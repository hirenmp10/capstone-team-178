"""Transport to the out-of-process GR00T policy server.

Pure stdlib + NumPy. This module must never import Isaac Sim -- and equally, it
must never import torch or GR00T.

**Why this is always a separate process.** Isaac Sim 5.1 ships Python 3.11.13;
GR00T N1.7's ``pyproject.toml`` requires ``>=3.12,<3.13``. Its ``flash-attn`` and
``deepspeed`` dependencies are additionally gated on ``sys_platform == 'linux'``.
No configuration puts them in one interpreter, so the process boundary is a fact
of the stack rather than a design preference -- and this module is the only thing
that crosses it.

The transport is length-prefixed pickle over TCP. Pickle because observations are
NumPy arrays and a 224x224x3 image pair per request makes JSON encoding the
dominant cost; length-prefixed because a bare stream gives no way to know where
one message ends. Both ends are ours, so the usual objection to pickle does not
apply here -- but the server must never be exposed beyond localhost.
"""

from __future__ import annotations

import pickle
import socket
import struct
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import Gr00tConfig
from mfw.core.errors import PolicyError
from mfw.utils.logging import get_logger

__all__ = ["Gr00tTcpClient", "send_message", "receive_message", "MAX_MESSAGE_BYTES"]

_log = get_logger("gr00t.client")

#: Refuse absurd frames rather than trying to allocate them. A corrupted length
#: prefix would otherwise ask for gigabytes.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024

_HEADER = struct.Struct("!Q")


def send_message(sock: socket.socket, payload: Any) -> None:
    """Send one length-prefixed pickled message."""
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if len(body) > MAX_MESSAGE_BYTES:
        raise PolicyError(f"message of {len(body)} bytes exceeds the limit")
    sock.sendall(_HEADER.pack(len(body)) + body)


def receive_exactly(sock: socket.socket, count: int) -> bytes:
    """Read exactly ``count`` bytes.

    TCP is a stream: a single ``recv`` may return a partial message, so the loop
    is required, not defensive padding.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise PolicyError("connection closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(sock: socket.socket) -> Any:
    """Receive one length-prefixed pickled message."""
    (length,) = _HEADER.unpack(receive_exactly(sock, _HEADER.size))
    if length > MAX_MESSAGE_BYTES:
        raise PolicyError(f"declared message size {length} exceeds the limit")
    return pickle.loads(receive_exactly(sock, length))


class Gr00tTcpClient:
    """Client for the policy server. Satisfies :class:`~mfw.core.interfaces.IPolicyClient`.

    Host and port come from config, so moving the server from Windows to WSL2 --
    or to another machine entirely -- is a configuration change and needs no code
    change anywhere in the framework.
    """

    def __init__(self, config: Gr00tConfig) -> None:
        config.validate()
        self.config = config
        self._sock: socket.socket | None = None
        self._ready = False

    def connect(self, retries: int = 3, backoff_s: float = 0.5) -> None:
        """Open the connection and handshake.

        Retries because a server started alongside the simulator may still be
        loading a multi-gigabyte checkpoint when the first connection is attempted.
        """
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                sock = socket.create_connection(
                    (self.config.host, self.config.port), timeout=self.config.request_timeout_s
                )
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock

                send_message(sock, {"type": "handshake", "embodiment": self.config.embodiment_tag})
                reply = receive_message(sock)
                if reply.get("type") != "handshake_ok":
                    raise PolicyError(f"unexpected handshake reply: {reply}")

                server_tag = reply.get("embodiment_tag")
                if server_tag and server_tag != self.config.embodiment_tag:
                    # A mismatch means the action layout differs, and the
                    # framework would silently misinterpret every chunk.
                    raise PolicyError(
                        f"server embodiment {server_tag!r} does not match configured "
                        f"{self.config.embodiment_tag!r}"
                    )

                self._ready = True
                _log.info(
                    "Connected to GR00T server at %s:%d (embodiment %s)",
                    self.config.host,
                    self.config.port,
                    server_tag or self.config.embodiment_tag,
                )
                return
            except (OSError, PolicyError) as exc:
                last_error = exc
                self.close()
                if attempt < retries:
                    time.sleep(backoff_s * attempt)

        raise PolicyError(
            f"could not connect to the GR00T server at {self.config.host}:{self.config.port}: "
            f"{last_error}"
        )

    def is_ready(self) -> bool:
        return self._ready and self._sock is not None

    def predict(self, observation: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Send one observation and receive one action chunk.

        For ``oxe_droid`` the reply carries ``eef_9d`` (relative),
        ``gripper_position`` (absolute) and ``joint_position`` (relative), each
        over a 40-step horizon.
        """
        if not self.is_ready():
            raise PolicyError("policy client is not connected; call connect() first")

        assert self._sock is not None
        try:
            send_message(self._sock, {"type": "predict", "observation": observation})
            reply = receive_message(self._sock)
        except (OSError, PolicyError) as exc:
            # The connection is unusable after a transport error; forcing a
            # reconnect is safer than issuing motion from a half-read reply.
            self._ready = False
            raise PolicyError(f"policy request failed: {exc}") from exc

        if reply.get("type") == "error":
            raise PolicyError(f"server error: {reply.get('message')}")
        if reply.get("type") != "action":
            raise PolicyError(f"unexpected reply type: {reply.get('type')}")

        action = reply.get("action")
        if not isinstance(action, dict):
            raise PolicyError("reply contained no action dict")
        return {key: np.asarray(value, dtype=np.float64) for key, value in action.items()}

    def close(self) -> None:
        self._ready = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "Gr00tTcpClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

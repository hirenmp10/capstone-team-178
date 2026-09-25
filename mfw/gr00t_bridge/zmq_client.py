"""Client for NVIDIA's own GR00T policy server (ZeroMQ + msgpack).

Pure stdlib + NumPy at module scope. This module must never import Isaac Sim,
torch, or GR00T.

Why this exists alongside :mod:`mfw.gr00t_bridge.client`
--------------------------------------------------------
The Isaac-GR00T repo ships ``gr00t/policy/server_client.py`` -- a ``PolicyServer``
and ``PolicyClient`` pair speaking ZeroMQ with msgpack framing. Using NVIDIA's
protocol is strictly better than the pickle transport written earlier:

* **msgpack, not pickle.** Deserialising pickle executes arbitrary code, so that
  transport was only ever defensible on loopback. msgpack is data-only.
* **It is the supported wire format**, so a GR00T upgrade that changes the
  observation schema stays compatible.
* **Zero maintenance** on the framing, request/reply and reconnection logic.

Both clients satisfy :class:`~mfw.core.interfaces.IPolicyClient`, so
``Gr00tExecutor`` is unchanged either way -- which is precisely what the Protocol
seam was for. The pickle client remains for the mock server used in tests, which
must run without ZeroMQ installed.

``pyzmq`` and ``msgpack`` are imported lazily, so importing this module (and
therefore the whole bridge) still works in an interpreter without them.

Wire contract (``Isaac-GR00T/gr00t/policy/server_client.py`` at n1.7-release)
------------------------------------------------------------------------------
* request: msgpack map ``{"endpoint": str, "data": {...}, "api_token"?: str}``;
  the server calls ``handler(**data)``, so ``get_action`` needs
  ``{"observation": ..., "options": ...}``.
* reply: whatever the handler returned -- ``get_action`` returns the tuple
  ``(action, info)``, which msgpack delivers as a **list**. Any server-side
  exception, unknown endpoint or bad token comes back as ``{"error": str}``.
* The server registers a ``kill`` endpoint, so a server bound beyond loopback
  should require an ``api_token`` (``scripts/groot_server.py --api-token``).
  This client sends one when given ``api_token=`` or when
  ``MFW_GR00T_API_TOKEN`` is set.

``tests/test_gr00t_zmq.py`` exercises this client against an in-thread REP
server speaking that contract. It validates the bridge (framing, token, error
and timeout handling), **not** model inference: no checkpoint is loaded there.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import Gr00tConfig
from mfw.core.errors import PolicyError
from mfw.utils.logging import get_logger

__all__ = ["Gr00tZmqClient", "API_TOKEN_ENV"]

_log = get_logger("gr00t.zmq")

#: Environment variable holding the shared secret for a token-protected server.
API_TOKEN_ENV = "MFW_GR00T_API_TOKEN"


class Gr00tZmqClient:
    """Talks to ``gr00t.policy.server_client.PolicyServer``.

    Satisfies :class:`~mfw.core.interfaces.IPolicyClient` structurally; no
    inheritance, so the framework never imports GR00T types.
    """

    def __init__(self, config: Gr00tConfig, api_token: str | None = None) -> None:
        config.validate()
        self.config = config
        #: Sent with every request when set. Kept out of the YAML on purpose:
        #: a secret does not belong in a committed config file.
        self.api_token = api_token if api_token is not None else os.environ.get(API_TOKEN_ENV)
        self._socket: Any = None
        self._context: Any = None
        self._ready = False

    # ------------------------------------------------------------------

    def connect(self, retries: int = 3, backoff_s: float = 1.0) -> None:
        """Open the REQ socket and confirm the server answers.

        Retries because a server launched alongside the simulator may still be
        loading a multi-gigabyte checkpoint on the first attempt.
        """
        try:
            import zmq
        except ImportError as exc:
            raise PolicyError(
                "pyzmq is not installed in this interpreter, so the GR00T server "
                "cannot be reached.\n"
                "The client runs inside Isaac Sim's python; install there with:\n"
                "    python.bat -m pip install pyzmq msgpack msgpack-numpy"
            ) from exc

        import time

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                # Without a timeout a REQ socket blocks forever on a server that
                # is loading, and the whole robot session hangs with no message.
                timeout_ms = int(self.config.request_timeout_s * 1000)
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                # Discard queued messages on close instead of blocking shutdown.
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(f"tcp://{self.config.host}:{self.config.port}")

                self._context, self._socket = context, socket
                self._ready = True

                # A REQ socket "connects" even with nothing listening, so the
                # only real proof is a round trip.
                self.ping()
                _log.info(
                    "Connected to the GR00T server at %s:%d",
                    self.config.host,
                    self.config.port,
                )
                return
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt < retries:
                    time.sleep(backoff_s * attempt)

        raise PolicyError(
            f"could not reach the GR00T policy server at {self.config.host}:"
            f"{self.config.port}: {last_error}\n"
            "Start it first:\n"
            "    <groot-venv-python> scripts/groot_server.py --port "
            f"{self.config.port}"
        )

    def is_ready(self) -> bool:
        return self._ready and self._socket is not None

    # ------------------------------------------------------------------

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """One request/reply round trip."""
        if not self.is_ready():
            raise PolicyError("policy client is not connected; call connect() first")

        payload: dict[str, Any] = {"endpoint": endpoint, "data": data or {}}
        if self.api_token:
            payload["api_token"] = self.api_token
        try:
            self._socket.send(self._pack(payload))
            reply = self._unpack(self._socket.recv())
        except Exception as exc:
            # A REQ socket that errors mid-exchange is stuck in a bad state:
            # ZeroMQ enforces strict send/recv alternation, so the only safe
            # recovery is a fresh socket rather than another request.
            self._ready = False
            raise PolicyError(f"policy request failed: {exc}") from exc

        if isinstance(reply, dict) and reply.get("error"):
            raise PolicyError(f"server error: {reply['error']}")
        return reply

    @staticmethod
    def _pack(payload: dict[str, Any]) -> bytes:
        import msgpack
        import msgpack_numpy

        return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)

    @staticmethod
    def _unpack(raw: bytes) -> Any:
        import msgpack
        import msgpack_numpy

        return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)

    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Round trip to prove the server is actually answering."""
        self._call("ping")
        return True

    def predict(self, observation: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Send one observation, receive one action chunk.

        For ``oxe_droid`` the reply carries ``eef_9d``, ``gripper_position`` and
        ``joint_position`` over a 40-step horizon. N1.7's processor decodes the
        relative training representation back to absolute poses, which is why
        ``gr00t.actions_are_absolute`` defaults to true.

        The payload is wrapped as ``{"observation": ..., "options": ...}`` because
        the server dispatches with ``handler(**request["data"])`` -- it spreads
        the data dict as *keyword arguments* onto ``policy.get_action(observation,
        options)``. Sending the observation bare makes every modality look like a
        stray kwarg:

            BasePolicy.get_action() got an unexpected keyword argument 'video'
        """
        reply = self._call("get_action", {"observation": observation, "options": None})

        # get_action returns (action, info); msgpack delivers that as a list.
        if isinstance(reply, (list, tuple)):
            action = next((item for item in reply if isinstance(item, dict)), None)
        elif isinstance(reply, dict):
            action = reply.get("action", reply)
        else:
            action = None

        if not isinstance(action, dict):
            raise PolicyError(
                f"could not find an action dict in a {type(reply).__name__} reply"
            )

        return {
            key: np.asarray(value, dtype=np.float64)
            for key, value in action.items()
            if isinstance(value, (np.ndarray, list))
        }

    def get_modality_config(self) -> Any:
        """Ask the server what observation layout it expects.

        Worth calling once at startup: a mismatch between the framework's
        observation builder and the server's embodiment is otherwise invisible
        until the actions come back subtly wrong.
        """
        return self._call("get_modality_config")

    def close(self) -> None:
        self._ready = False
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                _log.debug("ignoring error closing the ZMQ socket: %s", exc)
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception as exc:  # noqa: BLE001
                _log.debug("ignoring error terminating the ZMQ context: %s", exc)
            self._context = None

    def __enter__(self) -> "Gr00tZmqClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

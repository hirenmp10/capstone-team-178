"""Minimal ZMQ request/reply client used to reach ``jetson/robot_server.py``.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers; ``pyzmq``, ``msgpack`` and ``msgpack_numpy`` are imported
lazily so the package imports on an interpreter without them and fails with a
message naming the pip packages when they are actually needed.

Wire format is the one :mod:`mfw.gr00t_bridge.zmq_client` already speaks
(request ``{"endpoint": str, "data": dict}`` -> reply dict or ``{"error":
str}``, msgpack + msgpack_numpy framing), so the robot server and the GR00T
server are interchangeable at the transport level and there is one framing
convention to debug.

Two traps this class is shaped around:

* A REQ socket that times out is *stuck*: ZeroMQ enforces strict send/recv
  alternation, and the late reply, if it ever arrives, would be delivered to
  the *next* request. The only safe recovery is to throw the socket away, so
  every failure closes it and the next call reconnects.
* ``follow_trajectory`` blocks on the server for the duration of the motion,
  which can be many times ``request_timeout_s``. The receive timeout is
  therefore set per call, not per socket.
"""

from __future__ import annotations

import time
from typing import Any

from mfw.core.errors import ExecutionError
from mfw.utils.logging import get_logger

__all__ = ["ZmqRpcClient", "RpcError"]

_log = get_logger("hardware.zmq_rpc")


class RpcError(ExecutionError):
    """The server was unreachable, timed out, or answered ``{"error": ...}``."""


def _import_zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:
        raise RpcError(
            "pyzmq is not installed in this interpreter; install with "
            "`py -3.12 -m pip install pyzmq msgpack msgpack-numpy`"
        ) from exc
    return zmq


def pack(payload: Any) -> bytes:
    """msgpack + msgpack_numpy encode (NumPy arrays travel as typed binary)."""
    try:
        import msgpack
        import msgpack_numpy
    except ImportError as exc:
        raise RpcError(
            "msgpack / msgpack-numpy are not installed; install with "
            "`py -3.12 -m pip install msgpack msgpack-numpy`"
        ) from exc
    return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(raw: bytes) -> Any:
    """Inverse of :func:`pack`."""
    import msgpack
    import msgpack_numpy

    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)


class ZmqRpcClient:
    """One REQ socket to one ``tcp://host:port`` server, with per-call timeouts."""

    def __init__(self, host: str, port: int, request_timeout_s: float = 5.0) -> None:
        if not (0 < int(port) < 65536):
            raise ValueError(f"port out of range: {port}")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")
        self.host = str(host)
        self.port = int(port)
        self.request_timeout_s = float(request_timeout_s)
        self._context: Any = None
        self._socket: Any = None

    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """``tcp://host:port`` this client talks to."""
        return f"tcp://{self.host}:{self.port}"

    def is_connected(self) -> bool:
        """Whether a socket currently exists (not proof the server answers)."""
        return self._socket is not None

    def connect(self) -> None:
        """Open the socket. Cheap; a REQ socket 'connects' even with no server."""
        if self._socket is not None:
            return
        zmq = _import_zmq()
        # The process-wide context, never terminated by this client. A per-client
        # Context that is garbage-collected while its socket is still open blocks
        # forever in Context.__del__ -> term(); measured: a leaked JetsonClient in
        # an exception traceback hung pytest's shutdown. Closing the socket with
        # LINGER 0 releases everything this client owns.
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, int(self.request_timeout_s * 1000))
        socket.setsockopt(zmq.RCVTIMEO, int(self.request_timeout_s * 1000))
        socket.connect(self.endpoint)
        self._context, self._socket = context, socket

    def close(self) -> None:
        """Close the socket. Idempotent. The shared context is left alone."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception as exc:  # pragma: no cover - best effort
                _log.debug("socket close failed: %s", exc)
            self._socket = None
        self._context = None

    def __del__(self) -> None:  # pragma: no cover - safety net for leaked clients
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------

    def call(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """One round trip. Raises :class:`RpcError` on any failure.

        ``timeout_s`` overrides the receive timeout for this call only; motion
        endpoints pass the expected duration plus a margin.
        """
        self.connect()
        zmq = _import_zmq()
        timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        payload = {"endpoint": str(endpoint), "data": dict(data or {})}
        try:
            self._socket.setsockopt(zmq.RCVTIMEO, int(max(timeout, 0.001) * 1000))
            self._socket.send(pack(payload))
            reply = unpack(self._socket.recv())
        except Exception as exc:
            # The socket is now in an undefined send/recv state; drop it.
            self.close()
            raise RpcError(
                f"{endpoint!r} to {self.endpoint} failed after {timeout:.1f}s: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if isinstance(reply, dict) and reply.get("error"):
            raise RpcError(f"{endpoint!r}: server error: {reply['error']}")
        return reply

    def ping(self, retries: int = 3, backoff_s: float = 0.5) -> dict[str, Any]:
        """Round trip until the server answers, or raise with a start hint."""
        last: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                reply = self.call("ping")
                if not isinstance(reply, dict):
                    raise RpcError(f"ping returned {type(reply).__name__}, expected a dict")
                return reply
            except RpcError as exc:
                last = exc
                if attempt < retries:
                    time.sleep(backoff_s * attempt)
        raise RpcError(f"no server answering at {self.endpoint}: {last}")

    def __enter__(self) -> "ZmqRpcClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

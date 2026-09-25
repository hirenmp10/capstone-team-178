"""ZeroMQ RPC client transport for hardware-lane services.

This module provides a lightweight, dependency-minimal (pyzmq + msgpack)
RPC client for communicating with `jetson/robot_server.py` over port 5560.

Note:
    `mfw.hardware.jetson_client` (owned by Adyanth) wraps this raw RPC client
    with typed, domain-specific methods. Do not implement `jetson_client.py` here.
"""

from __future__ import annotations

from typing import Any
import msgpack
import zmq


class RpcError(Exception):
    """Raised when an RPC request fails, times out, or returns `ok: false`."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"[{error_type}] {message}")
        self.error_type = str(error_type)
        self.message = str(message)


class ZmqRpcClient:
    """Client for ZeroMQ RPC services using msgpack serialization.

    Uses a `ZMQ.REQ` socket. Because `REQ` sockets have a strict alternating
    send/recv state machine, if a request times out the socket is closed
    and recreated before any retry.
    """

    def __init__(self, host: str, port: int, request_timeout_s: float = 5.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.request_timeout_s = float(request_timeout_s)
        self.endpoint = f"tcp://{self.host}:{self.port}"
        self._ctx = zmq.Context()
        self._sock: zmq.Socket | None = None
        self._req_id = 0
        self._init_socket()

    def _init_socket(self) -> None:
        """Create or recreate the REQ socket and connect to the endpoint."""
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None

        if not self._ctx.closed:
            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.connect(self.endpoint)

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        retries: int = 0,
    ) -> dict[str, Any]:
        """Send an RPC request and await the reply.

        Args:
            method: RPC method name.
            params: Dictionary of parameters (defaults to empty dict).
            timeout_s: Timeout in seconds (defaults to `self.request_timeout_s`).
            retries: Number of retry attempts on timeout.

        Returns:
            Dictionary returned by the server on success (`ok: true`).

        Raises:
            RpcError: On timeout, network failure, or when the server replies
                with `ok: false`.
        """
        if params is None:
            params = {}
        if timeout_s is None:
            timeout_s = self.request_timeout_s

        self._req_id += 1
        request = {
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        payload = msgpack.packb(request, use_bin_type=True)
        attempts = max(0, retries) + 1

        for attempt in range(attempts):
            try:
                if self._sock is None:
                    self._init_socket()

                assert self._sock is not None
                self._sock.send(payload)

                poller = zmq.Poller()
                poller.register(self._sock, zmq.POLLIN)
                events = dict(poller.poll(int(timeout_s * 1000)))

                if self._sock in events and events[self._sock] == zmq.POLLIN:
                    reply_bytes = self._sock.recv()
                    reply = msgpack.unpackb(reply_bytes, raw=False)
                    if not isinstance(reply, dict):
                        raise RpcError(
                            "InvalidReply",
                            f"Expected dict reply from server, got {type(reply).__name__}",
                        )
                    if not reply.get("ok", False):
                        raise RpcError(
                            reply.get("error_type", "RpcError"),
                            reply.get("error", "RPC call failed"),
                        )
                    return reply
                else:
                    # Timeout on this attempt: cycle the REQ socket
                    self._init_socket()
                    if attempt == attempts - 1:
                        raise RpcError(
                            "Timeout",
                            f"Timeout after {timeout_s:.2f}s waiting for {method} on {self.endpoint}",
                        )
            except zmq.ZMQError as exc:
                self._init_socket()
                if attempt == attempts - 1:
                    raise RpcError("ZmqError", str(exc)) from exc

        raise RpcError("Timeout", f"Timeout after {timeout_s:.2f}s on {self.endpoint}")

    def ping(self, retries: int = 1) -> dict[str, Any]:
        """Convenience method to ping the server."""
        return self.call("ping", {}, retries=retries)

    def close(self) -> None:
        """Close the socket and terminate the ZeroMQ context."""
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None

        if self._ctx is not None and not self._ctx.closed:
            try:
                self._ctx.term()
            except Exception:
                pass

    def __enter__(self) -> ZmqRpcClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

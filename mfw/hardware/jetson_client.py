"""Typed client for ``jetson/robot_server.py`` (the arm + webcam front door).

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers.

One method per server endpoint (table in ``docs/HARDWARE_BRIEF.md`` section 5),
each returning the server's reply dict verbatim so the caller can log it. Every
reply that carries a state also refreshes :attr:`last_state`, which is what
:class:`~mfw.hardware.remote_arm.RemoteArm` caches: after our own command the
commanded pose is known exactly, so a ``get_state`` round trip is only worth
paying occasionally to notice an estop or an external change.

Waypoints travel as **float64**. They used to be cast to float32 to save
bytes, and a waypoint sitting exactly on a measured joint limit (say 1.4362
rad) rounded 5e-8 past it, which the server's 1e-9 clamp threshold reported
as ``clamped: true`` and the controller then logged as a calibration mismatch
that did not exist (review P5). Fifty waypoints of four doubles are 1.6 kB;
bandwidth was never the constraint.

The recorded trap: ``follow_trajectory`` blocks server-side for the whole move,
and the server may *stretch* the move to respect its own velocity ceiling. The
receive timeout for that call is therefore the trajectory's own duration,
doubled to leave room for stretching, plus ``trajectory_timeout_margin_s`` --
never the plain request timeout, which would abandon every motion longer than
five seconds while the servos were still tracking it.

Host liveness. The server detaches an idle arm when no client has spoken for
``host_timeout_s`` (5 s), so a crashed laptop or a lost Wi-Fi link cannot
leave the jaw squeezing forever. A laptop that is alive but idle -- waiting at
the ``robot>`` prompt, running the LLM, listening -- must therefore say so:
:class:`Heartbeat` sends ``get_state`` on its **own** connection every
``period_s`` (a REQ socket is not thread-safe and the main one is blocked for
whole trajectory chunks). ``RemoteArm`` and ``calibrate_servos.py`` run one;
``calibrate_table.py --touch`` deliberately does not, so a jaw left pressed on
the table relaxes after the timeout.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from mfw.hardware.zmq_rpc import RpcError, ZmqRpcClient
from mfw.utils.logging import get_logger

__all__ = ["JetsonClient", "RpcError", "Heartbeat", "HEARTBEAT_PERIOD_S"]

HEARTBEAT_PERIOD_S = 1.0
"""Default :class:`Heartbeat` period: a fifth of the server's 5 s host timeout,
so four heartbeats in a row can be lost to Wi-Fi before the arm relaxes."""

_log = get_logger("hardware.jetson_client")

_EXPECTED_SERVER = "mfw-robot"
_HOME_TIMEOUT_S = 20.0
"""Upper bound on a homing move at the slowest plausible calibration."""


class JetsonClient:
    """Endpoint methods over :class:`~mfw.hardware.zmq_rpc.ZmqRpcClient`.

    Not connected on construction; call :meth:`connect`, which also verifies
    the peer is a robot server (``ping`` reports ``server: "mfw-robot"``) so a
    GR00T server on the wrong port is caught at startup rather than at the
    first trajectory.
    """

    def __init__(
        self,
        host: str,
        port: int,
        request_timeout_s: float = 5.0,
        trajectory_timeout_margin_s: float = 5.0,
    ) -> None:
        if trajectory_timeout_margin_s <= 0:
            raise ValueError("trajectory_timeout_margin_s must be > 0")
        self.rpc = ZmqRpcClient(host, port, request_timeout_s)
        self.trajectory_timeout_margin_s = float(trajectory_timeout_margin_s)
        self.last_state: dict[str, Any] | None = None
        self.server_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """``tcp://host:port``."""
        return self.rpc.endpoint

    def connect(self, retries: int = 3, backoff_s: float = 0.5) -> dict[str, Any]:
        """Ping until the server answers; returns the ping reply.

        Raises :class:`RpcError` naming the start command when nothing answers
        or the peer is not a robot server.
        """
        try:
            info = self.rpc.ping(retries=retries, backoff_s=backoff_s)
        except RpcError as exc:
            raise RpcError(
                f"{exc}\nStart it first (fake for the laptop lane):\n"
                f"    py -3.12 jetson/robot_server.py --driver fake --fake-camera --port {self.rpc.port}"
            ) from exc
        server = str(info.get("server", ""))
        if server != _EXPECTED_SERVER:
            self.rpc.close()
            raise RpcError(
                f"{self.endpoint} answered as {server or 'an unknown server'!r}, "
                f"expected {_EXPECTED_SERVER!r}; check hardware.jetson_port"
            )
        self.server_info = info
        _log.info(
            "Connected to robot server %s (driver=%s, fake=%s, protocol v%s)",
            self.endpoint, info.get("driver"), info.get("fake"), info.get("version"),
        )
        return info

    def is_ready(self) -> bool:
        """Whether :meth:`connect` succeeded and the socket is still open."""
        return self.server_info is not None and self.rpc.is_connected()

    @property
    def is_fake(self) -> bool:
        """True when the server reported a fake driver (no servos attached)."""
        return bool(self.server_info and self.server_info.get("fake"))

    def close(self) -> None:
        """Close the transport. Idempotent."""
        self.rpc.close()
        self.server_info = None

    # ------------------------------------------------------------------

    def _state_call(self, endpoint: str, data: dict[str, Any] | None = None,
                    timeout_s: float | None = None) -> dict[str, Any]:
        reply = self.rpc.call(endpoint, data, timeout_s=timeout_s)
        if not isinstance(reply, dict):
            raise RpcError(f"{endpoint!r} returned {type(reply).__name__}, expected a dict")
        if "q" in reply:
            self.last_state = reply
        return reply

    def ping(self) -> dict[str, Any]:
        """``{ok, server, version, driver, fake, t}``."""
        return self._state_call("ping")

    def get_state(self) -> dict[str, Any]:
        """``{q: [4] rad, gripper_width, moving, estopped, t}`` -- commanded values."""
        return self._state_call("get_state")

    def get_frame(self, quality: int | None = None) -> dict[str, Any]:
        """``{jpeg: bytes, width, height, t_capture, seq}`` from the Jetson webcam."""
        data: dict[str, Any] = {}
        if quality is not None:
            data["quality"] = int(quality)
        reply = self._state_call("get_frame", data)
        if not isinstance(reply.get("jpeg"), (bytes, bytearray)):
            raise RpcError("get_frame reply carries no JPEG bytes")
        return reply

    def set_gripper(self, width_m: float) -> dict[str, Any]:
        """Command the jaw opening in metres; returns the new state."""
        return self._state_call("set_gripper", {"width": float(width_m)})

    def follow_trajectory(self, waypoints: ArrayLike, dt: float) -> dict[str, Any]:
        """Stream ``waypoints`` (N, 4) spaced ``dt`` seconds apart; blocks until done.

        Returns ``{q, gripper_width, completed, elapsed_s, clamped, estopped,
        detached, reattached, detach_count, last_detach_reason, attached}``:
        ``reattached`` is true when the servos were limp and the server had
        to re-energise them before streaming (a detach the caller may not
        have asked for; ``detach_count`` says whether one happened). The
        receive timeout scales with the trajectory so long moves are not
        abandoned mid-flight.
        """
        wp = np.ascontiguousarray(np.asarray(waypoints, dtype=np.float64))
        if wp.ndim == 1:
            wp = wp.reshape(1, -1)
        if wp.ndim != 2 or wp.shape[0] == 0:
            raise ValueError(f"waypoints must be (N, dof), got shape {wp.shape}")
        if not dt > 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        duration = (wp.shape[0] - 1) * float(dt)
        timeout = 2.0 * duration + self.trajectory_timeout_margin_s + self.rpc.request_timeout_s
        started = time.perf_counter()
        reply = self._state_call(
            "follow_trajectory", {"waypoints": wp, "dt": float(dt)}, timeout_s=timeout
        )
        _log.debug(
            "follow_trajectory: %d waypoints, %.2fs planned, %.2fs round trip, completed=%s",
            wp.shape[0], duration, time.perf_counter() - started, reply.get("completed"),
        )
        return reply

    def home(self) -> dict[str, Any]:
        """Move to the Jetson-side calibrated home posture; returns the state."""
        return self._state_call(
            "home", timeout_s=_HOME_TIMEOUT_S + self.trajectory_timeout_margin_s
        )

    def estop(self) -> dict[str, Any]:
        """Detach every servo and refuse motion until :meth:`clear_estop`."""
        return self._state_call("estop")

    def clear_estop(self) -> dict[str, Any]:
        """Allow motion again. Servos re-attach on the next motion command."""
        return self._state_call("clear_estop")

    def set_torque(self, enabled: bool) -> dict[str, Any]:
        """``False`` lets the arm go limp (for hand-posing during calibration)."""
        return self._state_call("set_torque", {"enabled": bool(enabled)})

    def get_calibration(self) -> dict[str, Any]:
        """``{calibration: {...}, path}`` -- the Jetson-resident servo calibration."""
        return self._state_call("get_calibration")

    def set_calibration(self, calibration: dict[str, Any]) -> dict[str, Any]:
        """Replace (and persist server-side) the servo calibration."""
        return self._state_call("set_calibration", {"calibration": dict(calibration)})

    def get_fake_world(self) -> dict[str, Any]:
        """``{objects: {label: [x, y, z]}, attached, tcp}`` -- fake lane only.

        Served only by ``robot_server.py --driver fake --fake-world ...``; a
        real server answers with an error (raised here as :class:`RpcError`).
        Tests and ``calibrate_servos.py`` use it to see what the fake jaw did.
        """
        return self._state_call("get_fake_world")

    def __enter__(self) -> "JetsonClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class Heartbeat:
    """Proves to the robot server that this process is alive (see the module doc).

    Sends ``get_state`` on a private connection every ``period_s`` from a
    daemon thread, and hands each reply to ``on_state`` (``RemoteArm`` reads
    the detach counter from it, so a bridge detach during an idle gap is
    noticed within a period rather than at the next motion). A failed beat is
    logged once and retried next period; it never raises into the caller.
    Stop it with :meth:`stop` (idempotent); a stopped heartbeat is exactly
    what lets the server relax the arm when this process dies.
    """

    def __init__(
        self,
        host: str,
        port: int,
        period_s: float = HEARTBEAT_PERIOD_S,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        request_timeout_s: float = 2.0,
    ) -> None:
        if period_s <= 0:
            raise ValueError("heartbeat period must be > 0")
        self.host, self.port = str(host), int(port)
        self.period_s = float(period_s)
        self.on_state = on_state
        self.request_timeout_s = float(request_timeout_s)
        self.beats = 0
        """Successful beats since :meth:`start`."""
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: JetsonClient | None = None
        self._warned = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "Heartbeat":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jetson-heartbeat", daemon=True)
        self._thread.start()
        return self

    def beat(self) -> bool:
        """One ``get_state`` round trip now; True on success. Used by the thread."""
        try:
            if self._client is None:
                self._client = JetsonClient(self.host, self.port, request_timeout_s=self.request_timeout_s)
            reply = self._client.get_state()
        except RpcError as exc:
            self.failures += 1
            if not self._warned:
                _log.warning("heartbeat to %s:%d failed (%s); retrying every %.1f s", self.host, self.port, exc, self.period_s)
                self._warned = True
            return False
        self.beats += 1
        self._warned = False
        if self.on_state is not None:
            try:
                self.on_state(reply)
            except Exception as exc:  # noqa: BLE001 - a bad callback must not stop the heartbeat
                _log.warning("heartbeat callback failed: %s", exc)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.beat()
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.request_timeout_s + self.period_s + 1.0)
        self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None

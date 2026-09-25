"""Robot server for Jetson-hosted or simulated hardware arm control.

Runs as a service on the Jetson (or on a laptop in fake mode) exposing
a ZeroMQ ROUTER RPC endpoint over port 5560 using msgpack serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import asdict, dataclass, field
import io
import json
import logging
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

logger = logging.getLogger("robot_server")


# ==============================================================================
# Calibration & Channel Data Structures
# ==============================================================================


@dataclass
class ServoChannel:
    """Configuration for a single servo channel."""

    name: str
    channel: int
    pulse_min_us: int
    pulse_max_us: int
    pulse_zero_us: int
    us_per_rad: float  # Signed: sign encodes servo movement direction
    rad_min: float  # Kinematic lower limit
    rad_max: float  # Kinematic upper limit
    width_open_m: float = 0.08  # Jaw only: width when fully open
    width_closed_m: float = 0.0  # Jaw only: width when fully closed


@dataclass
class ServoCalibration:
    """Servo calibration for the 4-DOF hobby arm and gripper jaw."""

    joints: list[ServoChannel]
    jaw: ServoChannel

    @classmethod
    def default(cls) -> ServoCalibration:
        """Default calibration with MG90S placeholder parameters.

        # PLACEHOLDER: Sized to ~28cm reach PLA arm kit with MG90S servos.
        """
        # ~573 us/rad (180 deg = pi rad -> 1800 us / pi =~ 572.9578 us/rad)
        us_rad = 572.9577951308232
        joints = [
            ServoChannel(
                name="base_yaw",
                channel=0,
                pulse_min_us=500,
                pulse_max_us=2400,
                pulse_zero_us=1450,
                us_per_rad=us_rad,
                rad_min=-1.5708,
                rad_max=1.5708,
            ),
            ServoChannel(
                name="shoulder",
                channel=1,
                pulse_min_us=500,
                pulse_max_us=2400,
                pulse_zero_us=1450,
                us_per_rad=us_rad,
                rad_min=-0.5236,
                rad_max=1.5708,
            ),
            ServoChannel(
                name="elbow",
                channel=2,
                pulse_min_us=500,
                pulse_max_us=2400,
                pulse_zero_us=1450,
                us_per_rad=us_rad,
                rad_min=-1.5708,
                rad_max=1.5708,
            ),
            ServoChannel(
                name="wrist",
                channel=3,
                pulse_min_us=500,
                pulse_max_us=2400,
                pulse_zero_us=1450,
                us_per_rad=us_rad,
                rad_min=-1.5708,
                rad_max=1.5708,
            ),
        ]
        jaw = ServoChannel(
            name="jaw",
            channel=4,
            pulse_min_us=500,
            pulse_max_us=2400,
            pulse_zero_us=1450,
            us_per_rad=us_rad,
            rad_min=0.0,
            rad_max=1.5708,
            width_open_m=0.08,
            width_closed_m=0.0,
        )
        return cls(joints=joints, jaw=jaw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServoCalibration:
        joints = [ServoChannel(**j) for j in data["joints"]]
        jaw = ServoChannel(**data["jaw"])
        return cls(joints=joints, jaw=jaw)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> ServoCalibration:
        if isinstance(path_or_str, Path):
            text = path_or_str.read_text(encoding="utf-8")
        else:
            s = str(path_or_str).strip()
            if s.startswith("{"):
                text = s
            else:
                try:
                    p = Path(path_or_str)
                    if p.is_file():
                        text = p.read_text(encoding="utf-8")
                    else:
                        text = s
                except (OSError, ValueError):
                    text = s
        return cls.from_dict(json.loads(text))

    def rad_to_pulse(self, i: int, rad: float) -> tuple[float, float]:
        """Convert joint angle in radians to servo pulse in microseconds.

        Returns:
            (pulse_us, clamped_by_rad)
        """
        ch = self.joints[i]
        clamped_rad = min(max(rad, ch.rad_min), ch.rad_max)
        clamped_by_rad = abs(rad - clamped_rad)
        pulse = ch.pulse_zero_us + ch.us_per_rad * clamped_rad
        clamped_pulse = min(max(pulse, float(ch.pulse_min_us)), float(ch.pulse_max_us))
        return clamped_pulse, clamped_by_rad

    def pulse_to_rad(self, i: int, pulse: float | int) -> float:
        """Convert servo pulse in microseconds to joint angle in radians."""
        ch = self.joints[i]
        return (float(pulse) - ch.pulse_zero_us) / ch.us_per_rad

    def width_to_pulse(self, width_m: float) -> float:
        """Convert gripper jaw width in metres to servo pulse in microseconds."""
        jaw = self.jaw
        clamped_w = min(max(width_m, jaw.width_closed_m), jaw.width_open_m)
        span_w = jaw.width_open_m - jaw.width_closed_m
        fraction = (clamped_w - jaw.width_closed_m) / span_w if span_w != 0 else 0.0
        pulse = jaw.pulse_min_us + fraction * (jaw.pulse_max_us - jaw.pulse_min_us)
        return min(max(pulse, float(jaw.pulse_min_us)), float(jaw.pulse_max_us))

    def pulse_to_width(self, pulse: float | int) -> float:
        """Convert servo pulse in microseconds to gripper jaw width in metres."""
        jaw = self.jaw
        span_p = jaw.pulse_max_us - jaw.pulse_min_us
        fraction = (float(pulse) - jaw.pulse_min_us) / span_p if span_p != 0 else 0.0
        return jaw.width_closed_m + fraction * (jaw.width_open_m - jaw.width_closed_m)


# ==============================================================================
# Driver Abstraction & Implementations
# ==============================================================================


class DriverError(Exception):
    """Raised by a driver when a serial/hardware command fails or times out."""



class Driver(ABC):
    """Abstract hardware driver controlling servo actuation."""

    attached: bool
    name: str

    @abstractmethod
    def attach(self) -> None:
        """Attach / power servo motors."""

    @abstractmethod
    def detach(self) -> None:
        """Detach / de-energize servo motors."""

    @abstractmethod
    def write_pulses(self, pulses: list[int], duration_ms: int = 0) -> None:
        """Write microsecond pulses to all servo channels.

        Args:
            pulses: list of N pulse values in microseconds.
            duration_ms: if > 0, use the T (interpolate) command instead of P;
                         ignored by FakeDriver (which has no interpolation time).
        """


class UnoSerialDriver(Driver):
    """Arduino Uno USB-CDC serial driver.

    Talks to ``firmware/servo_bridge/servo_bridge.ino`` over a USB-CDC serial
    port (115200 8N1 ASCII lines).  Import of ``serial`` (pyserial) is deferred
    to ``open()`` so the fake lane never requires pyserial at import time.

    Serial protocol summary (full spec: docs/hardware/PROTOCOL.md §1.5):
        P p0…pN\\n  — set targets immediately
        T ms p0…pN\\n — interpolate over ms milliseconds
        D\\n         — detach all (E-stop)
        W ms\\n      — set watchdog timeout
        S\\n         — status (attached, moving, pulses); counts as keepalive
        V\\n         — firmware version
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        watchdog_ms: int = 500,
        timeout_s: float = 0.2,
        serial_factory: Any = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.watchdog_ms = int(watchdog_ms)
        self.timeout_s = float(timeout_s)
        self.serial_factory = serial_factory
        self._ser: Any = None
        self._lock = threading.Lock()
        self.attached = False
        self.moving = False
        self.name = "uno_serial"
        self._n_channels: int | None = None

    def open(self, calibration: "ServoCalibration | None" = None) -> None:
        """Open the serial port, handshake with the Uno, set watchdog, and sync state.

        Args:
            calibration: used to verify channel count; if None uses N=5.

        Raises:
            RuntimeError: if the firmware channel count doesn't match calibration.
            DriverError: if version handshake fails or times out.
        """
        # Lazy pyserial import so the fake lane never needs it
        if self.serial_factory is not None:
            self._ser = self.serial_factory(self.port, self.baud, timeout=self.timeout_s)
        else:
            import serial  # noqa: PLC0415
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)

        # Wait for Uno auto-reset (DTR toggle on open resets ATmega328P)
        time.sleep(2.0)
        if hasattr(self._ser, "reset_input_buffer"):
            self._ser.reset_input_buffer()

        # Version handshake
        v_reply = self._cmd("V")
        # expect: "V servo_bridge 1 <nch>"
        parts = v_reply.split()
        if len(parts) != 4 or parts[0] != "V" or parts[1] != "servo_bridge":
            raise DriverError(f"Unexpected version reply: {v_reply!r}")
        n_firmware = int(parts[3])
        n_expected = (len(calibration.joints) + 1) if calibration is not None else 5
        if n_firmware != n_expected:
            raise RuntimeError(
                f"Firmware channel count {n_firmware} != calibration channel count {n_expected}"
            )
        self._n_channels = n_firmware

        # Set watchdog
        self._cmd(f"W {self.watchdog_ms}")

        # Sync state
        self._sync_state()

    def _cmd(self, line: str) -> str:
        """Send one command line and return the reply, under lock.

        Raises:
            DriverError: on timeout or if the reply is ERR …
        """
        with self._lock:
            assert self._ser is not None, "serial port not open"
            out = (line + "\n").encode()
            logger.debug("Uno << %r", out)
            self._ser.write(out)
            reply = self._ser.readline().decode(errors="replace").strip()
            logger.debug("Uno >> %r", reply)
            if not reply:
                raise DriverError(f"no reply from Uno for command {line!r}")
            if reply.startswith("ERR"):
                raise DriverError(reply)
            return reply

    def _sync_state(self) -> None:
        """Parse an S reply and update attached / moving flags."""
        s_reply = self._cmd("S")
        parts = s_reply.split()
        # S <att> <mov> <wd_ms> p0…pN
        if len(parts) >= 3 and parts[0] == "S":
            self.attached = parts[1] == "1"
            self.moving = parts[2] == "1"

    def attach(self) -> None:
        """No-op: the Uno attaches on the first P/T command."""
        # Attach happens implicitly on first write_pulses call
        pass

    def detach(self) -> None:
        """Send D to detach all servos."""
        try:
            self._cmd("D")
            self.attached = False
        except DriverError as exc:
            logger.warning("Uno detach failed: %s", exc)

    def write_pulses(self, pulses: list[int], duration_ms: int = 0) -> None:
        """Send P or T command with the given pulses.

        Args:
            pulses: list of pulse values in µs.
            duration_ms: if > 0 uses T (interpolate) command; else P (immediate).

        Raises:
            DriverError: if the Uno replies ERR or times out.
        """
        pulse_str = " ".join(str(int(p)) for p in pulses)
        if duration_ms > 0:
            cmd = f"T {int(duration_ms)} {pulse_str}"
        else:
            cmd = f"P {pulse_str}"
        self._cmd(cmd)
        self.attached = True  # P/T always attaches

    def keepalive(self) -> None:
        """Send S to refresh attached/moving state without writing pulses.

        The S command counts as a keepalive frame in the Uno firmware.
        """
        try:
            self._sync_state()
        except DriverError as exc:
            logger.warning("Uno keepalive (S) failed: %s", exc)

    def close(self) -> None:
        """Detach and close the serial port."""
        try:
            self.detach()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:  # noqa: BLE001
            pass
        self._ser = None


class Pca9685Driver(Driver):
    """I2C PCA9685 PWM driver (Stage B - PLANNED)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Stage B: Pca9685Driver is planned for Stage B")

    def attach(self) -> None:
        raise NotImplementedError("Stage B")

    def detach(self) -> None:
        raise NotImplementedError("Stage B")

    def write_pulses(self, pulses: list[int]) -> None:
        raise NotImplementedError("Stage B")


class FakeDriver(Driver):
    """In-memory simulation driver for tests and dry runs."""

    def __init__(
        self,
        calibration: ServoCalibration,
        on_command: Callable[[list[int], list[float], float], Any] | None = None,
        watchdog_ms: float | None = None,
    ) -> None:
        self.calibration = calibration
        self.on_command = on_command
        self.watchdog_ms = watchdog_ms
        self.attached = False
        self.name = "fake"
        self.last_pulses: list[int] = [
            ch.pulse_zero_us for ch in calibration.joints
        ] + [int(calibration.jaw.pulse_min_us)]
        self.last_write_time: float | None = None
        self.watchdog_trips = 0
        self.write_count = 0

    def attach(self) -> None:
        self.attached = True

    def detach(self) -> None:
        self.attached = False

    def write_pulses(self, pulses: list[int], duration_ms: int = 0) -> None:
        now = time.monotonic()
        if (
            self.watchdog_ms is not None
            and self.attached
            and self.last_write_time is not None
        ):
            dt_ms = (now - self.last_write_time) * 1000.0
            if dt_ms > self.watchdog_ms:
                self.attached = False
                self.watchdog_trips += 1

        self.last_write_time = now
        self.write_count += 1

        clamped = []
        for i, p in enumerate(pulses):
            ch = self.calibration.joints[i] if i < 4 else self.calibration.jaw
            clamped.append(int(min(max(p, ch.pulse_min_us), ch.pulse_max_us)))
        self.last_pulses = clamped

        if self.on_command is not None:
            joints_rad = [
                self.calibration.pulse_to_rad(i, clamped[i]) for i in range(4)
            ]
            width_m = self.calibration.pulse_to_width(clamped[4])
            self.on_command(clamped, joints_rad, width_m)


# ==============================================================================
# RobotServer Implementation
# ==============================================================================


class RobotServer:
    """ZeroMQ ROUTER RPC server for robot control."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5560,
        driver: Driver | None = None,
        camera: Any = None,
        calibration: ServoCalibration | None = None,
        follower_sleep: Callable[[float], None] = time.sleep,
        host_timeout_s: float = 5.0,
        tick_s: float = 0.02,
        fake_world: str | dict[str, Any] | None = None,
        fake_camera: bool = False,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.calibration = calibration or ServoCalibration.default()
        self.driver = driver or FakeDriver(self.calibration)
        self.camera = camera
        self.fake_camera = fake_camera
        self.follower_sleep = follower_sleep
        self.host_timeout_s = float(host_timeout_s)
        self.tick_s = float(tick_s)
        self.fake_world = fake_world

        self._estopped = False
        self._moving = False
        self._current_q = [
            self.calibration.pulse_to_rad(i, ch.pulse_zero_us)
            for i, ch in enumerate(self.calibration.joints)
        ]
        self._current_gripper_w = self.calibration.jaw.width_open_m

        self._start_time = time.time()
        self._last_request_time = time.monotonic()
        self._running = False

        self._motion_queue: queue.Queue = queue.Queue(maxsize=1)
        self._abort_motion = threading.Event()

        self._ctx: zmq.Context | None = None
        self._router_sock: zmq.Socket | None = None
        self._backend_endpoint = f"inproc://backend_{id(self)}"

        self._server_thread: threading.Thread | None = None
        self._follower_thread: threading.Thread | None = None
        self._timeout_thread: threading.Thread | None = None
        self._bound_port = self.port

    def serve_in_thread(self, port: int | None = None) -> tuple[threading.Thread, int]:
        """Start the server in a background thread and return (thread, bound_port)."""
        if port is not None:
            self.port = int(port)

        ready_event = threading.Event()
        self._server_thread = threading.Thread(
            target=self._run_server,
            args=(ready_event,),
            name="RobotServer-Main",
            daemon=True,
        )
        self._server_thread.start()
        ready_event.wait(timeout=10.0)
        return self._server_thread, self._bound_port

    def serve_forever(self) -> None:
        """Start the server and block until stopped."""
        ready_event = threading.Event()
        self._run_server(ready_event)

    def _run_server(self, ready_event: threading.Event) -> None:
        self._ctx = zmq.Context()
        self._router_sock = self._ctx.socket(zmq.ROUTER)
        self._router_sock.setsockopt(zmq.LINGER, 0)
        bind_addr = f"tcp://{self.host}:{self.port}"
        self._router_sock.bind(bind_addr)

        # Resolve port if port 0 was passed
        last_endpoint = self._router_sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._bound_port = int(last_endpoint.rpartition(":")[2])

        # Internal backend socket for receiving motion replies
        backend_sock = self._ctx.socket(zmq.PULL)
        backend_sock.setsockopt(zmq.LINGER, 0)
        backend_sock.bind(self._backend_endpoint)

        self._running = True

        # Start follower thread
        self._follower_thread = threading.Thread(
            target=self._run_follower,
            name="RobotServer-Follower",
            daemon=True,
        )
        self._follower_thread.start()

        # Start host timeout thread
        self._timeout_thread = threading.Thread(
            target=self._run_timeout,
            name="RobotServer-Timeout",
            daemon=True,
        )
        self._timeout_thread.start()

        ready_event.set()
        logger.info("RobotServer listening on %s (bound port %d)", bind_addr, self._bound_port)

        poller = zmq.Poller()
        poller.register(self._router_sock, zmq.POLLIN)
        poller.register(backend_sock, zmq.POLLIN)

        try:
            while self._running:
                events = dict(poller.poll(50))
                if self._router_sock in events and events[self._router_sock] == zmq.POLLIN:
                    frames = self._router_sock.recv_multipart()
                    if len(frames) >= 3:
                        identity = frames[0]
                        empty = frames[1]
                        payload = frames[2]
                        self._handle_client_request(identity, empty, payload)

                if backend_sock in events and events[backend_sock] == zmq.POLLIN:
                    frames = backend_sock.recv_multipart()
                    if len(frames) == 3:
                        identity, empty, reply_bytes = frames
                        self._moving = False
                        if self._router_sock is not None:
                            try:
                                self._router_sock.send_multipart([identity, empty, reply_bytes])
                            except Exception as exc:
                                logger.error("Failed to send motion reply: %s", exc)
        except Exception as exc:
            if self._running:
                logger.error("RobotServer main loop error: %s", exc)
        finally:
            self._cleanup(backend_sock)

    def _cleanup(self, backend_sock: zmq.Socket) -> None:
        try:
            backend_sock.close(linger=0)
        except Exception:
            pass
        if self._router_sock is not None:
            try:
                self._router_sock.close(linger=0)
            except Exception:
                pass
            self._router_sock = None
        if self._ctx is not None and not self._ctx.closed:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None

    def stop(self) -> None:
        """Idempotently shut down the server, stop threads, and detach servos."""
        if not self._running:
            try:
                self.driver.detach()
            except Exception:
                pass
            return

        self._running = False
        self._abort_motion.set()

        try:
            self.driver.detach()
        except Exception:
            pass

        if self._follower_thread and self._follower_thread.is_alive():
            self._follower_thread.join(timeout=2.0)
        if self._timeout_thread and self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=2.0)
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)

    def _handle_client_request(
        self, identity: bytes, empty: bytes, payload: bytes
    ) -> None:
        self._last_request_time = time.monotonic()
        try:
            req = msgpack.unpackb(payload, raw=False)
            if not isinstance(req, dict):
                raise ValueError("Payload must be a dict")
            method = req.get("method")
            params = req.get("params", {}) or {}
        except Exception as exc:
            err_reply = {
                "ok": False,
                "error": f"Invalid request envelope: {exc}",
                "error_type": "BadRequest",
            }
            self._send_router_reply(identity, empty, err_reply)
            return

        # Immediate out-of-band methods: estop, clear_estop, ping, get_state, get_calibration, get_frame, get_world
        if method == "ping":
            reply = {
                "ok": True,
                "server": "robot_server",
                "version": 1,
                "driver": self.driver.name,
                "uptime_s": time.time() - self._start_time,
            }
            self._send_router_reply(identity, empty, reply)
            return

        if method == "get_state":
            reply = {
                "ok": True,
                "joint_positions": list(self._current_q),
                "joint_names": ["base_yaw", "shoulder", "elbow", "wrist"],
                "gripper_width_m": float(self._current_gripper_w),
                "attached": bool(self.driver.attached),
                "moving": bool(self._moving),
                "estopped": bool(self._estopped),
                "last_request_age_s": float(time.monotonic() - self._last_request_time),
            }
            self._send_router_reply(identity, empty, reply)
            return

        if method == "estop":
            self._estopped = True
            self._abort_motion.set()
            self.driver.detach()
            logger.info("E-Stop engaged; servos detached")
            reply = {"ok": True, "estopped": True}
            self._send_router_reply(identity, empty, reply)
            return

        if method == "clear_estop":
            self._estopped = False
            self._abort_motion.clear()
            logger.info("E-Stop cleared (servos remain detached until next motion)")
            reply = {"ok": True, "estopped": False}
            self._send_router_reply(identity, empty, reply)
            return

        if method == "get_calibration":
            cal_dict = self.calibration.to_dict()
            reply = {"ok": True, **cal_dict}
            self._send_router_reply(identity, empty, reply)
            return

        if method == "get_frame":
            if not self.fake_camera and self.camera is None:
                reply = {
                    "ok": False,
                    "error": "No camera configured",
                    "error_type": "NoCamera",
                }
            else:
                try:
                    jpeg_bytes, w, h = self._generate_frame()
                    reply = {
                        "ok": True,
                        "jpeg": jpeg_bytes,
                        "width": w,
                        "height": h,
                        "stamp_s": time.time(),
                    }
                except Exception as exc:
                    reply = {
                        "ok": False,
                        "error": str(exc),
                        "error_type": "NoEncoder",
                    }
            self._send_router_reply(identity, empty, reply)
            return

        if method == "get_world":
            if self.driver.name != "fake":
                reply = {
                    "ok": False,
                    "error": "get_world is only supported with fake driver",
                    "error_type": "NotSupported",
                }
            else:
                objects = self._parse_fake_world()
                # Reference: attachment physics uses mfw.hardware.kinematics.PlanarKinematics
                reply = {
                    "ok": True,
                    "objects": objects,
                    "attached": None,
                    "tcp": None,
                }
            self._send_router_reply(identity, empty, reply)
            return

        # Motion methods: home, set_gripper, follow_trajectory
        if method in ("home", "set_gripper", "follow_trajectory"):
            if self._estopped:
                reply = {
                    "ok": False,
                    "error": "Robot is in E-Stop state",
                    "error_type": "EStopped",
                }
                self._send_router_reply(identity, empty, reply)
                return

            if method == "follow_trajectory":
                points = params.get("points")
                if not isinstance(points, list) or len(points) == 0:
                    reply = {
                        "ok": False,
                        "error": "follow_trajectory requires non-empty points list",
                        "error_type": "BadRequest",
                    }
                    self._send_router_reply(identity, empty, reply)
                    return

                for pt in points:
                    q = pt.get("q")
                    if not isinstance(q, (list, tuple)) or len(q) != 4:
                        reply = {
                            "ok": False,
                            "error": f"Trajectory point q must have 4 entries, got {q}",
                            "error_type": "BadRequest",
                        }
                        self._send_router_reply(identity, empty, reply)
                        return

                    for i, q_val in enumerate(q):
                        _, clamped_by = self.calibration.rad_to_pulse(i, float(q_val))
                        if clamped_by > 0.02:
                            reply = {
                                "ok": False,
                                "error": f"Joint {i} target {q_val} exceeds limit by {clamped_by:.4f} rad",
                                "error_type": "JointLimit",
                            }
                            self._send_router_reply(identity, empty, reply)
                            return

            # Check if motion worker is busy
            if self._moving or not self._motion_queue.empty():
                reply = {
                    "ok": False,
                    "error": "A motion is already in progress",
                    "error_type": "Busy",
                }
                self._send_router_reply(identity, empty, reply)
                return

            # Dispatch motion
            self._abort_motion.clear()
            self._moving = True
            self._motion_queue.put((identity, empty, method, params))
            return

        # Unknown method
        reply = {
            "ok": False,
            "error": f"Unknown method: {method}",
            "error_type": "UnknownMethod",
        }
        self._send_router_reply(identity, empty, reply)

    def _send_router_reply(
        self, identity: bytes, empty: bytes, reply_dict: dict[str, Any]
    ) -> None:
        if self._router_sock is not None:
            try:
                payload = msgpack.packb(reply_dict, use_bin_type=True)
                self._router_sock.send_multipart([identity, empty, payload])
            except Exception as exc:
                logger.error("Failed to send ROUTER reply: %s", exc)

    def _run_follower(self) -> None:
        """Worker thread for motion interpolation and periodic keepalive."""
        assert self._ctx is not None
        push_sock = self._ctx.socket(zmq.PUSH)
        push_sock.setsockopt(zmq.LINGER, 0)
        push_sock.connect(self._backend_endpoint)

        try:
            while self._running:
                try:
                    job = self._motion_queue.get(timeout=self.tick_s)
                except queue.Empty:
                    # Idle keepalive tick:
                    # prefer driver.keepalive() when available (UnoSerialDriver sends S,
                    # which counts as a keepalive frame in the firmware without moving anything);
                    # fall back to writing the current pulses (FakeDriver watchdog proof).
                    if self.driver.attached and not self._estopped:
                        if hasattr(self.driver, "keepalive"):
                            self.driver.keepalive()  # type: ignore[union-attr]
                        else:
                            pulses = [
                                int(round(self.calibration.rad_to_pulse(i, self._current_q[i])[0]))
                                for i in range(4)
                            ] + [int(round(self.calibration.width_to_pulse(self._current_gripper_w)))]
                            self.driver.write_pulses(pulses)
                    continue

                identity, empty, method, params = job
                reply = self._execute_motion_job(method, params)
                try:
                    payload = msgpack.packb(reply, use_bin_type=True)
                    push_sock.send_multipart([identity, empty, payload])
                except Exception as exc:
                    logger.error("Failed pushing motion reply to backend: %s", exc)
        finally:
            push_sock.close(linger=0)

    def _execute_motion_job(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.driver.attached:
            self.driver.attach()

        if method == "home":
            duration_s = float(params.get("duration_s", 2.0))
            target_q = [0.0, 0.0, 0.0, 0.0]
            target_w = self.calibration.jaw.width_open_m
            aborted = self._interpolate_to(target_q, target_w, duration_s)
            if aborted or self._estopped:
                return {
                    "ok": False,
                    "error": "Motion aborted by E-stop",
                    "error_type": "EStopped",
                }
            return {
                "ok": True,
                "joint_positions": list(self._current_q),
            }

        if method == "set_gripper":
            width_m = float(params.get("width_m", self.calibration.jaw.width_open_m))
            duration_s = float(params.get("duration_s", 0.5))
            aborted = self._interpolate_to(self._current_q, width_m, duration_s)
            if aborted or self._estopped:
                return {
                    "ok": False,
                    "error": "Motion aborted by E-stop",
                    "error_type": "EStopped",
                }
            return {
                "ok": True,
                "gripper_width_m": float(self._current_gripper_w),
            }

        if method == "follow_trajectory":
            points = params["points"]
            gripper_w = params.get("gripper_width_m")
            aborted = self._execute_trajectory(points, gripper_w)
            if aborted or self._estopped:
                return {
                    "ok": False,
                    "error": "Motion aborted by E-stop",
                    "error_type": "EStopped",
                }
            return {
                "ok": True,
                "joint_positions": list(self._current_q),
                "gripper_width_m": float(self._current_gripper_w),
                "aborted": False,
            }

        return {
            "ok": False,
            "error": f"Unknown motion: {method}",
            "error_type": "BadRequest",
        }

    def _interpolate_to(
        self, target_q: list[float], target_w: float, duration_s: float
    ) -> bool:
        start_q = list(self._current_q)
        start_w = float(self._current_gripper_w)
        steps = max(1, int(round(duration_s / self.tick_s)))

        for step in range(1, steps + 1):
            if self._abort_motion.is_set() or self._estopped:
                return True

            alpha = step / float(steps)
            q_step = [
                start_q[i] + alpha * (target_q[i] - start_q[i]) for i in range(4)
            ]
            w_step = start_w + alpha * (target_w - start_w)

            self._apply_state(q_step, w_step)
            self.follower_sleep(self.tick_s)

        return self._abort_motion.is_set() or self._estopped

    def _execute_trajectory(
        self, points: list[dict[str, Any]], gripper_w: float | None
    ) -> bool:
        t0 = float(points[0]["t"])
        t_end = float(points[-1]["t"])
        total_time = max(0.0, t_end - t0)

        target_w = (
            float(gripper_w)
            if gripper_w is not None
            else float(self._current_gripper_w)
        )
        start_w = float(self._current_gripper_w)

        if total_time <= 0.0 or len(points) == 1:
            if self._abort_motion.is_set() or self._estopped:
                return True
            final_q = [float(x) for x in points[-1]["q"]]
            self._apply_state(final_q, target_w)
            return False

        steps = max(len(points), int(round(total_time / self.tick_s)))
        times = [float(p["t"]) - t0 for p in points]
        q_arr = np.array([p["q"] for p in points], dtype=float)

        for step in range(steps + 1):
            if self._abort_motion.is_set() or self._estopped:
                return True

            elapsed = (step / float(steps)) * total_time

            # Interpolate q at elapsed time
            q_interp = [
                float(np.interp(elapsed, times, q_arr[:, j])) for j in range(4)
            ]
            w_interp = start_w + (elapsed / total_time) * (target_w - start_w)

            self._apply_state(q_interp, w_interp)
            self.follower_sleep(self.tick_s)

        # Ensure exact final point is applied
        if not (self._abort_motion.is_set() or self._estopped):
            final_q = [float(x) for x in points[-1]["q"]]
            self._apply_state(final_q, target_w)

        return self._abort_motion.is_set() or self._estopped

    def _apply_state(self, q: list[float], gripper_w: float) -> None:
        self._current_q = list(q)
        self._current_gripper_w = float(gripper_w)
        pulses = [
            int(round(self.calibration.rad_to_pulse(i, self._current_q[i])[0]))
            for i in range(4)
        ] + [int(round(self.calibration.width_to_pulse(self._current_gripper_w)))]
        self.driver.write_pulses(pulses)

    def _run_timeout(self) -> None:
        """Periodic thread monitoring host activity timeout."""
        while self._running:
            time.sleep(0.05)
            if self.driver.attached and not self._moving:
                age = time.monotonic() - self._last_request_time
                if age > self.host_timeout_s:
                    logger.warning(
                        "Host inactivity timeout (age=%.2fs > %.2fs); detaching servos",
                        age,
                        self.host_timeout_s,
                    )
                    self.driver.detach()

    def _generate_frame(self) -> tuple[bytes, int, int]:
        """Generate a synthetic 640x480 JPEG frame."""
        width, height = 640, 480
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (width, height), color=(128, 128, 128))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "fake-camera", fill=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue(), width, height
        except ImportError:
            pass

        try:
            import cv2

            img = np.full((height, width, 3), 128, dtype=np.uint8)
            cv2.putText(
                img,
                "fake-camera",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            success, enc = cv2.imencode(".jpg", img)
            if not success:
                raise RuntimeError("cv2.imencode failed")
            return enc.tobytes(), width, height
        except ImportError:
            pass

        raise RuntimeError("No JPEG encoder available (install Pillow or OpenCV)")

    def _parse_fake_world(self) -> dict[str, list[float]]:
        """Parse the --fake-world string into a dictionary of object positions."""
        if not self.fake_world:
            return {}

        if isinstance(self.fake_world, dict):
            return self.fake_world

        result = {}
        for token in str(self.fake_world).strip().split():
            if ":" not in token:
                continue
            name, _, coords_str = token.partition(":")
            coords = [float(c.strip()) for c in coords_str.split(",") if c.strip()]
            if len(coords) == 2:
                coords.append(0.0)
            result[name] = coords
        return result


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson / Fake Robot Server for Hardware Arm Control"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5560, help="Bind port (default: 5560)")
    parser.add_argument(
        "--driver",
        choices=["fake", "uno_serial", "pca9685"],
        default="fake",
        help="Servo driver implementation (default: fake)",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to JSON servo calibration file",
    )
    parser.add_argument(
        "--serial-port",
        default="/dev/ttyACM0",
        help="Serial port for UnoSerialDriver (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200)",
    )
    parser.add_argument(
        "--host-timeout-s",
        type=float,
        default=5.0,
        help="Inactivity timeout in seconds before detaching servos (default: 5.0)",
    )
    parser.add_argument(
        "--fake-camera",
        action="store_true",
        help="Enable synthetic camera frames (640x480 JPEG)",
    )
    parser.add_argument(
        "--fake-world",
        default=None,
        help="Space-separated list of name:x,y[,z] objects (e.g. 'marker:0.18,0.05 bowl:0.15,-0.12')",
    )
    parser.add_argument(
        "--fake-watchdog-ms",
        type=float,
        default=None,
        help="Watchdog timeout in ms for the fake AND uno_serial drivers (default: None / firmware default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.calibration:
        calibration = ServoCalibration.from_json(args.calibration)
    else:
        calibration = ServoCalibration.default()

    if args.driver == "fake":
        driver = FakeDriver(calibration, watchdog_ms=args.fake_watchdog_ms)
    elif args.driver == "uno_serial":
        watchdog_ms = int(args.fake_watchdog_ms) if args.fake_watchdog_ms is not None else 500
        driver = UnoSerialDriver(
            port=args.serial_port,
            baud=args.baud,
            watchdog_ms=watchdog_ms,
        )
        driver.open(calibration=calibration)
    elif args.driver == "pca9685":
        driver = Pca9685Driver()
    else:
        raise ValueError(f"Unknown driver: {args.driver}")

    server = RobotServer(
        host=args.host,
        port=args.port,
        driver=driver,
        calibration=calibration,
        host_timeout_s=args.host_timeout_s,
        fake_camera=args.fake_camera,
        fake_world=args.fake_world,
    )

    def sig_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %d; shutting down...", signum)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

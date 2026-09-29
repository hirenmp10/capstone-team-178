"""Jetson-side robot server: ZMQ front door for the hobby arm and the webcam.

Runs **standalone on the Jetson** (``python3 jetson/robot_server.py``) and must
never import ``mfw`` -- the laptop's framework is not installed there. Hard
dependencies: numpy, pyyaml, pyzmq, msgpack, msgpack-numpy. ``pyserial``,
``smbus2`` and ``cv2`` are imported lazily inside the driver / camera that
needs them, so ``--driver fake --no-camera`` runs on a bare interpreter.

Why this process exists
-----------------------
The servos have no position feedback and the Uno only understands pulse widths.
Everything that turns *radians* into *microseconds* -- pulse-angle maps, sign,
zero offsets, per-joint limits, gripper width -- is calibration measured on the
assembled arm, and it lives **here**, in ``robot_config.yaml`` next to the arm,
never in the laptop's ``configs/``. The laptop speaks radians and metres and is
told how far it actually got.

Wire protocol (frozen end of week 1; the laptop client is
``mfw/hardware/jetson_client.py``): ZMQ, msgpack + msgpack_numpy framing,
request ``{"endpoint": str, "data": dict}`` -> reply dict or ``{"error": str}``.
Endpoint table in ``docs/HARDWARE_BRIEF.md`` section 5.

Measured traps this file is shaped around
-----------------------------------------
* A plain REP socket cannot answer ``estop`` while ``follow_trajectory`` is
  blocking, and a trajectory blocks for seconds. The server therefore binds a
  ROUTER socket: motion endpoints run in a worker thread and reply later,
  while ``estop``/``get_state`` from a second client are answered at once.
  REQ clients see no difference.
* Servos re-attach on the first frame after a detach and jump to the target at
  full speed. After ``estop`` the arm may have dropped, so ``clear_estop``
  does **not** re-attach; the next motion command does, and ``home`` is the
  right next call.
* The Uno resets when the serial port opens (DTR). Bytes sent in the first
  ~2 s are lost silently, so the driver waits before the first frame.
* The commanded state is *derived from the stored pulses through the current
  calibration* rather than cached in radians. That way ``set_calibration``
  (zero offsets, direction flips) re-labels the pose without moving anything.
* The Uno detaches every servo after ``WATCHDOG_MS`` (500 ms) without a frame,
  and every normal pause in a pick (settle, perception, the student talking)
  is longer than that. A frame is only written when a command moves, so the
  arm went limp between commands and snapped back on the next one. Every
  driver therefore runs a keepalive thread (:meth:`ServoDriver.start_keepalive`)
  that feeds the watchdog while attached and idle; ``get_state`` asks the Uno
  whether it still is attached and believes the Uno over its own flag.
* A re-attach is a jump: feedback-less servos go from wherever gravity left
  them to the first pulse at full speed, and nothing on the Jetson can slew
  that. What can be avoided is the *second* jump -- the old code re-attached
  at the pre-detach pose and then streamed a fresh trajectory at once, and
  the old sketch started every re-attach *at the new target* so even a slow
  ``T`` frame snapped. After any detach (estop, torque off, watchdog, serial
  loss) the first motion endpoint -- ``follow_trajectory``, ``set_gripper``,
  ``home`` and ``set_torque`` alike, because every one of them re-energises
  all five channels -- now goes through :meth:`ServoDriver.reattach`: the
  stored pose is re-energised and the move to the target takes
  ``reattach_s`` (``home_move_s`` for ``home``) as one ``T`` frame on the
  Uno, fed with ``K`` frames while it interpolates, or a software ramp
  elsewhere. The sketch keeps its last pulsed position across a detach and
  slews from there. ``write`` refuses to run while the hold is in force, so
  no forgotten path can re-attach with a bare ``P`` frame.
* The sketch clamps every pulse to ``MIN_US``/``MAX_US`` and still answers
  ``OK``. A calibration that reaches past that range is executed *wrong*
  silently, so :data:`SKETCH_ARM_PULSE_US` mirrors the sketch and
  ``validate`` refuses limits the servo cannot reach; the follower reports
  ``clamped`` from the pulse level, not just the radian limits.
* A serial reply that arrives after ``reply_timeout_s`` is not lost: it sits
  in the input buffer and every later exchange reads the *previous* frame's
  reply. Draining alone does not fix it -- measured on a FIFO wire with real
  arrival times, a reply 1-20 ms late is still *in flight* when the buffer is
  drained, the resent frame reads it, and 16 of the next 17 reads returned
  the previous frame's reply for good. On a timeout (or a reply of the wrong
  kind) the driver therefore drains, sends ``S<nonce>`` and reads until the
  sketch echoes that nonce: the Uno answers in order, so everything queued
  before the echo is stale and skipped, and the stream is provably in step
  before the frame is resent. If the *resent* frame is answered late too
  (audit Missed-7: the stream then stayed one frame behind for good, because
  a late ``OK`` looks exactly like the next frame's ``OK``), the exchange
  fails and the stream is marked out of step: the next exchange syncs
  *before* it writes anything, and sends nothing (except a ``D``) until a
  sync succeeds. After :data:`SERIAL_MAX_TIMEOUTS` consecutive failures it
  declares the servos detached.
* ``robot_config.yaml`` says ``measured: false`` until S5 and
  ``calibrate_servos.py`` have measured it. A real driver (``uno``,
  ``pca9685``) refuses to start on it; ``--allow-placeholder-calibration``
  (first bench bring-up only) starts anyway with every pulse kept inside
  :data:`PLACEHOLDER_PULSE_BAND_US` and the speed capped
  (:func:`placeholder_safe_calibration`).
* One lost reply must not trip the watchdog: the driver lock is held for the
  whole exchange, so the keepalive waits behind it. The reply timeout is
  :data:`DEFAULT_REPLY_TIMEOUT_S` (a reply takes ~3 ms at 115200) and the
  sketch counts ``?`` and ``S`` as proof of a live host, so even a lost frame
  *and* a lost sync leave the feeding gap under the 500 ms watchdog
  (``tests/test_hardware_bridge.py`` pins the budget).
* The Uno resets whenever the port opens (DTR), so at every server start --
  and after any USB re-enumeration or brownout -- the bridge knows no
  position: its first frame attaches every servo AT that frame's target at
  full speed, and a ``T`` frame's duration has nothing to interpolate from.
  The driver tracks that (``position_known``; the sketch also reports it in
  its ``Q`` reply) and the server refuses every motion but ``home`` until the
  first attach, so ``calibrate_servos.py``'s ``torque on`` can no longer
  snap five servos to 1500 us while promising that nothing moves. ``home``
  from an unknown position is the documented hand-posed boot: the reply
  says ``first_attach: true`` and the log says it was a jump.
* The keepalive holds the arm for as long as the server runs, including the
  jaw's squeeze on a held object. A crashed laptop or a lost Wi-Fi link must
  not mean "forever": after ``host_timeout_s`` (``robot_config.yaml``, 5 s)
  without a request from any client, and with no motion running, the server
  detaches the servos (the Uno's watchdog would follow anyway once the
  keepalive stops) and says so; the laptop's ``RemoteArm`` sends a
  ``get_state`` heartbeat every second from its own thread, so only a laptop
  process that died, lost its network or froze entirely trips it (a main
  loop stuck in a long call still heartbeats). The fake detector's
  ``get_fake_world`` polling does not count as the laptop. The next motion
  re-attaches slowly as after any detach.
* Every detach -- estop, torque off, bridge watchdog, serial loss, host
  timeout -- increments ``detach_count`` with a ``last_detach_reason``, and
  every motion reply says whether it had to ``reattached`` first. A detach
  repaired inside the next motion call used to be invisible to the laptop
  (the reply said ``detached: false``) even though a held object may have
  dropped in the gap.
* ``home_q`` is driven to at every boot by an endpoint with no planner in
  the loop, so a home recorded with the forearm below the table would be
  executed at every start. :func:`check_home_clearance` runs the same planar
  FK as the fake world on the server *and* in ``calibrate_servos.py``.

Fake world (``--driver fake --fake-world "marker:0.18,0.05 bowl:0.15,-0.12"``)
------------------------------------------------------------------------------
Two fake services in two processes cannot agree on a grasp unless one of them
owns the objects. This server does: :class:`FakeWorld` watches every pulse the
fake driver writes, computes the jaw midpoint with the same planar model the
laptop plans with (link lengths passed on the command line, defaults equal to
``configs/hardware.yaml``), attaches an object when the jaw closes on it and
carries it with the TCP. The scripted detector reads the live positions
through the ``get_fake_world`` endpoint (``detector_service.py --world-from``),
so a lifted object rises out of its view exactly as it would on a real table.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "JOINT_NAMES",
    "GRIPPER_NAME",
    "WIRE_ORDER",
    "SERVER_NAME",
    "PROTOCOL_VERSION",
    "SKETCH_ARM_PULSE_US",
    "SKETCH_GRIPPER_PULSE_US",
    "SKETCH_PULSE_LIMITS_US",
    "SKETCH_WATCHDOG_MS",
    "KEEPALIVE_PERIOD_S",
    "SERIAL_MAX_TIMEOUTS",
    "DEFAULT_REPLY_TIMEOUT_S",
    "DETACH_ATTEMPTS",
    "HOST_TIMEOUT_S",
    "SYNC_MAX_LINES",
    "CLAMP_TOL_RAD",
    "TABLE_CLEARANCE_M",
    "PLACEHOLDER_PULSE_BAND_US",
    "PLACEHOLDER_MAX_JOINT_VELOCITY",
    "PLACEHOLDER_MIN_REATTACH_S",
    "PLACEHOLDER_MIN_HOME_MOVE_S",
    "PLACEHOLDER_HOME_Q",
    "DEFAULT_MAX_FRAME_AGE_S",
    "placeholder_safe_calibration",
    "startup_calibration",
    "DriverError",
    "CalibrationError",
    "encode_p_frame",
    "encode_t_frame",
    "encode_s_frame",
    "decode_command_frame",
    "decode_q_frame",
    "decode_q_status",
    "BridgeStatus",
    "parse_reply",
    "JointCalibration",
    "GripperCalibration",
    "CameraSettings",
    "ServoCalibration",
    "check_home_clearance",
    "ServoDriver",
    "FakeDriver",
    "UnoSerialDriver",
    "Pca9685Driver",
    "TrajectoryFollower",
    "CameraGrabber",
    "ArmGeometry",
    "planar_tcp",
    "planar_link_points",
    "FakeWorld",
    "RobotServer",
    "main",
]

_log = logging.getLogger("robot_server")

SKETCH_ARM_PULSE_US: tuple[int, int] = (600, 2400)
"""Pulse clamp of the four arm channels in ``servo_bridge.ino`` (``MIN_US`` /
``MAX_US`` entries 0-3). MUST equal the sketch: it clamps silently and still
answers ``OK``, so a calibration allowed to reach past this range would report
a pulse as executed while the servo sat at the clamp, and a limit recorded
there would be a lie. ``JointCalibration.validate`` refuses such a range; when
S5 measures tighter end-stops, tighten this table AND the sketch together.
``tests/test_hardware_bridge.py`` parses the sketch and pins the two."""

SKETCH_GRIPPER_PULSE_US: tuple[int, int] = (900, 2100)
"""Same for the gripper channel (``MIN_US[4]`` / ``MAX_US[4]``)."""

SKETCH_PULSE_LIMITS_US: tuple[tuple[int, int], ...] = (SKETCH_ARM_PULSE_US,) * 4 + (
    SKETCH_GRIPPER_PULSE_US,
)
"""Per-channel ``(min_us, max_us)`` in wire order, exactly the sketch's tables."""

SKETCH_WATCHDOG_MS = 500
"""``WATCHDOG_MS`` in the sketch: silence longer than this detaches every
servo. Pinned by the test suite against the sketch source, like the pulse
tables, so the keepalive period below cannot drift past it."""

KEEPALIVE_PERIOD_S = 0.15
"""Idle interval after which a driver with a watchdog re-feeds it. The thread
polls at half this period, so the worst-case silence is 1.5 periods (225 ms)
plus one serial round trip -- under half the 500 ms watchdog, which leaves
room for a GC pause or a slow ``?`` reply without a trip."""

SERIAL_MAX_TIMEOUTS = 3
"""Consecutive failed exchanges after which :class:`UnoSerialDriver` declares
the servos detached. One timeout is repaired by a sync (``S<nonce>``) and a
resend; three failed exchanges in a row mean the Uno is gone or reset, and its
own watchdog has already dropped the arm."""

DEFAULT_REPLY_TIMEOUT_S = 0.1
"""Serial reply timeout. A reply is ~3 ms of line time at 115200 plus USB
latency; 0.1 s is a >10x margin. It used to be 0.25 s, and a lost ``?`` reply
then held the driver lock for 2 x 0.25 s while the keepalive waited behind
it -- longer than the 500 ms watchdog (review, bridge lens: 7 of 8 phases
tripped). Budget now: 1.5 keepalive periods + a lost frame + a lost sync =
0.225 + 0.1 + 0.1 s < 0.5 s, and the sketch feeds on ``?``/``S`` besides."""

SYNC_MAX_LINES = 8
"""Most lines read while waiting for the ``S<nonce>`` echo. Only one frame is
ever outstanding, so at most one stale reply precedes the echo; the slack
covers noise lines."""

DETACH_ATTEMPTS = 3
"""Times ``D`` is sent before a detach is declared failed. A ``D`` corrupted
on the wire is answered ``ERR unknown command`` -- an in-step reply, so the
exchange does not resend it -- and the arm used to stay energised under the
keepalive (review, bridge lens). The hold is raised *before* the first
attempt, so even a detach that fails every time stops the keepalive feeding
and the Uno's watchdog relaxes the arm within 500 ms."""

HOST_TIMEOUT_S = 5.0
"""Default ``host_timeout_s``: seconds without any client request after which
an idle server detaches the servos (see the module docstring)."""

CLAMP_TOL_RAD = 1e-6
"""Radians below which a limit clamp is not a clamp. Waypoints cross the wire
as float32 (``JetsonClient.follow_trajectory``), which perturbs a value that
sits exactly on a limit by up to ~1e-7 rad; the old 1e-9 threshold flagged
about half of all measured limit values as 'clamped' on a perfectly legal
move. 1e-6 rad is 0.006 us on the placeholder map: invisible on the wire."""

PLACEHOLDER_PULSE_BAND_US: tuple[int, int] = (1000, 2000)
"""Pulse band every channel is kept inside while ``robot_server.py`` drives
real servos with an UNMEASURED calibration (``measured: false`` in
``robot_config.yaml``) under ``--allow-placeholder-calibration``: +-50 deg
around each servo's centre on the placeholder 600..2400 us map, far from any
end-stop a kit servo might have. Nothing measured says where the end-stops
are, and a servo pushed into one stalls at 1.4-2.5 A. ``--placeholder-band-us``
widens it deliberately (e.g. 600,2400 for smoke test S5 with the linkages
off); smoke-test S5 measurements replace the placeholder and lift it."""

PLACEHOLDER_HOME_Q: tuple[float, float, float, float] = (0.0, 0.0, -0.6109, 0.1745)
"""Placeholder ``home_q``: upper arm vertical, forearm folded 35 deg back,
wrist 10 deg forward (jaws up-and-back) -- the TCP ~10 cm behind the base and
~34 cm high, out of the overhead camera's workspace, every link far above the
table. On the placeholder map that is 1500/1500/1150/1600 us: at least 550 us
inside every pulse extreme and 150 us inside
:data:`PLACEHOLDER_PULSE_BAND_US`. The previous placeholder put the elbow at
-90 deg = 600 us, exactly the pulse extreme, which on a real MG996R may sit
at or past the mechanical end-stop and stall at every boot (home is the first
frame after every server start). Smoke test S5 end-stop measurements and
``calibrate_servos.py`` ``home set`` replace it; mirror the value in
``robot.home_joint_positions`` of ``configs/hardware.yaml``."""

PLACEHOLDER_MAX_JOINT_VELOCITY = 0.35
"""rad/s ceiling in placeholder mode (the placeholder file says 1.2)."""

PLACEHOLDER_MIN_REATTACH_S = 2.0
"""Floor on ``reattach_s`` in placeholder mode."""

PLACEHOLDER_MIN_HOME_MOVE_S = 4.0
"""Floor on ``home_move_s`` in placeholder mode."""

DEFAULT_MAX_FRAME_AGE_S = 1.0
"""``get_frame`` refuses a frame older than this (on the Jetson's monotonic
clock) unless the request passes its own ``max_age_s``. The grabber thread
refreshes the frame at the camera rate (30 fps); a frame a second old means
the camera stalled or was unplugged, and a detection on it would place an
object where it was, not where it is."""

TABLE_CLEARANCE_M = 0.01
"""Every link point after the shoulder must sit at least this high above the
table for a pose to be accepted as ``home_q``. Home is driven to at every boot
by an endpoint with no planner in the loop, so a home below the table would
push the forearm through it on every start."""

JOINT_NAMES: tuple[str, str, str, str] = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
)
"""Arm joints in wire order. Matches ``robot.arm_joint_names`` in
``configs/hardware.yaml`` and the Uno's D3/D5/D6/D9 pins. Never reorder."""

GRIPPER_NAME = "gripper"
WIRE_ORDER: tuple[str, ...] = JOINT_NAMES + (GRIPPER_NAME,)
"""The five channels of a P/T frame, in order."""

SERVER_NAME = "mfw-robot"
PROTOCOL_VERSION = 1

_N_ARM = len(JOINT_NAMES)
_N_WIRE = len(WIRE_ORDER)
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "robot_config.yaml"


class DriverError(RuntimeError):
    """The servo bridge rejected or failed a command."""


class CalibrationError(ValueError):
    """The calibration file is missing, malformed or self-contradictory."""


# ---------------------------------------------------------------------------
# Uno frame helpers -- pure functions, shared by the driver and the tests.
# ---------------------------------------------------------------------------


def _check_pulses(us: Sequence[int]) -> tuple[int, ...]:
    if len(us) != _N_WIRE:
        raise ValueError(f"a frame carries {_N_WIRE} pulses, got {len(us)}")
    out = []
    for value in us:
        v = int(round(float(value)))
        if v < 0:
            raise ValueError(f"pulse widths are non-negative, got {v}")
        out.append(v)
    return tuple(out)


def encode_p_frame(us: Sequence[int]) -> bytes:
    """``P<us>x5\\n``: set targets, slewed by the Uno at 20 us per 20 ms tick."""
    return ("P" + ",".join(str(v) for v in _check_pulses(us)) + "\n").encode("ascii")


def encode_t_frame(us: Sequence[int], ms: int) -> bytes:
    """``T<us>x5,<ms>\\n``: reach the targets by linear interpolation over ``ms``."""
    ms_i = int(round(float(ms)))
    if ms_i < 0:
        raise ValueError(f"move time must be >= 0 ms, got {ms_i}")
    return ("T" + ",".join(str(v) for v in _check_pulses(us)) + f",{ms_i}\n").encode(
        "ascii"
    )


def encode_s_frame(nonce: int) -> bytes:
    """``S<nonce>\\n``: the sketch echoes ``S<nonce>`` (serial resync)."""
    n = int(nonce)
    if n < 0:
        raise ValueError(f"sync nonce must be >= 0, got {n}")
    return f"S{n}\n".encode("ascii")


def decode_command_frame(line: bytes | str) -> tuple[Any, ...]:
    """Parse a host->Uno frame exactly as the sketch does.

    Returns ``("P", us)``, ``("T", us, ms)``, ``("?",)``, ``("D",)``,
    ``("K",)`` (keepalive: refreshes the watchdog, moves nothing) or
    ``("S", nonce)`` (sync: echoed verbatim). Raises ``ValueError`` with the
    sketch's ``ERR`` wording on a bad frame, so the test suite pins both ends
    of the protocol to one parser.
    """
    text = line.decode("ascii") if isinstance(line, bytes) else line
    text = text.rstrip("\r\n")
    if not text:
        raise ValueError("ERR bad frame")
    cmd, body = text[0], text[1:]
    if cmd == "?":
        return ("?",)
    if cmd == "D":
        return ("D",)
    if cmd == "K":
        if body:
            raise ValueError("ERR bad frame")
        return ("K",)
    if cmd == "S":
        if not body or not body.isdigit():
            raise ValueError("ERR bad frame")
        return ("S", int(body))
    if cmd in ("P", "T"):
        want = _N_WIRE + 1 if cmd == "T" else _N_WIRE
        parts = body.split(",")
        if len(parts) != want or not all(p.isdigit() for p in parts):
            raise ValueError("ERR bad frame")
        values = [int(p) for p in parts]
        if cmd == "T":
            return ("T", tuple(values[:_N_WIRE]), values[_N_WIRE])
        return ("P", tuple(values))
    raise ValueError(f"ERR unknown command {cmd}")


@dataclass(frozen=True)
class BridgeStatus:
    """The Uno's answer to ``?``."""

    pulses: tuple[int, ...]
    """What the bridge is pulsing (or last pulsed, when detached), wire order."""
    attached: bool
    position_known: bool | None
    """``have_position``: False on a freshly reset Uno, whose 1500 us are a
    library default rather than a position. ``None`` from a sketch older than
    the 7-field ``Q`` reply, which cannot tell."""


def decode_q_status(line: bytes | str) -> BridgeStatus:
    """Parse ``Q<us>x5,<attached>[,<known>]``; the 6-field form is the old sketch."""
    text = line.decode("ascii") if isinstance(line, bytes) else line
    text = text.strip()
    if not text.startswith("Q"):
        raise DriverError(f"expected a Q frame, got {text!r}")
    parts = text[1:].split(",")
    if len(parts) not in (_N_WIRE + 1, _N_WIRE + 2) or not all(p.strip().isdigit() for p in parts):
        raise DriverError(f"malformed Q frame {text!r}")
    values = [int(p) for p in parts]
    known = bool(values[_N_WIRE + 1]) if len(values) == _N_WIRE + 2 else None
    return BridgeStatus(tuple(values[:_N_WIRE]), bool(values[_N_WIRE]), known)


def decode_q_frame(line: bytes | str) -> tuple[tuple[int, ...], bool]:
    """Parse the Uno's ``Q`` reply into ``(pulses, attached)`` (see :func:`decode_q_status`)."""
    status = decode_q_status(line)
    return status.pulses, status.attached


def parse_reply(line: bytes | str) -> None:
    """Raise :class:`DriverError` unless the Uno answered ``OK``."""
    text = line.decode("ascii", errors="replace") if isinstance(line, bytes) else line
    text = text.strip()
    if text == "OK":
        return
    if text.startswith("ERR"):
        raise DriverError(f"Uno rejected the frame: {text}")
    if not text:
        raise DriverError("no reply from the Uno (unplugged, wrong port, or still resetting)")
    raise DriverError(f"unexpected reply from the Uno: {text!r}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointCalibration:
    """Pulse <-> angle map for one arm servo.

    The servo's own angle is linear in pulse width between
    ``(pulse_min_us, angle_at_min_deg)`` and ``(pulse_max_us, angle_at_max_deg)``.
    The *joint* angle the planner uses is ``q = direction * (servo_rad -
    zero_offset_rad)``; ``zero_offset_rad`` is the servo angle at which the
    joint reads zero, ``direction`` flips a servo mounted mirror-image.
    ``limit_*_rad`` are the joint-space limits enforced on every command.

    Both limits must map *inside* the pulse range: a 180-degree servo cannot
    span a 240-degree kinematic range whatever the zero offset, and a limit
    the pulse map cannot reach is executed at the clamp with the planner none
    the wiser (measured: a top-down grasp at r = 0.10 m asked the elbow for
    119 deg, got 90, and the jaw closed 88 mm from the object).
    """

    pulse_min_us: int = 600
    pulse_max_us: int = 2400
    angle_at_min_deg: float = -90.0
    angle_at_max_deg: float = 90.0
    zero_offset_rad: float = 0.0
    direction: int = 1
    limit_lower_rad: float = -1.5708
    limit_upper_rad: float = 1.5708

    def validate(self, name: str) -> None:
        if self.pulse_min_us >= self.pulse_max_us:
            raise CalibrationError(
                f"joints.{name}: pulse_min_us ({self.pulse_min_us}) must be < "
                f"pulse_max_us ({self.pulse_max_us})"
            )
        lo, hi = SKETCH_ARM_PULSE_US
        if self.pulse_min_us < lo or self.pulse_max_us > hi:
            raise CalibrationError(
                f"joints.{name}: pulse range {self.pulse_min_us}..{self.pulse_max_us} us reaches "
                f"past the Uno's clamp {lo}..{hi} us (MIN_US/MAX_US in servo_bridge.ino); the "
                "sketch would clamp silently and answer OK. Widen both or neither."
            )
        if math.isclose(self.angle_at_min_deg, self.angle_at_max_deg):
            raise CalibrationError(f"joints.{name}: angle_at_min_deg == angle_at_max_deg")
        if self.direction not in (1, -1):
            raise CalibrationError(f"joints.{name}: direction must be 1 or -1")
        if self.limit_lower_rad >= self.limit_upper_rad:
            raise CalibrationError(
                f"joints.{name}: limit_lower_rad must be < limit_upper_rad"
            )
        for label in ("limit_lower_rad", "limit_upper_rad"):
            limit = getattr(self, label)
            if not self.fits_pulse_map(limit):
                raise CalibrationError(
                    f"joints.{name}: {label} = {limit:.4f} rad ({math.degrees(limit):.1f} deg) maps to "
                    f"{self.rad_to_us_unclamped(limit):.0f} us, outside the pulse map "
                    f"{self.pulse_min_us}..{self.pulse_max_us} us: the servo cannot reach it and "
                    "every command near it would be clamped silently. Tighten the limit (and "
                    "hardware.arm.joint_lower/upper on the laptop) or re-measure angle_at_min/max_deg."
                )

    # Servo-angle helpers (radians in the servo's own frame).
    def _servo_rad_to_us_unclamped(self, servo_rad: float) -> float:
        deg = math.degrees(servo_rad)
        frac = (deg - self.angle_at_min_deg) / (self.angle_at_max_deg - self.angle_at_min_deg)
        return float(self.pulse_min_us + frac * (self.pulse_max_us - self.pulse_min_us))

    def _servo_rad_to_us(self, servo_rad: float) -> float:
        us = self._servo_rad_to_us_unclamped(servo_rad)
        return float(min(max(us, self.pulse_min_us), self.pulse_max_us))

    def _us_to_servo_rad(self, us: float) -> float:
        frac = (float(us) - self.pulse_min_us) / (self.pulse_max_us - self.pulse_min_us)
        deg = self.angle_at_min_deg + frac * (self.angle_at_max_deg - self.angle_at_min_deg)
        return math.radians(deg)

    def rad_to_us(self, rad: float) -> float:
        """Joint radians -> pulse width (float us, clamped to the pulse range)."""
        return self._servo_rad_to_us(self.direction * float(rad) + self.zero_offset_rad)

    def rad_to_us_unclamped(self, rad: float) -> float:
        """Joint radians -> the pulse the map *asks for*, before the range clamp."""
        return self._servo_rad_to_us_unclamped(self.direction * float(rad) + self.zero_offset_rad)

    def fits_pulse_map(self, rad: float) -> bool:
        """True when ``rad`` lands on the wire unchanged.

        The wire carries integer microseconds, so the honest test is whether
        the *rounded* pulse lies inside the map: the placeholder limit of
        1.5708 rad asks for 2400.02 us, rounds to 2400 and is not a clamp.
        """
        us = int(round(self.rad_to_us_unclamped(rad)))
        return self.pulse_min_us <= us <= self.pulse_max_us

    def us_to_rad(self, us: float) -> float:
        """Pulse width -> joint radians (inverse of :meth:`rad_to_us`)."""
        return self.direction * (self._us_to_servo_rad(us) - self.zero_offset_rad)


@dataclass(frozen=True)
class GripperCalibration:
    """Two measured points: the pulse at which the jaws are fully open / closed."""

    pulse_open_us: int = 1000
    pulse_closed_us: int = 1900
    width_open_m: float = 0.045
    width_closed_m: float = 0.0

    def validate(self) -> None:
        if self.pulse_open_us == self.pulse_closed_us:
            raise CalibrationError("gripper: pulse_open_us == pulse_closed_us")
        lo, hi = SKETCH_GRIPPER_PULSE_US
        for name in ("pulse_open_us", "pulse_closed_us"):
            value = getattr(self, name)
            if value < lo or value > hi:
                raise CalibrationError(
                    f"gripper.{name} = {value} us is outside the Uno's gripper clamp {lo}..{hi} us "
                    "(MIN_US[4]/MAX_US[4] in servo_bridge.ino)"
                )
        if self.width_open_m <= self.width_closed_m:
            raise CalibrationError("gripper: width_open_m must be > width_closed_m")
        if self.width_closed_m < 0:
            raise CalibrationError("gripper: width_closed_m must be >= 0")

    def width_to_us(self, width_m: float) -> float:
        span = self.width_open_m - self.width_closed_m
        frac = (float(width_m) - self.width_closed_m) / span
        frac = min(max(frac, 0.0), 1.0)
        return float(self.pulse_closed_us + frac * (self.pulse_open_us - self.pulse_closed_us))

    def us_to_width(self, us: float) -> float:
        frac = (float(us) - self.pulse_closed_us) / (self.pulse_open_us - self.pulse_closed_us)
        frac = min(max(frac, 0.0), 1.0)
        return float(self.width_closed_m + frac * (self.width_open_m - self.width_closed_m))


@dataclass(frozen=True)
class CameraSettings:
    """V4L2 capture settings for the overhead webcam."""

    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    jpeg_quality: int = 85

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise CalibrationError("camera: width, height and fps must be > 0")
        if not 1 <= self.jpeg_quality <= 100:
            raise CalibrationError("camera.jpeg_quality must be in 1..100")


@dataclass(frozen=True)
class ServoCalibration:
    """Everything the Jetson needs to turn radians and metres into pulses.

    Immutable; :meth:`replace` and :meth:`from_dict` build new instances so a
    half-applied edit can never leave the server with a torn calibration.
    """

    joints: Mapping[str, JointCalibration] = field(
        default_factory=lambda: {name: JointCalibration() for name in JOINT_NAMES}
    )
    gripper: GripperCalibration = GripperCalibration()
    channels: Mapping[str, int] = field(
        default_factory=lambda: {name: i for i, name in enumerate(WIRE_ORDER)}
    )
    """Servo channel per name. Uno: the index into the P frame (0..4, fixed
    by the pin table). PCA9685: the board channel."""
    home_q: tuple[float, float, float, float] = PLACEHOLDER_HOME_Q
    control_rate_hz: float = 50.0
    max_joint_velocity: float = 1.2
    """rad/s; a trajectory faster than this is stretched, never truncated."""
    reattach_s: float = 1.0
    """Seconds the first motion after a detach (estop, torque off, watchdog)
    spends re-energising at the stored pose before the trajectory proper. The
    servos jump from wherever gravity left the arm to that pose regardless;
    this gives them time to arrive before anything else is asked of them."""
    home_move_s: float = 2.5
    """Seconds a ``home`` from the detached state takes to interpolate from the
    last pulsed pose to ``home_q`` (stretched further if max_joint_velocity
    demands it). NOT at boot: a freshly reset Uno knows no position, and its
    first frame attaches AT home instantly (see the module docstring)."""
    host_timeout_s: float = HOST_TIMEOUT_S
    """Seconds without any client request after which an idle server detaches
    the servos (laptop crash, Wi-Fi loss). 0 disables it -- never on the arm."""
    camera: CameraSettings = CameraSettings()
    measured: bool = False
    """True once the pulse maps, zero offsets, limits, gripper points and
    ``home_q`` were measured on the assembled arm (smoke test S5 and
    ``calibrate_servos.py``; its ``measured yes`` + ``save`` sets it). A real
    driver refuses to start on ``measured: false`` unless
    ``--allow-placeholder-calibration`` is given, which then applies
    :func:`placeholder_safe_calibration`. Missing from a file means false."""

    # -- construction ------------------------------------------------------

    @classmethod
    def default(cls) -> "ServoCalibration":
        """Placeholder calibration sized to the kit; measured values replace it.

        Every limit is +-90 deg or tighter: that is all a 600..2400 us map can
        reach, and a wider limit here (the elbow used to say +-120 deg) is a
        promise the servo silently breaks.
        """
        joints = {
            "base_yaw": JointCalibration(limit_lower_rad=-1.5708, limit_upper_rad=1.5708),
            "shoulder_pitch": JointCalibration(limit_lower_rad=-0.5236, limit_upper_rad=1.5708),
            "elbow_pitch": JointCalibration(limit_lower_rad=-1.5708, limit_upper_rad=1.5708),
            "wrist_pitch": JointCalibration(limit_lower_rad=-1.5708, limit_upper_rad=1.5708),
        }
        cal = cls(joints=joints)
        cal.validate()
        return cal

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], geometry: "ArmGeometry | None" = None
    ) -> "ServoCalibration":
        """Build from the YAML structure (see ``robot_config.yaml``).

        With ``geometry`` the home pose is also checked against the table
        (:func:`check_home_clearance`).
        """
        if not isinstance(raw, Mapping):
            raise CalibrationError(f"calibration must be a mapping, got {type(raw).__name__}")
        joints_raw = raw.get("joints", {})
        joints: dict[str, JointCalibration] = {}
        for name in JOINT_NAMES:
            entry = joints_raw.get(name, {}) if isinstance(joints_raw, Mapping) else {}
            joints[name] = _build(JointCalibration, entry, f"joints.{name}")
        gripper = _build(GripperCalibration, raw.get("gripper", {}), "gripper")
        camera = _build(CameraSettings, raw.get("camera", {}), "camera")
        channels_raw = raw.get("channels", {})
        channels = {name: i for i, name in enumerate(WIRE_ORDER)}
        if isinstance(channels_raw, Mapping):
            for name, ch in channels_raw.items():
                channels[str(name)] = int(ch)
        home_raw = raw.get("home_q", cls.home_q)
        home_q = tuple(float(v) for v in home_raw)
        cal = cls(
            joints=joints,
            gripper=gripper,
            channels=channels,
            home_q=home_q,  # type: ignore[arg-type]
            control_rate_hz=float(raw.get("control_rate_hz", cls.control_rate_hz)),
            max_joint_velocity=float(raw.get("max_joint_velocity", cls.max_joint_velocity)),
            reattach_s=float(raw.get("reattach_s", cls.reattach_s)),
            home_move_s=float(raw.get("home_move_s", cls.home_move_s)),
            host_timeout_s=float(raw.get("host_timeout_s", cls.host_timeout_s)),
            camera=camera,
            measured=_as_bool(raw.get("measured", False), "measured"),
        )
        cal.validate(geometry)
        return cal

    def to_dict(self) -> dict[str, Any]:
        """Plain-YAML-serialisable structure; inverse of :meth:`from_dict`."""
        return {
            "joints": {name: dataclasses.asdict(j) for name, j in self.joints.items()},
            "gripper": dataclasses.asdict(self.gripper),
            "channels": dict(self.channels),
            "home_q": [float(v) for v in self.home_q],
            "control_rate_hz": float(self.control_rate_hz),
            "max_joint_velocity": float(self.max_joint_velocity),
            "reattach_s": float(self.reattach_s),
            "home_move_s": float(self.home_move_s),
            "host_timeout_s": float(self.host_timeout_s),
            "camera": dataclasses.asdict(self.camera),
            "measured": bool(self.measured),
        }

    @classmethod
    def load(cls, path: str | Path, geometry: "ArmGeometry | None" = None) -> "ServoCalibration":
        """Read ``robot_config.yaml``; raises :class:`CalibrationError` on any
        problem, including (with ``geometry``) a home pose below the table."""
        import yaml

        p = Path(path)
        if not p.exists():
            raise CalibrationError(f"calibration file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw, geometry)

    def save(self, path: str | Path) -> None:
        """Write atomically (temp file + replace) so a crash mid-write cannot
        leave an empty calibration behind."""
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write("# Servo calibration for jetson/robot_server.py. Edited by\n")
            fh.write("# scripts/calibrate_servos.py; hand edits are fine too.\n")
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        tmp.replace(p)

    def validate(self, geometry: "ArmGeometry | None" = None) -> None:
        """Raise :class:`CalibrationError` on anything self-contradictory.

        ``geometry`` (link lengths) enables the home-pose table check; the
        server and the calibration REPL always pass it, the pure conversions
        in tests need not.
        """
        missing = [n for n in JOINT_NAMES if n not in self.joints]
        if missing:
            raise CalibrationError(f"joints missing calibration: {missing}")
        for name, joint in self.joints.items():
            joint.validate(name)
        self.gripper.validate()
        self.camera.validate()
        if len(self.home_q) != _N_ARM:
            raise CalibrationError(f"home_q must have {_N_ARM} entries, got {len(self.home_q)}")
        for name, q in zip(JOINT_NAMES, self.home_q):
            j = self.joints[name]
            if not (j.limit_lower_rad <= q <= j.limit_upper_rad):
                raise CalibrationError(
                    f"home_q[{name}] = {q:.4f} is outside its limits "
                    f"[{j.limit_lower_rad:.4f}, {j.limit_upper_rad:.4f}]"
                )
        if geometry is not None:
            check_home_clearance(self.home_q, geometry)
        if self.control_rate_hz <= 0 or self.control_rate_hz > 200:
            raise CalibrationError("control_rate_hz must be in (0, 200]")
        if self.max_joint_velocity <= 0:
            raise CalibrationError("max_joint_velocity must be > 0")
        if self.reattach_s < 0 or not math.isfinite(self.reattach_s):
            raise CalibrationError("reattach_s must be >= 0")
        if self.home_move_s < 0 or not math.isfinite(self.home_move_s):
            raise CalibrationError("home_move_s must be >= 0")
        if self.host_timeout_s < 0 or not math.isfinite(self.host_timeout_s):
            raise CalibrationError("host_timeout_s must be >= 0 (0 disables the host-liveness bound)")
        for name in WIRE_ORDER:
            if name not in self.channels:
                raise CalibrationError(f"channels missing {name!r}")
        if len(set(self.channels[n] for n in WIRE_ORDER)) != _N_WIRE:
            raise CalibrationError("channels must be distinct")

    # -- conversions --------------------------------------------------------

    def rad_to_us(self, joint: str, rad: float) -> float:
        """Joint radians -> pulse width (us) through that joint's calibration."""
        return self.joints[joint].rad_to_us(rad)

    def us_to_rad(self, joint: str, us: float) -> float:
        """Pulse width -> joint radians."""
        return self.joints[joint].us_to_rad(us)

    def width_to_us(self, width_m: float) -> float:
        """Jaw opening (m) -> gripper pulse width (us)."""
        return self.gripper.width_to_us(width_m)

    def us_to_width(self, us: float) -> float:
        """Gripper pulse width -> jaw opening (m)."""
        return self.gripper.us_to_width(us)

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` joint limits in wire order, radians."""
        lower = np.array([self.joints[n].limit_lower_rad for n in JOINT_NAMES], dtype=np.float64)
        upper = np.array([self.joints[n].limit_upper_rad for n in JOINT_NAMES], dtype=np.float64)
        return lower, upper

    def clamp_q(self, q: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp a joint vector to the limits; ``clamped`` tells whether anything
        moved by more than :data:`CLAMP_TOL_RAD`."""
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != _N_ARM:
            raise ValueError(f"q must have {_N_ARM} entries, got {arr.shape[0]}")
        lower, upper = self.limits()
        out = np.minimum(np.maximum(arr, lower), upper)
        return out, bool(np.any(np.abs(out - arr) > CLAMP_TOL_RAD))

    def pulse_clamped(self, q: np.ndarray) -> bool:
        """True when any joint of ``q`` would be clamped at the *pulse* level.

        The radian limits and the pulse map are two separate clamps; this is
        the second one, the one the sketch applies silently. ``validate``
        keeps the limits inside the map, so on a valid calibration this only
        fires for a pose past the limits -- but it is what the follower
        reports, so a stale or hand-edited map cannot hide a clamp.
        """
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        return not all(self.joints[name].fits_pulse_map(float(v)) for name, v in zip(JOINT_NAMES, arr))

    def q_to_pulses(self, q: np.ndarray) -> list[float]:
        return [self.rad_to_us(name, float(v)) for name, v in zip(JOINT_NAMES, q)]

    def pulses_to_q(self, pulses: Sequence[float]) -> np.ndarray:
        return np.array(
            [self.us_to_rad(name, float(us)) for name, us in zip(JOINT_NAMES, pulses)],
            dtype=np.float64,
        )


def _as_bool(value: Any, path: str) -> bool:
    """A YAML boolean, strictly: ``measured: "no"`` must not read as true."""
    if isinstance(value, bool):
        return value
    raise CalibrationError(f"{path} must be true or false, got {value!r}")


def placeholder_safe_calibration(
    calibration: ServoCalibration,
    band_us: tuple[int, int] = PLACEHOLDER_PULSE_BAND_US,
    geometry: "ArmGeometry | None" = None,
) -> ServoCalibration:
    """The calibration a real driver runs under ``--allow-placeholder-calibration``.

    Nothing in an unmeasured file says where the servos' end-stops are, so
    for the first bench bring-up every joint's radian limits are tightened to
    the angles that map into ``band_us`` (the follower then clamps there and
    reports ``clamped``), the gripper's two points are pulled into the band
    along its own linear map, ``max_joint_velocity`` is capped at
    :data:`PLACEHOLDER_MAX_JOINT_VELOCITY` and the re-attach / home moves are
    slowed. ``home_q`` must already lie inside the band (it is the first frame
    after every start); a calibration that cannot fit is refused.
    """
    lo_us, hi_us = (int(v) for v in band_us)
    if lo_us >= hi_us:
        raise CalibrationError(f"placeholder pulse band {lo_us}..{hi_us} us is empty")
    if lo_us < SKETCH_ARM_PULSE_US[0] or hi_us > SKETCH_ARM_PULSE_US[1]:
        raise CalibrationError(
            f"placeholder pulse band {lo_us}..{hi_us} us reaches past the Uno's clamp "
            f"{SKETCH_ARM_PULSE_US[0]}..{SKETCH_ARM_PULSE_US[1]} us"
        )
    joints: dict[str, JointCalibration] = {}
    for name in JOINT_NAMES:
        joint = calibration.joints[name]
        a = joint.us_to_rad(max(lo_us, joint.pulse_min_us))
        b = joint.us_to_rad(min(hi_us, joint.pulse_max_us))
        band_lo, band_hi = min(a, b), max(a, b)
        lower = max(joint.limit_lower_rad, band_lo)
        upper = min(joint.limit_upper_rad, band_hi)
        if lower >= upper:
            raise CalibrationError(
                f"joints.{name}: its limits do not overlap the placeholder pulse band {lo_us}..{hi_us} us"
            )
        joints[name] = dataclasses.replace(joint, limit_lower_rad=float(lower), limit_upper_rad=float(upper))
    for name, q in zip(JOINT_NAMES, calibration.home_q):
        j = joints[name]
        if not (j.limit_lower_rad - 1e-9 <= q <= j.limit_upper_rad + 1e-9):
            raise CalibrationError(
                f"home_q[{name}] = {q:.4f} rad ({calibration.rad_to_us(name, q):.0f} us) is outside the "
                f"placeholder pulse band {lo_us}..{hi_us} us; home is the first frame after every start, so "
                "move home_q inside the band (or widen it with --placeholder-band-us)"
            )
    g = calibration.gripper
    open_us, closed_us = float(g.pulse_open_us), float(g.pulse_closed_us)
    new_open = min(max(open_us, lo_us), hi_us)
    new_closed = min(max(closed_us, lo_us), hi_us)
    if new_open == new_closed:
        raise CalibrationError(f"gripper: both pulse points collapse onto the placeholder band edge {new_open:.0f} us")
    gripper = GripperCalibration(
        pulse_open_us=int(round(new_open)),
        pulse_closed_us=int(round(new_closed)),
        width_open_m=float(g.us_to_width(new_open)),
        width_closed_m=float(g.us_to_width(new_closed)),
    )
    safe = dataclasses.replace(
        calibration,
        joints=joints,
        gripper=gripper,
        max_joint_velocity=min(float(calibration.max_joint_velocity), PLACEHOLDER_MAX_JOINT_VELOCITY),
        reattach_s=max(float(calibration.reattach_s), PLACEHOLDER_MIN_REATTACH_S),
        home_move_s=max(float(calibration.home_move_s), PLACEHOLDER_MIN_HOME_MOVE_S),
    )
    safe.validate(geometry)
    return safe


def _build(cls: type, raw: Any, path: str) -> Any:
    """Construct a frozen dataclass from a YAML mapping, rejecting unknown keys."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise CalibrationError(f"{path}: expected a mapping, got {type(raw).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise CalibrationError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        target = fields[name].type
        try:
            if target in ("int", int):
                kwargs[name] = int(value)
            elif target in ("float", float):
                kwargs[name] = float(value)
            elif target in ("str", str):
                kwargs[name] = str(value)
            else:
                kwargs[name] = value
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"{path}.{name}: {exc}") from exc
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class ServoDriver:
    """Common surface of the three servo back-ends.

    A driver stores the **last commanded pulse per wire channel** (float us,
    unrounded) and derives radians / metres through ``self.calibration`` on
    demand. Nothing here reads the servos: PWM servos cannot be read.

    Attach state. :attr:`attached` is the driver's belief that the servos are
    being pulsed. It goes False on :meth:`detach` (estop, torque off, host
    timeout, server stop), on :meth:`mark_detached` (the bridge reported that
    its watchdog fired) and, on the Uno, after repeated serial failures. Every
    one of those also raises a *hold*: :meth:`write` / :meth:`write_gripper`
    refuse to run until :meth:`reattach` -- which the motion endpoints call
    with a duration -- lifts it. The hold is what stops an in-flight follower,
    a keepalive or any forgotten path from re-attaching five servos with a
    full-speed snap the moment after an estop. Each attached -> detached
    transition is counted (:attr:`detach_count`, :attr:`last_detach_reason`)
    so the laptop can learn about a detach that the next motion repaired.

    Known position. :attr:`position_known` is False until the first frame
    since the bridge reset (construction: opening the Uno's port resets it)
    and again after the bridge reports a reset. While it is False the bridge's
    first frame attaches AT its target at full speed -- there is nothing to
    slew from -- so :meth:`reattach` does exactly that, in one frame, and says
    so; the server only allows ``home`` in that state.

    Keepalive. :meth:`start_keepalive` runs a daemon thread that calls
    :meth:`_send_keepalive` whenever the wire has been idle for
    ``keepalive_s`` while attached. The Uno detaches after 500 ms of silence
    and every pause in a pick is longer than that, so without this the arm
    went limp between commands. 150 ms leaves room for a GC pause; the thread
    polls at half the period so the worst-case gap is 1.5 periods.
    """

    name = "base"
    has_watchdog = True
    """Whether the bridge relaxes the servos on its own when the host goes
    quiet (the Uno's 500 ms watchdog). The PCA9685 has none."""

    def __init__(self, calibration: ServoCalibration, keepalive_s: float | None = None) -> None:
        self.calibration = calibration
        self._lock = threading.RLock()
        self._pulses: list[float] = calibration.q_to_pulses(np.asarray(calibration.home_q))
        self._pulses.append(calibration.width_to_us(calibration.gripper.width_open_m))
        self._attached = False
        self._hold_detached = False
        self._position_known = False
        self._last_frame_t = -math.inf
        self.detach_count = 0
        """Attached -> detached transitions since construction (any cause)."""
        self.last_detach_reason: str | None = None
        self.sleep: Callable[[float], None] = time.sleep
        """Pacing of the software re-attach ramp; the server injects a no-op
        in tests exactly as it does for the follower."""
        self.keepalive_s = keepalive_s
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_failures = 0

    # -- to override --------------------------------------------------------

    def _send_pulses(self, pulses: Sequence[float]) -> None:
        raise NotImplementedError

    def _send_detach(self) -> None:
        raise NotImplementedError

    def _send_keepalive(self) -> None:
        """Feed the bridge's watchdog without changing any target.

        Default: re-send the stored pulses (a ``P`` frame with the current
        targets moves nothing). Back-ends with a dedicated frame override it.
        """
        self._send_pulses(list(self._pulses))

    def _send_reattach(
        self,
        start: Sequence[float],
        target: Sequence[float],
        duration_s: float,
        abort: threading.Event | None,
    ) -> None:
        """Energise at ``start`` and reach ``target`` over ``duration_s``.

        Default: a software ramp of ``P`` frames at the control rate -- the
        first one exactly at ``start`` (re-energise where the servos were
        last pulsed), the last exactly at ``target`` -- each one also feeding
        the watchdog. ``duration_s == 0`` is one frame at ``target`` (the
        unknown-position attach). Returns early (servos left wherever the
        ramp got to, already detached by whoever set it) when ``abort`` fires
        or a detach lands mid-ramp. Overrides may hand the interpolation to
        the bridge (the Uno's ``T`` frame) and must then update ``_pulses``
        themselves.
        """
        if duration_s <= 0.0:
            with self._lock:
                if self._hold_detached or (abort is not None and abort.is_set()):
                    return
                self._pulses = [float(t) for t in target]
                self._emit()
            return
        tick = 1.0 / float(self.calibration.control_rate_hz)
        n = max(1, int(math.ceil(duration_s / tick)))
        for i in range(0, n + 1):
            with self._lock:
                if self._hold_detached or (abort is not None and abort.is_set()):
                    return
                frac = i / n
                self._pulses = [s + (t - s) * frac for s, t in zip(start, target)]
                self._emit()
            if i < n:
                self.sleep(tick)

    def _refresh_attach_state(self) -> None:
        """Hook for back-ends that can notice a detach on their own (the fake
        watchdog); called under the lock before any decision that depends on
        :attr:`attached`."""

    def probe(self) -> tuple[tuple[int, ...], bool] | None:
        """``(pulses being output, attached)`` as the *bridge* reports them,
        or ``None`` when the back-end cannot tell (PCA9685). A back-end that
        can also tell a reset bridge updates :attr:`position_known` here."""
        return None

    def close(self) -> None:
        """Release the port / bus. Idempotent."""
        self.stop_keepalive()

    # -- shared behaviour ---------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        """The driver lock. The follower holds it across its estop check and
        the write so an estop can never slip between the two."""
        return self._lock

    def _note_frame(self) -> None:
        self._last_frame_t = time.monotonic()

    def _emit(self) -> None:
        self._send_pulses(list(self._pulses))
        self._attached = True
        self._position_known = True
        self._note_frame()

    def _count_detach(self, reason: str) -> None:
        self.detach_count += 1
        self.last_detach_reason = reason

    def _require_energised(self) -> None:
        self._refresh_attach_state()
        if self._hold_detached:
            raise DriverError(
                "servos are detached (estop, torque off, watchdog or serial loss); "
                "a motion endpoint must re-attach them slowly before writing"
            )

    def write(self, q_rad: np.ndarray) -> None:
        """Command the four arm joints (radians, wire order). Clamps to limits.

        Refuses while a detach hold is in force (see the class docstring).
        """
        q, _ = self.calibration.clamp_q(q_rad)
        with self._lock:
            self._require_energised()
            self._pulses[:_N_ARM] = self.calibration.q_to_pulses(q)
            self._emit()

    def write_gripper(self, width_m: float) -> None:
        """Command the jaw opening in metres (clamped to the calibrated span).

        Sends all five channels, so it re-energises the whole arm; refused
        while a detach hold is in force for the same reason as :meth:`write`.
        """
        with self._lock:
            self._require_energised()
            self._pulses[_N_ARM] = self.calibration.width_to_us(width_m)
            self._emit()

    def torque(self, enabled: bool, duration_s: float = 1.0, abort: threading.Event | None = None) -> None:
        """``False`` detaches (arm goes limp); ``True`` re-attaches slowly at
        the stored pose over ``duration_s`` (see :meth:`reattach`)."""
        if enabled:
            self.reattach(duration_s=duration_s, abort=abort)
        else:
            self.detach("torque off")

    def detach(self, reason: str = "detach") -> None:
        """Stop pulsing every servo and raise the hold. The commanded pose is
        remembered and is where :meth:`reattach` starts from.

        The hold goes up *before* the wire is touched, so a ``D`` that the
        bridge rejects or never answers still stops the keepalive feeding
        (the Uno's watchdog then relaxes the arm) and still refuses every
        write. ``D`` is tried :data:`DETACH_ATTEMPTS` times; the last error
        is re-raised when none got through.
        """
        with self._lock:
            was_attached = self._attached
            self._attached = False
            self._hold_detached = True
            if was_attached:
                self._count_detach(reason)
            last: DriverError | None = None
            for attempt in range(1, DETACH_ATTEMPTS + 1):
                try:
                    self._send_detach()
                    return
                except DriverError as exc:
                    last = exc
                    _log.error("detach (D) attempt %d/%d failed: %s", attempt, DETACH_ATTEMPTS, exc)
            if not self.has_watchdog:
                raise DriverError(
                    f"the bridge did not confirm the detach after {DETACH_ATTEMPTS} attempts ({last}); "
                    f"the {self.name} has NO watchdog and may still be pulsing every servo -- CUT THE "
                    "+6 V SERVO SUPPLY NOW"
                ) from last
            raise DriverError(
                f"the bridge did not confirm the detach after {DETACH_ATTEMPTS} attempts ({last}); "
                "the hold is raised and the keepalive no longer feeds the watchdog, so the Uno "
                f"relaxes the servos within {SKETCH_WATCHDOG_MS} ms -- keep a hand on the +6 V switch"
            ) from last

    def mark_detached(self, reason: str) -> None:
        """Record that the bridge detached on its own (watchdog, serial loss)
        without sending anything; raises the hold like :meth:`detach`."""
        with self._lock:
            if self._attached:
                _log.warning("servos detached by the bridge (%s); the next motion re-attaches slowly", reason)
                self._count_detach(reason)
            self._attached = False
            self._hold_detached = True

    def mark_position_lost(self, reason: str) -> None:
        """The bridge reset: it pulses nothing and knows no position. The next
        frame will attach AT its target, so only ``home`` is allowed next."""
        with self._lock:
            if self._position_known:
                _log.error(
                    "the servo bridge has no known position any more (%s): the arm is limp and the "
                    "next frame would attach every servo AT its target at full speed. Hand-pose the "
                    "arm at home, then send `home`.",
                    reason,
                )
            self._position_known = False
            self.mark_detached(reason)

    def reattach(
        self,
        pulses: Sequence[float] | None = None,
        duration_s: float = 1.0,
        abort: threading.Event | None = None,
    ) -> None:
        """Re-energise after a detach and move to ``pulses`` over ``duration_s``.

        ``pulses`` defaults to the stored pose (re-energise in place). The
        first pulse goes out at the stored pose, which is the last place the
        servos were known to be; the move to the target then takes
        ``duration_s`` (never less than one control tick). Already attached:
        the target is written at once, as an ordinary command. Blocks until
        the move is over or ``abort`` (the server's estop event) fires.

        With no known position (:attr:`position_known` False) the bridge
        attaches AT the target whatever is asked, so this is one frame at
        ``pulses``, logged as the jump it is; the server allows it for
        ``home`` only.
        """
        with self._lock:
            target = [float(v) for v in (self._pulses if pulses is None else pulses)]
            if len(target) != _N_WIRE:
                raise ValueError(f"reattach needs {_N_WIRE} pulses, got {len(target)}")
            self._refresh_attach_state()
            if self._attached and not self._hold_detached:
                self._pulses = target
                self._emit()
                return
            first_attach = not self._position_known
            start = list(target) if first_attach else list(self._pulses)
            self._hold_detached = False
            self._attached = True
        if first_attach:
            _log.warning(
                "first frame since the bridge reset: every servo attaches AT its target at full "
                "speed (no position is known to slew from); the arm must be hand-posed there"
            )
            self._send_reattach(start, target, 0.0, abort)
            return
        _log.info("re-attaching servos over %.2f s", duration_s)
        self._send_reattach(start, target, max(0.0, float(duration_s)), abort)

    def read_commanded(self) -> tuple[np.ndarray, float]:
        """``(q_rad[4], gripper_width_m)`` derived from the stored pulses."""
        with self._lock:
            q = self.calibration.pulses_to_q(self._pulses[:_N_ARM])
            width = self.calibration.us_to_width(self._pulses[_N_ARM])
        return q, float(width)

    @property
    def attached(self) -> bool:
        """Whether the servos are believed to be pulsed (see the class docstring)."""
        with self._lock:
            self._refresh_attach_state()
            return self._attached

    @property
    def position_known(self) -> bool:
        """Whether the bridge has a position to slew from (see the class docstring)."""
        with self._lock:
            return self._position_known

    @property
    def hold_detached(self) -> bool:
        """Whether a detach hold is in force (``write`` refuses until ``reattach``)."""
        with self._lock:
            self._refresh_attach_state()
            return self._hold_detached

    @property
    def needs_reattach(self) -> bool:
        """True when the next motion must go through :meth:`reattach`: at boot
        (nothing pulsed yet) and after any detach."""
        with self._lock:
            self._refresh_attach_state()
            return self._hold_detached or not self._attached

    def pulses(self) -> tuple[int, ...]:
        """Last commanded pulses, rounded to the integer us that go on the wire."""
        with self._lock:
            return tuple(int(round(p)) for p in self._pulses)

    # -- keepalive ----------------------------------------------------------

    def start_keepalive(self, period_s: float | None = None) -> None:
        """Start the watchdog-feeding thread (no-op without a period)."""
        if period_s is not None:
            self.keepalive_s = float(period_s)
        if self.keepalive_s is None or self.keepalive_s <= 0:
            return
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name=f"{self.name}-keepalive", daemon=True
        )
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        """Stop the thread; idempotent."""
        self._keepalive_stop.set()
        thread = self._keepalive_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._keepalive_thread = None

    @property
    def keepalive_running(self) -> bool:
        return self._keepalive_thread is not None and self._keepalive_thread.is_alive()

    def keepalive_tick(self) -> bool:
        """One keepalive decision, as the thread makes it; True when a frame went out.

        Feeds only while attached with no hold, and only when the wire has
        been idle for a period (the follower or a re-attach feeds it
        otherwise). Public so a test can drive it without the thread.
        """
        period = float(self.keepalive_s or KEEPALIVE_PERIOD_S)
        with self._lock:
            self._refresh_attach_state()
            if not self._attached or self._hold_detached:
                return False
            if time.monotonic() - self._last_frame_t < period:
                return False  # the follower (or a reattach ramp) is feeding it
            try:
                self._send_keepalive()
                self._note_frame()
                self._keepalive_failures = 0
                return True
            except DriverError as exc:
                self._keepalive_failures += 1
                if self._keepalive_failures == 1:
                    _log.warning("keepalive failed: %s", exc)
                if self._keepalive_failures >= SERIAL_MAX_TIMEOUTS:
                    # The bridge's own watchdog has fired by now; stop
                    # pretending the arm is held and force a slow re-attach.
                    self.mark_detached(
                        f"keepalive failed {self._keepalive_failures} times in a row: {exc}"
                    )
                    self._keepalive_failures = 0
                return False

    def _keepalive_loop(self) -> None:
        period = float(self.keepalive_s or 0.0)
        while not self._keepalive_stop.wait(period / 2.0):
            self.keepalive_tick()


class FakeDriver(ServoDriver):
    """Records every command; moves instantly. The test double and ``--driver fake``.

    ``on_command(q_rad, gripper_width_m)`` is invoked on every pulse write so a
    test (or a fake world) can watch the arm move.

    ``watchdog_ms`` makes the fake behave like the Uno's watchdog: after that
    many milliseconds without a ``P``/``K`` frame it detaches itself, records
    ``("W", ())`` and counts a trip. With it set the keepalive thread starts
    by default (150 ms, as on the Uno), so ``--fake-watchdog-ms 500`` lets
    the fake lane prove that idle gaps no longer drop the arm -- and
    ``keepalive_s=0`` proves the test can fail.

    Like the Uno after its port opens, a new fake knows no position: the
    first attach is one frame AT its target (the fake lane used to glide home
    over ``home_move_s`` at boot, a ramp the Uno never performs), and the
    server allows only ``home`` until then. :meth:`simulate_reset` models a
    mid-session USB reset or brownout.
    """

    name = "fake"

    def __init__(
        self,
        calibration: ServoCalibration | None = None,
        on_command: Callable[[np.ndarray, float], None] | None = None,
        watchdog_ms: float | None = None,
        keepalive_s: float | None = None,
    ) -> None:
        if keepalive_s is None and watchdog_ms is not None:
            keepalive_s = KEEPALIVE_PERIOD_S
        super().__init__(calibration or ServoCalibration.default(), keepalive_s=keepalive_s)
        self.on_command = on_command
        self.commands: list[tuple[str, tuple[int, ...]]] = []
        """``("P", pulses)`` per write, ``("D", ())`` per detach, ``("K", ())``
        per keepalive, ``("W", ())`` per watchdog trip, ``("R", ())`` per
        simulated bridge reset."""
        self.watchdog_ms = None if watchdog_ms is None else float(watchdog_ms)
        self.watchdog_trips = 0
        self.fail_detach: int = 0
        """Test hook: the next this-many ``D`` frames are rejected like a
        corrupted frame the Uno answers with ``ERR``."""
        if keepalive_s:
            self.start_keepalive()

    def _refresh_attach_state(self) -> None:
        if self.watchdog_ms is None or not self._attached:
            return
        if (time.monotonic() - self._last_frame_t) * 1000.0 > self.watchdog_ms:
            self.commands.append(("W", ()))
            self.watchdog_trips += 1
            self.mark_detached(f"fake watchdog: no frame for {self.watchdog_ms:.0f} ms")

    def _send_pulses(self, pulses: Sequence[float]) -> None:
        rounded = tuple(int(round(p)) for p in pulses)
        self.commands.append(("P", rounded))
        if self.on_command is not None:
            q = self.calibration.pulses_to_q(pulses[:_N_ARM])
            self.on_command(q, self.calibration.us_to_width(pulses[_N_ARM]))

    def _send_detach(self) -> None:
        if self.fail_detach > 0:
            self.fail_detach -= 1
            self.commands.append(("E", ()))
            raise DriverError("Uno rejected the frame: ERR unknown command (fake)")
        self.commands.append(("D", ()))

    def _send_keepalive(self) -> None:
        self.commands.append(("K", ()))

    def probe(self) -> tuple[tuple[int, ...], bool]:
        with self._lock:
            self._refresh_attach_state()
            return self.pulses(), self._attached

    def simulate_reset(self) -> None:
        """The bridge reset under us (USB re-enumeration, brownout)."""
        with self._lock:
            self.commands.append(("R", ()))
            self.mark_position_lost("fake bridge reset")

    @property
    def write_count(self) -> int:
        return sum(1 for kind, _ in self.commands if kind == "P")

    @property
    def keepalive_count(self) -> int:
        return sum(1 for kind, _ in self.commands if kind == "K")


class UnoSerialDriver(ServoDriver):
    """Arduino Uno running ``servo_bridge.ino`` over USB-CDC.

    Opens the port once. Opening toggles DTR, which resets the Uno; anything
    sent in the next ~2 s is lost, hence the wait -- and the bridge knows no
    position afterwards (:attr:`position_known` starts False). Motion is
    streamed as ``P`` frames: the follower already resamples to the control
    rate, so the Uno's own slew limit (20 us per tick) is the last line of
    defence, not the interpolator. The one ``T`` frame is the re-attach
    after a detach, where the Uno's interpolator is exactly what is wanted.

    Watchdog. The sketch detaches after :data:`SKETCH_WATCHDOG_MS` of silence
    and every pause in a pick is longer, so the keepalive thread (started
    here, period ``keepalive_s``) sends a ``K`` frame -- refreshes the watchdog,
    moves nothing -- whenever the wire has been idle. :meth:`probe` asks the
    Uno with ``?`` whether it agrees the servos are attached (and whether it
    has a position at all); the server believes the Uno over the driver's
    own flags.

    Resynchronisation. ``readline`` returning nothing means the reply is late,
    not lost; see :meth:`_resync` for how the stream is put back in step.
    """

    name = "uno_serial"

    def __init__(
        self,
        calibration: ServoCalibration,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        reset_wait_s: float = 2.0,
        reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S,
        keepalive_s: float | None = KEEPALIVE_PERIOD_S,
    ) -> None:
        super().__init__(calibration, keepalive_s=keepalive_s)
        try:
            import serial  # noqa: F401
        except ImportError as exc:
            raise DriverError(
                "pyserial is not installed; on the Jetson run: pip3 install pyserial"
            ) from exc
        import serial as _serial

        self.port = port
        self.reply_timeout_s = float(reply_timeout_s)
        self.consecutive_timeouts = 0
        """Failed exchanges since the last good one (diagnostics)."""
        self.resyncs = 0
        """Times the stream was re-synchronised (drain + ``S`` echo) and a frame resent."""
        self.stale_replies_skipped = 0
        """Late replies read and discarded while waiting for a sync echo."""
        self._sync_nonce = 0
        self._sync_unsupported_logged = False
        self._out_of_step = False
        """A failed exchange may still be answered late: sync before the next frame."""
        try:
            self._ser = _serial.Serial(port, baudrate=baud, timeout=reply_timeout_s)
        except Exception as exc:
            raise DriverError(
                f"could not open {port}: {exc}. Is the Uno plugged in, is the user in "
                "the dialout group, and is it a genuine Uno (CH340 clones do not "
                "enumerate on JetPack 6)?"
            ) from exc
        _log.info(
            "Opened %s at %d baud; waiting %.1fs for the Uno reset. The bridge now knows no "
            "position: the first frame attaches AT home, so the arm must be hand-posed there.",
            port, baud, reset_wait_s,
        )
        time.sleep(reset_wait_s)
        self._ser.reset_input_buffer()
        self.start_keepalive()

    # -- wire ---------------------------------------------------------------

    def _exchange(self, frame: bytes, expect: str = "OK") -> bytes:
        """Write ``frame``, return its reply line, repairing a stream that
        fell out of step.

        ``expect`` is ``"OK"`` for P/T/D/K frames and ``"Q"`` for ``?``. A
        timeout, or a reply that cannot answer this frame (a ``Q`` or ``S``
        line, or noise, when ``OK``/``ERR`` is due), triggers :meth:`_resync`
        and one resend. When the sync itself fails, or the resend times out
        or is answered wrongly too, the exchange counts as failed and ``b""``
        is returned (callers raise "no reply"); after
        :data:`SERIAL_MAX_TIMEOUTS` failed exchanges in a row the servos are
        marked detached.

        A failed exchange leaves a reply *owed*: the resent frame (or the sync)
        may still be answered late, and that late ``OK`` is indistinguishable
        from the next frame's ``OK`` (audit Missed-7: two late replies in a
        row left every later read one frame behind for good). The stream is
        therefore marked out of step, and the next exchange re-synchronises
        *before* writing anything; if that sync fails too, the frame is not
        sent at all -- nothing goes out on a stream that is not provably in
        step.
        """
        with self._lock:
            if self._out_of_step:
                self.resyncs += 1
                if not self._resync():
                    if frame.startswith(b"D"):
                        # A detach is idempotent and makes the arm safe: send
                        # it even unconfirmed (its reply, if any, is skipped
                        # by the next sync). detach() retries it anyway.
                        self._ser.write(frame)
                        self._ser.flush()
                    return self._fail_exchange(b"")
                self._out_of_step = False
            reply = self._write_read(frame)
            if reply.strip() and self._kind_matches(reply, expect):
                self.consecutive_timeouts = 0
                return reply
            if reply.strip():
                _log.warning(
                    "serial stream out of step (sent %r, got %r); re-synchronising",
                    frame, reply,
                )
            self.resyncs += 1
            if not self._resync():
                return self._fail_exchange(b"")
            reply = self._write_read(frame)
            if reply.strip() and self._kind_matches(reply, expect):
                self.consecutive_timeouts = 0
                return reply
            if reply.strip():
                _log.warning(
                    "serial stream still out of step after a sync (sent %r, got %r)", frame, reply,
                )
            return self._fail_exchange(reply)

    def _fail_exchange(self, reply: bytes) -> bytes:
        """Book-keeping for a failed exchange: a reply may still be owed, so
        the next exchange syncs first; enough failures in a row mean the Uno
        is gone and its watchdog has dropped the arm."""
        self._out_of_step = True
        self.consecutive_timeouts += 1
        if self.consecutive_timeouts >= SERIAL_MAX_TIMEOUTS:
            self.mark_detached(
                f"{self.consecutive_timeouts} consecutive failed serial exchanges on {self.port}"
            )
            self.consecutive_timeouts = 0
        return reply

    def _resync(self) -> bool:
        """Put the reply stream back in step; True when it provably is.

        Drain what has arrived, send ``S<nonce>``, then read until the sketch
        echoes that exact nonce. The Uno answers strictly in order, so every
        line before the echo -- a late reply still in flight when the buffer
        was drained, which draining alone never catches -- is stale and
        skipped. False when the echo does not come (Uno gone, or a sketch
        too old to know ``S``: then flash the current one).
        """
        self._ser.reset_input_buffer()
        self._sync_nonce = (self._sync_nonce + 1) % 1_000_000
        token = encode_s_frame(self._sync_nonce)
        self._ser.write(token)
        self._ser.flush()
        expected = token.strip()
        for _ in range(SYNC_MAX_LINES):
            line = self._ser.readline()
            text = line.strip()
            if not text:
                return False
            if text == expected:
                return True
            if text.startswith(b"ERR unknown command S") and not self._sync_unsupported_logged:
                _log.error(
                    "the Uno does not understand the sync frame: it runs an old servo_bridge.ino. "
                    "Flash jetson/arduino/servo_bridge/servo_bridge.ino; until then a late reply "
                    "cannot be recovered from"
                )
                self._sync_unsupported_logged = True
            self.stale_replies_skipped += 1
        return False

    def _write_read(self, frame: bytes) -> bytes:
        self._ser.write(frame)
        self._ser.flush()
        return self._ser.readline()

    @staticmethod
    def _kind_matches(reply: bytes, expect: str) -> bool:
        text = reply.strip()
        if expect == "Q":
            return text.startswith(b"Q")
        # OK or ERR are the only in-step answers to a command frame; Q and S
        # echoes are replies to something else, and anything else is noise
        # (a line that is not a reply cannot prove the stream is in step).
        return text == b"OK" or text.startswith(b"ERR")

    def _send_pulses(self, pulses: Sequence[float]) -> None:
        ordered = self._reorder(pulses)
        parse_reply(self._exchange(encode_p_frame(ordered)))

    def _send_detach(self) -> None:
        parse_reply(self._exchange(b"D\n"))

    def _send_keepalive(self) -> None:
        """``K``: refreshes the Uno's watchdog and nothing else."""
        parse_reply(self._exchange(b"K\n"))

    def _send_reattach(
        self,
        start: Sequence[float],
        target: Sequence[float],
        duration_s: float,
        abort: threading.Event | None,
    ) -> None:
        """One ``T`` frame from the Uno's own last position to ``target`` over
        ``duration_s``, fed with ``K`` frames while it interpolates.

        ``start`` is not sent: the sketch keeps its last pulsed position
        across a detach and interpolates from *that*. A freshly reset Uno has
        none and attaches AT ``target`` (the caller passes 0 s then). The
        wait is paced by ``self.sleep`` in steps of the keepalive period so
        the watchdog never fires mid-move and an estop can take the lock
        between steps.
        """
        with self._lock:
            if self._hold_detached or (abort is not None and abort.is_set()):
                return
            ms = int(round(max(0.0, duration_s) * 1000.0))
            parse_reply(self._exchange(encode_t_frame(self._reorder(target), ms), expect="OK"))
            self._pulses = [float(v) for v in target]
            self._attached = True
            self._position_known = True
            self._note_frame()
        step = float(self.keepalive_s) if self.keepalive_s else KEEPALIVE_PERIOD_S
        steps = int(math.ceil(max(0.0, duration_s) / step))
        for _ in range(steps):
            self.sleep(step)
            with self._lock:
                if self._hold_detached or (abort is not None and abort.is_set()):
                    return
                self._send_keepalive()
                self._note_frame()

    def _reorder(self, pulses: Sequence[float]) -> list[int]:
        """Map wire-order pulses onto the channel map (identity for the stock pin table)."""
        out = [1500] * _N_WIRE
        for name, us in zip(WIRE_ORDER, pulses):
            out[self.calibration.channels[name]] = int(round(us))
        return out

    def _unorder(self, pulses: Sequence[int]) -> tuple[int, ...]:
        """Inverse of :meth:`_reorder`: channel-ordered pulses back to wire order."""
        return tuple(int(pulses[self.calibration.channels[name]]) for name in WIRE_ORDER)

    def query_status(self) -> BridgeStatus:
        """``?`` -> the full :class:`BridgeStatus` (pulses in channel order)."""
        return decode_q_status(self._exchange(b"?\n", expect="Q"))

    def query(self) -> tuple[tuple[int, ...], bool]:
        """``?`` -> ``(current pulses, attached)`` as the Uno sees them."""
        status = self.query_status()
        return status.pulses, status.attached

    def probe(self) -> tuple[tuple[int, ...], bool]:
        """The Uno's answer to ``?``; a reset Uno (``have_position`` 0) marks
        the position lost, whatever this driver believed."""
        with self._lock:
            status = self.query_status()
            if status.position_known is False and (self._position_known or self._attached):
                self.mark_position_lost("the Uno reports have_position=0: it reset (USB, brownout, port reopened)")
            return self._unorder(status.pulses), status.attached

    def close(self) -> None:
        self.stop_keepalive()
        with self._lock:
            try:
                self._ser.close()
            except Exception as exc:  # pragma: no cover - hardware
                _log.warning("closing %s failed: %s", self.port, exc)


class Pca9685Driver(ServoDriver):
    """PCA9685 on I2C, registers written directly with ``smbus2``.

    Direct register access sidesteps Jetson.GPIO 2.1.7 / PlatformDetect
    breakage on JetPack 6. Sequence: sleep, PRESCALE 121 (25 MHz / 4096 / 50 Hz
    - 1 = 121), wake, restart with auto-increment. Channel n lives at
    ``0x06 + 4n`` (ON_L, ON_H, OFF_L, OFF_H); a pulse of ``us`` is
    ``OFF = us * 4096 / 20000`` counts with ON = 0. Setting bit 4 of OFF_H
    (0x10) forces the channel fully off, which is how a channel is detached.

    NO HARDWARE WATCHDOG. The PCA9685 keeps generating the last pulse on
    every channel for as long as it has power: a stalled or crashed Jetson
    process leaves the arm energised and the jaw squeezing. There is no
    keepalive to stop feeding, so the only software bound is the server's
    host timeout (``host_timeout_s``): after that much client silence the
    server calls :meth:`detach`, which writes FULL-OFF (bit 4 of OFF_H) to
    every servo channel. Never run this driver with ``--host-timeout-s 0``,
    and keep a hand on the +6 V switch -- a dead ``robot_server.py`` cannot
    detach anything. :meth:`close` writes full-off too, best effort. Prefer
    the Uno (its sketch relaxes the arm 500 ms after the host goes quiet)
    unless it is a clone.
    """

    name = "pca9685"
    has_watchdog = False

    _MODE1 = 0x00
    _PRESCALE = 0xFE
    _LED0_ON_L = 0x06
    _MODE1_SLEEP = 0x10
    _MODE1_AI = 0x20
    _MODE1_RESTART = 0x80
    _PRESCALE_50HZ = 121

    def __init__(self, calibration: ServoCalibration, bus: int = 7, address: int = 0x40) -> None:
        super().__init__(calibration)
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise DriverError(
                "smbus2 is not installed; on the Jetson run: pip3 install smbus2"
            ) from exc
        self.address = address
        self._bus: Any = None
        try:
            self._bus = SMBus(bus)
        except Exception as exc:
            raise DriverError(f"could not open /dev/i2c-{bus}: {exc}") from exc
        self._write8(self._MODE1, self._MODE1_SLEEP)
        self._write8(self._PRESCALE, self._PRESCALE_50HZ)
        self._write8(self._MODE1, 0x00)
        time.sleep(0.005)
        self._write8(self._MODE1, self._MODE1_RESTART | self._MODE1_AI)
        _log.info("PCA9685 at 0x%02x on bus %d set to 50 Hz", address, bus)

    def _write8(self, reg: int, value: int) -> None:
        try:
            self._bus.write_byte_data(self.address, reg, value & 0xFF)
        except OSError as exc:
            # A NACK / bus glitch surfaces as OSError (errno 121 on Linux).
            # As a DriverError it reaches the retry in detach() and the
            # follower's stop-on-failure path instead of escaping them.
            raise DriverError(f"I2C write to 0x{self.address:02x} register 0x{reg:02x} failed: {exc}") from exc

    def _set_channel(self, channel: int, us: float | None) -> None:
        """``us=None``: FULL-OFF (bit 4 of OFF_H) -- the channel stops pulsing."""
        base = self._LED0_ON_L + 4 * channel
        if us is None:
            data = [0, 0, 0, 0x10]
        else:
            off = int(round(float(us) * 4096.0 / 20000.0))
            off = min(max(off, 0), 4095)
            data = [0, 0, off & 0xFF, (off >> 8) & 0x0F]
        try:
            self._bus.write_i2c_block_data(self.address, base, data)
        except OSError as exc:
            raise DriverError(f"I2C write to PCA9685 channel {channel} failed: {exc}") from exc

    def _send_pulses(self, pulses: Sequence[float]) -> None:
        with self._lock:
            for name, us in zip(WIRE_ORDER, pulses):
                self._set_channel(self.calibration.channels[name], us)

    def _send_detach(self) -> None:
        with self._lock:
            for name in WIRE_ORDER:
                self._set_channel(self.calibration.channels[name], None)

    def close(self) -> None:
        """Full-off on every servo channel (no watchdog will do it), then
        release the bus. Idempotent."""
        self.stop_keepalive()
        with self._lock:
            if self._bus is None:
                return
            try:
                self._send_detach()
            except Exception as exc:  # pragma: no cover - hardware
                _log.warning("full-off on close failed: %s", exc)
            self._attached = False
            self._hold_detached = True
            try:
                self._bus.close()
            except Exception as exc:  # pragma: no cover - hardware
                _log.warning("closing the I2C bus failed: %s", exc)
            self._bus = None


# ---------------------------------------------------------------------------
# Fake world (fake driver only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmGeometry:
    """Link lengths of the planar arm, metres; defaults match ``hardware.arm``.

    Kept as a five-number CLI argument rather than read from the laptop's
    config because this file must not import ``mfw``. ``tests/test_hardware_e2e.py``
    pins :func:`planar_tcp` against ``PlanarKinematics.fk`` so the two cannot
    drift silently.
    """

    base_height: float = 0.07
    shoulder_offset: float = 0.0
    upper_arm: float = 0.105
    forearm: float = 0.10
    tool: float = 0.09

    @classmethod
    def parse(cls, text: str) -> "ArmGeometry":
        parts = [float(v) for v in text.split(",")]
        if len(parts) != 5:
            raise ValueError("arm geometry needs 5 numbers: base_height,shoulder_offset,upper_arm,forearm,tool")
        return cls(*parts)


def planar_link_points(q: Sequence[float], geometry: ArmGeometry) -> np.ndarray:
    """World positions of ``[base, shoulder, elbow, wrist, tcp]``, shape (5, 3).

    Same convention as :func:`planar_tcp` and ``PlanarKinematics.link_points``
    on the laptop: yaw about +Z, zero along +X; every pitch measured from
    vertical toward the radial direction, elbow and wrist relative to the
    previous link. The base point is the yaw axis at table level.
    """
    yaw, shoulder, elbow, wrist = (float(v) for v in q)
    g = geometry
    phi1 = shoulder
    phi2 = phi1 + elbow
    phi3 = phi2 + wrist
    r_s, z_s = g.shoulder_offset, g.base_height
    r_e, z_e = r_s + g.upper_arm * math.sin(phi1), z_s + g.upper_arm * math.cos(phi1)
    r_w, z_w = r_e + g.forearm * math.sin(phi2), z_e + g.forearm * math.cos(phi2)
    r_t, z_t = r_w + g.tool * math.sin(phi3), z_w + g.tool * math.cos(phi3)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [r_s * cy, r_s * sy, z_s],
            [r_e * cy, r_e * sy, z_e],
            [r_w * cy, r_w * sy, z_w],
            [r_t * cy, r_t * sy, z_t],
        ],
        dtype=np.float64,
    )


def planar_tcp(q: Sequence[float], geometry: ArmGeometry) -> np.ndarray:
    """Jaw-midpoint position for ``q = [yaw, shoulder, elbow, wrist]`` (radians).

    Convention (identical to ``mfw/hardware/kinematics.py``): yaw about +Z,
    zero along +X; every pitch measured from vertical toward the radial
    direction, elbow and wrist relative to the previous link.
    """
    return planar_link_points(q, geometry)[4]


def check_home_clearance(
    home_q: Sequence[float],
    geometry: ArmGeometry,
    clearance_m: float = TABLE_CLEARANCE_M,
) -> None:
    """Raise :class:`CalibrationError` when any link of ``home_q`` sits below
    ``clearance_m`` above the table.

    Measured trap: ``home_q = [0, 1.5708, 1.5708, 0]`` passes every joint
    limit and puts the wrist 3 cm and the jaw 12 cm *below* the table; the
    runtime drives to home at every boot before any planner exists, so a
    home like that pushes the forearm through the table on every start.
    The base and shoulder points are fixed by the mounting and skipped.
    """
    if len(home_q) != _N_ARM:
        raise CalibrationError(f"home_q must have {_N_ARM} entries, got {len(home_q)}")
    points = planar_link_points(home_q, geometry)
    names = ("elbow", "wrist", "jaw")
    low = [(name, float(z)) for name, z in zip(names, points[2:, 2]) if z < clearance_m]
    if low:
        detail = ", ".join(f"{name} at z = {z * 1000.0:+.0f} mm" for name, z in low)
        raise CalibrationError(
            f"home_q {[round(float(v), 4) for v in home_q]} puts the arm below the table "
            f"clearance ({clearance_m * 1000.0:.0f} mm): {detail}. Home is driven to at every "
            "boot with no planner in the loop; record a tucked pose with every link above the table."
        )


class FakeWorld:
    """Objects on the table that the fake jaw can pick up and carry.

    Rules (metres): attach when the commanded width drops below the object's
    narrow side + 5 mm while the TCP is within 2 cm (XY) and below its top +
    1 cm; an attached object's base follows the TCP; release when the width
    opens beyond the narrow side + 5 mm, dropping the object under the TCP.
    Thread-safe: the follower writes from the motion thread, ``snapshot`` is
    read from the request loop.
    """

    ATTACH_XY_M = 0.02
    ATTACH_Z_ABOVE_TOP_M = 0.01
    WIDTH_SLACK_M = 0.005

    def __init__(
        self,
        scene: Mapping[str, Sequence[float]],
        footprints: Mapping[str, Sequence[float]],
        geometry: ArmGeometry | None = None,
    ) -> None:
        self.geometry = geometry or ArmGeometry()
        self.footprints = {str(k): tuple(float(v) for v in size) for k, size in footprints.items()}
        self.objects: dict[str, np.ndarray] = {}
        for label, xyz in scene.items():
            vals = [float(v) for v in xyz] + [0.0]
            self.objects[str(label)] = np.array(vals[:3], dtype=np.float64)
        self.attached: str | None = None
        self.tcp: np.ndarray | None = None
        self._lock = threading.Lock()

    def _width_of(self, label: str) -> float:
        sx, sy, _sz = self.footprints.get(label, (0.04, 0.04, 0.04))
        return float(min(sx, sy))

    def _height_of(self, label: str) -> float:
        return float(self.footprints.get(label, (0.04, 0.04, 0.04))[2])

    def on_command(self, q: np.ndarray, width: float) -> None:
        """``FakeDriver.on_command`` hook: one call per pulse write."""
        tcp = planar_tcp(q, self.geometry)
        with self._lock:
            self.tcp = tcp
            if self.attached is not None:
                label = self.attached
                if width > self._width_of(label) + self.WIDTH_SLACK_M:
                    self.objects[label] = np.array([tcp[0], tcp[1], 0.0])
                    self.attached = None
                    _log.info("fake world: released %s at %s", label, np.round(tcp[:2], 3).tolist())
                else:
                    self.objects[label] = tcp.copy()
                return
            best: tuple[float, str] | None = None
            for label, pos in self.objects.items():
                if width >= self._width_of(label) + self.WIDTH_SLACK_M:
                    continue
                xy = float(np.hypot(tcp[0] - pos[0], tcp[1] - pos[1]))
                top = float(pos[2] + self._height_of(label))
                if xy <= self.ATTACH_XY_M and tcp[2] <= top + self.ATTACH_Z_ABOVE_TOP_M:
                    if best is None or xy < best[0]:
                        best = (xy, label)
            if best is not None:
                self.attached = best[1]
                self.objects[best[1]] = tcp.copy()
                _log.info("fake world: attached %s (%.1f mm off centre)", best[1], best[0] * 1000.0)

    def snapshot(self) -> dict[str, Any]:
        """``{objects: {label: [x, y, z]}, attached, tcp}`` for the detector."""
        with self._lock:
            return {
                "objects": {label: [float(v) for v in pos] for label, pos in self.objects.items()},
                "attached": self.attached,
                "tcp": None if self.tcp is None else [float(v) for v in self.tcp],
            }


# ---------------------------------------------------------------------------
# Trajectory follower
# ---------------------------------------------------------------------------


class TrajectoryFollower:
    """Streams a joint trajectory to the driver at the control rate.

    The planner on the laptop emits waypoints at its own ``dt``. Here they are
    clamped to the joint limits, the whole move is **stretched in time** until
    no joint exceeds ``max_joint_velocity`` (never truncated: a shortened path
    ends somewhere the planner did not intend), then resampled at
    ``control_rate_hz`` and written tick by tick. The estop flag is checked
    every tick *under the driver lock, together with the write*: ``estop``
    takes the same lock to set the flag and detach, so a stop can never land
    between the check and the frame that follows it (review HS-7). When the
    flag is raised, or the driver reports a detach hold mid-move (watchdog,
    serial loss), the follower stops writing and reports where it got to.

    ``clamped`` in the reply is true when a waypoint moved by more than
    :data:`CLAMP_TOL_RAD` at the radian limits **or** would be clamped at the
    pulse level -- the second clamp is the one the sketch applies silently.
    """

    def __init__(
        self,
        driver: ServoDriver,
        calibration: ServoCalibration,
        estop: threading.Event,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.driver = driver
        self.calibration = calibration
        self._estop = estop
        self._sleep = sleep
        self._clock = clock

    def plan(self, waypoints: np.ndarray, dt: float) -> tuple[np.ndarray, float, bool, float]:
        """Return ``(samples[M,4], tick_s, clamped, stretch)`` without moving."""
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim == 1 and wp.shape[0] == _N_ARM:
            wp = wp.reshape(1, _N_ARM)
        if wp.ndim != 2 or wp.shape[1] != _N_ARM or wp.shape[0] == 0:
            raise ValueError(f"waypoints must be (N, {_N_ARM}), got shape {wp.shape}")
        if not np.all(np.isfinite(wp)):
            raise ValueError("waypoints contain NaN or inf")
        if not (dt > 0.0) or not math.isfinite(dt):
            raise ValueError(f"dt must be a positive finite number, got {dt}")

        lower, upper = self.calibration.limits()
        clamped_wp = np.minimum(np.maximum(wp, lower), upper)
        clamped = bool(np.any(np.abs(clamped_wp - wp) > CLAMP_TOL_RAD))
        if not clamped:
            # The pulse map is a second, silent clamp; a stale or hand-edited
            # calibration must not hide it. Linear map: checking the waypoints
            # covers every interpolated sample between them.
            clamped = any(self.calibration.pulse_clamped(row) for row in clamped_wp)

        n = clamped_wp.shape[0]
        tick_s = 1.0 / float(self.calibration.control_rate_hz)
        if n == 1:
            return clamped_wp, tick_s, clamped, 1.0

        steps = np.abs(np.diff(clamped_wp, axis=0))
        peak_velocity = float(steps.max()) / dt if steps.size else 0.0
        stretch = max(1.0, peak_velocity / float(self.calibration.max_joint_velocity))
        duration = (n - 1) * dt * stretch

        t_in = np.arange(n) * dt * stretch
        m = max(2, int(math.ceil(duration / tick_s)) + 1)
        t_out = np.linspace(0.0, duration, m)
        samples = np.column_stack(
            [np.interp(t_out, t_in, clamped_wp[:, j]) for j in range(_N_ARM)]
        )
        samples[-1] = clamped_wp[-1]
        return samples, tick_s, clamped, stretch

    def follow(self, waypoints: np.ndarray, dt: float) -> dict[str, Any]:
        """Execute; blocks. Returns the ``follow_trajectory`` reply dict."""
        samples, tick_s, clamped, stretch = self.plan(waypoints, dt)
        start = self._clock()
        estopped = False
        completed = False
        detached = False

        if self._estop.is_set():
            estopped = True
        else:
            next_t = start
            for i, sample in enumerate(samples):
                with self.driver.lock:
                    if self._estop.is_set():
                        estopped = True
                        break
                    try:
                        self.driver.write(sample)
                    except DriverError as exc:
                        # The hold was raised under us (watchdog trip, serial
                        # loss, or an estop that beat us to the lock). Stop
                        # here; the next motion endpoint re-attaches slowly.
                        if self._estop.is_set():
                            estopped = True
                        else:
                            detached = True
                            _log.warning("trajectory stopped at sample %d/%d: %s", i, len(samples), exc)
                        break
                if i == len(samples) - 1:
                    completed = True
                    break
                next_t += tick_s
                remaining = next_t - self._clock()
                if remaining > 0:
                    self._sleep(remaining)

        elapsed = self._clock() - start
        q, width = self.driver.read_commanded()
        if stretch > 1.0 + 1e-9:
            _log.info(
                "trajectory stretched x%.2f to respect max_joint_velocity=%.2f rad/s",
                stretch,
                self.calibration.max_joint_velocity,
            )
        return {
            "q": q.tolist(),
            "gripper_width": float(width),
            "completed": bool(completed),
            "elapsed_s": float(elapsed),
            "clamped": bool(clamped),
            "estopped": bool(estopped),
            "detached": bool(detached),
        }


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class CameraGrabber:
    """Keeps the latest webcam frame; hands out JPEG bytes on request.

    OWNERSHIP. On the Jetson this is the ONLY process that opens the webcam
    (``/dev/video0``). Two V4L2 readers of one UVC camera usually fail with
    EBUSY, and whichever loses silently has no frames; the detector, the
    table calibration and the laptop all take frames through ``get_frame``.

    A background thread drains the V4L2 queue continuously. Without that, the
    driver's internal buffer (typically 4 frames) means ``read()`` returns a
    frame that is 4 frames *old*, and an overhead detection lags the arm by
    130 ms -- enough to place a moving marker 2 cm from where it is.

    Staleness. Every frame carries ``t_capture`` (Jetson wall clock) and
    ``age_s`` measured on the Jetson's *monotonic* clock at reply time, so a
    client can reject an old frame without trusting the two machines' clocks
    to agree (they need not: the Jetson has no RTC battery and syncs NTP
    late). ``get_frame`` refuses a frame older than ``max_age_s`` outright
    (:data:`DEFAULT_MAX_FRAME_AGE_S` unless the request says otherwise): a
    camera that stalls or is unplugged stops the grabber's updates, and the
    last good frame would otherwise be served forever. ``read_failures``
    counts consecutive failed reads (a webcam that loses USB power returns
    ``False`` from every read).

    ``fake=True`` synthesises a 640x480 grey frame with a moving square and a
    timestamp so the transport can be exercised without a camera.
    ``capture_factory(device)`` replaces ``cv2.VideoCapture`` (tests inject a
    capture that stalls or drops frames); ``clock`` is the monotonic clock.
    """

    def __init__(
        self,
        settings: CameraSettings,
        fake: bool = False,
        capture_factory: Callable[[Any], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.fake = fake
        self._capture_factory = capture_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t_capture = 0.0
        self._t_capture_mono = -math.inf
        self._seq = 0
        self.read_failures = 0
        """Consecutive failed ``read()`` calls since the last good frame."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: Any = None
        self._t0 = time.time()

    @staticmethod
    def _cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is not installed; pip3 install opencv-python "
                "(or run with --no-camera)"
            ) from exc
        return cv2

    def start(self) -> None:
        """Open the device (real mode) and start the grabber thread."""
        if self.fake:
            return
        s = self.settings
        device: Any = s.device
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        if self._capture_factory is not None:
            self._cap = self._capture_factory(device)
            if not self._cap.isOpened():
                raise RuntimeError(f"could not open camera {s.device}")
        else:
            cv2 = self._cv2()
            backend = getattr(cv2, "CAP_V4L2", 0)
            self._cap = cv2.VideoCapture(device, backend) if backend else cv2.VideoCapture(device)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"could not open camera {s.device} (is another process -- a detector camera "
                    "loop, a browser, a viewer -- holding it? robot_server must be its only owner)"
                )
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
            self._cap.set(cv2.CAP_PROP_FPS, s.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._loop, name="camera-grabber", daemon=True)
        self._thread.start()
        _log.info("Camera %s open at %dx%d", s.device, s.width, s.height)

    def grab_once(self) -> bool:
        """One read from the capture, as the grabber thread does it; True when
        a frame was stored. Public so a test can drive it without the thread."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            with self._lock:
                self.read_failures += 1
            return False
        with self._lock:
            self._frame = frame
            self._t_capture = time.time()
            self._t_capture_mono = self._clock()
            self._seq += 1
            self.read_failures = 0
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.grab_once():
                time.sleep(0.01)

    def _synthesise(self) -> np.ndarray:
        cv2 = self._cv2()
        w, h = self.settings.width, self.settings.height
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        t = time.time() - self._t0
        cx = int(w * (0.5 + 0.35 * math.sin(t)))
        cy = int(h * (0.5 + 0.25 * math.cos(0.7 * t)))
        cv2.rectangle(frame, (cx - 25, cy - 25), (cx + 25, cy + 25), (40, 40, 200), -1)
        cv2.putText(
            frame,
            f"fake {time.strftime('%H:%M:%S')} #{self._seq}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame

    def get_frame(self, quality: int | None = None, max_age_s: float | None = None) -> dict[str, Any]:
        """Latest frame as ``{jpeg, width, height, t_capture, age_s, seq, t_server}``.

        ``age_s`` is how old the frame is *now*, on the Jetson's monotonic
        clock; ``t_capture`` / ``t_server`` are the Jetson's wall clock at
        capture and at reply. ``seq`` counts captured frames, so two replies
        with the same ``seq`` are the same frame. Raises ``RuntimeError``
        naming the age when the latest frame is older than ``max_age_s``
        (default :data:`DEFAULT_MAX_FRAME_AGE_S`; ``<= 0`` disables the check).
        """
        cv2 = self._cv2()
        q = int(quality) if quality is not None else self.settings.jpeg_quality
        q = min(max(q, 1), 100)
        limit = DEFAULT_MAX_FRAME_AGE_S if max_age_s is None else float(max_age_s)
        if self.fake:
            with self._lock:
                self._seq += 1
                seq = self._seq
            frame = self._synthesise()
            t_capture = time.time()
            age = 0.0
        else:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
                t_capture, seq = self._t_capture, self._seq
                age = self._clock() - self._t_capture_mono
                failures = self.read_failures
            if frame is None:
                raise RuntimeError(f"no camera frame captured yet ({failures} failed reads so far)")
            if limit > 0.0 and age > limit:
                raise RuntimeError(
                    f"stale camera frame: the latest is {age:.2f} s old (limit {limit:.2f} s, "
                    f"{failures} failed reads since); the camera stalled or was unplugged"
                )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        h, w = frame.shape[:2]
        return {
            "jpeg": buf.tobytes(),
            "width": int(w),
            "height": int(h),
            "t_capture": float(t_capture),
            "age_s": float(max(0.0, age)),
            "seq": int(seq),
            "t_server": time.time(),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:  # pragma: no cover - hardware
                _log.warning("camera release failed: %s", exc)
            self._cap = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _pack(payload: Any) -> bytes:
    import msgpack
    import msgpack_numpy

    return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)


def _unpack(raw: bytes) -> Any:
    import msgpack
    import msgpack_numpy

    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)


class RobotServer:
    """ZMQ front door: one ROUTER socket, one worker for motion, instant estop.

    ``serve_forever()`` for the deployed process, ``serve_in_thread()`` for
    tests (``port=0`` picks a free port and returns it). ``stop()`` is safe
    from any thread and always detaches the servos.

    Host liveness: every request from any client refreshes the host clock;
    :meth:`check_host_liveness` (run by the serving loop every poll) detaches
    the servos once ``host_timeout_s`` passes with no request and no motion
    running. ``clock`` is injectable so a test can drive it without waiting.
    """

    MOTION_ENDPOINTS = frozenset({"follow_trajectory", "home", "set_gripper", "set_torque"})
    NOT_HOST_LIVENESS = frozenset({"get_fake_world"})
    """Requests that do not prove the *laptop* is alive: the fake detector
    polls ``get_fake_world`` on its own and would otherwise keep a fake arm
    attached after the laptop died."""

    def __init__(
        self,
        host: str,
        port: int,
        driver: ServoDriver,
        camera: CameraGrabber | None,
        calibration: ServoCalibration,
        calibration_path: str | Path | None = None,
        follower_sleep: Callable[[float], None] | None = None,
        fake_world: FakeWorld | None = None,
        geometry: ArmGeometry | None = None,
        host_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        file_calibration: ServoCalibration | None = None,
        placeholder_band_us: tuple[int, int] | None = None,
    ) -> None:
        """``geometry`` (link lengths) backs the home-pose table check on
        ``set_calibration``; defaults match ``hardware.arm``.
        ``host_timeout_s`` overrides ``calibration.host_timeout_s`` (0
        disables the bound).

        Placeholder mode (``placeholder_band_us`` set; see
        :func:`startup_calibration`): ``calibration`` is the *active*,
        band-limited one the driver enforces and ``file_calibration`` the one
        on disk, which ``get_calibration`` also reports and ``set_calibration``
        persists -- the band is a property of this run, never written to the
        file. Every calibration pushed during the run is band-limited again
        until the server restarts without the flag."""
        self.host = host
        self.fake_world = fake_world
        self.port = port
        self.driver = driver
        self.camera = camera
        self.calibration = calibration
        self.file_calibration = file_calibration if file_calibration is not None else calibration
        self.placeholder_band_us = None if placeholder_band_us is None else (
            int(placeholder_band_us[0]), int(placeholder_band_us[1])
        )
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.geometry = geometry or ArmGeometry()
        self.fake = isinstance(driver, FakeDriver)
        self._host_timeout_override = None if host_timeout_s is None else float(host_timeout_s)
        self._clock = clock

        self._estop = threading.Event()
        self._moving = threading.Event()
        self._stop = threading.Event()
        self._replies: "queue.Queue[tuple[bytes, Any]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._context: Any = None
        self._socket: Any = None
        self.bound_port: int | None = None
        self._probe_failed_logged = False
        self._last_probe: tuple[tuple[int, ...], bool] | None = None
        self._last_request_t = self._clock()
        self.host_lost = False
        """True from a host-timeout detach until the next request arrives."""
        # ``follower_sleep`` replaces the follower's real-time pacing. Tests pass a
        # no-op so a 2 s trajectory costs no wall time; the deployed server keeps
        # ``time.sleep`` so the Uno receives frames at the control rate. The
        # driver's re-attach ramp is paced the same way.
        sleep = time.sleep if follower_sleep is None else follower_sleep
        self.follower = TrajectoryFollower(driver, calibration, self._estop, sleep=sleep)
        driver.sleep = sleep
        driver.start_keepalive()

    @property
    def host_timeout_s(self) -> float:
        """Seconds of client silence before an idle arm is detached (0: never)."""
        if self._host_timeout_override is not None:
            return self._host_timeout_override
        return float(self.calibration.host_timeout_s)

    # -- lifecycle ----------------------------------------------------------

    def bind(self) -> int:
        """Bind the ROUTER socket; returns the port actually bound."""
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is not installed; pip3 install pyzmq") from exc
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        if self.port == 0:
            self.bound_port = int(self._socket.bind_to_random_port(f"tcp://{self.host}"))
        else:
            self._socket.bind(f"tcp://{self.host}:{self.port}")
            self.bound_port = self.port
        _log.info("robot_server listening on tcp://%s:%d (%s)", self.host, self.bound_port, self.driver.name)
        return self.bound_port

    def serve_forever(self) -> None:
        """Bind (if needed) and run the loop in the calling thread until :meth:`stop`."""
        if self._socket is None:
            self.bind()
        self._loop()

    def serve_in_thread(self, port: int | None = None) -> tuple["RobotServer", int]:
        """Bind now, serve in a daemon thread; returns ``(self, bound_port)``."""
        if port is not None:
            self.port = port
        bound = self.bind()
        self._thread = threading.Thread(target=self._loop, name="robot-server", daemon=True)
        self._thread.start()
        return self, bound

    def stop(self) -> None:
        """Stop serving, detach the servos, close the socket. Idempotent."""
        self._stop.set()
        self._estop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        self.driver.stop_keepalive()
        try:
            self.driver.detach("server stop")
        except Exception as exc:
            _log.warning("detach on stop failed: %s", exc)
        if self.camera is not None:
            self.camera.stop()
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

    def __enter__(self) -> "RobotServer":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # -- loop ---------------------------------------------------------------

    def _loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                events = dict(poller.poll(timeout=50))
                if self._socket in events:
                    frames = self._socket.recv_multipart()
                    self._dispatch(frames)
                self._flush_replies()
                self.check_host_liveness()
        except Exception as exc:  # pragma: no cover - only on a torn-down socket
            if not self._stop.is_set():
                _log.exception("server loop died: %s", exc)
        finally:
            self._flush_replies()

    def _flush_replies(self) -> None:
        while True:
            try:
                ident, reply = self._replies.get_nowait()
            except queue.Empty:
                return
            try:
                self._socket.send_multipart([ident, b"", _pack(reply)])
            except Exception as exc:
                _log.warning("reply to %r failed: %s", ident, exc)

    def note_request(self) -> None:
        """A client spoke: refresh the host clock (and say so after a timeout)."""
        self._last_request_t = self._clock()
        if self.host_lost:
            self.host_lost = False
            _log.warning("a client is talking again after the host timeout; the next motion re-attaches slowly")

    def check_host_liveness(self) -> bool:
        """Detach an idle, attached arm nobody has talked to for ``host_timeout_s``.

        True when it detached. Never while a motion runs (the motion's own
        client is blocked waiting for its reply, legitimately silent); the
        silence is measured from the later of the last request and the end of
        the last motion. The detach raises the hold, so the keepalive stops
        feeding even if the ``D`` itself fails, and the Uno's watchdog then
        relaxes the arm within 500 ms.
        """
        timeout = self.host_timeout_s
        if timeout <= 0.0 or self.host_lost or self._moving.is_set():
            return False
        if self._worker is not None and self._worker.is_alive():
            return False
        silent = self._clock() - self._last_request_t
        if silent < timeout:
            return False
        if not self.driver.attached:
            return False
        _log.error(
            "no request from any client for %.1f s (host_timeout_s = %.1f): the laptop process died, "
            "lost its network, or froze. Detaching the servos so nothing is held, squeezed or pushed "
            "indefinitely; the arm goes limp and drops what it holds. The next motion re-attaches slowly.",
            silent, timeout,
        )
        self.host_lost = True
        try:
            with self.driver.lock:
                self.driver.detach(f"host timeout: no client request for {silent:.1f} s")
        except DriverError as exc:
            _log.error("host-timeout detach failed: %s", exc)
        return True

    def _dispatch(self, frames: list[bytes]) -> None:
        if len(frames) < 2:
            _log.warning("malformed ROUTER frame set (%d parts)", len(frames))
            return
        ident, payload = frames[0], frames[-1]
        try:
            request = _unpack(payload)
            endpoint = str(request.get("endpoint", ""))
            data = request.get("data") or {}
            if not isinstance(data, dict):
                raise ValueError("data must be a mapping")
        except Exception as exc:
            self.note_request()
            self._replies.put((ident, {"error": f"bad request: {exc}"}))
            return
        if endpoint not in self.NOT_HOST_LIVENESS:
            self.note_request()

        if endpoint in self.MOTION_ENDPOINTS:
            if self._worker is not None and self._worker.is_alive():
                self._replies.put((ident, {"error": "busy: a motion command is still running"}))
                return
            # Set here, not in the worker: the liveness check must never see
            # a motion that has been accepted but not yet started as idle.
            self._moving.set()
            self._worker = threading.Thread(
                target=self._run_motion, args=(ident, endpoint, data), name="robot-motion", daemon=True
            )
            self._worker.start()
            return

        self._replies.put((ident, self._handle(endpoint, data)))

    def _run_motion(self, ident: bytes, endpoint: str, data: dict[str, Any]) -> None:
        try:
            reply = self._handle(endpoint, data)
        finally:
            self._last_request_t = self._clock()
            self._moving.clear()
        self._replies.put((ident, reply))

    def _handle(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_ep_{endpoint}", None)
        if handler is None or not endpoint or endpoint.startswith("_"):
            return {"error": f"unknown endpoint {endpoint!r}"}
        try:
            return handler(data)
        except Exception as exc:
            _log.warning("%s failed: %s: %s", endpoint, type(exc).__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -- endpoints ----------------------------------------------------------

    def state(self, probe: bool = True) -> dict[str, Any]:
        """The ``get_state`` reply: commanded values, never measured ones.

        ``attached`` is whether the servos are being pulsed. Drivers that can
        ask their bridge (the Uno's ``?``, the fake's watchdog model) are asked
        here, cheaply, and believed over the server's own flag: when the Uno
        reports detached while the server thought otherwise, a warning is
        logged, the driver is marked detached and the next motion re-attaches
        slowly. ``bridge_q`` / ``bridge_gripper_width`` are the bridge's own
        last pulsed position through the current calibration (``None`` when
        the back-end cannot tell, *and* when the bridge knows no position --
        a reset Uno's 1500 us are a library default, not a pose), quantised
        to the whole microseconds the bridge holds (about 1e-3 rad); the
        calibration REPL uses them as the target for ``torque on`` so
        re-energising moves nothing. ``bridge_position_known`` is False after
        every bridge reset until the first attach; ``detach_count`` /
        ``last_detach_reason`` count every detach so a client can notice one
        it did not ask for.

        ``probe=False`` skips the bridge query and answers from the server's
        own belief only. The estop reply uses it: a dead or out-of-step Uno
        costs a reply timeout (plus a failed sync) per ``?``, and the estop
        reply must not wait on the wire once its ``D`` attempts are over.
        ``bridge_q`` / ``bridge_gripper_width`` are then ``None`` ("cannot
        tell"), never the last probe on record: nothing refreshes that probe
        after a motion, so it can be a whole trajectory old. ``bridge_probed``
        says which kind of reply this is.
        """
        if probe:
            self._reconcile_attach_state()
        q, width = self.driver.read_commanded()
        bridge_q: list[float] | None = None
        bridge_width: float | None = None
        known = self.driver.position_known
        last_probe = self._last_probe if probe else None
        if last_probe is not None and known:
            pulses, _attached = last_probe
            bridge_q = [float(v) for v in self.calibration.pulses_to_q(pulses[:_N_ARM])]
            bridge_width = float(self.calibration.us_to_width(pulses[_N_ARM]))
        return {
            "q": [float(v) for v in q],
            "gripper_width": float(width),
            "moving": bool(self._moving.is_set()),
            "estopped": bool(self._estop.is_set()),
            "attached": bool(self.driver.attached),
            "bridge_q": bridge_q,
            "bridge_gripper_width": bridge_width,
            "bridge_position_known": bool(known),
            "bridge_probed": bool(probe),
            "detach_count": int(self.driver.detach_count),
            "last_detach_reason": self.driver.last_detach_reason,
            "host_timeout_s": float(self.host_timeout_s),
            "host_lost": bool(self.host_lost),
            "calibration_measured": bool(self.file_calibration.measured),
            "placeholder_band_us": None if self.placeholder_band_us is None else list(self.placeholder_band_us),
            "t": time.time(),
        }

    def _reconcile_attach_state(self) -> None:
        """Ask the bridge whether the servos are attached; believe it."""
        try:
            probe = self.driver.probe()
        except DriverError as exc:
            if not self._probe_failed_logged:
                _log.warning("bridge probe failed (%s); attach state is the server's own belief", exc)
                self._probe_failed_logged = True
            self._last_probe = None
            return
        self._probe_failed_logged = False
        self._last_probe = probe
        if probe is None:
            return
        _pulses, bridge_attached = probe
        if not bridge_attached and self.driver.attached:
            self.driver.mark_detached("the bridge reports its servos detached (watchdog fired?)")

    def _require_motion_allowed(self) -> None:
        if self._estop.is_set():
            raise DriverError("estopped: motion refused until clear_estop")

    def _require_known_position(self, endpoint: str) -> None:
        """Refuse every motion but ``home`` while the bridge knows no position."""
        self._reconcile_attach_state()
        if self.driver.position_known:
            return
        raise DriverError(
            f"{endpoint} refused: the servo bridge has no known position (the Uno resets whenever "
            "the serial port opens -- every robot_server start -- and after a USB reset or "
            "brownout), so its first frame would attach every servo AT the target at full speed. "
            "Hand-pose the arm at home_q, then send `home` (calibrate_servos.py: `home`); every "
            "command works normally after that"
        )

    def _ensure_attached(
        self, target_pulses: Sequence[float] | None, duration_s: float, reconcile: bool = True
    ) -> bool:
        """Re-attach slowly if the servos are not being pulsed; True when it did.

        Called at the top of every motion endpoint. ``target_pulses`` (all
        five channels; ``None`` = the stored pose) is where the move ends;
        the bridge starts from its last pulsed position and takes
        ``duration_s`` to get there. Skipped -- an ordinary write follows --
        when the servos are already attached. ``reconcile=False`` when the
        caller has just asked the bridge (one ``?`` per command is enough).
        """
        if reconcile:
            self._reconcile_attach_state()
        if not self.driver.needs_reattach:
            return False
        self.driver.reattach(target_pulses, duration_s=duration_s, abort=self._estop)
        return True

    def _reattach_duration(self, target_q: np.ndarray, floor_s: float) -> float:
        """``floor_s`` stretched so no joint exceeds ``max_joint_velocity``.

        Travel is measured from where the *bridge* is (its last ``?`` answer)
        when that is known: an estop or watchdog trip during a long ``T``
        leaves the Uno partway, while the stored pulses already equal the old
        target, and the interpolation starts from the Uno's real position.
        """
        q_now, _ = self.driver.read_commanded()
        target = np.asarray(target_q, dtype=np.float64)
        travel = float(np.max(np.abs(target - q_now)))
        probe = self._last_probe
        if probe is not None and self.driver.position_known:
            bridge_q = self.calibration.pulses_to_q(probe[0][:_N_ARM])
            travel = max(travel, float(np.max(np.abs(target - bridge_q))))
        return max(float(floor_s), travel / float(self.calibration.max_joint_velocity))

    def _motion_state(self, reattached: bool, **extra: Any) -> dict[str, Any]:
        state = self.state()
        state["reattached"] = bool(reattached)
        state.update(extra)
        return state

    def _ep_ping(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "server": SERVER_NAME,
            "version": PROTOCOL_VERSION,
            "driver": self.driver.name,
            "fake": self.fake,
            "arm_geometry": dataclasses.asdict(self.geometry),
            "host_timeout_s": float(self.host_timeout_s),
            "calibration_measured": bool(self.file_calibration.measured),
            "placeholder_band_us": None if self.placeholder_band_us is None else list(self.placeholder_band_us),
            "camera": self.camera is not None,
            "t": time.time(),
        }

    def _ep_get_state(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.state()

    def _ep_get_frame(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.camera is None:
            raise RuntimeError(
                "no camera on this server (started with --no-camera, or the camera failed to open "
                "at start -- see the server log)"
            )
        max_age = data.get("max_age_s")
        return self.camera.get_frame(data.get("quality"), None if max_age is None else float(max_age))

    def _ep_set_gripper(self, data: dict[str, Any]) -> dict[str, Any]:
        self._require_motion_allowed()
        if "width" not in data:
            raise ValueError("set_gripper needs {'width': metres}")
        width = float(data["width"])
        if not math.isfinite(width):
            raise ValueError("width must be finite")
        self._require_known_position("set_gripper")
        self._moving.set()
        try:
            # A gripper command re-energises all five channels; after a detach
            # it must therefore take the slow path like any arm move.
            pulses = list(self.driver.pulses())
            pulses[_N_ARM] = self.calibration.width_to_us(width)
            reattached = self._ensure_attached(pulses, self.calibration.reattach_s, reconcile=False)
            if not reattached:
                self.driver.write_gripper(width)
        finally:
            self._moving.clear()
        return self._motion_state(reattached)

    def _ep_follow_trajectory(self, data: dict[str, Any]) -> dict[str, Any]:
        self._require_motion_allowed()
        if "waypoints" not in data or "dt" not in data:
            raise ValueError("follow_trajectory needs {'waypoints': f32[N,4], 'dt': s}")
        waypoints = np.asarray(data["waypoints"], dtype=np.float64)
        dt = float(data["dt"])
        self._require_known_position("follow_trajectory")
        self._moving.set()
        try:
            samples, _tick, _clamped, _stretch = self.follower.plan(waypoints, dt)
            first = list(self.calibration.q_to_pulses(samples[0])) + [self.driver.pulses()[_N_ARM]]
            reattached = self._ensure_attached(
                first, self._reattach_duration(samples[0], self.calibration.reattach_s), reconcile=False
            )
            reply = self.follower.follow(waypoints, dt)
        finally:
            self._moving.clear()
        reply["reattached"] = bool(reattached)
        reply["detach_count"] = int(self.driver.detach_count)
        reply["last_detach_reason"] = self.driver.last_detach_reason
        reply["attached"] = bool(self.driver.attached)
        return reply

    def _ep_home(self, data: dict[str, Any]) -> dict[str, Any]:
        self._require_motion_allowed()
        home = np.asarray(self.calibration.home_q, dtype=np.float64)
        self._reconcile_attach_state()
        first_attach = not self.driver.position_known
        self._moving.set()
        try:
            home_pulses = list(self.calibration.q_to_pulses(home)) + [self.driver.pulses()[_N_ARM]]
            duration = self._reattach_duration(home, self.calibration.home_move_s)
            if self._ensure_attached(home_pulses, duration, reconcile=False):
                # Detached (estop, torque off, watchdog): one slow move from
                # the last pulsed pose straight to home; no re-attach at the
                # pre-detach pose first. From an unknown position (boot, a
                # bridge reset) the bridge attached AT home, instantly.
                if first_attach:
                    _log.warning(
                        "home from an unknown position: the servos attached AT home at full speed. "
                        "If the arm was not hand-posed at home it jumped there."
                    )
                return self._motion_state(True, first_attach=first_attach)
            q_now, _ = self.driver.read_commanded()
            # Two waypoints; the follower stretches the time to max_joint_velocity.
            waypoints = np.stack([q_now, home])
            dt = float(np.max(np.abs(home - q_now)) / self.calibration.max_joint_velocity)
            dt = max(dt, 1.0 / self.calibration.control_rate_hz)
            self.follower.follow(waypoints, dt)
        finally:
            self._moving.clear()
        return self._motion_state(False, first_attach=False)

    def _ep_estop(self, data: dict[str, Any]) -> dict[str, Any]:
        # Flag and detach under the driver lock: the follower holds the same
        # lock across its estop check and write, so no P frame can follow the
        # D frame (review HS-7). The hold raised by detach() covers any path
        # that does not check the flag at all -- including a D the bridge
        # rejected, which used to leave the arm energised under the keepalive.
        detach_error: str | None = None
        with self.driver.lock:
            self._estop.set()
            try:
                self.driver.detach("estop")
            except DriverError as exc:
                detach_error = str(exc)
        if detach_error is None:
            _log.warning("ESTOP: servos detached; motion refused until clear_estop")
        else:
            _log.error("ESTOP: %s", detach_error)
        # Cached/commanded state only: no ``?`` after the D attempts, so a dead
        # or out-of-step Uno cannot delay the reply by further timeouts.
        # ``attached`` is already False (detach() lowers it before the wire).
        state = self.state(probe=False)
        state["detach_error"] = detach_error
        return state

    def _ep_clear_estop(self, data: dict[str, Any]) -> dict[str, Any]:
        self._estop.clear()
        _log.info("estop cleared; the next motion command re-attaches the servos slowly")
        return self.state()

    def _ep_set_torque(self, data: dict[str, Any]) -> dict[str, Any]:
        """``{enabled}``; with ``enabled: true`` an optional ``q`` (radians,
        4) and ``gripper_width`` name the pose to re-energise *toward* instead
        of the stored one.

        Torque-on ALWAYS re-energises at the bridge's LAST PULSED pose: the
        sketch keeps ``current_us`` across a detach and its first pulse after
        re-attach is that pulse, whatever the arm was hand-posed to since.
        Feedback-less servos cannot be read, so if the arm was moved by hand
        while limp, every moved servo JUMPS back to its last pulsed position
        at full speed (hundreds of ms for an MG996R), and only then does the
        ``T`` interpolation toward ``q`` start. The calibration REPL passes
        the bridge's own last position as ``q`` so that, *for an arm that was
        not moved*, nothing moves after the jump-free re-attach -- it warns
        and asks for ``yes`` before sending this. Refused while the bridge
        knows no position: the first frame would snap every servo to its
        target."""
        enabled = bool(data.get("enabled", True))
        if not enabled:
            self.driver.torque(False)
            return self.state()
        self._require_motion_allowed()
        self._require_known_position("set_torque")
        target = list(self.driver.pulses())
        if data.get("q") is not None:
            q = np.asarray(data["q"], dtype=np.float64).reshape(-1)
            if q.shape[0] != _N_ARM or not np.all(np.isfinite(q)):
                raise ValueError(f"set_torque q must be {_N_ARM} finite radians")
            q, _ = self.calibration.clamp_q(q)
            target[:_N_ARM] = self.calibration.q_to_pulses(q)
        if data.get("gripper_width") is not None:
            target[_N_ARM] = self.calibration.width_to_us(float(data["gripper_width"]))
        self._moving.set()
        try:
            target_q = self.calibration.pulses_to_q(target[:_N_ARM])
            reattached = self._ensure_attached(
                target, self._reattach_duration(target_q, self.calibration.reattach_s), reconcile=False
            )
            if not reattached:
                self.driver.reattach(target, duration_s=self.calibration.reattach_s, abort=self._estop)
        finally:
            self._moving.clear()
        return self._motion_state(reattached)

    def _ep_get_fake_world(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.fake_world is None:
            raise RuntimeError("no fake world on this server (start with --driver fake --fake-world ...)")
        return self.fake_world.snapshot()

    def _ep_get_calibration(self, data: dict[str, Any]) -> dict[str, Any]:
        """``calibration`` is what the server enforces now (band-limited in
        placeholder mode); ``file_calibration`` is what is on disk and what a
        calibration tool should edit."""
        return {
            "calibration": self.calibration.to_dict(),
            "file_calibration": self.file_calibration.to_dict(),
            "measured": bool(self.file_calibration.measured),
            "placeholder_band_us": None if self.placeholder_band_us is None else list(self.placeholder_band_us),
            "path": str(self.calibration_path) if self.calibration_path else None,
            "arm_geometry": dataclasses.asdict(self.geometry),
        }

    def _ep_set_calibration(self, data: dict[str, Any]) -> dict[str, Any]:
        if "calibration" not in data:
            raise ValueError("set_calibration needs {'calibration': {...}}")
        new = ServoCalibration.from_dict(data["calibration"], self.geometry)
        active = new
        if self.placeholder_band_us is not None:
            # The band belongs to this run (--allow-placeholder-calibration),
            # not to the file: re-apply it, and lift it only by a restart.
            active = placeholder_safe_calibration(new, self.placeholder_band_us, self.geometry)
            if new.measured:
                _log.warning(
                    "calibration marked measured; the placeholder band %s us stays in force until "
                    "robot_server.py restarts without --allow-placeholder-calibration",
                    self.placeholder_band_us,
                )
        elif self.file_calibration.measured and not new.measured:
            _log.warning(
                "the pushed calibration is marked measured: false; a real driver will refuse it at "
                "the next start without --allow-placeholder-calibration"
            )
        # Swap atomically: the driver derives q from pulses through this object,
        # so re-labelling never moves the arm.
        self.calibration = active
        self.file_calibration = new
        self.driver.calibration = active
        self.follower.calibration = active
        if self.calibration_path is not None:
            new.save(self.calibration_path)
            _log.info("calibration saved to %s", self.calibration_path)
        return {
            "ok": True,
            "path": str(self.calibration_path) if self.calibration_path else None,
            "measured": bool(new.measured),
            "placeholder_band_us": None if self.placeholder_band_us is None else list(self.placeholder_band_us),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


REAL_DRIVERS = frozenset({"uno", "pca9685"})
"""Drivers that move real servos: they refuse an unmeasured calibration."""


def startup_calibration(
    driver_kind: str,
    config_path: str | Path,
    geometry: ArmGeometry,
    allow_placeholder: bool = False,
    band_us: tuple[int, int] = PLACEHOLDER_PULSE_BAND_US,
) -> tuple[ServoCalibration, ServoCalibration, tuple[int, int] | None]:
    """``(file_calibration, active_calibration, band_or_None)`` for a start.

    Raises :class:`CalibrationError` with the reason to print when the server
    must not start:

    * a missing file with a real driver (the fake driver uses the placeholder);
    * ``measured: false`` with a real driver (``uno``/``pca9685``) unless
      ``allow_placeholder``: every value in the shipped ``robot_config.yaml``
      is a kit-sized guess, and nothing stops a guessed pulse map from driving
      a servo into its end-stop;
    * with ``allow_placeholder`` and an unmeasured file, the active
      calibration is :func:`placeholder_safe_calibration` -- pulses inside
      ``band_us``, speed capped, slower re-attach -- and the band is returned
      so the server keeps applying it. A measured file ignores the flag.
    """
    path = Path(config_path)
    real = driver_kind in REAL_DRIVERS
    if path.exists():
        file_cal = ServoCalibration.load(path, geometry)
    elif not real:
        _log.warning("%s not found; using the placeholder calibration (fake driver)", path)
        file_cal = ServoCalibration.default()
    else:
        raise CalibrationError(f"calibration file {path} not found; refusing to drive real servos blind")
    if file_cal.measured:
        if allow_placeholder:
            _log.info("%s is measured; --allow-placeholder-calibration has no effect", path)
        return file_cal, file_cal, None
    if not allow_placeholder:
        if real:
            raise CalibrationError(
                f"{path} says measured: false -- its pulse maps, limits and home_q are placeholders, "
                "and nothing stops a guessed pulse from driving a servo into its end-stop (1.4-2.5 A "
                "stall). Measure it (smoke test S5 + scripts/calibrate_servos.py, then `measured yes; "
                "save`), or for the FIRST bench bring-up only pass --allow-placeholder-calibration, "
                f"which keeps every pulse inside {band_us[0]}..{band_us[1]} us and caps the speed"
            )
        return file_cal, file_cal, None
    active = placeholder_safe_calibration(file_cal, band_us, geometry)
    _log.warning(
        "PLACEHOLDER CALIBRATION (%s, measured: false): pulses kept inside %d..%d us, max joint "
        "velocity %.2f rad/s, re-attach %.1f s, home move %.1f s. For bench bring-up only.",
        path, band_us[0], band_us[1], active.max_joint_velocity, active.reattach_s, active.home_move_s,
    )
    return file_cal, active, (int(band_us[0]), int(band_us[1]))


def _parse_band(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("expected LO,HI in microseconds, e.g. 1000,2000")
    lo, hi = int(parts[0]), int(parts[1])
    if lo >= hi:
        raise ValueError(f"empty band {lo}..{hi} us")
    return lo, hi


def build_driver(
    kind: str,
    calibration: ServoCalibration,
    serial_port: str,
    i2c_bus: int,
    fake_watchdog_ms: float | None = None,
) -> ServoDriver:
    """Instantiate the driver named on the command line."""
    if kind == "fake":
        return FakeDriver(calibration, watchdog_ms=fake_watchdog_ms)
    if kind == "uno":
        return UnoSerialDriver(calibration, port=serial_port)
    if kind == "pca9685":
        return Pca9685Driver(calibration, bus=i2c_bus)
    raise ValueError(f"unknown driver {kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson robot server for the hobby arm")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="servo calibration YAML")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; 0.0.0.0 for the LAN")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--driver", choices=("uno", "pca9685", "fake"), default="uno")
    parser.add_argument("--serial", default="/dev/ttyACM0", help="Uno port (--driver uno)")
    parser.add_argument("--i2c-bus", type=int, default=7, help="I2C bus (--driver pca9685)")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="serve without get_frame. The camera is ON by default: this process is the only one "
        "that opens the webcam, everything else (detector, calibrate_table, laptop) takes frames "
        "through get_frame",
    )
    parser.add_argument(
        "--require-camera",
        action="store_true",
        help="exit instead of serving without get_frame when the camera does not open (the MVP run "
        "line: a demo without frames is not a demo)",
    )
    parser.add_argument(
        "--allow-placeholder-calibration",
        action="store_true",
        help="(uno/pca9685) start on a calibration that says measured: false -- FIRST bench bring-up "
        "only; pulses are kept inside --placeholder-band-us and the speed is capped",
    )
    parser.add_argument(
        "--placeholder-band-us",
        default=f"{PLACEHOLDER_PULSE_BAND_US[0]},{PLACEHOLDER_PULSE_BAND_US[1]}",
        metavar="LO,HI",
        help="pulse band for --allow-placeholder-calibration (default %(default)s; 600,2400 only with "
        "the linkages off, for the S5 end-stop search)",
    )
    parser.add_argument("--fake-camera", action="store_true", help="synthesise frames")
    parser.add_argument(
        "--fake-world",
        default=None,
        metavar="SCENE",
        help='(--driver fake) objects the fake jaw can pick up, e.g. "marker:0.18,0.05 bowl:0.15,-0.12"; '
        "served on get_fake_world for detector_service.py --world-from",
    )
    parser.add_argument(
        "--arm-geometry",
        default="0.07,0.0,0.105,0.10,0.09",
        help="base_height,shoulder_offset,upper_arm,forearm,tool in metres for the fake world's FK "
        "and the home-pose table check (must equal hardware.arm in the laptop config)",
    )
    parser.add_argument(
        "--fake-watchdog-ms",
        type=float,
        default=None,
        metavar="MS",
        help="(--driver fake) detach after this many ms without a frame, like the Uno's 500 ms "
        "watchdog; the keepalive then has to keep the fake arm attached through every idle gap",
    )
    parser.add_argument(
        "--host-timeout-s",
        type=float,
        default=None,
        metavar="S",
        help="detach the servos after this many seconds without any client request (laptop crash, "
        "Wi-Fi loss); overrides host_timeout_s in the calibration file (default 5 s); 0 disables it",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        geometry = ArmGeometry.parse(args.arm_geometry)
    except ValueError as exc:
        _log.error("--arm-geometry: %s", exc)
        return 2

    config_path = Path(args.config)
    try:
        band = _parse_band(args.placeholder_band_us)
    except ValueError as exc:
        _log.error("--placeholder-band-us: %s", exc)
        return 2
    try:
        file_calibration, calibration, placeholder_band = startup_calibration(
            args.driver, config_path, geometry, args.allow_placeholder_calibration, band
        )
        _log.info("Calibration from %s (measured: %s)", config_path, file_calibration.measured)
    except CalibrationError as exc:
        _log.error("calibration %s rejected: %s", config_path, exc)
        return 2

    if args.fake_watchdog_ms is not None and args.driver != "fake":
        _log.error("--fake-watchdog-ms needs --driver fake")
        return 2
    try:
        driver = build_driver(args.driver, calibration, args.serial, args.i2c_bus, args.fake_watchdog_ms)
    except DriverError as exc:
        _log.error("%s", exc)
        return 2

    fake_world: FakeWorld | None = None
    if args.fake_world:
        if not isinstance(driver, FakeDriver):
            _log.error("--fake-world needs --driver fake")
            return 2
        # The footprint table and scene grammar live next to the scripted
        # detector; both files ship together on the Jetson, so import by path.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from jetson.detector_service import DEFAULT_FOOTPRINTS, parse_scene  # noqa: PLC0415

        fake_world = FakeWorld(parse_scene(args.fake_world), DEFAULT_FOOTPRINTS, geometry)
        driver.on_command = fake_world.on_command
        _log.info("Fake world: %s", ", ".join(f"{k}@{v[:2].tolist()}" for k, v in fake_world.objects.items()))

    camera: CameraGrabber | None = None
    if args.no_camera and args.require_camera:
        _log.error("--no-camera and --require-camera contradict each other")
        driver.close()
        return 2
    if not args.no_camera:
        camera = CameraGrabber(calibration.camera, fake=args.fake_camera or args.driver == "fake")
        try:
            camera.start()
        except Exception as exc:
            if args.require_camera:
                _log.error("camera unavailable (%s) and --require-camera is set; exiting", exc)
                driver.close()
                return 2
            _log.warning("camera unavailable (%s); serving without get_frame", exc)
            camera = None

    if args.host_timeout_s is not None and args.host_timeout_s < 0:
        _log.error("--host-timeout-s must be >= 0")
        return 2
    server = RobotServer(
        args.host, args.port, driver, camera, calibration, config_path,
        fake_world=fake_world, geometry=geometry, host_timeout_s=args.host_timeout_s,
        file_calibration=file_calibration, placeholder_band_us=placeholder_band,
    )
    if isinstance(driver, Pca9685Driver) and server.host_timeout_s <= 0:
        _log.error(
            "the PCA9685 has no watchdog: with the host timeout disabled nothing would ever relax the "
            "arm after a laptop crash. Refusing --host-timeout-s 0 with --driver pca9685"
        )
        server.stop()
        driver.close()
        return 2
    _log.info(
        "Host-liveness bound: servos detach after %.1f s without a client request%s",
        server.host_timeout_s, " (DISABLED)" if server.host_timeout_s <= 0 else "",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("Shutting down")
    finally:
        server.stop()
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``IRobot`` for the hobby arm behind ``jetson/robot_server.py``.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers.

The framework's ``IRobot`` surface actually used outside the simulation lane is
thirteen methods (audited in the plan: ``joint_names, config, get_state,
tcp_pose, forward_kinematics, inverse_kinematics, get_arm_joint_positions,
set_arm_joint_targets, open_gripper, close_gripper, get_gripper_width,
get_gripper_state, go_home_immediate``). This class provides exactly those,
plus what the hardware planner and controller need (``follow_trajectory``,
``estop``, ``both_branches``, ``within_limits``, ``close_gripper_to``), and
makes ``articulation`` raise: the two sim-lane readers of it are replaced on
this lane, so any other access is a bug and must not be silently satisfied.

Why every state here is *commanded* rather than *measured*: PWM hobby servos
have no readback. The server derives ``q`` from the last pulses it sent
through its calibration, and that is the only "position" that exists. The
consequences are deliberate and visible in the API:

* velocities in :class:`RobotState` are zero,
* ``GripperState.is_grasping`` is always ``False`` -- a commanded close is not
  evidence, so grasp verification is perception-based (see
  ``mfw.physics.contact.verify_grasp`` with ``gripper_feedback=False``),
* the cache is refreshed from ``get_state`` at most every ``refresh_s`` so an
  estop or a calibration change on the Jetson is noticed without a round trip
  per query. Every reply to one of our own commands refreshes it immediately,
  so the cache is exact right after a motion, which is when it matters.

Two sockets, not one. ``follow_trajectory`` blocks the main REQ socket for a
whole chunk (``hardware.trajectory_chunk_s`` plus whatever the Jetson
stretches it to), and a REQ socket cannot send anything until that reply is
in, so a stop queued behind it would not be a stop. Every RemoteArm therefore
opens a second :class:`JetsonClient` to the same server that is used only for
``estop`` and ``get_state``; the server's ROUTER answers those inline while
its worker thread is still streaming the chunk (review P4). The Ctrl-C
handler in ``scripts/run_assistant.py`` and ``RemoteController.emergency_stop``
go through it. The second connection is opened with a short ping timeout
(:data:`_ESTOP_CONNECT_TIMEOUT_S`, at most 2 s) and **never blocks or fails
construction**: if it does not come up, a warning is logged and the stop
channel falls back to the main client, which is exactly the pre-review
behaviour. Honest limit: the assistant is single-threaded and a Python
signal handler runs on the main thread, so a software stop lands within one
chunk of the request, not mid-servo-frame; the mid-motion stop is the +6 V
switch (``docs/HARDWARE_BRIEF.md`` section 9).

The cache also carries the server's ``attached`` flag (the bridge stream's
keepalive work): when a refresh shows the servos went from attached to
detached without an estop from us -- the Uno's watchdog fired, serial was
lost, someone hit ``torque off`` -- a warning tells the operator that the arm
is limp and that the next motion re-attaches slowly from wherever the last
pulse left it.

That flag alone missed the common case (review, bridge lens): a detach during
an idle gap *and* its slow repair both happen inside the next motion call, so
no ``get_state`` ever saw the arm limp and the reply said ``detached: false``
while the held marker may have dropped. Every server reply now carries a
``detach_count`` (and motion replies ``reattached``); a count that grew
without an estop from this arm is an **unexpected detach**: it is logged as a
warning, emitted as a ``robot.bridge_detached`` event, queued for the
controller (:meth:`RemoteArm.pop_detach_notice`, which stops the motion) and
marks whatever the jaw holds as unverified (:attr:`RemoteArm.hold_unverified`)
until the jaw is opened.

Heartbeat. The server relaxes an arm nobody has spoken to for
``host_timeout_s``; with ``heartbeat_s`` set (the runtime sets it) a
:class:`~mfw.hardware.jetson_client.Heartbeat` keeps an idle-but-alive laptop
counted as alive and feeds the detach counter above into the cache between
commands. Only the status fields of a heartbeat reply are absorbed -- never
``q`` -- because a heartbeat answered mid-trajectory would otherwise overwrite
the pose the chunk reply had just delivered.

Refuses to build when ``hardware.arm.joint_lower/upper`` are wider than the
Jetson's servo pulse map can reach (review GEO-1 / HS-2). A 180-degree servo
mapped over 600..2400 us cannot span a 240-degree kinematic range whatever
its zero offset; the Jetson would clamp the pulse silently, answer
``clamped=false``, and the TCP would land up to 88 mm from the plan on a near
top-down grasp. :func:`assert_limits_reachable` compares the laptop limits
with the angle band each joint's ``JointCalibration`` can actually command and
names ``scripts/calibrate_servos.py`` in the refusal.

Trap recorded: the Jetson clamps targets to *its* calibrated pulse range, which
can be narrower than ``hardware.arm.joint_lower/upper``. When that happens the
reply's ``q`` differs from what was commanded; the controller logs it. Do not
"fix" the cache to the commanded value -- the arm is where the Jetson says.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mfw.config.schema import RobotConfig
from mfw.core.errors import ConfigurationError, ExecutionError
from mfw.core.interfaces import IRobot
from mfw.core.types import GripperState, JointState, Pose, RobotState
from mfw.hardware.jetson_client import Heartbeat, JetsonClient
from mfw.hardware.kinematics import PlanarKinematics
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["RemoteArm", "reachable_joint_range", "assert_limits_reachable"]

_log = get_logger("hardware.remote_arm")

_LIMIT_TOLERANCE_RAD = 1e-3
"""Slack when comparing ``hardware.arm`` limits with the pulse map's reach:
the YAML writes pi/2 as 1.5708 while the map's 90 deg converts to 1.570796."""
_ESTOP_CONNECT_TIMEOUT_S = 2.0
"""Ceiling on the ping that opens the dedicated stop channel. The main client
has already proven the server answers, so a second socket to the same
endpoint either connects at once or something is wrong with it specifically;
waiting the full ``request_timeout_s`` (5 s, three retries) would only delay
bring-up."""


def reachable_joint_range(joint: Mapping[str, Any]) -> tuple[float, float]:
    """Joint angles, radians, that one ``JointCalibration`` entry can command.

    The servo's own angle spans ``angle_at_min_deg..angle_at_max_deg`` over
    ``pulse_min_us..pulse_max_us``; the joint angle is ``direction *
    (servo_rad - zero_offset_rad)``. Anything outside the returned interval is
    clamped to the pulse range on the Jetson without ``clamped`` being set.
    ``joint`` is one entry of ``get_calibration()['calibration']['joints']``.
    """
    direction = int(joint.get("direction", 1))
    zero = float(joint.get("zero_offset_rad", 0.0))
    ends = [
        direction * (math.radians(float(joint[key])) - zero)
        for key in ("angle_at_min_deg", "angle_at_max_deg")
    ]
    return min(ends), max(ends)


def _enforced_limits(
    joint_names: Sequence[str], calibration: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """``(lower, upper)`` from a ``get_calibration()['calibration']`` dict, or ``None``."""
    joints = calibration.get("joints") if isinstance(calibration, Mapping) else None
    if not isinstance(joints, Mapping):
        return None
    lower, upper = [], []
    for name in joint_names:
        entry = joints.get(name)
        try:
            lower.append(float(entry["limit_lower_rad"]))
            upper.append(float(entry["limit_upper_rad"]))
        except (KeyError, TypeError, ValueError):
            return None
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def assert_limits_reachable(
    kinematics: PlanarKinematics, calibration: Mapping[str, Any]
) -> None:
    """Refuse ``hardware.arm`` limits the Jetson's pulse map cannot reach.

    ``calibration`` is the ``calibration`` dict of a ``get_calibration`` reply.
    Raises :class:`ConfigurationError` naming every offending joint, the band
    the map reaches and ``scripts/calibrate_servos.py``. Only *warns* when
    the Jetson's own ``limit_*_rad`` are narrower than ``hardware.arm``: that
    case is clamped in radians with ``clamped=true`` and the controller
    reports it, so it is a visible mismatch rather than a silent one.
    """
    joints = calibration.get("joints") if isinstance(calibration, Mapping) else None
    if not isinstance(joints, Mapping):
        raise ConfigurationError(
            "Jetson calibration carries no 'joints' table; inspect it with "
            "`py -3.12 scripts/calibrate_servos.py --jetson <ip>:5560` (read)"
        )
    problems: list[str] = []
    narrower: list[str] = []
    for index, name in enumerate(kinematics.joint_names):
        entry = joints.get(name)
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: no entry in the Jetson calibration")
            continue
        try:
            lo, hi = reachable_joint_range(entry)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{name}: unreadable pulse map ({exc})")
            continue
        arm_lo = float(kinematics.lower[index])
        arm_hi = float(kinematics.upper[index])
        if arm_lo < lo - _LIMIT_TOLERANCE_RAD or arm_hi > hi + _LIMIT_TOLERANCE_RAD:
            problems.append(
                f"{name}: hardware.arm limits [{math.degrees(arm_lo):.1f}, "
                f"{math.degrees(arm_hi):.1f}] deg exceed the pulse map's reach "
                f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg "
                f"({entry.get('pulse_min_us')}..{entry.get('pulse_max_us')} us)"
            )
            continue
        limit_lower, limit_upper = entry.get("limit_lower_rad"), entry.get("limit_upper_rad")
        if limit_lower is not None and limit_upper is not None and (
            float(limit_lower) > arm_lo + _LIMIT_TOLERANCE_RAD
            or float(limit_upper) < arm_hi - _LIMIT_TOLERANCE_RAD
        ):
            narrower.append(name)
    if problems:
        raise ConfigurationError(
            "hardware.arm.joint_lower/upper ask for angles the Jetson's servo pulse map "
            "cannot reach; the Jetson would clamp the pulse silently and the TCP would land "
            "centimetres from the plan (review GEO-1). "
            + "; ".join(problems)
            + ". Either tighten joint_lower/joint_upper in configs/hardware.yaml to the "
            "reachable band, or measure the servo end-stops and widen the map with "
            "`py -3.12 scripts/calibrate_servos.py --jetson <ip>:5560` (pulse / limit / save)."
        )
    if narrower:
        _log.warning(
            "Jetson limit_*_rad for %s are narrower than hardware.arm limits; targets past "
            "them are clamped and reported clamped=true (keep both tables equal after "
            "calibration)",
            narrower,
        )


class RemoteArm(IRobot):
    """Radians in, radians out; the Jetson owns pulses and calibration."""

    def __init__(
        self,
        client: JetsonClient,
        kinematics: PlanarKinematics,
        config: RobotConfig,
        clock: Any,
        refresh_s: float = 0.5,
        set_target_dt: float = 0.05,
        estop_client: JetsonClient | None = None,
        check_calibration: bool = True,
        event_logger: EventLogger | None = None,
        heartbeat_s: float | None = None,
    ) -> None:
        """Wrap a *connected* ``client``.

        ``estop_client`` is the dedicated stop/state channel; ``None`` opens a
        second connection to the same server (the normal case). Tests inject
        a stub. ``check_calibration=False`` skips the pulse-map check, for
        tools that connect before a calibration exists; the runtime never
        passes it. ``event_logger`` receives ``robot.bridge_detached``
        events; ``heartbeat_s`` starts a :class:`Heartbeat` on a third
        connection (``None``: none -- tests and one-shot tools).
        """
        config.validate()
        if tuple(config.arm_joint_names) != tuple(kinematics.joint_names):
            raise ExecutionError(
                "robot.arm_joint_names and the kinematics model disagree: "
                f"{list(config.arm_joint_names)} vs {list(kinematics.joint_names)}"
            )
        self._client = client
        self._kin = kinematics
        self._config = config
        self._clock = clock
        self._refresh_s = float(refresh_s)
        self._set_target_dt = float(set_target_dt)

        self._q: NDArray[np.float64] = np.asarray(config.home_joint_positions, dtype=np.float64)
        self._width: float = float(config.gripper_open_width)
        self._estopped = False
        self._attached: bool | None = None
        self._refreshed_at = -np.inf
        self._events = event_logger
        self._status_lock = threading.Lock()
        self._detach_seen: int | None = None
        self._own_detach_pending = False
        self._expect_reattach = True
        """The first motion re-attaches by design (boot); so does the first
        one after our own estop."""
        self._detach_notices: list[str] = []
        self._hold_unverified = False
        self._heartbeat: Heartbeat | None = None
        self.server_joint_limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        """The Jetson's ENFORCED joint limits (``limit_*_rad`` of the active
        calibration, e.g. the +-50 deg placeholder band), in ``joint_names``
        order; ``None`` until :meth:`_check_calibration` read them. Motions
        planned past them execute clamped and then stop, so a planner that
        chooses its own targets (the scan's sweep) stays inside them."""
        if client.last_state is not None:
            self._absorb(client.last_state)

        if check_calibration:
            self._check_calibration()
        self._estop_client: JetsonClient = (
            estop_client if estop_client is not None else self._open_estop_client()
        )
        self._estop_channel_dedicated = self._estop_client is not self._client
        if heartbeat_s is not None and heartbeat_s > 0:
            rpc = self._client.rpc
            self._heartbeat = Heartbeat(
                rpc.host,
                rpc.port,
                period_s=float(heartbeat_s),
                on_state=lambda state: self._absorb_status(state, source="heartbeat"),
                request_timeout_s=min(float(rpc.request_timeout_s), _ESTOP_CONNECT_TIMEOUT_S),
            ).start()

    # ------------------------------------------------------------------
    # bring-up checks

    def _check_calibration(self) -> None:
        try:
            reply = self._client.get_calibration()
        except ExecutionError as exc:
            raise ExecutionError(
                "could not fetch the Jetson calibration to check hardware.arm limits "
                f"against its pulse map: {exc}"
            ) from exc
        calibration = reply.get("calibration") if isinstance(reply, dict) else None
        assert_limits_reachable(self._kin, calibration or {})
        self.server_joint_limits = _enforced_limits(self._kin.joint_names, calibration)
        self._check_geometry(reply.get("arm_geometry") if isinstance(reply, dict) else None)

    _GEOMETRY_FIELDS = ("base_height", "shoulder_offset", "upper_arm", "forearm", "tool")

    def _check_geometry(self, server_geometry: Any) -> None:
        """Warn when the Jetson checks ``home_q`` against other link lengths.

        ``robot_server.py --arm-geometry`` backs the only guard on the pose it
        drives at every boot (home below the table), and it defaults to the
        placeholder links; once ``hardware.arm`` is measured the two must be
        passed together (re-review, motion lens). A warning, not a refusal:
        the geometry changes no command the Jetson executes.
        """
        if not isinstance(server_geometry, Mapping):
            return
        arm = getattr(self._kin, "arm", None)
        if arm is None:
            return
        diffs = []
        for name in self._GEOMETRY_FIELDS:
            if name not in server_geometry or not hasattr(arm, name):
                continue
            ours, theirs = float(getattr(arm, name)), float(server_geometry[name])
            if abs(ours - theirs) > 1e-4:
                diffs.append(f"{name} {theirs:.4f} (Jetson) vs {ours:.4f} (hardware.arm)")
        if diffs:
            _log.warning(
                "robot_server.py --arm-geometry disagrees with hardware.arm: %s. The Jetson's "
                "home-below-the-table check validates the boot pose against the wrong arm; restart "
                "it with --arm-geometry %s",
                "; ".join(diffs),
                ",".join(f"{float(getattr(arm, n)):g}" for n in self._GEOMETRY_FIELDS),
            )

    def _open_estop_client(self) -> JetsonClient:
        """Second connection for stop/state, or the main client if it will not open.

        Bounded: one ping with a timeout of at most
        :data:`_ESTOP_CONNECT_TIMEOUT_S`. Never raises -- a missing stop channel
        degrades to the single-socket behaviour with a logged warning; it must
        not stop the arm from being built. That includes errors raised
        outside the RPC's own error handling (a socket that cannot even be
        created when the process is out of file descriptors raises
        ``zmq.ZMQError``/``OSError``, not :class:`ExecutionError`), hence the
        broad catch.
        """
        rpc = self._client.rpc
        second: JetsonClient | None = None
        try:
            second = JetsonClient(
                rpc.host,
                rpc.port,
                request_timeout_s=min(float(rpc.request_timeout_s), _ESTOP_CONNECT_TIMEOUT_S),
                trajectory_timeout_margin_s=self._client.trajectory_timeout_margin_s,
            )
            second.connect(retries=1)
        except Exception as exc:  # noqa: BLE001 - see the docstring: never fail bring-up
            if second is not None:
                try:
                    second.close()
                except Exception as close_exc:  # noqa: BLE001
                    _log.debug("closing the failed estop channel raised: %s", close_exc)
            _log.warning(
                "could not open the dedicated estop channel to %s (%s); estop and "
                "get_state will share the main socket and a stop queued behind a "
                "trajectory chunk waits for that chunk",
                rpc.endpoint, exc,
            )
            return self._client
        _log.info("Dedicated estop/get_state channel open to %s", rpc.endpoint)
        return second

    # ------------------------------------------------------------------
    # cache

    def _absorb(self, state: dict[str, Any]) -> None:
        """Take the commanded pose out of a server reply."""
        q = state.get("q")
        if q is not None:
            arr = np.asarray(q, dtype=np.float64).reshape(-1)
            if arr.shape[0] == self._kin.dof:
                self._q = arr
        if "gripper_width" in state:
            self._width = float(state["gripper_width"])
        self._absorb_status(state, source="reply")
        self._refreshed_at = time.monotonic()

    def _absorb_status(self, state: dict[str, Any], source: str) -> None:
        """Estop / attach / detach-counter fields of a reply; never the pose.

        Runs on the heartbeat thread too, hence the lock.
        """
        if not isinstance(state, dict):
            return
        unexpected: str | None = None
        count_now: int | None = None
        with self._status_lock:
            if "estopped" in state:
                self._estopped = bool(state["estopped"])
            if "attached" in state:
                attached = bool(state["attached"])
                if (
                    self._attached is True
                    and not attached
                    and not self._estopped
                    and state.get("detach_count") is None
                ):
                    # Older servers carry no counter: keep the old warning.
                    _log.warning(
                        "Jetson reports the servos DETACHED without an estop from us (Uno "
                        "watchdog, serial loss or torque off): the arm is limp and may have "
                        "sagged; the next motion re-attaches slowly from the last pulsed pose. "
                        "Check the arm before commanding it."
                    )
                self._attached = attached
            raw_count = state.get("detach_count")
            if raw_count is not None:
                count_now = int(raw_count)
                seen = self._detach_seen
                if seen is not None and count_now > seen:
                    if self._own_detach_pending:
                        self._own_detach_pending = False
                    else:
                        unexpected = str(state.get("last_detach_reason") or "no reason reported")
                if seen is None or count_now > seen:
                    self._detach_seen = count_now
            if state.get("reattached"):
                if self._expect_reattach:
                    self._expect_reattach = False
                elif raw_count is None and unexpected is None:
                    unexpected = "the server re-attached limp servos before this motion"
            if unexpected is not None:
                self._hold_unverified = True
                self._detach_notices.append(unexpected)
        if unexpected is not None:
            _log.warning(
                "The Jetson DETACHED the servos without a request from this laptop (%s). The arm "
                "went limp -- it may have sagged and dropped what it held -- and the next motion "
                "re-energises it slowly. Anything in the jaw is UNVERIFIED: look at the gripper.",
                unexpected,
            )
            if self._events is not None:
                self._events.emit(
                    "robot.bridge_detached",
                    {"reason": unexpected, "detach_count": count_now, "source": source},
                )

    def _maybe_refresh(self) -> None:
        if time.monotonic() - self._refreshed_at >= self._refresh_s:
            self._absorb(self._estop_client.get_state())

    def refresh(self) -> None:
        """Force a ``get_state`` round trip now (on the stop/state channel)."""
        self._absorb(self._estop_client.get_state())

    @property
    def estopped(self) -> bool:
        """Whether the server last reported itself estopped (cached)."""
        return self._estopped

    @property
    def attached(self) -> bool | None:
        """Whether the servos were being pulsed at the last refresh (cached).

        ``None`` until a reply that carries the field has been seen (older
        servers do not send it).
        """
        return self._attached

    @property
    def hold_unverified(self) -> bool:
        """True after an unexpected detach, until :meth:`clear_hold_unverified`.

        A limp arm may have dropped what the jaw held; nothing on this arm
        can tell, so a held object must be treated as unverified (look, or
        re-verify by perception) rather than as carried.
        """
        with self._status_lock:
            return self._hold_unverified

    def clear_hold_unverified(self) -> None:
        """Forget the unverified-hold mark (the jaw was opened, or a person looked)."""
        with self._status_lock:
            self._hold_unverified = False

    def pop_detach_notice(self) -> str | None:
        """Unexpected detaches seen since the last call, joined, or ``None``.

        The controller calls this before and after every chunk and stops the
        motion when it returns something.
        """
        with self._status_lock:
            if not self._detach_notices:
                return None
            notice = "; ".join(self._detach_notices)
            self._detach_notices.clear()
            return notice

    @property
    def heartbeat(self) -> Heartbeat | None:
        """The running heartbeat, if ``heartbeat_s`` was given."""
        return self._heartbeat

    @property
    def has_dedicated_estop_channel(self) -> bool:
        """False when the stop channel fell back to the main client at construction."""
        return self._estop_channel_dedicated

    @property
    def kinematics(self) -> PlanarKinematics:
        """The closed-form model this arm is described by."""
        return self._kin

    @property
    def client(self) -> JetsonClient:
        """The transport, for calibration tools that need raw endpoints."""
        return self._client

    @property
    def estop_client(self) -> JetsonClient:
        """The second connection, reserved for ``estop`` and ``get_state``."""
        return self._estop_client

    def close(self) -> None:
        """Stop the heartbeat and close the stop/state channel. The main client
        belongs to whoever built it."""
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._estop_client is not self._client:
            self._estop_client.close()

    # ------------------------------------------------------------------
    # IRobot: identity and state

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._config.arm_joint_names)

    @property
    def config(self) -> RobotConfig:
        """The robot section of the framework config (read by planners)."""
        return self._config

    @property
    def articulation(self) -> Any:
        """Always raises: there is no Isaac articulation on this lane."""
        raise ExecutionError(
            "RemoteArm has no articulation; the hardware lane replaces every reader of it"
        )

    def get_arm_joint_positions(self) -> NDArray[np.float64]:
        """Commanded joint vector (radians, wire order). Copy; safe to mutate."""
        self._maybe_refresh()
        return self._q.copy()

    def get_state(self) -> RobotState:
        """Snapshot built like the sim robot's, with zero velocities and clock time."""
        self._maybe_refresh()
        joint_state = JointState(
            positions=self._q.copy(),
            velocities=np.zeros(self._kin.dof),
            names=self.joint_names,
        )
        return RobotState(
            joint_state=joint_state,
            tcp_pose=self.tcp_pose(),
            gripper=self.get_gripper_state(),
            sim_time=float(self._clock.sim_time),
            step_index=int(self._clock.step_index),
        )

    def tcp_pose(self) -> Pose:
        """FK of the commanded configuration: jaw midpoint, never a finger."""
        self._maybe_refresh()
        return self._kin.fk(self._q)

    # ------------------------------------------------------------------
    # IRobot: kinematics

    def forward_kinematics(self, joint_positions: ArrayLike) -> Pose:
        return self._kin.fk(joint_positions)

    def inverse_kinematics(
        self, target: Pose, seed: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64] | None:
        return self._kin.ik(target, seed=seed)

    def both_branches(self, target: Pose) -> list[NDArray[np.float64]]:
        """Every in-limit IK solution, preferred first (see ``PlanarKinematics``)."""
        return self._kin.both_branches(target)

    def within_limits(self, joint_positions: ArrayLike) -> bool:
        """Whether ``q`` respects ``hardware.arm.joint_lower/upper``."""
        return self._kin.within_limits(joint_positions)

    # ------------------------------------------------------------------
    # IRobot: motion

    def follow_trajectory(self, waypoints: ArrayLike, dt: float) -> dict[str, Any]:
        """Ship waypoints to the Jetson and block; returns the server reply.

        Raises :class:`ExecutionError` (via the client) when the server refuses
        (estopped, busy, unreachable). A reply with ``estopped: true`` is
        returned, not raised: the controller decides how to report it.
        """
        reply = self._client.follow_trajectory(waypoints, dt)
        self._absorb(reply)
        return reply

    def set_arm_joint_targets(self, positions: ArrayLike) -> None:
        """Command one configuration: a single-waypoint trajectory.

        The server writes it immediately and its Uno-side slew limit smooths
        the step, which is what a bare position target means on this arm.
        """
        q = np.asarray(positions, dtype=np.float64).reshape(-1)
        if q.shape[0] != self._kin.dof:
            raise ExecutionError(f"expected {self._kin.dof} arm targets, got {q.shape[0]}")
        self.follow_trajectory(q.reshape(1, -1), self._set_target_dt)

    def go_home_immediate(self) -> None:
        """Move to the Jetson's calibrated home. Blocks for the move.

        Not a teleport: servos cannot teleport, so ``home`` is a real motion at
        the server's velocity ceiling. Safe at bring-up; never call it while
        holding an object. At bring-up after a bridge reset (every server
        start on the Uno) the servos attach AT home instantly -- the reply
        says ``first_attach`` and a warning repeats the hand-pose rule. This
        method does not ask anyone: :class:`~mfw.hardware.runtime.HardwareRuntime`
        runs its first-home gate (operator confirmation) before calling it.
        """
        reply = self._client.home()
        self._absorb(reply)
        if reply.get("first_attach"):
            _log.warning(
                "The servo bridge had no known position (it resets whenever the Jetson opens its "
                "port): the servos attached AT home at full speed. If the arm was not hand-posed "
                "at home it jumped there -- hand-pose it right before this first home (run_assistant "
                "--hardware asks for that at its prompt, or --home-confirmed asserts it), after every "
                "robot_server start."
            )
        home = np.asarray(self._config.home_joint_positions, dtype=np.float64)
        error = float(np.max(np.abs(self._q - home)))
        if error > 0.05:
            _log.warning(
                "Jetson home %s differs from robot.home_joint_positions %s by %.3f rad; "
                "the two calibrations disagree",
                np.round(self._q, 3).tolist(), np.round(home, 3).tolist(), error,
            )

    def estop(self) -> None:
        """Detach every servo and refuse motion until :meth:`clear_estop`.

        Sent on the dedicated channel, so it goes out even while the main
        client is blocked inside :meth:`follow_trajectory`. If that channel
        fails (its socket timed out, the Jetson restarted), the stop is retried
        once on the main client before the error propagates: a stop must not
        be lost to the channel meant to make it faster.
        """
        with self._status_lock:
            self._own_detach_pending = True
            self._expect_reattach = True
        try:
            try:
                self._absorb(self._estop_client.estop())
                return
            except ExecutionError as exc:
                if self._estop_client is self._client:
                    raise
                _log.error("estop on the dedicated channel failed (%s); retrying on the main client", exc)
            self._absorb(self._client.estop())
        finally:
            with self._status_lock:
                self._own_detach_pending = False

    def clear_estop(self) -> None:
        """Allow motion again after :meth:`estop`."""
        self._absorb(self._client.clear_estop())

    # ------------------------------------------------------------------
    # IRobot: gripper

    def open_gripper(self) -> None:
        """Open fully. Nothing is held afterwards, so an unverified hold is cleared."""
        self._absorb(self._client.set_gripper(self._config.gripper_open_width))
        self.clear_hold_unverified()

    def close_gripper(self) -> None:
        """Close to ``robot.gripper_closed_width`` (an empty close)."""
        self.close_gripper_to(self._config.gripper_closed_width)

    def close_gripper_to(self, width_m: float) -> None:
        """Command the jaw to ``width_m``, clamped to ``[closed, open]``.

        This is how a grasp is closed on this arm: the MG90S has no torque
        limit, so a close to 0 mm on a 19 mm marker stalls it at full current
        for the whole carry. The controller passes the planned grasp width
        minus a small margin instead.
        """
        lo = float(self._config.gripper_closed_width)
        hi = float(self._config.gripper_open_width)
        target = min(max(float(width_m), lo), hi)
        self._absorb(self._client.set_gripper(target))

    def get_gripper_width(self) -> float:
        """Commanded jaw opening in metres. There is no measured width."""
        self._maybe_refresh()
        return float(self._width)

    def get_gripper_state(self) -> GripperState:
        """Commanded width with ``is_grasping=False`` -- no sensor can say otherwise."""
        self._maybe_refresh()
        return GripperState(
            width=float(self._width),
            target_width=float(self._width),
            is_moving=False,
            is_grasping=False,
        )

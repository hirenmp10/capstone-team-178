"""Trajectory execution for the remote hobby arm.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers.

Subclasses :class:`~mfw.controllers.joint_controller.JointTrajectoryController`
so the constructor, the workspace guard and the gripper/hold/stop surface the
skills rely on stay byte-identical; only the parts that assumed a simulator
are overridden.

What changes and why
--------------------
* **No tracking supervision.** The base class compares measured joint
  positions with the path every step to detect a blocked arm. PWM servos
  report nothing, so there is nothing to compare; the only failure the Jetson
  can report is that it stopped early (estop), clamped a target to its
  calibrated range or pulse map, or lost the servos (``detached``). All three
  stop the motion: a clamped servo put the TCP somewhere the table and
  obstacle checks never saw (review GEO-1/HS-2), so nothing more is shipped.
* **Chunked shipping.** The trajectory is resampled onto the controller grid
  as before, then sent in chunks of ``hardware.trajectory_chunk_s``. Between
  chunks ``on_step`` runs (so ``Pick.hold_grip`` still re-asserts the close)
  and ``_stop_requested`` is honoured. One round trip per chunk is what keeps
  Wi-Fi jitter out of the servo timing: interpolation happens on the Jetson
  and the Uno, never across the LAN.
* **Workspace check up front.** Every waypoint is checked before the first
  byte is shipped. In sim a violation mid-trajectory merely stops the arm; on
  metal the arm has already swept somewhere it should not have been.
* **Home is exempt from the workspace box.** The box says where the TCP may
  *operate*; the tucked rest posture lives outside it by design (folded back,
  jaws up, clear of the camera's view of the table). ``go_home`` would
  otherwise raise ``SafetyViolation`` on every run.
* **Settling is time.** ``_settle_at`` waits ``hardware.settle_steps_after_motion``
  clock steps for the PLA links to stop ringing; the arm cannot report arrival.
* **Emergency stop never touches ``articulation``.** It detaches the servos on
  the Jetson (``estop``), which is the only immediate stop this arm has. The
  arm goes limp and may drop what it holds; that is documented and padded for.
  The request goes out on ``RemoteArm``'s dedicated stop channel, so it is
  answered even while the main socket is blocked inside a chunk (review P4);
  but the assistant is single-threaded, so a *spoken or typed* stop still only
  runs between skills. Mid-motion, the +6 V switch is the stop (review HS-4).
* **The jaw closes to the planned grasp width, not to zero.** The MG90S has no
  torque limit and no feedback: ``close_gripper()`` on a 19 mm marker asks for
  0 mm and stalls the servo at full current for the whole carry, re-stalled by
  every ``maintain_grasp`` call (review, missed item 1). The grasp skill hands
  the planned chord width to :meth:`RemoteController.set_grasp_width` before
  closing; :meth:`close_gripper_blocking` then closes to that width minus
  :data:`GRASP_CLOSE_MARGIN_M`, floored at ``robot.gripper_closed_width``, and
  :meth:`maintain_grasp` re-asserts the *same* width. ``open_gripper_blocking``
  forgets the width, so a close with no grasp planned is still an empty close.
* **The squeeze margin is a measure-to-tune number.** On the shipped gripper
  map (1000..1900 us over 45 mm) 4 mm is 80 us, about 8 degrees of servo
  error -- at or past an analog MG90S's proportional band, so the servo may
  sit near its ~0.7 A stall current for the whole carry, and the Jetson's
  keepalive now holds that for as long as the laptop is alive (review,
  missed-1). Measure the gripper current holding a marker on the bench and
  tune it, likely to 1-2 mm: pass ``close_margin_m`` or set
  ``hardware.grasp_close_margin_m`` once the schema carries it (read here
  with ``getattr`` so the field can land without touching this file).
* **A detach the laptop did not ask for stops the motion.** ``RemoteArm``
  queues every unexpected bridge detach (the Uno watchdog, serial loss, the
  server's host timeout, a ``torque off`` from another tool) -- including one
  that was repaired inside the chunk that reported it, which used to be
  invisible. The controller checks the queue before and after every chunk
  and stops with a ``controller.bridge_detached`` event: the arm went limp,
  anything held may have dropped, and ``RemoteArm.hold_unverified`` says so
  until the jaw is opened.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import HardwareConfig, MotionConfig, RobotConfig
from mfw.controllers.joint_controller import JointTrajectoryController
from mfw.core.errors import ExecutionError
from mfw.core.types import Pose, Trajectory
from mfw.motion.trajectory import resample
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["RemoteController"]

_log = get_logger("hardware.controller")

_HOME_EXEMPT_TOL_RAD = 1e-3

GRASP_CLOSE_MARGIN_M = 0.004
"""Default metres the jaw closes past the planned grasp width so the fingers
squeeze rather than merely touch: enough to take up PLA linkage slack and the
~1 mm pulse quantisation. NOT measured: whether the MG90S then sits near
stall current is exactly what the bench check in the module docstring must
answer (``close_margin_m`` / ``hardware.grasp_close_margin_m`` override it)."""


class RemoteController(JointTrajectoryController):
    """Ships trajectories to the Jetson in chunks; keeps the base class's guards."""

    def __init__(
        self,
        sim: Any,
        robot: Any,
        motion_config: MotionConfig,
        robot_config: RobotConfig,
        hardware_config: HardwareConfig,
        workspace_min: NDArray[np.float64] | None = None,
        workspace_max: NDArray[np.float64] | None = None,
        event_logger: EventLogger | None = None,
        close_margin_m: float | None = None,
    ) -> None:
        super().__init__(
            sim=sim,
            robot=robot,
            motion_config=motion_config,
            robot_config=robot_config,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            event_logger=event_logger,
        )
        hardware_config.validate()
        self.hardware_config = hardware_config
        self._home = np.asarray(robot_config.home_joint_positions, dtype=np.float64)
        self.chunks_shipped = 0
        """Total chunks sent since construction; tests and the event log read it."""
        self._grasp_width: float | None = None
        margin = close_margin_m
        if margin is None:
            margin = getattr(hardware_config, "grasp_close_margin_m", GRASP_CLOSE_MARGIN_M)
        margin = float(margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ExecutionError(f"grasp close margin must be a finite number >= 0 m, got {margin!r}")
        self.close_margin_m = margin
        """Metres closed past the planned grasp width (see :data:`GRASP_CLOSE_MARGIN_M`)."""
        self.last_detach_notice: str | None = None
        """The unexpected bridge detach that stopped the last motion, if any
        (cleared when the next motion starts)."""

    # ------------------------------------------------------------------
    # grid

    def _effective_dt(self) -> float:
        steps = max(
            1, int(round(self.motion_config.interpolation_dt / self._sim.config.physics_dt))
        )
        return steps * float(self._sim.config.physics_dt)

    def _chunk_length(self, dt: float) -> int:
        return max(2, int(round(self.hardware_config.trajectory_chunk_s / dt)))

    # ------------------------------------------------------------------
    # IController

    def follow_trajectory(
        self,
        trajectory: Trajectory,
        on_step: Callable[[int, NDArray[np.float64]], bool] | None = None,
    ) -> bool:
        """Resample, guard every waypoint, then ship in chunks.

        ``on_step(index, q)`` is called after every chunk with the index of the
        last waypoint shipped and the Jetson's reported configuration;
        returning ``False`` aborts. Returns ``False`` on abort, estop, a
        server-side early stop, a clamped target, a bridge detach, or an
        unexpected detach reported by the arm before or after any chunk
        (see the module docstring). Raises ``SafetyViolation`` before any
        motion if a waypoint leaves the workspace.
        """
        if len(trajectory) == 0:
            return True

        self._stop_requested = False
        # A notice describes THIS motion's stop; a stale one from an earlier
        # motion would make every later failure read as "the arm went limp".
        self.last_detach_notice = None
        dt = self._effective_dt()
        dense = resample(trajectory, dt)
        positions = np.stack([w.positions for w in dense.waypoints])
        last = positions.shape[0] - 1

        # Guard first, ship second. A violation raised here costs nothing.
        for index, q in enumerate(positions):
            self._assert_within_workspace(
                q, margin=0.0 if index == last else self.motion_config.workspace_transit_margin
            )

        started = time.perf_counter()
        chunk = self._chunk_length(dt)
        start = 0
        while start <= last:
            if self._stop_requested:
                self._emit_execution("aborted", trajectory, start, 0.0, started)
                return False
            if self._stop_for_detach(trajectory, start, started, before_chunk=True):
                return False

            stop = min(last + 1, start + chunk)
            try:
                reply = self._robot.follow_trajectory(positions[start:stop], dt)
            except ExecutionError as exc:
                _log.error("Jetson rejected the trajectory chunk at %d: %s", start, exc)
                self._emit_execution("rejected", trajectory, start, 0.0, started)
                return False
            self.chunks_shipped += 1
            shipped_to = stop - 1
            reported = np.asarray(reply.get("q", positions[shipped_to]), dtype=np.float64)
            if self._stop_for_detach(trajectory, shipped_to, started, before_chunk=False):
                return False

            deviation = float(np.max(np.abs(reported - positions[shipped_to])))
            if reply.get("clamped"):
                # Review GEO-1/HS-2: a clamp (radian limit or pulse map) means a
                # servo could not reach the plan, so the TCP is somewhere the
                # table/obstacle checks never saw. The chunk has already run;
                # the conservative move is to ship nothing more.
                _log.error(
                    "Jetson clamped a target in chunk ending at waypoint %d: a servo "
                    "cannot reach the planned pose (its calibrated range or pulse map "
                    "is narrower than hardware.arm limits). Stopping this motion; "
                    "recalibrate with calibrate_servos.py.",
                    shipped_to,
                )
                self._emit_execution("clamped", trajectory, shipped_to, deviation, started)
                return False
            if reply.get("detached"):
                _log.error(
                    "The bridge detached the servos during the chunk ending at waypoint %d "
                    "(Uno watchdog or serial loss): the arm is limp. Stopping this motion.",
                    shipped_to,
                )
                self._emit_execution("detached", trajectory, shipped_to, deviation, started)
                return False
            if deviation > self.robot_config.joint_position_tolerance:
                _log.warning(
                    "Jetson reports %s after waypoint %d, commanded %s (%.3f rad apart)",
                    np.round(reported, 3).tolist(), shipped_to,
                    np.round(positions[shipped_to], 3).tolist(), deviation,
                )

            if reply.get("estopped"):
                _log.warning("Trajectory stopped by estop at waypoint %d", shipped_to)
                self._emit_execution("estopped", trajectory, shipped_to, deviation, started)
                return False
            if not reply.get("completed", False):
                self._emit_execution("incomplete", trajectory, shipped_to, deviation, started)
                return False

            if on_step is not None and not on_step(shipped_to, reported):
                self.hold_position()
                self._emit_execution("callback_abort", trajectory, shipped_to, deviation, started)
                return False
            start = stop

        self._settle_at(positions[last])
        self._emit_execution("completed", trajectory, len(dense), 0.0, started)
        return True

    def _stop_for_detach(
        self, trajectory: Trajectory, index: int, started: float, before_chunk: bool
    ) -> bool:
        """True (and reported) when the arm queued an unexpected detach."""
        pop = getattr(self._robot, "pop_detach_notice", None)
        notice = pop() if callable(pop) else None
        if not notice:
            return False
        self.last_detach_notice = str(notice)
        when = "before shipping" if before_chunk else "during"
        _log.error(
            "The servo bridge detached without a request from this laptop (%s), noticed %s the "
            "chunk at waypoint %d: the arm went limp and anything held may have dropped. "
            "Stopping this motion; look at the arm before commanding it.",
            notice, when, index,
        )
        self._emit(
            "controller.bridge_detached",
            {"reason": str(notice), "waypoint": int(index), "before_chunk": bool(before_chunk)},
        )
        self._emit_execution("detached", trajectory, index, 0.0, started)
        return True

    @property
    def hold_unverified(self) -> bool:
        """Whether the arm reports that a held object is unverified (a limp spell)."""
        return bool(getattr(self._robot, "hold_unverified", False))

    def servo_to_pose(self, target: Pose) -> bool:
        """One IK step toward ``target`` shipped as a single waypoint."""
        current = self._robot.get_arm_joint_positions()
        solution = self._robot.inverse_kinematics(target, seed=current)
        if solution is None:
            return False
        self._assert_within_workspace(solution)
        self._robot.set_arm_joint_targets(solution)
        self._sim.step(
            max(1, int(round(self.motion_config.interpolation_dt / self._sim.config.physics_dt)))
        )
        return True

    def emergency_stop(self) -> None:
        """Detach the servos on the Jetson. Never reaches for an articulation.

        The arm goes limp: with no brakes, cutting the pulses is the only
        instantaneous stop these servos offer. Whatever is held will drop.
        Sent on the arm's dedicated stop channel (see ``RemoteArm.estop``), so
        it is delivered even if the main socket is mid-chunk; the caller,
        however, is on the same thread as ``follow_trajectory``, so from the
        assistant it can only run between skills.
        """
        self._stop_requested = True
        try:
            self._robot.estop()
        except ExecutionError as exc:
            _log.error("estop request failed: %s", exc)
        _log.warning("Emergency stop engaged: servos detached")
        self._emit("controller.emergency_stop", {})

    def stop(self) -> None:
        """Cooperative abort between chunks; the current chunk finishes on the Jetson."""
        self._stop_requested = True

    def hold_position(self) -> None:
        """No-op: the Jetson keeps pulsing the last target until told otherwise."""
        return None

    # ------------------------------------------------------------------
    # gripper

    def set_grasp_width(self, width_m: float) -> None:
        """Remember the planned grasp width for the next close.

        Called by the grasp skill with the candidate's chord width (the extent
        the fingers actually meet). Until :meth:`open_gripper_blocking` clears
        it, :meth:`close_gripper_blocking` and :meth:`maintain_grasp` close to
        ``max(width_m - close_margin_m, robot.gripper_closed_width)``.
        Non-finite or negative widths are ignored (an empty close follows).
        """
        value = float(width_m)
        if not np.isfinite(value) or value < 0.0:
            _log.warning("ignoring grasp width %r; the next close will be an empty close", width_m)
            self._grasp_width = None
            return
        self._grasp_width = value

    @property
    def grasp_width(self) -> float | None:
        """The planned grasp width set by :meth:`set_grasp_width`, or ``None``."""
        return self._grasp_width

    def close_target_width(self) -> float:
        """Width the next close commands: planned width minus the margin, floored at closed."""
        closed = float(self.robot_config.gripper_closed_width)
        if self._grasp_width is None:
            return closed
        return max(self._grasp_width - self.close_margin_m, closed)

    def _close_jaw(self) -> None:
        """Close to the planned width when one is known, else an empty close."""
        if self._grasp_width is None or not hasattr(self._robot, "close_gripper_to"):
            self._robot.close_gripper()
            return
        self._robot.close_gripper_to(self.close_target_width())

    def open_gripper_blocking(self, settle_steps: int | None = None) -> float:
        """Command open, forget the grasp width, wait the settle time, return the width."""
        self._grasp_width = None
        self._robot.open_gripper()
        self._sim.step(self._settle_steps(settle_steps))
        return float(self._robot.get_gripper_width())

    def close_gripper_blocking(
        self, settle_steps: int | None = None, stall_tolerance: float = 1e-4
    ) -> float:
        """Close to the planned grasp width (or fully), settle, return the commanded width.

        There is no stall to detect: the reported width is what was asked for.
        ``stall_tolerance`` is accepted for signature compatibility only.
        """
        self._close_jaw()
        self._sim.step(self._settle_steps(settle_steps))
        return float(self._robot.get_gripper_width())

    def maintain_grasp(self) -> None:
        """Re-assert the close at the *same* width; never tighten to zero mid-carry."""
        self._close_jaw()

    def _settle_steps(self, requested: int | None) -> int:
        configured = int(self.hardware_config.settle_steps_after_motion)
        if requested is None:
            return configured
        return max(configured, int(requested)) if configured > 0 else int(requested)

    # ------------------------------------------------------------------
    # internals

    def _settle_at(self, target: NDArray[np.float64], max_steps: int = 90) -> None:
        """Wait for the servos to physically arrive; nothing can confirm it."""
        self._sim.step(int(self.hardware_config.settle_steps_after_motion))

    def _assert_within_workspace(
        self, joint_positions: NDArray[np.float64], margin: float = 0.0
    ) -> None:
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape == self._home.shape and np.allclose(q, self._home, atol=_HOME_EXEMPT_TOL_RAD):
            return
        super()._assert_within_workspace(q, margin=margin)

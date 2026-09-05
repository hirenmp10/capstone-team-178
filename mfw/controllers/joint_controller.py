"""Trajectory execution.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

This is the **only** layer permitted to write joint targets, and nothing in the
framework writes an object's pose. Everything an object does is a consequence of
contact forces from the gripper.

Execution is closed-loop against tracking error: if the arm falls behind its
commanded configuration beyond ``tracking_error_limit``, the trajectory aborts.
That check is what turns "the arm hit something" into a reported failure the
state machine can recover from, instead of the arm grinding into an obstacle
while the log claims progress.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import MotionConfig, RobotConfig
from mfw.core.errors import ExecutionError, SafetyViolation
from mfw.core.interfaces import IController
from mfw.core.types import Pose, Trajectory
from mfw.motion.trajectory import resample
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["JointTrajectoryController"]

_log = get_logger("controllers.joint")


class JointTrajectoryController(IController):
    """Executes joint trajectories with tracking supervision."""

    def __init__(
        self,
        sim: Any,
        robot: Any,
        motion_config: MotionConfig,
        robot_config: RobotConfig,
        workspace_min: NDArray[np.float64] | None = None,
        workspace_max: NDArray[np.float64] | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.motion_config = motion_config
        self.robot_config = robot_config
        self._sim = sim
        self._robot = robot
        self._events = event_logger
        self._stop_requested = False
        self._workspace_min = (
            None if workspace_min is None else np.asarray(workspace_min, dtype=np.float64)
        )
        self._workspace_max = (
            None if workspace_max is None else np.asarray(workspace_max, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # IController
    # ------------------------------------------------------------------

    def follow_trajectory(
        self,
        trajectory: Trajectory,
        on_step: Callable[[int, NDArray[np.float64]], bool] | None = None,
    ) -> bool:
        """Execute a trajectory to completion.

        ``on_step`` is called with ``(index, current_joint_positions)`` after each
        control step; returning ``False`` aborts. This is how the pick skill
        watches contact forces while the gripper is closing without the
        controller needing to know what a grasp is.

        Returns ``False`` if aborted or if tracking error exceeded the limit.
        """
        if len(trajectory) == 0:
            return True

        self._stop_requested = False

        # Resample onto a grid that is an *exact* multiple of the physics step.
        #
        # If the waypoint spacing and the number of physics steps per waypoint
        # disagree, the commanded target advances faster than simulated time and
        # the error accumulates over the whole trajectory rather than staying
        # bounded. With interpolation_dt=0.02 and physics_dt=1/120 the mismatch is
        # 20%, which on a 2 s motion builds ~0.4 rad of lag -- far past the
        # tracking limit -- so every long reconfiguration aborted partway.
        steps_per_waypoint = max(
            1, int(round(self.motion_config.interpolation_dt / self._sim.config.physics_dt))
        )
        effective_dt = steps_per_waypoint * self._sim.config.physics_dt
        dense = resample(trajectory, effective_dt)

        started = time.perf_counter()
        worst_deviation = 0.0
        consecutive_violations = 0
        progress = 0
        steps_without_progress = 0

        positions = np.stack([w.positions for w in dense.waypoints])

        for index, waypoint in enumerate(dense.waypoints):
            if self._stop_requested:
                self._emit_execution("aborted", trajectory, index, worst_deviation, started)
                return False

            # Goals strictly; transit points with slack. See
            # MotionConfig.workspace_transit_margin.
            self._assert_within_workspace(
                waypoint.positions,
                margin=(
                    0.0
                    if index == len(dense.waypoints) - 1
                    else self.motion_config.workspace_transit_margin
                ),
            )
            self._robot.set_arm_joint_targets(waypoint.positions)
            # Rendering policy is left to the simulation context, which decides
            # from the headless flag. Forcing it off here would make the arm
            # appear frozen in the GUI for the entire duration of every motion --
            # the viewport would only update once the trajectory finished.
            self._sim.step(steps_per_waypoint)

            actual = self._robot.get_arm_joint_positions()

            # Measure deviation from the *path*, not from the time-indexed target.
            #
            # A position-controlled arm always trails its moving setpoint: measured
            # here, healthy free-space motion at 1 rad/s lags 0.176 rad, which
            # would trip any limit tight enough to catch a real collision. But that
            # lag is *along* the path -- the arm is on the right curve, just behind
            # schedule, and it catches up when the trajectory ends.
            #
            # A blocked or deflected arm is different: it leaves the path. Matching
            # against the nearest upcoming waypoint separates the two, so the limit
            # can be tight without firing on normal motion.
            previous_progress = progress
            progress, deviation = self._closest_on_path(positions, actual, progress)
            worst_deviation = max(worst_deviation, deviation)

            if progress > previous_progress:
                steps_without_progress = 0
            else:
                steps_without_progress += 1

            if deviation > self.motion_config.tracking_error_limit:
                consecutive_violations += 1
                if consecutive_violations >= self.motion_config.tracking_violation_steps:
                    _log.warning(
                        "Path deviation %.3f rad stayed above %.3f for %d steps at waypoint %d/%d",
                        deviation,
                        self.motion_config.tracking_error_limit,
                        consecutive_violations,
                        index,
                        len(dense),
                    )
                    self.hold_position()
                    self._emit_execution(
                        "path_deviation", trajectory, index, worst_deviation, started
                    )
                    return False
            else:
                consecutive_violations = 0

            # A stuck arm stays *on* the path but stops advancing along it, so
            # deviation alone would never notice.
            if steps_without_progress >= self.motion_config.stall_abort_steps:
                _log.warning(
                    "Arm made no progress along the path for %d steps at waypoint %d/%d",
                    steps_without_progress,
                    index,
                    len(dense),
                )
                self.hold_position()
                self._emit_execution("stalled", trajectory, index, worst_deviation, started)
                return False

            if on_step is not None and not on_step(index, actual):
                self.hold_position()
                self._emit_execution("callback_abort", trajectory, index, worst_deviation, started)
                return False

        # Settle onto the final target: the last commanded position is a target,
        # not an achieved state, and a grasp planned to millimetres cannot start
        # from "nearly there". This is also where the benign along-path lag is
        # absorbed.
        self._settle_at(dense.waypoints[-1].positions)
        self._emit_execution("completed", trajectory, len(dense), worst_deviation, started)
        return True

    @staticmethod
    def _closest_on_path(
        positions: NDArray[np.float64], actual: NDArray[np.float64], from_index: int
    ) -> tuple[int, float]:
        """Nearest upcoming waypoint to ``actual``, and the distance to it.

        Searches forward only, from the last matched index: the arm cannot move
        backwards along its own trajectory, and a bounded forward window keeps
        this O(1) per control step rather than O(n).
        """
        window = positions[from_index : from_index + 40]
        if window.shape[0] == 0:
            return from_index, 0.0
        distances = np.max(np.abs(window - actual), axis=1)
        offset = int(np.argmin(distances))
        return from_index + offset, float(distances[offset])

    def servo_to_pose(self, target: Pose) -> bool:
        """Single closed-loop Cartesian step toward ``target``.

        The reactive path used by the GR00T backend, which emits relative
        end-effector deltas rather than joint trajectories.
        """
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

    def hold_position(self) -> None:
        """Command the current configuration, so the arm resists gravity in place.

        Not a no-op: leaving stale targets means the drives keep pulling toward
        wherever the arm was last told to go.
        """
        self._robot.set_arm_joint_targets(self._robot.get_arm_joint_positions())

    def stop(self) -> None:
        """Request an abort at the next control step, keeping the arm controlled.

        Cooperative rather than immediate: cutting drive targets mid-motion would
        let the arm fall under gravity, dropping whatever it holds.
        """
        self._stop_requested = True
        self.hold_position()

    def emergency_stop(self) -> None:
        """Halt immediately and zero velocities.

        Distinct from :meth:`stop`: this does not wait for a control step. It
        still holds position rather than releasing, because dropping a held
        object is itself a hazard.
        """
        self._stop_requested = True
        self.hold_position()
        articulation = self._robot.articulation
        articulation.set_joint_velocities(
            np.zeros_like(np.asarray(articulation.get_joint_velocities(), dtype=np.float64))
        )
        _log.warning("Emergency stop engaged")
        self._emit("controller.emergency_stop", {})

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def open_gripper_blocking(self, settle_steps: int = 60) -> float:
        """Open the fingers and wait for them to arrive. Returns the final width."""
        self._robot.open_gripper()
        self._sim.step(settle_steps)
        return self._robot.get_gripper_width()

    def close_gripper_blocking(
        self, settle_steps: int = 120, stall_tolerance: float = 1e-4
    ) -> float:
        """Close the fingers until they stall, then hold. Returns the final width.

        Stops early once the width stops changing, which is the physical signal
        that the fingers have met either each other or an object. Waiting a fixed
        number of steps instead would either cut the squeeze short or waste time,
        and the stall itself is the evidence a grasp is forming.
        """
        self._robot.close_gripper()

        previous = self._robot.get_gripper_width()
        stalled_for = 0
        for _ in range(settle_steps):
            self._sim.step(1)
            width = self._robot.get_gripper_width()
            if abs(width - previous) < stall_tolerance:
                stalled_for += 1
                # Require several consecutive stalled steps: a single one can be
                # a solver artefact rather than real contact.
                if stalled_for >= 10:
                    break
            else:
                stalled_for = 0
            previous = width

        # Re-assert the closing command so the grip is maintained during the
        # motion that follows. Without this the fingers relax and drop the load.
        self._robot.close_gripper()
        return self._robot.get_gripper_width()

    def maintain_grasp(self) -> None:
        """Re-assert the close command. Call every control step while carrying."""
        self._robot.close_gripper()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _settle_at(self, target: NDArray[np.float64], max_steps: int = 90) -> None:
        """Hold a target until the arm converges or the budget runs out."""
        for _ in range(max_steps):
            self._robot.set_arm_joint_targets(target)
            self._sim.step(1)
            error = float(np.max(np.abs(self._robot.get_arm_joint_positions() - target)))
            if error < self.robot_config.joint_position_tolerance:
                return

    def _assert_within_workspace(
        self, joint_positions: NDArray[np.float64], margin: float = 0.0
    ) -> None:
        """Reject a configuration whose TCP leaves the allowed workspace.

        A safety limit, so it raises rather than returning a status: a commanded
        motion outside the workspace is a bug or a bad policy output, and
        retrying it is never the right response.

        ``margin`` widens the envelope for configurations the arm merely passes
        through. The box says where the arm should *operate*; RRT plans in joint
        space, so a path between two in-box poses routinely swings the TCP
        outside it, and rejecting that aborts a valid motion. Callers pass zero
        for goals and ``workspace_transit_margin`` for intermediate waypoints.
        """
        if self._workspace_min is None or self._workspace_max is None:
            return

        tcp = self._robot.forward_kinematics(joint_positions)
        low = self._workspace_min - margin
        high = self._workspace_max + margin
        if np.any(tcp.position < low) or np.any(tcp.position > high):
            where = "TCP target" if margin == 0.0 else "TCP in transit"
            raise SafetyViolation(
                f"{where} {np.round(tcp.position, 3).tolist()} is outside the workspace "
                f"{np.round(low, 3).tolist()} to {np.round(high, 3).tolist()}"
            )

    def _emit_execution(
        self,
        outcome: str,
        trajectory: Trajectory,
        index: int,
        worst_deviation: float,
        started: float,
    ) -> None:
        self._emit(
            "controller.follow_trajectory",
            {
                "outcome": outcome,
                "waypoint_index": index,
                "worst_path_deviation": worst_deviation,
                "wall_time_s": time.perf_counter() - started,
                "trajectory": trajectory.to_log(),
            },
        )

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(event, payload)

"""Online collision-aware motion planning.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

Two planners, both driven entirely by perception:

* **Lula RRT** for free-space motion. Franka is the only robot in this install
  shipping an RRT config (``path_planner_configs/franka``), which is a large part
  of why it was chosen.
* **Straight-line Cartesian** for grasp approach, lift and retreat. Kept separate
  from free-space planning because those motions must not curve: a curved
  approach sweeps the fingers sideways through the object being grasped.

Nothing here loads a recorded trajectory. The collision world is rebuilt from
the scene graph before every plan, so obstacles are whatever perception
currently reports -- objects that were not observed simply do not exist to the
planner, which is the honest behaviour for a robot that cannot see them.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import MotionConfig
from mfw.core.errors import PlanningError
from mfw.core.interfaces import IMotionPlanner
from mfw.core.types import Frame, JointState, Pose, SceneGraph, Trajectory
from mfw.motion.trajectory import densify_path, max_joint_step, time_parameterise
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["LulaMotionPlanner", "interpolate_pose"]

_log = get_logger("motion.planner")

_OBSTACLE_ROOT = "/World/planner_obstacles"


class LulaMotionPlanner(IMotionPlanner):
    """RRT + Cartesian planning against a perception-driven collision world."""

    def __init__(
        self,
        sim: Any,
        robot: Any,
        config: MotionConfig,
        event_logger: EventLogger | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._sim = sim
        self._robot = robot
        self._events = event_logger
        self._rrt: Any = None
        self._visualizer: Any = None
        self._obstacle_prims: dict[str, Any] = {}
        # Mirrors Lula's internal enabled flags. Lula raises on a redundant
        # enable/disable, so transitions must be tracked on this side.
        self._enabled_obstacles: set[str] = set()
        # Track ids deliberately excluded from collision, e.g. the object
        # currently being grasped. Held separately from _enabled_obstacles
        # because the per-plan rebuild would otherwise re-enable them.
        self._excluded: set[str] = set()
        self._build()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _build(self) -> None:
        from isaacsim.robot_motion.motion_generation import (  # noqa: PLC0415
            PathPlannerVisualizer,
            interface_config_loader,
        )
        from isaacsim.robot_motion.motion_generation.lula import RRT  # noqa: PLC0415

        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        # Plan for the hand frame, not a fingertip. The TCP offset is applied by
        # the robot layer, so every frame in this framework agrees.
        rrt_config["end_effector_frame_name"] = self._robot.config.tcp_parent_prim

        self._rrt = RRT(**rrt_config)
        self._rrt.set_max_iterations(10000)
        self._rrt.set_robot_base_pose(
            robot_position=np.array(self._robot.config.base_position, dtype=np.float64),
            robot_orientation=np.array(self._robot.config.base_quat, dtype=np.float64),
        )
        # The table and floor are permanent; registering them as static lets Lula
        # skip re-uploading their geometry on every world update.
        self._add_static_environment()

        self._visualizer = PathPlannerVisualizer(self._robot.articulation, self._rrt)
        _log.info("Lula RRT ready (end effector frame: %s)", rrt_config["end_effector_frame_name"])

    def _add_static_environment(self) -> None:
        """Register the floor and table as immovable obstacles.

        Both are fetched from the world's object registry rather than
        re-wrapped from their prim paths. Constructing a fresh ``FixedCuboid``
        over an existing prim risks re-authoring geometry that physics has
        already parsed; the registry hands back the very object the scene
        builder created.
        """
        scene = self._sim.world.scene

        for name in ("default_ground_plane", "ground_plane"):
            ground = scene.get_object(name)
            if ground is not None:
                try:
                    self._rrt.add_ground_plane(ground)
                    _log.info("Registered %s as a static planning obstacle", name)
                except Exception as exc:  # pragma: no cover - API variance
                    _log.warning("Could not register ground plane %s: %s", name, exc)
                break

        table = scene.get_object("table")
        if table is None:
            _log.info("No table in the scene registry; skipping static table obstacle")
            return
        try:
            if self._rrt.add_obstacle(table, static=True):
                _log.info("Registered table as a static planning obstacle")
            else:
                _log.warning("Lula rejected the table as an obstacle")
        except Exception as exc:  # pragma: no cover - API variance
            _log.warning("Could not register table obstacle: %s", exc)

    # ------------------------------------------------------------------
    # collision world
    # ------------------------------------------------------------------

    def update_collision_world(self, scene: SceneGraph) -> None:
        """Rebuild obstacles from the scene graph. Called before every plan.

        Perceived objects are mirrored onto a pool of invisible proxy cuboids
        that Lula can consume. A pool is reused rather than creating and
        destroying prims per observation, because stage mutation is expensive and
        would stall the control loop.

        The proxies are ``VisualCuboid``: geometry only, no collider. They must
        never participate in physics -- an invisible collider in the workspace
        would push the real objects around.
        """
        from isaacsim.core.api.objects import VisualCuboid  # noqa: PLC0415

        seen: set[str] = set()

        for track_id, obj in scene.objects.items():
            seen.add(track_id)
            inflated = obj.bbox.extents + 2.0 * self.config.obstacle_inflation
            prim = self._obstacle_prims.get(track_id)

            if prim is None:
                prim = VisualCuboid(
                    prim_path=f"{_OBSTACLE_ROOT}/{track_id}",
                    name=f"obstacle_{track_id}",
                    position=obj.bbox.center.position,
                    orientation=obj.bbox.center.quat,
                    scale=inflated,
                    visible=False,
                )
                self._obstacle_prims[track_id] = prim
                if self._rrt.add_obstacle(prim, static=False):
                    # Lula adds obstacles in the enabled state.
                    self._enabled_obstacles.add(track_id)
                    # An object first seen while excluded must not become an
                    # obstacle just because its proxy was created late.
                    if track_id in self._excluded:
                        self._set_obstacle_enabled(track_id, False)
                else:
                    _log.warning("Lula rejected obstacle for %s", track_id)
            else:
                prim.set_world_pose(
                    position=obj.bbox.center.position, orientation=obj.bbox.center.quat
                )
                prim.set_local_scale(inflated)
                # Do NOT re-enable an object that was deliberately excluded.
                #
                # This rebuild runs at the start of every plan, and Pick calls
                # exclude_from_collision(target) immediately *before* planning
                # its approach. Unconditionally enabling every object here undid
                # that exclusion on the very next line, so the target was always
                # an obstacle while the arm planned to grasp it.
                #
                # The symptom looked nothing like this. It presented as "no path
                # to the pregrasp standoff" for large objects only, because the
                # standoff sits 100 mm above the grasp point: a 51 mm foam brick's
                # inflated box tops out below it and the plan succeeds, while a
                # 213 mm cracker box swallows it. That produced a clean
                # correlation with object height which survived every other
                # hypothesis -- clutter, inner reach, distance from base -- and
                # pointed at geometry rather than at this.
                if track_id not in self._excluded:
                    self._set_obstacle_enabled(track_id, True)

        # Objects that vanished must stop blocking the planner, or the arm will
        # keep avoiding empty space forever.
        for track_id in set(self._obstacle_prims) - seen:
            self._set_obstacle_enabled(track_id, False)

        self._rrt.update_world()

    def _set_obstacle_enabled(self, track_id: str, enabled: bool) -> None:
        """Toggle an obstacle, but only on an actual state change.

        Lula treats a redundant toggle as a hard error -- "Attempted to enable an
        already-enabled obstacle" is a ``RuntimeError``, not a warning -- so the
        enabled set is mirrored here and only transitions are forwarded.
        """
        prim = self._obstacle_prims.get(track_id)
        if prim is None:
            return
        currently_enabled = track_id in self._enabled_obstacles
        if enabled == currently_enabled:
            return

        if enabled:
            self._rrt.enable_obstacle(prim)
            self._enabled_obstacles.add(track_id)
        else:
            self._rrt.disable_obstacle(prim)
            self._enabled_obstacles.discard(track_id)

    def exclude_from_collision(self, track_id: str) -> None:
        """Stop treating one object as an obstacle.

        Required for the grasp approach: the target object must not be an
        obstacle while the gripper is deliberately closing around it, or every
        approach plan is rejected as a collision. Re-enabled after release.
        """
        self._excluded.add(track_id)
        self._set_obstacle_enabled(track_id, False)
        self._rrt.update_world()

    def include_in_collision(self, track_id: str) -> None:
        """Resume treating one object as an obstacle."""
        self._excluded.discard(track_id)
        self._set_obstacle_enabled(track_id, True)
        self._rrt.update_world()

    # ------------------------------------------------------------------
    # IMotionPlanner
    # ------------------------------------------------------------------

    def plan_to_pose(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Plan a collision-free path to a Cartesian TCP goal.

        The goal is converted from TCP to the hand frame first, since the planner
        works in the hand frame.
        """
        goal_joints = self._robot.inverse_kinematics(goal_pose, seed=start.positions)
        if goal_joints is None:
            _log.debug("plan_to_pose: no IK solution for the goal")
            return None
        return self.plan_to_joint(start, goal_joints, scene)

    def plan_to_joint(
        self, start: JointState, goal: NDArray[np.float64], scene: SceneGraph
    ) -> Trajectory | None:
        """Plan a collision-free path to an explicit joint configuration."""
        self.update_collision_world(scene)
        started = time.perf_counter()

        self._rrt.set_cspace_target(np.asarray(goal, dtype=np.float64))
        start_positions = np.asarray(start.positions, dtype=np.float64)

        # Plan several times and keep the shortest path.
        #
        # RRT returns the first feasible path it finds, and a raw tree path
        # meanders: one measured here reached an object 0.33 m in front of the
        # robot by swinging the TCP to x = -0.202, behind the arm's own base.
        # Collision-free and valid; simply awful, and it aborted the pick.
        #
        # A collision-checked shortcut pass is the textbook fix, but Lula's RRT
        # wrapper exposes no state-validity query, so a candidate straight-line
        # segment cannot be verified as safe. Re-sampling does not need one: the
        # meandering is an artefact of where the tree happened to grow, so a
        # different seed usually produces a much better path, and every
        # candidate is collision-free by construction.
        best_path: NDArray[np.float64] | None = None
        best_length = float("inf")
        attempts = max(1, int(self.config.plan_attempts))

        for attempt in range(attempts):
            if attempt > 0 and hasattr(self._rrt, "set_random_seed"):
                # Vary the seed explicitly rather than relying on the planner's
                # internal state, so a retry is a genuinely different sample and
                # a fixed seed keeps a run reproducible.
                self._rrt.set_random_seed(attempt * 7919 + 13)

            raw_path = self._rrt.compute_path(
                start_positions, np.array([], dtype=np.float64)
            )
            if raw_path is None or len(raw_path) == 0:
                continue

            candidate = np.asarray(raw_path, dtype=np.float64)
            length = float(np.sum(np.linalg.norm(np.diff(candidate, axis=0), axis=1)))
            if length < best_length:
                best_path, best_length = candidate, length

        elapsed = time.perf_counter() - started

        if best_path is None:
            self._emit("motion.plan_failed", {"kind": "joint", "planning_time_s": elapsed})
            _log.debug("RRT found no path in %d attempt(s) (%.2f s)", attempts, elapsed)
            return None

        _log.debug(
            "RRT: best of %d attempt(s), joint path length %.2f rad (%.2f s)",
            attempts,
            best_length,
            elapsed,
        )
        path = best_path
        # RRT returns sparse waypoints; refine so the controller has something
        # smooth to track.
        if max_joint_step(path) > 0.1:
            path = densify_path(path, max_step=0.05)

        trajectory = time_parameterise(
            path,
            joint_names=self._robot.joint_names,
            max_velocity=self.config.max_joint_velocity,
            max_acceleration=self.config.max_joint_acceleration,
            planner_name="lula_rrt",
            planning_time_s=elapsed,
        )
        self._emit("motion.plan", {"kind": "joint", **trajectory.to_log()})
        return trajectory

    def plan_cartesian_line(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Plan a straight-line TCP motion, solving IK along the way.

        Used for grasp approach, lift and retreat. A curved path through those
        phases drags the fingers sideways across the object.

        Deliberately does *not* consult the collision world: during an approach
        the gripper is intentionally converging on the target, so a collision
        check against that target would reject every valid plan. Safety comes
        from the motion being short, straight, and bounded by the pregrasp
        standoff.
        """
        current_pose = self._robot.forward_kinematics(start.positions)
        if goal_pose.frame is not Frame.WORLD:
            raise PlanningError(f"Cartesian goal must be in world frame, got {goal_pose.frame}")

        distance = current_pose.translation_distance(goal_pose)
        steps = max(2, int(np.ceil(distance / self.config.cartesian_step)))
        started = time.perf_counter()

        joint_path = [np.asarray(start.positions, dtype=np.float64)]
        seed = joint_path[0]

        for i in range(1, steps + 1):
            fraction = i / steps
            waypoint_pose = interpolate_pose(current_pose, goal_pose, fraction)
            solution = self._robot.inverse_kinematics(waypoint_pose, seed=seed)
            if solution is None:
                self._emit(
                    "motion.cartesian_failed",
                    {"failed_at_fraction": fraction, "distance": distance},
                )
                _log.debug("Cartesian IK failed at %.0f%% of the path", fraction * 100)
                return None
            joint_path.append(solution)
            seed = solution

        path = np.vstack(joint_path)
        trajectory = time_parameterise(
            path,
            joint_names=self._robot.joint_names,
            max_velocity=self.config.max_joint_velocity,
            max_acceleration=self.config.max_joint_acceleration,
            planner_name="cartesian_line",
            planning_time_s=time.perf_counter() - started,
        )
        self._emit("motion.plan", {"kind": "cartesian", **trajectory.to_log()})
        return trajectory

    def plan_with_retries(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Plan to a Cartesian goal, varying the IK branch as well as the tree seed.

        Re-seeding the RRT alone is not enough, and the distinction is the whole
        point of this method. ``plan_to_pose`` solves IK with
        ``seed=start.positions`` every time, so retrying it produces the *same*
        goal configuration on every attempt. A 7-DOF arm has several IK branches
        for one Cartesian pose and only some of them clear the table, so when the
        branch IK happens to return is in collision, Lula rejects it identically
        each time and the retries are pure cost.

        Measured on the benchmark mug: three attempts, 141 s, the same joint
        vector printed as ``Invalid configuration`` throughout, and the pick
        failed. The identical pick had succeeded on an earlier run purely
        because IK returned the other branch -- which is what made the failure
        look non-deterministic rather than like the deterministic bug it is.

        So each attempt re-solves IK from a different warm start, and a goal
        configuration already tried is skipped without planning: a fresh tree
        seed cannot rescue a goal state the planner has already rejected.
        """
        start_positions = np.asarray(start.positions, dtype=np.float64)
        attempts = max(1, int(self.config.max_replan_attempts))
        # Seeded explicitly so a run stays reproducible; an unseeded generator
        # would make a failing pick unrepeatable and therefore undebuggable.
        rng = np.random.default_rng(20260901)

        tried: list[NDArray[np.float64]] = []
        planned = 0
        # More seeds than attempts: a seed can fail IK outright or re-converge
        # on a branch already tried, and neither should consume planning budget.
        for index in range(attempts * 4):
            if planned >= attempts:
                break

            seed = self._ik_seed(index, start_positions, rng)
            goal_joints = self._robot.inverse_kinematics(goal_pose, seed=seed)
            if goal_joints is None:
                continue
            if any(np.allclose(goal_joints, done, atol=1e-3) for done in tried):
                continue

            tried.append(goal_joints)
            planned += 1

            self._rrt.set_random_seed(planned * 7919)
            trajectory = self.plan_to_joint(start, goal_joints, scene)
            if trajectory is not None:
                if planned > 1:
                    _log.info("Plan found on IK branch %d", planned)
                return trajectory

        _log.debug(
            "plan_with_retries: %d distinct IK branch(es) tried, none plannable",
            len(tried),
        )
        return None

    def _ik_seed(
        self,
        index: int,
        start_positions: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Warm start for the ``index``-th IK solve, cheapest branch first.

        Index 0 is the current configuration: it yields the nearest branch,
        which is the shortest motion and usually collision-free. Index 1 is the
        home posture, which is known reachable and clear of the table, so it
        tends to land in the "natural" elbow configuration when the current one
        does not. Beyond that, perturbations wide enough (+/-0.9 rad) to escape
        the basin that just failed rather than re-converging on it.
        """
        if index == 0:
            return start_positions
        if index == 1:
            home = np.asarray(
                self._robot.config.home_joint_positions, dtype=np.float64
            )
            if home.shape == start_positions.shape:
                return home
        return start_positions + rng.uniform(-0.9, 0.9, size=start_positions.shape)

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(event, payload)


def interpolate_pose(start: Pose, end: Pose, fraction: float) -> Pose:
    """Linear position interpolation with SLERP orientation.

    Public because it is Isaac-free and the hardware lane's Cartesian planner
    steps along the same line; two copies of a SLERP would drift apart.

    SLERP rather than component-wise quaternion lerp: a naive lerp does not
    travel at constant angular rate and can pass through a non-unit quaternion,
    which would make the wrist speed up and slow down mid-approach.
    """
    from mfw.utils import transforms as tf

    position = start.position + fraction * (end.position - start.position)

    q0 = start.quat
    q1 = end.quat
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        # Take the short way round the double cover.
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Nearly identical; lerp is numerically safer than SLERP here.
        quat = q0 + fraction * (q1 - q0)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        quat = (
            np.sin((1.0 - fraction) * theta) / sin_theta * q0
            + np.sin(fraction * theta) / sin_theta * q1
        )

    return Pose(position, quat / np.linalg.norm(quat), Frame.WORLD)


# Former private name, kept so nothing that imported it breaks.
_interpolate_pose = interpolate_pose

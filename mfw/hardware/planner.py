"""Joint-space motion planning for the planar hobby arm.

Pure NumPy. This module must never import Isaac Sim, torch or transformers.

There is no collision world on this lane -- no RRT, no obstacle meshes -- and
a 4-DoF planar arm does not need one: between any two configurations the
straight joint-space segment is a smooth, predictable sweep, and the only
things it can hit are the table and the objects on it. Both are checked
analytically on every waypoint:

* every link point (shoulder, elbow, wrist) and the TCP stays above the
  support plane (a small tolerance lets the TCP touch it, which the
  ``--touch`` calibration needs);
* the TCP does not pass through any perceived object's inflated box, except
  the ones :meth:`exclude_from_collision` names (the grasp target while
  closing on it, the held object while carrying it).

When the direct segment fails, the planner retries through a **transit
height**: raise the start TCP to the transit height, cross there, descend to
the goal. That is the whole of the transit logic, and it is what carries an
object over a bowl rim instead of through it.

The transit height is computed **per plan**, never read blindly from the
config (review GEO-4). ``hardware.arm.transit_height`` is the floor; the
planner raises it to the tallest non-excluded obstacle's top plus the
obstacle inflation, plus the held object's hang-down below the TCP (half its
height: it is grasped at mid-height), plus the finger dip of a tilted transit
(:data:`_TRANSIT_FINGER_DIP_M`). Measured trap: ``transit_height: 0.10`` with a
0.10 m ``bin`` in ``object_sizes`` meant the via points sat exactly at the
bin's height, inside its 0.02 m inflated box, so every crossing of a bin had
*no plan at all* -- a refusal, not a collision, but a demo that cannot place
into its own bin. On the placeholder geometry the crossing first succeeds at
0.125 m; the config now floors at 0.15 m and the per-plan rule covers taller
things. Reach at 0.15 m: the 45-degree transit pitch reaches it from about
0.10 m to 0.20 m out (``tests/test_hardware_grasp_planner.py`` pins it), so a
transit that starts or ends outside that band gets no via point there and the
plan is refused rather than crossed low.

While something is carried, every obstacle is additionally inflated by the
held object's half-height (all axes): the TCP-in-box test knows nothing about
what hangs below the TCP, and a cube carried over a bowl otherwise had 15 mm
of real clearance where the planner believed it had 20. The planner cannot
see the gripper, so "held" is inferred from :meth:`exclude_from_collision`:
an excluded object that is still in the scene is treated as held. During the
descent onto a grasp target that over-approximates (the target is excluded
while nothing is held yet), which costs a centimetre or two of transit height
on the way in and nothing else.

The held object is a box, not a point (re-review, motion lens). The hang
above is only right for a vertical tool. A marker is gripped across its 19 mm
width, so its 140 mm length lies along the tool's in-plane X axis, and the
45-75 degree transit pitch tilts that axis: the marker's end hangs about
``L/2 sin(tilt) + H/2 cos(tilt)`` = 49-56 mm below the TCP, not 9.5 mm. 26
planner-approved crossings of the 0.10 m bin put the marker's end 3.6-8.8 mm
inside the real bin. Now, when something is *held*, every waypoint also
checks the held object's own box -- placed in the TCP frame (long side along
tool X, narrow side along the closing axis Y, height along the approach Z) and
sampled every 2 cm -- against every obstacle and the table, and the transit
heights tried include the hang at each transit pitch. "Held" is the planner's
inference: an excluded object whose centre lies within reach of the plan's
start TCP (the jaw is at it), or one named by :meth:`set_held_object`. An
excluded grasp target the arm is still travelling to is therefore not boxed
around the TCP on the way in.

Measured trap -- the lift must be allowed to tilt the tool. On the placeholder
geometry (upper arm 0.105, tool 0.09, wrist limit +/-90 deg) a *vertical* tool
at a 0.10 m transit height puts the wrist 0.12 m above the shoulder, which
the upper arm cannot lift while the wrist limit pins the forearm at or below
horizontal; IK of "the same pose, higher" is therefore ``None`` at every
radius, and the transit fallback never fired for a top-down grasp: a wall
between two low poses simply failed the plan. Tilting the tool outward in the
arm plane (75, 60, then 45 degrees from horizontal) makes a 0.10 m transit
reachable from 12 cm out to 24 cm (0.15 m: 10 to 20 cm, 45 degrees only, with
the elbow held to +/-90 degrees), so :meth:`JointSpacePlanner._raised` falls
back to those pitches when the original orientation is out of reach. The
object rides tilted in the jaw for the crossing, which is harmless; the
grasp and release poses themselves are never changed.

``plan_with_retries`` walks every IK branch of the goal (elbow-up first, then
elbow-down, then the antipodal yaw when it fits the limits) so a goal the
preferred branch cannot reach without sweeping the table is still reached on
the other branch. The three methods skills call that are not on
``IMotionPlanner`` -- ``plan_with_retries``, ``exclude_from_collision``,
``include_in_collision`` -- are provided with the Lula planner's semantics.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import HardwareArmConfig, MotionConfig
from mfw.core.errors import PlanningError
from mfw.core.interfaces import IMotionPlanner
from mfw.core.types import Frame, JointState, Pose, SceneGraph, Trajectory
from mfw.grasp.generator import build_grasp_pose
from mfw.motion.planner import interpolate_pose
from mfw.motion.trajectory import densify_path, time_parameterise
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["JointSpacePlanner"]

_log = get_logger("hardware.planner")

_DENSIFY_STEP_RAD = 0.05
_TABLE_TOLERANCE_M = 0.01
"""How far below the support a link point may dip before the path is rejected."""
_OBSTACLE_INFLATION_M = 0.02
_TRANSIT_XY_THRESHOLD_M = 0.03
"""Below this horizontal travel a motion is a local adjust and needs no transit."""
_TRANSIT_PITCHES_DEG = (75.0, 60.0, 45.0)
"""Tool pitches (degrees from horizontal, most vertical first) tried for the
transit lift when the pose's own orientation cannot reach the transit height.
See the module docstring for why a vertical tool never can on a short arm."""
_TRANSIT_FINGER_DIP_M = 0.016
"""How far one finger can hang below the TCP when the tool is tilted 45
degrees for the transit: half a 45 mm jaw times sin 45. Added to the per-plan
transit height on top of the obstacle inflation."""
_HELD_SAMPLE_STEP_M = 0.02
"""Spacing of the points sampled along a held object's box."""
_HELD_BOX_INFLATION_M = 0.01
"""Obstacle inflation for the held box's samples: half the sample spacing, so
an obstacle corner poking between two samples still has one inside its
inflated box. Smaller than the TCP's 2 cm on purpose -- the box *is* the
object's extent (the size table), not a point standing in for it."""
_HELD_REACH_MARGIN_M = 0.03
"""An excluded object counts as held when its centre is within half its
largest horizontal extent plus this of the plan's start TCP (XY)."""


class JointSpacePlanner(IMotionPlanner):
    """Straight joint interpolation with analytic table and object checks."""

    def __init__(
        self,
        robot: Any,
        kinematics: Any,
        config: MotionConfig,
        arm: HardwareArmConfig,
        support_height: float = 0.0,
        event_logger: EventLogger | None = None,
    ) -> None:
        config.validate()
        arm.validate()
        self.config = config
        self.arm = arm
        self._robot = robot
        self._kin = kinematics
        self._support = float(support_height)
        self._events = event_logger
        self._scene: SceneGraph | None = None
        self._excluded: set[str] = set()
        self._held_explicit: str | None = None
        self._plan_held: list[tuple[str, NDArray[np.float64]]] = []
        """Held boxes (tool-frame extents) for the plan in progress."""

    # ------------------------------------------------------------------
    # collision world

    def update_collision_world(self, scene: SceneGraph) -> None:
        """Remember the latest scene; obstacles are its objects' boxes."""
        self._scene = scene

    def exclude_from_collision(self, track_id: str) -> None:
        """Stop treating one object as an obstacle (the grasp target, the held one)."""
        self._excluded.add(track_id)

    def include_in_collision(self, track_id: str) -> None:
        """Resume treating one object as an obstacle."""
        self._excluded.discard(track_id)
        if self._held_explicit == track_id:
            self._held_explicit = None

    def set_held_object(self, track_id: str | None) -> None:
        """Name the object in the jaw (or ``None``) instead of inferring it.

        The skills know; the planner otherwise infers "held" from exclusion
        plus proximity to the plan's start (see the module docstring). A
        named object is excluded from collision as well.
        """
        self._held_explicit = track_id
        if track_id is not None:
            self._excluded.add(track_id)

    # ------------------------------------------------------------------
    # validity

    def held_half_height(self) -> float:
        """Half the height of the tallest excluded object still in the scene, metres.

        Zero when nothing is excluded or the excluded tracks have left the
        scene. This is the planner's only notion of "something is in the jaw":
        the skills exclude the grasp target before closing on it and the held
        object while carrying it (``Pick``/``Place`` in ``mfw.skills.primitives``).
        """
        if self._scene is None or not self._excluded:
            return 0.0
        tallest = 0.0
        for track_id in self._excluded:
            obj = self._scene.objects.get(track_id)
            if obj is None:
                continue
            tallest = max(tallest, float(obj.bbox.extents[2]) / 2.0)
        return tallest

    @staticmethod
    def held_tool_extents(obj: Any) -> NDArray[np.float64]:
        """A held object's extents along the tool's (X, Y, Z) axes, metres.

        The jaw has no roll, so it closes across the object's narrower
        horizontal side (along the closing axis Y); the longer side then lies
        along tool X, in the arm plane, and tilts with the pitch; the height
        runs along the approach axis Z.
        """
        ext = np.asarray(obj.bbox.extents, dtype=np.float64)
        return np.array([max(ext[0], ext[1]), min(ext[0], ext[1]), ext[2]], dtype=np.float64)

    @staticmethod
    def hang_below_tcp(tool_extents: NDArray[np.float64], tilt_rad: float) -> float:
        """How far a held box reaches below the TCP with the tool tilted
        ``tilt_rad`` from vertical in the arm plane."""
        half = np.asarray(tool_extents, dtype=np.float64) / 2.0
        return float(half[0] * abs(math.sin(tilt_rad)) + half[2] * abs(math.cos(tilt_rad)))

    def held_objects(self, start_q: NDArray[np.float64] | None = None) -> list[tuple[str, NDArray[np.float64]]]:
        """``(track_id, tool-frame extents)`` of what the jaw holds for a plan from ``start_q``.

        The explicitly named object (:meth:`set_held_object`) when there is
        one; otherwise every excluded object still in the scene whose centre
        is within its own half-extent plus :data:`_HELD_REACH_MARGIN_M` of the
        start TCP in XY. Without ``start_q`` every excluded object counts.
        """
        if self._scene is None or not self._excluded:
            return []
        tcp_xy = None if start_q is None else np.asarray(self._kin.fk(start_q).position[:2], dtype=np.float64)
        out: list[tuple[str, NDArray[np.float64]]] = []
        for track_id in sorted(self._excluded):
            obj = self._scene.objects.get(track_id)
            if obj is None:
                continue
            if self._held_explicit is not None:
                if track_id != self._held_explicit:
                    continue
            elif tcp_xy is not None:
                centre = np.asarray(obj.bbox.center.position[:2], dtype=np.float64)
                reach = float(np.max(np.asarray(obj.bbox.extents, dtype=np.float64)[:2])) / 2.0
                if float(np.linalg.norm(centre - tcp_xy)) > reach + _HELD_REACH_MARGIN_M:
                    continue
            out.append((track_id, self.held_tool_extents(obj)))
        return out

    def transit_heights_for_scene(
        self, held: list[tuple[str, NDArray[np.float64]]] | None = None
    ) -> list[float]:
        """Candidate transit heights, ascending; the first is :meth:`transit_height_for_scene`.

        One per transit pitch the lift may take (vertical, 75, 60, 45 deg):
        a held box hangs further below the TCP the more the tool tilts, so
        the crossing a tilted transit needs is higher. The path check decides
        which one is actually clear.
        """
        base = self.transit_height_for_scene()
        if not held:
            return [base]
        heights = {round(base, 6)}
        tops = self._obstacle_tops()
        for pitch_deg in _TRANSIT_PITCHES_DEG:
            tilt = math.radians(90.0 - pitch_deg)
            hang = max(self.hang_below_tcp(ext, tilt) for _, ext in held)
            height = float(self.arm.transit_height)
            for top in tops:
                height = max(height, top + _OBSTACLE_INFLATION_M + hang + _TRANSIT_FINGER_DIP_M)
            heights.add(round(max(height, base), 6))
        return sorted(heights)

    def _obstacle_tops(self) -> list[float]:
        if self._scene is None:
            return []
        return [
            float(obj.bbox.center.position[2]) + float(obj.bbox.extents[2]) / 2.0
            for track_id, obj in self._scene.objects.items()
            if track_id not in self._excluded
        ]

    def transit_height_for_scene(self) -> float:
        """Transit height for the current scene and held object, metres.

        ``max(hardware.arm.transit_height, tallest obstacle top + inflation +
        held half-height + finger dip)``. Excluded objects are not obstacles,
        so the held object never raises its own crossing; only what it must
        clear does.
        """
        height = float(self.arm.transit_height)
        if self._scene is None:
            return height
        hang = self.held_half_height()
        for track_id, obj in self._scene.objects.items():
            if track_id in self._excluded:
                continue
            top = float(obj.bbox.center.position[2]) + float(obj.bbox.extents[2]) / 2.0
            height = max(height, top + _OBSTACLE_INFLATION_M + hang + _TRANSIT_FINGER_DIP_M)
        return height

    def _tcp_in_obstacle(self, tcp: NDArray[np.float64]) -> str | None:
        if self._scene is None:
            return None
        inflation = _OBSTACLE_INFLATION_M + self.held_half_height()
        for track_id, obj in self._scene.objects.items():
            if track_id in self._excluded:
                continue
            centre = np.asarray(obj.bbox.center.position, dtype=np.float64)
            half = np.asarray(obj.bbox.extents, dtype=np.float64) / 2.0 + inflation
            local = obj.bbox.center.rotation_matrix().T @ (tcp - centre)
            if np.all(np.abs(local) <= half):
                return track_id
        return None

    @staticmethod
    def _held_samples(tool_extents: NDArray[np.float64]) -> NDArray[np.float64]:
        """Points on a held box in the tool frame: its eight corners and the
        long edges every :data:`_HELD_SAMPLE_STEP_M`."""
        half = np.asarray(tool_extents, dtype=np.float64) / 2.0
        axes = []
        for h in half:
            n = max(2, int(math.ceil(2.0 * h / _HELD_SAMPLE_STEP_M)) + 1)
            axes.append(np.linspace(-h, h, n))
        xs, ys, zs = axes
        # Edges along X at the four (y, z) corners, plus the corner rings.
        grid = np.array([[x, y, z] for x in xs for y in (ys[0], ys[-1]) for z in (zs[0], zs[-1])])
        return grid

    def _held_box_hit(self, q: NDArray[np.float64], held: list[tuple[str, NDArray[np.float64]]]) -> str | None:
        """Why the held box at ``q`` collides (an obstacle or the table), else ``None``."""
        pose = self._kin.fk(q)
        rot = pose.rotation_matrix()
        tcp = np.asarray(pose.position, dtype=np.float64)
        floor = self._support - _TABLE_TOLERANCE_M
        for track_id, extents in held:
            world = tcp + self._held_samples(extents) @ rot.T
            if np.any(world[:, 2] < floor):
                return f"held {track_id} below the table"
            if self._scene is None:
                continue
            for other_id, obj in self._scene.objects.items():
                if other_id in self._excluded:
                    continue
                centre = np.asarray(obj.bbox.center.position, dtype=np.float64)
                half = np.asarray(obj.bbox.extents, dtype=np.float64) / 2.0 + _HELD_BOX_INFLATION_M
                local = (world - centre) @ obj.bbox.center.rotation_matrix()
                if np.any(np.all(np.abs(local) <= half, axis=1)):
                    return f"held {track_id} through {other_id}"
        return None

    def _path_is_clear(self, path: NDArray[np.float64], check_objects: bool = True) -> tuple[bool, str]:
        """Table clearance for every link, plus object boxes for the TCP and,
        while something is held, for the held object's own box."""
        floor = self._support - _TABLE_TOLERANCE_M
        held = self._plan_held if check_objects else []
        for q in path:
            if not self._kin.within_limits(q):
                return False, "joint limit"
            points = self._kin.link_points(q)
            if np.any(points[1:, 2] < floor):
                return False, "link below the table"
            if check_objects:
                hit = self._tcp_in_obstacle(points[4])
                if hit is not None:
                    return False, f"TCP through {hit}"
                if held:
                    why = self._held_box_hit(q, held)
                    if why is not None:
                        return False, why
        return True, ""

    # ------------------------------------------------------------------
    # segments

    def _segment(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        return densify_path(np.vstack([a, b]), max_step=_DENSIFY_STEP_RAD)

    def _join(self, configurations: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        pieces = [configurations[0].reshape(1, -1)]
        for a, b in zip(configurations[:-1], configurations[1:]):
            pieces.append(self._segment(a, b)[1:])
        return np.vstack(pieces)

    def _raised(
        self, q: NDArray[np.float64], height: float | None = None
    ) -> NDArray[np.float64] | None:
        """IK of the TCP of ``q`` lifted to ``height`` (default: the per-scene transit height).

        The pose's own orientation is tried first. When that is out of reach
        (always, for a vertical tool on the placeholder geometry -- see the
        module docstring) the tool is tilted outward in the arm plane through
        ``_TRANSIT_PITCHES_DEG`` until one pitch fits. ``None`` when the TCP is
        already at or above the transit height, or no pitch reaches it.
        """
        transit_height = self.transit_height_for_scene() if height is None else float(height)
        pose = self._kin.fk(q)
        if pose.position[2] >= transit_height - 1e-6:
            return None
        x, y = float(pose.position[0]), float(pose.position[1])
        lifted = np.array([x, y, transit_height])
        solution = self._kin.ik(Pose(lifted, pose.quat, Frame.WORLD), seed=q)
        if solution is not None:
            return solution

        # Roll is not a degree of freedom, so only the approach axis matters
        # to IK; the tangential closing axis merely completes a valid frame.
        bearing = math.atan2(y, x)
        radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        tangential = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
        for pitch_deg in _TRANSIT_PITCHES_DEG:
            tilt = math.radians(90.0 - pitch_deg)
            approach = math.sin(tilt) * radial - math.cos(tilt) * np.array([0.0, 0.0, 1.0])
            solution = self._kin.ik(build_grasp_pose(lifted, tangential, approach), seed=q)
            if solution is not None:
                _log.debug(
                    "Transit lift at (%.3f, %.3f) needs the tool tilted to %.0f deg",
                    x, y, pitch_deg,
                )
                return solution
        return None

    def _candidate_paths(self, start: NDArray[np.float64], goal: NDArray[np.float64]) -> list[tuple[str, NDArray[np.float64]]]:
        direct = self._join([start, goal])
        paths: list[tuple[str, NDArray[np.float64]]] = [("direct", direct)]

        tcp_start = self._kin.fk(start).position
        tcp_goal = self._kin.fk(goal).position
        if float(np.linalg.norm(tcp_goal[:2] - tcp_start[:2])) < _TRANSIT_XY_THRESHOLD_M:
            return paths

        for height in self.transit_heights_for_scene(self._plan_held):
            vias: list[NDArray[np.float64]] = []
            up = self._raised(start, height)
            if up is not None:
                vias.append(up)
            down = self._raised(goal, height)
            if down is not None:
                vias.append(down)
            if vias:
                paths.append(("transit", self._join([start, *vias, goal])))
            elif height > self.arm.transit_height + 1e-9:
                _log.debug(
                    "Transit height %.3f m (raised above %.3f for the scene) is out of reach at "
                    "both ends; no transit at this height",
                    height, self.arm.transit_height,
                )
        return paths

    def _finish(self, path: NDArray[np.float64], name: str, started: float, kind: str) -> Trajectory:
        trajectory = time_parameterise(
            path,
            joint_names=self._robot.joint_names,
            max_velocity=self.config.max_joint_velocity,
            max_acceleration=self.config.max_joint_acceleration,
            planner_name=name,
            planning_time_s=time.perf_counter() - started,
        )
        self._emit("motion.plan", {"kind": kind, **trajectory.to_log()})
        return trajectory

    # ------------------------------------------------------------------
    # IMotionPlanner

    def plan_to_joint(
        self, start: JointState, goal: NDArray[np.float64], scene: SceneGraph
    ) -> Trajectory | None:
        """Direct joint interpolation, else via the transit height, else ``None``."""
        self.update_collision_world(scene)
        started = time.perf_counter()
        q0 = np.asarray(start.positions, dtype=np.float64)
        q1 = np.asarray(goal, dtype=np.float64)
        if q1.shape != q0.shape:
            raise PlanningError(f"goal has {q1.shape[0]} joints, start has {q0.shape[0]}")
        if not self._kin.within_limits(q1):
            self._emit("motion.plan_failed", {"kind": "joint", "reason": "goal outside joint limits"})
            return None

        self._plan_held = self.held_objects(q0)
        reasons: list[str] = []
        for name, path in self._candidate_paths(q0, q1):
            ok, why = self._path_is_clear(path)
            if ok:
                return self._finish(path, f"joint_space_{name}", started, "joint")
            reasons.append(f"{name}: {why}")
        self._emit("motion.plan_failed", {"kind": "joint", "reason": "; ".join(reasons)})
        _log.debug("plan_to_joint failed: %s", "; ".join(reasons))
        return None

    def plan_to_pose(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """IK seeded from the start, then :meth:`plan_to_joint`."""
        goal = self._kin.ik(goal_pose, seed=np.asarray(start.positions, dtype=np.float64))
        if goal is None:
            self._emit("motion.plan_failed", {"kind": "pose", "reason": "no IK solution"})
            return None
        return self.plan_to_joint(start, goal, scene)

    def plan_cartesian_line(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph, check_objects: bool = False
    ) -> Trajectory | None:
        """Straight TCP line, IK per step, no obstacle check by default (as on the sim lane).

        Approach, lift and retreat converge on or depart from an object on
        purpose; only the table is checked. ``check_objects=True`` applies
        the full planned-path check (TCP and held box against every
        non-excluded obstacle) -- a *transit* must never fall back to an
        unchecked line (re-review: Place's fallback drove a held cube 37-41 mm
        into the bin the planner had refused to cross).
        """
        if goal_pose.frame is not Frame.WORLD:
            raise PlanningError(f"Cartesian goal must be in world frame, got {goal_pose.frame}")
        started = time.perf_counter()
        q0 = np.asarray(start.positions, dtype=np.float64)
        current = self._kin.fk(q0)
        distance = current.translation_distance(goal_pose)
        steps = max(2, int(np.ceil(distance / self.config.cartesian_step)))

        path = [q0]
        seed = q0
        for i in range(1, steps + 1):
            fraction = i / steps
            waypoint = interpolate_pose(current, goal_pose, fraction)
            solution = self._kin.ik(waypoint, seed=seed)
            if solution is None:
                self._emit("motion.cartesian_failed", {"failed_at_fraction": fraction, "distance": distance})
                _log.debug("Cartesian IK failed at %.0f%% of the line", fraction * 100)
                return None
            path.append(solution)
            seed = solution
        arr = np.vstack(path)
        if check_objects:
            self.update_collision_world(scene)
            self._plan_held = self.held_objects(q0)
        ok, why = self._path_is_clear(arr, check_objects=check_objects)
        if not ok:
            self._emit("motion.cartesian_failed", {"reason": why, "distance": distance})
            return None
        return self._finish(arr, "cartesian_line", started, "cartesian")

    def plan_with_retries(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Try every IK branch of the goal, preferred first."""
        branches = self._kin.both_branches(goal_pose)
        if not branches:
            self._emit("motion.plan_failed", {"kind": "pose", "reason": "no IK branch reaches the goal"})
            return None
        q0 = np.asarray(start.positions, dtype=np.float64)
        for index, goal in enumerate(branches):
            trajectory = self.plan_to_joint(start, goal, scene)
            if trajectory is not None:
                if index > 0:
                    _log.info("Plan found on IK branch %d", index + 1)
                return trajectory
        _log.debug("plan_with_retries: %d branch(es) tried from %s, none clear", len(branches), np.round(q0, 3))
        return None

    # ------------------------------------------------------------------

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(event, payload)

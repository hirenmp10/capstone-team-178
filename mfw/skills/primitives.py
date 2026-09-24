"""The atomic skill library.

Isaac Sim is reached only through injected components; this module imports none
of it directly.

Every skill here is independently executable and **ends by returning**. None of
them calls another. In particular:

* ``Pick`` observes, plans a grasp, approaches, closes, verifies, lifts, and then
  **holds**. It does not place.
* ``Place`` finds a destination, lowers, releases, retreats, and then **stops**.
  It does not return home.

Manipulation is pure PhysX throughout: no parenting, no teleporting, no pose
writing, no fake attachment. Objects move only because the fingers push on them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.core.errors import AmbiguousReference, ObjectNotFound
from mfw.core.types import Frame, Pose, SkillResult
from mfw.grasp.generator import generate_grasp_candidates
from mfw.physics.contact import verify_grasp
from mfw.skills.base import Skill
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = [
    "Observe",
    "ScanScene",
    "LookAt",
    "MoveTo",
    "MoveRelative",
    "Pick",
    "Place",
    "OpenGripper",
    "CloseGripper",
    "RotateWrist",
    "GoHome",
    "Wait",
    "Stop",
    "EmergencyStop",
    "ALL_SKILLS",
]

_log = get_logger("skills.primitives")


def _clamp_to_workspace(ctx: Any, position: NDArray[np.float64], margin: float = 0.01) -> NDArray[np.float64]:
    """Clip a target position into the configured workspace.

    Skills must not *command* a pose the controller will reject: a retreat that
    overshoots the ceiling by a millimetre would raise ``SafetyViolation`` and
    abort the whole command, when the intent -- "get clear" -- was perfectly
    achievable a hair lower. The controller's check stays as the real guard; this
    just keeps well-intentioned motions inside it.
    """
    low = np.asarray(ctx.config.scene.workspace_min, dtype=np.float64) + margin
    high = np.asarray(ctx.config.scene.workspace_max, dtype=np.float64) - margin
    return np.clip(np.asarray(position, dtype=np.float64), low, high)


def _resolve_target(ctx: Any, params: dict[str, Any], key: str = "target") -> str:
    """Resolve a target reference to a track id.

    Accepts an explicit ``track_id``, a class label, or a pronoun -- pronouns and
    labels both go through memory, which is what makes "place it" work. Raises
    rather than guessing when a reference is ambiguous.
    """
    reference = params.get(key)
    if reference is None:
        raise ObjectNotFound(f"no {key} specified")

    reference = str(reference).strip()
    scene = ctx.vision.require_fresh_scene()

    if reference in scene.objects:
        return reference

    if ctx.memory is not None:
        resolved = ctx.memory.resolve_reference(reference)
        if resolved is not None and resolved in scene.objects:
            return resolved

    matches = scene.by_label(reference)
    if not matches:
        available = sorted({o.label for o in scene.objects.values() if o.label})
        raise ObjectNotFound(
            f"cannot find {reference!r}; currently visible: {available or 'nothing'}"
        )
    if len(matches) > 1:
        raise AmbiguousReference(
            f"{reference!r} matches {len(matches)} objects; say which one"
        )
    return matches[0].track_id


# ----------------------------------------------------------------------
# perception
# ----------------------------------------------------------------------


class Observe(Skill):
    """Perceive the scene and report it. Moves nothing."""

    skill_name = "observe"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        # Render before capturing: without a rendered frame the annotators hold
        # whatever they last produced.
        self.ctx.sim.render_step(2)
        scene = self.ctx.vision.observe()
        if self.ctx.memory is not None:
            self.ctx.memory.update_scene(scene)

        described = [
            {
                "track_id": o.track_id,
                "label": o.label,
                "position": np.round(o.pose.position, 4).tolist(),
                "size": np.round(o.bbox.extents, 4).tolist(),
                "confidence": round(o.confidence, 3),
            }
            for o in scene.objects.values()
        ]
        return self._ok(
            f"observed {len(described)} object(s)", objects=described, relations=scene.relations
        )


class ScanScene(Skill):
    """Sweep the wrist through several viewpoints, observing at each.

    A single viewpoint leaves occlusion shadows. Scanning from a few base
    rotations reveals objects hidden behind others, and each observation feeds the
    same tracker, so identities persist across the sweep.
    """

    skill_name = "scan_scene"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        sweep = float(params.get("sweep", 0.6))
        num_views = int(params.get("views", 3))

        home = np.asarray(self.ctx.config.robot.home_joint_positions, dtype=np.float64)
        offsets = np.linspace(-sweep, sweep, max(1, num_views))
        seen: dict[str, str] = {}

        for offset in offsets:
            target = home.copy()
            target[0] += float(offset)
            scene = self.ctx.vision.require_fresh_scene()
            trajectory = self.ctx.planner.plan_to_joint(
                self.ctx.robot.get_state().joint_state, target, scene
            )
            if trajectory is None:
                continue
            self.ctx.controller.follow_trajectory(trajectory)
            self.ctx.sim.render_step(3)
            observed = self.ctx.vision.observe()
            for track_id, obj in observed.objects.items():
                seen[track_id] = obj.label

        if self.ctx.memory is not None:
            self.ctx.memory.update_scene(self.ctx.vision.last_scene_graph())

        return self._ok(f"scanned {len(offsets)} viewpoint(s), found {len(seen)} object(s)",
                        objects=seen)


class LookAt(Skill):
    """Point the wrist camera at an object without approaching it."""

    skill_name = "look_at"

    def validate(self, params: dict[str, Any]) -> str | None:
        if not params.get("target"):
            return "look_at requires a target"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        track_id = _resolve_target(self.ctx, params)
        scene = self.ctx.vision.require_fresh_scene()
        obj = scene.objects[track_id]

        # Stand off above the object looking straight down: the most reliable
        # inspection pose, and the one that occludes least.
        standoff = float(params.get("distance", 0.25))
        view_position = obj.bbox.center.position + np.array([0.0, 0.0, standoff])
        view_pose = Pose(
            view_position,
            tf.matrix_to_quat(np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])),
            Frame.WORLD,
        )

        trajectory = self.ctx.planner.plan_to_pose(
            self.ctx.robot.get_state().joint_state, view_pose, scene
        )
        if trajectory is None:
            return self._infeasible(f"cannot reach a viewing pose for {obj.label!r}")
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted while moving to the viewing pose")

        self.ctx.sim.render_step(3)
        self.ctx.vision.observe()
        return self._ok(f"looking at {obj.label!r}", track_id=track_id)


# ----------------------------------------------------------------------
# motion
# ----------------------------------------------------------------------


class MoveTo(Skill):
    """Move the TCP to an absolute pose or a named object's standoff."""

    skill_name = "move_to"

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("position") is None and not params.get("target"):
            return "move_to requires either a position or a target"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        scene = self.ctx.vision.require_fresh_scene()
        current = self.ctx.robot.tcp_pose()

        if params.get("position") is not None:
            position = np.asarray(params["position"], dtype=np.float64)
        else:
            track_id = _resolve_target(self.ctx, params)
            obj = scene.objects[track_id]
            position = obj.bbox.center.position + np.array(
                [0.0, 0.0, float(params.get("standoff", 0.15))]
            )

        goal = Pose(position, np.asarray(params.get("quat", current.quat)), Frame.WORLD)
        trajectory = self.ctx.planner.plan_with_retries(
            self.ctx.robot.get_state().joint_state, goal, scene
        )
        if trajectory is None:
            return self._infeasible(
                f"no collision-free path to {np.round(position, 3).tolist()}"
            )
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted")

        achieved = self.ctx.robot.tcp_pose()
        return self._ok(
            f"moved to {np.round(achieved.position, 3).tolist()}",
            position=achieved.position.tolist(),
            error=achieved.translation_distance(goal),
        )


class MoveRelative(Skill):
    """Move the TCP by a Cartesian delta. "Move left" and nothing else."""

    skill_name = "move_relative"

    #: Direction words mapped to unit vectors in the robot's frame. The base sits
    #: at the origin facing +X, so the operator's "left" is +Y.
    DIRECTIONS = {
        "left": np.array([0.0, 1.0, 0.0]),
        "right": np.array([0.0, -1.0, 0.0]),
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, 0.0, 0.0]),
        "back": np.array([-1.0, 0.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([0.0, 0.0, -1.0]),
    }

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("delta") is None and not params.get("direction"):
            return "move_relative requires a delta or a direction"
        direction = params.get("direction")
        if direction and str(direction).lower() not in self.DIRECTIONS:
            return (
                f"unknown direction {direction!r}; "
                f"expected one of {sorted(self.DIRECTIONS)}"
            )
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        if params.get("delta") is not None:
            delta = np.asarray(params["delta"], dtype=np.float64)
        else:
            distance = float(params.get("distance", 0.10))
            delta = self.DIRECTIONS[str(params["direction"]).lower()] * distance

        current = self.ctx.robot.tcp_pose()
        goal = Pose(current.position + delta, current.quat, Frame.WORLD)
        scene = self.ctx.vision.require_fresh_scene()

        # Straight line: a relative move should go where it was told, not curve.
        trajectory = self.ctx.planner.plan_cartesian_line(
            self.ctx.robot.get_state().joint_state, goal, scene
        )
        if trajectory is None:
            return self._infeasible(
                f"cannot move by {np.round(delta, 3).tolist()}: outside reach or no IK"
            )
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted")

        achieved = self.ctx.robot.tcp_pose()
        return self._ok(
            f"moved by {np.round(achieved.position - current.position, 3).tolist()}",
            delta=(achieved.position - current.position).tolist(),
        )


class RotateWrist(Skill):
    """Rotate the last joint. Rotation only, no translation."""

    skill_name = "rotate_wrist"

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("angle") is None:
            return "rotate_wrist requires an angle in radians"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        angle = float(params["angle"])
        joints = self.ctx.robot.get_arm_joint_positions().copy()
        joints[-1] += angle

        scene = self.ctx.vision.require_fresh_scene()
        trajectory = self.ctx.planner.plan_to_joint(
            self.ctx.robot.get_state().joint_state, joints, scene
        )
        if trajectory is None:
            return self._infeasible(f"cannot rotate wrist by {angle:.3f} rad (joint limit?)")
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("rotation aborted")
        return self._ok(f"rotated wrist by {angle:.3f} rad")


class GoHome(Skill):
    """Return to the home posture by planning, not by teleporting."""

    skill_name = "go_home"

    #: Retreats tried, in order, when the arm cannot plan home from where it
    #: stands. Straight up first: it is the shortest escape and the one that
    #: works after an ordinary pick. Then higher. Then up-and-back, which is
    #: what frees the arm when the table edge rather than the height is what
    #: blocks the local configuration space -- the case a pure lift cannot fix,
    #: and the one that left the arm stranded after a place.
    _RETREAT_OFFSETS = (
        np.array([0.0, 0.0, 0.15]),
        np.array([0.0, 0.0, 0.25]),
        np.array([-0.10, 0.0, 0.20]),
        np.array([-0.18, 0.0, 0.10]),
    )

    def _run(self, params: dict[str, Any]) -> SkillResult:
        home = np.asarray(self.ctx.config.robot.home_joint_positions, dtype=np.float64)
        scene = self.ctx.vision.require_fresh_scene()
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        trajectory = planner.plan_to_joint(robot.get_state().joint_state, home, scene)
        if trajectory is None:
            trajectory = self._escape_then_plan_home(home)

        if trajectory is None:
            return self._infeasible(
                "no collision-free path home after "
                f"{len(self._RETREAT_OFFSETS)} retreat attempts"
            )
        if not controller.follow_trajectory(trajectory):
            return self._fail("motion home aborted")

        error = float(np.max(np.abs(robot.get_arm_joint_positions() - home)))
        return self._ok(f"at home posture (max joint error {error:.3f} rad)")

    def _escape_then_plan_home(self, home: NDArray[np.float64]) -> Any:
        """Retreat out of the blocked region, then re-plan the path home.

        Why a ladder rather than one lift. After a pick or place the arm sits
        low and close to the table, where enough of the local configuration
        space is blocked that RRT rejects the start state outright -- and a
        rejected *start* cannot be rescued by re-seeding the tree, only by
        moving somewhere else first. The previous version tried exactly one
        +150 mm lift and, if that single straight line had no IK solution, gave
        up: measured after a place, that left the arm stranded with "no
        collision-free path home" and only a restart to recover from.

        Each offset is attempted twice, straight line first. ``plan_cartesian_line``
        is preferred because it is short and predictable, but it solves IK at
        every waypoint and fails outright when any one of them has no solution.
        RRT can still route around that, so the same target is retried with
        ``plan_with_retries`` -- which also varies the IK branch, the fix that
        made the pregrasp reachable.

        Retreats accumulate: a partial escape that does not yet free the arm
        still leaves it somewhere better for the next offset to work from.
        """
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        for index, offset in enumerate(self._RETREAT_OFFSETS, start=1):
            tcp = robot.tcp_pose()
            target = Pose(
                _clamp_to_workspace(self.ctx, tcp.position + offset),
                tcp.quat,
                Frame.WORLD,
            )
            scene = self.ctx.vision.require_fresh_scene()

            retreat = planner.plan_cartesian_line(
                robot.get_state().joint_state, target, scene
            )
            if retreat is None:
                retreat = planner.plan_with_retries(
                    robot.get_state().joint_state, target, scene
                )
            if retreat is None:
                _log.debug("Retreat %d: no motion to %s", index, offset.tolist())
                continue
            if not controller.follow_trajectory(retreat):
                _log.debug("Retreat %d aborted mid-motion", index)
                continue

            scene = self.ctx.vision.require_fresh_scene()
            trajectory = planner.plan_to_joint(
                robot.get_state().joint_state, home, scene
            )
            if trajectory is not None:
                _log.info("Path home found after retreat %d %s", index, offset.tolist())
                return trajectory

        return None


# ----------------------------------------------------------------------
# gripper
# ----------------------------------------------------------------------


class OpenGripper(Skill):
    """Open the fingers. Opens only -- the arm does not move.

    Releases any held object as a physical consequence of opening, and updates
    memory accordingly, but performs no motion of its own.
    """

    skill_name = "open_gripper"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        width = self.ctx.controller.open_gripper_blocking()
        if self.ctx.memory is not None:
            self.ctx.memory.set_held_object(None)
        return self._ok(f"gripper open at {width * 1000:.1f} mm", width=width)


class CloseGripper(Skill):
    """Close the fingers. Closes only -- no approach, no lift, no verification.

    Reports whether the fingers stalled, which is evidence something is between
    them, but deliberately does not claim a grasp: that requires a lift, and
    lifting is ``Pick``'s job.
    """

    skill_name = "close_gripper"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        width = self.ctx.controller.close_gripper_blocking()
        stalled = width > self.ctx.config.robot.gripper_closed_width + 0.004
        return self._ok(
            f"gripper closed to {width * 1000:.1f} mm"
            + (" (stalled on an object)" if stalled else " (closed on air)"),
            width=width,
            stalled_on_object=stalled,
        )


# ----------------------------------------------------------------------
# pick and place
# ----------------------------------------------------------------------


class Pick(Skill):
    """Pick up an object and **hold** it.

    Sequence: observe, synthesise grasps, choose the best reachable one, plan to
    the pregrasp standoff, approach in a straight line, close with force limiting,
    lift, verify from sensor evidence, then hold and WAIT.

    Does **not** place. Nothing here triggers another skill.
    """

    skill_name = "pick"

    def validate(self, params: dict[str, Any]) -> str | None:
        if not params.get("target"):
            return "pick requires a target"
        if self.ctx.memory is not None and self.ctx.memory.get_held_object() is not None:
            return "already holding something; place or release it first"
        return None

    def _generate(self, obj: Any, scene: Any) -> list[Any]:
        """Grasp candidates for ``obj``, from the injected generator when there is one.

        The sim lane leaves ``ctx.grasp_generator`` unset and keeps the direct
        OBB synthesis. The hardware lane injects a top-down generator: its
        perception has no depth, so the "box" is a size-table entry, and a
        jaw with no wrist roll cannot meet an arbitrary box axis anyway.
        """
        generator = getattr(self.ctx, "grasp_generator", None)
        if generator is not None:
            return list(generator.generate(scene, obj.track_id))
        return generate_grasp_candidates(obj, self.ctx.config.grasp)

    def _run(self, params: dict[str, Any]) -> SkillResult:
        # 1. Perceive. Never act on a remembered pose.
        self.ctx.sim.render_step(2)
        scene = self.ctx.vision.observe()
        track_id = _resolve_target(self.ctx, params)
        obj = scene.objects[track_id]

        # 2. Synthesise and rank grasps from the perceived geometry.
        candidates = self._generate(obj, scene)
        hardware_generator = getattr(self.ctx, "grasp_generator", None) is not None
        narrow_enough = float(np.min(obj.bbox.extents[:2])) <= self.ctx.config.grasp.max_grasp_width
        if hardware_generator and narrow_enough and (getattr(obj, "attributes", None) or {}).get("yaw_ambiguous"):
            # Depthless perception only (the sim lane never sets it): the pixel
            # box fits neither the along-X nor the across orientation well
            # enough, so both the centre and the chord the candidates were
            # checked against are guesses. A wrong one closes the jaw on air,
            # or drives the fingers onto the body.
            return self._infeasible(
                f"cannot tell precisely which way the {obj.label} is lying (the camera sees it "
                "at an angle), so the jaw could miss it; turn it to point straight at the "
                "arm's base and ask again"
            )
        if not candidates and hardware_generator and narrow_enough:
            # Hardware lane: the jaw cannot rotate about the tool axis, so an
            # object narrow enough overall can still be too wide *across the
            # jaw* the way it is lying. Saying "19 mm, the gripper spans 35 mm"
            # there is a contradiction the student cannot act on.
            return self._infeasible(
                f"the {obj.label} is narrow enough ({np.min(obj.bbox.extents[:2]) * 1000:.0f} mm) "
                "but not lying the way this jaw closes (it cannot rotate); turn it to point "
                "straight at the arm's base and ask again"
            )
        if not candidates:
            return self._infeasible(
                f"{obj.label!r} is {np.min(obj.bbox.extents) * 1000:.0f} mm at its narrowest; "
                f"the gripper spans {self.ctx.config.grasp.max_grasp_width * 1000:.0f} mm"
            )

        state = self.ctx.robot.get_state()
        ranked = self.ctx.grasp_scorer.score(candidates, scene, state)
        if not ranked:
            return self._infeasible(
                f"generated {len(candidates)} grasp(s) for {obj.label!r} but none are "
                "reachable and collision-free"
            )

        self.ctx.emit(
            "pick.grasp_selected",
            {"target": track_id, "label": obj.label, "chosen": ranked[0].to_log(),
             "num_candidates": len(candidates), "num_feasible": len(ranked)},
        )

        # 3. Attempt the best few grasps in order. A failed approach is a normal
        #    outcome, not an error: the next candidate may well work.
        #
        # Every retry re-observes and re-plans from scratch, because a failed
        # attempt usually MOVES the object. Instrumented on the soup can: the
        # first attempt gripped it (fingers stalled at 60.3 mm on a 68 mm can),
        # lost it during the lift, and left it 34 mm from where it started. The
        # remaining candidates -- computed from the original observation --
        # then sent the gripper to empty space 100 mm away, twice, and the skill
        # reported "closed without contact".
        #
        # That message described the stale retries, not the real failure, and it
        # is what made this look like a perception centring problem for far
        # longer than it should have. A retry that plans against a pose the
        # previous attempt invalidated is not a second chance; it is a guaranteed
        # miss that also overwrites the useful error.
        last_error = "no grasp attempted"
        attempts = int(self.ctx.config.motion.max_replan_attempts)

        for attempt in range(attempts):
            if attempt == 0:
                grasp = ranked[0]
            else:
                # Reset before looking, so the arm is not occluding its own view.
                self.ctx.controller.open_gripper_blocking()
                fresh = self.ctx.vision.observe()
                moved = fresh.get(track_id)
                if moved is None:
                    return self._fail(
                        f"{last_error} (and {obj.label!r} is no longer visible to retry)"
                    )

                regenerated = self._generate(moved, fresh)
                reranked = self.ctx.grasp_scorer.score(
                    regenerated, fresh, self.ctx.robot.get_state()
                )
                if not reranked:
                    return self._fail(
                        f"{last_error} (no feasible grasp remains after re-observing)"
                    )
                idx = min(attempt, len(reranked) - 1)
                grasp, obj, scene = reranked[idx], moved, fresh

            outcome = self._attempt_grasp(grasp, obj, scene)
            if outcome.ok:
                return outcome
            if not self.ctx.config.robot.gripper_feedback and (
                outcome.data.get("verification_failed") or outcome.data.get("jaw_closed")
            ):
                # Without gripper feedback the verdict comes from pixels and can
                # be wrong in either direction, and the jaw is closed at lift
                # height. A retry would open it there -- dropping the object if
                # it *is* held -- and its own failure would then replace this
                # message with an unrelated one (review F1/F3). Stop here: the
                # verifier's reason is the answer, and a human decides.
                return self._infeasible(
                    f"{outcome.message}. The jaw is left closed in case "
                    f"{obj.label!r} is held: look at the gripper, then say 'open the gripper' "
                    "to release it",
                    **outcome.data,
                )
            last_error = outcome.message

        return self._fail(f"all grasp attempts failed: {last_error}")

    _MIN_RECENTRE_LIMIT_M = 0.05

    def _recentre_from_standoff(self, grasp: Any) -> Pose | None:
        """Re-measure the target from the pre-grasp and correct the grasp XY.

        Returns ``None`` when the correction should not be applied, which is the
        safe default: the original pose was at least scored for reachability and
        clearance, and a correction derived from a bad observation is worse than
        no correction.

        Declines when the target is no longer tracked -- the arm may have nudged
        it, and a grasp aimed at a stale track is worse than one aimed at a stale
        pose -- or when the shift is larger than the object itself.

        That second bound replaces a fixed 50 mm limit, which was arbitrary and
        rejected exactly the corrections it existed to enable: it fired eight
        times in one run, blocking shifts of 51 to 97 mm on the three objects
        still closing on air. Identity is not what a magnitude test protects
        here anyway -- the detection is fetched by ``track_id``, so the tracker
        has already established this is the same object.

        What is worth testing is physical plausibility. A correction larger than
        the object's own footprint cannot be a re-measurement of that object,
        whatever the tracker says, so the object's largest horizontal extent is
        the natural limit. The floor keeps small objects correctable at all.
        """
        try:
            fresh = self.ctx.vision.observe()
        except Exception as exc:  # noqa: BLE001 - fall back to the planned pose
            _log.debug("Re-observation at the standoff failed: %s", exc)
            return None

        target = fresh.get(grasp.target_track_id)
        if target is None:
            _log.debug(
                "Target %s not tracked from the standoff; keeping the planned grasp",
                grasp.target_track_id,
            )
            return None

        planned = np.asarray(grasp.pose.position, dtype=np.float64)
        corrected = planned.copy()
        corrected[:2] = np.asarray(target.bbox.center.position, dtype=np.float64)[:2]

        shift = float(np.linalg.norm(corrected[:2] - planned[:2]))
        limit = max(
            self._MIN_RECENTRE_LIMIT_M,
            float(np.max(np.asarray(target.bbox.extents, dtype=np.float64)[:2])),
        )
        if shift > limit:
            _log.warning(
                "Re-centring would move the grasp %.0f mm, more than the object's own "
                "%.0f mm footprint; keeping the planned pose",
                shift * 1000.0,
                limit * 1000.0,
            )
            return None

        if shift > 1e-4:
            _log.info("Re-centred the grasp by %.1f mm from the standoff view", shift * 1000.0)
        return Pose(corrected, grasp.pose.quat, Frame.WORLD)

    def _reference_view(self, track_id: str, planning_scene: Any) -> Any:
        """The pre-descent scene the feedback-less verdict compares against.

        Observed now, from the standoff, before the jaw descends. If the arm
        at the standoff already hides the target, the scene the grasp was
        planned from (taken before the arm moved over it) is used instead,
        provided the target was actually detected in it. Either way the
        verifier checks the target was seen *in that very observation*, so a
        track the tracker merely kept alive never becomes a reference.
        """
        try:
            view = self.ctx.vision.observe()
        except Exception as exc:  # noqa: BLE001 - fall back to the planning view
            _log.warning("Pre-descent observation failed (%s); using the planning view", exc)
            return planning_scene
        target = view.get(track_id)
        if target is not None and target.last_seen_step >= view.step_index:
            return view
        planned = planning_scene.get(track_id)
        if planned is not None and planned.last_seen_step >= planning_scene.step_index:
            _log.info("Target hidden from the standoff; the planning view is the pre-descent reference")
            return planning_scene
        return view

    def _lift_prediction(self, scene_before: Any, track_id: str) -> Any:
        """What the target should look like now if it is held (``None``: no predictor)."""
        predictor = getattr(self.ctx.vision, "predict_lift", None)
        target = scene_before.get(track_id) if scene_before is not None else None
        if predictor is None or target is None:
            return None
        try:
            return predictor(target, self.ctx.robot.tcp_pose())
        except Exception as exc:  # noqa: BLE001 - verdict falls back to bare pixel growth
            _log.warning("Lift prediction failed (%s); verifying from pixel growth alone", exc)
            return None

    def _attempt_grasp(self, grasp: Any, obj: Any, scene: Any) -> SkillResult:
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        # The target must not be an obstacle while we deliberately close on it.
        planner.exclude_from_collision(grasp.target_track_id)
        try:
            controller.open_gripper_blocking()

            # 4. Plan to the standoff with full collision avoidance.
            approach_plan = planner.plan_with_retries(
                robot.get_state().joint_state, grasp.pregrasp_pose, scene
            )
            if approach_plan is None:
                return self._fail("no path to the pregrasp standoff")
            if not controller.follow_trajectory(approach_plan):
                return self._fail("motion to the pregrasp standoff aborted")

            feedback = bool(self.ctx.config.robot.gripper_feedback)
            before_descent = None
            if not feedback:
                # Without gripper feedback the verdict compares the object's
                # pixel box before and after. Observed after the descent, the
                # arm is parked over the object and a small one is already
                # hidden -- its later absence then proves nothing (review:
                # "before scene observed with the arm over the object"). So
                # the reference is taken here, from the standoff, and the sim
                # lane keeps its post-descent observation below.
                before_descent = self._reference_view(grasp.target_track_id, scene)

            # 4b. Re-observe from the standoff and re-centre the grasp.
            #
            # The pose planned from is measured from wherever the object first
            # became visible, often an oblique exterior view. Measured on the
            # benchmark scene, that carries up to 71 mm of centre error -- more
            # than the soup can is wide -- so the fingers close on empty air
            # beside the object. The errors track visibility exactly:
            # exterior-only objects come in at 71/53/43/24 mm, objects seen by
            # both cameras at 22/22/18/14/3 mm.
            #
            # Here the wrist camera sits ~100 mm directly above the target,
            # looking straight down: minimal occlusion, maximal resolution, no
            # oblique foreshortening. It is the best view this robot will ever
            # get of the object, and it is free -- the arm is already parked.
            #
            # Only the horizontal centre is taken. Height, orientation and grasp
            # width come from the original candidate, because those were what
            # the scorer checked for reachability and finger clearance, and
            # changing them here would invalidate that reasoning. Correcting
            # *where* the fingers close without changing *how* is the whole
            # point.
            grasp_pose = grasp.pose
            if self.ctx.config.grasp.recentre_from_standoff:
                # Off on the hardware lane: one fixed overhead camera sees
                # the same thing from the standoff as it did before.
                recentred = self._recentre_from_standoff(grasp)
                if recentred is not None:
                    grasp_pose = recentred

            # 5. Straight-line approach. A curved path would sweep the fingers
            #    sideways through the object.
            descent = planner.plan_cartesian_line(
                robot.get_state().joint_state, grasp_pose, scene
            )
            if descent is None:
                return self._fail("no straight-line approach to the grasp pose")
            if not controller.follow_trajectory(descent):
                return self._fail("approach aborted")

            if feedback:
                scene_before_lift = self.ctx.vision.observe()
            else:
                scene_before_lift = before_descent

            # 6. Force-limited close. Real contact, real friction. A controller
            #    that can close to a width (hobby servos, no force limit) is
            #    told the chord the candidate was checked against, so the jaw
            #    squeezes the object instead of stalling against it.
            if hasattr(self.ctx.controller, "set_grasp_width"):
                self.ctx.controller.set_grasp_width(grasp.width)
            closed_width = controller.close_gripper_blocking()

            # 7. Lift straight up, re-asserting the grip every step. Without
            #    re-assertion the fingers relax and drop the load mid-lift.
            lift_height = float(self.ctx.config.grasp.lift_height)
            grasp_pose_now = robot.tcp_pose()
            lift_target = Pose(
                _clamp_to_workspace(
                    self.ctx, grasp_pose_now.position + np.array([0.0, 0.0, lift_height])
                ),
                grasp_pose_now.quat,
                Frame.WORLD,
            )
            lift = planner.plan_cartesian_line(
                robot.get_state().joint_state, lift_target, scene
            )
            if lift is None:
                return self._fail("cannot lift from the grasp pose")

            def hold_grip(_index: int, _positions: NDArray[np.float64]) -> bool:
                controller.maintain_grasp()
                return True

            if not controller.follow_trajectory(lift, on_step=hold_grip):
                if feedback:
                    return self._fail("lift aborted")
                # Stopped part-way up with the jaw closed on whatever it
                # holds: the retry's open would drop it (see _run).
                return self._fail("lift aborted", jaw_closed=True)

            # 8. Verify from evidence, not from having issued the commands.
            self.ctx.sim.render_step(3)
            scene_after = self.ctx.vision.observe()
            extra: dict[str, Any] = {}
            prediction = None
            if not feedback:
                prediction = self._lift_prediction(scene_before_lift, grasp.target_track_id)
                extra["lift_prediction"] = prediction
            evidence = verify_grasp(
                robot=robot,
                scene_before=scene_before_lift,
                scene_after=scene_after,
                track_id=grasp.target_track_id,
                closed_width=self.ctx.config.robot.gripper_closed_width,
                expected_height_gain=lift_height,
                gripper_feedback=self.ctx.config.robot.gripper_feedback,
                min_displacement=self.ctx.config.grasp.verify_min_displacement,
                **extra,
            )
            payload = {"target": grasp.target_track_id, **evidence.to_log()}
            if not feedback:
                payload["before_view"] = "planning" if scene_before_lift is scene else "standoff"
            if prediction is not None:
                payload["lift_prediction"] = prediction.to_log()
            self.ctx.emit("pick.verification", payload)

            if not evidence.holding:
                if feedback:
                    return self._fail(evidence.reason())
                return self._fail(
                    evidence.reason(), verification_failed=True, evidence=evidence.to_log()
                )

            if self.ctx.memory is not None:
                self.ctx.memory.set_held_object(grasp.target_track_id)

            # 9. Hold and WAIT. No place, no return home.
            controller.maintain_grasp()
            return self._ok(
                f"holding {obj.label!r} ({evidence.reason()})",
                track_id=grasp.target_track_id,
                label=obj.label,
                grasp=grasp.to_log(),
                evidence=evidence.to_log(),
            )
        finally:
            planner.include_in_collision(grasp.target_track_id)


class Place(Skill):
    """Place the held object and **stop**.

    Sequence: re-observe the scene, locate the destination, synthesise a release
    pose above it, move there (using a top-down wrist orientation for maximum IK
    reachability), lower, open, retreat, then WAIT.

    Does **not** return home. Requires something to be held: "place it" with an
    empty gripper is a user error, reported as such.

    Orientation policy for placement transit
    ----------------------------------------
    After a pick the arm holds a wrist orientation suited to the grasp it just
    completed. When the destination lies at a significantly different XY position
    (e.g. block at x=0.45 picked with a side-approach, box at x=0.62 y=0.28),
    the carry quaternion frequently has no IK solution above the destination:
    the elbow must reconfigure and Lula cannot find a path.

    A top-down approach (TCP +Z pointing world -Z) is the most IK-reachable
    orientation on a tabletop -- it is the configuration the arm is in at the
    home posture. Trying it first, with the carry orientation as a fallback,
    covers the overwhelming majority of real placements without abandoning the
    carry posture for the rare case where top-down is itself infeasible.
    """

    skill_name = "place"

    # Top-down TCP orientation: approach axis points world -Z (downward).
    # Rotation matrix: X=[-1,0,0], Y=[0,-1,0], Z=[0,0,-1] => quat [0,1,0,0].
    # This is a 180-degree rotation about world X, the standard overhead-grasp
    # orientation for a Franka Panda and the most IK-reachable placement posture.
    _TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    def validate(self, params: dict[str, Any]) -> str | None:
        if self.ctx.memory is None:
            return "place requires memory to know what is held"
        if self.ctx.memory.get_held_object() is None:
            return "not holding anything to place"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        held_id = self.ctx.memory.get_held_object()

        # Re-observe before computing the destination pose. After a pick the arm
        # has moved; the pre-pick scene graph is stale and the target object may
        # have shifted (knocked during approach, or just tracked with drift).
        self.ctx.sim.render_step(2)
        scene = self.ctx.vision.observe()

        release_pose, description = self._destination_pose(params, scene, held_id)
        if release_pose is None:
            return self._infeasible(description)

        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        # The carried object must not be an obstacle to its own placement.
        planner.exclude_from_collision(held_id)
        try:
            def hold_grip(_index: int, _positions: NDArray[np.float64]) -> bool:
                controller.maintain_grasp()
                return True

            # Build transit-above with a top-down orientation first, then fall
            # back to the carry quaternion. Top-down is the most IK-reachable
            # orientation on a tabletop: the arm reaches it from home without
            # reconfiguring through table-level clearance. The carry quat is
            # preserved as a fallback because occasionally the destination is
            # placed where a top-down approach collides (e.g. placing inside a
            # deep container requires the arm to enter at an angle).
            above_position = _clamp_to_workspace(
                self.ctx, release_pose.position + np.array([0.0, 0.0, 0.08])
            )

            dest_x, dest_y = release_pose.position[0], release_pose.position[1]
            bearing = np.arctan2(dest_y, dest_x)

            # Candidate orientations: pure top-down first, angled approaches
            # along the radial direction for distant targets, and carry quat as fallback.
            candidate_quats = [self._TOP_DOWN_QUAT]
            for pitch_deg in (60.0, 45.0):
                rad_pitch = np.radians(pitch_deg)
                z_approach = np.array([
                    np.cos(bearing) * np.cos(rad_pitch),
                    np.sin(bearing) * np.cos(rad_pitch),
                    -np.sin(rad_pitch),
                ])
                y_axis = np.array([-np.sin(bearing), np.cos(bearing), 0.0])
                x_axis = np.cross(y_axis, z_approach)
                R = np.stack([x_axis, y_axis, z_approach], axis=1)
                candidate_quats.append(tf.matrix_to_quat(R))
            candidate_quats.append(robot.tcp_pose().quat)

            transit = None
            chosen_above = None
            any_ik_reachable = False

            for cand_quat in candidate_quats:
                above_cand = Pose(above_position, cand_quat, Frame.WORLD)
                if robot.inverse_kinematics(above_cand) is None:
                    continue
                any_ik_reachable = True
                transit = planner.plan_with_retries(robot.get_state().joint_state, above_cand, scene)
                if transit is None:
                    transit = planner.plan_cartesian_line(
                        robot.get_state().joint_state, above_cand, scene
                    )
                if transit is not None:
                    chosen_above = above_cand
                    break

            if transit is None:
                return self._infeasible(
                    f"cannot reach a position above {description}: "
                    + (
                        "no collision-free path"
                        if any_ik_reachable
                        else f"pose {np.round(above_position, 3).tolist()} has no IK solution"
                    )
                )

            if not controller.follow_trajectory(transit, on_step=hold_grip):
                return self._fail("transit to the destination aborted")

            # Lower straight down.
            lower_target = Pose(release_pose.position, robot.tcp_pose().quat, Frame.WORLD)
            lower = planner.plan_cartesian_line(
                robot.get_state().joint_state, lower_target, scene
            )
            if lower is None:
                lower = planner.plan_with_retries(
                    robot.get_state().joint_state, lower_target, scene
                )
            if lower is None:
                return self._infeasible("cannot lower to the release height")
            if not controller.follow_trajectory(lower, on_step=hold_grip):
                return self._fail("lowering aborted")

            # Release. The object settles under gravity, on its own.
            controller.open_gripper_blocking()
            self.ctx.sim.step(self.ctx.config.simulation.settle_steps)

            # Retreat straight up so the fingers clear what was just placed.
            retreat_target = Pose(
                _clamp_to_workspace(
                    self.ctx, robot.tcp_pose().position + np.array([0.0, 0.0, 0.12])
                ),
                robot.tcp_pose().quat,
                Frame.WORLD,
            )
            retreat = planner.plan_cartesian_line(
                robot.get_state().joint_state, retreat_target, scene
            )
            if retreat is not None:
                controller.follow_trajectory(retreat)

            self.ctx.memory.set_held_object(None)

            # Confirm by perception that the object is where it was put.
            self.ctx.sim.render_step(3)
            after = self.ctx.vision.observe()
            placed = after.get(held_id)
            if self.ctx.config.robot.gripper_feedback:
                settled = (
                    float(np.linalg.norm(placed.pose.position - release_pose.position))
                    if placed is not None
                    else float("inf")
                )
                return self._ok(
                    f"placed at {description}"
                    + (f" (settled {settled * 1000:.0f} mm from the target)" if placed else ""),
                    track_id=held_id,
                    release_position=release_pose.position.tolist(),
                    settled_offset=settled,
                )

            # Depthless lane: the estimator pins every object's z to the table
            # (support + half its size-table height), so a marker placed in a
            # bowl reads ~70 mm "below" a release height of 0.08 m -- a vertical
            # miss that never happened. Only the table-plane offset is measured.
            settled = (
                float(np.linalg.norm(placed.pose.position[:2] - release_pose.position[:2]))
                if placed is not None
                else float("inf")
            )
            return self._ok(
                f"placed at {description}"
                + (f" (settled {settled * 1000:.0f} mm from the target, horizontally)" if placed else ""),
                track_id=held_id,
                release_position=release_pose.position.tolist(),
                settled_offset=settled,
                settled_offset_axes="xy",
            )
        finally:
            planner.include_in_collision(held_id)

    def _destination_pose(
        self, params: dict[str, Any], scene: Any, held_id: str
    ) -> tuple[Pose | None, str]:
        """Work out where to release, from an explicit point or a named target."""
        clearance = float(self.ctx.config.grasp.place_clearance)
        current_quat = self.ctx.robot.tcp_pose().quat

        if params.get("position") is not None:
            position = np.asarray(params["position"], dtype=np.float64)
            return Pose(position, current_quat, Frame.WORLD), (
                f"{np.round(position, 3).tolist()}"
            )

        if params.get("target"):
            try:
                destination_id = _resolve_target(self.ctx, params)
            except (ObjectNotFound, AmbiguousReference) as exc:
                return None, str(exc)
            if destination_id == held_id:
                return None, "the destination is the object being held"

            destination = scene.objects[destination_id]
            relation = str(params.get("relation", "on")).lower()

            if relation in ("in", "inside"):
                # Release just above the rim so the object drops in rather than
                # being pressed against the base.
                top = destination.bbox.center.position[2] + destination.bbox.extents[2] / 2.0
                position = np.array(
                    [
                        destination.bbox.center.position[0],
                        destination.bbox.center.position[1],
                        top + clearance,
                    ]
                )
            elif relation in ("next_to", "beside", "next"):
                offset = destination.bbox.extents[1] / 2.0 + 0.08
                position = destination.bbox.center.position + np.array([0.0, offset, 0.0])
                position[2] = self.ctx.support_height + clearance + 0.03
            else:  # "on"
                top = destination.bbox.center.position[2] + destination.bbox.extents[2] / 2.0
                position = np.array(
                    [
                        destination.bbox.center.position[0],
                        destination.bbox.center.position[1],
                        top + clearance + 0.03,
                    ]
                )
            # Aim the OBJECT at the destination, not the hand.
            #
            # Everything above positions the TCP over the destination's centre,
            # which is only the same thing when the object sits exactly at the
            # fingertips. It does not. Measured while held, the marker's centre
            # is 22-26 mm from the TCP, and because a 118 mm marker is gripped
            # near one end that offset is mostly sideways. Put the hand over the
            # middle of a 172 mm bowl and the marker itself hangs over the rim,
            # which is where it lands.
            #
            # Compensating means the release point is wherever the hand has to
            # be for the *object* to end up on target. Only XY is corrected: the
            # height already comes from the destination's own surface, and the
            # perceived z of a gripper-occluded object is not worth trusting.
            position = self._aim_object_not_hand(position, scene, held_id)
            return Pose(position, current_quat, Frame.WORLD), (
                f"{relation} {destination.label!r}"
            )

        # No destination given: put it back where it was picked up.
        #
        # Releasing at the hand's current XY looks equivalent and is not. The
        # hand is wherever the last command left it -- after "move up 10 cm" it
        # has also drifted a centimetre or two sideways, and after a re-centred
        # grasp it is offset from the object's origin. Measured on the mug: the
        # release landed far enough off that the mug rolled out of the scene
        # entirely and knocked the bottle flat on its way, turning one
        # successful pick into two displaced objects and a "success" log.
        #
        # The pick origin is recovered from the scene memory snapshotted when
        # the object was grasped, so "place it" is the inverse of "pick it up".
        # Height still comes from the support surface rather than the recorded
        # centre: perception measures a held object through a partly occluding
        # gripper, and trusting that z would release it into the table.
        origin_xy: NDArray[np.float64] | None = None
        picked_from = (
            self.ctx.memory.scene_at_pick() if self.ctx.memory is not None else None
        )
        if picked_from is not None:
            resting = picked_from.get(held_id)
            if resting is not None:
                origin_xy = np.asarray(resting.bbox.center.position[:2], dtype=np.float64)

        if origin_xy is None:
            # No usable snapshot (picked before the first observe, or the track
            # was lost). The hand's position is a worse answer, but it is the
            # only one left, and refusing to place would strand the object.
            origin_xy = np.asarray(self.ctx.robot.tcp_pose().position[:2], dtype=np.float64)
            where = "the current position"
        else:
            where = "where it was picked up"

        position = np.array(
            [
                origin_xy[0],
                origin_xy[1],
                self.ctx.support_height + clearance + self._held_half_height(scene, held_id),
            ]
        )
        return Pose(position, current_quat, Frame.WORLD), where

    #: Ignore a compensation larger than this. The held object's pose comes from
    #: perception looking past a gripper that occludes it, so a bad frame can
    #: put its "centre" somewhere absurd. Beyond this the offset is not a grasp
    #: offset, it is a mis-detection, and shifting the release by it would throw
    #: the object further than doing nothing at all.
    _MAX_AIM_CORRECTION_M = 0.12

    def _aim_object_not_hand(
        self, position: NDArray[np.float64], scene: Any, held_id: str
    ) -> NDArray[np.float64]:
        """Shift a destination so the held object, not the TCP, lands on it.

        Not on the depthless lane (``robot.gripper_feedback`` false): there
        the held object's perceived centre is the table-plane point its
        *lifted* box maps to, parallax-shifted 7-29 mm from where it really
        hangs, so the "offset" is a camera artefact. Measured on the honest
        fake lane: a marker held dead centre read 28 mm off, the release was
        moved 28 mm the wrong way and the lowering had no plan. The top-down
        grasp there is aimed at the object's centre, so no correction is the
        better estimate.
        """
        if not self.ctx.config.robot.gripper_feedback:
            return position
        held = scene.get(held_id)
        if held is None:
            return position

        tcp = self.ctx.robot.tcp_pose()
        offset = np.asarray(held.bbox.center.position, dtype=np.float64)[:2] - np.asarray(
            tcp.position, dtype=np.float64
        )[:2]

        magnitude = float(np.linalg.norm(offset))
        if magnitude > self._MAX_AIM_CORRECTION_M:
            _log.debug(
                "Ignoring %.0f mm aim correction for %s: larger than a plausible "
                "grasp offset, so the held pose is probably a mis-detection",
                magnitude * 1000.0,
                held_id,
            )
            return position

        corrected = np.array(position, dtype=np.float64)
        corrected[:2] -= offset
        if magnitude > 0.005:
            _log.info(
                "Aim corrected by %.0f mm so the object lands on target, not the hand",
                magnitude * 1000.0,
            )
        return corrected

    def _held_half_height(self, scene: Any, held_id: str, fallback: float = 0.03) -> float:
        """Half the height of the held object, from perception.

        The TCP sits roughly at the object's middle, so releasing at
        ``surface + clearance`` alone would drop the object from half its own
        height -- enough for a tall object to topple or bounce away from where it
        was meant to go. Using the measured extent releases it just clear of the
        surface instead of dropping it.
        """
        obj = scene.get(held_id) if scene is not None else None
        if obj is None:
            return fallback
        return float(obj.bbox.extents[2]) / 2.0


# ----------------------------------------------------------------------
# control
# ----------------------------------------------------------------------


class Wait(Skill):
    """Do nothing for a while, letting physics settle."""

    skill_name = "wait"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        duration = float(params.get("duration", 1.0))
        steps = max(1, int(duration / self.ctx.config.simulation.physics_dt))
        self.ctx.sim.step(steps)
        return self._ok(f"waited {duration:.2f} s")


class Stop(Skill):
    """Halt motion, keeping the arm actively controlled.

    Does not release: dropping whatever is held would be a new action, and this
    skill was only asked to stop.
    """

    skill_name = "stop"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        self.ctx.controller.stop()
        return self._aborted("stopped; holding position")


class EmergencyStop(Skill):
    """Halt immediately and zero velocities. Still holds, rather than dropping."""

    skill_name = "emergency_stop"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        self.ctx.controller.emergency_stop()
        return self._aborted("emergency stop engaged")


#: Every skill class, for registry construction.
ALL_SKILLS: tuple[type[Skill], ...] = (
    Observe,
    ScanScene,
    LookAt,
    MoveTo,
    MoveRelative,
    Pick,
    Place,
    OpenGripper,
    CloseGripper,
    RotateWrist,
    GoHome,
    Wait,
    Stop,
    EmergencyStop,
)

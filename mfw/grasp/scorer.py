"""Grasp candidate scoring and feasibility filtering.

Pure NumPy plus an :class:`~mfw.core.interfaces.IRobot` for kinematics; no Isaac
Sim import. The robot is injected, so scoring is testable against a fake.

Separated from generation because the two ask different questions. Generation
asks "does this geometry admit a grasp?" -- a property of the object alone.
Scoring asks "can *this robot* reach it without hitting anything?" -- which needs
kinematics and the rest of the scene. Keeping them apart means the geometric part
stays cheap and the expensive IK is only paid on candidates worth checking.

Hard rejections happen before soft scoring. A candidate that would drive the hand
through the table or has no IK solution is not a low-quality grasp, it is not a
grasp at all, and averaging it into a weighted score would let a high approach
score resurrect it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import GraspConfig
from mfw.core.interfaces import IGraspScorer, IRobot
from mfw.core.types import GraspCandidate, RobotState, SceneGraph
from mfw.grasp.generator import WORLD_DOWN
from mfw.utils.logging import get_logger

__all__ = ["GraspScorer"]

_log = get_logger("grasp.scorer")


class GraspScorer(IGraspScorer):
    """Ranks grasp candidates by reachability, approach quality and clearance."""

    def __init__(
        self,
        robot: IRobot,
        config: GraspConfig,
        support_height: float = 0.0,
        hand_setback: float = 0.1034,
    ) -> None:
        """``hand_setback`` is the distance from the fingertip midpoint (the TCP)
        back to the hand origin -- 103.4 mm on the Franka, measured via Lula FK.
        It defines how much gripper body trails behind the fingertips."""
        config.validate()
        self.robot = robot
        self.config = config
        self.support_height = support_height
        self.hand_setback = hand_setback

    def score(
        self,
        candidates: list[GraspCandidate] | tuple[GraspCandidate, ...],
        scene: SceneGraph,
        robot_state: RobotState,
    ) -> list[GraspCandidate]:
        """Return feasible candidates, best first.

        Infeasible candidates are dropped entirely rather than scored low.
        """
        scored: list[GraspCandidate] = []
        rejected: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        for candidate in candidates:
            # --- hard rejections, cheapest first ---
            if candidate.pose.position[2] < self.support_height + 0.002:
                reject("below_support")
                continue

            if not self._fingers_clear_support(candidate):
                reject("fingers_hit_support")
                continue

            if self._collides_with_other_objects(candidate, scene):
                reject("collision_with_other_object")
                continue

            grasp_joints = self.robot.inverse_kinematics(
                candidate.pose, seed=robot_state.joint_state.positions
            )
            if grasp_joints is None:
                reject("grasp_unreachable")
                continue

            pregrasp_joints = self.robot.inverse_kinematics(
                candidate.pregrasp_pose, seed=grasp_joints
            )
            if pregrasp_joints is None:
                reject("pregrasp_unreachable")
                continue

            # --- soft scoring ---
            breakdown = self._score_components(candidate, scene, robot_state, pregrasp_joints)
            total = float(np.clip(sum(breakdown.values()) / len(breakdown), 0.0, 1.0))
            if total < self.config.min_score:
                reject("below_min_score")
                continue

            scored.append(
                GraspCandidate(
                    pose=candidate.pose,
                    pregrasp_pose=candidate.pregrasp_pose,
                    approach_axis=candidate.approach_axis,
                    width=candidate.width,
                    score=total,
                    target_track_id=candidate.target_track_id,
                    scores_breakdown=breakdown,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        if rejected:
            _log.debug("Grasp rejections: %s", rejected)
        return scored

    # ------------------------------------------------------------------
    # hard feasibility
    # ------------------------------------------------------------------

    def _fingers_clear_support(self, candidate: GraspCandidate) -> bool:
        """Whether the whole gripper body stays above the support surface.

        The TCP *is* the fingertip midpoint, so the fingers and hand extend
        **backwards** from it, opposite the approach direction -- not below it.
        For a top-down grasp that puts the entire gripper above the TCP, and the
        only requirement is that the fingertips clear the table.

        The case this actually catches is a **low side grasp**: approaching
        horizontally at a height near the tabletop places the hand body beside
        the object at the same height, where it strikes the worktop even though
        the fingertips are clear. Sampling the segment from the fingertips back to
        the hand origin covers both cases with one test.
        """
        approach = np.asarray(candidate.approach_axis, dtype=np.float64)
        tcp = np.asarray(candidate.pose.position, dtype=np.float64)
        # The hand origin sits this far back along the approach axis.
        hand_origin = tcp - approach * self.hand_setback

        heights = np.linspace(tcp[2], hand_origin[2], 5)
        # Account for hand half-thickness (35 mm) on horizontal approach components,
        # where the fingers and palm extend below the centerline toward the table.
        horizontal_extent = float(np.sqrt(max(0.0, 1.0 - approach[2] ** 2)))
        hand_half_thickness = 0.035 * horizontal_extent
        return bool(np.all(heights - hand_half_thickness > self.support_height - 0.002))

    def _collides_with_other_objects(
        self, candidate: GraspCandidate, scene: SceneGraph
    ) -> bool:
        """Whether the hand would strike a *different* object on its way in.

        The target itself is excluded: converging on it is the entire point, and
        treating it as an obstacle would reject every valid grasp.

        Samples along the approach segment against inflated bounding boxes. An
        approximation, but the grasp only has to be good enough for the motion
        planner to verify properly afterwards.
        """
        start = np.asarray(candidate.pregrasp_pose.position, dtype=np.float64)
        end = np.asarray(candidate.pose.position, dtype=np.float64)
        samples = np.linspace(
            start, end, max(2, self.config.collision_check_samples)
        )

        for track_id, obj in scene.objects.items():
            if track_id == candidate.target_track_id:
                continue

            center = np.asarray(obj.bbox.center.position, dtype=np.float64)
            rotation = obj.bbox.center.rotation_matrix()
            half = np.asarray(obj.bbox.extents, dtype=np.float64) / 2.0 + 0.02

            local = (samples - center) @ rotation
            if np.any(np.all(np.abs(local) <= half, axis=1)):
                return True

        return False

    # ------------------------------------------------------------------
    # soft scoring
    # ------------------------------------------------------------------

    def _score_components(
        self,
        candidate: GraspCandidate,
        scene: SceneGraph,
        robot_state: RobotState,
        pregrasp_joints: NDArray[np.float64],
    ) -> dict[str, float]:
        """Individual quality terms, each in ``[0, 1]``."""
        return {
            "width_margin": self._width_margin_score(candidate),
            "approach_alignment": self._approach_score(candidate),
            "wrist_travel": self._wrist_travel_score(robot_state, pregrasp_joints),
            "centroid_offset": self._centroid_score(candidate, scene),
            "clearance": self._clearance_score(candidate, scene),
        }

    def _width_margin_score(self, candidate: GraspCandidate) -> float:
        """Prefer a comfortable margin between object width and gripper span.

        A grasp at the very limit of the gripper leaves nothing for perception
        error, and the extents are measured from a partial point cloud. A grasp
        far too small for the gripper wastes finger travel and closes slowly.
        """
        usable = self.config.max_grasp_width - self.config.finger_width_margin
        if usable <= self.config.min_grasp_width:
            return 0.0
        # Peak at ~55% of usable span, tapering either side.
        ideal = 0.55 * usable
        deviation = abs(candidate.width - ideal) / usable
        return float(np.clip(1.0 - deviation, 0.0, 1.0))

    def _approach_score(self, candidate: GraspCandidate) -> float:
        """Favour top-down approaches, weighted by ``top_grasp_bias``.

        Top-down is the most reachable direction on a table and keeps the elbow
        clear of the worktop. Side grasps stay available -- they are necessary for
        tall or wide objects -- just ranked below.
        """
        alignment = float(np.dot(np.asarray(candidate.approach_axis), WORLD_DOWN))
        downward = (alignment + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
        return float(
            np.clip(
                self.config.top_grasp_bias * downward + (1.0 - self.config.top_grasp_bias) * 0.5,
                0.0,
                1.0,
            )
        )

    def _wrist_travel_score(
        self, robot_state: RobotState, pregrasp_joints: NDArray[np.float64]
    ) -> float:
        """Prefer grasps requiring less joint motion from the current posture.

        Shorter reconfigurations are faster, less likely to sweep through the
        scene, and less likely to hit a joint limit mid-motion.
        """
        delta = np.abs(np.asarray(pregrasp_joints) - robot_state.joint_state.positions)
        travel = float(np.max(delta))
        return float(np.clip(1.0 - travel / np.pi, 0.0, 1.0))

    def _centroid_score(self, candidate: GraspCandidate, scene: SceneGraph) -> float:
        """Prefer grasping near the object's centroid.

        Gripping far from the centre of mass lets the object pivot in the fingers
        once lifted, which is a common cause of a grasp that holds initially and
        fails during transport.
        """
        obj = scene.get(candidate.target_track_id)
        if obj is None:
            return 0.5
        offset = float(
            np.linalg.norm(candidate.pose.position - obj.bbox.center.position)
        )
        scale = max(float(np.max(obj.bbox.extents)), 1e-3)
        return float(np.clip(1.0 - offset / scale, 0.0, 1.0))

    def _clearance_score(self, candidate: GraspCandidate, scene: SceneGraph) -> float:
        """Prefer grasps with room around them.

        Distance from the pregrasp position to the nearest other object, so that
        two adjacent objects are approached from the open side rather than the
        crowded one.
        """
        position = np.asarray(candidate.pregrasp_pose.position, dtype=np.float64)
        distances = [
            float(np.linalg.norm(position - np.asarray(obj.bbox.center.position)))
            for track_id, obj in scene.objects.items()
            if track_id != candidate.target_track_id
        ]
        if not distances:
            return 1.0
        return float(np.clip(min(distances) / 0.3, 0.0, 1.0))

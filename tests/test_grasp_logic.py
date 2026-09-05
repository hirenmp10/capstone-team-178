"""Phase 4 pure-logic tests: grasp synthesis and scoring.

No Isaac Sim. The scorer takes an IRobot, so a fake with controllable IK lets us
test reachability filtering deterministically -- something that is very hard to
arrange in simulation, where you cannot ask for "IK that fails exactly here".
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.config.schema import GraspConfig
from mfw.core.interfaces import IRobot
from mfw.core.types import (
    BoundingBox3D,
    Frame,
    GripperState,
    JointState,
    ObjectHypothesis,
    Pose,
    RobotState,
    SceneGraph,
)
from mfw.grasp.generator import build_grasp_pose, generate_grasp_candidates
from mfw.grasp.scorer import GraspScorer
from mfw.utils import transforms as tf

pytestmark = pytest.mark.phase4

TABLE_Z = 0.40


class FakeRobot(IRobot):
    """Minimal IRobot whose IK behaviour is fully controllable."""

    def __init__(self, reachable=lambda pose: True, dof=7):
        self._reachable = reachable
        self._dof = dof
        self.ik_calls = 0

    @property
    def joint_names(self):
        return tuple(f"j{i}" for i in range(self._dof))

    def get_state(self):
        return _robot_state(self._dof)

    def tcp_pose(self):
        return Pose.identity()

    def forward_kinematics(self, joint_positions):
        return Pose.identity()

    def inverse_kinematics(self, target, seed=None):
        self.ik_calls += 1
        if not self._reachable(target):
            return None
        return np.zeros(self._dof)

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass


def _robot_state(dof=7):
    return RobotState(
        joint_state=JointState(positions=np.zeros(dof), names=tuple(f"j{i}" for i in range(dof))),
        tcp_pose=Pose(np.array([0.4, 0.0, 0.6]), np.array([1.0, 0, 0, 0]), Frame.WORLD),
        gripper=GripperState(width=0.08, target_width=0.08, is_moving=False, is_grasping=False),
        sim_time=0.0,
        step_index=0,
    )


def _object(track_id="obj_1", center=(0.5, 0.0, TABLE_Z + 0.025),
            extents=(0.05, 0.05, 0.05), yaw=0.0, label="block"):
    quat = tf.matrix_to_quat(
        np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    pose = Pose(np.asarray(center, dtype=float), quat, Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id,
        label=label,
        pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.asarray(extents, dtype=float)),
        confidence=0.9,
        num_points=800,
        last_seen_sim_time=0.0,
        last_seen_step=0,
    )


def _scene(*objects):
    return SceneGraph(
        objects={o.track_id: o for o in objects}, sim_time=0.0, step_index=0
    )


class TestGraspPoseConstruction:
    def test_axes_follow_the_gripper_convention(self):
        """Local Y closes the fingers, local Z is the approach direction.

        Measured from the Franka asset: fingertips sit at +/-0.025 along hand Y
        and 0.1034 along hand Z.
        """
        pose = build_grasp_pose(
            grasp_point=np.array([0.5, 0.0, 0.5]),
            closing_axis=np.array([0.0, 1.0, 0.0]),
            approach_axis=np.array([0.0, 0.0, -1.0]),
        )
        rotation = pose.rotation_matrix()
        assert np.allclose(rotation[:, 1], [0.0, 1.0, 0.0], atol=1e-9), "local Y must close"
        assert np.allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-9), "local Z must approach"

    def test_frame_is_right_handed_and_orthonormal(self):
        pose = build_grasp_pose(
            np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.0, -1.0])
        )
        rotation = pose.rotation_matrix()
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)

    def test_non_perpendicular_inputs_are_orthogonalised(self):
        """Fitted box axes are only approximately perpendicular."""
        pose = build_grasp_pose(
            np.zeros(3), np.array([0.0, 1.0, 0.3]), np.array([0.0, 0.0, -1.0])
        )
        rotation = pose.rotation_matrix()
        assert abs(float(np.dot(rotation[:, 1], rotation[:, 2]))) < 1e-9

    def test_parallel_axes_rejected(self):
        with pytest.raises(ValueError, match="parallel"):
            build_grasp_pose(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))

    def test_degenerate_approach_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            build_grasp_pose(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.zeros(3))


class TestGeneration:
    def test_generates_candidates_for_a_graspable_cube(self):
        candidates = generate_grasp_candidates(_object(), GraspConfig())
        assert candidates, "no candidates for a 5 cm cube"
        assert all(c.target_track_id == "obj_1" for c in candidates)

    def test_rejects_object_wider_than_the_gripper(self):
        """A 20 cm box cannot be spanned; nothing should be proposed."""
        wide = _object(extents=(0.20, 0.20, 0.20))
        assert generate_grasp_candidates(wide, GraspConfig()) == []

    def test_rejects_object_thinner_than_the_minimum(self):
        thin = _object(extents=(0.001, 0.001, 0.001))
        assert generate_grasp_candidates(thin, GraspConfig()) == []

    def test_all_widths_are_within_gripper_range(self):
        config = GraspConfig()
        usable = config.max_grasp_width - config.finger_width_margin
        for candidate in generate_grasp_candidates(_object(extents=(0.04, 0.06, 0.09)), config):
            assert config.min_grasp_width <= candidate.width <= usable

    def test_pregrasp_is_offset_backwards_along_the_approach(self):
        """A straight-line approach from the standoff is what keeps the fingers
        from sweeping sideways through the object."""
        config = GraspConfig()
        for candidate in generate_grasp_candidates(_object(), config):
            offset = candidate.pose.position - candidate.pregrasp_pose.position
            distance = float(np.linalg.norm(offset))
            assert distance == pytest.approx(config.approach_offset, abs=1e-9)
            direction = offset / distance
            assert np.allclose(direction, candidate.approach_axis, atol=1e-9)

    def test_pregrasp_shares_the_grasp_orientation(self):
        """Rotating during the approach would twist the fingers into the object."""
        for candidate in generate_grasp_candidates(_object(), GraspConfig()):
            assert np.allclose(candidate.pregrasp_pose.quat, candidate.pose.quat)

    def test_closing_axis_matches_the_reported_width(self):
        """The width must be the extent along the axis the fingers actually close."""
        obj = _object(extents=(0.04, 0.06, 0.09))
        for candidate in generate_grasp_candidates(obj, GraspConfig()):
            closing_axis = candidate.pose.rotation_matrix()[:, 1]
            box_rotation = obj.bbox.center.rotation_matrix()
            alignments = np.abs(box_rotation.T @ closing_axis)
            matched = obj.bbox.extents[int(np.argmax(alignments))]
            assert candidate.width == pytest.approx(matched, abs=1e-9)

    def test_respects_a_rotated_object(self):
        """Candidates must follow the object's yaw, not world axes."""
        obj = _object(extents=(0.04, 0.07, 0.05), yaw=0.6)
        candidates = generate_grasp_candidates(obj, GraspConfig())
        assert candidates
        box_axes = obj.bbox.center.rotation_matrix()
        for candidate in candidates:
            closing = candidate.pose.rotation_matrix()[:, 1]
            # Must align with one of the box axes.
            assert np.max(np.abs(box_axes.T @ closing)) > 0.99

    def test_honours_max_candidates(self):
        config = GraspConfig(max_candidates=5)
        assert len(generate_grasp_candidates(_object(extents=(0.04, 0.05, 0.06)), config)) <= 5

    def test_produces_diverse_approach_directions(self):
        """A single approach direction would make the scorer's job meaningless."""
        candidates = generate_grasp_candidates(_object(extents=(0.04, 0.05, 0.06)), GraspConfig())
        directions = {tuple(np.round(c.approach_axis, 3)) for c in candidates}
        assert len(directions) >= 4


class TestScoring:
    def test_scores_and_ranks_feasible_candidates(self):
        obj = _object()
        candidates = generate_grasp_candidates(obj, GraspConfig())
        scorer = GraspScorer(FakeRobot(), GraspConfig(), support_height=TABLE_Z)

        scored = scorer.score(candidates, _scene(obj), _robot_state())
        assert scored, "everything was rejected for a reachable cube on a table"
        assert all(0.0 <= c.score <= 1.0 for c in scored)
        assert scored == sorted(scored, key=lambda c: c.score, reverse=True)

    def test_unreachable_candidates_are_dropped_not_ranked_low(self):
        """A grasp with no IK solution is not a grasp at all."""
        obj = _object()
        candidates = generate_grasp_candidates(obj, GraspConfig())
        scorer = GraspScorer(
            FakeRobot(reachable=lambda pose: False), GraspConfig(), support_height=TABLE_Z
        )
        assert scorer.score(candidates, _scene(obj), _robot_state()) == []

    def test_prefers_top_down_when_biased(self):
        obj = _object(extents=(0.05, 0.05, 0.05))
        candidates = generate_grasp_candidates(obj, GraspConfig(top_grasp_bias=0.95))
        scorer = GraspScorer(FakeRobot(), GraspConfig(top_grasp_bias=0.95), support_height=TABLE_Z)

        scored = scorer.score(candidates, _scene(obj), _robot_state())
        assert scored
        best_alignment = float(np.dot(scored[0].approach_axis, [0.0, 0.0, -1.0]))
        assert best_alignment > 0.9, (
            f"best grasp approaches at alignment {best_alignment:.2f}, not top-down"
        )

    def test_rejects_low_side_grasps_that_drive_the_hand_into_the_table(self):
        """The TCP is the fingertip midpoint, so the hand trails *behind* it.

        A horizontal approach at tabletop height keeps the fingertips clear but
        puts the hand body at the same height, striking the worktop.
        """
        low = _object(center=(0.5, 0.0, TABLE_Z + 0.012), extents=(0.05, 0.05, 0.024))
        candidates = generate_grasp_candidates(low, GraspConfig())
        scorer = GraspScorer(
            FakeRobot(), GraspConfig(), support_height=TABLE_Z, hand_setback=0.1034
        )

        for scored in scorer.score(candidates, _scene(low), _robot_state()):
            hand_origin_z = float(
                scored.pose.position[2] - scored.approach_axis[2] * 0.1034
            )
            assert hand_origin_z > TABLE_Z - 0.005, (
                "kept a grasp whose hand body is below the table"
            )

    def test_top_down_grasps_are_not_rejected_by_the_support_check(self):
        """Regression guard.

        Treating the fingers as hanging *below* the TCP (they do not -- the TCP is
        the fingertip midpoint) rejected every top-down grasp of a short object,
        leaving only side grasps.
        """
        obj = _object(extents=(0.05, 0.05, 0.05))
        scorer = GraspScorer(FakeRobot(), GraspConfig(), support_height=TABLE_Z)
        scored = scorer.score(
            generate_grasp_candidates(obj, GraspConfig()), _scene(obj), _robot_state()
        )
        top_down = [c for c in scored if float(np.dot(c.approach_axis, [0, 0, -1])) > 0.9]
        assert top_down, "all top-down grasps of a 5 cm cube were rejected"

    def test_rejects_approach_through_a_neighbouring_object(self):
        """A grasp whose approach passes through another object must be dropped."""
        target = _object("target", center=(0.5, 0.0, TABLE_Z + 0.025))
        # A blocker sitting directly above the target blocks top-down approaches.
        blocker = _object(
            "blocker", center=(0.5, 0.0, TABLE_Z + 0.11), extents=(0.10, 0.10, 0.06)
        )
        scene = _scene(target, blocker)
        candidates = generate_grasp_candidates(target, GraspConfig())
        scorer = GraspScorer(FakeRobot(), GraspConfig(), support_height=TABLE_Z)

        scored = scorer.score(candidates, scene, _robot_state())
        for candidate in scored:
            straight_down = float(np.dot(candidate.approach_axis, [0.0, 0.0, -1.0]))
            assert straight_down < 0.99, "kept a grasp approaching through the blocker"

    def test_target_itself_is_not_treated_as_an_obstacle(self):
        """Converging on the target is the point; treating it as an obstacle
        would reject every valid grasp."""
        obj = _object()
        scorer = GraspScorer(FakeRobot(), GraspConfig(), support_height=TABLE_Z)
        assert scorer.score(generate_grasp_candidates(obj, GraspConfig()), _scene(obj), _robot_state())

    def test_breakdown_is_populated_for_diagnosis(self):
        obj = _object()
        scorer = GraspScorer(FakeRobot(), GraspConfig(), support_height=TABLE_Z)
        scored = scorer.score(generate_grasp_candidates(obj, GraspConfig()), _scene(obj), _robot_state())
        assert scored
        breakdown = scored[0].scores_breakdown
        assert {"width_margin", "approach_alignment", "clearance"} <= set(breakdown)
        assert all(0.0 <= v <= 1.0 for v in breakdown.values())

    def test_min_score_filters_weak_grasps(self):
        obj = _object()
        candidates = generate_grasp_candidates(obj, GraspConfig())
        strict = GraspConfig(min_score=0.99)
        scorer = GraspScorer(FakeRobot(), strict, support_height=TABLE_Z)
        assert len(scorer.score(candidates, _scene(obj), _robot_state())) < len(candidates)

    def test_ik_is_only_called_on_candidates_that_survive_cheap_checks(self):
        """Ordering matters: IK is the expensive step and must come last."""
        low = _object(center=(0.5, 0.0, TABLE_Z - 0.05))
        robot = FakeRobot()
        scorer = GraspScorer(robot, GraspConfig(), support_height=TABLE_Z)
        scorer.score(generate_grasp_candidates(low, GraspConfig()), _scene(low), _robot_state())
        assert robot.ik_calls == 0, "IK ran on candidates already rejected geometrically"

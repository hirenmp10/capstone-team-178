"""Phase 9 gate: top-down grasp synthesis, the joint-space planner and
feedback-less grasp verification, on the real ``PlanarKinematics``.

No Isaac Sim, no servos. The robot is a thin ``IRobot`` shim over the closed
form model, which is exactly what ``RemoteArm`` is minus the wire, so every
IK answer here is the one the hardware lane gets. Geometry comes from
``configs/hardware_fake.yaml`` (which restates nothing kinematic, so it is the
placeholder arm of ``hardware.yaml``).

Three measured traps this file pins:

* A vertical tool cannot reach the transit height (0.10 m then, 0.15 m now
  after review GEO-4) anywhere on the placeholder arm (the wrist would sit
  well above a 10.5 cm upper arm with the forearm pinned at horizontal by
  the wrist limit). The planner's transit
  lift therefore has to tilt the tool, or a wall between two low poses fails
  the plan outright. ``test_plan_to_pose_crosses_at_transit_height`` fails on
  a planner that lifts with the original orientation only.
* At 18 cm out a marker is graspable straight down but cannot be lifted 6 cm
  without tilting, so the generator's lift pre-filter drops the 90-degree
  candidate and the 75-degree one leads. The end-to-end pick relies on that.
* The elbow is limited to +/-90 degrees (the placeholder pulse map; review
  GEO-1/HS-2). A top-down grasp at table level is then reachable only from
  about 0.145 m to 0.20 m out, a 75-degree one from about 0.17 m: every pose
  below is chosen inside that band, and ``TestElbowRange`` pins that the
  closer grasps the old +/-120-degree limits accepted are now *refused*
  instead of executed 5-9 cm off by a clamped servo.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mfw.config.schema import load_config
from mfw.core.errors import PlanningError
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
from mfw.grasp.generator import build_grasp_pose
from mfw.grasp.scorer import GraspScorer
from mfw.hardware.grasp import TopDownGraspGenerator, chord_width
from mfw.hardware.kinematics import PlanarKinematics
from mfw.hardware.planner import JointSpacePlanner
from mfw.physics.contact import GraspEvidence, verify_grasp
from mfw.utils import transforms as tf

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_FAKE_YAML = REPO_ROOT / "configs" / "hardware_fake.yaml"

CFG = load_config(HARDWARE_FAKE_YAML)
ARM = CFG.hardware.arm
JOINTS = tuple(CFG.robot.arm_joint_names)
HOME = np.asarray(CFG.robot.home_joint_positions, dtype=np.float64)
SIZES = {label: tuple(size) for label, size in CFG.hardware.object_sizes.items()}
DOWN = np.array([0.0, 0.0, -1.0])

MARKER_XY = (0.18, 0.05)
#: The nearest top-down grasp the +/-90-degree elbow reaches (with its 6 cm lift).
CUBE_XY = (0.15, 0.0)
#: A 75-degree grasp needs more reach: the tilted band starts near 0.17 m.
TILTED_XY = (0.19, 0.0)
#: A goal with BOTH elbow branches inside +/-90 degrees (elbow +/-1.49 rad).
BRANCH_GOAL_XYZ = (0.10, 0.0, 0.28)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


class PlanarRobot(IRobot):
    """``RemoteArm`` without the wire: FK/IK from the closed form, commanded gripper width."""

    def __init__(self, kinematics: PlanarKinematics, q=HOME) -> None:
        self._kin = kinematics
        self.q = np.asarray(q, dtype=np.float64)
        self.width = float(CFG.robot.gripper_open_width)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._kin.joint_names

    def get_state(self) -> RobotState:
        return RobotState(
            joint_state=JointState(positions=self.q.copy(), names=self.joint_names),
            tcp_pose=self.tcp_pose(),
            gripper=GripperState(self.width, self.width, False, False),
            sim_time=0.0,
            step_index=0,
        )

    def tcp_pose(self) -> Pose:
        return self._kin.fk(self.q)

    def forward_kinematics(self, joint_positions) -> Pose:
        return self._kin.fk(joint_positions)

    def inverse_kinematics(self, target: Pose, seed=None):
        return self._kin.ik(target, seed=seed)

    def open_gripper(self) -> None:
        self.width = float(CFG.robot.gripper_open_width)

    def close_gripper(self) -> None:
        self.width = float(CFG.robot.gripper_closed_width)

    def get_gripper_width(self) -> float:
        return self.width


def _yaw_quat(yaw: float) -> np.ndarray:
    return tf.matrix_to_quat(
        np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    )


def _object(track_id: str, label: str, xy, size, yaw: float = 0.0, z: float | None = None,
            last_seen_step: int = 0) -> ObjectHypothesis:
    """A perceived box resting on the table (centre at half its height), as PlanarPerception reports."""
    size = tuple(float(s) for s in size)
    centre = np.array([xy[0], xy[1], size[2] / 2.0 if z is None else z], dtype=np.float64)
    pose = Pose(centre, _yaw_quat(yaw), Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id, label=label, pose=pose, bbox=BoundingBox3D(center=pose, extents=np.array(size)),
        confidence=0.9, num_points=0, last_seen_sim_time=0.0, last_seen_step=last_seen_step,
    )


def _scene(*objects: ObjectHypothesis, step: int = 0) -> SceneGraph:
    return SceneGraph(objects={o.track_id: o for o in objects}, sim_time=0.0, step_index=step)


def _tangential(xy) -> np.ndarray:
    bearing = math.atan2(float(xy[1]), float(xy[0]))
    return np.array([-math.sin(bearing), math.cos(bearing), 0.0])


def _top_down(xy, z: float) -> Pose:
    return build_grasp_pose(np.array([xy[0], xy[1], z], dtype=np.float64), _tangential(xy), DOWN)


def _tcp_path(kin: PlanarKinematics, trajectory) -> np.ndarray:
    return np.array([kin.fk(w.positions).position for w in trajectory.waypoints])


def _times(trajectory) -> np.ndarray:
    return np.array([w.time_from_start for w in trajectory.waypoints])


def _inside_inflated_box(point, obj: ObjectHypothesis, inflation: float = 0.02) -> bool:
    local = obj.bbox.center.rotation_matrix().T @ (np.asarray(point) - obj.bbox.center.position)
    return bool(np.all(np.abs(local) <= obj.bbox.extents / 2.0 + inflation))


@pytest.fixture
def kin() -> PlanarKinematics:
    return PlanarKinematics(ARM, JOINTS)


@pytest.fixture
def robot(kin) -> PlanarRobot:
    return PlanarRobot(kin)


@pytest.fixture
def planner(robot, kin) -> JointSpacePlanner:
    return JointSpacePlanner(robot, kin, CFG.motion, ARM, support_height=0.0)


@pytest.fixture
def generator(kin) -> TopDownGraspGenerator:
    return TopDownGraspGenerator(CFG.grasp, CFG.hardware, support_height=0.0, kinematics=kin)


@pytest.fixture
def geometric_generator() -> TopDownGraspGenerator:
    """No kinematics: pure geometry, every configured pitch survives."""
    return TopDownGraspGenerator(CFG.grasp, CFG.hardware, support_height=0.0, kinematics=None)


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------


class TestTopDownGraspGenerator:
    def test_first_candidate_approaches_straight_down(self, geometric_generator):
        marker = _object("m", "marker", MARKER_XY, SIZES["marker"])
        candidates = geometric_generator.candidates_for(marker)
        assert len(candidates) == len(CFG.hardware.grasp_pitch_angles_deg)
        first = candidates[0]
        np.testing.assert_allclose(first.approach_axis, DOWN, atol=1e-12)
        np.testing.assert_allclose(first.pose.rotation_matrix()[:, 2], DOWN, atol=1e-12)
        assert first.scores_breakdown["pitch_deg"] == 90.0
        assert first.target_track_id == "m"
        # Subsequent pitches lean outward, in the order configured.
        pitches = [c.scores_breakdown["pitch_deg"] for c in candidates]
        assert pitches == list(CFG.hardware.grasp_pitch_angles_deg)

    def test_first_candidate_is_vertical_with_kinematics_for_a_near_object(self, generator):
        cube = _object("c", "cube", CUBE_XY, (0.02, 0.02, 0.02))
        candidates = generator.candidates_for(cube)
        assert candidates
        np.testing.assert_allclose(candidates[0].approach_axis, DOWN, atol=1e-12)

    def test_pregrasp_is_offset_backwards_by_approach_offset(self, generator, geometric_generator):
        for gen in (generator, geometric_generator):
            for candidate in gen.candidates_for(_object("m", "marker", MARKER_XY, SIZES["marker"])):
                offset = candidate.pose.position - candidate.pregrasp_pose.position
                assert np.linalg.norm(offset) == pytest.approx(CFG.grasp.approach_offset, abs=1e-12)
                np.testing.assert_allclose(offset / np.linalg.norm(offset), candidate.approach_axis, atol=1e-12)
                np.testing.assert_allclose(candidate.pregrasp_pose.quat, candidate.pose.quat)

    def test_width_is_the_extent_along_the_jaw_axis(self, geometric_generator):
        """Marker lying along the ray from the base: a tangential jaw meets its 19 mm side."""
        bearing = math.atan2(MARKER_XY[1], MARKER_XY[0])
        marker = _object("m", "marker", MARKER_XY, SIZES["marker"], yaw=bearing)
        candidates = geometric_generator.candidates_for(marker)
        assert candidates
        for candidate in candidates:
            assert candidate.width == pytest.approx(SIZES["marker"][1], abs=1e-9)
            np.testing.assert_allclose(candidate.pose.rotation_matrix()[:, 1], _tangential(MARKER_XY), atol=1e-12)

    def test_width_is_the_chord_not_the_shadow(self, geometric_generator):
        """15 degrees off the jaw line the shadow is 56 mm; the chord the fingers meet is 19.7 mm."""
        bearing = math.atan2(MARKER_XY[1], MARKER_XY[0])
        marker = _object("m", "marker", MARKER_XY, SIZES["marker"], yaw=bearing + math.radians(15.0))
        candidates = geometric_generator.candidates_for(marker)
        assert candidates
        expected = SIZES["marker"][1] / math.cos(math.radians(15.0))
        assert candidates[0].width == pytest.approx(expected, abs=1e-9)
        rotation = marker.bbox.center.rotation_matrix()
        assert chord_width(SIZES["marker"], rotation, _tangential(MARKER_XY)) == pytest.approx(expected, abs=1e-9)
        shadow = float(np.abs(rotation.T @ _tangential(MARKER_XY)) @ (np.array(SIZES["marker"]) / 2.0)) * 2.0
        assert shadow > CFG.grasp.max_grasp_width, "the shadow would have rejected this marker"

    @pytest.mark.parametrize("degrees_off", [45.0, 90.0])
    def test_thin_object_rotated_far_off_the_jaw_line_is_rejected(self, geometric_generator, generator, degrees_off):
        bearing = math.atan2(MARKER_XY[1], MARKER_XY[0])
        marker = _object("m", "marker", MARKER_XY, SIZES["marker"], yaw=bearing + math.radians(degrees_off))
        assert geometric_generator.candidates_for(marker) == []
        assert generator.generate(_scene(marker), "m") == []

    def test_object_wider_than_the_jaw_is_rejected(self, geometric_generator):
        bowl = _object("b", "bowl", (0.15, -0.12), SIZES["bowl"])
        assert geometric_generator.candidates_for(bowl) == []

    def test_lift_prefilter_drops_the_vertical_pitch_at_the_workspace_edge(self, generator, geometric_generator, kin):
        """Marker at 18 cm: graspable straight down, not liftable 6 cm straight up."""
        marker = _object("m", "marker", MARKER_XY, SIZES["marker"])
        vertical = geometric_generator.candidates_for(marker)[0]
        assert kin.ik(vertical.pose) is not None, "the grasp itself is reachable"
        lifted = Pose(vertical.pose.position + [0.0, 0.0, CFG.grasp.lift_height], vertical.pose.quat, Frame.WORLD)
        assert kin.ik(lifted) is None, "but the vertical lift is not"

        kept = generator.candidates_for(marker)
        assert kept, "some tilted pitch must survive"
        assert kept[0].scores_breakdown["pitch_deg"] == 75.0
        assert all(c.scores_breakdown["pitch_deg"] != 90.0 for c in kept)
        for candidate in kept:
            lifted = Pose(candidate.pose.position + [0.0, 0.0, CFG.grasp.lift_height], candidate.pose.quat, Frame.WORLD)
            assert kin.ik(lifted) is not None

    def test_grasp_point_never_sinks_into_the_support(self):
        gen = TopDownGraspGenerator(CFG.grasp, CFG.hardware, support_height=0.40, kinematics=None)
        flat = _object("f", "cube", CUBE_XY, (0.02, 0.02, 0.004), z=0.40 + 0.002)
        candidates = gen.candidates_for(flat)
        assert candidates
        for candidate in candidates:
            assert candidate.pose.position[2] >= 0.40 + 0.005 - 1e-12
            np.testing.assert_allclose(candidate.pose.position[:2], CUBE_XY)

    def test_unknown_track_yields_nothing(self, generator):
        assert generator.generate(_scene(), "ghost") == []

    def test_scorer_ranks_at_least_one_candidate_for_a_reachable_object(self, generator, robot):
        scorer = GraspScorer(
            robot, CFG.grasp, support_height=0.0,
            hand_setback=float(np.linalg.norm(CFG.robot.tcp_offset_from_hand)),
        )
        for obj in (_object("m", "marker", MARKER_XY, SIZES["marker"]), _object("c", "cube", CUBE_XY, (0.02, 0.02, 0.02))):
            scene = _scene(obj)
            candidates = generator.generate(scene, obj.track_id)
            ranked = scorer.score(candidates, scene, robot.get_state())
            assert ranked, f"every top-down grasp of {obj.label!r} was rejected"
            assert all(0.0 <= c.score <= 1.0 for c in ranked)
            assert ranked == sorted(ranked, key=lambda c: c.score, reverse=True)
            assert all(c.target_track_id == obj.track_id for c in ranked)
            assert robot.inverse_kinematics(ranked[0].pose) is not None
            assert robot.inverse_kinematics(ranked[0].pregrasp_pose) is not None


# ----------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------

#: The end-to-end test's in-workspace configuration (TCP about x=0.21, y=0.04, z=0.03).
GOAL_Q = np.array([0.2, 0.9, 0.9, 0.9])


class TestJointSpacePlanner:
    def test_plan_to_joint_is_monotone_and_within_velocity_limit(self, planner, kin):
        trajectory = planner.plan_to_joint(JointState(HOME, names=JOINTS), GOAL_Q, _scene())
        assert trajectory is not None
        assert trajectory.joint_names == JOINTS
        assert trajectory.planner_name == "joint_space_direct"
        times = _times(trajectory)
        assert times[0] == 0.0
        assert np.all(np.diff(times) > 0.0)
        positions = np.stack([w.positions for w in trajectory.waypoints])
        np.testing.assert_allclose(positions[0], HOME)
        np.testing.assert_allclose(positions[-1], GOAL_Q)
        velocities = np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None])
        assert np.max(velocities) <= CFG.motion.max_joint_velocity + 1e-9
        assert all(kin.within_limits(q) for q in positions)

    def test_plan_to_joint_outside_limits_is_none(self, planner):
        beyond = np.array([0.0, 2.0, 0.0, 0.0])
        assert planner.plan_to_joint(JointState(HOME, names=JOINTS), beyond, _scene()) is None

    def test_plan_to_joint_rejects_a_dof_mismatch(self, planner):
        with pytest.raises(PlanningError, match="joints"):
            planner.plan_to_joint(JointState(HOME, names=JOINTS), np.zeros(3), _scene())

    def test_plan_to_joint_refuses_to_drive_the_tcp_into_the_table(self, planner, kin):
        sunk = kin.ik(_top_down(CUBE_XY, -0.03))
        assert sunk is not None and kin.within_limits(sunk), "must be a table rejection, not IK"
        assert planner.plan_to_joint(JointState(HOME, names=JOINTS), sunk, _scene()) is None

    def test_plan_to_pose_goes_direct_when_nothing_is_in_the_way(self, planner, kin):
        start = kin.ik(_top_down((0.14 * math.cos(0.7), 0.14 * math.sin(0.7)), 0.02))
        goal = _top_down((0.14 * math.cos(-0.7), 0.14 * math.sin(-0.7)), 0.02)
        trajectory = planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene())
        assert trajectory is not None
        assert trajectory.planner_name == "joint_space_direct"
        np.testing.assert_allclose(kin.fk(trajectory.waypoints[-1].positions).position, goal.position, atol=1e-9)

    def test_plan_to_pose_crosses_at_transit_height_over_an_obstacle(self, planner, kin):
        """Two low poses 0.14 m out, a 5 cm box on the arc between them."""
        rho = 0.14
        start = kin.ik(_top_down((rho * math.cos(0.7), rho * math.sin(0.7)), 0.02))
        goal = _top_down((rho * math.cos(-0.7), rho * math.sin(-0.7)), 0.02)
        wall = _object("wall", "box", (rho, 0.0), (0.04, 0.04, 0.05))
        assert start is not None

        trajectory = planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(wall))
        assert trajectory is not None, "the transit fallback never fired"
        assert trajectory.planner_name == "joint_space_transit"
        path = _tcp_path(kin, trajectory)
        assert np.max(path[:, 2]) >= ARM.transit_height - 1e-3
        assert not any(_inside_inflated_box(p, wall) for p in path)
        np.testing.assert_allclose(path[-1], goal.position, atol=1e-9)
        assert np.all(np.diff(_times(trajectory)) > 0.0)
        assert all(kin.within_limits(w.positions) for w in trajectory.waypoints)

    def test_excluding_the_obstacle_restores_the_direct_path(self, planner, kin):
        rho = 0.14
        start = kin.ik(_top_down((rho * math.cos(0.7), rho * math.sin(0.7)), 0.02))
        goal = _top_down((rho * math.cos(-0.7), rho * math.sin(-0.7)), 0.02)
        wall = _object("wall", "box", (rho, 0.0), (0.04, 0.04, 0.05))
        planner.exclude_from_collision("wall")
        assert planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(wall)).planner_name == "joint_space_direct"
        planner.include_in_collision("wall")
        assert planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(wall)).planner_name == "joint_space_transit"

    def test_transit_lift_tilts_the_tool_when_vertical_cannot_reach(self, planner, kin):
        """Regression: same-orientation lift is unreachable everywhere for a vertical tool."""
        q = kin.ik(_top_down(CUBE_XY, 0.02))
        pose = kin.fk(q)
        same_orientation = Pose(np.array([*pose.position[:2], ARM.transit_height]), pose.quat, Frame.WORLD)
        assert kin.ik(same_orientation, seed=q) is None, "if this becomes reachable the arm geometry changed"

        raised = planner._raised(q)
        assert raised is not None
        lifted = kin.fk(raised)
        np.testing.assert_allclose(lifted.position[:2], pose.position[:2], atol=1e-9)
        assert lifted.position[2] == pytest.approx(ARM.transit_height, abs=1e-9)
        assert lifted.rotation_matrix()[2, 2] < -0.5, "the tool still points mostly down"
        assert kin.within_limits(raised)

    def test_no_transit_for_a_local_adjust(self, planner, kin):
        """Under 3 cm of horizontal travel there is nothing to cross; only the direct path is tried."""
        start = kin.ik(_top_down(CUBE_XY, 0.02))
        goal = _top_down((CUBE_XY[0] + 0.01, CUBE_XY[1]), 0.02)
        blocker = _object("blk", "cube", (CUBE_XY[0] + 0.005, CUBE_XY[1]), (0.01, 0.01, 0.01))
        assert planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(blocker)) is None

    @pytest.mark.parametrize(("pitch_deg", "xy"), [(90.0, CUBE_XY), (75.0, TILTED_XY)])
    def test_cartesian_line_stays_within_a_step_of_the_line(self, planner, kin, pitch_deg, xy):
        bearing = math.atan2(xy[1], xy[0])
        approach, closing = TopDownGraspGenerator._frame(bearing, math.radians(90.0 - pitch_deg), ARM.jaw_axis)
        grasp = build_grasp_pose(np.array([*xy, 0.02]), closing, approach)
        pregrasp = Pose(grasp.position - approach * CFG.grasp.approach_offset, grasp.quat, Frame.WORLD)
        start = kin.ik(pregrasp)
        assert start is not None

        trajectory = planner.plan_cartesian_line(JointState(start, names=JOINTS), grasp, _scene())
        assert trajectory is not None
        assert trajectory.planner_name == "cartesian_line"
        expected_steps = int(np.ceil(CFG.grasp.approach_offset / CFG.motion.cartesian_step))
        assert len(trajectory) == expected_steps + 1

        path = _tcp_path(kin, trajectory)
        p0, p1 = pregrasp.position, grasp.position
        direction = (p1 - p0) / np.linalg.norm(p1 - p0)
        for point in path:
            along = np.dot(point - p0, direction)
            lateral = np.linalg.norm((point - p0) - along * direction)
            assert lateral < CFG.motion.cartesian_step
            assert -1e-9 <= along <= np.linalg.norm(p1 - p0) + 1e-9
        np.testing.assert_allclose(path[-1], p1, atol=1e-9)
        assert np.all(np.diff(_times(trajectory)) > 0.0)
        for waypoint in trajectory.waypoints:
            np.testing.assert_allclose(kin.fk(waypoint.positions).rotation_matrix()[:, 2], approach, atol=1e-9)

    def test_cartesian_line_beyond_reach_is_none(self, planner, kin):
        start = kin.ik(_top_down(CUBE_XY, 0.07))
        far = _top_down((0.5, 0.0), 0.07)
        assert planner.plan_cartesian_line(JointState(start, names=JOINTS), far, _scene()) is None

    def test_cartesian_line_requires_a_world_frame_goal(self, planner):
        local = Pose(np.array([0.14, 0.0, 0.02]), np.array([1.0, 0.0, 0.0, 0.0]), Frame.ROBOT_BASE)
        with pytest.raises(PlanningError, match="world frame"):
            planner.plan_cartesian_line(JointState(HOME, names=JOINTS), local, _scene())

    def test_plan_with_retries_takes_the_preferred_branch_when_clear(self, planner, kin):
        goal = build_grasp_pose(np.array(BRANCH_GOAL_XYZ), np.array([0.0, 1.0, 0.0]), np.array([0.7, 0.0, 0.7]))
        branches = kin.both_branches(goal)
        assert len(branches) == 2 and branches[0][2] > 0.0 and branches[1][2] < 0.0
        assert all(kin.within_limits(b) for b in branches), "both branches must be physically reachable"
        trajectory = planner.plan_with_retries(JointState(HOME, names=JOINTS), goal, _scene())
        assert trajectory is not None
        np.testing.assert_allclose(trajectory.waypoints[-1].positions, branches[0], atol=1e-9)

    def test_plan_with_retries_succeeds_on_the_second_branch(self, planner, kin):
        """A perceived box sits on the elbow-up sweep from home; the elbow-down sweep misses it.

        Both branches share the TCP, so only the *path* to each can differ:
        the obstacle is placed on the first branch's TCP sweep, about 5 cm
        clear of the second's. Both branches keep the elbow inside +/-90 deg.
        """
        goal = build_grasp_pose(np.array(BRANCH_GOAL_XYZ), np.array([0.0, 1.0, 0.0]), np.array([0.7, 0.0, 0.7]))
        branches = kin.both_branches(goal)
        assert len(branches) == 2 and all(kin.within_limits(b) for b in branches)
        obstacle = _object("shelf", "box", (0.085, 0.0), (0.03, 0.03, 0.03), z=0.351)
        scene = _scene(obstacle)
        start = JointState(HOME, names=JOINTS)

        assert planner.plan_to_joint(start, branches[0], scene) is None, "the first branch must be blocked"
        assert planner.plan_to_joint(start, branches[1], scene) is not None

        trajectory = planner.plan_with_retries(start, goal, scene)
        assert trajectory is not None
        final = trajectory.waypoints[-1].positions
        np.testing.assert_allclose(final, branches[1], atol=1e-9)
        assert final[2] < 0.0, "the fallback is the elbow-down branch"
        np.testing.assert_allclose(kin.fk(final).position, goal.position, atol=1e-9)
        assert not any(_inside_inflated_box(p, obstacle) for p in _tcp_path(kin, trajectory))

    def test_plan_with_retries_with_no_branch_is_none(self, planner):
        far = _top_down((0.5, 0.0), 0.02)
        assert planner.plan_with_retries(JointState(HOME, names=JOINTS), far, _scene()) is None

    def test_elbow_down_preference_is_honoured(self, robot):
        arm_down = replace(ARM, elbow_up=False)
        kin_down = PlanarKinematics(arm_down, JOINTS)
        planner_down = JointSpacePlanner(PlanarRobot(kin_down), kin_down, CFG.motion, arm_down, support_height=0.0)
        goal = build_grasp_pose(np.array(BRANCH_GOAL_XYZ), np.array([0.0, 1.0, 0.0]), np.array([0.7, 0.0, 0.7]))
        trajectory = planner_down.plan_with_retries(JointState(HOME, names=JOINTS), goal, _scene())
        assert trajectory is not None
        assert trajectory.waypoints[-1].positions[2] < 0.0


# ----------------------------------------------------------------------
# transit height (review GEO-4 + the held-object hang-down)
# ----------------------------------------------------------------------

BIN_XY = (0.17, 0.0)


def _bin() -> ObjectHypothesis:
    return _object("bin", "bin", BIN_XY, SIZES["bin"])


ACROSS_BEARING = 0.9
"""Radians either side of the bin: far enough out in y that the start pose
sits outside the bin's box even when it is inflated by a held cube."""


def _across_the_bin(kin: PlanarKinematics, rho: float = 0.15) -> tuple[np.ndarray, Pose]:
    """A low top-down start on one side of the bin and a low goal on the other."""
    b = ACROSS_BEARING
    start = kin.ik(_top_down((rho * math.cos(b), rho * math.sin(b)), 0.03))
    goal = _top_down((rho * math.cos(-b), rho * math.sin(-b)), 0.03)
    assert start is not None
    return start, goal


class TestTransitHeight:
    def test_config_floor_is_above_the_tallest_listed_object(self):
        """The floor alone must clear every object the demo lists, inflated."""
        tallest = max(size[2] for size in SIZES.values())
        assert ARM.transit_height >= tallest + 0.02 + 1e-9

    def test_empty_scene_uses_the_config_floor(self, planner):
        planner.update_collision_world(_scene())
        assert planner.transit_height_for_scene() == pytest.approx(ARM.transit_height)

    def test_tall_obstacle_raises_the_transit_height(self, planner):
        tower = _object("tower", "box", (0.17, 0.0), (0.04, 0.04, 0.16))
        planner.update_collision_world(_scene(tower))
        assert planner.transit_height_for_scene() >= 0.16 + 0.02 - 1e-9

    def test_held_object_raises_the_crossing_and_inflates_obstacles(self, planner):
        bin_ = _bin()
        tower = _object("tower", "box", (0.20, -0.10), (0.04, 0.04, 0.16))
        cube = _object("cube", "cube", CUBE_XY, SIZES["cube"])
        planner.update_collision_world(_scene(bin_, tower, cube))
        empty_handed = planner.transit_height_for_scene()
        assert empty_handed > ARM.transit_height, "the tower, not the floor, sets this crossing"
        planner.exclude_from_collision("cube")
        assert planner.held_half_height() == pytest.approx(SIZES["cube"][2] / 2.0)
        assert planner.transit_height_for_scene() == pytest.approx(empty_handed + SIZES["cube"][2] / 2.0)
        # A TCP 3 cm above the bin rim: clear empty-handed, a collision with a
        # 5 cm cube hanging 2.5 cm below the fingertips.
        probe = np.array([BIN_XY[0], BIN_XY[1], SIZES["bin"][2] + 0.03])
        assert planner._tcp_in_obstacle(probe) == "bin"
        planner.include_in_collision("cube")
        planner.exclude_from_collision("ghost")  # excluded but not in the scene: nothing held
        assert planner.held_half_height() == 0.0
        assert planner._tcp_in_obstacle(probe) is None

    def test_a_transit_over_the_bin_clears_it(self, planner, kin):
        """The measured trap: at transit_height 0.10 the vias sat at the bin rim -> no plan."""
        start, goal = _across_the_bin(kin)
        bin_ = _bin()
        trajectory = planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(bin_))
        assert trajectory is not None, "a 0.10 m bin must be crossable"
        assert trajectory.planner_name == "joint_space_transit"
        path = _tcp_path(kin, trajectory)
        assert not any(_inside_inflated_box(p, bin_) for p in path)
        over = [p for p in path if _inside_inflated_box([p[0], p[1], bin_.bbox.center.position[2]], bin_)]
        assert over, "the transit must actually pass over the bin"
        assert min(p[2] for p in over) >= SIZES["bin"][2] + 0.02 - 1e-9
        np.testing.assert_allclose(path[-1], goal.position, atol=1e-9)

    def test_a_held_cube_is_carried_over_the_bin_with_its_own_clearance(self, planner, kin):
        start, goal = _across_the_bin(kin)
        bin_ = _bin()
        # The held cube rides with the TCP; perception still has it in the scene.
        b = ACROSS_BEARING
        held = _object("held", "cube", (0.15 * math.cos(b), 0.15 * math.sin(b)), SIZES["cube"], z=0.03)
        planner.exclude_from_collision("held")
        trajectory = planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(bin_, held))
        assert trajectory is not None
        hang = SIZES["cube"][2] / 2.0
        for p in _tcp_path(kin, trajectory):
            if _inside_inflated_box([p[0], p[1], bin_.bbox.center.position[2]], bin_):
                assert p[2] - hang >= SIZES["bin"][2] + 0.02 - 1e-9, "the cube's bottom scraped the rim"


# ----------------------------------------------------------------------
# the held object is a box that tilts with the transit (re-review, motion lens)
# ----------------------------------------------------------------------


def _held_box_clearance(kin: PlanarKinematics, trajectory, obstacle: ObjectHypothesis, held_size) -> float:
    """Smallest signed distance, metres, from the held box's corners (in the
    TCP frame: long side along X, narrow along the closing axis Y, height
    along Z) to the obstacle's *real* box over the whole trajectory;
    negative is penetration. The same geometry as the re-review's probe."""
    import itertools

    centre = obstacle.bbox.center.position
    half = obstacle.bbox.extents / 2.0
    length, width, height = max(held_size[0], held_size[1]), min(held_size[0], held_size[1]), held_size[2]
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * (np.array([length, width, height]) / 2.0)
    worst = math.inf
    for waypoint in trajectory.waypoints:
        pose = kin.fk(waypoint.positions)
        for point in pose.position + corners @ pose.rotation_matrix().T:
            d = np.abs(point - centre) - half
            worst = min(worst, float(np.linalg.norm(np.maximum(d, 0.0))) if np.any(d > 0) else float(d.max()))
    return worst


def _reach_lifted(kin: PlanarKinematics, xy, z: float) -> np.ndarray | None:
    """IK of a lifted pose, top-down first, then tilted like the transit lift."""
    q = kin.ik(_top_down(xy, z))
    if q is not None:
        return q
    bearing = math.atan2(xy[1], xy[0])
    radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
    for pitch in (75.0, 60.0, 45.0):
        tilt = math.radians(90.0 - pitch)
        approach = math.sin(tilt) * radial - math.cos(tilt) * np.array([0.0, 0.0, 1.0])
        q = kin.ik(build_grasp_pose(np.array([xy[0], xy[1], z]), _tangential(xy), approach))
        if q is not None:
            return q
    return None


def _place_goals(xy, z: float) -> list[Pose]:
    """Place's candidate orientations above a spot: top-down, then 60 and 45 deg."""
    bearing = math.atan2(xy[1], xy[0])
    goals = [_top_down(xy, z)]
    for pitch in (60.0, 45.0):
        rp = math.radians(pitch)
        approach = np.array([math.cos(bearing) * math.cos(rp), math.sin(bearing) * math.cos(rp), -math.sin(rp)])
        goals.append(build_grasp_pose(np.array([xy[0], xy[1], z]), _tangential(xy), approach))
    return goals


class TestHeldObjectGeometry:
    def test_the_hang_grows_with_the_tilt_for_a_long_object(self):
        marker = JointSpacePlanner.held_tool_extents(_object("m", "marker", (0.18, 0.0), SIZES["marker"]))
        assert marker.tolist() == pytest.approx([0.140, 0.019, 0.019])
        assert JointSpacePlanner.hang_below_tcp(marker, 0.0) == pytest.approx(0.0095)
        # The re-review's number: ~70 mm x sin 45 of marker below the TCP, not 9.5 mm.
        assert JointSpacePlanner.hang_below_tcp(marker, math.radians(45.0)) == pytest.approx(
            (0.070 + 0.0095) * math.sin(math.radians(45.0))
        )
        turned = _object("m", "marker", (0.18, 0.0), (0.019, 0.140, 0.019))
        assert JointSpacePlanner.held_tool_extents(turned).tolist() == pytest.approx([0.140, 0.019, 0.019])

    def test_held_is_inferred_from_the_start_not_from_exclusion_alone(self, planner, kin):
        target = _object("target", "marker", MARKER_XY, SIZES["marker"])
        planner.update_collision_world(_scene(target, _bin()))
        planner.exclude_from_collision("target")
        at_target = _reach_lifted(kin, MARKER_XY, 0.07)
        assert at_target is not None
        assert [tid for tid, _ in planner.held_objects(at_target)] == ["target"]
        # Travelling to a grasp target from home: nothing is in the jaw yet.
        assert planner.held_objects(HOME) == []
        planner.set_held_object("target")
        assert [tid for tid, _ in planner.held_objects(HOME)] == ["target"]
        planner.include_in_collision("target")
        assert planner.held_objects(at_target) == []

    def test_transit_heights_include_the_tilted_hang(self, planner):
        held = _object("held", "marker", (0.10, 0.12), SIZES["marker"], z=0.07)
        planner.update_collision_world(_scene(_bin(), held))
        planner.exclude_from_collision("held")
        heights = planner.transit_heights_for_scene(planner.held_objects())
        assert heights[0] == pytest.approx(planner.transit_height_for_scene())
        top = SIZES["bin"][2]
        tilted = top + 0.02 + JointSpacePlanner.hang_below_tcp(
            JointSpacePlanner.held_tool_extents(held), math.radians(45.0)) + 0.016
        assert heights[-1] == pytest.approx(tilted)
        assert heights == sorted(heights)

    def test_a_held_marker_is_never_carried_into_the_bin_rim(self, planner, kin):
        """The re-review's probe: 26 planner-approved crossings of the 0.10 m
        bin put the carried marker's end 3.6-8.8 mm inside it (example: start
        (0.101, 0.119), goal (0.105, -0.119), TCP at z = 0.150, marker end at
        z = 0.094 against a 0.10 m rim). Every plan now either clears the
        real bin with the marker's real box, or is refused."""
        bin_ = _bin()
        held_size = SIZES["marker"]
        planned = 0
        for rs in (0.13, 0.15, 0.17, 0.19, 0.21):
            for rg in (0.13, 0.15, 0.17, 0.19, 0.21):
                for bs, bg in ((0.9, -0.9), (0.7, -0.7), (1.1, -0.6)):
                    start_xy = (rs * math.cos(bs), rs * math.sin(bs))
                    goal_xy = (rg * math.cos(bg), rg * math.sin(bg))
                    q0 = _reach_lifted(kin, start_xy, held_size[2] / 2.0 + 0.06)
                    if q0 is None:
                        continue
                    held = _object("held", "marker", start_xy, held_size)
                    planner.include_in_collision("held")
                    planner.exclude_from_collision("held")
                    scene = _scene(bin_, held)
                    trajectory = None
                    # Place's goal candidates: top-down, then 60 and 45 deg.
                    for goal in _place_goals(goal_xy, min(held_size[2] / 2.0 + 0.085, 0.20)):
                        if kin.ik(goal) is None:
                            continue
                        trajectory = planner.plan_with_retries(JointState(q0, names=JOINTS), goal, scene)
                        if trajectory is not None:
                            break
                    if trajectory is None:
                        continue
                    planned += 1
                    clearance = _held_box_clearance(kin, trajectory, bin_, held_size)
                    assert clearance > 0.0, (
                        f"marker carried {-clearance * 1000:.1f} mm into the bin: start {np.round(start_xy, 3)} "
                        f"goal {np.round(goal_xy, 3)} via {trajectory.planner_name}"
                    )
        # Not a vacuous pass: the old planner approved 29 of these crossings,
        # all 29 into the rim; some honest ones remain.
        assert planned >= 1
        example_start = _reach_lifted(kin, (0.101, 0.119), 0.0695)
        assert example_start is not None
        held = _object("held", "marker", (0.101, 0.119), held_size)
        trajectory = planner.plan_with_retries(
            JointState(example_start, names=JOINTS), _top_down((0.105, -0.119), 0.0945), _scene(bin_, held)
        )
        if trajectory is not None:
            assert _held_box_clearance(kin, trajectory, bin_, held_size) > 0.0

    def test_a_held_cube_still_crosses_the_bin(self, planner, kin):
        """The box check must not turn every crossing into a refusal: the
        cube case the old planner handled keeps a plan, with real clearance."""
        start, goal = _across_the_bin(kin)
        bin_ = _bin()
        b = ACROSS_BEARING
        held = _object("held", "cube", (0.15 * math.cos(b), 0.15 * math.sin(b)), SIZES["cube"], z=0.03)
        planner.exclude_from_collision("held")
        trajectory = planner.plan_to_pose(JointState(start, names=JOINTS), goal, _scene(bin_, held))
        assert trajectory is not None
        assert _held_box_clearance(kin, trajectory, bin_, SIZES["cube"]) > 0.0

    def test_cartesian_line_can_be_asked_to_check_objects(self, planner, kin):
        """For a transit fallback (the skills stream): the unchecked line the
        re-review caught driving through the bin is refused when asked."""
        b, rho, z = 0.7, 0.17, 0.09  # both ends outside the inflated bin, the line through it
        start = _reach_lifted(kin, (rho * math.cos(b), rho * math.sin(b)), z)
        end = _reach_lifted(kin, (rho * math.cos(-b), rho * math.sin(-b)), z)
        assert start is not None and end is not None
        goal = kin.fk(end)
        scene = _scene(_bin())
        assert not _inside_inflated_box(kin.fk(start).position, _bin())
        assert planner.plan_cartesian_line(JointState(start, names=JOINTS), goal, scene) is not None
        assert planner.plan_cartesian_line(JointState(start, names=JOINTS), goal, scene, check_objects=True) is None


# ----------------------------------------------------------------------
# the elbow range (review GEO-1 / HS-2): plan only what the servo executes
# ----------------------------------------------------------------------


class TestElbowRange:
    def test_laptop_limits_equal_the_jetson_placeholder_map(self):
        from jetson.robot_server import JOINT_NAMES, ServoCalibration

        cal = ServoCalibration.default()
        for i, name in enumerate(JOINT_NAMES):
            joint = cal.joints[name]
            assert ARM.joint_lower[i] == pytest.approx(joint.limit_lower_rad, abs=1e-9)
            assert ARM.joint_upper[i] == pytest.approx(joint.limit_upper_rad, abs=1e-9)

    @pytest.mark.parametrize("rho", [0.10, 0.12])
    def test_grasps_the_old_limits_clamped_are_now_refused(self, kin, generator, rho):
        """At 10-12 cm a top-down grasp needs a 106-119 degree elbow; the servo stops at 90."""
        assert kin.ik(_top_down((rho, 0.0), 0.01)) is None
        assert generator.candidates_for(_object("c", "cube", (rho, 0.0), (0.02, 0.02, 0.02))) == []

    def test_every_planned_grasp_is_executed_unclamped_by_the_placeholder_map(self, kin):
        """Whatever IK accepts, the Jetson's pulse map reproduces to well under a millidegree."""
        from jetson.robot_server import ServoCalibration

        cal = ServoCalibration.default()
        checked = 0
        for rho in np.arange(0.06, 0.30, 0.01):
            for bearing in (-0.8, 0.0, 0.8):
                xy = (rho * math.cos(bearing), rho * math.sin(bearing))
                for z in (0.01, 0.05, 0.10):
                    q = kin.ik(_top_down(xy, z))
                    if q is None:
                        continue
                    executed = cal.pulses_to_q(cal.q_to_pulses(q))
                    np.testing.assert_allclose(executed, q, atol=2e-3)
                    checked += 1
        assert checked > 10


# ----------------------------------------------------------------------
# grasp verification without gripper feedback
# ----------------------------------------------------------------------


class VerifyRobot:
    """What ``verify_grasp`` reads: a gripper width and a TCP pose."""

    def __init__(self, width: float, tcp: Pose) -> None:
        self._width = width
        self._tcp = tcp

    def get_gripper_width(self) -> float:
        return self._width

    def tcp_pose(self) -> Pose:
        return self._tcp


LIFT = CFG.grasp.lift_height
CLOSED = CFG.robot.gripper_closed_width
MIN_DISPLACEMENT = CFG.grasp.verify_min_displacement
STEP_AFTER = 40


def _marker_scenes(after_xy=MARKER_XY, after_z: float | None = None, seen_step: int = STEP_AFTER,
                   present: bool = True):
    before = _scene(_object("m", "marker", MARKER_XY, SIZES["marker"], last_seen_step=0), step=0)
    if not present:
        return before, _scene(step=STEP_AFTER)
    after_obj = _object("m", "marker", after_xy, SIZES["marker"], z=after_z, last_seen_step=seen_step)
    return before, _scene(after_obj, step=STEP_AFTER)


def _tcp_above(xy, z: float) -> Pose:
    return _top_down(xy, z)


def _verify(robot, before, after, **overrides) -> GraspEvidence:
    kwargs = dict(
        robot=robot, scene_before=before, scene_after=after, track_id="m", closed_width=CLOSED,
        expected_height_gain=LIFT, gripper_feedback=False, min_displacement=MIN_DISPLACEMENT,
    )
    kwargs.update(overrides)
    return verify_grasp(**kwargs)


class TestVerifyGraspWithoutFeedback:
    """The feedback-less verdict, from the pixels an honest camera produces (review F1).

    Owned by the perception stream. Every case renders real pixel boxes with
    the scripted detector's analytic camera, lifts them onto the table with
    the production estimator, and asks the production ``predict_lift`` what a
    carried object would look like -- so the numbers are the ones the fake lane
    and (with a measured camera) the real arm produce, not hand-picked shifts.

    The old rule ("moved >= 30 mm on the table, or vanished") is what these
    cases pin as wrong: a carried marker moves only 10-30 mm on the table
    (parallax) and a missed one can vanish under the jaw.
    """

    GRASP_Z = SIZES["marker"][2] / 2.0

    class _Clock:
        step_index = 0
        sim_time = 0.0

    @staticmethod
    def _stack(camera=None, measured: bool = True, footprints=None):
        """``(perception, detector)`` for ``camera`` (default: the fake lane's)."""
        from jetson.detector_service import ScriptedDetector, SyntheticPinhole
        from mfw.hardware.perception import PlanarPerception

        cam = camera if camera is not None else SyntheticPinhole()
        cam_cfg = replace(
            CFG.exterior_camera,
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, resolution=(cam.width, cam.height),
            position=tuple(cam.position), look_at=tuple(cam.look_at), up=tuple(cam.up),
            homography=tuple(cam.homography_pixel_to_table().reshape(-1)), pose_measured=measured,
        )
        hardware = CFG.hardware
        if footprints:
            sizes = dict(hardware.object_sizes)
            sizes.update({k: list(v) for k, v in footprints.items()})
            hardware = replace(hardware, object_sizes=sizes)
        perception = PlanarPerception(
            clock=TestVerifyGraspWithoutFeedback._Clock(), detector=None, config=CFG.perception,
            hardware=hardware, homography=cam_cfg.homography, camera=cam_cfg,
        )
        return perception, ScriptedDetector({}, cam, footprints=footprints)

    @staticmethod
    def _seen(perception, detector, label: str, base_xyz, step: int, occluder=None, distort=None):
        """The hypothesis ``PlanarPerception`` would report for an object whose base is at ``base_xyz``.

        ``distort(box)`` edits the detector's box first (jitter, a merged jaw).
        The orientation attributes are the ones the production estimator adds.
        """
        box = detector.box_for(label, base_xyz, occluder)
        if box is None:
            return None
        if distort is not None:
            box = distort(box)
        est = perception.estimator.estimate_footprint({"label": label, "bbox_px": list(box)})
        return ObjectHypothesis(
            track_id="m", label=label, pose=est.pose, bbox=BoundingBox3D(center=est.pose, extents=est.extents),
            confidence=0.9, num_points=1, last_seen_sim_time=0.0, last_seen_step=step,
            attributes={"bbox_px": [float(v) for v in box], "yaw_rad": est.yaw_rad,
                        "yaw_ambiguous": est.yaw_ambiguous},
        )

    def _run(self, *, xy=MARKER_XY, label="marker", after_base=None, hidden=False, camera=None,
             measured=True, footprints=None, visible_before=True, carried_forward=False, distort=None):
        """Verdict for one physical outcome.

        ``after_base``: where the object's base really is after the lift
        (default: carried, base 1/2 height below the TCP); a fourth value is
        its yaw (a spin). ``hidden`` hides it exactly where the honest
        detector would (under the jaw footprint). ``distort(box, detector,
        tcp)`` edits the post-lift pixel box the way a real detector might.
        """
        perception, detector = self._stack(camera, measured, footprints)
        half_h = perception.estimator.size_for(label)[2] / 2.0
        rest_base = (xy[0], xy[1], 0.0)
        tcp_after = np.array([xy[0], xy[1], half_h + LIFT])
        if after_base is None:
            after_base = (xy[0], xy[1], LIFT)  # carried: centre at the TCP
        before_obj = self._seen(perception, detector, label, rest_base, step=0)
        assert before_obj is not None
        if not visible_before:
            before_obj.last_seen_step = -1
        before = _scene(before_obj, step=0)
        occluder = detector.arm_box(tcp_after) if hidden else None
        edit = None if distort is None else (lambda box: distort(box, detector, tcp_after))
        after_obj = self._seen(perception, detector, label, after_base, STEP_AFTER, occluder, edit)
        if carried_forward:
            after_obj = before_obj  # the tracker kept the old track alive
        after = _scene(*( [after_obj] if after_obj is not None else [] ), step=STEP_AFTER)
        prediction = perception.predict_lift(before_obj, tcp_after)
        robot = VerifyRobot(CLOSED, _tcp_above(tcp_after[:2], tcp_after[2]))
        return _verify(robot, before, after, lift_prediction=prediction), prediction

    # -- carried --------------------------------------------------------

    def test_carried_marker_is_held_although_it_moved_less_than_the_old_gate(self):
        evidence, prediction = self._run()
        assert evidence.verdict == "carried" and evidence.holding
        assert evidence.object_visible and evidence.pixel_grew
        assert evidence.pixel_scale > 1.04 and evidence.expected_pixel_scale > 1.04
        assert evidence.tcp_in_object_box is True and evidence.on_prediction is True
        # The parallax shift the old rule demanded be >= 30 mm: it is not.
        assert 0.005 < evidence.object_displacement < MIN_DISPLACEMENT
        assert evidence.object_displaced is False
        assert prediction.model_available
        assert "rose with the jaw" in evidence.reason()

    @pytest.mark.parametrize("xy", [(0.12, -0.10), (0.25, 0.15), (0.10, 0.0), (0.22, -0.15)])
    def test_carried_marker_is_held_across_the_workspace(self, xy):
        evidence, _ = self._run(xy=xy)
        assert evidence.verdict == "carried", evidence.reason()

    def test_without_a_measured_camera_growth_alone_is_never_carried(self):
        """Re-review (high, nadir): growth alone is what a spin, 6 % jitter or
        a box that takes in the fingers also produce, so an unmeasured camera
        can only say the box grew *like* a lift -- unknown, not holding."""
        evidence, prediction = self._run(measured=False)
        assert not prediction.model_available and prediction.expected_pixel_scale > 1.04
        assert evidence.tcp_in_object_box is None and evidence.on_prediction is None
        assert evidence.pixel_grew and evidence.shape_consistent is True
        assert evidence.verdict == "unknown" and not evidence.holding
        assert "pose_measured is false" in evidence.reason() and "spun" in evidence.reason()

    # -- spin, jitter, a merged jaw (re-review critical + high) ---------

    NADIR = dict(fx=600, fy=600, cx=320, cy=240, width=640, height=480,
                 position=(0.18, 0.0, 0.60), look_at=(0.18, 0.0, 0.0), up=(1.0, 0.0, 0.0))
    POSITIONS = [(0.15, 0.0), (0.18, 0.05), (0.12, -0.10), (0.25, 0.15), (0.22, -0.18), (0.28, 0.20), (0.10, 0.0)]

    def _cameras(self):
        from jetson.detector_service import SyntheticPinhole

        nadir = SyntheticPinhole(**self.NADIR)
        return [("fake", None, True), ("nadir", nadir, True), ("nadir-unmeasured", nadir, False)]

    @pytest.mark.parametrize("yaw_deg", [5.0, 15.0, 30.0, 45.0])
    def test_a_marker_spun_in_place_is_never_carried(self, yaw_deg):
        """The closing jaw spins the 140 x 19 mm marker without lifting it. Its
        axis-aligned box has a longer diagonal (x1.06-1.13), which read
        'carried' on every camera; Place then carried an empty jaw."""
        for name, camera, measured in self._cameras():
            for xy in self.POSITIONS:
                spun = (xy[0], xy[1], 0.0, math.radians(yaw_deg))
                evidence, _ = self._run(xy=xy, after_base=spun, camera=camera, measured=measured)
                assert not evidence.holding, (name, xy, yaw_deg, evidence.to_log())
                assert evidence.verdict in {"reshaped", "resting", "knocked", "unknown"}

    @pytest.mark.parametrize("scale", [1.04, 1.05, 1.06, 1.07])
    def test_uniform_box_jitter_without_a_lift_is_never_carried(self, scale):
        def grow(box, _detector, _tcp):
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            hw, hh = (box[2] - box[0]) / 2.0 * scale, (box[3] - box[1]) / 2.0 * scale
            return (cx - hw, cy - hh, cx + hw, cy + hh)

        for name, camera, measured in self._cameras():
            for xy in self.POSITIONS:
                evidence, _ = self._run(xy=xy, after_base=(xy[0], xy[1], 0.0), camera=camera,
                                        measured=measured, distort=grow)
                assert not evidence.holding, (name, xy, scale, evidence.to_log())

    def test_a_box_that_takes_in_the_fingers_is_never_carried(self):
        def merge(box, detector, tcp):
            arm = detector.arm_box(tcp)
            return (min(box[0], arm[0]), min(box[1], arm[1]), max(box[2], arm[2]), max(box[3], arm[3]))

        for name, camera, measured in self._cameras():
            for xy in self.POSITIONS:
                evidence, _ = self._run(xy=xy, after_base=(xy[0], xy[1], 0.0), camera=camera,
                                        measured=measured, distort=merge)
                assert not evidence.holding, (name, xy, evidence.to_log())

    def test_a_real_carry_still_reads_carried_on_a_measured_nadir_camera(self):
        _, nadir, _ = self._cameras()[1]
        for xy in self.POSITIONS:
            evidence, _ = self._run(xy=xy, camera=nadir, measured=True)
            assert evidence.verdict == "carried", (xy, evidence.reason())
            assert evidence.shape_consistent is True

    def test_carried_needs_the_centre_to_move_like_a_lift(self):
        """Same size as a carried box, but left at the rest spot: on the
        oblique camera a lift moves the box ~50 px, so this is not a carry."""
        def at_rest_but_grown(box, detector, tcp):
            carried = detector.box_for("marker", (tcp[0], tcp[1], LIFT))
            dx = (box[0] + box[2] - carried[0] - carried[2]) / 2.0
            dy = (box[1] + box[3] - carried[1] - carried[3]) / 2.0
            return (carried[0] + dx, carried[1] + dy, carried[2] + dx, carried[3] + dy)

        evidence, prediction = self._run(after_base=(MARKER_XY[0], MARKER_XY[1], 0.0), distort=at_rest_but_grown)
        assert evidence.shape_consistent is True and evidence.centroid_consistent is False
        assert not evidence.holding and prediction.rest_bbox_px is not None

    def test_a_shrunk_box_is_a_partial_detection_not_a_knock(self):
        """Re-review (medium): a jaw that splits the marker and a detector that
        boxes one end gave 'knocked, moved 94 mm' while the marker was held."""
        from jetson.detector_service import ScriptedDetector

        end = ScriptedDetector({}, footprints={"marker": (0.0475, 0.019, 0.019)})

        def one_end(_box, _detector, tcp):
            return end.box_for("marker", (tcp[0] + 0.046, tcp[1], LIFT))

        evidence, _ = self._run(distort=one_end)
        assert evidence.verdict == "unknown" and not evidence.holding
        assert "partial" in evidence.reason() and "knocked" not in evidence.reason()

    def test_sim_lane_log_record_is_unchanged(self):
        """The feedback lane's record keeps exactly its old keys (byte-for-byte
        sim path; tests/test_grasp_logic.py pins the verdicts)."""
        evidence = GraspEvidence(
            fingers_stalled=True, gripper_width=0.02, object_tracked=True, object_to_tcp_distance=0.01,
            height_gain=0.05, expected_height_gain=0.06, contact_force=None,
        )
        assert list(evidence.to_log()) == [
            "holding", "fingers_stalled", "gripper_width", "object_tracked", "object_to_tcp_distance",
            "height_gain", "expected_height_gain", "contact_force", "object_min_extent", "width_plausible",
            "gripper_feedback", "object_visible", "object_displacement", "object_displaced", "visible_before",
            "pixel_scale", "expected_pixel_scale", "pixel_grew", "tcp_in_object_box", "predicted_offset",
            "on_prediction", "occluded_by_arm", "rest_hidden_by_arm", "model_available", "verdict",
        ]

    # -- missed / knocked ----------------------------------------------

    def test_missed_grasp_leaves_the_object_resting(self):
        evidence, _ = self._run(after_base=(MARKER_XY[0], MARKER_XY[1], 0.0))
        assert evidence.verdict == "resting" and not evidence.holding
        assert evidence.pixel_scale == pytest.approx(1.0, abs=0.01)
        assert "still resting" in evidence.reason()
        # The commanded width is logged but never decides anything here.
        assert evidence.gripper_width == CLOSED and evidence.to_log()["holding"] is False

    def test_knocked_object_slid_but_did_not_rise(self):
        evidence, _ = self._run(after_base=(MARKER_XY[0] + 0.05, MARKER_XY[1], 0.0))
        assert evidence.object_displaced and not evidence.pixel_grew
        assert evidence.verdict == "knocked" and not evidence.holding
        assert "knocked" in evidence.reason()

    def test_object_lifted_but_not_at_the_tool_is_not_carried(self):
        """Raised as far as a carried one, but 6 cm beside the jaw: not in this jaw."""
        evidence, _ = self._run(after_base=(MARKER_XY[0], MARKER_XY[1] + 0.06, LIFT))
        assert evidence.on_prediction is False
        assert evidence.verdict != "carried"
        assert not evidence.holding

    # -- not seen --------------------------------------------------------

    CUBE = {"cube": (0.03, 0.03, 0.03)}

    def test_small_object_hidden_under_the_jaw_counts_as_held(self):
        """Fake camera, 30 mm cube at (0.25, 0.15): carried box 100 % under the
        jaw, rest spot 31 % covered -- a left-behind cube would be seen."""
        evidence, _ = self._run(xy=(0.25, 0.15), label="cube", footprints=self.CUBE, hidden=True)
        assert evidence.object_visible is False and evidence.visible_before
        assert evidence.rest_hidden_by_arm is False and evidence.occluded_by_arm
        assert evidence.verdict == "occluded" and evidence.holding
        assert "confirm by eye" in evidence.reason()

    def test_missed_small_object_is_not_held_when_it_is_still_visible_at_rest(self):
        evidence, _ = self._run(xy=(0.25, 0.15), label="cube", footprints=self.CUBE,
                                after_base=(0.25, 0.15, 0.0), hidden=True)
        assert evidence.object_visible and evidence.verdict == "resting" and not evidence.holding

    def test_absence_under_a_jaw_that_also_covers_the_rest_spot_is_unknown(self):
        """The trap: a nadir camera sees the lifted jaw straight above the rest
        spot, so a missed cube lying there is hidden exactly like a carried one."""
        from jetson.detector_service import SyntheticPinhole

        nadir = SyntheticPinhole(fx=600, fy=600, cx=320, cy=240, width=640, height=480,
                                 position=(0.18, 0.0, 0.60), look_at=(0.18, 0.0, 0.0), up=(1.0, 0.0, 0.0))
        for after_base in (None, (0.18, 0.05, 0.0)):  # carried, or missed and lying there
            evidence, _ = self._run(xy=(0.18, 0.05), label="cube", footprints=self.CUBE, camera=nadir,
                                    after_base=after_base, hidden=True)
            assert evidence.object_visible is False
            assert evidence.rest_hidden_by_arm is True
            assert evidence.verdict == "unknown" and not evidence.holding
            assert "also covers the spot where it rested" in evidence.reason()

    def test_vanished_marker_is_not_explained_by_the_jaw(self):
        """The legacy scripted rule (lifted -> gone) is not evidence: a marker
        sticks far out of a 45 mm jaw, so the jaw cannot be what hides it."""
        evidence, _ = self._run(after_base=(MARKER_XY[0], MARKER_XY[1], 1.0))  # out of view
        assert evidence.object_visible is False
        assert evidence.verdict == "unknown" and not evidence.holding

    def test_not_seen_now_and_not_seen_before_the_descent_is_unknown(self):
        evidence, _ = self._run(label="cube", xy=(0.25, 0.15), footprints=self.CUBE, hidden=True,
                                visible_before=False)
        assert evidence.visible_before is False and evidence.verdict == "unknown"
        assert not evidence.holding and "not seen before the descent" in evidence.reason()

    def test_track_carried_forward_by_the_tracker_counts_as_not_seen(self):
        evidence, _ = self._run(carried_forward=True)
        assert evidence.object_visible is False
        assert evidence.verdict == "unknown" and not evidence.holding

    def test_absence_without_a_measured_camera_is_unknown(self):
        evidence, _ = self._run(xy=(0.25, 0.15), label="cube", footprints=self.CUBE, hidden=True,
                                measured=False)
        assert evidence.verdict == "unknown" and not evidence.holding
        assert "pose_measured" in evidence.reason()

    def test_log_carries_the_verdict_and_the_pixel_fields(self):
        evidence, _ = self._run()
        log = evidence.to_log()
        assert {"gripper_feedback", "object_visible", "visible_before", "pixel_scale", "expected_pixel_scale",
                "pixel_grew", "tcp_in_object_box", "on_prediction", "occluded_by_arm", "rest_hidden_by_arm",
                "verdict", "holding"} <= set(log)
        assert log["gripper_feedback"] is False and log["verdict"] == "carried" and log["holding"] is True


class TestVerifyGraspWithFeedback:
    """The sim rule is untouched: stall + tracked + plausible width, displacement ignored."""

    MIN_EXTENT = min(SIZES["marker"])

    def _held_setup(self):
        tcp = _tcp_above(MARKER_XY, SIZES["marker"][2] / 2 + LIFT)
        before, after = _marker_scenes(after_z=SIZES["marker"][2] / 2 + LIFT)
        return tcp, before, after

    def test_stalled_tracked_and_plausible_is_holding_without_any_displacement(self):
        tcp, before, after = self._held_setup()
        evidence = _verify(VerifyRobot(self.MIN_EXTENT, tcp), before, after, gripper_feedback=True)
        assert evidence.fingers_stalled and evidence.object_tracked and evidence.width_plausible
        assert evidence.object_displaced is False
        assert evidence.holding
        assert evidence.height_gain == pytest.approx(LIFT)
        assert "stalled" in evidence.reason()

    def test_fully_closed_fingers_are_not_holding_even_if_the_object_moved(self):
        moved_xy = (MARKER_XY[0] + 0.05, MARKER_XY[1])
        before, after = _marker_scenes(after_xy=moved_xy)
        tcp = _tcp_above(moved_xy, SIZES["marker"][2] / 2)
        evidence = _verify(VerifyRobot(CLOSED, tcp), before, after, gripper_feedback=True)
        assert evidence.object_displaced
        assert not evidence.fingers_stalled
        assert not evidence.holding
        assert "without stalling" in evidence.reason()

    def test_stall_far_under_the_object_size_is_not_holding(self):
        """The measured false positive: 13.3 mm stall beside a 35 mm side."""
        tcp = _tcp_above(MARKER_XY, 0.10)
        before, after = _marker_scenes()
        wide_marker = _object("m", "marker", MARKER_XY, (0.14, 0.035, 0.035), last_seen_step=STEP_AFTER)
        after = _scene(wide_marker, step=STEP_AFTER)
        evidence = _verify(VerifyRobot(0.0133, tcp), before, after, gripper_feedback=True, max_object_to_tcp=0.2)
        assert evidence.fingers_stalled and evidence.object_tracked
        assert not evidence.width_plausible
        assert not evidence.holding
        assert "closed past it" in evidence.reason()

    def test_object_far_from_the_fingertips_is_not_holding(self):
        before, after = _marker_scenes()
        tcp = _tcp_above(MARKER_XY, SIZES["marker"][2] / 2 + 0.15)
        evidence = _verify(VerifyRobot(self.MIN_EXTENT, tcp), before, after, gripper_feedback=True)
        assert evidence.fingers_stalled and not evidence.object_tracked
        assert not evidence.holding
        assert "not carried" in evidence.reason()

    def test_object_gone_is_not_holding_with_feedback(self):
        """Disappearance alone is never a carry, on either lane (review F1).

        With feedback the stalled fingers need a tracked object. Without it,
        a vanished object with no pixel evidence (no box growth, no projected
        occlusion) is ``unknown`` -- it may equally have been knocked off the
        table -- and ``unknown`` is not holding.
        """
        before, after = _marker_scenes(present=False)
        tcp = _tcp_above(MARKER_XY, 0.08)
        with_feedback = _verify(VerifyRobot(self.MIN_EXTENT, tcp), before, after, gripper_feedback=True)
        without = _verify(VerifyRobot(self.MIN_EXTENT, tcp), before, after, gripper_feedback=False)
        assert with_feedback.object_tracked is False and not with_feedback.holding
        assert not without.holding
        assert without.verdict not in ("carried", "occluded")

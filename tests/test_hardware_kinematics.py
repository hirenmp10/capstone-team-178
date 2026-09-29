"""Phase 9 gate: closed-form kinematics and the wall clock, no Isaac.

Everything the hardware lane does downstream -- grasp synthesis, the planner,
the controller's workspace check, the fake world in the end-to-end test --
leans on ``PlanarKinematics.fk`` and ``ik`` agreeing with each other and with
``build_grasp_pose``. A silent mismatch there does not crash anything: it
plants the jaw a centimetre off and the pick just misses. So the round trip is
tested at machine precision on random in-limit configurations, and every
rejection path (reach, limits, out-of-plane approach) is checked against a
case whose *reason* the test can prove from the geometry.

Measured trap pinned here: ``ik`` used to keep the seed's yaw for targets
within 1 mm of the yaw axis and then solve the plane along that radial, which
put the TCP up to 2 mm from the request while still returning a solution.
About 0.3% of random in-limit configurations land that close to the axis, so
the 200-draw round trip flaked with roughly four RNG seeds in ten. Seed 7
below is one that hits it.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pytest

from mfw.config.schema import ConfigError, HardwareArmConfig
from mfw.core.errors import ConfigurationError
from mfw.core.types import Frame, Pose
from mfw.grasp.generator import build_grasp_pose
from mfw.hardware.clock import WallClock
from mfw.hardware.kinematics import PlanarKinematics

pytestmark = pytest.mark.phase9

JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")
UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP

#: Random configuration (seed 7, draw 158) whose TCP sits 0.77 mm from the yaw
#: axis: the case that exposed the inexact near-axis IK.
NEAR_AXIS_Q = np.array([-1.20399783, -0.15329425, 0.31396285, -0.16878863])


@pytest.fixture
def arm() -> HardwareArmConfig:
    return HardwareArmConfig()


@pytest.fixture
def kin(arm) -> PlanarKinematics:
    return PlanarKinematics(arm, JOINTS)


def _bearing_axes(point) -> tuple[np.ndarray, np.ndarray]:
    """Radial and tangential unit vectors of the arm plane through ``point``."""
    bearing = math.atan2(float(point[1]), float(point[0]))
    radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
    tangential = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
    return radial, tangential


def _top_down(point) -> Pose:
    """Top-down grasp pose with a tangential jaw, exactly as the generator builds it."""
    _, tangential = _bearing_axes(point)
    return build_grasp_pose(np.asarray(point, dtype=np.float64), tangential, DOWN)


def _pose(point, approach, closing=(0.0, 1.0, 0.0)) -> Pose:
    return build_grasp_pose(
        np.asarray(point, dtype=np.float64),
        np.asarray(closing, dtype=np.float64),
        np.asarray(approach, dtype=np.float64),
    )


def _wrist_within_reach(arm: HardwareArmConfig, pose: Pose) -> bool:
    """Whether the wrist point implied by ``pose`` lies in the 2-link reach annulus.

    Independent of the module under test: the wrist sits ``tool`` behind the
    TCP along the approach axis, and the upper arm + forearm must span the
    distance from the shoulder axis to it.
    """
    approach = pose.rotation_matrix()[:, 2]
    wrist = np.asarray(pose.position) - arm.tool * approach
    radial = math.hypot(float(wrist[0]), float(wrist[1])) - arm.shoulder_offset
    dist = math.hypot(radial, float(wrist[2]) - arm.base_height)
    return abs(arm.upper_arm - arm.forearm) <= dist <= arm.upper_arm + arm.forearm


def _assert_reaches(kin: PlanarKinematics, q, pose: Pose, tol: float = 1e-6) -> None:
    reached = kin.fk(q)
    assert np.linalg.norm(reached.position - pose.position) < tol
    assert np.linalg.norm(reached.rotation_matrix()[:, 2] - pose.rotation_matrix()[:, 2]) < tol


# ----------------------------------------------------------------------
# FK / IK round trip
# ----------------------------------------------------------------------


class TestRoundTrip:
    def test_fk_ik_round_trip_on_200_random_in_limit_configurations(self, kin):
        """Seeded from the configuration itself, IK must land on its own pose exactly."""
        rng = np.random.default_rng(7)
        for _ in range(200):
            q = rng.uniform(kin.lower, kin.upper)
            pose = kin.fk(q)
            q2 = kin.ik(pose, seed=q)
            assert q2 is not None, f"IK lost a pose FK produced: q={q}, tcp={pose.position}"
            assert kin.within_limits(q2)
            _assert_reaches(kin, q2, pose)

    def test_unseeded_round_trip_is_exact_too(self, kin):
        rng = np.random.default_rng(11)
        for _ in range(200):
            q = rng.uniform(kin.lower, kin.upper)
            pose = kin.fk(q)
            q2 = kin.ik(pose)
            assert q2 is not None
            _assert_reaches(kin, q2, pose)

    def test_near_axis_target_is_reached_exactly(self, kin):
        """Regression: a target 0.77 mm from the yaw axis used to come back 1.5 mm off."""
        pose = kin.fk(NEAR_AXIS_Q)
        assert math.hypot(*pose.position[:2]) < 1e-3, "fixture no longer sits near the axis"
        q2 = kin.ik(pose, seed=NEAR_AXIS_Q)
        assert q2 is not None
        _assert_reaches(kin, q2, pose)

    def test_seed_selects_the_nearer_branch(self, kin):
        pose = _pose((0.16, 0.0, 0.25), approach=(1.0, 0.0, 0.0))
        branches = kin.both_branches(pose)
        assert len(branches) == 2
        for branch in branches:
            chosen = kin.ik(pose, seed=branch)
            np.testing.assert_allclose(chosen, branch, atol=1e-9)

    def test_fk_rejects_wrong_joint_count(self, kin):
        with pytest.raises(ValueError, match="4 elements"):
            kin.fk([0.0, 0.0, 0.0])

    def test_ik_rejects_non_world_pose(self, kin):
        pose = _top_down((0.14, 0.0, 0.02))
        local = Pose(pose.position, pose.quat, Frame.ROBOT_BASE)
        with pytest.raises(ValueError, match="WORLD"):
            kin.ik(local)


# ----------------------------------------------------------------------
# elbow branches
# ----------------------------------------------------------------------


class TestBranches:
    POSE = ((0.16, 0.0, 0.25), (1.0, 0.0, 0.0))
    """Horizontal-forward tool 25 cm up: both elbow branches fit the limits."""

    def test_both_elbow_branches_are_distinct_and_valid(self, kin):
        pose = _pose(*self.POSE)
        branches = kin.both_branches(pose)
        assert len(branches) == 2
        first, second = branches
        assert np.sign(first[2]) == -np.sign(second[2]), "branches must bend the elbow opposite ways"
        assert np.linalg.norm(first - second) > 0.1
        for q in branches:
            assert kin.within_limits(q)
            _assert_reaches(kin, q, pose)

    def test_preferred_branch_follows_elbow_up(self, arm):
        pose = _pose(*self.POSE)
        up_first = PlanarKinematics(arm, JOINTS).both_branches(pose)[0]
        down_first = PlanarKinematics(replace(arm, elbow_up=False), JOINTS).both_branches(pose)[0]
        assert up_first[2] > 0.0, "positive elbow is elbow-up by the module's convention"
        assert down_first[2] < 0.0
        np.testing.assert_allclose(up_first[2], -down_first[2], atol=1e-9)

    def test_ik_without_seed_returns_the_preferred_branch(self, kin):
        pose = _pose(*self.POSE)
        np.testing.assert_allclose(kin.ik(pose), kin.both_branches(pose)[0], atol=1e-12)

    def test_top_down_grasp_in_the_workspace_has_one_branch(self, kin):
        """The wrist limit kills the folded-back branch for a vertical tool low down."""
        assert len(kin.both_branches(_top_down((0.14, 0.0, 0.02)))) == 1


# ----------------------------------------------------------------------
# rejections
# ----------------------------------------------------------------------


class TestRejections:
    def test_unreachable_target_is_none(self, arm, kin):
        far = _top_down((0.5, 0.0, 0.02))
        assert not _wrist_within_reach(arm, far)
        assert kin.ik(far) is None
        assert kin.both_branches(far) == []

    def test_limit_violation_is_none_even_though_within_reach(self, arm, kin):
        """Tool pointing back at the base 10 cm out: the wrist would have to fold past 90 deg."""
        pose = _pose((0.10, 0.0, 0.05), approach=(-1.0, 0.0, 0.0))
        assert _wrist_within_reach(arm, pose), "this case must be a limit failure, not a reach failure"
        assert kin.ik(pose) is None
        assert kin.both_branches(pose) == []

    def test_out_of_limit_configuration_is_reported(self, kin):
        assert not kin.within_limits([0.0, 2.0, 0.0, 0.0])
        assert kin.within_limits(kin.upper) and kin.within_limits(kin.lower)
        clamped = kin.clamp([0.0, 2.0, -3.0, 0.0])
        assert kin.within_limits(clamped)
        np.testing.assert_allclose(clamped, [0.0, kin.upper[1], kin.lower[2], 0.0])

    def test_out_of_plane_approach_is_none(self, kin):
        """No wrist roll: an approach axis across the arm plane cannot be met."""
        sideways = _pose((0.18, 0.0, 0.05), approach=(0.0, 1.0, 0.0), closing=(0.0, 0.0, 1.0))
        assert kin.ik(sideways) is None

    def test_slightly_out_of_plane_approach_is_tolerated(self, kin):
        target = _top_down((0.14, 0.0, 0.02))
        tilted = _pose(target.position, approach=(0.0, 0.1, -1.0), closing=(0.0, 1.0, 0.0))
        q = kin.ik(tilted)
        assert q is not None
        # Position is honoured exactly; the approach is re-aimed into the plane.
        assert np.linalg.norm(kin.fk(q).position - tilted.position) < 1e-9
        assert kin.ik(tilted, orientation_tolerance=0.01) is None

    def test_wrong_joint_name_count_is_a_config_error(self, arm):
        with pytest.raises(ConfigError, match="exactly 4"):
            PlanarKinematics(arm, ("a", "b", "c"))

    def test_bad_jaw_axis_is_a_config_error(self, arm):
        with pytest.raises(ConfigError, match="jaw_axis"):
            PlanarKinematics(arm, JOINTS, jaw_axis="sideways")


# ----------------------------------------------------------------------
# conventions shared with grasp synthesis
# ----------------------------------------------------------------------


class TestConventions:
    @pytest.mark.parametrize("point", [(0.14, 0.03, 0.02), (0.12, -0.08, 0.015), (0.16, 0.0, 0.03)])
    def test_fk_jaw_axis_equals_build_grasp_pose_y_for_a_top_down_grasp(self, kin, point):
        """Build the grasp, IK it, FK back: the closing axis must be the same vector."""
        grasp = _top_down(point)
        q = kin.ik(grasp)
        assert q is not None, f"top-down grasp at {point} should be reachable"
        back = kin.fk(q)
        np.testing.assert_allclose(back.rotation_matrix()[:, 1], grasp.rotation_matrix()[:, 1], atol=1e-9)
        np.testing.assert_allclose(back.rotation_matrix(), grasp.rotation_matrix(), atol=1e-9)
        np.testing.assert_allclose(back.position, grasp.position, atol=1e-9)

    def test_radial_jaw_closes_forward_when_the_tool_points_down(self, arm):
        kin = PlanarKinematics(arm, JOINTS, jaw_axis="radial")
        point = np.array([0.14, 0.05, 0.02])
        radial, _ = _bearing_axes(point)
        grasp = build_grasp_pose(point, radial, DOWN)
        q = kin.ik(grasp)
        assert q is not None
        np.testing.assert_allclose(kin.fk(q).rotation_matrix()[:, 1], radial, atol=1e-9)

    # Every target sits inside the top-down band the +/-90-degree elbow
    # reaches (about 0.14-0.19 m out at z = 0.03); (0.10, -0.09) used to be
    # here and is now correctly unreachable (r = 0.135 m needs a 92-degree elbow).
    @pytest.mark.parametrize("xy", [(0.14, 0.05), (0.11, -0.10), (0.02, 0.15), (0.16, 0.0)])
    def test_yaw_is_atan2_of_the_target(self, kin, xy):
        pose = _top_down((xy[0], xy[1], 0.03))
        q = kin.ik(pose)
        assert q is not None
        assert q[0] == pytest.approx(math.atan2(xy[1], xy[0]), abs=1e-12)

    def test_tool_pitch_is_the_sum_of_the_pitches(self, kin):
        rng = np.random.default_rng(5)
        for _ in range(50):
            q = rng.uniform(kin.lower, kin.upper)
            radial, tangential = _bearing_axes([math.cos(q[0]), math.sin(q[0])])
            pitch = q[1] + q[2] + q[3]
            expected = math.sin(pitch) * radial + math.cos(pitch) * UP
            rotation = kin.fk(q).rotation_matrix()
            np.testing.assert_allclose(rotation[:, 2], expected, atol=1e-12)
            np.testing.assert_allclose(rotation[:, 1], tangential, atol=1e-12)

    def test_placeholder_home_is_folded_back_with_jaws_up(self, kin):
        """The yaml's home ``[0, 1.0472, -1.5708, 0.5236]``: TCP ~4 cm forward, ~30 cm high, tool up."""
        home = np.array([0.0, 1.0472, -1.5708, 0.5236])
        assert kin.within_limits(home)
        pose = kin.fk(home)
        assert pose.position[0] == pytest.approx(0.041, abs=0.002)
        assert pose.position[1] == pytest.approx(0.0, abs=1e-9)
        assert pose.position[2] == pytest.approx(0.299, abs=0.002)
        np.testing.assert_allclose(pose.rotation_matrix()[:, 2], UP, atol=1e-4)

    def test_link_points_chain_from_the_base(self, arm, kin):
        pts = kin.link_points(np.zeros(4))
        assert pts.shape == (5, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pts[1], [arm.shoulder_offset, 0.0, arm.base_height])
        # All pitches zero: every link stacks straight up.
        heights = arm.base_height + np.cumsum([0.0, arm.upper_arm, arm.forearm, arm.tool])
        np.testing.assert_allclose(pts[1:, 2], heights)
        np.testing.assert_allclose(kin.wrist_point(np.zeros(4)), pts[3])
        assert kin.reach_radius() == pytest.approx(arm.shoulder_offset + arm.upper_arm + arm.forearm + arm.tool)

    def test_limits_are_read_only(self, kin):
        with pytest.raises(ValueError):
            kin.lower[0] = 0.0


# ----------------------------------------------------------------------
# WallClock
# ----------------------------------------------------------------------


class TestWallClock:
    def test_fast_mode_advances_step_index_deterministically(self):
        clock = WallClock(physics_dt=0.005, settle_steps=10, fast=True)
        assert clock.step_index == 0 and clock.sim_time == 0.0
        clock.step(3)
        assert clock.step_index == 3
        assert clock.sim_time == pytest.approx(0.015)
        clock.settle()
        assert clock.step_index == 13
        clock.settle(steps=2)
        assert clock.step_index == 15
        clock.step(0)
        clock.step(-4)
        assert clock.step_index == 15, "non-positive counts are no-ops"
        clock.render_step(50)
        assert clock.step_index == 15, "render_step never advances the clock"
        clock.reset()
        assert clock.step_index == 0 and clock.sim_time == 0.0

    def test_fast_mode_does_not_sleep(self):
        clock = WallClock(physics_dt=0.1, settle_steps=0, fast=True)
        started = time.perf_counter()
        clock.step(100_000)  # would be 10 000 s of real sleeping
        assert time.perf_counter() - started < 0.5
        assert clock.step_index == 100_000

    def test_slow_mode_sleeps_for_steps_times_dt(self):
        clock = WallClock(physics_dt=0.005, settle_steps=0)
        started = time.perf_counter()
        clock.step(10)
        elapsed = time.perf_counter() - started
        assert 0.03 < elapsed < 0.5, f"10 x 5 ms should sleep about 50 ms, took {elapsed * 1000:.0f} ms"
        assert clock.sim_time >= 0.03
        assert clock.step_index >= 6

    def test_slow_mode_time_passes_without_stepping(self):
        """A detector round trip blocks the laptop while the world keeps moving."""
        clock = WallClock(physics_dt=0.01, settle_steps=0)
        time.sleep(0.03)
        assert clock.sim_time > 0.02
        assert clock.step_index >= 2

    def test_config_is_validated_and_exposed(self):
        clock = WallClock(physics_dt=1 / 120, settle_steps=24)
        assert clock.config.physics_dt == pytest.approx(1 / 120)
        assert clock.config.settle_steps == 24
        assert clock.config.headless is True
        with pytest.raises(ConfigurationError, match="physics_dt"):
            WallClock(physics_dt=0.0, settle_steps=1)
        with pytest.raises(ConfigurationError, match="settle_steps"):
            WallClock(physics_dt=0.01, settle_steps=-1)

    def test_close_is_idempotent(self):
        clock = WallClock(physics_dt=0.01, settle_steps=1, fast=True)
        clock.close()
        clock.close()
        clock.step()  # still usable: close only logs on this lane
        assert clock.step_index == 1

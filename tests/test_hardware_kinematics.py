"""Unit and property tests for planar 4-DoF arm kinematics.

Phase 9: Mechanics, Kinematics, Calibration.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mfw.config.schema import HardwareArmConfig
from mfw.core.errors import KinematicsError
from mfw.core.types import Frame, Pose
from mfw.grasp.generator import build_grasp_pose
from mfw.hardware.kinematics import PlanarKinematics
from mfw.utils import transforms as tf

JOINTS = ("base_yaw", "shoulder", "elbow", "wrist")


@pytest.fixture
def default_arm_config() -> HardwareArmConfig:
    """Standard baseline arm config for unit tests."""
    return HardwareArmConfig(
        base_height=0.07,
        shoulder_offset=0.0,
        upper_arm=0.105,
        forearm=0.10,
        tool=0.09,
        joint_lower=(-1.5708, -0.5236, -1.5708, -1.5708),
        joint_upper=(1.5708, 1.5708, 1.5708, 1.5708),
        elbow_up=True,
        jaw_axis="tangential",
        transit_height=0.15,
    )


@pytest.fixture
def kinematics(default_arm_config: HardwareArmConfig) -> PlanarKinematics:
    return PlanarKinematics(default_arm_config, JOINTS)


@pytest.mark.phase9
class TestPlanarKinematicsValidation:
    """Test constructor validation and error handling."""

    def test_valid_initialization(self, default_arm_config: HardwareArmConfig):
        kin = PlanarKinematics(default_arm_config, JOINTS)
        assert kin.joint_names == JOINTS
        assert len(kin.joint_lower) == 4
        assert len(kin.joint_upper) == 4
        assert kin.s == 0.0
        assert kin.h == 0.07
        assert kin.L1 == 0.105
        assert kin.L2 == 0.10
        assert kin.L3 == 0.09

    def test_wrong_joint_count(self, default_arm_config: HardwareArmConfig):
        with pytest.raises(KinematicsError, match="requires exactly 4 joint names"):
            PlanarKinematics(default_arm_config, ("base_yaw", "shoulder", "elbow"))

        with pytest.raises(KinematicsError, match="requires exactly 4 joint names"):
            PlanarKinematics(default_arm_config, ("j1", "j2", "j3", "j4", "j5"))

    def test_wrong_joint_names(self, default_arm_config: HardwareArmConfig):
        with pytest.raises(KinematicsError, match="Expected joint names"):
            PlanarKinematics(default_arm_config, ("joint1", "joint2", "joint3", "joint4"))

    def test_invalid_geometry(self):
        with pytest.raises(KinematicsError, match="base_height must be positive"):
            PlanarKinematics(HardwareArmConfig(base_height=-0.05), JOINTS)

        with pytest.raises(KinematicsError, match="upper_arm length must be positive"):
            PlanarKinematics(HardwareArmConfig(upper_arm=0.0), JOINTS)

        with pytest.raises(KinematicsError, match="shoulder_offset must be non-negative"):
            PlanarKinematics(HardwareArmConfig(shoulder_offset=-0.01), JOINTS)

    def test_invalid_limits(self):
        with pytest.raises(KinematicsError, match="must be < upper limit"):
            PlanarKinematics(
                HardwareArmConfig(joint_lower=(1.0, 0.0, 0.0, 0.0), joint_upper=(0.0, 1.0, 1.0, 1.0)),
                JOINTS,
            )

        with pytest.raises(KinematicsError, match="must be finite"):
            PlanarKinematics(
                HardwareArmConfig(joint_lower=(float("nan"), 0.0, 0.0, 0.0)),
                JOINTS,
            )

    def test_within_limits(self, kinematics: PlanarKinematics):
        assert kinematics.within_limits([0.0, 0.0, 0.0, 0.0])
        assert kinematics.within_limits([-1.57, -0.5, 0.0, 1.57])
        assert not kinematics.within_limits([2.0, 0.0, 0.0, 0.0])
        assert not kinematics.within_limits([0.0, -1.0, 0.0, 0.0])
        assert not kinematics.within_limits([0.0, 0.0, 0.0])  # wrong length -> False
        assert not kinematics.within_limits([float("nan"), 0.0, 0.0, 0.0])


@pytest.mark.phase9
class TestForwardKinematics:
    """Test Forward Kinematics against analytical and hand-calculated solutions."""

    def test_home_fk(self, kinematics: PlanarKinematics):
        """Zero pose: arm points straight up along +Z."""
        q = [0.0, 0.0, 0.0, 0.0]
        pose = kinematics.fk(q)
        expected_z = 0.07 + 0.105 + 0.10 + 0.09  # h + L1 + L2 + L3 = 0.365 m
        np.testing.assert_allclose(pose.position, [0.0, 0.0, expected_z], atol=1e-6)

        # Approach vector is local +Z (pointing vertically +Z)
        rot = pose.rotation_matrix()
        approach = rot[:, 2]
        np.testing.assert_allclose(approach, [0.0, 0.0, 1.0], atol=1e-6)

    def test_hand_calculated_case_1(self, kinematics: PlanarKinematics):
        """Case 1: Shoulder pitched 90 deg forward (q = [0, pi/2, 0, 0])."""
        q = [0.0, math.pi / 2.0, 0.0, 0.0]
        pose = kinematics.fk(q)
        expected_reach = 0.105 + 0.10 + 0.09  # L1 + L2 + L3 = 0.295 m
        expected_z = 0.07  # h
        np.testing.assert_allclose(pose.position, [expected_reach, 0.0, expected_z], atol=1e-6)
        rot = pose.rotation_matrix()
        np.testing.assert_allclose(rot[:, 2], [1.0, 0.0, 0.0], atol=1e-6)  # approach points +X

    def test_hand_calculated_case_2(self, kinematics: PlanarKinematics):
        """Case 2: Base yaw 90 deg, shoulder upright, elbow 90 deg, wrist 90 deg (top-down grasp)."""
        q = [math.pi / 2.0, 0.0, math.pi / 2.0, math.pi / 2.0]
        pose = kinematics.fk(q)
        # phi1 = 0, phi2 = pi/2, phi3 = pi
        # r = 0.0 + 0.105*0 + 0.10*1 + 0.09*0 = 0.10 m
        # z = 0.07 + 0.105*1 + 0.10*0 + 0.09*(-1) = 0.07 + 0.105 - 0.09 = 0.085 m
        # x = 0, y = 0.10
        np.testing.assert_allclose(pose.position, [0.0, 0.10, 0.085], atol=1e-6)
        rot = pose.rotation_matrix()
        np.testing.assert_allclose(rot[:, 2], [0.0, 0.0, -1.0], atol=1e-6)  # approach points straight down (-Z)

    def test_hand_calculated_case_3(self, kinematics: PlanarKinematics):
        """Case 3: Base yaw -45 deg, shoulder 45 deg, elbow -45 deg, wrist 0."""
        q = [-math.pi / 4.0, math.pi / 4.0, -math.pi / 4.0, 0.0]
        pose = kinematics.fk(q)
        # phi1 = pi/4, phi2 = 0, phi3 = 0
        # r = 0.105 * sin(pi/4) = 0.105 / sqrt(2) = 0.0742462 m
        # z = 0.07 + 0.105 * cos(pi/4) + 0.10 + 0.09 = 0.07 + 0.0742462 + 0.19 = 0.3342462 m
        r_expected = 0.105 * math.sin(math.pi / 4.0)
        z_expected = 0.07 + 0.105 * math.cos(math.pi / 4.0) + 0.10 + 0.09
        x_expected = r_expected * math.cos(-math.pi / 4.0)
        y_expected = r_expected * math.sin(-math.pi / 4.0)
        np.testing.assert_allclose(pose.position, [x_expected, y_expected, z_expected], atol=1e-6)
        rot = pose.rotation_matrix()
        np.testing.assert_allclose(rot[:, 2], [0.0, 0.0, 1.0], atol=1e-6)

    def test_fk_exceptions(self, kinematics: PlanarKinematics):
        with pytest.raises(KinematicsError, match="Expected 4 joint values"):
            kinematics.fk([0.0, 0.0, 0.0])

        with pytest.raises(KinematicsError, match="non-finite"):
            kinematics.fk([0.0, float("nan"), 0.0, 0.0])

        with pytest.raises(KinematicsError, match="non-finite"):
            kinematics.fk([0.0, float("inf"), 0.0, 0.0])


@pytest.mark.phase9
class TestInverseKinematics:
    """Test Inverse Kinematics solver correctness and edge cases."""

    def test_unreachable_far_target(self, kinematics: PlanarKinematics):
        """Target far beyond max extension returns None (no exception)."""
        far_pose = build_grasp_pose(
            np.array([1.5, 0.0, 0.10]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        assert kinematics.ik(far_pose) is None

    def test_unreachable_near_target(self, kinematics: PlanarKinematics):
        """Target inside minimum inner reach returns None."""
        near_pose = build_grasp_pose(
            np.array([0.01, 0.0, 0.50]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        assert kinematics.ik(near_pose) is None

    def test_behind_base_target(self, kinematics: PlanarKinematics):
        """Target with r <= 0 or invalid orientation returns None."""
        zero_r_pose = Pose(np.array([0.0, 0.0, 0.10]), np.array([1.0, 0.0, 0.0, 0.0]), Frame.WORLD)
        assert kinematics.ik(zero_r_pose) is None

    def test_below_table_target(self, kinematics: PlanarKinematics):
        """Target way below table plane returns None."""
        below_pose = build_grasp_pose(
            np.array([0.20, 0.0, -0.50]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        assert kinematics.ik(below_pose) is None

    def test_ik_malformed_inputs(self, kinematics: PlanarKinematics):
        with pytest.raises(KinematicsError, match="Expected Pose instance"):
            kinematics.ik("not a pose")  # type: ignore[arg-type]

        with pytest.raises(KinematicsError, match="non-finite"):
            nan_pose = Pose(np.array([float("nan"), 0.2, 0.1]), np.array([1.0, 0.0, 0.0, 0.0]))
            kinematics.ik(nan_pose)

        with pytest.raises(KinematicsError, match="Expected 4 joint seed"):
            valid_pose = kinematics.fk([0.0, 0.5, -0.5, 0.0])
            kinematics.ik(valid_pose, seed=[0.0, 0.0])

    def test_elbow_branches(self):
        """Verify elbow-up vs elbow-down branch selection."""
        arm_up = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=True,
            jaw_axis="tangential",
        )
        arm_down = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=False,
            jaw_axis="tangential",
        )
        kin_up = PlanarKinematics(arm_up, JOINTS)
        kin_down = PlanarKinematics(arm_down, JOINTS)

        target_pose = kin_up.fk([0.0, 0.8, -0.6, 0.8])

        q_up = kin_up.ik(target_pose)
        q_down = kin_down.ik(target_pose)

        assert q_up is not None
        assert q_down is not None
        assert q_up[2] <= 0.0  # elbow-up branch has q3 <= 0
        assert q_down[2] >= 0.0  # elbow-down branch has q3 >= 0

        # Both forward kinematic poses must land at the exact target
        pose_up = kin_up.fk(q_up)
        pose_down = kin_down.fk(q_down)
        np.testing.assert_allclose(pose_up.position, target_pose.position, atol=1e-6)
        np.testing.assert_allclose(pose_down.position, target_pose.position, atol=1e-6)

    def test_deterministic_seed_tie_breaking(self):
        """When both branches are valid, seed provides deterministic tie breaking."""
        arm_symmetric = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=True,
            jaw_axis="tangential",
        )
        kin = PlanarKinematics(arm_symmetric, JOINTS)
        target = kin.fk([0.0, 0.8, -0.6, 0.8])

        # Seed close to elbow-down solution
        seed_down = [0.0, 0.2, 0.6, 1.2]
        q_recovered_down = kin.ik(target, seed=seed_down)
        assert q_recovered_down is not None
        assert q_recovered_down[2] > 0.0  # Selected elbow-down branch

        # Seed close to elbow-up solution
        seed_up = [0.0, 0.8, -0.6, 0.8]
        q_recovered_up = kin.ik(target, seed=seed_up)
        assert q_recovered_up is not None
        assert q_recovered_up[2] < 0.0  # Selected elbow-up branch

    def test_both_jaw_axis_orientations(self):
        """Test tangential vs radial jaw orientations and symmetric jaw flips."""
        arm_tangential = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=True,
            jaw_axis="tangential",
        )
        arm_radial = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=True,
            jaw_axis="radial",
        )
        kin_tan = PlanarKinematics(arm_tangential, JOINTS)
        kin_rad = PlanarKinematics(arm_radial, JOINTS)

        pose_tan = kin_tan.fk([0.3, 0.6, -0.4, 0.5])
        pose_rad = kin_rad.fk([0.3, 0.6, -0.4, 0.5])

        assert kin_tan.ik(pose_tan) is not None
        assert kin_tan.ik(pose_rad) is None  # Wrong jaw axis rejected

        assert kin_rad.ik(pose_rad) is not None
        assert kin_rad.ik(pose_tan) is None  # Wrong jaw axis rejected

        # Test 180 deg jaw flip (symmetric parallel fingers)
        rot_tan = pose_tan.rotation_matrix()
        # Flip local Y and local X
        rot_flipped = np.stack([-rot_tan[:, 0], -rot_tan[:, 1], rot_tan[:, 2]], axis=1)
        pose_tan_flipped = Pose(pose_tan.position, tf.matrix_to_quat(rot_flipped), Frame.WORLD)
        assert kin_tan.ik(pose_tan_flipped) is not None

    @pytest.mark.parametrize("pitch_deg", [90.0, 75.0, 60.0])
    def test_grasp_pitches(self, pitch_deg: float):
        """Test IK for standard grasp pitches 90°, 75°, 60°."""
        arm = HardwareArmConfig(
            base_height=0.07,
            upper_arm=0.105,
            forearm=0.10,
            tool=0.09,
            joint_lower=(-1.5708, -1.5708, -2.5, -2.5),
            joint_upper=(1.5708, 1.5708, 2.5, 2.5),
            elbow_up=True,
            jaw_axis="tangential",
        )
        kinematics = PlanarKinematics(arm, JOINTS)

        r = 0.17
        bearing = 0.3
        xy = np.array([r * math.cos(bearing), r * math.sin(bearing)])
        z = 0.04

        tilt = math.radians(90.0 - pitch_deg)
        r_hat = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        t_hat = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
        approach = math.sin(tilt) * r_hat - math.cos(tilt) * np.array([0.0, 0.0, 1.0])

        pose = build_grasp_pose(np.array([xy[0], xy[1], z]), t_hat, approach)
        q = kinematics.ik(pose)
        assert q is not None, f"Failed IK for pitch {pitch_deg} deg"

        pose_rec = kinematics.fk(q)
        np.testing.assert_allclose(pose_rec.position, pose.position, atol=1e-6)
        rot_rec = pose_rec.rotation_matrix()
        np.testing.assert_allclose(rot_rec[:, 2], pose.rotation_matrix()[:, 2], atol=1e-6)


@pytest.mark.phase9
class Test1000ConfigurationsRoundtrip:
    """Rigorous property test: 1000 valid configurations roundtrip through FK -> IK -> FK."""

    def test_1000_valid_configurations_roundtrip(self, default_arm_config: HardwareArmConfig):
        arm_wide = HardwareArmConfig(
            base_height=default_arm_config.base_height,
            shoulder_offset=default_arm_config.shoulder_offset,
            upper_arm=default_arm_config.upper_arm,
            forearm=default_arm_config.forearm,
            tool=default_arm_config.tool,
            joint_lower=(-1.5, -0.4, -1.4, -1.4),
            joint_upper=(1.5, 1.4, 1.4, 1.4),
            elbow_up=True,
            jaw_axis="tangential",
        )
        kin = PlanarKinematics(arm_wide, JOINTS)

        rng = np.random.RandomState(42)
        lowers = np.array(arm_wide.joint_lower)
        uppers = np.array(arm_wide.joint_upper)

        tested_count = 0
        while tested_count < 1000:
            q_initial = rng.uniform(lowers, uppers)

            phi1 = q_initial[1]
            phi2 = q_initial[1] + q_initial[2]
            phi3 = q_initial[1] + q_initial[2] + q_initial[3]

            r = arm_wide.upper_arm * math.sin(phi1) + arm_wide.forearm * math.sin(phi2) + arm_wide.tool * math.sin(phi3)
            r_w = arm_wide.upper_arm * math.sin(phi1) + arm_wide.forearm * math.sin(phi2)

            # Filter out backwards-folding configurations where arm penetrates base
            if r < 0.02 or r_w < 0.0:
                continue

            pose = kin.fk(q_initial)
            q_recovered = kin.ik(pose, seed=q_initial)
            assert q_recovered is not None, f"Sample {tested_count} failed IK for pose {pose.position}"

            # Verify recovered joint limits
            assert kin.within_limits(q_recovered), f"Sample {tested_count} recovered q out of limits: {q_recovered}"

            # Verify recovered FK accuracy
            pose_recovered = kin.fk(q_recovered)

            pos_err = float(np.linalg.norm(pose_recovered.position - pose.position))
            assert pos_err < 1e-6, f"Sample {tested_count} position error {pos_err:.2e} exceeds 1e-6 m"

            rot_init = pose.rotation_matrix()
            rot_rec = pose_recovered.rotation_matrix()

            approach_err = float(np.linalg.norm(rot_rec[:, 2] - rot_init[:, 2]))
            assert approach_err < 1e-6, f"Sample {tested_count} approach error {approach_err:.2e} exceeds 1e-6"

            rot_err = float(np.linalg.norm(rot_rec - rot_init))
            assert rot_err < 1e-6, f"Sample {tested_count} rotation matrix error {rot_err:.2e} exceeds 1e-6"

            tested_count += 1

        assert tested_count == 1000

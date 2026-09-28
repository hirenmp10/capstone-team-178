"""Hardware-lane configuration tests (Phase 9 gate).

The hardware lane adds a second backend to a schema that was written for one.
These tests pin the seams: the sim defaults must be untouched (every existing
YAML still loads with the same values), the new dataclasses must reject the
misconfigurations a teammate is most likely to write while measuring the arm,
and ``CameraConfig.pixel_intrinsics`` must reproduce the exact derivation the
sim camera uses -- fx from the horizontal aperture, fy from the vertical one --
or the hardware and sim lanes would disagree about where a pixel points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfw.config.schema import (
    CameraConfig,
    ConfigError,
    FrameworkConfig,
    GraspConfig,
    HardwareArmConfig,
    HardwareConfig,
    MotionConfig,
    RobotConfig,
    load_config,
)

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_YAML = CONFIGS / "default.yaml"
HARDWARE_YAML = CONFIGS / "hardware.yaml"
HARDWARE_FAKE_YAML = CONFIGS / "hardware_fake.yaml"

PLANAR_JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")


def _planar_robot(**overrides) -> RobotConfig:
    kwargs = dict(
        kinematics="planar_4dof",
        arm_joint_names=PLANAR_JOINTS,
        finger_joint_names=("gripper",),
        home_joint_positions=(0.0, 0.0, 0.0, 0.0),
    )
    kwargs.update(overrides)
    return RobotConfig(**kwargs)


class TestYamlsLoad:
    def test_hardware_yaml_loads_and_validates(self):
        cfg = load_config(HARDWARE_YAML)
        assert isinstance(cfg, FrameworkConfig)
        assert cfg.backend == "hardware"
        assert cfg.robot.kinematics == "planar_4dof"
        assert cfg.robot.arm_joint_names == PLANAR_JOINTS
        assert cfg.robot.finger_joint_names == ("gripper",)
        assert cfg.robot.gripper_feedback is False
        assert cfg.motion.planner == "joint_space"
        assert cfg.wrist_camera.enabled is False
        assert cfg.exterior_camera.enabled is True
        assert cfg.scene.add_table is False
        assert cfg.hardware.jetson_host == "192.168.1.50"
        assert cfg.hardware.arm_bridge == "uno_serial"

    def test_hardware_yaml_homography_is_deliberately_empty(self):
        """Empty is a valid *config*; refusing to run on it is the runtime's job."""
        cfg = load_config(HARDWARE_YAML)
        assert cfg.exterior_camera.homography == ()

    def test_hardware_yaml_link_lengths_match_dataclass_placeholders(self):
        """The YAML placeholders are the dataclass defaults, by construction, so
        whichever one a teammate edits first there is a single set of numbers
        until the arm is measured."""
        cfg = load_config(HARDWARE_YAML)
        assert cfg.hardware.arm == HardwareArmConfig()

    def test_hardware_yaml_tool_matches_tcp_offset(self):
        cfg = load_config(HARDWARE_YAML)
        assert cfg.hardware.arm.tool == pytest.approx(cfg.robot.tcp_offset_from_hand[2])

    def test_hardware_yaml_home_inside_arm_limits(self):
        cfg = load_config(HARDWARE_YAML)
        arm = cfg.hardware.arm
        for q, lo, hi in zip(cfg.robot.home_joint_positions, arm.joint_lower, arm.joint_upper):
            assert lo <= q <= hi

    def test_hardware_yaml_grasp_width_fits_the_jaw(self):
        """The inherited 78 mm ceiling would accept objects a 45 mm jaw cannot span."""
        cfg = load_config(HARDWARE_YAML)
        assert cfg.grasp.max_grasp_width + cfg.grasp.finger_width_margin <= cfg.robot.gripper_open_width + 1e-9

    def test_hardware_fake_yaml_loads_and_validates(self):
        cfg = load_config(HARDWARE_FAKE_YAML)
        assert cfg.backend == "hardware"
        assert cfg.hardware.jetson_host == "127.0.0.1"
        assert cfg.hardware.detector_host == "127.0.0.1"
        assert cfg.hardware.fake_clock is True
        assert cfg.hardware.arm_bridge == "fake"
        assert len(cfg.exterior_camera.homography) == 9
        assert isinstance(cfg.exterior_camera.homography, tuple)

    def test_hardware_fake_inherits_everything_else(self):
        real = load_config(HARDWARE_YAML)
        fake = load_config(HARDWARE_FAKE_YAML)
        assert fake.hardware.arm == real.hardware.arm
        assert fake.hardware.object_sizes == real.hardware.object_sizes
        assert fake.hardware.jetson_port == real.hardware.jetson_port
        assert fake.scene.workspace_min == real.scene.workspace_min
        assert fake.robot == real.robot

    def test_object_sizes_are_coerced_to_tuples(self):
        """A typed Mapping must coerce its values or YAML lists leak through
        and the 'sequences become tuples' guarantee silently fails."""
        cfg = load_config(HARDWARE_YAML)
        assert set(cfg.hardware.labels) <= set(cfg.hardware.object_sizes)
        for label, size in cfg.hardware.object_sizes.items():
            assert isinstance(size, tuple), label
            assert len(size) == 3
            assert all(isinstance(v, float) for v in size)

    def test_default_yaml_unchanged_by_additions(self):
        """The sim lane must not notice any of this."""
        cfg = load_config(DEFAULT_YAML)
        assert cfg.backend == "sim"
        assert cfg.robot.kinematics == "lula"
        assert cfg.robot.gripper_feedback is True
        assert len(cfg.robot.arm_joint_names) == 7
        assert cfg.motion.planner == "rrt"
        assert cfg.grasp.recentre_from_standoff is True
        assert cfg.wrist_camera.enabled is True
        assert cfg.exterior_camera.fx == 0.0
        assert cfg.exterior_camera.homography == ()
        assert cfg.exterior_camera.distortion == ()
        assert cfg.hardware == HardwareConfig()


class TestBackendValidation:
    def test_unknown_backend_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("backend: quantum\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="backend"):
            load_config(p)

    def test_hardware_backend_requires_planar_kinematics(self, tmp_path):
        """Switching backend without switching kinematics would fail deep in
        runtime construction (no Lula); catch it at load."""
        p = tmp_path / "c.yaml"
        p.write_text("backend: hardware\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="planar_4dof"):
            load_config(p)

    def test_sim_backend_accepts_planar_kinematics(self):
        """The constraint is one-directional: a planar arm in sim is allowed."""
        FrameworkConfig(robot=_planar_robot()).validate()

    def test_hardware_backend_with_planar_robot_validates(self):
        FrameworkConfig(backend="hardware", robot=_planar_robot()).validate()


class TestRobotKinematics:
    def test_default_still_needs_six_dof(self):
        with pytest.raises(ConfigError, match="at least 6 DoF"):
            RobotConfig(arm_joint_names=("a", "b", "c")).validate()

    def test_planar_with_four_joints_validates(self):
        _planar_robot().validate()

    def test_planar_with_three_joints_rejected(self):
        with pytest.raises(ConfigError, match="exactly 4"):
            _planar_robot(arm_joint_names=("a", "b", "c"), home_joint_positions=(0.0,) * 3).validate()

    def test_planar_with_seven_joints_rejected(self):
        with pytest.raises(ConfigError, match="exactly 4"):
            RobotConfig(kinematics="planar_4dof").validate()

    def test_unknown_kinematics_rejected(self):
        with pytest.raises(ConfigError, match="kinematics"):
            RobotConfig(kinematics="dh_table").validate()

    def test_home_length_still_checked_for_planar(self):
        with pytest.raises(ConfigError, match="home_joint_positions"):
            _planar_robot(home_joint_positions=(0.0, 0.0)).validate()


class TestMotionPlanner:
    @pytest.mark.parametrize("planner", ["rrt", "rmpflow", "joint_space"])
    def test_allowed_planners(self, planner):
        MotionConfig(planner=planner).validate()

    def test_unknown_planner_rejected(self):
        with pytest.raises(ConfigError, match="planner"):
            MotionConfig(planner="astar").validate()


class TestCameraConfig:
    @pytest.mark.parametrize("n", [0, 9])
    def test_homography_lengths_allowed(self, n):
        CameraConfig(homography=tuple(float(i) for i in range(n))).validate()

    @pytest.mark.parametrize("n", [1, 4, 8, 10, 16])
    def test_homography_lengths_rejected(self, n):
        with pytest.raises(ConfigError, match="homography"):
            CameraConfig(homography=tuple(float(i) for i in range(n))).validate()

    @pytest.mark.parametrize("n", [0, 4, 5, 8])
    def test_distortion_lengths_allowed(self, n):
        CameraConfig(distortion=(0.0,) * n).validate()

    @pytest.mark.parametrize("n", [1, 2, 3, 6, 7, 9])
    def test_distortion_lengths_rejected(self, n):
        with pytest.raises(ConfigError, match="distortion"):
            CameraConfig(distortion=(0.0,) * n).validate()

    def test_negative_intrinsic_rejected(self):
        with pytest.raises(ConfigError, match="fx"):
            CameraConfig(fx=-1.0).validate()

    def test_pixel_intrinsics_match_camera_py_for_default_exterior(self):
        """Must equal Camera.get_intrinsics exactly: fx from the horizontal
        aperture, fy from the *vertical* aperture. The default apertures are not
        4:3, so fx != fy, and that asymmetry is measured-correct."""
        cam = load_config(DEFAULT_YAML).exterior_camera
        width, height = cam.resolution
        expected_fx = cam.focal_length * width / cam.horizontal_aperture
        expected_fy = cam.focal_length * height / cam.vertical_aperture
        fx, fy, cx, cy = cam.pixel_intrinsics()
        assert fx == pytest.approx(expected_fx)
        assert fy == pytest.approx(expected_fy)
        assert cx == pytest.approx(width / 2.0)
        assert cy == pytest.approx(height / 2.0)
        # Sanity on the literal numbers so a silent change to the defaults shows.
        assert fx == pytest.approx(2.4 * 640 / 3.896)
        assert fy == pytest.approx(2.4 * 480 / 2.453)
        assert fx != pytest.approx(fy)

    def test_pixel_intrinsics_derivation_for_wrist_defaults(self):
        cam = load_config(DEFAULT_YAML).wrist_camera
        fx, fy, cx, cy = cam.pixel_intrinsics()
        assert fx == pytest.approx(1.93 * 640 / 3.896)
        assert fy == pytest.approx(1.93 * 480 / 2.453)
        assert (cx, cy) == (320.0, 240.0)

    def test_pixel_intrinsics_measured_values_win(self):
        cam = CameraConfig(resolution=(1280, 720), fx=910.0, fy=905.0, cx=650.0, cy=355.0)
        assert cam.pixel_intrinsics() == (910.0, 905.0, 650.0, 355.0)

    def test_pixel_intrinsics_measured_fx_defaults_the_rest(self):
        """A webcam datasheet usually gives one focal length and no centre."""
        cam = CameraConfig(resolution=(640, 480), fx=600.0)
        assert cam.pixel_intrinsics() == (600.0, 600.0, 320.0, 240.0)

    def test_pixel_intrinsics_returns_floats(self):
        for v in CameraConfig().pixel_intrinsics():
            assert isinstance(v, float)

    def test_yaml_homography_and_distortion_become_tuples(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(
            "exterior_camera:\n  homography: [1,0,0, 0,1,0, 0,0,1]\n  distortion: [0.1, -0.2, 0, 0]\n",
            encoding="utf-8",
        )
        cam = load_config(p).exterior_camera
        assert isinstance(cam.homography, tuple) and len(cam.homography) == 9
        assert isinstance(cam.distortion, tuple) and cam.distortion == (0.1, -0.2, 0.0, 0.0)
        assert all(isinstance(v, float) for v in cam.homography)


class TestGraspConfig:
    def test_defaults(self):
        g = GraspConfig()
        assert g.recentre_from_standoff is True
        assert g.verify_min_displacement == pytest.approx(0.03)

    def test_zero_displacement_rejected(self):
        with pytest.raises(ConfigError, match="verify_min_displacement"):
            GraspConfig(verify_min_displacement=0.0).validate()


class TestHardwareArmConfig:
    def test_defaults_validate(self):
        HardwareArmConfig().validate()

    def test_default_values(self):
        arm = HardwareArmConfig()
        assert arm.base_height == pytest.approx(0.07)
        assert arm.shoulder_offset == pytest.approx(0.0)
        assert arm.upper_arm == pytest.approx(0.105)
        assert arm.forearm == pytest.approx(0.10)
        assert arm.tool == pytest.approx(0.09)
        assert len(arm.joint_lower) == 4 and len(arm.joint_upper) == 4
        assert arm.elbow_up is True
        assert arm.jaw_axis == "tangential"
        # GEO-4: 0.10 sat at the bin's own height; the planner raises this floor per plan.
        assert arm.transit_height == pytest.approx(0.15)

    @pytest.mark.parametrize("name", ["base_height", "upper_arm", "forearm", "tool"])
    def test_zero_length_rejected(self, name):
        with pytest.raises(ConfigError, match=name):
            HardwareArmConfig(**{name: 0.0}).validate()

    def test_negative_shoulder_offset_rejected(self):
        with pytest.raises(ConfigError, match="shoulder_offset"):
            HardwareArmConfig(shoulder_offset=-0.01).validate()

    def test_zero_shoulder_offset_allowed(self):
        HardwareArmConfig(shoulder_offset=0.0).validate()

    def test_inverted_limits_rejected(self):
        with pytest.raises(ConfigError, match="joint_lower"):
            HardwareArmConfig(joint_lower=(0.0, 0.0, 0.0, 0.0), joint_upper=(1.0, 1.0, 0.0, 1.0)).validate()

    def test_limits_length_rejected_from_yaml(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("hardware:\n  arm:\n    joint_lower: [0, 0, 0]\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="joint_lower"):
            load_config(p)

    @pytest.mark.parametrize("axis", ["tangential", "radial"])
    def test_jaw_axes_allowed(self, axis):
        HardwareArmConfig(jaw_axis=axis).validate()

    def test_jaw_axis_rejected(self):
        with pytest.raises(ConfigError, match="jaw_axis"):
            HardwareArmConfig(jaw_axis="diagonal").validate()

    def test_transit_height_rejected(self):
        with pytest.raises(ConfigError, match="transit_height"):
            HardwareArmConfig(transit_height=0.0).validate()


class TestHardwareConfig:
    def test_defaults(self):
        hw = HardwareConfig()
        hw.validate()
        assert hw.jetson_host == "127.0.0.1"
        assert hw.jetson_port == 5560
        assert hw.detector_host == "127.0.0.1"
        assert hw.detector_port == 5558
        assert hw.arm_bridge == "uno_serial"
        assert hw.request_timeout_s == pytest.approx(5.0)
        assert hw.trajectory_timeout_margin_s == pytest.approx(5.0)
        assert hw.trajectory_chunk_s == pytest.approx(1.0)
        assert hw.settle_steps_after_motion == 24
        assert hw.fake_clock is False
        assert isinstance(hw.arm, HardwareArmConfig)
        assert dict(hw.object_sizes) == {}
        assert hw.default_object_size == (0.04, 0.04, 0.04)
        assert hw.labels == ()
        assert hw.detection_min_score == pytest.approx(0.1)
        assert hw.pixel_anchor == "bottom_center"
        assert hw.workspace_margin_xy == pytest.approx(0.03)
        assert hw.grasp_pitch_angles_deg == (90.0, 75.0, 60.0)

    @pytest.mark.parametrize("bridge", ["uno_serial", "pca9685", "fake"])
    def test_arm_bridge_allowed(self, bridge):
        HardwareConfig(arm_bridge=bridge).validate()

    def test_arm_bridge_rejected(self):
        with pytest.raises(ConfigError, match="arm_bridge"):
            HardwareConfig(arm_bridge="usb_gpio").validate()

    @pytest.mark.parametrize("name", ["jetson_port", "detector_port"])
    @pytest.mark.parametrize("port", [0, 65536, -1])
    def test_port_range(self, name, port):
        with pytest.raises(ConfigError, match=name):
            HardwareConfig(**{name: port}).validate()

    def test_object_size_must_be_positive_triple(self):
        with pytest.raises(ConfigError, match="object_sizes"):
            HardwareConfig(object_sizes={"marker": (0.1, 0.0, 0.02)}).validate()
        with pytest.raises(ConfigError, match="object_sizes"):
            HardwareConfig(object_sizes={"marker": (0.1, 0.02)}).validate()

    def test_default_object_size_must_be_positive(self):
        with pytest.raises(ConfigError, match="default_object_size"):
            HardwareConfig(default_object_size=(0.04, -0.04, 0.04)).validate()

    @pytest.mark.parametrize("anchor", ["bottom_center", "center"])
    def test_pixel_anchor_allowed(self, anchor):
        HardwareConfig(pixel_anchor=anchor).validate()

    def test_pixel_anchor_rejected(self):
        with pytest.raises(ConfigError, match="pixel_anchor"):
            HardwareConfig(pixel_anchor="top_left").validate()

    @pytest.mark.parametrize("pitch", [0.0, -10.0, 90.5, 180.0])
    def test_pitch_out_of_range_rejected(self, pitch):
        with pytest.raises(ConfigError, match="grasp_pitch_angles_deg"):
            HardwareConfig(grasp_pitch_angles_deg=(90.0, pitch)).validate()

    def test_pitch_boundary_90_allowed(self):
        HardwareConfig(grasp_pitch_angles_deg=(90.0,)).validate()

    def test_empty_pitches_rejected(self):
        with pytest.raises(ConfigError, match="grasp_pitch_angles_deg"):
            HardwareConfig(grasp_pitch_angles_deg=()).validate()

    def test_timeouts_must_be_positive(self):
        with pytest.raises(ConfigError, match="trajectory_chunk_s"):
            HardwareConfig(trajectory_chunk_s=0.0).validate()

    def test_arm_validation_is_wired_through(self):
        with pytest.raises(ConfigError, match="jaw_axis"):
            HardwareConfig(arm=HardwareArmConfig(jaw_axis="x")).validate()

    def test_hardware_validation_is_wired_into_framework(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("hardware:\n  arm_bridge: telepathy\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="arm_bridge"):
            load_config(p)

    def test_unknown_hardware_key_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("hardware:\n  jetson_hots: 1.2.3.4\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(p)

    def test_object_sizes_non_mapping_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("hardware:\n  object_sizes: [marker]\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="object_sizes"):
            load_config(p)


class TestHardwareLaneImportHygiene:
    """Checked in a fresh interpreter: inside the test session torch or Isaac
    may already be resident from another module, which would mask a leak."""

    FORBIDDEN = ("isaacsim", "omni", "pxr", "carb", "torch", "transformers")

    def _loaded_after_import(self, module: str) -> set[str]:
        import json
        import subprocess
        import sys

        probe = (
            "import sys, json; "
            f"import {module}; "
            "print(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        return set(json.loads(out.strip().splitlines()[-1]))

    def test_hardware_package_imports_without_isaac_or_torch(self):
        loaded = self._loaded_after_import("mfw.hardware")
        assert not (loaded & set(self.FORBIDDEN)), loaded & set(self.FORBIDDEN)

    def test_config_schema_imports_without_isaac_or_torch(self):
        """The schema is imported by both lanes; it must stay stdlib + NumPy."""
        loaded = self._loaded_after_import("mfw.config.schema")
        assert not (loaded & set(self.FORBIDDEN)), loaded & set(self.FORBIDDEN)

    def test_jetson_package_does_not_import_mfw(self):
        loaded = self._loaded_after_import("jetson")
        assert "mfw" not in loaded

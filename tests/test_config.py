"""Configuration loading and validation tests.

The point of these is that a bad config fails loudly at startup. A silently
accepted typo means a run proceeds with a default the operator did not intend,
which in manipulation shows up as inexplicable behaviour hours later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfw.config.schema import (
    ConfigError,
    FrameworkConfig,
    PhysicsConfig,
    RobotConfig,
    SceneConfig,
    SceneObjectConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class TestDefaultConfig:
    def test_default_yaml_loads_and_validates(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert isinstance(cfg, FrameworkConfig)

    def test_sequences_become_tuples_not_lists(self):
        """PEP 563 makes dataclass field types strings; if hints are not resolved,
        coercion silently no-ops and these stay lists."""
        cfg = load_config(DEFAULT_CONFIG)
        assert isinstance(cfg.robot.arm_joint_names, tuple)
        assert isinstance(cfg.robot.tcp_offset_from_hand, tuple)
        assert isinstance(cfg.scene.objects, tuple)
        assert isinstance(cfg.wrist_camera.resolution, tuple)

    def test_nested_dataclasses_are_built(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert isinstance(cfg.physics, PhysicsConfig)
        assert isinstance(cfg.scene.objects[0], SceneObjectConfig)

    def test_tcp_offset_is_not_a_finger(self):
        """The whole point of the derived TCP: parent must be the hand, with a
        real forward offset to the fingertip midpoint."""
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.robot.tcp_parent_prim == "panda_hand"
        assert cfg.robot.tcp_offset_from_hand[2] > 0.09
        assert "finger" not in cfg.robot.tcp_parent_prim

    def test_gr00t_contract_matches_embodiment_config(self):
        """These three values are dictated by GR00T's oxe_droid embodiment config.

        action_horizon comes from delta_indices=list(range(40)); the history
        length serves video delta_indices=[-15, 0]; the camera keys are the
        declared modality_keys.
        """
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.gr00t.action_horizon == 40
        assert cfg.gr00t.observation_history >= 16
        assert cfg.gr00t.exterior_camera_key == "exterior_image_1_left"
        assert cfg.gr00t.wrist_camera_key == "wrist_image_left"

    def test_two_distinct_cameras_configured(self):
        """oxe_droid requires both an exterior and a wrist view."""
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.wrist_camera.name != cfg.exterior_camera.name
        assert cfg.wrist_camera.parent_prim, "wrist camera must be attached to a moving link"
        assert not cfg.exterior_camera.parent_prim, "exterior camera must be static"


class TestStrictness:
    def test_unknown_top_level_key_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("simulaton:\n  headless: true\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(p)

    def test_unknown_nested_key_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("physics:\n  static_frixion: 1.0\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(p)

    def test_error_names_the_offending_path(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("motion:\n  bogus: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="motion"):
            load_config(p)

    def test_missing_file_rejected(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("does/not/exist.yaml")

    def test_empty_file_yields_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("", encoding="utf-8")
        assert load_config(p).simulation.headless is True


class TestValidation:
    def test_rest_offset_above_contact_offset_rejected(self):
        with pytest.raises(ConfigError, match="rest_offset"):
            PhysicsConfig(contact_offset=0.001, rest_offset=0.01).validate()

    def test_rendering_faster_than_physics_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("simulation:\n  physics_dt: 0.02\n  rendering_dt: 0.001\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="rendering_dt"):
            load_config(p)

    def test_too_few_dof_rejected(self):
        with pytest.raises(ConfigError, match="at least 6 DoF"):
            RobotConfig(arm_joint_names=("a", "b", "c")).validate()

    def test_home_position_length_mismatch_rejected(self):
        with pytest.raises(ConfigError, match="home_joint_positions"):
            RobotConfig(home_joint_positions=(0.0, 0.0)).validate()

    def test_gripper_widths_must_be_ordered(self):
        with pytest.raises(ConfigError, match="gripper_open_width"):
            RobotConfig(gripper_open_width=0.0, gripper_closed_width=0.05).validate()

    def test_duplicate_scene_object_names_rejected(self):
        obj = SceneObjectConfig(name="dup")
        with pytest.raises(ConfigError, match="duplicate object/furniture names"):
            SceneConfig(objects=(obj, SceneObjectConfig(name="dup"))).validate()

    def test_object_and_furniture_sharing_a_name_rejected(self):
        """Objects and furniture share the /World namespace, so a collision
        would have one prim silently overwrite the other."""
        from mfw.config.schema import SceneFurnitureConfig

        with pytest.raises(ConfigError, match="duplicate object/furniture names"):
            SceneConfig(
                objects=(SceneObjectConfig(name="shelf"),),
                furniture=(SceneFurnitureConfig(name="shelf", asset="storage_cabinet"),),
            ).validate()

    def test_inverted_workspace_rejected(self):
        with pytest.raises(ConfigError, match="workspace_min"):
            SceneConfig(workspace_min=(1.0, 1.0, 1.0), workspace_max=(0.0, 0.0, 0.0)).validate()

    def test_usd_object_without_path_rejected(self):
        with pytest.raises(ConfigError, match="requires usd_subpath"):
            SceneObjectConfig(name="o", kind="usd").validate()

    def test_bad_planner_name_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("motion:\n  planner: astar\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="planner"):
            load_config(p)


class TestExecutorSelection:
    def test_default_is_classical(self):
        assert load_config(DEFAULT_CONFIG).executor_for("pick") == "classical"

    def test_override_selects_backend(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(
            "gr00t:\n  enabled: true\nexecutor_overrides:\n  pick: gr00t\n", encoding="utf-8"
        )
        cfg = load_config(p)
        assert cfg.executor_for("pick") == "gr00t"
        assert cfg.executor_for("place") == "classical"

    def test_selecting_gr00t_while_disabled_is_rejected(self, tmp_path):
        """Guards the most likely misconfiguration: switching backend and
        forgetting to enable the server, which would otherwise fail deep in a
        skill rather than at startup."""
        p = tmp_path / "c.yaml"
        p.write_text("executor_overrides:\n  pick: gr00t\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="gr00t.enabled"):
            load_config(p)

    def test_default_gr00t_while_disabled_is_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("default_executor: gr00t\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="gr00t.enabled"):
            load_config(p)

    def test_unknown_backend_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("default_executor: magic\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="classical"):
            load_config(p)


class TestOverrides:
    def test_overrides_are_deep_merged(self):
        cfg = load_config(DEFAULT_CONFIG, overrides={"simulation": {"headless": False}})
        assert cfg.simulation.headless is False
        # Sibling keys in the same section must survive the merge.
        assert cfg.simulation.settle_steps == 60

    def test_overrides_are_validated(self):
        with pytest.raises(ConfigError):
            load_config(DEFAULT_CONFIG, overrides={"motion": {"planner": "nope"}})

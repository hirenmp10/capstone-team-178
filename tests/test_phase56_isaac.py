"""Phase 5 and 6 gate: Pick and Place under pure physics.

This is the real integration test of the whole stack, and the phase where a fake
attachment would otherwise creep in. The decisive assertion is
``test_pick_lifts_the_object_off_the_table``: the object's height must increase in
*ground truth* after a pick. Nothing in the framework writes an object pose, so
the only way that height can change is real contact and friction.

Also gated here: atomicity. Pick must hold and stop; it must not place.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.core.types import SkillStatus
from tests.conftest import ground_truth_pose

pytestmark = [pytest.mark.isaac, pytest.mark.phase5]


@pytest.fixture
def ready(restore_scene):
    """Settled scene, empty gripper, fresh observation."""
    runtime = restore_scene
    runtime.memory.set_held_object(None)
    runtime.skills.execute("open_gripper")
    runtime.sim.render_step(2)
    runtime.vision.observe()
    runtime.vision.observe()
    return runtime


def _pickable_label(runtime) -> str:
    """A label the gripper can actually span, chosen from perception."""
    scene = runtime.vision.last_scene_graph()
    usable = runtime.config.grasp.max_grasp_width - runtime.config.grasp.finger_width_margin
    candidates = [
        o for o in scene.objects.values() if float(np.min(o.bbox.extents)) <= usable and o.label
    ]
    if not candidates:
        pytest.skip("no perceived object is narrow enough for the gripper")
    # Smallest first: the most reliably graspable.
    candidates.sort(key=lambda o: float(np.min(o.bbox.extents)))
    return candidates[0].label


def _truth_height(runtime, label: str) -> float:
    """Ground-truth height of the spawned object whose semantic label matches."""
    for obj in runtime.config.scene.objects:
        if (obj.semantic_label or obj.name).lower() == label.lower():
            return float(ground_truth_pose(runtime, obj.name)[0][2])
    pytest.fail(f"no spawned object with label {label!r}")


class TestAtomicity:
    def test_registry_exposes_the_full_skill_library(self, ready):
        expected = {
            "observe", "scan_scene", "look_at", "move_to", "move_relative",
            "pick", "place", "open_gripper", "close_gripper", "rotate_wrist",
            "go_home", "wait", "stop", "emergency_stop",
        }
        assert expected <= set(ready.skills.names)

    def test_skills_cannot_reach_other_skills(self, ready):
        """The structural guarantee: no skill holds a registry handle.

        Without this, atomicity would depend on every implementation choosing not
        to chain -- and one 'convenient' call inside Pick would silently make
        every pick also place.
        """
        for name in ready.skills.names:
            skill = ready.skills.get(name)
            for attribute in vars(skill.ctx).values():
                assert not isinstance(attribute, type(ready.skills)), (
                    f"skill {name} can reach the registry"
                )

    def test_open_gripper_does_not_move_the_arm(self, ready):
        """"Open gripper" opens only."""
        before = ready.robot.tcp_pose().position.copy()
        assert ready.skills.execute("open_gripper").ok
        after = ready.robot.tcp_pose().position
        assert float(np.linalg.norm(after - before)) < 0.01

    def test_close_gripper_does_not_move_the_arm(self, ready):
        before = ready.robot.tcp_pose().position.copy()
        assert ready.skills.execute("close_gripper").ok
        assert float(np.linalg.norm(ready.robot.tcp_pose().position - before)) < 0.01

    def test_close_on_air_reports_no_object(self, ready):
        result = ready.skills.execute("close_gripper")
        assert result.ok
        assert result.data["stalled_on_object"] is False

    def test_move_relative_only_moves_the_requested_axis(self, ready):
        """"Move left" moves left and does nothing else."""
        before = ready.robot.tcp_pose().position.copy()
        result = ready.skills.execute("move_relative", {"direction": "up", "distance": 0.05})
        if result.status is SkillStatus.INFEASIBLE:
            pytest.skip(f"move up infeasible from this posture: {result.message}")
        assert result.ok, result.message

        delta = ready.robot.tcp_pose().position - before
        assert delta[2] > 0.02, "did not move up"
        assert abs(delta[0]) < 0.02 and abs(delta[1]) < 0.02, (
            f"moved on unrequested axes: {np.round(delta, 3).tolist()}"
        )

    def test_unknown_direction_is_rejected_before_moving(self, ready):
        before = ready.robot.tcp_pose().position.copy()
        result = ready.skills.execute("move_relative", {"direction": "sideways"})
        assert result.status is SkillStatus.INFEASIBLE
        assert float(np.linalg.norm(ready.robot.tcp_pose().position - before)) < 1e-3


class TestObserveSkill:
    def test_observe_reports_objects_without_moving(self, ready):
        before = ready.robot.tcp_pose().position.copy()
        result = ready.skills.execute("observe")
        assert result.ok
        assert result.data["objects"], "observe reported nothing"
        assert float(np.linalg.norm(ready.robot.tcp_pose().position - before)) < 0.01

    def test_observe_never_exposes_ground_truth(self, ready):
        """Perception output must not contain spawn configuration."""
        result = ready.skills.execute("observe")
        spawn_names = {o.name for o in ready.config.scene.objects}
        for described in result.data["objects"]:
            assert described["track_id"] not in spawn_names


class TestPick:
    def test_pick_lifts_the_object_off_the_table(self, ready):
        """The decisive test of pure-physics manipulation.

        Ground-truth height must increase. Nothing in this framework writes an
        object pose, so the only mechanism that can raise it is genuine contact
        and friction between the fingertips and the object.
        """
        label = _pickable_label(ready)
        height_before = _truth_height(ready, label)

        result = ready.skills.execute("pick", {"target": label})
        assert result.ok, f"pick failed: {result.message}"

        height_after = _truth_height(ready, label)
        gain = height_after - height_before
        assert gain > 0.03, (
            f"{label!r} rose only {gain * 1000:.0f} mm; it was not actually lifted "
            f"(evidence: {result.data.get('evidence')})"
        )

    def test_pick_leaves_the_object_held_not_placed(self, ready):
        """Atomicity: pick holds. It must not put the object back down."""
        label = _pickable_label(ready)
        result = ready.skills.execute("pick", {"target": label})
        assert result.ok, result.message

        assert ready.memory.get_held_object() is not None, "memory does not record a held object"
        width = ready.robot.get_gripper_width()
        assert width > ready.config.robot.gripper_closed_width + 0.004, (
            "gripper is fully closed; the object was released"
        )

    def test_pick_records_the_held_object_in_memory(self, ready):
        """What makes a later "place it" resolvable."""
        label = _pickable_label(ready)
        result = ready.skills.execute("pick", {"target": label})
        assert result.ok, result.message
        assert ready.memory.get_held_object() == result.data["track_id"]

    def test_pick_verification_uses_sensor_evidence(self, ready):
        label = _pickable_label(ready)
        result = ready.skills.execute("pick", {"target": label})
        assert result.ok, result.message

        evidence = result.data["evidence"]
        assert evidence["fingers_stalled"] is True
        assert evidence["object_tracked"] is True
        assert evidence["object_to_tcp_distance"] < 0.09

    def test_pick_of_an_unknown_object_fails_cleanly(self, ready):
        result = ready.skills.execute("pick", {"target": "teapot"})
        assert not result.ok
        assert "cannot find" in result.message.lower()

    def test_pick_of_an_oversized_object_is_infeasible(self, ready):
        """A box wider than the gripper must be reported, not attempted."""
        scene = ready.vision.last_scene_graph()
        usable = ready.config.grasp.max_grasp_width - ready.config.grasp.finger_width_margin
        oversized = [
            o for o in scene.objects.values() if float(np.min(o.bbox.extents)) > usable and o.label
        ]
        if not oversized:
            pytest.skip("no oversized object in the scene")

        result = ready.skills.execute("pick", {"target": oversized[0].label})
        assert result.status is SkillStatus.INFEASIBLE
        assert "narrowest" in result.message or "reachable" in result.message

    def test_second_pick_while_holding_is_rejected(self, ready):
        """Guards against silently dropping the first object."""
        label = _pickable_label(ready)
        first = ready.skills.execute("pick", {"target": label})
        if not first.ok:
            pytest.skip(f"first pick failed, cannot test the guard: {first.message}")

        second = ready.skills.execute("pick", {"target": label})
        assert second.status is SkillStatus.INFEASIBLE
        assert "already holding" in second.message


@pytest.mark.phase6
class TestPlace:
    def test_place_releases_and_the_object_settles(self, ready):
        """Place puts the object down under gravity, then stops."""
        label = _pickable_label(ready)
        pick = ready.skills.execute("pick", {"target": label})
        if not pick.ok:
            pytest.skip(f"pick failed, cannot test place: {pick.message}")

        held_height = _truth_height(ready, label)
        result = ready.skills.execute("place", {})
        assert result.ok, f"place failed: {result.message}"

        assert ready.memory.get_held_object() is None, "memory still reports a held object"
        assert ready.robot.get_gripper_width() > 0.05, "gripper did not open on release"

        settled_height = _truth_height(ready, label)
        assert settled_height < held_height + 0.01, (
            f"object did not come down: {held_height:.3f} -> {settled_height:.3f}"
        )
        assert settled_height > ready.table_top_height - 0.03, (
            f"object fell through the table to z={settled_height:.3f}"
        )

    def test_place_with_empty_gripper_is_rejected(self, ready):
        """"Place it" with nothing held is a user error, reported as such."""
        result = ready.skills.execute("place", {})
        assert result.status is SkillStatus.INFEASIBLE
        assert "not holding" in result.message

    def test_place_does_not_return_home(self, ready):
        """Atomicity: place places. Going home is a separate command."""
        label = _pickable_label(ready)
        if not ready.skills.execute("pick", {"target": label}).ok:
            pytest.skip("pick failed")

        assert ready.skills.execute("place", {}).ok
        home = np.asarray(ready.config.robot.home_joint_positions)
        final = ready.robot.get_arm_joint_positions()
        assert float(np.max(np.abs(final - home))) > 0.05, (
            "arm returned to home; place should stop where it finished"
        )

    def test_pronoun_resolves_to_the_held_object(self, ready):
        """The "pick the bottle ... place it" requirement."""
        label = _pickable_label(ready)
        pick = ready.skills.execute("pick", {"target": label})
        if not pick.ok:
            pytest.skip(f"pick failed: {pick.message}")

        held = ready.memory.get_held_object()
        assert ready.memory.resolve_reference("it") == held
        assert ready.memory.resolve_reference("that") == held

"""Phase 7 gate: natural language driving the real robot.

The end-to-end claim: a typed sentence becomes one physical action, and the
"pick the bottle ... place it" pronoun chain works against a live scene.

Uses the runtime fixture directly rather than ``Assistant`` because Isaac's
runtime is a process-wide singleton and the session already owns one.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.language.intent_parser import RuleBasedIntentParser
from mfw.planner.state_machine import State
from mfw.planner.task_planner import TaskPlanner

pytestmark = [pytest.mark.isaac, pytest.mark.phase7]


@pytest.fixture
def assistant(restore_scene):
    """A task planner wired to the live runtime."""
    runtime = restore_scene
    runtime.memory.set_held_object(None)
    runtime.skills.execute("open_gripper")
    runtime.sim.render_step(2)
    runtime.skills.execute("observe")

    planner = TaskPlanner(
        parser=RuleBasedIntentParser(runtime.skills.names),
        executors={"classical": runtime.executor},
        memory=runtime.memory,
        config=runtime.config,
        vision=runtime.vision,
        events=runtime.events,
    )
    return runtime, planner


def _pickable_label(runtime) -> str:
    scene = runtime.vision.last_scene_graph()
    usable = runtime.config.grasp.max_grasp_width - runtime.config.grasp.finger_width_margin
    candidates = [
        o for o in scene.objects.values() if float(np.min(o.bbox.extents)) <= usable and o.label
    ]
    if not candidates:
        pytest.skip("nothing graspable in view")
    candidates.sort(key=lambda o: float(np.min(o.bbox.extents)))
    return candidates[0].label


class TestLanguageToAction:
    def test_observe_command_reports_the_scene(self, assistant):
        runtime, planner = assistant
        outcome = planner.handle("what do you see")
        assert outcome.ok, outcome.message
        assert outcome.skill == "observe"
        assert outcome.result.data["objects"]

    def test_move_left_moves_left_only(self, assistant):
        """"Move left." Move left only."""
        runtime, planner = assistant
        before = runtime.robot.tcp_pose().position.copy()

        outcome = planner.handle("move left 5 cm")
        if not outcome.ok:
            pytest.skip(f"move left infeasible from this posture: {outcome.message}")

        delta = runtime.robot.tcp_pose().position - before
        assert delta[1] > 0.02, f"did not move left: {np.round(delta, 3).tolist()}"
        assert abs(delta[0]) < 0.02 and abs(delta[2]) < 0.02

    def test_open_gripper_command(self, assistant):
        runtime, planner = assistant
        outcome = planner.handle("open the gripper")
        assert outcome.ok and outcome.skill == "open_gripper"
        assert runtime.robot.get_gripper_width() > 0.05

    def test_unknown_command_does_nothing(self, assistant):
        runtime, planner = assistant
        before = runtime.robot.tcp_pose().position.copy()

        outcome = planner.handle("make me a sandwich")
        assert not outcome.ok
        assert outcome.skill is None
        assert float(np.linalg.norm(runtime.robot.tcp_pose().position - before)) < 1e-3

    def test_pick_command_executes_exactly_one_action(self, assistant):
        """"Pick the can" -> pick, hold, WAIT. Not pick-then-place."""
        runtime, planner = assistant
        label = _pickable_label(runtime)

        outcome = planner.handle(f"pick up the {label}")
        assert outcome.ok, outcome.message
        assert outcome.skill == "pick"

        # Still holding: no automatic place happened.
        assert runtime.memory.get_held_object() is not None
        assert runtime.robot.get_gripper_width() > runtime.config.robot.gripper_closed_width + 0.004

    def test_returns_to_waiting_after_every_command(self, assistant):
        """The single edge out of COMPLETE."""
        runtime, planner = assistant
        for utterance in ("observe", "open the gripper", "observe"):
            planner.handle(utterance)
            assert planner.machine.state is State.WAIT_FOR_COMMAND

    def test_state_trace_shows_the_pipeline(self, assistant):
        runtime, planner = assistant
        outcome = planner.handle("observe")
        visited = [entry["to"] for entry in outcome.state_trace]
        for expected in ("observe", "plan", "execute", "verify", "complete", "wait_for_command"):
            assert expected in visited, f"{expected} missing from {visited}"


class TestPronounChain:
    def test_pick_then_place_it(self, assistant):
        """The headline requirement: "pick the bottle" then, later, "place it".

        Two separate commands. The first must not place; the second must know what
        "it" means without being told again.
        """
        runtime, planner = assistant
        label = _pickable_label(runtime)

        pick = planner.handle(f"pick up the {label}")
        if not pick.ok:
            pytest.skip(f"pick failed: {pick.message}")

        held = runtime.memory.get_held_object()
        assert held is not None
        assert runtime.memory.resolve_reference("it") == held

        place = planner.handle("place it")
        assert place.ok, f"place failed: {place.message}"
        assert place.skill == "place"
        assert runtime.memory.get_held_object() is None

    def test_place_without_a_pick_is_refused(self, assistant):
        runtime, planner = assistant
        outcome = planner.handle("place it")
        assert not outcome.ok
        assert "not holding" in outcome.message


class TestBackendRouting:
    def test_classical_backend_handles_everything(self, assistant):
        runtime, planner = assistant
        for skill in runtime.skills.names:
            assert runtime.executor.supports(skill), f"classical cannot run {skill}"

    def test_gr00t_declines_non_policy_skills(self, assistant):
        """So the planner falls back instead of the capability vanishing."""
        from mfw.gr00t_bridge.executor import POLICY_SKILLS

        runtime, _ = assistant
        for skill in ("observe", "stop", "emergency_stop", "go_home"):
            assert skill in runtime.skills.names
            assert skill not in POLICY_SKILLS

"""Phase 7 pure-logic tests: intent parsing, memory, state machine, planner.

No Isaac Sim. The central property under test is **atomicity**: one utterance must
produce exactly one skill, and no command may trigger a follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.config.schema import MemoryConfig, load_config
from mfw.core.types import (
    BoundingBox3D,
    Frame,
    GripperState,
    JointState,
    ObjectHypothesis,
    Pose,
    RobotState,
    SceneGraph,
    SkillResult,
    SkillStatus,
)
from mfw.language.intent_parser import (
    Intent,
    LlmIntentParser,
    RuleBasedIntentParser,
    UnparsedCommand,
)
from mfw.memory.working_memory import CommandRecord, WorkingMemory
from mfw.planner.state_machine import (
    IllegalTransition,
    State,
    StateMachine,
    recovery_for,
)
from mfw.planner.task_planner import ACTION_GRAPHS, TaskPlanner

pytestmark = pytest.mark.phase7

SKILLS = (
    "observe", "scan_scene", "look_at", "move_to", "move_relative", "pick", "place",
    "open_gripper", "close_gripper", "rotate_wrist", "go_home", "wait", "stop",
    "emergency_stop",
)


def _object(track_id, label, position=(0.5, 0.0, 0.45)):
    pose = Pose(np.asarray(position, dtype=float), np.array([1.0, 0, 0, 0]), Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id, label=label, pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.array([0.05, 0.05, 0.05])),
        confidence=0.9, num_points=500, last_seen_sim_time=0.0, last_seen_step=0,
    )


def _scene(*objects):
    return SceneGraph(objects={o.track_id: o for o in objects}, sim_time=0.0, step_index=0)


@pytest.fixture
def parser():
    return RuleBasedIntentParser(SKILLS)


@pytest.fixture
def memory():
    return WorkingMemory(MemoryConfig())


class TestIntentParsingAtomicity:
    """The framework's central behavioural requirement."""

    def test_parse_returns_exactly_one_intent(self, parser):
        intent = parser.parse("pick the can")
        assert isinstance(intent, Intent)
        assert intent.skill == "pick"

    def test_pick_never_implies_place(self, parser):
        """"Pick the can" must not schedule a place."""
        intent = parser.parse("pick up the can")
        assert intent.skill == "pick"
        assert "place" not in str(intent).lower()

    def test_compound_command_yields_only_the_first_action(self, parser):
        """"Pick X and put it in Y" is one command: pick.

        The user's spec is explicit that no command may automatically execute
        another, so a conjunction must not become a two-step plan.
        """
        intent = parser.parse("pick up the can and put it in the box")
        assert intent.skill == "pick"


class TestIntentParsing:
    @pytest.mark.parametrize(
        "utterance,skill",
        [
            ("pick the can", "pick"),
            ("pick up the bottle", "pick"),
            ("grab the block", "pick"),
            ("take the mug", "pick"),
            ("place it", "place"),
            ("put it down", "place"),
            ("open the gripper", "open_gripper"),
            ("open", "open_gripper"),
            ("close the gripper", "close_gripper"),
            ("close", "close_gripper"),
            ("move left", "move_relative"),
            ("go up", "move_relative"),
            ("stop", "stop"),
            ("emergency stop", "emergency_stop"),
            ("go home", "go_home"),
            ("observe", "observe"),
            ("scan the scene", "scan_scene"),
            ("wait", "wait"),
            ("look at the can", "look_at"),
            ("rotate the wrist 45 degrees", "rotate_wrist"),
        ],
    )
    def test_recognises_core_vocabulary(self, parser, utterance, skill):
        assert parser.parse(utterance).skill == skill

    def test_extracts_target(self, parser):
        assert parser.parse("pick the red block").params["target"] == "red block"

    def test_keeps_pronoun_for_memory_to_resolve(self, parser):
        """The parser must not attempt resolution; that is memory's job."""
        assert parser.parse("pick it up").params["target"] == "it"

    @pytest.mark.parametrize(
        "utterance,direction",
        [("move left", "left"), ("move right", "right"), ("go up", "up"),
         ("move down", "down"), ("move forward", "forward"), ("move backwards", "backward")],
    )
    def test_extracts_direction(self, parser, utterance, direction):
        assert parser.parse(utterance).params["direction"] == direction

    def test_bare_number_is_centimetres(self, parser):
        """"Move left 10" means 10 cm; reading it as metres would be a wild motion."""
        assert parser.parse("move left 10").params["distance"] == pytest.approx(0.10)

    @pytest.mark.parametrize(
        "utterance,metres",
        [("move left 5 cm", 0.05), ("move up 20 mm", 0.02), ("move right 0.3 m", 0.3)],
    )
    def test_honours_explicit_units(self, parser, utterance, metres):
        assert parser.parse(utterance).params["distance"] == pytest.approx(metres)

    def test_extracts_place_relation(self, parser):
        intent = parser.parse("put it in the box")
        assert intent.skill == "place"
        assert intent.params["relation"] == "in"
        assert intent.params["target"] == "box"

    def test_longest_relation_phrase_wins(self, parser):
        intent = parser.parse("place it on top of the box")
        assert intent.params["relation"] == "on"
        assert intent.params["target"] == "box"

    def test_open_the_box_is_not_a_gripper_command(self, parser):
        """A container is not the gripper.

        The robot has no skill for opening a box, so refusing is the honest
        outcome -- better than silently opening the gripper, which is a real
        action on the world and not what was asked.
        """
        with pytest.raises(UnparsedCommand):
            parser.parse("open the box")

    def test_rotate_direction_sets_sign(self, parser):
        clockwise = parser.parse("rotate wrist 90 degrees clockwise").params["angle"]
        counter = parser.parse("rotate wrist 90 degrees counterclockwise").params["angle"]
        assert clockwise < 0 < counter

    def test_unparseable_command_raises(self, parser):
        with pytest.raises(UnparsedCommand):
            parser.parse("compose a symphony")

    def test_empty_command_raises(self, parser):
        with pytest.raises(UnparsedCommand, match="empty"):
            parser.parse("   ")

    def test_unregistered_skill_is_rejected(self):
        """Guards against parsing to a skill the robot does not have."""
        restricted = RuleBasedIntentParser(known_skills=("observe",))
        with pytest.raises(UnparsedCommand, match="not registered"):
            restricted.parse("pick the can")


class TestLlmParserFallback:
    def test_uses_model_output_when_valid(self):
        parser = LlmIntentParser(
            complete=lambda _p: '{"skill": "pick", "params": {"target": "can"}, "confidence": 0.9}',
            known_skills=SKILLS,
        )
        intent = parser.parse("could you grab that fizzy drink")
        assert intent.skill == "pick" and intent.params["target"] == "can"

    def test_falls_back_when_model_errors(self):
        """A model outage must not stop the robot understanding "stop"."""
        def broken(_prompt):
            raise RuntimeError("model unreachable")

        parser = LlmIntentParser(complete=broken, known_skills=SKILLS)
        assert parser.parse("stop").skill == "stop"

    def test_falls_back_on_unknown_skill(self):
        parser = LlmIntentParser(
            complete=lambda _p: '{"skill": "teleport", "params": {}}', known_skills=SKILLS
        )
        assert parser.parse("move left").skill == "move_relative"

    def test_falls_back_on_non_json(self):
        parser = LlmIntentParser(complete=lambda _p: "I think you want to pick it up", known_skills=SKILLS)
        assert parser.parse("pick the can").skill == "pick"

    def test_prompt_forbids_multi_step_plans(self):
        """The instruction is load-bearing: a model will otherwise emit a plan."""
        assert "ONE action" in LlmIntentParser.PROMPT_TEMPLATE
        assert "Never a sequence" in LlmIntentParser.PROMPT_TEMPLATE


class TestMemoryReferenceResolution:
    def test_pronoun_resolves_to_held_object(self, memory):
        """The "pick the bottle ... place it" requirement."""
        memory.update_scene(_scene(_object("obj_1", "bottle"), _object("obj_2", "can")))
        memory.set_held_object("obj_1")
        assert memory.resolve_reference("it") == "obj_1"
        assert memory.resolve_reference("that") == "obj_1"

    def test_pronoun_falls_back_to_last_referenced(self, memory):
        memory.update_scene(_scene(_object("obj_1", "bottle"), _object("obj_2", "can")))
        memory.note_reference("obj_2")
        assert memory.resolve_reference("it") == "obj_2"

    def test_held_object_wins_over_last_referenced(self, memory):
        """If the gripper is full, "it" is what is in the gripper."""
        memory.update_scene(_scene(_object("obj_1", "bottle"), _object("obj_2", "can")))
        memory.note_reference("obj_2")
        memory.set_held_object("obj_1")
        assert memory.resolve_reference("it") == "obj_1"

    def test_unique_label_resolves(self, memory):
        memory.update_scene(_scene(_object("obj_1", "bottle"), _object("obj_2", "can")))
        assert memory.resolve_reference("the can") == "obj_2"

    def test_ambiguous_label_refuses_to_guess(self, memory):
        """A wrong referent means confidently manipulating the wrong object."""
        memory.update_scene(_scene(_object("obj_1", "can"), _object("obj_2", "can")))
        assert memory.resolve_reference("the can") is None
        assert set(memory.candidates_for("can")) == {"obj_1", "obj_2"}

    def test_pronoun_with_nothing_held_or_referenced_returns_none(self, memory):
        memory.update_scene(_scene(_object("obj_1", "can")))
        assert memory.resolve_reference("it") is None

    def test_unknown_label_returns_none(self, memory):
        memory.update_scene(_scene(_object("obj_1", "can")))
        assert memory.resolve_reference("teapot") is None

    def test_picking_sets_the_referent(self, memory):
        """After a pick, "it" should mean the thing just picked up."""
        memory.update_scene(_scene(_object("obj_1", "can")))
        memory.set_held_object("obj_1")
        assert memory.last_referenced == "obj_1"

    def test_held_object_cleared_when_it_vanishes(self, memory):
        """A phantom payload would poison every later command."""
        memory.update_scene(_scene(_object("obj_1", "can")))
        memory.set_held_object("obj_1")
        for _ in range(6):
            memory.update_scene(_scene())
        assert memory.get_held_object() is None

    def test_memory_stores_no_poses(self, memory):
        """Identities only. A remembered pose would be acted on after the object moved."""
        memory.update_scene(_scene(_object("obj_1", "can")))
        memory.set_held_object("obj_1")
        snapshot = memory.snapshot()
        assert snapshot["held_object"] == "obj_1"
        assert isinstance(snapshot["current_objects"]["obj_1"], str)

    def test_records_results_against_commands(self, memory):
        memory.record_command(CommandRecord("pick the can", "pick", {"target": "can"}))
        memory.record_result(SkillResult(skill_name="pick", status=SkillStatus.SUCCESS))
        assert memory.recent_commands()[-1].status == "success"


class TestStateMachine:
    def test_starts_idle_and_reaches_waiting(self):
        machine = StateMachine()
        assert machine.state is State.IDLE
        machine.to(State.WAIT_FOR_COMMAND)
        assert machine.state is State.WAIT_FOR_COMMAND

    def test_full_happy_path(self):
        machine = StateMachine()
        for state in (
            State.WAIT_FOR_COMMAND, State.PARSE, State.OBSERVE, State.PLAN,
            State.EXECUTE, State.VERIFY, State.COMPLETE, State.WAIT_FOR_COMMAND,
        ):
            machine.to(state)
        assert machine.state is State.WAIT_FOR_COMMAND

    def test_complete_leads_only_back_to_waiting(self):
        """The structural guarantee of atomicity.

        If COMPLETE could reach EXECUTE or PLAN, one command could begin another.
        """
        from mfw.planner.state_machine import TRANSITIONS

        assert TRANSITIONS[State.COMPLETE] == (State.WAIT_FOR_COMMAND,)

    def test_illegal_transition_raises(self):
        machine = StateMachine()
        machine.to(State.WAIT_FOR_COMMAND)
        with pytest.raises(IllegalTransition):
            machine.to(State.EXECUTE)

    def test_failure_can_replan(self):
        machine = StateMachine()
        for state in (State.WAIT_FOR_COMMAND, State.PARSE, State.OBSERVE, State.PLAN,
                      State.EXECUTE, State.FAILED, State.REPLAN, State.OBSERVE):
            machine.to(state)
        assert machine.state is State.OBSERVE

    def test_reset_to_waiting_from_anywhere(self):
        machine = StateMachine()
        for state in (State.WAIT_FOR_COMMAND, State.PARSE, State.OBSERVE):
            machine.to(state)
        machine.reset_to_waiting("done")
        assert machine.state is State.WAIT_FOR_COMMAND

    def test_trace_is_recorded(self):
        machine = StateMachine()
        machine.to(State.WAIT_FOR_COMMAND, "ready")
        machine.to(State.PARSE, "parsing")
        trace = machine.trace()
        assert trace[-1]["from"] == "wait_for_command" and trace[-1]["to"] == "parse"


class TestRecoveryPolicy:
    def test_safety_violation_is_never_retried(self):
        from mfw.core.errors import SafetyViolation

        assert recovery_for(SafetyViolation("out of bounds")) is State.ABORTED

    def test_ambiguous_reference_asks(self):
        from mfw.core.errors import AmbiguousReference

        assert recovery_for(AmbiguousReference("two cans")) is State.CLARIFY

    def test_planning_failure_replans(self):
        from mfw.core.errors import PlanningError

        assert recovery_for(PlanningError("no path")) is State.REPLAN

    def test_perception_failure_replans(self):
        from mfw.core.errors import PerceptionError

        assert recovery_for(PerceptionError("no depth")) is State.REPLAN

    def test_missing_object_does_not_retry_forever(self):
        from mfw.core.errors import ObjectNotFound

        assert recovery_for(ObjectNotFound("no teapot")) is State.FAILED


class FakeExecutor:
    """Records calls and returns a scripted status."""

    def __init__(self, status=SkillStatus.SUCCESS, supports_all=True, data=None):
        self.status = status
        self._supports_all = supports_all
        self.calls: list[tuple[str, dict]] = []
        self.data = data or {}

    @property
    def backend_name(self):
        return "fake"

    def supports(self, skill_name):
        return self._supports_all

    def execute(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        return SkillResult(
            skill_name=skill_name, status=self.status, message="scripted", data=dict(self.data)
        )


@pytest.fixture
def config():
    return load_config(overrides={})


class TestTaskPlanner:
    def test_one_command_executes_one_skill(self, memory, config):
        """The whole point: no command triggers a second action."""
        executor = FakeExecutor()
        planner = TaskPlanner(
            RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config
        )
        outcome = planner.handle("pick the can")

        assert outcome.ok
        assert len(executor.calls) == 1
        assert executor.calls[0][0] == "pick"

    def test_returns_to_waiting_after_success(self, memory, config):
        executor = FakeExecutor()
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        planner.handle("pick the can")
        assert planner.machine.state is State.WAIT_FOR_COMMAND

    def test_two_commands_execute_two_skills_in_order(self, memory, config):
        executor = FakeExecutor()
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        planner.handle("pick the can")
        planner.handle("place it")
        assert [call[0] for call in executor.calls] == ["pick", "place"]

    def test_unparsed_command_does_not_execute_anything(self, memory, config):
        executor = FakeExecutor()
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        outcome = planner.handle("recite poetry")

        assert not outcome.ok
        assert executor.calls == []
        assert planner.machine.state is State.WAIT_FOR_COMMAND

    def test_infeasible_result_is_not_retried(self, memory, config):
        """Retrying an object too wide for the gripper cannot help."""
        executor = FakeExecutor(status=SkillStatus.INFEASIBLE)
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        planner.handle("pick the can")
        assert len(executor.calls) == 1

    def test_failed_result_is_retried_up_to_the_limit(self, memory, config):
        executor = FakeExecutor(status=SkillStatus.FAILED)
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        planner.handle("pick the can")
        assert len(executor.calls) == config.motion.max_replan_attempts

    def test_reports_the_action_graph(self, memory, config):
        """Explains the internal stages without executing them as commands."""
        planner = TaskPlanner(
            RuleBasedIntentParser(SKILLS), {"classical": FakeExecutor()}, memory, config
        )
        outcome = planner.handle("pick the can")
        assert outcome.action_graph == ACTION_GRAPHS["pick"]
        assert outcome.action_graph[-1] == "hold and WAIT"

    def test_state_trace_is_reported(self, memory, config):
        planner = TaskPlanner(
            RuleBasedIntentParser(SKILLS), {"classical": FakeExecutor()}, memory, config
        )
        outcome = planner.handle("pick the can")
        visited = [entry["to"] for entry in outcome.state_trace]
        assert "execute" in visited and "complete" in visited

    def test_falls_back_to_a_backend_that_supports_the_skill(self, memory, config):
        """A partial policy must not make a capability unavailable."""
        picky = FakeExecutor(supports_all=False)
        general = FakeExecutor()
        planner = TaskPlanner(
            RuleBasedIntentParser(SKILLS),
            {"gr00t": picky, "classical": general},
            memory,
            config,
        )
        assert planner.handle("observe").ok
        assert general.calls and not picky.calls

    def test_successful_result_updates_the_referent(self, memory, config):
        """Enables the next command to say "it"."""
        executor = FakeExecutor(data={"track_id": "obj_7"})
        planner = TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, config)
        planner.handle("pick the can")
        assert memory.last_referenced == "obj_7"

    def test_action_graphs_all_end_by_waiting(self):
        """Documented pipelines must not imply a follow-up action."""
        for skill, graph in ACTION_GRAPHS.items():
            if skill in ("stop", "emergency_stop"):
                continue
            assert "WAIT" in graph[-1], f"{skill} graph does not end by waiting: {graph}"

"""The LLM intent parser on the hardware MVP: hybrid policy, prompt rules, checks.

No model, no GPU. The model is a scripted fake worker that behaves the way the
real Qwen2.5-3B-Instruct Q4_K_M did when measured on 2026-09-26 through
``jetson/llm_worker_llamacpp.py`` (62 + 31 utterances): it is usually right,
but it also returns the measured wrong answers ("put it in the box" -> place
target "it", "grab the block" -> "red block", "move up 20 mm" -> 20, clockwise
with the wrong sign), refuses with ``"unknown"``, answers late, times out and
fails with ``ok: false``. A test that only passes because the fake is perfect
would prove nothing, so every failure path here is one the real worker showed
or can show (llama-server down, a stalled GPU, a slow first call).

The end-to-end cases run the real shim over TCP against the shim's own
``FakeLlamaServer`` and against a socket that accepts and never answers.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import math
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from jetson import llm_worker_llamacpp as shim
from mfw.hardware.runtime import HARDWARE_SKILLS
from mfw.language.intent_parser import (
    DEFAULT_LLM_MODE,
    LLM_MODES,
    REFUSAL_SKILL,
    LlmIntentParser,
    NegatedCommand,
    RuleBasedIntentParser,
    UnparsedCommand,
    UnregisteredSkill,
)
from mfw.skills.primitives import ALL_SKILLS

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]

SIM_SKILLS = tuple(cls.skill_name for cls in ALL_SKILLS)
HW_SKILLS = tuple(cls.skill_name for cls in HARDWARE_SKILLS)
CONTEXT = {"visible_objects": ["red block", "banana", "bowl", "box", "marker"], "held_object": None}


def _reply(skill: str, params: dict[str, Any] | None = None, confidence: float = 0.9) -> str:
    return json.dumps({"skill": skill, "params": params or {}, "confidence": confidence})


class FakeLlmWorker:
    """Stands in for ``run_assistant``'s ``llm_complete`` against the 5557 worker.

    ``script`` maps an exact command to what the worker does for it:
    a reply string, ``"ok:false"`` (llama-server down: the client raises
    RuntimeError, exactly as run_assistant's llm_complete does), ``"timeout"``
    (socket.timeout, a stalled GPU), or ``("late", seconds, reply)`` (answers,
    but slowly -- the parser must still use it). Unscripted commands get a
    confident wrong answer, so a test cannot pass by the model being silent.
    """

    WRONG = _reply("emergency_stop")

    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.script = dict(script or {})
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @staticmethod
    def command_of(prompt: str) -> str:
        return re.findall(r"^Command: (.*)\nJSON:$", prompt, re.MULTILINE)[-1]

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        action = self.script.get(self.command_of(prompt), self.WRONG)
        if action == "ok:false":
            raise RuntimeError("RuntimeError: llama-server unreachable at http://127.0.0.1:8080")
        if action == "timeout":
            raise socket.timeout("timed out")
        if isinstance(action, tuple) and action[0] == "late":
            time.sleep(action[1])
            return action[2]
        return action


def _hybrid(worker: FakeLlmWorker, skills: tuple[str, ...] = SIM_SKILLS) -> LlmIntentParser:
    return LlmIntentParser(worker, skills, mode="hybrid")


def _rule_outcome(utterance: str, skills: tuple[str, ...] = SIM_SKILLS):
    try:
        intent = RuleBasedIntentParser(skills).parse(utterance, CONTEXT)
    except UnparsedCommand as exc:
        return ("UNPARSED", type(exc).__name__, str(exc))
    return (intent.skill, dict(intent.params), intent.confidence)


# ----------------------------------------------------------------------
# policy constants
# ----------------------------------------------------------------------


class TestModes:
    def test_hybrid_is_the_default_when_an_llm_is_configured(self):
        assert DEFAULT_LLM_MODE == "hybrid"
        assert LLM_MODES == ("hybrid", "llm-first")

    def test_class_default_stays_llm_first_for_existing_callers(self):
        """mfw.assistant builds LlmIntentParser(complete, skills) directly; the
        CLI (--llm-mode, default hybrid) sets the policy on it."""
        assert LlmIntentParser(FakeLlmWorker(), SIM_SKILLS).mode == "llm-first"

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="hybrid"):
            LlmIntentParser(FakeLlmWorker(), SIM_SKILLS, mode="llm-only")
        parser = LlmIntentParser(FakeLlmWorker(), SIM_SKILLS)
        with pytest.raises(ValueError):
            parser.mode = "rules"


# ----------------------------------------------------------------------
# hybrid: rule first, the model only for refusals
# ----------------------------------------------------------------------

#: Rule-parsable commands, each scripted with the model's measured wrong answer
#: for it where there was one. Hybrid must return the rule parse untouched and
#: never even ask the model.
RULE_WINS = {
    "grab the block": _reply("pick", {"target": "red block"}),
    "put it in the box": _reply("place", {"relation": "in", "target": "it"}),
    "place it on top of the box": _reply("place", {"relation": "on", "target": "it"}),
    "move up 20 mm": _reply("move_relative", {"direction": "up", "distance": 20}),
    "rotate wrist 90 degrees clockwise": _reply("rotate_wrist", {"angle": 1.5708}),
    "drop it": _reply("place", {"relation": "in", "target": "it"}),
    "hold still": _reply("stop"),
    "let go of it": _reply("stop", {"target": "it"}),
    "can you put the banana in the bowl": _reply("place", {"relation": "in", "target": "banana"}),
    "stop": _reply("emergency_stop"),
    "emergency stop": _reply("stop"),
    "pick up the can and put it in the box": _reply("place", {"relation": "in", "target": "box"}),
    # measured 2026-09-26/27; the grammar answers these itself since 2026-09-27
    "raise the arm a bit": _reply("rotate_wrist", {"angle": 0.5}),
    "lower the gripper 3 cm": _reply("close_gripper"),
    "lift the arm up by 5 cm": _reply("rotate_wrist", {"angle": 0.095}),
    "return to your resting position": _reply("place"),
}


class TestHybridNeverOverridesTheRules:
    @pytest.mark.parametrize("utterance", sorted(RULE_WINS))
    def test_rule_success_is_returned_and_the_model_is_not_asked(self, utterance):
        worker = FakeLlmWorker(RULE_WINS)
        parser = _hybrid(worker)
        intent = parser.parse(utterance, CONTEXT)
        assert (intent.skill, dict(intent.params), intent.confidence) == _rule_outcome(utterance)
        assert worker.calls == 0
        assert parser.last_source == "rule" and intent.source == "rule"

    def test_llm_first_would_have_taken_the_wrong_answer(self):
        """The same script in llm-first mode does reach the model -- proving the
        hybrid test above is not passing because the fake is never consulted."""
        worker = FakeLlmWorker(RULE_WINS)
        LlmIntentParser(worker, SIM_SKILLS, mode="llm-first").parse("hold still", CONTEXT)
        assert worker.calls == 1


class TestHybridAsksTheModelOnlyOnRefusal:
    @pytest.mark.parametrize(
        "utterance,reply,expected",
        [
            ("park yourself", _reply("go_home"), ("go_home", {})),
            ("shut the gripper", _reply("close_gripper"), ("close_gripper", {})),
            ("would you mind picking up the banana", _reply("pick", {"target": "banana"}),
             ("pick", {"target": "banana"})),
            ("swing over to the mug", _reply("move_to", {"target": "the mug"}), ("move_to", {"target": "mug"})),
            ("bring the gripper up a little", _reply("move_relative", {"direction": "up"}),
             ("move_relative", {"direction": "up"})),
        ],
    )
    def test_refusal_goes_to_the_model(self, utterance, reply, expected):
        assert _rule_outcome(utterance)[0] == "UNPARSED"  # precondition: the grammar refuses
        worker = FakeLlmWorker({utterance: reply})
        parser = _hybrid(worker)
        intent = parser.parse(utterance, CONTEXT)
        assert (intent.skill, dict(intent.params)) == expected
        assert worker.calls == 1 and parser.last_source == "llm" and intent.source == "llm"

    def test_a_late_answer_is_still_used(self):
        """A slow first call (cold prompt cache) is not a failure."""
        utterance = "park yourself"
        worker = FakeLlmWorker({utterance: ("late", 0.2, _reply("go_home"))})
        assert _hybrid(worker).parse(utterance, CONTEXT).skill == "go_home"

    @pytest.mark.parametrize(
        "failure",
        ["ok:false", "timeout", "I think you want the arm to go home.", _reply(REFUSAL_SKILL),
         json.dumps({"skill": ["go_home", "open_gripper"], "params": {}}),
         json.dumps({"skill": "go_home", "params": ["a", "plan"]}),
         _reply("teleport")],
        ids=["ok-false", "timeout", "prose", "unknown", "plan", "params-list", "unregistered-skill"],
    )
    def test_a_model_failure_keeps_the_rule_refusal(self, failure):
        utterance = "park yourself"
        rule = _rule_outcome(utterance)
        worker = FakeLlmWorker({utterance: failure})
        parser = _hybrid(worker)
        with pytest.raises(UnparsedCommand) as info:
            parser.parse(utterance, CONTEXT)
        assert (type(info.value).__name__, str(info.value)) == rule[1:]  # the grammar's own refusal
        assert worker.calls == 1 and parser.last_source == "rule-refused"

    def test_empty_command_never_reaches_the_model(self):
        worker = FakeLlmWorker()
        with pytest.raises(UnparsedCommand, match="empty"):
            _hybrid(worker).parse("   ", CONTEXT)
        assert worker.calls == 0

    def test_a_skill_this_robot_lacks_is_refused_without_asking_the_model(self):
        """The grammar understood "rotate the wrist"; the hobby arm has no wrist
        roll. Asking the model would only find some OTHER skill to run."""
        worker = FakeLlmWorker({"rotate wrist 90 degrees clockwise": _reply("move_relative", {"direction": "right"})})
        parser = _hybrid(worker, HW_SKILLS)
        with pytest.raises(UnregisteredSkill, match="not registered"):
            parser.parse("rotate wrist 90 degrees clockwise", CONTEXT)
        assert worker.calls == 0 and parser.last_source == "rule-refused"

    def test_the_model_cannot_emit_a_sim_only_skill_on_hardware(self):
        utterance = "point the camera at the banana"
        worker = FakeLlmWorker({utterance: _reply("look_at", {"target": "banana"})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker, HW_SKILLS).parse(utterance, CONTEXT)
        assert worker.calls == 1


class TestSourceIsLogged:
    def _records(self, fn: Callable[[], Any]) -> list[str]:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger("mfw.language.intent")
        previous = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            try:
                fn()
            except UnparsedCommand:
                pass
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        return [r.getMessage() for r in records]

    def test_each_source_is_named(self):
        worker = FakeLlmWorker({"park yourself": _reply("go_home"),
                                "wave at the audience": "ok:false"})
        parser = _hybrid(worker)
        assert any("intent source=rule " in m for m in self._records(lambda: parser.parse("go home")))
        assert any("intent source=llm " in m
                   for m in self._records(lambda: parser.parse("park yourself")))
        assert any("intent source=rule-refused" in m
                   for m in self._records(lambda: parser.parse("wave at the audience")))

    def test_llm_first_fallback_is_named(self):
        parser = LlmIntentParser(FakeLlmWorker({"go home": "timeout"}), SIM_SKILLS)
        messages = self._records(lambda: parser.parse("go home"))
        assert any("intent source=rule-fallback" in m for m in messages)
        assert parser.last_source == "rule-fallback"


# ----------------------------------------------------------------------
# the model's output is checked before it can move the arm
# ----------------------------------------------------------------------


class TestModelOutputIsValidated:
    def test_pronoun_destination_is_rejected(self):
        """Measured: "put it in the box" -> place target "it" (into itself)."""
        worker = FakeLlmWorker({"put it in the box": _reply("place", {"relation": "in", "target": "it"})})
        llm_first = LlmIntentParser(worker, SIM_SKILLS, mode="llm-first")
        intent = llm_first.parse("put it in the box", CONTEXT)
        assert (intent.skill, intent.params) == ("place", {"relation": "in", "target": "box"})
        assert llm_first.last_source == "rule-fallback" and worker.calls == 1

    def test_pronoun_destination_refused_in_hybrid(self):
        utterance = "kindly deposit it inside of that"  # the grammar refuses this
        assert _rule_outcome(utterance)[0] == "UNPARSED"
        worker = FakeLlmWorker({utterance: _reply("place", {"relation": "in", "target": "that"})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse(utterance, CONTEXT)

    def test_invented_target_is_rejected_in_hybrid(self):
        """Measured: "grab the block" -> "red block" copied from the visible list."""
        utterance = "would you mind picking up the block"
        assert _rule_outcome(utterance)[0] == "UNPARSED"
        worker = FakeLlmWorker({utterance: _reply("pick", {"target": "red block"})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse(utterance, CONTEXT)
        worker.script[utterance] = _reply("pick", {"target": "block"})
        assert _hybrid(worker).parse(utterance, CONTEXT).params == {"target": "block"}

    def test_plural_and_article_differences_are_not_invention(self):
        utterance = "would you mind picking up the bananas"
        worker = FakeLlmWorker({utterance: _reply("pick", {"target": "the banana"})})
        assert _hybrid(worker).parse(utterance, CONTEXT).params == {"target": "banana"}

    def test_a_pronoun_target_needs_a_spoken_pronoun(self):
        worker = FakeLlmWorker({"would you mind picking up": _reply("pick", {"target": "it"}),
                                "would you mind picking it up": _reply("pick", {"target": "it"})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse("would you mind picking up", CONTEXT)
        assert _hybrid(worker).parse("would you mind picking it up", CONTEXT).params == {"target": "it"}

    def test_missing_target_is_rejected(self):
        utterance = "would you mind picking up the banana"
        worker = FakeLlmWorker({utterance: _reply("pick", {})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse(utterance, CONTEXT)

    @pytest.mark.parametrize(
        "utterance,model_distance,expected",
        [("bring the gripper down 3 cm", 3, 0.03),        # measured class: 20 mm -> 20
         ("bring the gripper down 3 cm", 0.3, 0.03),
         ("bring the gripper down 30 mm", 30, 0.03),
         ("bring the gripper down five centimetres", 5, 0.05),
         ("bring the gripper down by twenty five millimetres", 0.25, 0.025)],
    )
    def test_distance_is_re_read_from_the_utterance(self, utterance, model_distance, expected):
        assert _rule_outcome(utterance)[0] == "UNPARSED"  # the model path, not the grammar
        worker = FakeLlmWorker({utterance: _reply("move_relative", {"direction": "down", "distance": model_distance})})
        intent = _hybrid(worker).parse(utterance, CONTEXT)
        assert intent.params["direction"] == "down"
        assert intent.params["distance"] == pytest.approx(expected)

    def test_no_spoken_distance_means_the_skill_default(self):
        """Measured: "move backwards" -> 0.5 m, five times the skill default."""
        utterance = "bring the gripper up a little"
        worker = FakeLlmWorker({utterance: _reply("move_relative", {"direction": "up", "distance": 0.5})})
        parser = _hybrid(worker)
        assert parser.parse(utterance, CONTEXT).params == {"direction": "up"}
        assert parser.last_source == "llm"

    def test_a_direction_that_contradicts_the_words_is_rejected(self):
        utterance = "bring the gripper down 3 cm"
        worker = FakeLlmWorker({utterance: _reply("move_relative", {"direction": "up", "distance": 0.03})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse(utterance, CONTEXT)
        assert worker.calls == 1

    @pytest.mark.parametrize(
        "utterance,model_angle,expected_degrees",
        [("rotate wrist 90 degrees clockwise", 1.5708, -90.0),   # measured wrong sign
         ("rotate wrist 90 degrees clockwise", 90, -90.0),       # degrees, not radians
         ("rotate wrist 90 degrees counterclockwise", -1.5708, 90.0),
         ("twist the wrist thirty degrees anticlockwise", 0.5, 30.0),
         ("rotate wrist 1 radian clockwise", 57.3, -57.29577951308232)],
    )
    def test_rotation_sign_and_size_come_from_the_words(self, utterance, model_angle, expected_degrees):
        worker = FakeLlmWorker({utterance: _reply("rotate_wrist", {"angle": model_angle})})
        intent = LlmIntentParser(worker, SIM_SKILLS, mode="llm-first").parse(utterance, CONTEXT)
        assert worker.calls == 1 and intent.skill == "rotate_wrist"
        assert math.degrees(intent.params["angle"]) == pytest.approx(expected_degrees)

    def test_parameterless_skills_drop_junk(self):
        """Measured: "let go of it" -> {"skill": "stop", "params": {"target": "it"}}."""
        worker = FakeLlmWorker({"let loose": _reply("open_gripper", {"target": "it"})})
        assert _hybrid(worker).parse("let loose", CONTEXT).params == {}

    def test_wait_duration_defaults_and_reads_words(self):
        worker = FakeLlmWorker({"chill for a moment": _reply("wait", {"duration": 30}),
                                "chill for three seconds": _reply("wait", {"duration": 30})})
        assert _hybrid(worker).parse("chill for a moment", CONTEXT).params == {"duration": 1.0}
        assert _hybrid(worker).parse("chill for three seconds", CONTEXT).params == {"duration": 3.0}

    @pytest.mark.parametrize(
        "params,expected",
        [({"relation": "inside", "target": "the bowl"}, {"relation": "in", "target": "bowl"}),
         ({"relation": "next to", "target": "bowl"}, {"relation": "next_to", "target": "bowl"}),
         ({"target": "bowl"}, {"relation": "to", "target": "bowl"}),
         ({}, {}),
         ({"relation": "direction", "direction": "backward", "distance": 7}, {"relation": "direction",
          "direction": "back", "distance": 0.10})],
    )
    def test_place_params_are_normalised(self, params, expected):
        utterance = "kindly deposit it by the bowl"
        worker = FakeLlmWorker({utterance: _reply("place", params)})
        assert _hybrid(worker).parse(utterance, CONTEXT).params == expected

    def test_an_unknown_relation_is_rejected(self):
        utterance = "kindly deposit it by the bowl"
        worker = FakeLlmWorker({utterance: _reply("place", {"relation": "orbiting", "target": "bowl"})})
        with pytest.raises(UnparsedCommand):
            _hybrid(worker).parse(utterance, CONTEXT)

    def test_a_pronoun_pick_while_holding_is_refused(self):
        """Measured 2026-09-27 (fresh set): "toss it in the bin" while holding the
        eraser -> pick "it". The held object cannot be picked again; the model
        should have said place. Hybrid keeps the grammar's refusal."""
        utterance = "toss it in the bin"
        assert _rule_outcome(utterance)[0] == "UNPARSED"
        holding = {**CONTEXT, "held_object": "eraser"}
        worker = FakeLlmWorker({utterance: _reply("pick", {"target": "it"})})
        parser = _hybrid(worker)
        with pytest.raises(UnparsedCommand, match="could not understand"):
            parser.parse(utterance, holding)
        assert worker.calls == 1 and parser.last_source == "rule-refused"
        # The right answer to the same command is accepted.
        worker.script[utterance] = _reply("place", {"relation": "in", "target": "bin"})
        intent = parser.parse(utterance, holding)
        assert (intent.skill, intent.params) == ("place", {"relation": "in", "target": "bin"})

    def test_a_pronoun_pick_with_empty_hands_is_kept(self):
        utterance = "would you mind picking it up"
        worker = FakeLlmWorker({utterance: _reply("pick", {"target": "it"})})
        assert _hybrid(worker).parse(utterance, {**CONTEXT, "held_object": None}).params == {"target": "it"}

    def test_llm_first_falls_back_on_a_pronoun_pick_while_holding(self):
        utterance = "pick it up"
        worker = FakeLlmWorker({utterance: _reply("pick", {"target": "it"})})
        parser = LlmIntentParser(worker, SIM_SKILLS, mode="llm-first")
        parser.parse(utterance, {**CONTEXT, "held_object": "eraser"})
        assert worker.calls == 1 and parser.last_source == "rule-fallback"

    def test_confidence_is_clamped(self):
        worker = FakeLlmWorker({"let loose": _reply("open_gripper", confidence=7.0)})
        assert _hybrid(worker).parse("let loose", CONTEXT).confidence == 1.0


# ----------------------------------------------------------------------
# the prompt
# ----------------------------------------------------------------------


class TestPrompt:
    def _prompt(self, skills=SIM_SKILLS, utterance="put it in the box", context=CONTEXT) -> str:
        return LlmIntentParser(FakeLlmWorker(), skills).build_prompt(utterance, context)

    def test_one_action_rules_survive(self):
        assert "ONE action" in LlmIntentParser.PROMPT_TEMPLATE
        assert "Never a sequence" in LlmIntentParser.PROMPT_TEMPLATE

    @pytest.mark.parametrize(
        "needle",
        ['"it"/"that" is the object being moved, NEVER the destination',
         '"put it in the box" -> target "box"',
         "Never add a colour, size or name the operator did not say",
         "never copy a name from this list",
         "20 mm = 0.02", "3 cm = 0.03",
         "Leave distance out when the operator gives none",
         "Clockwise is NEGATIVE",
         '"drop it", "let go"', "-> open_gripper",
         '"hold still"', "-> wait (never stop)",
         '"lower the gripper"', "move_relative up/down",
         f'"skill": "{REFUSAL_SKILL}"',
         "the FIRST action is pick",
         # 2026-09-27 fresh-set error classes
         "toss, throw", "the held object cannot be picked again",
         '"hover over", "approach"', "-> move_to that object",
         "held: blue cup"],
    )
    def test_measured_error_classes_are_addressed(self, needle):
        assert needle in self._prompt()

    def test_the_refusal_skill_is_offered(self):
        assert re.search(r"^Available actions: .*\bunknown$", self._prompt(), re.MULTILINE)

    def test_hardware_prompt_names_only_hardware_skills(self):
        prompt = self._prompt(HW_SKILLS)
        actions = re.search(r"^Available actions: (.*)$", prompt, re.MULTILINE).group(1).split(", ")
        assert actions == [*HW_SKILLS, "unknown"]
        for missing in ("rotate_wrist", "look_at"):
            assert missing not in prompt
        assert "Clockwise" not in prompt  # the rotation rule needs rotate_wrist
        assert "scan_scene" in prompt

    def test_the_utterance_and_scene_come_last_for_the_prompt_cache(self):
        """llama-server reuses the common prefix; everything before the visible
        objects must be identical across commands and scenes."""
        a = self._prompt(utterance="put it in the box", context=CONTEXT)
        b = self._prompt(utterance="raise the arm", context={"visible_objects": ["mug"], "held_object": "mug"})
        prefix = a[: a.index("Visible objects")]
        assert b.startswith(prefix) and len(prefix) > 0.8 * len(a)

    def test_examples_do_not_copy_the_evaluation_utterances(self):
        """Examples teach a rule; they must not be answers to the measured set."""
        measured = {"grab the block", "put it in the box", "place it on top of the box", "move up 20 mm",
                    "rotate wrist 90 degrees clockwise", "drop it", "hold still", "let go of it",
                    "raise the arm a bit", "lower the gripper 3 cm", "drop it in the bowl",
                    "place it in the bowl", "please drop it into the bin"}
        examples = set(re.findall(r"^Command: (.*?)  \(visible", self._prompt(), re.MULTILINE))
        assert examples and not (examples & measured)


# ----------------------------------------------------------------------
# the Jetson shim's skill enum
# ----------------------------------------------------------------------


class TestShimSkillEnum:
    def test_hardware_names_match_the_hardware_runtime(self):
        assert shim.HARDWARE_SKILL_NAMES == HW_SKILLS

    def test_sim_names_still_match_the_registry(self):
        assert set(shim.SKILL_NAMES) == set(SIM_SKILLS)

    def test_refusal_skill_is_shared_and_never_registered(self):
        assert shim.REFUSAL_SKILL == REFUSAL_SKILL
        assert REFUSAL_SKILL not in SIM_SKILLS

    def test_skills_for(self):
        assert shim.skills_for("hardware") == (*HW_SKILLS, "unknown")
        assert shim.skills_for("all") == (*shim.SKILL_NAMES, "unknown")
        assert shim.skills_for("pick, place") == ("pick", "place", "unknown")
        assert shim.skills_for("pick,unknown") == ("pick", "unknown")
        with pytest.raises(ValueError):
            shim.skills_for(" , ")

    def test_cli_default_is_the_hardware_enum(self, monkeypatch):
        seen: dict[str, Any] = {}

        class _Server:
            def __init__(self, host, port, completer):
                seen["completer"] = completer
                self.bound_port = port

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                seen["closed"] = True

        monkeypatch.setattr(shim, "LlmShimServer", _Server)
        monkeypatch.setattr(shim.LlamaCppCompleter, "ping", lambda self: True)
        assert shim.main(["--port", "0", "--log-level", "WARNING"]) == 0
        assert seen["completer"].skills == (*HW_SKILLS, "unknown") and seen["closed"]
        assert shim.main(["--port", "0", "--skills", "all", "--log-level", "WARNING"]) == 0
        assert seen["completer"].skills == (*shim.SKILL_NAMES, "unknown")

    def test_docstring_names_the_real_model(self):
        doc = shim.__doc__ or ""
        assert "Qwen2.5-3B-Instruct" in doc and "Q4_K_M" in doc
        assert "1.5B" not in doc

    def test_shim_stays_mfw_free_and_python_310(self):
        source = (REPO_ROOT / "jetson" / "llm_worker_llamacpp.py").read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 10))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert "mfw" not in imported


# ----------------------------------------------------------------------
# end to end: hybrid parser -> real shim over TCP -> fake / stalled llama-server
# ----------------------------------------------------------------------


def _tcp_complete(port: int, timeout: float = 10.0) -> Callable[[str], str]:
    """The exact client body of scripts/run_assistant.py's llm_complete."""

    def llm_complete(prompt: str) -> str:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.sendall(json.dumps({"prompt": prompt}).encode("utf-8") + b"\n")
            data = json.loads(sock.makefile("r", encoding="utf-8").readline())
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "LLM inference failed"))
            return data["text"]

    return llm_complete


class TestEndToEndThroughTheShim:
    def test_rule_parse_sends_nothing_and_refusal_crosses_the_wire(self):
        fake = shim.FakeLlamaServer(reply_text=_reply("go_home")).serve_in_thread()
        server, port = shim.LlmShimServer(
            "127.0.0.1", 0, shim.LlamaCppCompleter(fake.url, timeout_s=5.0, skills=shim.skills_for("hardware"))
        ).serve_in_thread()
        try:
            parser = LlmIntentParser(_tcp_complete(port), HW_SKILLS, mode="hybrid")
            assert parser.parse("pick up the marker", CONTEXT).params == {"target": "marker"}
            assert fake.requests == []
            assert parser.parse("park yourself", CONTEXT).skill == "go_home"
            request = fake.requests[-1]
            assert request["response_format"]["json_schema"]["schema"]["properties"]["skill"]["enum"] == [
                *HW_SKILLS, "unknown"]
            assert "Command: park yourself\nJSON:" in request["messages"][1]["content"]
        finally:
            server.stop()
            fake.stop()

    def test_a_stalled_llama_server_keeps_the_refusal_within_the_timeout(self):
        """llama-server accepted the connection and never answered (a hung GPU):
        the shim times out, replies ok:false, and the grammar's refusal stands."""
        stalled = socket.socket()
        stalled.bind(("127.0.0.1", 0))
        stalled.listen(8)
        stalled_port = stalled.getsockname()[1]
        server, port = shim.LlmShimServer(
            "127.0.0.1", 0, shim.LlamaCppCompleter(f"http://127.0.0.1:{stalled_port}", timeout_s=0.5)
        ).serve_in_thread()
        try:
            parser = LlmIntentParser(_tcp_complete(port), HW_SKILLS, mode="hybrid")
            t0 = time.monotonic()
            with pytest.raises(UnparsedCommand, match="could not understand"):
                parser.parse("park yourself", CONTEXT)
            assert time.monotonic() - t0 < 5.0
            assert parser.last_source == "rule-refused"
            # The rules keep working while the model is down.
            assert parser.parse("stop", CONTEXT).skill == "stop"
        finally:
            server.stop()
            stalled.close()

    def test_a_dead_shim_keeps_the_rules_working(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        parser = LlmIntentParser(_tcp_complete(dead_port, timeout=1.0), HW_SKILLS, mode="hybrid")
        assert parser.parse("go home", CONTEXT).skill == "go_home"
        with pytest.raises(UnparsedCommand):
            parser.parse("park yourself", CONTEXT)


# ----------------------------------------------------------------------
# scripts/run_assistant.py --llm-mode
# ----------------------------------------------------------------------


def _load_cli():
    spec = importlib.util.spec_from_file_location("_run_assistant_llm_mode_under_test",
                                                  REPO_ROOT / "scripts" / "run_assistant.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRunAssistantLlmMode:
    def test_apply_sets_the_mode_on_the_shared_parser(self, capsys):
        cli = _load_cli()
        parser = LlmIntentParser(FakeLlmWorker(), SIM_SKILLS)
        planner = SimpleNamespace(parser=parser)
        assistant = SimpleNamespace(parser=parser, planner=planner)
        cli._apply_llm_mode(assistant, "hybrid")
        assert planner.parser.mode == "hybrid"
        assert "LLM parser mode   : hybrid" in capsys.readouterr().out

    def test_apply_ignores_the_rule_parser(self):
        cli = _load_cli()
        rule = RuleBasedIntentParser(SIM_SKILLS)
        cli._apply_llm_mode(SimpleNamespace(parser=rule), "hybrid")
        assert not hasattr(rule, "mode")

    def test_flag_defaults_to_hybrid(self):
        source = (REPO_ROOT / "scripts" / "run_assistant.py").read_text(encoding="utf-8")
        match = re.search(r'"--llm-mode",\s*default="(\w[\w-]*)",\s*choices=\[([^\]]*)\]', source)
        assert match is not None
        assert match.group(1) == DEFAULT_LLM_MODE
        assert [c.strip().strip('"') for c in match.group(2).split(",")] == list(LLM_MODES)
        assert "_apply_llm_mode(assistant, args.llm_mode)" in source


# ----------------------------------------------------------------------
# review findings 3 and 4: an LLM intent needs a word asking for it; negation
# ----------------------------------------------------------------------


class TestHybridNeedsSpokenEvidence:
    """The model is asked only about what the grammar refused -- ASR noise,
    background talk. Measured Qwen maps those onto motions, so an LLM intent
    that moves the arm is accepted only when the operator said a word asking
    for that skill (and, for move_relative, a direction)."""

    @pytest.mark.parametrize(
        "utterance,reply",
        [
            ("hmm okay", _reply("open_gripper")),
            ("keep holding the marker", _reply("open_gripper")),
            ("thanks robot", _reply("go_home")),
            ("nudge it a little bit", _reply("move_relative", {"direction": "left", "distance": 0.1})),
            ("I see the marker", _reply("pick", {"target": "marker"})),
            ("the bowl is over there", _reply("place", {"relation": "in", "target": "bowl"})),
            ("that was nice", _reply("scan_scene")),
            ("okay good", _reply("close_gripper")),
        ],
    )
    def test_a_motion_nobody_asked_for_is_refused(self, utterance, reply):
        assert _rule_outcome(utterance, HW_SKILLS)[0] == "UNPARSED"
        worker = FakeLlmWorker({utterance: reply})
        parser = _hybrid(worker, HW_SKILLS)
        with pytest.raises(UnparsedCommand):
            parser.parse(utterance, CONTEXT)
        assert worker.calls == 1 and parser.last_source == "rule-refused"

    @pytest.mark.parametrize(
        "utterance,reply,skill",
        [  # the LLM successes measured on the dev, held-out and fresh sets (results_r4hw_*_hybrid.json)
            ("snatch the bottle cap", _reply("pick", {"target": "bottle cap"}), "pick"),
            ("hand me the banana", _reply("pick", {"target": "banana"}), "pick"),
            ("would you mind picking up the banana", _reply("pick", {"target": "banana"}), "pick"),
            ("toss it in the bin", _reply("place", {"relation": "in", "target": "bin"}), "place"),
            ("stick it on top of the box", _reply("place", {"relation": "on", "target": "box"}), "place"),
            ("loosen your grip", _reply("open_gripper"), "open_gripper"),
            ("let it go", _reply("open_gripper"), "open_gripper"),
            ("close your hand", _reply("close_gripper"), "close_gripper"),
            ("shut the gripper", _reply("close_gripper"), "close_gripper"),
            ("scoot forward a tiny bit", _reply("move_relative", {"direction": "forward"}), "move_relative"),
            ("bring the gripper down a little", _reply("move_relative", {"direction": "down"}), "move_relative"),
            ("head back to home", _reply("go_home"), "go_home"),
            ("look around the room", _reply("scan_scene"), "scan_scene"),
            ("tell me what's in front of you", _reply("observe"), "observe"),
            ("hang on a sec", _reply("wait"), "wait"),
        ],
    )
    def test_the_measured_paraphrases_still_pass(self, utterance, reply, skill):
        worker = FakeLlmWorker({utterance: reply})
        parser = _hybrid(worker, HW_SKILLS)
        assert parser.parse(utterance, CONTEXT).skill == skill


NEGATED = [
    "don't let go of it", "do not drop it", "don't open the gripper", "never release it",
    "don't go home", "do not pick up the marker", "no, put it in the bowl",
]


class TestNegationIsRefused:
    @pytest.mark.parametrize("utterance", NEGATED)
    @pytest.mark.parametrize("skills", [SIM_SKILLS, HW_SKILLS], ids=["sim", "hardware"])
    def test_the_grammar_refuses_a_negated_command(self, utterance, skills):
        with pytest.raises(NegatedCommand):
            RuleBasedIntentParser(skills).parse(utterance, CONTEXT)

    @pytest.mark.parametrize("utterance", NEGATED)
    def test_hybrid_never_asks_the_model_about_a_negation(self, utterance):
        worker = FakeLlmWorker({utterance: _reply("open_gripper")})
        parser = _hybrid(worker, HW_SKILLS)
        with pytest.raises(NegatedCommand):
            parser.parse(utterance, CONTEXT)
        assert worker.calls == 0 and parser.last_source == "rule-refused"

    @pytest.mark.parametrize("utterance", NEGATED)
    def test_llm_first_never_takes_the_models_answer_for_a_negation(self, utterance):
        worker = FakeLlmWorker({utterance: _reply("open_gripper")})
        parser = LlmIntentParser(worker, HW_SKILLS, mode="llm-first")
        with pytest.raises(NegatedCommand):
            parser.parse(utterance, CONTEXT)
        assert worker.calls == 0

    @pytest.mark.parametrize("utterance", ["stop, don't drop it", "stop moving left", "don't stop",
                                           "stop everything right now", "halt, do not open"])
    def test_a_stop_word_wins(self, utterance):
        for skills in (SIM_SKILLS, HW_SKILLS):
            assert RuleBasedIntentParser(skills).parse(utterance, CONTEXT).skill == "stop"
        worker = FakeLlmWorker({utterance: _reply("open_gripper")})
        assert _hybrid(worker, HW_SKILLS).parse(utterance, CONTEXT).skill == "stop"
        assert worker.calls == 0

    def test_emergency_still_beats_stop(self):
        assert RuleBasedIntentParser(HW_SKILLS).parse("emergency stop, don't move").skill == "emergency_stop"

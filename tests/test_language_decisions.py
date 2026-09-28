"""Language decisions (2026-09-28): an LLM-chosen release of a held object is confirmed.

In hybrid mode the language model is asked only about what the grammar
refused -- ASR noise, background talk, unusual phrasing -- and the measured
Qwen2.5-3B maps such noise onto releases ("hmm okay" -> open_gripper), while the
evidence check accepts "drop" for open_gripper. So when the MODEL (not the rule
parser) chose ``open_gripper`` or ``place`` while something is held, the
Assistant asks first in interactive and voice modes ("Did you mean: open the
gripper and drop the blue can? say yes or no"), and refuses in scripted ``-c``
mode, where nobody can answer.

No model and no runtime: the model is a scripted fake completion that answers
the way the real one did (and a confident wrong answer for anything
unscripted), and the executor is tests/test_grounding.py's HoldAwareExecutor,
which models what the gate protects -- a release empties the gripper.
"""

from __future__ import annotations

import json
import re

import pytest

from mfw.assistant import (
    LLM_RELEASE_SKILLS,
    confirm_by_asking,
    describe_release,
    release_question,
)
from mfw.language.intent_parser import Intent, LlmIntentParser, RuleBasedIntentParser
from mfw.planner.task_planner import TaskPlanner
from tests.test_grounding import SKILLS, HoldAwareExecutor, _assistant, _holding

pytestmark = pytest.mark.phase9


def _reply(skill, params=None):
    return json.dumps({"skill": skill, "params": params or {}, "confidence": 0.9})


class FakeModel:
    """A completion callable: the scripted answer for a command, else a confident wrong one."""

    WRONG = _reply("emergency_stop")

    def __init__(self, script):
        self.script = dict(script)
        self.commands: list[str] = []

    def __call__(self, prompt):
        command = re.findall(r"^Command: (.*)\nJSON:$", prompt, re.MULTILINE)[-1]
        self.commands.append(command)
        return self.script.get(command, self.WRONG)


SCRIPT = {
    "loosen your grip": _reply("open_gripper"),
    "let it go": _reply("open_gripper"),
    "toss it in the box": _reply("place", {"relation": "in", "target": "box"}),
    "pop it back": _reply("place", {}),
    "head back to home": _reply("go_home"),
}


def _llm_assistant(held="obj_002", script=SCRIPT, mode="hybrid"):
    """A bare Assistant (no runtime) whose parser is the hybrid LLM parser, gate installed."""
    memory, scene = _holding(held)
    executor = HoldAwareExecutor(memory, scene)
    assistant = _assistant(executor, memory=memory)
    model = FakeModel(script)
    assistant.parser = LlmIntentParser(model, SKILLS, mode=mode)
    assistant.planner = TaskPlanner(assistant.parser, assistant.executors, memory, assistant.config)
    assistant._install_llm_gate()
    return assistant, executor, memory, model


class Asker:
    """An ``ask`` callback with scripted replies; records every question."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.questions: list[str] = []

    def __call__(self, question):
        self.questions.append(question)
        return self.replies.pop(0) if self.replies else None


QUESTION = "Did you mean: open the gripper and drop the blue can? say yes or no"


class TestInteractiveAndVoiceAsk:
    def test_yes_releases(self):
        assistant, executor, memory, _ = _llm_assistant()
        ask = Asker("yes")
        outcome = assistant.command_with_clarification("loosen your grip", ask)
        assert ask.questions == [QUESTION]
        assert outcome.ok and executor.calls == [("open_gripper", {})]

    def test_a_spoken_yes_is_read_by_whole_words(self):
        assistant, executor, _, _ = _llm_assistant()
        assert assistant.command_with_clarification("loosen your grip", Asker("yeah go ahead")).ok
        assert executor.calls == [("open_gripper", {})]

    @pytest.mark.parametrize("reply", ["no", "nope", "cancel", "no, keep holding it"])
    def test_no_keeps_the_object(self, reply):
        assistant, executor, memory, _ = _llm_assistant()
        outcome = assistant.command_with_clarification("loosen your grip", Asker(reply))
        assert not outcome.ok and outcome.message.startswith("cancelled")
        assert executor.calls == [] and memory.get_held_object() == "obj_002"

    def test_no_reply_is_not_consent(self):
        assistant, executor, _, _ = _llm_assistant()
        outcome = assistant.command_with_clarification("loosen your grip", Asker())
        assert not outcome.ok and executor.calls == []

    def test_an_unclear_reply_is_asked_again_then_given_up(self):
        assistant, executor, _, _ = _llm_assistant()
        ask = Asker("hmm", "I know", "what")
        outcome = assistant.command_with_clarification("loosen your grip", ask)
        assert len(ask.questions) == 3 and ask.questions[1] == f"Please say yes or no. {QUESTION}"
        assert not outcome.ok and executor.calls == []

    def test_an_unclear_reply_then_yes(self):
        assistant, executor, _, _ = _llm_assistant()
        assert assistant.command_with_clarification("loosen your grip", Asker("hmm", "yes")).ok
        assert executor.calls == [("open_gripper", {})]

    def test_an_llm_place_names_the_held_object_and_the_destination(self):
        assistant, executor, _, _ = _llm_assistant()
        ask = Asker("yes")
        assert assistant.command_with_clarification("toss it in the box", ask).ok
        assert ask.questions == ["Did you mean: place the blue can in the box? say yes or no"]
        assert executor.calls == [("place", {"relation": "in", "target": "box"})]

    def test_the_ask_is_released_after_the_command(self):
        """A later scripted command must not reuse an interactive ``ask``."""
        assistant, executor, _, _ = _llm_assistant()
        assistant.command_with_clarification("head back to home", Asker())
        outcome = assistant.command("loosen your grip")
        assert not outcome.ok and "cannot be confirmed" in outcome.message


class TestScriptedRefuses:
    @pytest.mark.parametrize("utterance,action", [
        ("loosen your grip", "open the gripper and drop the blue can"),
        ("toss it in the box", "place the blue can in the box"),
        ("pop it back", "put the blue can back where it was picked up"),
    ])
    def test_scripted_mode_cannot_confirm_so_it_refuses(self, utterance, action):
        assistant, executor, memory, _ = _llm_assistant()
        outcome = assistant.command(utterance)
        assert not outcome.ok and executor.calls == []
        assert action in outcome.message and "cannot be confirmed" in outcome.message
        assert memory.get_held_object() == "obj_002"
        assert assistant.parser.last_source == "llm-unconfirmed"


class TestWhatIsNotAsked:
    def test_a_rule_parsed_release_is_not_asked(self):
        assistant, executor, _, model = _llm_assistant()
        ask = Asker()
        assert assistant.command_with_clarification("open the gripper", ask).ok
        assert ask.questions == [] and model.commands == [] and executor.calls == [("open_gripper", {})]

    def test_an_llm_release_with_nothing_held_is_not_asked(self):
        assistant, executor, _, _ = _llm_assistant(held=None)
        ask = Asker()
        assert assistant.command("let it go").ok
        assert ask.questions == [] and executor.calls == [("open_gripper", {})]

    def test_an_llm_non_release_while_holding_is_not_asked(self):
        assistant, executor, _, _ = _llm_assistant()
        assert assistant.command("head back to home").ok
        assert executor.calls == [("go_home", {})]

    def test_the_rule_parser_assistant_has_no_gate(self):
        assistant, executor, _, _ = _llm_assistant()
        assistant.parser = RuleBasedIntentParser(SKILLS)
        assistant._install_llm_gate()  # a no-op for the grammar
        assert not hasattr(assistant.parser, "llm_gate")

    def test_a_preview_reports_the_release_without_asking_and_it_is_still_a_motion(self):
        """The voice loop previews first (needs_confirmation); that must not ask or refuse."""
        assistant, executor, _, _ = _llm_assistant()
        assert assistant.planned_skills("loosen your grip") == ["open_gripper"]
        assert assistant.needs_confirmation("loosen your grip") is True
        assert executor.calls == []
        # ...and the preview flag does not leak: the real command still refuses.
        assert not assistant.command("loosen your grip").ok

    def test_llm_first_mode_is_gated_too(self):
        assistant, executor, _, _ = _llm_assistant(
            mode="llm-first", script={"open the gripper": _reply("open_gripper")})
        outcome = assistant.command("open the gripper")
        assert not outcome.ok and executor.calls == []


class TestHelpers:
    def test_release_skills(self):
        assert LLM_RELEASE_SKILLS == {"open_gripper", "place"}

    @pytest.mark.parametrize("intent,held,text", [
        (Intent("open_gripper"), "marker", "open the gripper and drop the marker"),
        (Intent("open_gripper"), None, "open the gripper and drop what it is holding"),
        (Intent("place", {"relation": "in", "target": "bowl"}), "marker", "place the marker in the bowl"),
        (Intent("place", {"relation": "next_to", "target": "box"}), "marker", "place the marker next to the box"),
        (Intent("place", {"relation": "left_of", "target": "box"}), "marker",
         "place the marker to the left of the box"),
        (Intent("place", {}), "marker", "put the marker back where it was picked up"),
        (Intent("place", {"relation": "direction", "direction": "left", "distance": 0.1}), "marker",
         "place the marker 10 cm to the left"),
    ])
    def test_describe_release(self, intent, held, text):
        assert describe_release(intent, held) == text

    def test_the_question(self):
        assert release_question("open the gripper and drop the marker") == (
            "Did you mean: open the gripper and drop the marker? say yes or no")

    def test_confirm_by_asking(self):
        assert confirm_by_asking("q?", Asker("yes")) is True
        assert confirm_by_asking("q?", Asker("no")) is False
        assert confirm_by_asking("q?", Asker(None)) is False
        assert confirm_by_asking("q?", Asker("yes no")) is False  # mixed is never consent
        assert confirm_by_asking("q?", Asker("maybe", "maybe", "yes"), max_questions=2) is False


class TestHesitationIsNotConsent:
    """Fixer, 2026-09-28. was: "okay wait" answering "Did you mean: open the
    gripper and drop the blue can?" read as yes (the "okay") and dropped it."""

    @pytest.mark.parametrize("reply", ["okay wait", "yes, hold on", "ok hang on", "sure, one sec", "wait"])
    def test_a_hesitant_reply_is_asked_again_and_never_releases(self, reply):
        assistant, executor, memory, _ = _llm_assistant()
        ask = Asker(reply, reply, reply)
        outcome = assistant.command_with_clarification("loosen your grip", ask)
        assert len(ask.questions) == 3
        assert not outcome.ok and executor.calls == [] and memory.get_held_object() == "obj_002"

    def test_hesitation_then_a_plain_yes_releases(self):
        assistant, executor, _, _ = _llm_assistant()
        assert assistant.command_with_clarification("loosen your grip", Asker("okay wait", "yes")).ok
        assert executor.calls == [("open_gripper", {})]

    def test_no_wait_is_still_no(self):
        assistant, executor, _, _ = _llm_assistant()
        outcome = assistant.command_with_clarification("loosen your grip", Asker("no wait"))
        assert not outcome.ok and outcome.message.startswith("cancelled") and executor.calls == []


class TestANegationThatStartsWithNoIsNeverStripped:
    """Fixer, 2026-09-28. The correction marker used to swallow the "no" that
    starts a negation: "no need to drop it" opened the gripper (rule parse, so
    the LLM gate never saw it), and in hybrid mode "no going home" reached the
    model as "going home" and homed. Now each is refused as a negation and the
    model is never asked. The executor is HoldAwareExecutor: a release would
    empty the gripper, so "still holding" is a real check.
    """

    @pytest.mark.parametrize("utterance", [
        "no need to drop it", "no need to open the gripper", "no, no need to drop it",
        "actually, no need to drop it", "no dropping it", "no letting go", "no going home",
        "no more moving left", "no need to go home", "no need to move left",
        "no need to put it in the bowl",
    ])
    def test_scripted_and_confirmed_alike_nothing_moves(self, utterance):
        script = dict(SCRIPT, **{"going home": _reply("go_home"),
                                 "more moving left": _reply("move_relative", {"direction": "left"}),
                                 "dropping it": _reply("open_gripper"),
                                 "need to drop it": _reply("open_gripper")})
        assistant, executor, memory, model = _llm_assistant(script=script)
        outcome = assistant.command(utterance)
        assert not outcome.ok and executor.calls == [] and model.commands == []
        # The voice/interactive path with an honest "yes" to any question: still nothing.
        outcome = assistant.command_with_clarification(utterance, Asker("yes", "yes", "yes"))
        assert not outcome.ok and executor.calls == [] and model.commands == []
        assert memory.get_held_object() == "obj_002"

    def test_a_real_correction_still_reaches_the_model_without_its_marker(self):
        """The fix keeps what the marker is for: "nope, toss it in the box"."""
        assistant, executor, _, model = _llm_assistant()
        assert assistant.command_with_clarification("nope, toss it in the box", Asker("yes")).ok
        assert model.commands == ["toss it in the box"]
        assert executor.calls == [("place", {"relation": "in", "target": "box"})]


class TestTheGateQuotesTheOperator:
    """Fixer, 2026-09-28. was: the refusal quoted the command after the marker
    ("the language model read 'toss it in the box' ..."), not what was said."""

    def test_the_refusal_quotes_the_whole_utterance(self):
        assistant, executor, _, model = _llm_assistant()
        outcome = assistant.command("nope, toss it in the box")
        assert model.commands == ["toss it in the box"]  # the model still sees no marker
        assert not outcome.ok and executor.calls == []
        assert "'nope, toss it in the box'" in outcome.message

    def test_a_declined_confirmation_quotes_it_too(self):
        assistant, _, _, _ = _llm_assistant()
        outcome = assistant.command_with_clarification("nope, toss it in the box", Asker("no"))
        assert "'nope, toss it in the box'" in outcome.message


#: The sim lane (Isaac, ``backend: sim``) builds the same Assistant, so both
#: language decisions reach it too. Recorded here as was -> now rows (fixer,
#: 2026-09-28; review finding "LLM gate installed for every Assistant"). The
#: Isaac suite is unaffected: tests/test_phase7_isaac.py drives TaskPlanner
#: with the rule parser directly and never goes through Assistant.clauses or
#: an LLM parser. Only ``run_assistant.py --llm`` in the sim lane sees row 1.
SIM_LANE_WAS_NOW = [
    ("let it go", "scripted -c with --llm, holding the blue can",
     "open_gripper executed (model-chosen release)", "refused: cannot be confirmed in scripted mode"),
    ("put the blue can in the box", "holding the blue can",
     "pick refused ('already holding something'), nothing placed", "place in the box"),
]


class TestSimLaneWasNow:
    def test_the_default_config_is_the_sim_lane(self):
        assistant, _, _, _ = _llm_assistant()
        assert assistant.config.backend == "sim"

    def test_row_1_an_llm_release_is_refused_in_scripted_mode(self):
        utterance, _, _, _ = SIM_LANE_WAS_NOW[0]
        assistant, executor, memory, _ = _llm_assistant()
        outcome = assistant.command(utterance)
        assert not outcome.ok and "cannot be confirmed" in outcome.message
        assert executor.calls == [] and memory.get_held_object() == "obj_002"
        # was: without the gate the same parser released the object.
        assistant.parser.llm_gate = None
        assert assistant.command(utterance).ok and executor.calls == [("open_gripper", {})]

    def test_row_2_a_transfer_of_the_held_object_is_its_place(self):
        utterance, _, _, _ = SIM_LANE_WAS_NOW[1]
        assistant, executor, _, _ = _llm_assistant()
        assistant.parser = RuleBasedIntentParser(SKILLS)
        assistant.planner = TaskPlanner(assistant.parser, assistant.executors, assistant.runtime.memory,
                                        assistant.config)
        assert assistant.clauses(utterance) == ["place it in the box"]
        assert assistant.command(utterance).ok
        assert executor.calls == [("place", {"relation": "in", "target": "box"})]

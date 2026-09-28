"""Phase 9 language tests: the three rule-parser fixes and the Parakeet engine.

No Isaac Sim, no microphone, no model download. The hardware lane relies on the
rule-based parser alone (no LLM fits the Jetson budget), so the three phrasings
that used to misparse -- and would have moved a real arm the wrong way -- are
pinned here, alongside a table of existing phrases that must keep mapping exactly
as before: a regex fix that silently changes a neighbouring mapping is worse than
the quirk it removed.

Three tables carry that contract (review finding F4 in docs/HARDWARE_REVIEW.md):

* ``HARDWARE_MAPPINGS`` -- every phrase whose mapping phase 9 changed, with the
  exact intent it produced before and after. The parser's own comment tables
  mirror it row for row; a mapping change that is not in this table is a
  regression, not a feature.
* ``RESTORED`` -- phrases the first phase 9 patterns swallowed by accident
  ("drop it in" with no target became a place; "take a picture frame" became an
  observe) and which now map exactly as before.
* ``UNCHANGED`` / ``REFUSED`` -- the frozen pre-phase-9 intents (and refusals)
  of the sim-lane vocabulary. The property check at the bottom takes every
  phrase straight out of tests/test_language_logic.py (never copies them): the
  parametrize tables through their marks, and every string literal handed to
  ``.parse(...)`` or ``.handle(...)`` through the module's AST. Each must be in
  one of these two tables and must still produce the frozen outcome, so a phrase
  added to the sim-lane suite without being frozen here fails, and the two
  suites cannot drift apart.

The speech worker is imported as a module straight from ``scripts/`` (its
top-level imports are stdlib only; sounddevice, numpy and every ASR backend are
imported lazily inside functions), so ``build_engine`` can be exercised without
an audio stack.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import logging
import math
import sys
import types
from pathlib import Path

import pytest

from mfw.language.intent_parser import (
    LlmIntentParser,
    NegatedCommand,
    RuleBasedIntentParser,
    UnparsedCommand,
    _RELATION_WORDS,
    split_transfer,
    strip_correction,
)
from tests import test_language_logic as sim_lane

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "speech_worker.py"

SKILLS = (
    "observe", "scan_scene", "look_at", "move_to", "move_relative", "pick", "place",
    "open_gripper", "close_gripper", "rotate_wrist", "go_home", "wait", "stop",
    "emergency_stop",
)


@pytest.fixture
def parser():
    return RuleBasedIntentParser(SKILLS)


# ----------------------------------------------------------------------
# the three fixes
# ----------------------------------------------------------------------


class TestDropWithDestinationIsPlace:
    """"Drop it in the bowl" is a place, not a gripper release over nothing."""

    @pytest.mark.parametrize(
        "utterance,relation,target",
        [
            ("drop it in the bowl", "in", "bowl"),
            ("drop it on the box", "on", "box"),
            ("drop it into the bowl", "in", "bowl"),
            # was here until 2026-09-28: ("drop the can next to the bowl", "next_to", "bowl").
            # A NAMED object is now a transfer (pick it, then place): see
            # TestDropANamedObjectIsATransfer below and HARDWARE_MAPPINGS.
            ("drop it on top of the box", "on", "box"),
        ],
    )
    def test_drop_with_destination_is_place(self, parser, utterance, relation, target):
        intent = parser.parse(utterance)
        assert intent.skill == "place"
        assert intent.params["relation"] == relation
        assert intent.params["target"] == target

    @pytest.mark.parametrize(
        "utterance",
        ["drop it", "drop", "let go", "release",
         # A trailing relation with nothing after it names no destination (F4).
         "drop it in", "drop in", "drop it inside", "drop it in the",
         # Nor does a word the robot cannot resolve to an object.
         "drop it in there", "drop it in here", "drop it in please", "drop it on top",
         "drop it in there please",
         # A pronoun names no destination either; HEAD released for these too.
         "drop it in it", "drop it on that one", "drop it in them", "drop the can on it",
         # Nor does a stacked relation word.
         "drop it on in it", "drop it in on that one"],
    )
    def test_bare_drop_stays_a_gripper_release(self, parser, utterance):
        """Without a destination there is nothing to plan; release immediately.

        A ``place`` with empty params would carry the object back to its pick
        origin, which is the opposite of what "drop it in" asks for.
        """
        intent = parser.parse(utterance)
        assert intent.skill == "open_gripper"
        assert intent.params == {}

    @pytest.mark.parametrize(
        "utterance,relation,target",
        [
            # A real word after the filler is a destination again.
            ("drop it in the bowl please", "in", "bowl"),
            ("drop it on the top shelf", "on", "top shelf"),
            ("drop it here in the box", "in", "box"),
        ],
    )
    def test_filler_before_a_real_target_does_not_hide_it(self, parser, utterance, relation, target):
        intent = parser.parse(utterance)
        assert intent.skill == "place"
        assert intent.params["relation"] == relation
        assert intent.params["target"] == target

    def test_first_clause_still_wins(self, parser):
        """The conjunction rule is unchanged: one utterance, one action."""
        intent = parser.parse("drop it in the bowl and then go home")
        assert intent.skill == "place"
        assert intent.params["target"] == "bowl"


class TestDropANamedObjectIsATransfer:
    """Language decisions stream, 2026-09-28: "drop <named object> <relation> <place>".

    was: ``place`` -- whatever the arm held went into the destination, whichever
    object was named ("drop the eraser in the bowl" while holding the marker put
    the marker in the bowl). now: a transfer, exactly like "put the can in the
    bowl": the parser's intent is the first action (pick the named object) and
    :func:`split_transfer` gives the Assistant both clauses. The Assistant runs
    only the place when the named object is the one held (tests/test_grounding.py).
    """

    @pytest.mark.parametrize(
        "utterance,obj,relation,target",
        [
            ("drop the can next to the bowl", "can", "next_to", "bowl"),
            ("drop the eraser in the bowl", "eraser", "in", "bowl"),
            ("drop the red block onto the box please", "red block", "on", "box"),
            ("please drop the marker into the bin", "marker", "in", "bin"),
            ("drop the can to the left of the bowl", "can", "left_of", "bowl"),
            ("drop off the can in the bowl", "can", "in", "bowl"),
            ("drop the can off in the bowl", "can", "in", "bowl"),
        ],
    )
    def test_named_object_is_picked_first(self, parser, utterance, obj, relation, target):
        intent = parser.parse(utterance)
        assert (intent.skill, intent.params) == ("pick", {"target": obj})
        split = split_transfer(utterance)
        assert split is not None and split.pick_clause is not None
        assert split.object_phrase == obj
        assert split.place_params == {"relation": relation, "target": target}

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            # A pronoun names the held object: a plain place, as before.
            ("drop it in the bowl", ("place", {"relation": "in", "target": "bowl"})),
            ("drop it off in the bowl", ("place", {"relation": "in", "target": "bowl"})),
            ("drop that next to the box", ("place", {"relation": "next_to", "target": "box"})),
            # No destination, a pronoun destination, or a direction: unchanged releases.
            ("drop the can", ("open_gripper", {})),
            ("drop the can on it", ("open_gripper", {})),
            ("drop the can to the left", ("open_gripper", {})),
            ("drop the can in there", ("open_gripper", {})),
        ],
    )
    def test_pronoun_and_destinationless_drops_are_unchanged(self, parser, utterance, expected):
        assert _same_outcome(_outcome(parser, utterance), expected)
        split = split_transfer(utterance)
        assert split is None or split.pick_clause is None


class TestCorrectionMarkers:
    """Language decisions stream, 2026-09-28: a leading correction is not a negation.

    was: "no, put it in the bowl" -> NegatedCommand (the operator had to say it
    again). now: the command after the marker is parsed. A negation AFTER the
    marker is still refused, and a stop word still wins.
    """

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("no, put it in the bowl", ("place", {"relation": "in", "target": "bowl"})),
            ("no put it in the bowl", ("place", {"relation": "in", "target": "bowl"})),
            ("nope, pick the marker", ("pick", {"target": "marker"})),
            ("nah, go home", ("go_home", {})),
            ("no no, open the gripper", ("open_gripper", {})),
            ("actually, place it on the box", ("place", {"relation": "on", "target": "box"})),
            ("actually no, place it on the box", ("place", {"relation": "on", "target": "box"})),
            ("sorry, pick the marker", ("pick", {"target": "marker"})),
            ("i mean pick the can", ("pick", {"target": "can"})),
            ("no, stop", ("stop", {})),
            ("no, don't stop", ("stop", {})),
        ],
    )
    def test_the_command_after_the_marker_is_parsed(self, parser, utterance, expected):
        assert _same_outcome(_outcome(parser, utterance), expected)

    @pytest.mark.parametrize(
        "utterance",
        ["no, don't drop it", "no don't drop it", "nope, do not go home", "actually, don't open the gripper",
         "don't open the gripper", "do not go home", "no", "no no", "nope", "sorry no",
         "the marker, not the block", "put it in the bowl, no"],
    )
    def test_a_negated_command_is_still_refused(self, parser, utterance):
        with pytest.raises(NegatedCommand):
            parser.parse(utterance)

    def test_a_marker_before_something_that_is_not_a_command_is_just_unparsed(self, parser):
        with pytest.raises(UnparsedCommand) as info:
            parser.parse("nope, the red one")
        assert not isinstance(info.value, NegatedCommand)

    def test_a_correction_keeps_a_transfer_whole(self, parser):
        """"actually, put the marker in the bowl" hid the transfer: place in bowl, marker ignored."""
        assert parser.parse("actually, put the marker in the bowl").params == {"target": "marker"}
        split = split_transfer("no, put the marker in the bowl")
        assert split is not None and split.clauses == ("pick the marker", "place it in the bowl")

    def test_strip_correction_leaves_a_bare_marker_alone(self):
        assert strip_correction("no") == "no"
        assert strip_correction("no no") == "no no"
        assert strip_correction("no put it in the bowl") == "put it in the bowl"
        assert strip_correction("nothing to do") == "nothing to do"
        assert strip_correction("notice the can") == "notice the can"

    @pytest.mark.parametrize("text", [
        "no need to drop it", "no more moving left", "no dropping it", "no going home", "no one",
        "nope not that one", "no no need to drop it", "actually no need to drop it", "nah never mind",
        "no way", "no problem", "no longer holding it", "no can do", "no reason to drop it",
    ])
    def test_a_no_that_starts_a_negation_is_kept(self, text):
        """Fixer, 2026-09-28: "no"/"nope"/"nah" are stripped only before a command start."""
        assert strip_correction(text) == text

    @pytest.mark.parametrize("text,rest", [
        ("no drop it", "drop it"), ("no let go", "let go"), ("nope the red one", "the red one"),
        ("no can you put it in the bowl", "can you put it in the bowl"), ("nah just go home", "just go home"),
        ("sorry need to drop it", "need to drop it"),  # "sorry" is no negation; the rest is judged
    ])
    def test_a_no_before_a_command_start_is_stripped(self, text, rest):
        assert strip_correction(text) == rest

    @pytest.mark.parametrize("utterance", [
        "no need to drop it", "no need to go home", "no dropping it", "no going home",
        "no more moving left", "no, no need to drop it",
    ])
    def test_a_no_that_starts_a_negation_is_refused_as_one(self, parser, utterance):
        with pytest.raises(NegatedCommand):
            parser.parse(utterance)


class TestTakeAPictureIsObserve:
    """"Take a picture" asks the robot to look, not to pick up a "picture"."""

    @pytest.mark.parametrize(
        "utterance",
        ["take a picture", "take a photo", "take a look", "take a snapshot",
         "take a picture of the table", "please take a look"],
    )
    def test_take_a_picture_is_observe(self, parser, utterance):
        intent = parser.parse(utterance)
        assert intent.skill == "observe"
        assert intent.params == {}

    def test_take_a_look_at_something_is_still_look_at(self, parser):
        intent = parser.parse("take a look at the can")
        assert intent.skill == "look_at"
        assert intent.params["target"] == "can"

    @pytest.mark.parametrize(
        "utterance,target",
        [
            ("take the mug", "mug"),
            # The observe noun must END the request: a word after it names an object.
            ("take a picture frame", "picture frame"),
            ("take a photo album", "photo album"),
            ("take a snapshot box", "snapshot box"),
        ],
    )
    def test_take_an_object_is_still_a_pick(self, parser, utterance, target):
        intent = parser.parse(utterance)
        assert intent.skill == "pick"
        assert intent.params["target"] == target

    @pytest.mark.parametrize("utterance", ["take a photo please", "take a snapshot now", "take a photo now please"])
    def test_trailing_filler_after_the_noun_is_still_observe(self, parser, utterance):
        """The same filler ``_clean_target`` strips from a pick is tolerated here."""
        assert parser.parse(utterance).skill == "observe"

    def test_observe_then_pick_yields_only_observe(self, parser):
        assert parser.parse("take a picture then pick up the can").skill == "observe"


class TestStayAndHoldAreWait:
    @pytest.mark.parametrize(
        "utterance",
        ["stay", "hold on", "hold still", "hold there", "hold position", "stand by", "standby"],
    )
    def test_stay_and_hold_are_wait(self, parser, utterance):
        intent = parser.parse(utterance)
        assert intent.skill == "wait"
        assert intent.params["duration"] == pytest.approx(1.0)

    def test_wait_with_a_number_keeps_the_duration(self, parser):
        assert parser.parse("hold on 3 seconds").params["duration"] == pytest.approx(3.0)

    def test_hold_an_object_is_not_a_wait(self, parser):
        """"Hold the can" is not a pause; refusing beats freezing mid-task."""
        with pytest.raises(UnparsedCommand):
            parser.parse("hold the can")


# ----------------------------------------------------------------------
# no regressions on the existing vocabulary
# ----------------------------------------------------------------------

#: Existing phrases and the exact intent they produced before the fixes
#: (derived from tests/test_language_logic.py). ``params`` is compared exactly
#: where it was previously asserted; ``...`` means only the skill is pinned.
UNCHANGED = [
    ("pick the can", "pick", {"target": "can"}),
    ("pick up the can", "pick", {"target": "can"}),
    ("pick up the bottle", "pick", {"target": "bottle"}),
    ("grab the block", "pick", {"target": "block"}),
    ("take the mug", "pick", {"target": "mug"}),
    ("pick the red block", "pick", {"target": "red block"}),
    ("pick it up", "pick", {"target": "it"}),
    ("pick up the can and put it in the box", "pick", {"target": "can"}),
    ("place it", "place", {}),
    ("put it down", "place", {}),
    ("put it in the box", "place", {"relation": "in", "target": "box"}),
    ("place it on top of the box", "place", {"relation": "on", "target": "box"}),
    ("open the gripper", "open_gripper", {}),
    ("open", "open_gripper", {}),
    ("close the gripper", "close_gripper", {}),
    ("close", "close_gripper", {}),
    ("move left", "move_relative", {"direction": "left"}),
    ("move right", "move_relative", {"direction": "right"}),
    ("go up", "move_relative", {"direction": "up"}),
    ("move down", "move_relative", {"direction": "down"}),
    ("move forward", "move_relative", {"direction": "forward"}),
    ("move backwards", "move_relative", {"direction": "backward"}),
    ("move left 10", "move_relative", {"direction": "left", "distance": 0.10}),
    ("move left 5 cm", "move_relative", {"direction": "left", "distance": 0.05}),
    ("move up 20 mm", "move_relative", {"direction": "up", "distance": 0.02}),
    ("move right 0.3 m", "move_relative", {"direction": "right", "distance": 0.3}),
    ("stop", "stop", {}),
    ("emergency stop", "emergency_stop", {}),
    ("go home", "go_home", {}),
    ("observe", "observe", {}),
    ("what do you see", "observe", {}),
    ("scan the scene", "scan_scene", {}),
    ("wait", "wait", {"duration": 1.0}),
    ("look at the can", "look_at", {"target": "can"}),
    ("rotate the wrist 45 degrees", "rotate_wrist", ...),
    ("rotate wrist 90 degrees clockwise", "rotate_wrist", ...),
    ("rotate wrist 90 degrees counterclockwise", "rotate_wrist", ...),
    ("move to the can", "move_to", {"target": "can"}),
    # Handed to the LLM parser in the sim-lane suite; the rule parser is its
    # fallback, so its reading is frozen too.
    ("could you grab that fizzy drink", "pick", {"target": "fizzy drink"}),
]

#: Sim-lane phrases the parser refused before phase 9 and must still refuse.
REFUSED = ("open the box", "compose a symphony", "recite poetry", "   ")


class TestExistingVocabularyUnchanged:
    @pytest.mark.parametrize("utterance,skill,params", UNCHANGED, ids=[u for u, _, _ in UNCHANGED])
    def test_maps_exactly_as_before(self, parser, utterance, skill, params):
        intent = parser.parse(utterance)
        assert intent.skill == skill
        if params is not ...:
            assert set(intent.params) == set(params)
            for key, expected in params.items():
                actual = intent.params[key]
                if isinstance(expected, float):
                    assert actual == pytest.approx(expected), key
                else:
                    assert actual == expected, key

    def test_table_is_large_enough_to_mean_something(self):
        assert len(UNCHANGED) >= 25

    @pytest.mark.parametrize("utterance", REFUSED)
    def test_still_refuses_what_it_refused(self, parser, utterance):
        with pytest.raises(UnparsedCommand):
            parser.parse(utterance)

    def test_rotate_direction_sign_unchanged(self, parser):
        clockwise = parser.parse("rotate wrist 90 degrees clockwise").params["angle"]
        counter = parser.parse("rotate wrist 90 degrees counterclockwise").params["angle"]
        assert clockwise < 0 < counter


# ----------------------------------------------------------------------
# every mapping phase 9 changed, with its before and after
# ----------------------------------------------------------------------

#: Marker for "the parser raised UnparsedCommand".
UNPARSED = "UNPARSED"

WAIT_1S = ("wait", {"duration": 1.0})

#: Every phrase whose mapping the phase 9 parser edits changed: utterance, the
#: exact intent before (HEAD, as reproduced by the F4 property check in
#: docs/HARDWARE_REVIEW.md), the exact intent now, and why the change is wanted.
#: The parser's comment tables (``_OBSERVE_PATTERNS`` and ``_match_wait``) list
#: the same rows; extend both together. A row whose before equals its after is a
#: documentation bug and is rejected below.
HARDWARE_MAPPINGS = [
    # drop + destination is a place, not a release over nothing
    ("drop it in the bowl", ("open_gripper", {}), ("place", {"relation": "in", "target": "bowl"}),
     "a destination was named; releasing where the arm happens to be drops the object on the table"),
    ("drop it on the box", ("open_gripper", {}), ("place", {"relation": "on", "target": "box"}), "same"),
    ("drop it into the bowl", ("open_gripper", {}), ("place", {"relation": "in", "target": "bowl"}), "same"),
    # was ("place", next_to bowl) from phase 9 until 2026-09-28 (language decisions
    # stream); now a transfer whose first action is the pick of the named object.
    ("drop the can next to the bowl", ("open_gripper", {}), ("pick", {"target": "can"}),
     "phase 9 made it a place of whatever was held; a named object is picked first now"),
    ("drop it on top of the box", ("open_gripper", {}), ("place", {"relation": "on", "target": "box"}), "same"),
    # take a picture/photo/snapshot is a request to perceive
    ("take a picture", ("pick", {"target": "picture"}), ("observe", {}),
     "there is no object called picture; the pick would fail after moving the arm"),
    ("take a photo", ("pick", {"target": "photo"}), ("observe", {}), "same"),
    ("take a snapshot", ("pick", {"target": "snapshot"}), ("observe", {}), "same"),
    ("take a picture of the table", ("pick", {"target": "picture of the table"}), ("observe", {}), "same, with 'of'"),
    ("take a photo please", ("pick", {"target": "photo"}), ("observe", {}), "same, trailing filler"),
    ("take a snapshot now", ("pick", {"target": "snapshot"}), ("observe", {}), "same, trailing filler"),
    ("take a photo now please", ("pick", {"target": "photo now"}), ("observe", {}), "same, stacked filler"),
    # take a look (no object) is the same request
    ("take a look", ("pick", {"target": "look"}), ("observe", {}),
     "'look' is not an object; 'take a look at X' is unchanged (look_at)"),
    ("please take a look", ("pick", {"target": "look"}), ("observe", {}), "same"),
    ("take a look around", ("pick", {"target": "look around"}), ("observe", {}), "same"),
    # hold still / stand by are what a student says to a moving arm
    ("hold still", UNPARSED, WAIT_1S, "was refused; a pause is the safe reading"),
    ("hold there", UNPARSED, WAIT_1S, "same"),
    ("hold position", ("place", {}), WAIT_1S,
     "'position' is a place verb; a place with no destination would carry the object back to its pick origin"),
    ("stand by", UNPARSED, WAIT_1S, "same request as wait"),
    ("standby", UNPARSED, WAIT_1S, "same, as one word (what ASR usually emits)"),
    # language stream, 2026-09-26: the hybrid parser trusts every rule parse, so
    # the rule errors measured on the Qwen evaluation sets are fixed here.
    # benefactive "me" / "for me" is not part of the object
    ("could you grab the banana for me", ("pick", {"target": "banana for me"}), ("pick", {"target": "banana"}),
     "grounding cannot resolve 'banana for me'; the pick failed as 'nothing in view is a me'"),
    ("grab me that banana please", ("pick", {"target": "me that banana"}), ("pick", {"target": "banana"}), "same"),
    ("fetch the red block for me", ("pick", {"target": "red block for me"}), ("pick", {"target": "red block"}), "same"),
    ("get me the marker", ("pick", {"target": "me the marker"}), ("pick", {"target": "marker"}), "same"),
    # spoken number words with a unit
    ("pause for five seconds", WAIT_1S, ("wait", {"duration": 5.0}), "the number was silently ignored"),
    ("wait for two seconds", WAIT_1S, ("wait", {"duration": 2.0}), "same"),
    ("move left five centimetres", ("move_relative", {"direction": "left"}),
     ("move_relative", {"direction": "left", "distance": 0.05}), "the skill default (10 cm) moved twice as far"),
    ("move up twenty mm", ("move_relative", {"direction": "up"}),
     ("move_relative", {"direction": "up", "distance": 0.02}), "same, 5x too far"),
    ("rotate the wrist forty five degrees", ("rotate_wrist", {"angle": math.pi / 2}),
     ("rotate_wrist", {"angle": math.pi / 4}), "the default 90 degrees replaced the spoken 45"),
    # "set it down next to X": "down" belongs to the verb
    ("set it down next to the mug", ("pick", {"target": "it"}), ("place", {"relation": "next_to", "target": "mug"}),
     "picked 'it' (the object already held) before placing; 'put it down' alone is unchanged"),
    ("put it down next to the bowl", ("pick", {"target": "it"}), ("place", {"relation": "next_to", "target": "bowl"}),
     "same"),
    # language stream, 2026-09-27: rule errors found by a fresh 34-utterance set
    # (the hybrid parser trusts every rule parse, so they cannot be left to the LLM)
    ("lift the arm up by 5 cm", ("pick", {"target": "arm up by 5 cm"}),
     ("move_relative", {"direction": "up", "distance": 0.05}), "'lift' is a pick verb; the arm is not an object"),
    ("raise the arm a bit", UNPARSED, ("move_relative", {"direction": "up"}), "the operator's word for 'move up'"),
    ("lower the gripper 3 cm", UNPARSED, ("move_relative", {"direction": "down", "distance": 0.03}), "same, down"),
    ("return to your resting position", ("place", {}), ("go_home", {}),
     "'position' is a place verb: it carried the held object back to its pick origin"),
    ("go back to your starting position", ("move_relative", {"direction": "backward"}), ("go_home", {}),
     "'back' read as a direction: the arm moved 10 cm towards its base"),
    ("go to the bowl", UNPARSED, ("move_to", {"target": "bowl"}), "approach phrasing"),
    ("hover over the dice", UNPARSED, ("move_to", {"target": "dice"}), "approach phrasing"),
    ("go over to the mug", UNPARSED, ("move_to", {"target": "mug"}), "same"),
    ("approach the sponge", UNPARSED, ("move_to", {"target": "sponge"}), "same"),
    ("come closer to the sponge", UNPARSED, ("move_to", {"target": "sponge"}), "same"),
    ("how many objects do you see", UNPARSED, ("observe", {}), "a question about the scene"),
    ("how many things are there", UNPARSED, ("observe", {}), "same"),
    # review finding 4 (2026-09-28): negation was ignored -- with an object held,
    # "stop, don't drop it" opened the jaw. A negated clause is refused; a stop
    # word wins over everything but the emergency stop.
    ("don't let go of it", ("open_gripper", {}), UNPARSED, "negated: the operator said NOT to release"),
    ("do not drop it", ("open_gripper", {}), UNPARSED, "same"),
    ("don't open the gripper", ("open_gripper", {}), UNPARSED, "same"),
    ("never release it", ("open_gripper", {}), UNPARSED, "same"),
    ("don't go home", ("go_home", {}), UNPARSED, "negated go_home"),
    ("do not pick up the marker", ("pick", {"target": "marker"}), UNPARSED, "negated pick"),
    ("stop, don't drop it", ("open_gripper", {}), ("stop", {}), "a stop word wins"),
    # language decisions stream, 2026-09-28. "before" is the parser as it was
    # just before this change (commit 7046082).
    # drop + a NAMED object + a destination is a transfer (pick it first)
    ("drop the eraser in the bowl", ("place", {"relation": "in", "target": "bowl"}), ("pick", {"target": "eraser"}),
     "the held object (whatever it was) went into the bowl; the eraser was ignored"),
    ("drop the can to the left of the bowl", ("open_gripper", {}), ("pick", {"target": "can"}),
     "released whatever was held where the arm happened to be"),
    ("drop the red block onto the box", ("place", {"relation": "on", "target": "box"}),
     ("pick", {"target": "red block"}), "same as the eraser"),
    # an adverb after the named object belongs to the verb (fixer, 2026-09-28;
    # in between it was pick "can gently" / "can back")
    ("drop the can gently in the bowl", ("place", {"relation": "in", "target": "bowl"}),
     ("pick", {"target": "can"}), "same as the eraser"),
    ("drop the can back in the bowl", ("place", {"relation": "in", "target": "bowl"}),
     ("pick", {"target": "can"}), "same as the eraser"),
    # a pronoun followed by a deictic is still the pronoun
    ("put it here in the box", ("pick", {"target": "it"}), ("place", {"relation": "in", "target": "box"}),
     "picked 'it' (the held object) before placing, like 'set it down next to' did"),
    # a leading correction marker is not a negation
    ("no, put it in the bowl", UNPARSED, ("place", {"relation": "in", "target": "bowl"}),
     "a correction was refused as a negation and had to be repeated"),
    ("nope, pick the marker", UNPARSED, ("pick", {"target": "marker"}), "same"),
    ("actually, put the marker in the bowl", ("place", {"relation": "in", "target": "bowl"}),
     ("pick", {"target": "marker"}), "the marker hid the transfer: whatever was held went into the bowl"),
]

#: Phrases the first phase 9 patterns changed by accident (F4) and which map
#: exactly as they did before phase 9 again.
RESTORED = [
    ("drop it in", ("open_gripper", {})),
    ("drop in", ("open_gripper", {})),
    ("drop it inside", ("open_gripper", {})),
    ("drop it in the", ("open_gripper", {})),
    ("drop it in there", ("open_gripper", {})),
    ("drop it in here", ("open_gripper", {})),
    ("drop it in please", ("open_gripper", {})),
    ("drop it on top", ("open_gripper", {})),
    ("drop it in it", ("open_gripper", {})),
    ("drop it on that one", ("open_gripper", {})),
    ("drop it in them", ("open_gripper", {})),
    ("drop the can on it", ("open_gripper", {})),
    ("drop it on in it", ("open_gripper", {})),
    ("drop it in on that one", ("open_gripper", {})),
    ("take a picture frame", ("pick", {"target": "picture frame"})),
    ("take a photo album", ("pick", {"target": "photo album"})),
    ("take a snapshot box", ("pick", {"target": "snapshot box"})),
]


def _outcome(parser, utterance):
    """Parse to a comparable ``(skill, params)`` or ``UNPARSED``."""
    try:
        intent = parser.parse(utterance)
    except UnparsedCommand:
        return UNPARSED
    return (intent.skill, dict(intent.params))


def _same_outcome(actual, expected) -> bool:
    """Compare two outcomes; floats (durations, distances) are compared approximately."""
    if UNPARSED in (actual, expected):
        return actual == expected
    (skill, params), (exp_skill, exp_params) = actual, expected
    if skill != exp_skill or set(params) != set(exp_params):
        return False
    for key, value in exp_params.items():
        if isinstance(value, float):
            if params[key] != pytest.approx(value):
                return False
        elif params[key] != value:
            return False
    return True


class TestHardwareMappingTable:
    @pytest.mark.parametrize("utterance,before,after,why", HARDWARE_MAPPINGS, ids=[r[0] for r in HARDWARE_MAPPINGS])
    def test_maps_to_the_documented_after(self, parser, utterance, before, after, why):
        assert _same_outcome(_outcome(parser, utterance), after), why

    @pytest.mark.parametrize("utterance,before,after,why", HARDWARE_MAPPINGS, ids=[r[0] for r in HARDWARE_MAPPINGS])
    def test_every_row_is_a_real_change(self, utterance, before, after, why):
        """A row whose before equals its after documents nothing."""
        assert before != after

    @pytest.mark.parametrize("utterance,expected", RESTORED, ids=[r[0] for r in RESTORED])
    def test_restored_phrases_map_as_before_phase_9(self, parser, utterance, expected):
        assert _same_outcome(_outcome(parser, utterance), expected)

    def test_changed_and_frozen_sets_are_disjoint(self):
        """A phrase cannot both be pinned as unchanged and listed as changed."""
        changed = {row[0] for row in HARDWARE_MAPPINGS}
        frozen = {row[0] for row in UNCHANGED} | {row[0] for row in RESTORED} | set(REFUSED)
        assert not (changed & frozen), changed & frozen

    def test_parser_comment_tables_name_this_table(self):
        """The parser's comment tables must point back here, so the two stay paired."""
        import inspect

        from mfw.language import intent_parser

        source = inspect.getsource(intent_parser)
        assert source.count("tests/test_hardware_language.py::HARDWARE_MAPPINGS") >= 2


# ----------------------------------------------------------------------
# grammar-corpus property: the boundaries of the new patterns
# ----------------------------------------------------------------------

#: Object names free of direction and relation words, including ones that begin
#: with an observe noun -- the exact place the first pattern over-matched.
CORPUS_OBJECTS = ("can", "bowl", "marker", "red block", "picture frame", "photo album", "snapshot box")
PICK_VERBS = ("pick up", "pick", "grab", "take", "lift", "get", "fetch")


class TestGrammarCorpusProperties:
    """Verb x object x relation sweeps; each asserts an invariant, not a snapshot."""

    @pytest.mark.parametrize("verb", PICK_VERBS)
    @pytest.mark.parametrize("obj", CORPUS_OBJECTS)
    @pytest.mark.parametrize("article", ("the", "a", ""))
    def test_a_verb_and_a_named_object_is_always_a_pick(self, parser, verb, obj, article):
        utterance = " ".join(part for part in (verb, article, obj) if part)
        intent = parser.parse(utterance)
        assert (intent.skill, intent.params) == ("pick", {"target": obj}), utterance

    @pytest.mark.parametrize("relation,canonical", sorted(_RELATION_WORDS.items()))
    @pytest.mark.parametrize("obj", CORPUS_OBJECTS)
    def test_drop_with_relation_and_target_is_always_a_place(self, parser, relation, canonical, obj):
        intent = parser.parse(f"drop it {relation} the {obj}")
        assert (intent.skill, intent.params) == ("place", {"relation": canonical, "target": obj})

    @pytest.mark.parametrize("relation", sorted(_RELATION_WORDS))
    @pytest.mark.parametrize("subject", ("it", "the can", ""))
    @pytest.mark.parametrize(
        "tail",
        ("", " the", " a", " there", " here", " please", " there please",
         " it", " that one", " them", " this", " the object"),
    )
    def test_drop_with_relation_but_no_target_is_always_a_release(self, parser, relation, subject, tail):
        utterance = " ".join(part for part in ("drop", subject, relation) if part) + tail
        intent = parser.parse(utterance)
        assert (intent.skill, intent.params) == ("open_gripper", {}), utterance

    @pytest.mark.parametrize("verb", ("drop", "drop it", "drop the can", "please drop it"))
    @pytest.mark.parametrize("relation", sorted(_RELATION_WORDS))
    @pytest.mark.parametrize(
        "tail",
        ("", " it", " the bowl", " that one", " the red one", " there please", " the box please", " here the"),
    )
    def test_a_drop_that_becomes_a_place_always_names_a_real_target(self, parser, verb, relation, tail):
        """Invariant over the whole drop grammar: a ``place`` only with a named object.

        Either the utterance stays the pre-phase-9 release, or it is a place whose
        target is a real word -- never a pronoun, article or filler that the
        place skill would resolve to the held object or to nothing.
        """
        utterance = f"{verb} {relation}{tail}"
        intent = parser.parse(utterance)
        if intent.skill == "open_gripper":
            assert intent.params == {}, utterance
            return
        if intent.skill == "pick":
            # 2026-09-28: "drop the can in the bowl" is a transfer. Only a NAMED
            # subject may become one, and its place clause must name a real target.
            assert verb == "drop the can", utterance
            assert intent.params == {"target": "can"}, utterance
            split = split_transfer(utterance)
            assert split is not None and split.pick_clause == "pick the can", utterance
            intent = split.place_params
            assert intent["target"] and intent["target"] not in RuleBasedIntentParser._NOT_A_TARGET, utterance
            assert intent["relation"] == _RELATION_WORDS[relation], utterance
            return
        assert intent.skill == "place", utterance
        target = intent.params.get("target")
        assert target, utterance
        assert target not in RuleBasedIntentParser._NOT_A_TARGET, utterance
        assert intent.params["relation"] == _RELATION_WORDS[relation], utterance

    @pytest.mark.parametrize("noun", ("picture", "photo", "snapshot"))
    @pytest.mark.parametrize("tail", ("", " please", " now", " of the table", " of everything"))
    def test_take_a_perception_noun_is_observe(self, parser, noun, tail):
        assert parser.parse(f"take a {noun}{tail}").skill == "observe"

    @pytest.mark.parametrize("noun", ("picture", "photo", "snapshot"))
    @pytest.mark.parametrize("obj", ("frame", "album", "box"))
    def test_take_a_perception_noun_plus_object_is_a_pick(self, parser, noun, obj):
        intent = parser.parse(f"take a {noun} {obj}")
        assert (intent.skill, intent.params) == ("pick", {"target": f"{noun} {obj}"})


# ----------------------------------------------------------------------
# property check against the sim-lane phrase tables (imported, not copied)
# ----------------------------------------------------------------------


def _parametrize_rows(test_function):
    """Return ``(argnames, rows)`` from a test's single ``parametrize`` mark."""
    marks = [mark for mark in test_function.pytestmark if mark.name == "parametrize"]
    assert len(marks) == 1, f"{test_function.__qualname__} should carry one parametrize mark"
    names, rows = marks[0].args
    return [name.strip() for name in names.split(",")], list(rows)


def _literal_phrases(module) -> list[str]:
    """Every string literal a test in ``module`` hands to ``.parse`` or ``.handle``.

    Read from the AST rather than copied, so a phrase typed into a sim-lane test
    body (not only into a parametrize table) is caught by the freeze below.
    """
    tree = ast.parse(inspect.getsource(module))
    phrases: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("parse", "handle") or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            phrases.add(first.value)
    return sorted(phrases)


SIM_LANE_SKILL_ROWS = _parametrize_rows(sim_lane.TestIntentParsing.test_recognises_core_vocabulary)[1]
SIM_LANE_DIRECTION_ROWS = _parametrize_rows(sim_lane.TestIntentParsing.test_extracts_direction)[1]
SIM_LANE_UNIT_ROWS = _parametrize_rows(sim_lane.TestIntentParsing.test_honours_explicit_units)[1]
SIM_LANE_LITERALS = _literal_phrases(sim_lane)
SIM_LANE_PHRASES = sorted(
    {row[0] for row in SIM_LANE_SKILL_ROWS + SIM_LANE_DIRECTION_ROWS + SIM_LANE_UNIT_ROWS}
    | set(SIM_LANE_LITERALS)
)
FROZEN = {row[0]: (row[1], row[2]) for row in UNCHANGED}
FROZEN.update({phrase: UNPARSED for phrase in REFUSED})


class TestSimLanePhrasesMapIdentically:
    """Every phrase the sim-lane suite uses maps as it did before phase 9.

    The tables and literals are read from tests/test_language_logic.py at
    collection time, so adding a phrase there without freezing it in
    ``UNCHANGED`` or ``REFUSED`` fails here.
    """

    def test_tables_were_imported(self):
        assert len(SIM_LANE_SKILL_ROWS) >= 20
        assert len(SIM_LANE_DIRECTION_ROWS) >= 6
        assert len(SIM_LANE_UNIT_ROWS) >= 3

    def test_literals_were_read_from_the_source(self):
        """The AST walk must see the single-phrase tests, not only the tables."""
        assert {"pick the red block", "put it in the box", "open the box", "recite poetry"} <= set(SIM_LANE_LITERALS)
        assert len(SIM_LANE_LITERALS) >= 12

    def test_both_suites_parse_against_the_same_registry(self):
        assert tuple(sim_lane.SKILLS) == SKILLS

    @pytest.mark.parametrize("utterance", SIM_LANE_PHRASES)
    def test_sim_lane_phrase_is_frozen_here(self, utterance):
        assert utterance in FROZEN, f"{utterance!r} is in test_language_logic.py but not in UNCHANGED"

    @pytest.mark.parametrize("utterance,skill", SIM_LANE_SKILL_ROWS, ids=[r[0] for r in SIM_LANE_SKILL_ROWS])
    def test_skill_agrees_with_the_sim_lane_table_and_the_frozen_table(self, parser, utterance, skill):
        assert FROZEN[utterance][0] == skill
        assert parser.parse(utterance).skill == skill

    @pytest.mark.parametrize("utterance,direction", SIM_LANE_DIRECTION_ROWS, ids=[r[0] for r in SIM_LANE_DIRECTION_ROWS])
    def test_direction_agrees(self, parser, utterance, direction):
        params = parser.parse(utterance).params
        assert params["direction"] == direction
        assert FROZEN[utterance] == ("move_relative", params)

    @pytest.mark.parametrize("utterance,metres", SIM_LANE_UNIT_ROWS, ids=[r[0] for r in SIM_LANE_UNIT_ROWS])
    def test_distance_agrees(self, parser, utterance, metres):
        params = parser.parse(utterance).params
        assert params["distance"] == pytest.approx(metres)
        frozen_skill, frozen_params = FROZEN[utterance]
        assert frozen_skill == "move_relative"
        assert frozen_params["distance"] == pytest.approx(metres)

    @pytest.mark.parametrize("utterance", SIM_LANE_PHRASES)
    def test_full_intent_matches_the_frozen_snapshot(self, parser, utterance):
        """Skill and params, not just skill: a changed target is a changed action.

        A refusal is part of the contract too: a phrase the sim lane expects to
        raise must still raise, not quietly become an action.
        """
        frozen = FROZEN[utterance]
        if frozen == UNPARSED:
            assert _outcome(parser, utterance) == UNPARSED
            return
        frozen_skill, frozen_params = frozen
        if frozen_params is ...:
            assert parser.parse(utterance).skill == frozen_skill
            return
        assert _same_outcome(_outcome(parser, utterance), (frozen_skill, frozen_params))


# ----------------------------------------------------------------------
# hygiene: the LLM parser logs, it does not print
# ----------------------------------------------------------------------


class TestLlmParserLogsInsteadOfPrinting:
    def test_module_contains_no_print_call(self):
        import ast
        import inspect

        from mfw.language import intent_parser

        tree = ast.parse(inspect.getsource(intent_parser))
        prints = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        ]
        assert prints == [], f"print() at lines {prints}"

    def test_raw_and_parsed_intent_go_to_the_module_logger(self):
        """Attached directly to the module logger: ``mfw`` may have propagation off."""
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger("mfw.language.intent")
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            llm = LlmIntentParser(
                complete=lambda _prompt: '{"skill": "pick", "params": {"target": "can"}, "confidence": 0.9}',
                known_skills=SKILLS,
            )
            intent = llm.parse("grab the can")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        assert intent.skill == "pick"
        messages = [record.getMessage() for record in records]
        assert any("raw response" in m and '"skill": "pick"' in m for m in messages), messages
        assert any("parsed" in m and "'pick'" in m and "'can'" in m for m in messages), messages
        assert all(record.levelno == logging.INFO for record in records)


# ----------------------------------------------------------------------
# Parakeet engine (no sherpa-onnx, no model, no audio device)
# ----------------------------------------------------------------------


def _load_worker():
    spec = importlib.util.spec_from_file_location("speech_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker():
    return _load_worker()


class TestParakeetEngine:
    def test_worker_imports_without_audio_stack(self, worker):
        """The module must stay importable where sounddevice is absent (CI, tests).

        Checked structurally: no module-level import of an audio or ASR package.
        A runtime check would pass vacuously on a machine that lacks them.
        """
        import ast

        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        heavy = {"sounddevice", "numpy", "sherpa_onnx", "faster_whisper", "nemo", "torch"}
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        assert not (top_level & heavy), top_level & heavy
        assert worker.ParakeetEngine.name == "parakeet"

    def test_missing_sherpa_onnx_is_reported_clearly(self, worker, monkeypatch):
        monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
        with pytest.raises(SystemExit, match="sherpa-onnx"):
            worker.build_engine("parakeet")

    def test_missing_model_names_the_release_asset(self, worker, monkeypatch, tmp_path):
        """A missing model tells the operator exactly what to fetch; it never fetches."""
        monkeypatch.setitem(sys.modules, "sherpa_onnx", types.ModuleType("sherpa_onnx"))
        with pytest.raises(SystemExit) as info:
            worker.build_engine("parakeet", model_dir=tmp_path)
        message = str(info.value)
        assert worker.PARAKEET_MODEL_NAME in message
        assert "k2-fsa" in message
        assert "encoder.int8.onnx" in message

    def test_download_hint_names_the_k2_fsa_asset(self, worker):
        hint = worker.ParakeetEngine.download_hint()
        assert "github.com/k2-fsa/sherpa-onnx/releases" in hint
        assert "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2" in hint

    def test_default_model_dir_is_under_the_user_cache(self, worker):
        expected = Path.home() / ".cache" / "mfw" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
        assert worker.PARAKEET_DEFAULT_MODEL_DIR == expected

    def test_transcribe_uses_exp_mean_log_probs(self, worker, monkeypatch, tmp_path):
        """Confidence is derived from the token log-probs when the binding has them."""
        np = pytest.importorskip("numpy")
        for name in worker.PARAKEET_REQUIRED_FILES:
            (tmp_path / name).write_bytes(b"")

        calls: dict = {}

        class FakeResult:
            text = " pick up the marker "
            ys_log_probs = [math.log(0.9), math.log(0.8), math.log(0.7)]

        class FakeStream:
            result = FakeResult()

            def accept_waveform(self, rate, samples):
                calls["rate"] = rate
                calls["dtype"] = samples.dtype

        class FakeRecognizer:
            def create_stream(self):
                return FakeStream()

            def decode_stream(self, stream):
                calls["decoded"] = True

        class OfflineRecognizer:
            @classmethod
            def from_transducer(cls, **kwargs):
                calls["kwargs"] = kwargs
                return FakeRecognizer()

        fake = types.ModuleType("sherpa_onnx")
        fake.OfflineRecognizer = OfflineRecognizer
        monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)

        engine = worker.build_engine("parakeet", model_dir=tmp_path)
        out = engine.transcribe(np.zeros(16000, dtype=np.float32) + 0.01)

        assert calls["kwargs"]["model_type"] == "nemo_transducer"
        assert calls["kwargs"]["num_threads"] == 4
        assert calls["kwargs"]["provider"] == "cpu"
        assert calls["kwargs"]["encoder"].endswith("encoder.int8.onnx")
        assert calls["kwargs"]["tokens"].endswith("tokens.txt")
        assert calls["rate"] == 16000 and str(calls["dtype"]) == "float32"
        assert out == [("pick up the marker", pytest.approx(math.exp(sum(FakeResult.ys_log_probs) / 3)))]

    def test_confidence_is_one_when_binding_has_no_log_probs(self, worker):
        class Bare:
            text = "stop"

        assert worker.ParakeetEngine.confidence_of(Bare()) == 1.0

        class Empty:
            text = "stop"
            ys_log_probs = []

        assert worker.ParakeetEngine.confidence_of(Empty()) == 1.0

    def test_parakeet_is_an_accepted_asr_choice(self, worker, monkeypatch):
        """The flag parses; --stdin mode returns before any model is built."""
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert worker.main(["--asr", "parakeet", "--stdin"]) == 0
        with pytest.raises(SystemExit):
            worker.main(["--asr", "no-such-engine", "--stdin"])

    def test_host_and_port_defaults_untouched(self, worker, monkeypatch):
        """The laptop client still finds a local worker at 127.0.0.1:5556."""
        import argparse

        captured: dict = {}
        original = argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):
            ns = original(self, args, namespace)
            captured.update(vars(ns))
            return ns

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        worker.main(["--stdin"])
        assert captured["host"] == "127.0.0.1" and captured["port"] == 5556
        assert captured["asr"] == "whisper" and captured["model_dir"] is None


# ----------------------------------------------------------------------
# language stream 2026-09-27: boundaries of the vertical / rest-pose / approach
# patterns (each row in HARDWARE_MAPPINGS has a neighbour here that must NOT move)
# ----------------------------------------------------------------------

LANGUAGE_STREAM_BOUNDARIES = [
    ("lift the can", ("pick", {"target": "can"}), "lifting a named object is still a pick"),
    ("lift the red block", ("pick", {"target": "red block"}), "same"),
    ("raise the arm to the left", UNPARSED, "a sideways direction is not what raise/lower say"),
    ("put it in the starting position", ("place", {"relation": "in", "target": "starting position"}),
     "no travel verb: a place, not go_home"),
    ("go back to the home position", ("go_home", {}), "unchanged"),
    ("go to the left", ("move_relative", {"direction": "left"}), "a direction is claimed before move_to"),
    ("go to it", UNPARSED, "a pronoun names no place to approach"),
    ("go to there", UNPARSED, "a deictic names no place"),
    ("come to me", UNPARSED, "the operator is not an object"),
    ("approach", UNPARSED, "nothing named"),
    ("move to the bowl", ("move_to", {"target": "bowl"}), "unchanged"),
    # language decisions stream, 2026-09-28: neighbours of the drop-transfer and
    # correction rows in HARDWARE_MAPPINGS
    ("drop it into the bin", ("place", {"relation": "in", "target": "bin"}), "a pronoun: still a plain place"),
    ("drop the can to the left", ("open_gripper", {}), "a direction is not a destination; unchanged"),
    ("sorry, pick the marker", ("pick", {"target": "marker"}), "'sorry' was never refused"),
    ("nah, go home", ("go_home", {}), "'nah' was never a negator"),
    ("actually, place it on the box", ("place", {"relation": "on", "target": "box"}), "unchanged"),
    ("no, stop", ("stop", {}), "a stop word still wins"),
    ("no, don't drop it", UNPARSED, "a negation after the marker is still refused"),
    ("nope, the red one", UNPARSED, "no command after the marker: refused (not as a negation)"),
    ("no", UNPARSED, "a bare marker is still a refused negation"),
    # fixer, 2026-09-28: a "no" that STARTS a negation is not a correction marker
    # (each of these executed in the working tree before the fix; all were
    # refused at 7046082 and are again)
    ("no need to drop it", UNPARSED, "'no need to' negates: it opened the gripper"),
    ("no need to open the gripper", UNPARSED, "same"),
    ("no need to go home", UNPARSED, "it homed"),
    ("no need to move left", UNPARSED, "it moved the arm"),
    ("no need to put it in the bowl", UNPARSED, "it placed"),
    ("no, no need to drop it", UNPARSED, "stacked markers, then the negation"),
    ("actually, no need to drop it", UNPARSED, "same"),
    ("no dropping it", UNPARSED, "'no <gerund>' negates"),
    ("no going home", UNPARSED, "same; in hybrid mode the model saw 'going home' and homed"),
    ("no more moving left", UNPARSED, "'no more' negates"),
    ("no let go", ("open_gripper", {}), "a verb after the marker: still a correction"),
    ("nope, the red one", UNPARSED, "an article after the marker: stripped, then not a command"),
    # fixer, 2026-09-28: a drop of "what you're holding" names the held object,
    # like "it": the plain place it was at 7046082 (the drop-transfer change had
    # made each a pick of the phrase, refused with "already holding something")
    ("drop what you're holding in the bowl", ("place", {"relation": "in", "target": "bowl"}),
     "the held object, not something to pick"),
    ("drop the one you are holding in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop what you have in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop your load in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop the object in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop the thing in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop the item into the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
    ("drop everything in the bowl", ("place", {"relation": "in", "target": "bowl"}), "same"),
]


class TestLanguageStreamBoundaries:
    @pytest.mark.parametrize("utterance,expected,why", LANGUAGE_STREAM_BOUNDARIES,
                             ids=[r[0] for r in LANGUAGE_STREAM_BOUNDARIES])
    def test_neighbours_do_not_move(self, parser, utterance, expected, why):
        assert _same_outcome(_outcome(parser, utterance), expected), why

    @pytest.mark.parametrize("verb", ("lift", "raise"))
    @pytest.mark.parametrize("obj", CORPUS_OBJECTS)
    def test_lifting_any_corpus_object_is_a_pick_never_a_move(self, parser, verb, obj):
        intent = parser.parse(f"{verb} the {obj}") if verb == "lift" else None
        if intent is not None:
            assert (intent.skill, intent.params) == ("pick", {"target": obj})
        else:  # "raise the can" was never a command; it must not become an arm move
            assert _outcome(parser, f"raise the {obj}") in (UNPARSED, ("pick", {"target": obj}))

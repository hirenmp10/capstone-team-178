"""Natural language to a single, structured intent.

Pure stdlib + NumPy. This module must never import Isaac Sim.

**One utterance yields exactly one intent.** "Pick the can" produces a single
``pick`` -- never ``pick`` followed by ``place``. This is the layer where the
framework's central behavioural requirement is enforced, so it is enforced by
construction: :meth:`RuleBasedIntentParser.parse` returns one dict, and there is
no code path that can return a sequence.

Two implementations behind one interface:

* :class:`RuleBasedIntentParser` -- deterministic, dependency-free, and the
  default. Manipulation commands are a small, closed vocabulary ("pick X",
  "move left", "open gripper"), and for that a grammar is more reliable and far
  faster than a model, with no risk of a language model inventing an extra step.
* :class:`LlmIntentParser` -- adds a language model for open-ended phrasing,
  constrained to emit one skill from the registry. Two policies
  (:data:`LLM_MODES`):

  - ``"hybrid"`` (:data:`DEFAULT_LLM_MODE`, what ``scripts/run_assistant.py
    --llm`` uses): the grammar FIRST; the model is asked only when the grammar
    refuses, and a model failure keeps the grammar's refusal. A successful
    rule parse is never overridden. Measured 2026-09-26 with Qwen2.5-3B Q4_K_M
    on the 62-utterance set: the model alone was right 46/62, the grammar
    48/62, and the model agreed with only 35 of the 48 rule-correct parses --
    so letting it override the grammar loses more than it gains. After the
    2026-09-27 prompt, validation and grammar changes (hardware skill enum):
    hybrid 62/62, 31/31 and 33/34 on three sets; llm-first 60/62 and 31/34.
    All three sets were seen while tuning, so treat these as optimistic; the
    one untuned run (fresh set, before the last round) was hybrid 27/34.
  - ``"llm-first"`` (the class default, kept for existing callers): the model
    first, the grammar only when the model is unavailable or invalid.

  Either way the model's output is checked before it can move the arm
  (:meth:`LlmIntentParser._validated`): numbers are re-read from the
  utterance, a place destination may not be a pronoun, and in hybrid mode a
  target must be words the operator actually said. Every model intent then
  passes :attr:`LlmIntentParser.llm_gate` when one is set: the Assistant's
  gate asks "Did you mean: open the gripper and drop the marker?" before a
  model-chosen release of a held object, and refuses it in scripted mode.

**Corrections.** A leading correction marker ("no, put it in the bowl",
"actually, ...", "sorry, ...") is dropped and the command after it parsed
(:func:`strip_correction`); a negation after the marker is still refused, and
a "no" that starts the negation itself ("no need to drop it", "no going
home") is not a marker at all.

**Moving a named object** ("move the red block to the bowl", "put the can in
the bowl", "move the red object to the left") is two actions: pick it, then
place it. The parser still returns one intent -- the first, ``pick`` -- exactly
as it does for "pick X and put it in Y". :func:`split_transfer` exposes the two
clauses so the Assistant can run them one after the other (stopping at the
first failure); that is the only place the expansion happens. Before this, the
audit measured "move the red object to the left" -> ``move_relative{left}``
(the object silently dropped, the empty arm moved) and "put the red block in
the bowl" -> ``place{in, bowl}`` (the block ignored; a place with nothing held).
"drop <named object> in/on/next to <place>" is the same transfer since
2026-09-28; a pronoun ("drop it in the bowl") stays a plain place. The parser
cannot know what is held (its context carries a track id, not a name), so the
Assistant runs only the place when the named object is the one in the gripper.

Place parameters (the contract with the Place skill):

* ``{}`` -- back to where the held object was picked
* ``{"relation": R, "target": phrase}`` with R in :data:`PLACE_RELATIONS`
  ("to" lets Place choose: in a container, on a flat top, else next to)
* ``{"relation": "direction", "direction": left|right|forward|back,
  "distance": metres}`` -- relative to the pick origin, robot frame (+X
  forward, +Y left), distance defaulting to :data:`DEFAULT_PLACE_OFFSET_M`.

The referent phrase is kept whole, qualifiers included ("small red block on
the left"): resolving it is :mod:`mfw.language.grounding`'s job, not the
grammar's.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from mfw.core.errors import MfwError
from mfw.core.interfaces import IIntentParser
from mfw.utils.logging import get_logger

__all__ = [
    "RuleBasedIntentParser",
    "LlmIntentParser",
    "Intent",
    "UnparsedCommand",
    "UnregisteredSkill",
    "NotConfirmed",
    "strip_correction",
    "LLM_MODES",
    "DEFAULT_LLM_MODE",
    "REFUSAL_SKILL",
    "TransferSplit",
    "split_transfer",
    "PLACE_RELATIONS",
    "PLACE_DIRECTIONS",
    "DEFAULT_PLACE_OFFSET_M",
]

_log = get_logger("language.intent")


class UnparsedCommand(MfwError):
    """The utterance did not match any known command."""


class UnregisteredSkill(UnparsedCommand):
    """The utterance was understood, but its skill is not on this robot.

    "Rotate the wrist 90 degrees" on the hardware arm (no wrist roll joint) is
    not a phrasing problem, so the hybrid parser does not ask the language
    model to find some other skill for it: the refusal stands.
    """


class NegatedCommand(UnparsedCommand):
    """The utterance negates its action ("don't drop it", "never release it").

    The grammar matches verbs, not sentences: before this refusal "stop, don't
    drop it" opened the jaw and "don't go home" went home (review finding 4).
    A negated clause is refused, and the hybrid parser does not hand it to the
    language model either -- there is no phrasing problem to solve, and the
    measured model mis-maps exactly these ("drop it"/"let go"/"hold still").
    The changed phrases are pinned in tests/test_hardware_language.py::HARDWARE_MAPPINGS.
    """


class NotConfirmed(UnparsedCommand):
    """An LLM-sourced intent that needed the operator's yes and did not get it.

    Raised by the gate :class:`LlmIntentParser` calls before it returns a
    model intent (:attr:`LlmIntentParser.llm_gate`; the Assistant installs one
    that asks before a model-chosen release of a held object). A subclass of
    :class:`UnparsedCommand`, so the planner treats it like any refusal: the
    message reaches the operator and nothing moves.
    """


#: A negation anywhere in the clause refuses it (after "normalise", so
#: "don't" is "don t"). "no"/"not" are included on purpose: "the marker, not
#: the block" is a correction whose safe reading is to do nothing and let the
#: operator say it again. A LEADING correction marker followed by a command
#: ("no, put it in the bowl") is removed first by :func:`strip_correction`, so
#: only the command after it is judged.
_NEGATION_RE = re.compile(
    r"\b(?:don t|dont|do not|does not|doesn t|didn t|never|not|no|nope|cannot|can t|mustn t|shouldn t|won t)\b"
)
#: A stop word in a clause wins over every other matcher except the emergency
#: stop: "stop, don't drop it" and "stop moving left" are stops.
_STOP_WORD_RE = re.compile(r"\b(?:stop|halt|abort)\b")

#: Words an operator puts IN FRONT of a corrected command (language decisions
#: stream, 2026-09-28): "no, put it in the bowl", "nope, the red one",
#: "actually, place it on the box", "sorry, pick the marker". Stacked markers
#: ("no no", "actually no", "no sorry") are all removed. Matched on normalised
#: text (the comma is gone), so "no put it in the bowl" is the same utterance.
#: "not" is NOT a marker ("not the block" names what not to do) and neither is
#: "wait" (a command of its own). Pinned in
#: tests/test_hardware_language.py::HARDWARE_MAPPINGS:
#:
#:   phrase                                  was                now
#:   "no, put it in the bowl"                UNPARSED (negated) place in bowl
#:   "nope, pick the marker"                 UNPARSED (negated) pick "marker"
#:   "actually, put the marker in the bowl"  place in bowl      pick "marker" (a transfer)
#:
#: "actually"/"sorry" were never refused, but they hid a transfer: the marker
#: was ignored and whatever was held went into the bowl.
#:
#: The negation refusal still applies to what follows the marker: "no, don't
#: drop it" is refused, and "no, stop" is a stop.
#:
#: "no"/"nope"/"nah" are also how a negation STARTS ("no need to drop it", "no
#: more moving left", "no dropping it"), and the normalised text has lost the
#: comma that told the two apart. So they are removed only when the next word
#: starts a command or an answer (:data:`_AFTER_A_NEGATING_MARKER`: a verb the
#: grammar knows, an article, a pronoun or an ordinal); otherwise the text is
#: kept whole and refused as the negation it is (fixer, 2026-09-28; before
#: this, "no need to drop it" opened the gripper and "no going home" homed).
#: Pinned in tests/test_hardware_language.py::LANGUAGE_STREAM_BOUNDARIES.
_CORRECTION_RE = re.compile(
    r"^(?:(?:no|nope|nah|actually|sorry|oops|correction|i mean|i meant|scratch that|my bad)\s+)+"
)
_NEGATING_MARKERS = frozenset({"no", "nope", "nah"})
#: Words that may follow a stripped "no"/"nope"/"nah". Deliberately a closed
#: list: a word missing from it keeps the utterance refused (safe), a word
#: wrongly in it would execute a negation. Never add a gerund ("dropping"),
#: "need", "more", "longer", "way", "problem", "reason", "point", "one" or
#: "not" -- each continues the negation.
_AFTER_A_NEGATING_MARKER = frozenset(
    {
        # verbs the grammar (or the model's vocabulary) starts a command with
        "pick", "grab", "take", "get", "lift", "grasp", "grip", "hold", "fetch", "collect",
        "place", "put", "set", "drop", "release", "let", "open", "close", "move", "go",
        "return", "head", "come", "look", "scan", "observe", "show", "tell", "describe",
        "find", "locate", "what", "which", "where", "wait", "pause", "stay", "stop", "halt",
        "abort", "cancel", "freeze", "emergency", "bring", "transfer", "carry", "shift",
        "slide", "relocate", "turn", "rotate", "twist", "lower", "raise", "elevate", "toss",
        "throw", "chuck", "dump", "deposit", "leave", "hover", "reach", "point", "push",
        "pull", "nudge", "squeeze", "clamp", "reset",
        # politeness / redirection in front of a command
        "please", "just", "instead", "rather", "now",
        # an answer or a referent ("nope, the red one", "no, the second one")
        "the", "a", "an", "that", "this", "these", "those", "it", "its", "them",
        "first", "second", "third", "fourth", "fifth", "last", "other",
    }
)
#: "no, can you put it in the bowl": a modal counts only before "you".
_MODALS = frozenset({"can", "could", "would", "will"})


def _starts_a_command(rest: str) -> bool:
    words = rest.split()
    if not words:
        return False
    if words[0] in _MODALS:
        return len(words) > 1 and words[1] == "you"
    return words[0] in _AFTER_A_NEGATING_MARKER


def strip_correction(text: str) -> str:
    """``text`` without a leading correction marker ("no, put it in the bowl").

    ``text`` is normalised (lowercase, no punctuation). When nothing but
    markers is left ("no", "no no", "sorry") the text is returned unchanged,
    so a bare "no" is still refused as a negation. When the markers include
    "no"/"nope"/"nah" and the next word does not start a command ("no need to
    drop it", "no more moving left", "no going home") the text is returned
    unchanged too: that "no" is the negation itself.
    """
    match = _CORRECTION_RE.match(text + " ")
    if match is None:
        return text
    rest = text[min(match.end(), len(text)):].strip()
    if not rest:
        return text
    if set(match.group(0).split()) & _NEGATING_MARKERS and not _starts_a_command(rest):
        return text
    return rest


def _without_correction(utterance: str) -> str:
    """The raw ``utterance``, or the normalised command after its correction marker."""
    normalised = RuleBasedIntentParser._normalise(utterance or "")
    stripped = strip_correction(normalised)
    return utterance if stripped == normalised else stripped


def is_negated(utterance: str) -> bool:
    """True when the first clause of ``utterance`` carries a negation.

    A leading correction marker is not a negation ("no, put it in the bowl").
    """
    text = RuleBasedIntentParser._first_clause(
        strip_correction(RuleBasedIntentParser._normalise(utterance or ""))
    )
    return _NEGATION_RE.search(text) is not None


#: Parsing policies of :class:`LlmIntentParser`.
LLM_MODES: tuple[str, ...] = ("hybrid", "llm-first")
#: The policy used when an operator configures an LLM (``run_assistant --llm``).
DEFAULT_LLM_MODE = "hybrid"
#: The skill name a model uses to decline ("not a robot command", "no object
#: named"). Never registered; it always means "refuse".
REFUSAL_SKILL = "unknown"


class Intent(dict):
    """A parsed intent: ``{"skill": str, "params": dict, "confidence": float}``.

    A ``dict`` subclass so it serialises straight into the event log while still
    being constructible with named fields.
    """

    def __init__(self, skill: str, params: dict[str, Any] | None = None, confidence: float = 1.0):
        super().__init__(skill=skill, params=params or {}, confidence=confidence)

    @property
    def skill(self) -> str:
        return self["skill"]

    @property
    def params(self) -> dict[str, Any]:
        return self["params"]

    @property
    def confidence(self) -> float:
        return self["confidence"]


# Filler words stripped before matching an object reference. "the", "that" and
# "this" are also pronouns in their own right, so they are removed only when
# something follows them.
_ARTICLES = ("the ", "a ", "an ", "that ", "this ", "my ")

_PRONOUNS = {"it", "that", "this", "them", "one", "object"}

_DIRECTION_WORDS = {
    "left": "left",
    "right": "right",
    "forward": "forward",
    "forwards": "forward",
    "ahead": "forward",
    "back": "backward",
    "backward": "backward",
    "backwards": "backward",
    "up": "up",
    "upward": "up",
    "upwards": "up",
    "down": "down",
    "downward": "down",
    "downwards": "down",
}

_RELATION_WORDS = {
    "in": "in",
    "inside": "in",
    "into": "in",
    "on": "on",
    "onto": "on",
    "on top of": "on",
    "next to": "next_to",
    "beside": "next_to",
    "near": "next_to",
}

#: Every relation a ``place`` intent may carry (contract P with the Place skill).
PLACE_RELATIONS = frozenset(
    {"in", "on", "next_to", "to", "left_of", "right_of", "in_front_of", "behind"}
)
#: Directions of a ``{"relation": "direction"}`` place, relative to the pick origin.
PLACE_DIRECTIONS = frozenset({"left", "right", "forward", "back"})
#: How far a directional place moves the object when no distance is spoken.
DEFAULT_PLACE_OFFSET_M = 0.10

#: Destination phrases a place understands. A superset of ``_RELATION_WORDS``
#: (which stays exactly as it was: the "drop" grammar and its frozen property
#: tests are built on it). Matched leftmost-first, longest-first at a position,
#: so "on top of" beats "on" and the destination keeps its own qualifiers
#: ("in the bowl on the left" -> target "bowl on the left").
_PLACE_RELATION_WORDS: dict[str, str] = {
    **_RELATION_WORDS,
    "close to": "next_to",
    "to the left side of": "left_of", "on the left side of": "left_of",
    "to the left of": "left_of", "on the left of": "left_of", "left of": "left_of",
    "to the right side of": "right_of", "on the right side of": "right_of",
    "to the right of": "right_of", "on the right of": "right_of", "right of": "right_of",
    "in front of": "in_front_of", "in back of": "behind", "behind": "behind",
    "inside of": "in", "within": "in", "atop": "on", "on to": "on",
    "over to": "to", "towards": "to", "toward": "to", "to": "to",
}
_PLACE_RELATION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(_PLACE_RELATION_WORDS, key=len, reverse=True))
    + r")\b"
)
#: Canonical spoken form of each relation, used to write the place clause.
_RELATION_TEXT = {
    "in": "in", "on": "on", "next_to": "next to", "to": "to", "left_of": "to the left of",
    "right_of": "to the right of", "in_front_of": "in front of", "behind": "behind",
}
_PLACE_DIRECTION_WORDS = {
    "left": "left", "right": "right", "forward": "forward", "forwards": "forward",
    "ahead": "forward", "back": "back", "backward": "back", "backwards": "back",
}
#: A destination that is only a direction: "to the left", "on the right side".
_DIRECTION_TAIL = re.compile(
    r"^(?:the |your |my )?(?P<dir>left|right)(?: hand)?(?: side)?$"
    r"|^(?:the )?(?P<dir2>forwards?|ahead|backwards?|back)$"
)
#: A spoken distance: "5 cm", "by 20 mm", "0.1 m", bare "10" (= cm).
_DISTANCE_SPAN = re.compile(
    r"\b(?:by\s+)?\d+(?:\.\d+)?\s*(?:mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?)?\b"
)
_VAGUE_AMOUNT = re.compile(r"\b(?:a (?:little )?bit|a little|slightly|a touch)\b")

#: The robot itself. "Move the gripper left" moves the arm; it names no object.
_SELF_WORDS = frozenset(
    {"arm", "gripper", "hand", "robot", "yourself", "wrist", "camera", "effector",
     "end", "tool", "claw", "fingers", "jaw", "jaws"}
)
#: Words that can sit between a motion verb and its direction without naming an
#: object ("move a bit to the left", "move your hand 5 cm left").
_MOTION_FILLER = frozenset(
    {"the", "a", "an", "it", "its", "itself", "your", "my", "to", "towards", "toward",
     "by", "bit", "little", "slightly", "more", "further", "over", "some", "just",
     "please", "now", "then", "again", "straight", "directly", "about", "around",
     "roughly", "approximately", "cm", "mm", "m", "centimeters", "centimetres",
     "centimeter", "centimetre", "millimeters", "millimetres", "meters", "metres",
     "meter", "metre", "and", "slowly", "carefully", "gently", "quickly", "touch"}
) | _SELF_WORDS

#: "drop" joined 2026-09-28 (language decisions stream) for a NAMED object and a
#: named destination only -- see :func:`split_transfer`. Pinned in
#: tests/test_hardware_language.py::HARDWARE_MAPPINGS:
#:
#:   phrase                               was                        now
#:   "drop the can next to the bowl"      place next_to bowl         pick "can" (then place)
#:   "drop the eraser in the bowl"        place in bowl              pick "eraser" (then place)
#:   "drop the can to the left of the bowl"  open_gripper            pick "can" (then place)
#:
#: Before, whatever the arm held went into the bowl, whichever object was
#: named. When the named object IS the held one, the Assistant runs only the
#: place (:meth:`mfw.assistant.Assistant.clauses`). "drop it in the bowl" (a
#: pronoun), "drop the can" and "drop the can on it" are unchanged.
_TRANSFER_VERBS = ("move", "put", "place", "set", "bring", "transfer", "carry", "take",
                   "shift", "slide", "relocate", "drop")
_TRANSFER_RE = re.compile(
    r"^(?:(?:please|kindly|now|can you|could you|would you|will you|go ahead and|"
    r"i want you to|i need you to|i would like you to|i d like you to)\s+)*"
    rf"(?P<verb>{'|'.join(_TRANSFER_VERBS)})\s+(?P<rest>.+?)(?:\s+(?:please|now))*$"
)
_OBJECT_PRONOUNS = frozenset({"it", "that", "this", "them", "that one", "this one"})
#: "drop <phrase> in the bowl" where the phrase means whatever is held; such a
#: drop stays the plain place it was before "drop" became a transfer verb.
#: Pinned in tests/test_hardware_language.py::LANGUAGE_STREAM_BOUNDARIES.
_DROP_HELD_OBJECT = re.compile(
    r"(?:"
    r"(?:(?:what|whatever|the (?:one|thing|object|item|stuff)|everything|anything)(?: that)?"
    r"(?: (?:you re|youre|you are|you ve|you have|you ve got|you have got|you got|you)"
    r"(?: (?:holding|carrying|gripping|grasping|got|have))?)?"
    r"(?: (?:in|on) (?:your|the) (?:hand|hands|gripper|grip|claw|jaws|fingers))?)"
    r"|(?:your|the) (?:load|cargo|payload)"
    r"|it all|all of it"
    r")"
)
_DROP_ADVERB_TAIL = re.compile(
    r"(?:\s+(?:gently|carefully|slowly|softly|back|right here|right there|here|there|down))+$"
)


@dataclass(frozen=True)
class TransferSplit:
    """"Move <object> <prep> <destination>" as its two atomic clauses.

    ``pick_clause`` is ``None`` when the object is a pronoun ("move it to the
    bowl"): the held object is placed, nothing is picked. ``place_params`` are
    exactly what ``place_clause`` parses to.
    """

    object_phrase: str
    pick_clause: str | None
    place_clause: str
    place_params: dict[str, Any] = field(default_factory=dict)

    @property
    def clauses(self) -> tuple[str, ...]:
        return tuple(c for c in (self.pick_clause, self.place_clause) if c)


def _strip_distance(text: str) -> tuple[str, float | None]:
    """Remove the first spoken distance from ``text``; return it in metres."""
    match = _DISTANCE_SPAN.search(text)
    if match is None:
        return text, None
    metres = RuleBasedIntentParser._find_distance(match.group(0))
    stripped = f"{text[: match.start()]} {text[match.end():]}"
    return re.sub(r"\s+", " ", stripped).strip(), metres


def _direction_of(tail: str) -> str | None:
    """The direction a destination phrase names, if it names only a direction."""
    match = _DIRECTION_TAIL.match(tail.strip())
    if match is None:
        return None
    return _PLACE_DIRECTION_WORDS[match.group("dir") or match.group("dir2")]


def _names_an_object(phrase: str) -> bool:
    words = [w for w in phrase.split() if not re.fullmatch(r"[\d.]+", w)]
    if not words or any(w in _SELF_WORDS for w in words):
        return False
    return any(w not in _MOTION_FILLER and w not in _PLACE_DIRECTION_WORDS for w in words)


def _names_a_destination(phrase: str) -> bool:
    not_a_place = RuleBasedIntentParser._NOT_A_TARGET | {"me", "you", "us", "yourself"}
    return any(w not in not_a_place for w in phrase.split())


def split_transfer(utterance: str) -> TransferSplit | None:
    """Split "move/put/bring/... <object> <prep> <destination>" into pick + place.

    Returns ``None`` when the utterance is not that shape -- including when the
    "object" is the robot itself ("move the gripper to the left" is a relative
    move), when no destination is named ("put the can down", "put the can
    back"), or when the destination is only a deictic ("put the can there").

    Choosing the split point: a referent can itself contain a preposition ("the
    block on the left", "the can next to the box"), so the LAST plain "to" is
    preferred ("move the block next to the can to the bowl"), otherwise the
    first relation phrase ("put the can in the bowl on the left" -> destination
    "bowl on the left"). "On the left/right" not followed by "of" is a position
    qualifier, never a split point, unless it ends the command ("put the can
    on the left" places it to the left).
    """
    text = RuleBasedIntentParser._normalise(utterance)
    text = RuleBasedIntentParser._first_clause(strip_correction(text))
    match = _TRANSFER_RE.match(text)
    if match is None:
        return None
    verb, rest = match.group("verb"), match.group("rest")
    if verb == "drop":
        # "drop off the can in the bowl" / "drop the can off in the bowl".
        rest = re.sub(r"^off\s+", "", rest)
        rest = re.sub(r"\s+off(?=\s+(?:in|into|inside|on|onto|next|beside|near|at|to)\b)", "", rest)
    rest, distance = _strip_distance(rest)
    rest = re.sub(r"\s+", " ", _VAGUE_AMOUNT.sub(" ", rest)).strip()

    side_spans = [
        m.span()
        for m in re.finditer(
            r"\bon (?:the |your |my )?(?:left|right)(?!(?:\s+hand)?(?:\s+side)?\s+of\b)(?: hand)?(?: side)?\b",
            rest,
        )
        if m.end() < len(rest)  # a trailing one is the destination, see below
    ]
    relations = [
        m for m in _PLACE_RELATION_RE.finditer(rest)
        if not any(start <= m.start() < end for start, end in side_spans)
    ]
    to_family = [m for m in relations if _PLACE_RELATION_WORDS[m.group(0)] == "to"]
    chosen = to_family[-1] if to_family else (relations[0] if relations else None)

    direction: str | None = None
    relation: str | None = None
    dest = ""
    if chosen is not None:
        obj = rest[: chosen.start()].strip()
        dest = rest[chosen.end():].strip()
        relation = _PLACE_RELATION_WORDS[chosen.group(0)]
        if relation in ("to", "on"):
            direction = _direction_of(dest)
    else:
        trailing = re.match(
            r"^(?P<obj>.+?)\s+(?:(?:to|towards|toward) (?:the |your |my )?)?"
            r"(?P<dir>left|right|forwards?|ahead|backwards?|back)(?: hand)?(?: side)?$",
            rest,
        )
        if trailing is None:
            return None
        obj = trailing.group("obj").strip()
        direction = _PLACE_DIRECTION_WORDS[trailing.group("dir")]
        # "Put the can back" means "return it", not "10 cm towards the base".
        if trailing.group("dir") == "back" and distance is None and verb in ("put", "place", "set"):
            return None

    # "Set it down next to the mug": "down" belongs to the verb, not the object.
    # It used to leave "it down" as the object -> pick "it" (the held object)
    # before the place (tests/test_hardware_language.py::HARDWARE_MAPPINGS).
    # Same for a deictic (2026-09-28): "put it here in the box" picked "it".
    obj = re.sub(r"^(it|that|this|them)\s+(?:down|here|there)$", r"\1", obj)
    if verb == "drop":
        # "drop the can gently / back / right here in the bowl": the adverb
        # belongs to the verb (it used to become the pick target "can gently").
        obj = _DROP_ADVERB_TAIL.sub("", obj).strip()
        if _DROP_HELD_OBJECT.fullmatch(obj):
            # "drop what you're holding in the bowl", "drop your load in the
            # bowl", "drop the object in the bowl": the held object, like "it".
            # A plain place, as before "drop" was a transfer verb (fixer,
            # 2026-09-28; they had become a pick of "what you re holding",
            # refused with "already holding something").
            return None
    if not obj:
        return None
    pronoun = obj in _OBJECT_PRONOUNS
    if not pronoun and not _names_an_object(obj):
        return None
    if verb == "drop" and (pronoun or direction is not None):
        # Only "drop <named object> <relation> <named place>" is a transfer.
        # "drop it in the bowl" is the place it always was (_match_place), and
        # "drop the can to the left" keeps its pre-existing reading; both stay
        # on the drop grammar's own path, untouched.
        return None

    if direction is not None:
        if pronoun:
            # "Move it to the left" keeps its meaning: a relative move of the arm.
            return None
        metres = distance if distance is not None else DEFAULT_PLACE_OFFSET_M
        params: dict[str, Any] = {"relation": "direction", "direction": direction, "distance": metres}
        place_clause = f"place it {direction} {metres * 100.0:g} cm"
    else:
        if relation is None or not dest or not _names_a_destination(dest):
            return None
        params = {"relation": relation, "target": RuleBasedIntentParser._clean_target(dest)}
        place_clause = f"place it {_RELATION_TEXT[relation]} {dest}"

    object_phrase = "it" if pronoun else RuleBasedIntentParser._clean_target(obj)
    return TransferSplit(
        object_phrase=object_phrase,
        pick_clause=None if pronoun else f"pick {obj}",
        place_clause=place_clause,
        place_params=params,
    )


class RuleBasedIntentParser(IIntentParser):
    """Deterministic grammar over the atomic skill vocabulary."""

    def __init__(self, known_skills: tuple[str, ...] = ()) -> None:
        self.known_skills = set(known_skills)

    def parse(self, utterance: str, context: dict[str, Any] | None = None) -> Intent:
        """Parse one utterance into one intent.

        Raises :class:`UnparsedCommand` rather than guessing: a misread
        manipulation command moves a real arm, so silence is safer than a
        plausible-looking wrong action.
        """
        if not utterance or not utterance.strip():
            raise UnparsedCommand("empty command")

        # A leading correction marker is dropped first ("no, put it in the
        # bowl" is "put it in the bowl"); see strip_correction.
        text = strip_correction(self._normalise(utterance))
        # "Pick the can and put it in the box" is ONE command: pick. Truncating at
        # the conjunction implements "emit only the first action" directly, rather
        # than leaving it to matcher ordering -- which got this backwards, matching
        # the later "put" and silently turning a pick into a place.
        text = self._first_clause(text)

        emergency = self._match_emergency(text)
        if emergency is not None:
            self._check_known(emergency)
            return emergency
        if _STOP_WORD_RE.search(text):
            # Review finding 4: a stop word wins ("stop, don't drop it").
            stop = Intent("stop", {}, 1.0)
            self._check_known(stop)
            return stop
        if _NEGATION_RE.search(text):
            raise NegatedCommand(
                f"refusing {utterance!r}: it negates its action (say what to do, not what not to do)"
            )

        for matcher in (
            self._match_emergency,
            self._match_stop,
            # Before the gripper matcher: "put the can in the open box" names an
            # object and a destination; the word "open" must not release the jaw.
            self._match_transfer,
            self._match_gripper,
            self._match_home,
            self._match_observe,
            self._match_scan,
            self._match_look,
            self._match_wait,
            self._match_rotate,
            self._match_relative_move,
            self._match_place,
            self._match_pick,
            self._match_move_to,
        ):
            intent = matcher(text)
            if intent is not None:
                self._check_known(intent)
                return intent

        raise UnparsedCommand(f"could not understand {utterance!r}")

    @staticmethod
    def _normalise(utterance: str) -> str:
        """Lowercase and strip punctuation, **preserving decimal points**.

        A blanket ``[^\\w\\s]`` strip turns "0.3 m" into "0 3 m", and the distance
        matcher then reads 3 metres instead of 0.3 -- a tenfold motion error from a
        punctuation rule.
        """
        text = utterance.lower()
        # Remove periods that are not decimal separators, then all other punctuation.
        text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
        text = re.sub(r"[^\w\s.]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _first_clause(text: str) -> str:
        """Keep only the first clause of a conjoined command.

        The framework's rule is that one utterance is one action, so a second
        clause is dropped rather than queued. Dropping it is the honest reading:
        the operator will be told what was done and can issue the next command.
        """
        split = re.split(r"\b(?:and then|and|then|after that)\b", text, maxsplit=1)
        first = split[0].strip()
        return first or text

    def _check_known(self, intent: Intent) -> None:
        if self.known_skills and intent.skill not in self.known_skills:
            raise UnregisteredSkill(
                f"parsed skill {intent.skill!r} is not registered; "
                f"available: {sorted(self.known_skills)}"
            )

    # ------------------------------------------------------------------
    # matchers, ordered most specific first
    # ------------------------------------------------------------------

    def _match_emergency(self, text: str) -> Intent | None:
        if re.search(r"\b(emergency|e ?stop|halt now|freeze)\b", text):
            return Intent("emergency_stop", {}, 1.0)
        return None

    def _match_stop(self, text: str) -> Intent | None:
        if re.fullmatch(r"(stop|halt|cancel|abort)( now| it| moving)?", text):
            return Intent("stop", {}, 1.0)
        return None

    #: "drop" followed by a destination ("drop it in the bowl") is a place, not a
    #: gripper release. This matcher runs before ``_match_place`` -- deliberately,
    #: so a bare "drop it" stays the instant, plan-free release it should be -- so
    #: it has to step aside itself when a relation phrase follows the verb.
    #:
    #: The relation must be followed by a word that names something. "drop it in"
    #: / "drop in" (a trailing preposition, no object named) is still the instant
    #: release: without a destination there is nothing to plan, and a ``place``
    #: with empty params would carry the object back to its pick origin instead
    #: of opening the jaw where the operator is pointing. The same goes for a
    #: tail made only of words the robot cannot resolve to an object (review
    #: finding F4): articles ("drop it in the"), deictics ("drop it in there"
    #: points at the table), politeness ("drop it in please"), and the "top"/"of"
    #: left over when "on top of" is cut short ("drop it on top"), and pronouns
    #: ("drop it in it", "drop it on that one", "drop it in them"): a pronoun
    #: names no destination, and ``place`` would hand memory a referent that
    #: usually resolves to the held object itself. A stacked relation word
    #: ("drop it on in it") names nothing either. Every one of those released
    #: the gripper before phase 9 and still does.
    _DROP_RELATION = re.compile(
        r"\bdrop\b.*?\b(?:on top of|next to|in|inside|into|on|onto|beside|near)\b(?P<tail>.*)$"
    )
    _NOT_A_TARGET = frozenset(
        {"the", "a", "an", "that", "this", "my", "there", "here", "please", "now", "top", "of"}
    ) | frozenset(_PRONOUNS) | frozenset(word for phrase in _RELATION_WORDS for word in phrase.split())

    @classmethod
    def _drop_names_a_destination(cls, text: str) -> bool:
        """True when a "drop" utterance names somewhere to put the object.

        The words after the first relation phrase must include at least one that
        is not an article, deictic, politeness, pronoun or "top of" fragment; only
        then is there a destination for ``_match_place`` to resolve.
        """
        match = cls._DROP_RELATION.search(text)
        if match is None:
            return False
        return any(word not in cls._NOT_A_TARGET for word in match.group("tail").split())

    def _match_gripper(self, text: str) -> Intent | None:
        # "open"/"close" alone are unambiguous in a manipulation context, but
        # must not fire on "open the box", which is not a gripper command.
        if (
            re.search(r"\b(open|release|let go|drop)\b", text)
            and not re.search(r"\bopen (the )?(box|door|lid|drawer)\b", text)
            and not self._drop_names_a_destination(text)
        ):
            return Intent("open_gripper", {}, 1.0)
        if re.search(r"\b(close|grip|squeeze|clamp)\b", text) and "gripper" in text:
            return Intent("close_gripper", {}, 1.0)
        if re.fullmatch(r"(close|grip|squeeze|clamp)", text):
            return Intent("close_gripper", {}, 1.0)
        return None

    #: Returning to a named rest pose is going home (language stream, 2026-09-27,
    #: fresh evaluation set). Pinned in
    #: tests/test_hardware_language.py::HARDWARE_MAPPINGS:
    #:
    #:   phrase                               was        now
    #:   "return to your resting position"    place {}   go_home
    #:   "go back to your starting position"  move_relative backward   go_home
    #:
    #: "position" is a place verb and "back" a direction word, so these carried
    #: the held object back to its pick origin or moved the arm 10 cm towards its
    #: base instead of going home. Only after a travel verb:
    #: "put it in the starting position" stays a place.
    _REST_POSE = re.compile(
        r"\b(?:return|go|head|move|get)(?: back)? to (?:your |the |its |my )?"
        r"(?:rest|resting|start|starting|initial|neutral|default|home|park|parking) (?:position|pose)\b"
    )

    def _match_home(self, text: str) -> Intent | None:
        if re.search(r"\b(go home|home position|return home|reset arm)\b", text) or self._REST_POSE.search(text):
            return Intent("go_home", {}, 1.0)
        return None

    #: Ways operators actually ask "what is there?". Matched by search, not
    #: fullmatch: real speech carries filler ("so, what objects are present now")
    #: and transcription adds its own noise, so an exact-phrase list rejects
    #: perfectly clear questions -- "what objects are present" failed against a
    #: fullmatch list that only knew "what do you see".
    _OBSERVE_PATTERNS = (
        r"\bwhat (?:do|can) you see\b",
        r"\bwhat(?:'s| is| are)?\s*(?:the\s+)?objects?\b",
        r"\bwhat objects? (?:are|do you|can you)\b",
        r"\bwhat(?:'s| is) (?:there|in front|on the table)\b",
        r"\b(?:list|name|tell me|show me)(?: the| all)? objects?\b",
        r"\bdescribe (?:the )?(?:scene|table|objects?)\b",
        r"\bwhat(?: objects?)? (?:are )?(?:present|visible|available)\b",
        r"^(?:observe|look|scan the table)$",
        # Hardware-lane additions (phase 9). Every changed row is pinned
        # one-for-one in tests/test_hardware_language.py::HARDWARE_MAPPINGS and
        # every unchanged boundary row in its RESTORED table; extend those tables
        # when you extend this tuple.
        #
        #   phrase                           was (pick verb "take")   now
        #   "take a picture|photo|snapshot"  pick "picture"           observe
        #   "... of <anything>"              pick "picture of ..."    observe
        #   "... please" / "... now"         pick "picture"           observe
        #   "take a picture frame"           pick "picture frame"     pick "picture frame"
        #   "take a look" (no "at")          pick "look"              observe
        #   "take a look at X"               look_at X                look_at X
        #
        # "take a picture" is a request to perceive, not to pick up an object
        # called "picture"; it must be claimed here, before the pick matcher sees
        # "take". The noun must END the request (allowing only the politeness
        # filler "please"/"now" after it) or be followed by "of": anything else
        # after it is an object name -- "take a picture frame" is a pick of the
        # frame, and the pattern must not swallow it.
        r"\btake a (?:picture|photo|snapshot)(?: of\b|(?: please| now)*$)",
        # "take a look" with no object is the same request; "take a look at X" is
        # left for the look_at matcher.
        r"\btake a look\b(?! at\b)",
        # Language stream, 2026-09-27 (fresh evaluation set; the 3B model
        # answered "unknown" to it after the prompt grew). Pinned in
        # tests/test_hardware_language.py::HARDWARE_MAPPINGS:
        #
        #   phrase                           was        now
        #   "how many objects do you see"    UNPARSED   observe
        #   "how many things are there"      UNPARSED   observe
        r"\bhow many (?:objects?|things|items)\b",
    )

    def _match_observe(self, text: str) -> Intent | None:
        for pattern in self._OBSERVE_PATTERNS:
            if re.search(pattern, text):
                return Intent("observe", {}, 1.0)
        return None

    def _match_scan(self, text: str) -> Intent | None:
        if re.search(r"\b(scan|survey|sweep)\b", text):
            return Intent("scan_scene", {}, 1.0)
        return None

    def _match_look(self, text: str) -> Intent | None:
        match = re.search(r"\blook at (.+)$", text)
        if match:
            return Intent("look_at", {"target": self._clean_target(match.group(1))}, 0.95)
        return None

    def _match_wait(self, text: str) -> Intent | None:
        # "hold" is deliberately not matched alone: "hold the can" is not a wait.
        #
        # Hardware-lane vocabulary (phase 9), pinned one-for-one in
        # tests/test_hardware_language.py::HARDWARE_MAPPINGS:
        #
        #   phrase               was        now    why
        #   "hold still"         UNPARSED   wait   what a student says to a moving arm
        #   "hold there"         UNPARSED   wait   same
        #   "hold position"      place {}   wait   "position" is a place verb; a place
        #                                          with no destination would carry the
        #                                          object back to its pick origin
        #   "stand by"/"standby" UNPARSED   wait   same request as "wait"
        #
        # "hold on" and "stay" already mapped to wait before phase 9.
        if re.search(r"\b(wait|hold (?:on|still|there|position)|stand ?by|pause|stay)\b", text):
            # Spoken number words count when a unit follows (language stream,
            # 2026-09-26; ASR writes numbers as words about as often as digits).
            # Same conversion in _match_rotate, _match_relative_move and
            # _place_direction. Pinned in
            # tests/test_hardware_language.py::HARDWARE_MAPPINGS:
            #
            #   phrase                               was              now
            #   "pause for five seconds"             wait 1 s         wait 5 s
            #   "wait for two seconds"               wait 1 s         wait 2 s
            #   "move left five centimetres"         no distance      0.05 m
            #   "move up twenty mm"                  no distance      0.02 m
            #   "rotate the wrist forty five degrees" 90 deg          45 deg
            #
            # "that one" / "the one on the left" have no unit after "one" and
            # are untouched (see _digits_for_number_words).
            seconds = self._find_number(_digits_for_number_words(text))
            return Intent("wait", {"duration": seconds if seconds is not None else 1.0}, 1.0)
        return None

    def _match_rotate(self, text: str) -> Intent | None:
        if not re.search(r"\b(rotate|turn|twist|spin)\b", text):
            return None
        if "wrist" not in text and "gripper" not in text and "hand" not in text:
            return None

        degrees = self._find_number(_digits_for_number_words(text))
        angle = (degrees or 90.0) * 3.141592653589793 / 180.0
        if re.search(r"\b(counter ?clockwise|left|anti ?clockwise)\b", text):
            angle = abs(angle)
        elif re.search(r"\b(clockwise|right)\b", text):
            angle = -abs(angle)
        return Intent("rotate_wrist", {"angle": angle}, 0.9)

    def _match_transfer(self, text: str) -> Intent | None:
        """"Move the red block to the bowl" -> its FIRST action, ``pick``.

        The Assistant runs both halves via :func:`split_transfer`; a caller that
        hands this utterance straight to the planner gets the pick alone, the
        same one-utterance-one-action rule as "pick X and put it in Y". With a
        pronoun ("move it next to the can") there is nothing to pick, and the
        intent is the place itself.
        """
        split = split_transfer(text)
        if split is None:
            return None
        if split.pick_clause is None:
            return Intent("place", dict(split.place_params), 0.95)
        return Intent("pick", {"target": split.object_phrase}, 0.9)

    #: "Raise/lift/lower the arm" moves the arm itself (language stream,
    #: 2026-09-27). Pinned in tests/test_hardware_language.py::HARDWARE_MAPPINGS:
    #:
    #:   phrase                        was                          now
    #:   "lift the arm up by 5 cm"     pick "arm up by 5 cm"        move_relative up 0.05
    #:   "raise the arm a bit"         UNPARSED                     move_relative up
    #:   "lower the gripper 3 cm"      UNPARSED                     move_relative down 0.03
    #:
    #: Only when the object is the robot's own arm/hand/gripper: "lift the can"
    #: is still a pick and "lower the can into the bowl" still a place.
    _VERTICAL_SELF = re.compile(
        r"^(?:(?:please|now|can you|could you|would you)\s+)*"
        r"(?P<verb>raise|lift|lower|elevate)\s+(?:(?:the|your|my|robot)\s+)*"
        r"(?:arm|hand|gripper|claw|wrist|tool|yourself)\b(?P<rest>.*)$"
    )

    def _match_vertical_self(self, text: str) -> Intent | None:
        match = self._VERTICAL_SELF.match(text)
        if match is None:
            return None
        rest = match.group("rest")
        words = set(rest.split())
        if any(w in words for w in ("left", "right", "forward", "forwards", "back", "backward", "backwards")):
            return None  # a sideways move is not what these verbs say; let the model or a refusal decide
        explicit = "up" if "up" in words else "down" if "down" in words else None
        direction = explicit or ("down" if match.group("verb") == "lower" else "up")
        params: dict[str, Any] = {"direction": direction}
        distance = self._find_distance(_digits_for_number_words(rest))
        if distance is not None:
            params["distance"] = distance
        return Intent("move_relative", params, 0.9)

    def _match_relative_move(self, text: str) -> Intent | None:
        vertical = self._match_vertical_self(text)
        if vertical is not None:
            return vertical
        verb = re.search(r"\b(move|go|shift|nudge|step)\b", text)
        if not verb:
            return None

        # A named object between the verb and the direction ("nudge the can
        # left", "go pick the block on the left") is not a move of the arm: the
        # audit measured "move the red object to the left" -> move_relative, the
        # object silently dropped. Transfers are claimed by _match_transfer;
        # anything else that names an object is left to the later matchers.
        positions = [
            m.start() for word in _DIRECTION_WORDS for m in re.finditer(rf"\b{word}\b", text)
        ]
        if positions:
            between = text[verb.end(): min(positions)].split()
            if any(
                w not in _MOTION_FILLER and not re.fullmatch(r"[\d.]+", w) for w in between
            ):
                return None

        for word, direction in _DIRECTION_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                params: dict[str, Any] = {"direction": direction}
                distance = self._find_distance(_digits_for_number_words(text))
                if distance is not None:
                    params["distance"] = distance
                return Intent("move_relative", params, 0.95)
        return None

    def _match_place(self, text: str) -> Intent | None:
        # "drop" reaches here only with a destination; ``_match_gripper`` has
        # already claimed the bare form.
        verb = re.search(r"\b(place|put|set|drop off|drop|position)\b", text)
        if not verb:
            return None

        after = text[verb.end():]
        # Leftmost relation after the verb, longest phrase at that position
        # ("on top of" beats "on"), so a destination keeps its qualifiers:
        # "put it in the bowl on the left" -> target "bowl on the left".
        for match in _PLACE_RELATION_RE.finditer(after):
            tail = after[match.end():].strip()
            if not tail:
                continue
            relation = _PLACE_RELATION_WORDS[match.group(0)]
            if relation in ("to", "on"):
                # "put it on the left" / "place it to the right by 5 cm" name a
                # direction, not an object called "left".
                direction = _direction_of(_strip_distance(tail)[0])
                if direction is not None:
                    return self._place_direction(direction, text)
            return Intent("place", {"relation": relation, "target": self._clean_target(tail)}, 0.95)

        direction = self._bare_direction(after)
        if direction is not None:
            return self._place_direction(direction, text)
        return Intent("place", {}, 0.95)

    def _place_direction(self, direction: str, text: str) -> Intent:
        distance = self._find_distance(_digits_for_number_words(text))
        return Intent(
            "place",
            {
                "relation": "direction",
                "direction": direction,
                "distance": distance if distance is not None else DEFAULT_PLACE_OFFSET_M,
            },
            0.9,
        )

    @staticmethod
    def _bare_direction(after: str) -> str | None:
        """"place it left 5 cm" -> "left". Not "put it right there" (an intensifier),
        and not a bare "put it back", which means return it where it came from."""
        match = re.search(r"\b(left|right)\b(?!\s+(?:there|here|now|away|of)\b)", after)
        if match:
            return match.group(1)
        match = re.search(r"\b(forwards?|ahead|backwards?)\b", after)
        if match:
            return _PLACE_DIRECTION_WORDS[match.group(1)]
        if re.search(r"\bback\b", after) and re.search(r"\d", after):
            return "back"
        return None

    def _match_pick(self, text: str) -> Intent | None:
        match = re.search(r"\b(pick up|pick|grab|take|lift|get|fetch)\b\s*(.*)$", text)
        if not match:
            return None

        remainder = match.group(2).strip()
        remainder = re.sub(r"^up\s+", "", remainder)
        if not remainder:
            # "pick it up" with nothing else: the referent comes from memory.
            return Intent("pick", {"target": "it"}, 0.8)
        return Intent("pick", {"target": self._clean_target(remainder)}, 0.95)

    #: Approach phrasings (language stream, 2026-09-27). Pinned in
    #: tests/test_hardware_language.py::HARDWARE_MAPPINGS:
    #:
    #:   phrase                        was        now
    #:   "hover over the dice"         UNPARSED   move_to "dice"
    #:   "go over to the mug"          UNPARSED   move_to "mug"
    #:   "approach the sponge"         UNPARSED   move_to "sponge"
    #:   "come closer to the sponge"   UNPARSED   move_to "sponge"
    #:   "go to the bowl"              UNPARSED   move_to "bowl"
    #:
    #: This matcher runs last, so a direction ("go to the left") or home ("go
    #: to the home position") has already been claimed; a target that is only
    #: the robot itself or a deictic is left unparsed.
    _MOVE_TO = re.compile(
        r"\b(?:move (?:to|toward|towards|over)|hover (?:over|above)|go (?:over )?to|approach"
        r"|come (?:closer )?to)\s+(.+)$"
    )

    def _match_move_to(self, text: str) -> Intent | None:
        match = self._MOVE_TO.search(text)
        if match:
            target = self._clean_target(match.group(1))
            if not match.group(0).startswith("move ") and (
                not _names_an_object(target) or target in _OBJECT_PRONOUNS
                or not _names_a_destination(target)
            ):
                return None
            return Intent("move_to", {"target": target}, 0.9)
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_target(phrase: str) -> str:
        """Reduce a noun phrase to the referring term.

        Leaves pronouns intact so memory can resolve them: "it" must survive as
        "it" rather than being stripped to nothing.
        """
        cleaned = phrase.strip()
        # Benefactive "me"/"for me" names who the object is for, never the
        # object (language stream, 2026-09-26; rows pinned in
        # tests/test_hardware_language.py::HARDWARE_MAPPINGS):
        #
        #   phrase                               was                  now
        #   "could you grab the banana for me"   "banana for me"      "banana"
        #   "grab me that banana please"         "me that banana"     "banana"
        #   "get me the marker"                  "me the marker"      "marker"
        #
        # Grounding cannot recover these: "me" and "for" are not modifiers it
        # knows, so every one of them failed as "nothing in view is a me".
        cleaned = re.sub(r"^(?:me|us)\s+(?=\S)", "", cleaned)
        cleaned = re.sub(r"\s+for (?:me|us)(?=(?:\s+(?:please|now))?$)", "", cleaned)
        for article in _ARTICLES:
            if cleaned.startswith(article):
                cleaned = cleaned[len(article) :]
                break
        # Drop trailing filler, e.g. "the can please" or "the can up".
        cleaned = re.sub(r"\s+(please|now|up|down|there|here)$", "", cleaned).strip()
        return cleaned or "it"

    @staticmethod
    def _find_number(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _find_distance(text: str) -> float | None:
        """Extract a distance in metres, honouring the stated unit.

        Bare numbers are read as centimetres: a spoken "move left 10" means 10 cm,
        and interpreting it as 10 metres would command a wild motion.
        """
        match = re.search(r"(\d+(?:\.\d+)?)\s*(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?)\b", text)
        if match:
            value, unit = float(match.group(1)), match.group(2)
            if unit.startswith("mm") or unit.startswith("millimet"):
                return value / 1000.0
            if unit.startswith("cm") or unit.startswith("centimet"):
                return value / 100.0
            return value

        bare = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(bare.group(1)) / 100.0 if bare else None


#: Spoken number words -> digits, so "five centimetres" is re-read exactly like
#: "5 centimetres" (the model's own number is never trusted, see ``_validated``).
_UNITS_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS_WORDS) + r")(?:[ -](?P<unit>"
    + "|".join(w for w in _UNITS_WORDS if _UNITS_WORDS[w] < 10 and w != "zero")
    + r"))?|(?P<single>" + "|".join(_UNITS_WORDS) + r"))\b"
)
#: A number word only counts as a quantity when a unit or quantity word follows;
#: "that one" / "the one on the left" are referents, not numbers.
_QUANTITY_AFTER = re.compile(
    r"^\s*(?:mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|"
    r"seconds?|secs?|s|degrees?|deg|radians?|rad)\b"
)

#: Words a raised/lowered arm is described with, when no direction word is said.
_VERTICAL_VERBS = {"raise": "up", "lift": "up", "elevate": "up", "lower": "down", "drop": "down"}

#: A place destination that is not a destination (it names the moved object).
_NOT_A_DESTINATION = frozenset(_OBJECT_PRONOUNS | {"one", "object", "here", "there", "itself"})
_TARGET_FILLER = frozenset({"the", "a", "an", "my", "your", "that", "this", "please", "now", "up"})


def _digits_for_number_words(text: str) -> str:
    """"move up five centimetres" -> "move up 5 centimetres" (quantities only)."""

    def repl(match: re.Match[str]) -> str:
        if not _QUANTITY_AFTER.match(text[match.end():]):
            return match.group(0)
        if match.group("single"):
            return str(_UNITS_WORDS[match.group("single")])
        return str(_TENS_WORDS[match.group("tens")] + _UNITS_WORDS.get(match.group("unit") or "", 0))

    return _NUMBER_WORD_RE.sub(repl, text)


def _evidence(*stems: str, exact: tuple[str, ...] = ()) -> re.Pattern[str]:
    parts = [re.escape(stem).replace(r"\ ", " ") + r"\w*" for stem in stems]
    parts += [re.escape(word).replace(r"\ ", " ") for word in exact]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


#: Hybrid mode (review finding 3): the model is asked only about what the
#: grammar refused -- ASR noise, background talk -- and measured Qwen turns
#: "hmm okay" into open_gripper and "thanks robot" into go_home. An LLM intent
#: for a skill in this table is accepted only when the operator's first clause
#: holds a word that asks for that skill (word stems; "pick" covers "picking").
#: move_relative additionally needs a spoken direction (checked in
#: ``_validated``). Skills that do not move the arm (observe, wait, stop,
#: emergency_stop) need no evidence. The measured LLM successes all pass.
_SKILL_EVIDENCE: dict[str, re.Pattern[str]] = {
    "open_gripper": _evidence("open", "releas", "drop", "loos", "ungrip", "unclench",
                              exact=("let go", "let it go", "let that go", "let them go")),
    "close_gripper": _evidence("close", "closing", "shut", "grip", "grasp", "clamp", "squeez", "clench",
                               "pinch", "tighten"),
    "go_home": _evidence("home", "rest", "park", "start", "initial", "neutral", "default", "reset"),
    "scan_scene": _evidence("scan", "survey", "search", "sweep", "explor", exact=("around", "room")),
    "pick": _evidence("pick", "grab", "take", "took", "get", "lift", "fetch", "grasp", "collect", "hold",
                      "bring", "snatch", "seiz", "carry", "hand", "nab", "retriev", "pluck", "scoop",
                      "gather"),
    "place": _evidence("put", "place", "drop", "set", "leave", "toss", "throw", "stick", "lay", "deposit",
                       "releas", "bring", "carry", "pop", "stash", "move", "dump", "stack", "insert",
                       "plop", "position", "return"),
    "move_to": _evidence("move", "go", "reach", "hover", "approach", "point", "head", "come", "travel",
                         exact=("over", "toward", "towards", "to", "near", "above")),
    "look_at": _evidence("look", "point", "face", "aim", "turn", "watch", "focus", "see", "check"),
    "rotate_wrist": _evidence("rotat", "turn", "twist", "spin", "roll", "clockwise", "wrist", "degree",
                              "flip"),
    "move_relative": _evidence("move", "go", "nudge", "shift", "slide", "scoot", "back", "raise", "lift",
                               "lower", "bring", "push", "pull", "step", "inch", "jog",
                               exact=("left", "right", "forward", "forwards", "ahead", "up", "down")),
}
#: Skills a negated utterance may still become (they stop or do nothing).
_ALWAYS_SAFE_SKILLS = frozenset({"stop", "emergency_stop"})


def _word_in(word: str, words: set[str]) -> bool:
    """``word`` was said, allowing a plural/singular difference."""
    return (
        word in words
        or f"{word}s" in words
        or f"{word}es" in words
        or (word.endswith("s") and word[:-1] in words)
        or (word.endswith("es") and word[:-2] in words)
    )


class LlmIntentParser(IIntentParser):
    """Language-model parser constrained to a single atomic skill.

    ``complete`` is any callable taking a prompt and returning text, so this works
    with a local model, a hosted API, or a stub in tests without the framework
    depending on any particular SDK.

    ``mode`` is one of :data:`LLM_MODES`. ``"hybrid"`` never lets the model
    override a rule parse that succeeded; see the module docstring for the
    measurement behind it. ``"llm-first"`` stays the class default only because
    existing callers construct the parser directly and rely on it;
    ``scripts/run_assistant.py`` selects :data:`DEFAULT_LLM_MODE`.

    Which parser produced each intent is logged (``intent source=...``) and kept
    on :attr:`last_source` and on the returned intent's ``source`` attribute
    (an attribute, not a key: the intent dict itself is unchanged).
    """

    #: The instruction is explicit that exactly one skill may be emitted. A model
    #: asked to "pick up the can and put it in the box" will otherwise happily
    #: return a two-step plan, which is precisely the behaviour this framework
    #: forbids.
    #:
    #: Everything that does not change per command comes first, so llama-server's
    #: prompt cache reuses it; only the visible objects, the held object and the
    #: command are evaluated per call (matters on the Jetson's prompt rate).
    #: ``{formats}``, ``{rules}`` and ``{examples}`` list only the robot's own
    #: skills: the hardware arm has no ``rotate_wrist``, so the model is never
    #: told about it.
    PROMPT_TEMPLATE = """You translate a robot operator's command into exactly ONE action for a small robot arm.

Available actions: {skills}

Parameter format for each action:
{formats}

Rules:
- Emit exactly ONE action. Never a sequence, never two actions.
- If the command implies several steps (e.g. "pick X and put it in Y"), emit only the FIRST action.
{rules}
- Reply with JSON only. No explanation, no extra text.

Examples:
{examples}

Visible objects (for context only; never copy a name from this list into "target"): {objects}
Currently held: {held}

Command: {utterance}
JSON:"""

    #: One line per skill; only registered skills are shown.
    _FORMATS: dict[str, str] = {
        "pick": '- pick: {"skill": "pick", "params": {"target": "<the object, in the operator\'s own words>"}, "confidence": <0-1>}',
        "place": (
            '- place: {"skill": "place", "params": {"relation": "in|on|next_to|to|left_of|right_of|in_front_of|behind", '
            '"target": "<the destination the operator named>"}, "confidence": <0-1>}\n'
            '  or {"skill": "place", "params": {"relation": "direction", "direction": "left|right|forward|back", '
            '"distance": <metres>}, "confidence": <0-1>}\n'
            '  or {"skill": "place", "params": {}, "confidence": <0-1>} to put it back where it was picked'
        ),
        "move_to": '- move_to: {"skill": "move_to", "params": {"target": "<object_name>"}, "confidence": <0-1>}',
        "look_at": '- look_at: {"skill": "look_at", "params": {"target": "<object_name>"}, "confidence": <0-1>}',
        "move_relative": (
            '- move_relative: {"skill": "move_relative", "params": {"direction": "left|right|forward|backward|up|down", '
            '"distance": <metres, only if spoken>}, "confidence": <0-1>}'
        ),
        "rotate_wrist": '- rotate_wrist: {"skill": "rotate_wrist", "params": {"angle": <radians>}, "confidence": <0-1>}',
        "wait": '- wait: {"skill": "wait", "params": {"duration": <seconds>}, "confidence": <0-1>}',
    }

    #: (skills the rule needs, rule text). A rule about a missing skill is left out.
    _RULES: tuple[tuple[frozenset[str], str], ...] = (
        (frozenset(),
         '- "target" copies the operator\'s own words for the object, qualifiers included '
         '(e.g. "small red block on the left"). Never add a colour, size or name the operator did not say.'),
        (frozenset({"pick"}),
         '- A pronoun ("it", "that", "this one") means the object being picked or already held: '
         'use "it" as the target and do not resolve it.'),
        (frozenset({"place"}),
         '- When something is already held, "<any verb> it in/into/on/onto <a place>" (put, toss, throw, '
         'stick, dump, drop...) is place, never pick: the held object cannot be picked again.'),
        (frozenset({"pick", "place"}),
         '- "put/move/bring <a named object> in/on/to <a place>" starts with picking that object: '
         'the FIRST action is pick with the named object as target.'),
        (frozenset({"place"}),
         '- For place, "it"/"that" is the object being moved, NEVER the destination. '
         'The place "target" is the named destination (bowl, box, bin, tray...). '
         '"put it in the box" -> target "box".'),
        (frozenset(),
         f'- If the command names no object where one is needed, or is not a robot command, reply '
         f'{{"skill": "{REFUSAL_SKILL}", "params": {{}}, "confidence": 0}}. Never guess an object.'),
        (frozenset({"move_relative"}),
         '- Distances are in metres: 20 mm = 0.02, 3 cm = 0.03, 0.3 m = 0.3. '
         'Leave distance out when the operator gives none.'),
        (frozenset({"rotate_wrist"}),
         '- Angles are radians. Clockwise is NEGATIVE, counterclockwise/anticlockwise is POSITIVE.'),
        (frozenset({"open_gripper"}),
         '- "drop it", "let go", "release it", "open your hand" -> open_gripper. '
         'Only "drop it in/on <a named place>" is a place.'),
        (frozenset({"wait"}),
         '- "hold still", "hold on", "pause", "stay", "stand by" -> wait (never stop).'),
        (frozenset({"move_relative"}),
         '- "raise"/"lower" move the whole arm: "raise the arm", "lift your hand", "lower the gripper", '
         '"go up/down" -> move_relative up/down. They never open or close the gripper and never rotate '
         'the wrist. Lifting a named object is a pick.'),
        (frozenset({"move_to"}),
         '- "go to", "go over to", "hover over", "approach", "come closer to" <an object> -> move_to that '
         'object (the arm moves above it; nothing is picked).'),
        (frozenset({"observe"}),
         '- Questions about what is there ("what do you see", "what is in front of you", '
         '"what objects are on the table", "take a picture") -> observe.'),
    )

    #: (skills the example needs, command, visible objects, reply). Written for
    #: the error classes measured on 2026-09-26, with objects and wordings that
    #: are NOT in the evaluation set, so they teach a rule rather than an answer.
    _EXAMPLES: tuple[tuple[frozenset[str], str, str, str], ...] = (
        (frozenset({"pick"}), "grab the cup", "blue cup, green tray",
         '{"skill": "pick", "params": {"target": "cup"}, "confidence": 0.9}'),
        (frozenset({"place"}), "now put it into the tray", "blue cup, green tray",
         '{"skill": "place", "params": {"relation": "in", "target": "tray"}, "confidence": 0.9}'),
        (frozenset({"place"}), "set that on the plate", "white plate",
         '{"skill": "place", "params": {"relation": "on", "target": "plate"}, "confidence": 0.9}'),
        (frozenset({"place"}), "chuck it onto the tray", "blue cup, green tray; held: blue cup",
         '{"skill": "place", "params": {"relation": "on", "target": "tray"}, "confidence": 0.9}'),
        (frozenset({"move_to"}), "hover above the cup", "blue cup, green tray",
         '{"skill": "move_to", "params": {"target": "cup"}, "confidence": 0.9}'),
        (frozenset({"open_gripper"}), "let it fall", "green tray",
         '{"skill": "open_gripper", "params": {}, "confidence": 0.9}'),
        (frozenset({"wait"}), "stay still for 2 seconds", "green tray",
         '{"skill": "wait", "params": {"duration": 2}, "confidence": 0.9}'),
        (frozenset({"move_relative"}), "move down 15 mm", "green tray",
         '{"skill": "move_relative", "params": {"direction": "down", "distance": 0.015}, "confidence": 0.9}'),
        (frozenset({"move_relative"}), "lift your hand up a little", "green tray",
         '{"skill": "move_relative", "params": {"direction": "up"}, "confidence": 0.8}'),
        (frozenset({"rotate_wrist"}), "twist the wrist 30 degrees clockwise", "green tray",
         '{"skill": "rotate_wrist", "params": {"angle": -0.5236}, "confidence": 0.9}'),
        (frozenset(), "sing me a song", "green tray",
         f'{{"skill": "{REFUSAL_SKILL}", "params": {{}}, "confidence": 0}}'),
    )

    def __init__(
        self,
        complete: Callable[[str], str],
        known_skills: tuple[str, ...],
        fallback: IIntentParser | None = None,
        mode: str = "llm-first",
    ) -> None:
        self._complete = complete
        self.known_skills = tuple(known_skills)
        self._fallback = fallback or RuleBasedIntentParser(known_skills)
        self.mode = mode
        self.last_source: str | None = None
        #: Called as ``llm_gate(intent, utterance, context)`` with every intent
        #: the MODEL produced (never a rule parse), after validation and before
        #: it is returned. ``utterance`` is what the operator said, correction
        #: marker included (the model saw the command after it), so a refusal
        #: quotes the operator exactly. It may raise :class:`UnparsedCommand` (usually
        #: :class:`NotConfirmed`) to refuse. ``None`` (the default): no gate,
        #: so the parser alone behaves exactly as measured. The Assistant
        #: installs one that asks before a model-chosen release of a held object.
        self.llm_gate: Callable[[Intent, str, dict[str, Any]], None] | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in LLM_MODES:
            raise ValueError(f"LLM parser mode must be one of {LLM_MODES}, got {value!r}")
        self._mode = value

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------

    def build_prompt(self, utterance: str, context: dict[str, Any] | None = None) -> str:
        """The full prompt for one command (public so tests and harnesses can inspect it)."""
        context = context or {}
        known = set(self.known_skills)
        formats = "\n".join(line for skill, line in self._FORMATS.items() if skill in known)
        no_params = [s for s in self.known_skills if s not in self._FORMATS]
        if no_params:
            formats += f"\n- Others ({', '.join(no_params)}): no params needed."
        rules = "\n".join(text for needs, text in self._RULES if needs <= known)
        examples = "\n".join(
            f"Command: {command}  (visible: {visible})\nJSON: {reply}"
            for needs, command, visible, reply in self._EXAMPLES
            if needs <= known
        )
        return self.PROMPT_TEMPLATE.format(
            skills=", ".join((*self.known_skills, REFUSAL_SKILL)),
            formats=formats,
            rules=rules,
            examples=examples,
            objects=", ".join(context.get("visible_objects", [])) or "none",
            held=context.get("held_object") or "nothing",
            utterance=utterance,
        )

    # ------------------------------------------------------------------
    # policies
    # ------------------------------------------------------------------

    def parse(self, utterance: str, context: dict[str, Any] | None = None) -> Intent:
        context = context or {}
        if self._mode == "hybrid":
            return self._parse_hybrid(utterance, context)
        return self._parse_llm_first(utterance, context)

    def _parse_hybrid(self, utterance: str, context: dict[str, Any]) -> Intent:
        """Grammar first; the model only for what the grammar refuses."""
        try:
            intent = self._fallback.parse(utterance, context)
        except (UnregisteredSkill, NegatedCommand):
            # Understood, but this robot lacks the skill -- or the operator
            # said what NOT to do: no phrasing problem for a model to solve,
            # and any skill it found would be wrong.
            self._record(None, "rule-refused")
            raise
        except UnparsedCommand as refusal:
            if not utterance or not utterance.strip():
                self._record(None, "rule-refused")
                raise
            # The model sees the command after a correction marker, never the
            # marker itself ("no, put it in the bowl" -> "put it in the bowl").
            command = _without_correction(utterance)
            try:
                intent = self._ask_model(command, context, strict=True)
            except Exception as exc:  # noqa: BLE001 - any model failure keeps the refusal
                _log.info("LLM could not parse %r either (%s); keeping the rule parser's refusal",
                          utterance, exc)
                self._record(None, "rule-refused")
                raise refusal from None
            self._apply_gate(intent, utterance, context)
            return self._record(intent, "llm")
        return self._record(intent, "rule")

    def _parse_llm_first(self, utterance: str, context: dict[str, Any]) -> Intent:
        """Model first; the grammar when the model is unavailable or invalid."""
        if is_negated(utterance):
            # The grammar owns negation (a stop word wins, anything else is refused).
            return self._record(self._fallback.parse(utterance, context), "rule-fallback")
        command = _without_correction(utterance)
        try:
            intent = self._ask_model(command, context, strict=False)
        except _UnknownSkillFromModel as exc:
            _log.warning("LLM proposed unknown skill %r; using the rule-based parser", exc.skill)
        except Exception as exc:  # noqa: BLE001
            # Degrade to the grammar rather than refusing: a model outage should
            # not stop the robot understanding "stop".
            _log.warning("LLM intent parsing failed (%s); using the rule-based parser", exc)
        else:
            self._apply_gate(intent, utterance, context)
            return self._record(intent, "llm")
        return self._record(self._fallback.parse(utterance, context), "rule-fallback")

    def _apply_gate(self, intent: Intent, utterance: str, context: dict[str, Any]) -> None:
        """Run :attr:`llm_gate` on a model intent; a refusal is recorded and raised."""
        if self.llm_gate is None:
            return
        try:
            self.llm_gate(intent, utterance, context)
        except UnparsedCommand:
            self._record(None, "llm-unconfirmed")
            raise

    def _record(self, intent: Intent | None, source: str) -> Intent:
        self.last_source = source
        if intent is None:
            _log.info("intent source=%s (mode=%s): refused", source, self._mode)
            return intent  # type: ignore[return-value]
        try:
            intent.source = source  # type: ignore[attr-defined]
        except AttributeError:  # a fallback returning a plain dict
            pass
        _log.info("intent source=%s (mode=%s) skill=%r params=%r",
                  source, self._mode, intent["skill"], intent["params"])
        return intent

    # ------------------------------------------------------------------
    # the model call and its checks
    # ------------------------------------------------------------------

    def _ask_model(self, utterance: str, context: dict[str, Any], strict: bool) -> Intent:
        """One model call -> a validated intent, or an exception (never a guess)."""
        raw = self._complete(self.build_prompt(utterance, context))
        _log.info("LLM intent raw response: %s", raw.strip())
        intent = self._parse_response(raw)
        if intent.skill == REFUSAL_SKILL:
            raise UnparsedCommand(f"the LLM declined {utterance!r}")
        if intent.skill not in self.known_skills:
            raise _UnknownSkillFromModel(intent.skill)
        intent = self._validated(intent, utterance, strict, context.get("held_object"))
        _log.info(
            "LLM intent parsed: skill=%r params=%r confidence=%.2f",
            intent.skill, intent.params, intent.confidence,
        )
        return intent

    def _validated(
        self, intent: Intent, utterance: str, strict: bool, held_object: Any = None
    ) -> Intent:
        """Check and normalise a model intent before it can move the arm.

        * Numbers (distance, duration, angle) are re-read from the utterance
          with the grammar's own unit-aware reader; the model only decides the
          structure. Measured: "move up 20 mm" came back as distance 20, and a
          bare "move backwards" as 0.5 m. No number spoken -> the skill default.
        * The rotation sign comes from "clockwise"/"counterclockwise" in the
          utterance (the model had it backwards).
        * A move direction must agree with a direction word the operator said.
        * A place destination may not be a pronoun (measured: "put it in the
          box" -> target "it", i.e. place the object into itself).
        * ``strict`` (hybrid): every target word must have been spoken; the
          model may not invent "red block" from the visible-objects list when
          the operator said "the block".
        * A pick of a pronoun while something is held is refused: the pronoun
          names the held object, which cannot be picked again (measured
          2026-09-27: "toss it in the bin" while holding the eraser -> pick
          "it").

        Raises :class:`UnparsedCommand` when the intent cannot be trusted.
        """
        skill = intent.skill
        params = dict(intent.params)
        text = RuleBasedIntentParser._first_clause(
            strip_correction(RuleBasedIntentParser._normalise(utterance))
        )
        if _NEGATION_RE.search(text) and skill not in _ALWAYS_SAFE_SKILLS:
            raise NegatedCommand(f"LLM {skill} for a negated command {utterance!r}")
        if strict:
            evidence = _SKILL_EVIDENCE.get(skill)
            if evidence is not None and evidence.search(text) is None:
                raise UnparsedCommand(
                    f"LLM {skill} but {utterance!r} has no word asking for it; refusing rather than guessing"
                )
        numeric = _digits_for_number_words(text)
        has_number = re.search(r"\d", numeric) is not None
        out: dict[str, Any]

        if skill in ("pick", "move_to", "look_at"):
            out = {"target": self._clean_model_target(params.get("target"), skill, utterance, strict)}
            if skill == "pick" and held_object and out["target"] in _OBJECT_PRONOUNS:
                raise UnparsedCommand(
                    f"LLM pick of {out['target']!r} while holding {held_object!r}: "
                    "the pronoun names the held object"
                )
        elif skill == "place":
            out = self._validated_place(params, numeric, has_number, utterance, strict)
        elif skill == "move_relative":
            direction = _DIRECTION_WORDS.get(str(params.get("direction", "")).strip().lower())
            if direction is None:
                raise UnparsedCommand(f"LLM move_relative without a valid direction: {params!r}")
            spoken = {_DIRECTION_WORDS[w] for w in text.split() if w in _DIRECTION_WORDS}
            spoken |= {_VERTICAL_VERBS[w] for w in text.split() if w in _VERTICAL_VERBS}
            if strict and not spoken:
                # Review finding 3: "nudge it a little bit" came back as a move
                # left -- a direction the operator never said.
                raise UnparsedCommand(f"LLM move_relative {direction!r} but no direction was spoken")
            if spoken and direction not in spoken:
                raise UnparsedCommand(
                    f"LLM direction {direction!r} contradicts the spoken {sorted(spoken)}"
                )
            out = {"direction": direction}
            if has_number:
                out["distance"] = RuleBasedIntentParser._find_distance(numeric)
        elif skill == "rotate_wrist":
            degrees = RuleBasedIntentParser._find_number(numeric) if has_number else None
            if degrees is not None and re.search(r"\brad(?:ian)?s?\b", numeric):
                angle = degrees
            else:
                angle = math.radians(degrees if degrees is not None else 90.0)
            # The grammar's sign convention (_match_rotate): counterclockwise or
            # left is positive, clockwise or right negative.
            if re.search(r"\b(?:counter ?clockwise|anti ?clockwise|left)\b", text):
                angle = abs(angle)
            elif re.search(r"\b(?:clockwise|right)\b", text):
                angle = -abs(angle)
            else:
                try:
                    model_angle = float(params.get("angle"))
                except (TypeError, ValueError):
                    model_angle = 1.0
                angle = abs(angle) if model_angle >= 0 else -abs(angle)
            out = {"angle": angle}
        elif skill == "wait":
            seconds = RuleBasedIntentParser._find_number(numeric) if has_number else None
            out = {"duration": float(seconds) if seconds is not None else 1.0}
        else:
            # observe, scan_scene, gripper, go_home, stop, emergency_stop: no
            # params (measured: "let go of it" -> stop {"target": "it"}).
            out = {}

        try:
            confidence = min(1.0, max(0.0, float(intent.confidence)))
        except (TypeError, ValueError):
            confidence = 0.7
        return Intent(skill, out, confidence)

    def _validated_place(
        self, params: dict[str, Any], numeric: str, has_number: bool, utterance: str, strict: bool
    ) -> dict[str, Any]:
        relation_raw = str(params.get("relation", "") or "").strip().lower().replace("_", " ")
        target = params.get("target")
        if not relation_raw and not target and "direction" not in params:
            return {}  # back where it was picked
        if relation_raw == "direction" or (not target and "direction" in params):
            direction = _PLACE_DIRECTION_WORDS.get(str(params.get("direction", "")).strip().lower())
            if direction is None:
                raise UnparsedCommand(f"LLM directional place without a valid direction: {params!r}")
            distance = RuleBasedIntentParser._find_distance(numeric) if has_number else None
            return {
                "relation": "direction",
                "direction": direction,
                "distance": distance if distance is not None else DEFAULT_PLACE_OFFSET_M,
            }
        relation: str | None
        if not relation_raw:
            relation = "to"  # a destination with no relation: Place decides
        elif relation_raw.replace(" ", "_") in PLACE_RELATIONS:
            relation = relation_raw.replace(" ", "_")
        else:
            relation = _PLACE_RELATION_WORDS.get(relation_raw)
        if relation is None:
            raise UnparsedCommand(f"LLM place with an unknown relation {relation_raw!r}")
        cleaned = self._clean_model_target(target, "place", utterance, strict)
        if cleaned in _NOT_A_DESTINATION:
            raise UnparsedCommand(
                f"LLM place destination {cleaned!r} is a pronoun: it names the held object, not a place"
            )
        return {"relation": relation, "target": cleaned}

    @staticmethod
    def _clean_model_target(target: Any, skill: str, utterance: str, strict: bool) -> str:
        if not isinstance(target, str) or not target.strip():
            raise UnparsedCommand(f"LLM {skill} without a target")
        cleaned = RuleBasedIntentParser._clean_target(RuleBasedIntentParser._normalise(target))
        if not strict:
            return cleaned
        said = set(RuleBasedIntentParser._normalise(utterance).split())
        if cleaned in _OBJECT_PRONOUNS:
            if not said & (_OBJECT_PRONOUNS | {"one"}):
                raise UnparsedCommand(f"LLM target {cleaned!r} but the operator used no pronoun")
            return cleaned
        invented = [w for w in cleaned.split() if w not in _TARGET_FILLER and not _word_in(w, said)]
        if invented:
            raise UnparsedCommand(
                f"LLM target {cleaned!r} has words the operator did not say: {invented}"
            )
        return cleaned

    def _parse_response(self, raw: str) -> Intent:
        """Extract the JSON object from a model response."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise UnparsedCommand(f"no JSON object in model response: {raw[:120]!r}")

        data = json.loads(match.group(0))
        if not isinstance(data, dict) or "skill" not in data:
            raise UnparsedCommand(f"model response lacks a skill: {data!r}")

        # Guard against a model returning a plan despite the instruction.
        if isinstance(data.get("params"), list) or isinstance(data["skill"], list):
            raise UnparsedCommand("model returned a sequence; only one action is permitted")
        if data.get("params") is not None and not isinstance(data.get("params"), dict):
            raise UnparsedCommand(f"model params are not an object: {data.get('params')!r}")

        return Intent(
            skill=str(data["skill"]),
            params=dict(data.get("params") or {}),
            confidence=float(data.get("confidence", 0.7)),
        )


class _UnknownSkillFromModel(UnparsedCommand):
    """The model named a skill that is not registered (llm-first falls back)."""

    def __init__(self, skill: str) -> None:
        super().__init__(f"LLM proposed unknown skill {skill!r}")
        self.skill = skill

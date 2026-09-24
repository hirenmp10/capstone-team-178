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
* :class:`LlmIntentParser` -- delegates to a language model for open-ended
  phrasing, constrained to emit one skill from the registry. Falls back to the
  rule-based parser when the model is unavailable or returns something invalid,
  so language understanding degrades rather than failing.

**Moving a named object** ("move the red block to the bowl", "put the can in
the bowl", "move the red object to the left") is two actions: pick it, then
place it. The parser still returns one intent -- the first, ``pick`` -- exactly
as it does for "pick X and put it in Y". :func:`split_transfer` exposes the two
clauses so the Assistant can run them one after the other (stopping at the
first failure); that is the only place the expansion happens. Before this, the
audit measured "move the red object to the left" -> ``move_relative{left}``
(the object silently dropped, the empty arm moved) and "put the red block in
the bowl" -> ``place{in, bowl}`` (the block ignored; a place with nothing held).

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
    "TransferSplit",
    "split_transfer",
    "PLACE_RELATIONS",
    "PLACE_DIRECTIONS",
    "DEFAULT_PLACE_OFFSET_M",
]

_log = get_logger("language.intent")


class UnparsedCommand(MfwError):
    """The utterance did not match any known command."""


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

_TRANSFER_VERBS = ("move", "put", "place", "set", "bring", "transfer", "carry", "take",
                   "shift", "slide", "relocate")
_TRANSFER_RE = re.compile(
    r"^(?:(?:please|kindly|now|can you|could you|would you|will you|go ahead and|"
    r"i want you to|i need you to|i would like you to|i d like you to)\s+)*"
    rf"(?P<verb>{'|'.join(_TRANSFER_VERBS)})\s+(?P<rest>.+?)(?:\s+(?:please|now))*$"
)
_OBJECT_PRONOUNS = frozenset({"it", "that", "this", "them", "that one", "this one"})


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
    text = RuleBasedIntentParser._first_clause(text)
    match = _TRANSFER_RE.match(text)
    if match is None:
        return None
    verb, rest = match.group("verb"), match.group("rest")
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

    if not obj:
        return None
    pronoun = obj in _OBJECT_PRONOUNS
    if not pronoun and not _names_an_object(obj):
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

        text = self._normalise(utterance)
        # "Pick the can and put it in the box" is ONE command: pick. Truncating at
        # the conjunction implements "emit only the first action" directly, rather
        # than leaving it to matcher ordering -- which got this backwards, matching
        # the later "put" and silently turning a pick into a place.
        text = self._first_clause(text)

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
            raise UnparsedCommand(
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

    def _match_home(self, text: str) -> Intent | None:
        if re.search(r"\b(go home|home position|return home|reset arm)\b", text):
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
            seconds = self._find_number(text)
            return Intent("wait", {"duration": seconds if seconds is not None else 1.0}, 1.0)
        return None

    def _match_rotate(self, text: str) -> Intent | None:
        if not re.search(r"\b(rotate|turn|twist|spin)\b", text):
            return None
        if "wrist" not in text and "gripper" not in text and "hand" not in text:
            return None

        degrees = self._find_number(text)
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

    def _match_relative_move(self, text: str) -> Intent | None:
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
                distance = self._find_distance(text)
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
        distance = self._find_distance(text)
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

    def _match_move_to(self, text: str) -> Intent | None:
        match = re.search(r"\bmove (?:to|toward|towards|over)\s+(.+)$", text)
        if match:
            return Intent("move_to", {"target": self._clean_target(match.group(1))}, 0.9)
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


class LlmIntentParser(IIntentParser):
    """Language-model parser constrained to a single atomic skill.

    ``complete`` is any callable taking a prompt and returning text, so this works
    with a local model, a hosted API, or a stub in tests without the framework
    depending on any particular SDK.
    """

    #: The instruction is explicit that exactly one skill may be emitted. A model
    #: asked to "pick up the can and put it in the box" will otherwise happily
    #: return a two-step plan, which is precisely the behaviour this framework
    #: forbids.
    PROMPT_TEMPLATE = """You translate a robot operator's command into exactly ONE action.

Available actions: {skills}

Required parameter format for each skill:
- pick:         {{"skill": "pick",  "params": {{"target": "<object_name>"}}, "confidence": <0-1>}}
- place:        {{"skill": "place", "params": {{"relation": "in|on|next_to|to|left_of|right_of|in_front_of|behind", "target": "<object_name>"}}, "confidence": <0-1>}}
                or {{"skill": "place", "params": {{"relation": "direction", "direction": "left|right|forward|back", "distance": <metres>}}, "confidence": <0-1>}}
                or {{"skill": "place", "params": {{}}, "confidence": <0-1>}} to put it back where it was picked
- move_to:      {{"skill": "move_to", "params": {{"target": "<object_name>"}}, "confidence": <0-1>}}
- look_at:      {{"skill": "look_at", "params": {{"target": "<object_name>"}}, "confidence": <0-1>}}
- move_relative:{{"skill": "move_relative", "params": {{"direction": "left|right|forward|backward|up|down", "distance": <metres>}}, "confidence": <0-1>}}
- rotate_wrist: {{"skill": "rotate_wrist", "params": {{"angle": <radians>}}, "confidence": <0-1>}}
- wait:         {{"skill": "wait", "params": {{"duration": <seconds>}}, "confidence": <0-1>}}
- Others (observe, scan_scene, open_gripper, close_gripper, go_home, stop, emergency_stop): no params needed.

Rules:
- Emit exactly ONE action. Never a sequence, never two actions.
- If the command implies several steps (e.g. "pick X and put it in Y"), emit only the FIRST action (pick).
- "put", "place", "drop", "set" map to the "place" action; "pick", "grab", "take", "lift", "get", "fetch" map to "pick".
- Always include the correct params as shown above — never leave params empty for pick.
- Keep the whole object description in "target", qualifiers included (e.g. "small red block on the left").
- Use "it"/"that" as the target when the operator used a pronoun; do not resolve it.
- Reply with JSON only. No explanation, no extra text.

Visible objects: {objects}
Currently held: {held}

Command: {utterance}
JSON:"""

    def __init__(
        self,
        complete: Callable[[str], str],
        known_skills: tuple[str, ...],
        fallback: IIntentParser | None = None,
    ) -> None:
        self._complete = complete
        self.known_skills = tuple(known_skills)
        self._fallback = fallback or RuleBasedIntentParser(known_skills)

    def parse(self, utterance: str, context: dict[str, Any] | None = None) -> Intent:
        context = context or {}
        prompt = self.PROMPT_TEMPLATE.format(
            skills=", ".join(self.known_skills),
            objects=", ".join(context.get("visible_objects", [])) or "none",
            held=context.get("held_object") or "nothing",
            utterance=utterance,
        )

        try:
            raw = self._complete(prompt)
            _log.info("LLM intent raw response: %s", raw.strip())
            intent = self._parse_response(raw)
            _log.info(
                "LLM intent parsed: skill=%r params=%r confidence=%.2f",
                intent.skill, intent.params, intent.confidence,
            )
        except Exception as exc:
            # Degrade to the grammar rather than refusing: a model outage should
            # not stop the robot understanding "stop".
            _log.warning("LLM intent parsing failed (%s); using the rule-based parser", exc)
            return self._fallback.parse(utterance, context)

        if intent.skill not in self.known_skills:
            _log.warning(
                "LLM proposed unknown skill %r; using the rule-based parser", intent.skill
            )
            return self._fallback.parse(utterance, context)
        return intent

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

        return Intent(
            skill=str(data["skill"]),
            params=dict(data.get("params") or {}),
            confidence=float(data.get("confidence", 0.7)),
        )

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
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Callable

from mfw.core.errors import MfwError
from mfw.core.interfaces import IIntentParser
from mfw.utils.logging import get_logger

__all__ = ["RuleBasedIntentParser", "LlmIntentParser", "Intent", "UnparsedCommand"]

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

    def _match_gripper(self, text: str) -> Intent | None:
        # "open"/"close" alone are unambiguous in a manipulation context, but
        # must not fire on "open the box", which is not a gripper command.
        if re.search(r"\b(open|release|let go|drop)\b", text) and not re.search(
            r"\bopen (the )?(box|door|lid|drawer)\b", text
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
        if re.search(r"\b(wait|hold on|pause|stay)\b", text):
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

    def _match_relative_move(self, text: str) -> Intent | None:
        if not re.search(r"\b(move|go|shift|nudge|step)\b", text):
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
        if not re.search(r"\b(place|put|set|drop off|position)\b", text):
            return None

        params: dict[str, Any] = {}
        # Longest relation phrases first so "on top of" wins over "on".
        for phrase in sorted(_RELATION_WORDS, key=len, reverse=True):
            match = re.search(rf"\b{phrase}\b\s+(.+)$", text)
            if match:
                params["relation"] = _RELATION_WORDS[phrase]
                params["target"] = self._clean_target(match.group(1))
                break
        return Intent("place", params, 0.95)

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
- place:        {{"skill": "place", "params": {{"relation": "in|on|next_to", "target": "<object_name>"}}, "confidence": <0-1>}}
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
- Always include the correct params as shown above — never leave params empty for pick or place.
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
            print(f"  [QWEN INFERENCE RAW] {raw.strip()}")
            intent = self._parse_response(raw)
            print(f"  [QWEN PARSED INTENT] skill='{intent.skill}', params={intent.params}, confidence={intent.confidence}")
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

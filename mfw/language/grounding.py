"""Language grounding: a spoken referent -> exactly one perceived object.

Pure stdlib + NumPy. This module must never import Isaac Sim.

Why this module exists
----------------------
``SceneGraph.by_label`` only understands "<colour> <class>". Everything else a
person actually says failed with ``ObjectNotFound`` while the robot was looking
straight at the object. Measured on the default scene (red block, blue can,
green box) on 2026-09-24 (logs/e2e/run_a_default.txt):

* "pick the red object"   -> ObjectNotFound   (generic noun never matched)
* "pick the large object" -> ObjectNotFound   (no size reasoning at all)
* "pick the object"       -> ObjectNotFound   with THREE objects in view --
  the one case that must become a question, not a refusal
* "blue cube" / "cylinder" / "object on the left" / "block next to the can"
  -> ObjectNotFound (audit probes; the scene graph computed the relations,
  nothing consumed them).

What it resolves
----------------
* **Generic nouns** (object, thing, one, item, stuff, piece) are a class
  wildcard, so "the red one" is a colour filter over everything.
* **Colour** from ``obj.attributes["color"]`` (hue-band names produced by
  ``mfw.vision.geometry.dominant_color_name``). An exact colour beats an
  adjacent hue (a lit red block can be named "orange"), which beats an object
  whose colour is unknown (the hardware lane reports ``""``). A *different*
  known colour never matches.
* **Class synonyms** (cube/cuboid/brick -> block & brick, cylinder/tin -> can,
  carton -> box, ...). The exact class always wins over a synonym, so a scene
  holding both a "block" and a "brick" does not merge them when the operator
  names one.
* **Size** adjectives and superlatives rank by bbox volume (height for
  tall/short) and require a clear 20 % margin; closer than that is a question.
* **Position**: "on the left/right" (+Y is the robot's left), leftmost,
  rightmost, middle, front/back, nearest/closest/farthest (from ``robot_xy``),
  and "<X> next to / left of / right of / on / in / under / in front of /
  behind <Y>" via ``scene.relations`` (recomputed from geometry when absent).
* Qualifiers combine: "the small red block on the left".

What it refuses
---------------
Nothing matches -> :class:`ObjectNotFound`, whose message says which
qualifier failed and on what ("cannot find 'blue cube': the only block is the
red block") and then lists *everything* visible ("currently visible: red
block, blue can, green box") -- never only what a filter left. Several equally
good matches -> :class:`AmbiguousReference` carrying distinguishing
descriptions and track ids, which the planner turns into a question. It never
picks one of several at random: a wrong referent means confidently moving the
wrong object.

Pronouns ("it", "that", "this one") still go through
``memory.resolve_reference`` first -- memory owns the referent chain.

The small dialogue helpers at the bottom (the clarification question, matching
the operator's answer against the options, rewriting the command with the
chosen object) live here because answering "which one?" is grounding too: the
answer "the red one" is resolved with the same grammar, restricted to the
options that were offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mfw.core.errors import AmbiguousReference, ObjectNotFound
from mfw.core.types import ObjectHypothesis, SceneGraph
from mfw.utils.logging import get_logger

__all__ = [
    "resolve_reference",
    "describe",
    "describe_candidates",
    "describe_scene",
    "parse_referent",
    "ReferentQuery",
    "clarification_question",
    "interpret_clarification",
    "ClarificationChoice",
    "substitute_referent",
    "is_pronoun",
    "SIZE_MARGIN",
    "POSITION_TOLERANCE_M",
    "NEXT_TO_DISTANCE_M",
]

_log = get_logger("language.grounding")

#: A size superlative needs the winner at least this much larger (by volume)
#: than the runner-up. "The large object" between a 125 cm^3 block and a
#: 130 cm^3 can is not a meaningful distinction, so it becomes a question.
SIZE_MARGIN = 1.20

#: Two candidates whose left/right, front/back or distance differ by less than
#: this are "level" and the superlative cannot separate them. 3 cm is under
#: the width of every graspable object in the configs, so a person looking at
#: two objects this close would also hesitate to call one "the left one".
POSITION_TOLERANCE_M = 0.03

#: Matches ``mfw.vision.scene_graph.build_scene_graph``'s default, used only when
#: a scene graph arrives without relations (synthetic scenes, some lanes).
NEXT_TO_DISTANCE_M = 0.15

_MAX_RELATION_DEPTH = 3

# ----------------------------------------------------------------------
# vocabulary
# ----------------------------------------------------------------------

_PRONOUN_PHRASES = frozenset(
    {"it", "that", "this", "them", "those", "these", "that one", "this one",
     "the same one", "same one", "the same"}
)

GENERIC_NOUNS = frozenset(
    {"object", "objects", "thing", "things", "one", "ones", "item", "items",
     "stuff", "piece", "pieces", "something", "anything"}
)

#: Canonical colour names are exactly those ``dominant_color_name`` emits.
_COLOUR_SYNONYMS: dict[str, str] = {
    "red": "red", "crimson": "red", "scarlet": "red", "maroon": "red",
    "orange": "orange",
    "yellow": "yellow", "gold": "yellow", "golden": "yellow",
    "green": "green", "lime": "green", "olive": "green",
    "cyan": "cyan", "teal": "cyan", "turquoise": "cyan", "aqua": "cyan",
    "blue": "blue", "navy": "blue",
    "purple": "purple", "violet": "purple", "lilac": "purple",
    "pink": "pink", "magenta": "pink",
    "white": "white", "black": "black",
    "grey": "grey", "gray": "grey", "silver": "grey",
    "brown": "brown", "tan": "brown",
}

#: Hue-band neighbours: a lit or shadowed surface can land one band over.
_COLOUR_NEIGHBOURS: dict[str, frozenset[str]] = {
    "red": frozenset({"orange", "pink"}),
    "orange": frozenset({"red", "yellow", "brown"}),
    "yellow": frozenset({"orange"}),
    "green": frozenset({"cyan"}),
    "cyan": frozenset({"green", "blue"}),
    "blue": frozenset({"cyan", "purple"}),
    "purple": frozenset({"blue", "pink"}),
    "pink": frozenset({"purple", "red"}),
    "white": frozenset({"grey"}),
    "grey": frozenset({"white", "black"}),
    "black": frozenset({"grey"}),
    "brown": frozenset({"orange"}),
}

#: Shape/class synonym groups. A word maps to every group it belongs to; an
#: object matches by synonym when its label is in one of those groups. Labels
#: come from configs/default.yaml (block, can, box), configs/benchmark.yaml via
#: configs/assets.yaml (can, brick, box, banana, marker, bowl, bin, mug, ...)
#: and configs/hardware.yaml (marker, banana, box, cube, bowl, bin).
_CLASS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"block", "cube", "cuboid", "brick"}),
    frozenset({"can", "tin", "cylinder", "canister"}),
    frozenset({"box", "carton", "package", "packet"}),
    frozenset({"bowl", "dish"}),
    frozenset({"bin", "basket", "tray"}),
    frozenset({"marker", "pen", "felt tip", "sharpie"}),
    frozenset({"mug", "cup"}),
    frozenset({"bottle", "flask"}),
    frozenset({"ball", "sphere"}),
    frozenset({"banana"}),
)

#: Words that name a container family without naming one class. A box is not
#: in the family: in these scenes a box is an object to pick (pudding box, the
#: default scene's green box), and "the container" must not make it ambiguous.
_CONTAINER_WORDS = frozenset({"container", "receptacle"})
_CONTAINER_CLASSES = frozenset({"bowl", "bin", "basket", "tray"})

#: Support surfaces are not perceived objects. "The can on the table" means a
#: can that is not stacked on anything, not a relation to an object "table".
_SURFACE_WORDS = frozenset(
    {"table", "desk", "floor", "ground", "surface", "counter", "bench", "tabletop", "workbench"}
)

_LARGE = frozenset({"large", "larger", "largest", "big", "bigger", "biggest", "huge",
                    "giant", "massive", "bulky"})
_SMALL = frozenset({"small", "smaller", "smallest", "little", "littlest", "tiny",
                    "tinier", "tiniest", "mini"})
_TALL = frozenset({"tall", "taller", "tallest"})
_SHORT = frozenset({"short", "shorter", "shortest", "flat", "flatter", "flattest"})
_LEFT = frozenset({"left", "leftmost", "leftward"})
_RIGHT = frozenset({"right", "rightmost", "rightward"})
_MIDDLE = frozenset({"middle", "center", "centre", "central"})
_FRONT = frozenset({"front", "frontmost"})
_BACK = frozenset({"back", "rear", "backmost", "rearmost"})
_NEAR = frozenset({"near", "nearest", "nearer", "closest", "closer", "close"})
_FAR = frozenset({"far", "farthest", "furthest", "farther", "further"})

_FILLER = frozenset(
    {"the", "a", "an", "please", "now", "there", "here", "over", "just", "some",
     "kind", "sort", "of", "which", "is", "that", "this", "these", "those", "my",
     "your", "our", "its", "side", "hand", "most", "very", "really", "dark",
     "light", "bright", "pale", "color", "colour", "coloured", "colored",
     "shaped", "looking", "sitting", "standing", "lying", "placed", "located",
     "with", "and", "to", "at", "on", "in"}
)

#: "Nearest to <these>" means nearest to the robot.
_ROBOT_WORDS = frozenset(
    {"you", "me", "us", "yourself", "robot", "the robot", "arm", "the arm",
     "your arm", "base", "the base", "gripper", "the gripper", "your gripper",
     "hand", "your hand", "the hand", "camera", "the camera"}
)

#: Relation keywords -> canonical relation. Ordered longest first so a regex
#: alternation tries "on top of" before "on" at the same position.
_RELATION_KEYWORDS: dict[str, str] = {
    "to the left side of": "left_of", "on the left side of": "left_of",
    "to the left of": "left_of", "on the left of": "left_of", "at the left of": "left_of",
    "left side of": "left_of", "left of": "left_of",
    "to the right side of": "right_of", "on the right side of": "right_of",
    "to the right of": "right_of", "on the right of": "right_of", "at the right of": "right_of",
    "right side of": "right_of", "right of": "right_of",
    "in front of": "in_front_of", "in back of": "behind", "behind": "behind",
    "on top of": "on", "atop": "on", "upon": "on", "on": "on",
    "inside of": "in", "inside": "in", "within": "in", "into": "in", "in": "in",
    "underneath": "under", "under": "under", "below": "under", "beneath": "under",
    "next to": "next_to", "adjacent to": "next_to", "close to": "next_to",
    "near to": "next_to", "beside": "next_to", "besides": "next_to", "near": "next_to",
    "alongside": "next_to", "by": "next_to",
}
_RELATION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(_RELATION_KEYWORDS, key=len, reverse=True))
    + r")\b"
)

_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}
_NUMBER_WORDS: dict[str, int] = {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}

_CANCEL_WORDS = frozenset(
    {"cancel", "abort", "stop", "none", "neither", "nothing", "nevermind", "quit", "no", "nope"}
)
_CANCEL_PHRASES = ("never mind", "forget it", "forget about it", "no one", "not any")


# ----------------------------------------------------------------------
# public: descriptions
# ----------------------------------------------------------------------


def _label_of(obj: ObjectHypothesis) -> str:
    return " ".join(str(obj.label or "").replace("_", " ").strip().lower().split())


def _colour_of(obj: ObjectHypothesis) -> str:
    raw = str(obj.attributes.get("color", "") or "").strip().lower()
    return _COLOUR_SYNONYMS.get(raw, raw)


def describe(obj: ObjectHypothesis) -> str:
    """"red block": colour + class, the class alone when colour is unknown."""
    label = _label_of(obj) or "object"
    colour = _colour_of(obj)
    return f"{colour} {label}" if colour else label


def describe_scene(scene: SceneGraph | None) -> str:
    """What is visible, for an ObjectNotFound message: "red block, blue can"."""
    if scene is None or not scene.objects:
        return "nothing"
    counts: dict[str, int] = {}
    for obj in scene.objects.values():
        name = describe(obj)
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(name if n == 1 else f"{n} x {name}" for name, n in counts.items())


def describe_candidates(candidates: Sequence[ObjectHypothesis]) -> list[str]:
    """Descriptions that tell the candidates apart, index-aligned with the input.

    Distinct objects get their plain description ("red block", "blue can").
    Two or three that share one get a position suffix read off the robot's
    left-right axis ("red block on the left" / "... in the middle" / "... on
    the right"), which :func:`resolve_reference` itself parses back to the same
    object. Larger or level groups fall back to the track id, which is always
    exact.
    """
    base = [describe(obj) for obj in candidates]
    out = list(base)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(base):
        groups.setdefault(name, []).append(index)
    for name, indices in groups.items():
        if len(indices) == 1:
            continue
        ordered = sorted(indices, key=lambda i: -float(candidates[i].pose.position[1]))
        ys = [float(candidates[i].pose.position[1]) for i in ordered]
        separated = all(ys[k] - ys[k + 1] > POSITION_TOLERANCE_M for k in range(len(ys) - 1))
        if separated and len(ordered) in (2, 3):
            suffixes = ("on the left", "on the right") if len(ordered) == 2 else (
                "on the left", "in the middle", "on the right")
            for i, suffix in zip(ordered, suffixes):
                out[i] = f"{name} {suffix}"
        else:
            for i in indices:
                out[i] = f"{name} {candidates[i].track_id}"
    return out


def _join_or(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"


def clarification_question(options: Sequence[str]) -> str:
    """"Which one do you mean: the red block, the blue can or the green box?\""""
    if not options:
        return "Which one do you mean?"
    return f"Which one do you mean: {_join_or([f'the {o}' for o in options])}?"


# ----------------------------------------------------------------------
# parsing a referent phrase
# ----------------------------------------------------------------------


def _normalise(text: str) -> str:
    text = str(text).lower().replace("-", " ")
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s.']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(left|right|front|back|rear) most\b", r"\1most", text)
    text = re.sub(r"\bfar (left|right)\b", r"\1most", text)
    return text


def is_pronoun(phrase: str) -> bool:
    """A referent memory should resolve ("it", "that one"), not the scene."""
    return _normalise(phrase) in _PRONOUN_PHRASES


@dataclass(frozen=True)
class ReferentQuery:
    """A referent phrase broken into what each word asks for.

    ``nouns`` keeps every word that was not recognised as a qualifier, in
    order; which of them are classes and which are unknown modifiers ("soup"
    in "soup can") depends on the scene's labels, so it is decided at match
    time, not here.
    """

    text: str
    nouns: tuple[str, ...] = ()
    colours: tuple[str, ...] = ()
    generic: bool = False
    selectors: tuple[str, ...] = ()
    relation: str | None = None
    relation_ref: str | None = None
    ref_selector: str | None = None
    """"nearest"/"farthest" measured from ``ref_phrase`` (another object) rather
    than from the robot."""
    ref_phrase: str | None = None


_SIDE_PHRASE = re.compile(
    r"\b(?:on|to|at|towards|toward|in) (?:the |your |my |our )?(left|right)"
    r"(?!(?:\s+hand)?(?:\s+side)?\s+of\b)(?: hand)?(?: side)?\b"
)
_MIDDLE_PHRASE = re.compile(r"\b(?:in|at) the (middle|center|centre)\b(?!\s+of\b)")
_FRONT_PHRASE = re.compile(r"\b(?:in|at) (?:the )?front\b(?!\s+of\b)")
_BACK_PHRASE = re.compile(r"\b(?:in|at) the (?:back|rear)\b(?!\s+of\b)")
_REF_SUPERLATIVE = re.compile(
    r"\b(?P<sup>nearest|closest|farthest|furthest)(?: one| ones)?\s+(?:(?:to|from)\s+)?(?P<ref>.+)$"
)
#: "the closest block to the can": the superlative BEFORE the head noun.
_REF_SUPERLATIVE_PREFIX = re.compile(
    r"^(?P<pre>.*?)\b(?P<sup>nearest|closest|farthest|furthest) (?P<head>.+?) (?:to|from) (?P<ref>.+)$"
)

#: A relation whose reference is the robot or the operator is a selector:
#: "the block near you" is the nearest block.
_ROBOT_RELATION_SELECTOR = {
    "next_to": "nearest", "in_front_of": "nearest", "behind": "farthest",
    "left_of": "left", "right_of": "right",
}


def _meaningful(words: Iterable[str]) -> list[str]:
    return [w for w in words if w not in _FILLER]


def parse_referent(phrase: str) -> ReferentQuery:
    """Split a referent phrase into class words, colours, selectors and a relation."""
    text = _normalise(phrase)
    selectors: list[str] = []
    ref_selector: str | None = None
    ref_phrase: str | None = None

    # "the block closest to the can" / "the block nearest you". Only when the
    # superlative follows a head ("the nearest block" is an adjective).
    match = _REF_SUPERLATIVE.search(text)
    prefix = _REF_SUPERLATIVE_PREFIX.match(text)
    sup_word, ref = None, None
    if match and _meaningful(text[: match.start()].split()):
        sup_word, ref = match.group("sup"), match.group("ref").strip()
        text = text[: match.start()].strip()
    elif prefix and _meaningful(prefix.group("head").split()):
        sup_word, ref = prefix.group("sup"), prefix.group("ref").strip()
        text = f"{prefix.group('pre')} {prefix.group('head')}".strip()
    if sup_word is not None and ref is not None:
        sup = "nearest" if sup_word in ("nearest", "closest") else "farthest"
        if ref in _ROBOT_WORDS or _strip_articles(ref) in _ROBOT_WORDS:
            selectors.append(sup)
        else:
            ref_selector, ref_phrase = sup, ref

    # Position phrases that look like relations but are not: "on the left" is
    # a side, "on the box" is a relation.
    text = _SIDE_PHRASE.sub(lambda m: f" {m.group(1)} ", text)
    text = _MIDDLE_PHRASE.sub(" middle ", text)
    text = _FRONT_PHRASE.sub(" front ", text)
    text = _BACK_PHRASE.sub(" back ", text)
    text = re.sub(r"\s+", " ", text).strip()

    relation: str | None = None
    relation_ref: str | None = None
    head = text
    for rel_match in _RELATION_RE.finditer(text):
        before = text[: rel_match.start()].split()
        after = text[rel_match.end():].strip()
        if not _meaningful(before) or not _meaningful(after.split()):
            continue
        # "the near block": "near" used as an adjective, with the head after it.
        relation = _RELATION_KEYWORDS[rel_match.group(0)]
        relation_ref = after
        head = text[: rel_match.start()].strip()
        if after in _ROBOT_WORDS or _strip_articles(after) in _ROBOT_WORDS:
            selectors.append(_ROBOT_RELATION_SELECTOR.get(relation, "nearest"))
            relation, relation_ref = None, None
        break

    nouns: list[str] = []
    colours: list[str] = []
    generic = False
    for word in head.split():
        if word in _COLOUR_SYNONYMS:
            colours.append(_COLOUR_SYNONYMS[word])
        elif word in _LARGE:
            selectors.append("largest")
        elif word in _SMALL:
            selectors.append("smallest")
        elif word in _TALL:
            selectors.append("tallest")
        elif word in _SHORT:
            selectors.append("shortest")
        elif word in _LEFT:
            selectors.append("left")
        elif word in _RIGHT:
            selectors.append("right")
        elif word in _MIDDLE:
            selectors.append("middle")
        elif word in _FRONT:
            selectors.append("front")
        elif word in _BACK:
            selectors.append("back")
        elif word in _NEAR:
            selectors.append("nearest")
        elif word in _FAR:
            selectors.append("farthest")
        elif word in GENERIC_NOUNS:
            generic = True
        elif word in _FILLER or re.fullmatch(r"[\d.]+", word):
            continue
        else:
            nouns.append(word)

    # A selector stated twice ("the left one on the left") is one selector.
    unique_selectors = tuple(dict.fromkeys(selectors))
    return ReferentQuery(
        text=_normalise(phrase),
        nouns=tuple(nouns),
        colours=tuple(dict.fromkeys(colours)),
        generic=generic,
        selectors=unique_selectors,
        relation=relation,
        relation_ref=relation_ref,
        ref_selector=ref_selector,
        ref_phrase=ref_phrase,
    )


def _strip_articles(text: str) -> str:
    for article in ("the ", "a ", "an ", "your ", "my "):
        if text.startswith(article):
            return text[len(article):]
    return text


# ----------------------------------------------------------------------
# matching against a scene
# ----------------------------------------------------------------------


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("es") and word[:-2] in _ALL_CLASS_WORDS:
        return word[:-2]
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_ALL_CLASS_WORDS = frozenset(w for group in _CLASS_GROUPS for w in group) | _CONTAINER_WORDS


def _groups_for(word: str) -> list[frozenset[str]]:
    return [group for group in _CLASS_GROUPS if word in group]


@dataclass(frozen=True)
class _ClassMatch:
    head: str | None
    unknown: tuple[str, ...]


def _find_class(query: ReferentQuery, scene: SceneGraph) -> _ClassMatch:
    """The head class word, and the words nothing in the vocabulary explains."""
    labels = {_label_of(o) for o in scene.objects.values()}
    label_words = {w for label in labels for w in label.split()}
    head: str | None = None
    partial: str | None = None
    unknown: list[str] = []
    # Multi-word labels ("picture frame") matched as a whole first.
    joined = " ".join(query.nouns)
    for label in sorted(labels, key=len, reverse=True):
        if " " in label and re.search(rf"\b{re.escape(label)}s?\b", joined):
            head = label
            break
    for word in query.nouns:
        singular = _singular(word)
        if head is not None and (word in head.split() or singular in head.split()):
            continue
        if word in labels or singular in labels or word in _ALL_CLASS_WORDS or singular in _ALL_CLASS_WORDS:
            head = word if (word in labels or word in _ALL_CLASS_WORDS) else singular
        elif word in label_words or singular in label_words:
            # Part of some label ("soup" of "soup can"): a modifier when a class
            # word is also present, the class itself when it is the only noun.
            partial = partial or (word if word in label_words else singular)
        else:
            unknown.append(word)
    return _ClassMatch(head=head or partial, unknown=tuple(unknown))


def _class_filter(
    head: str, candidates: list[ObjectHypothesis]
) -> list[ObjectHypothesis]:
    """Exact class first, synonym only when no exact match exists."""

    def label_forms(obj: ObjectHypothesis) -> set[str]:
        label = _label_of(obj)
        words = label.split()
        return {label, words[-1]} if words else {label}

    exact = [o for o in candidates if head in label_forms(o)]
    if exact:
        return exact
    if head in _CONTAINER_WORDS:
        return [o for o in candidates if label_forms(o) & _CONTAINER_CLASSES]
    synonyms = set().union(*_groups_for(head)) if _groups_for(head) else set()
    by_synonym = [o for o in candidates if label_forms(o) & synonyms]
    if by_synonym:
        return by_synonym
    return [o for o in candidates if head in _label_of(o).split()]


def _colour_filter(colour: str, candidates: list[ObjectHypothesis]) -> list[ObjectHypothesis]:
    exact = [o for o in candidates if _colour_of(o) == colour]
    if exact:
        return exact
    neighbours = [o for o in candidates if _colour_of(o) in _COLOUR_NEIGHBOURS.get(colour, ())]
    if neighbours:
        _log.debug("no %s object; using adjacent hue(s) %s", colour,
                   sorted({_colour_of(o) for o in neighbours}))
        return neighbours
    return [o for o in candidates if not _colour_of(o)]


def _xy(obj: ObjectHypothesis) -> np.ndarray:
    return np.asarray(obj.bbox.center.position[:2], dtype=np.float64)


def _relations(scene: SceneGraph) -> list[tuple[str, str, str]]:
    if scene.relations:
        return list(scene.relations)
    # Imported lazily: pure NumPy, but vision is not a hard dependency of language.
    from mfw.vision.scene_graph import compute_relations

    return compute_relations(scene.objects, next_to_distance=NEXT_TO_DISTANCE_M)


def _satisfies(
    relation: str,
    subject: ObjectHypothesis,
    ref: ObjectHypothesis,
    triples: set[tuple[str, str, str]],
    robot_xy: np.ndarray,
) -> bool:
    s, r = subject.track_id, ref.track_id
    if relation == "next_to":
        return (s, "next_to", r) in triples
    if relation == "left_of":
        return (s, "left_of", r) in triples
    if relation == "right_of":
        return (s, "right_of", r) in triples
    if relation == "on":
        return (s, "on_top_of", r) in triples
    if relation == "in":
        return (s, "inside", r) in triples
    if relation == "under":
        return (r, "on_top_of", s) in triples
    distance_s = float(np.linalg.norm(_xy(subject) - robot_xy))
    distance_r = float(np.linalg.norm(_xy(ref) - robot_xy))
    if relation == "in_front_of":
        return distance_s < distance_r - POSITION_TOLERANCE_M
    if relation == "behind":
        return distance_s > distance_r + POSITION_TOLERANCE_M
    return False


def _keep_tier(
    candidates: list[ObjectHypothesis], score: Any, *, higher_is_better: bool,
    tolerance: float = POSITION_TOLERANCE_M, ratio: float | None = None,
) -> list[ObjectHypothesis]:
    """Keep the candidates not clearly worse than the best by ``score``.

    ``ratio`` compares multiplicatively (sizes), ``tolerance`` additively
    (positions). One survivor means the selector separated them; several
    means it could not, and the caller decides whether to ask.
    """
    if len(candidates) <= 1:
        return candidates
    values = {o.track_id: float(score(o)) for o in candidates}
    best = max(values.values()) if higher_is_better else min(values.values())
    kept = []
    for obj in candidates:
        value = values[obj.track_id]
        if value == best:
            close = True
        elif ratio is not None:
            if higher_is_better:
                close = value * ratio > best
            else:
                close = value < best * ratio
        else:
            close = abs(value - best) < tolerance
        if close:
            kept.append(obj)
    return kept


def _apply_selector(
    selector: str, candidates: list[ObjectHypothesis], robot_xy: np.ndarray
) -> list[ObjectHypothesis]:
    if selector == "largest":
        return _keep_tier(candidates, lambda o: o.bbox.volume, higher_is_better=True, ratio=SIZE_MARGIN)
    if selector == "smallest":
        return _keep_tier(candidates, lambda o: o.bbox.volume, higher_is_better=False, ratio=SIZE_MARGIN)
    if selector == "tallest":
        return _keep_tier(candidates, lambda o: o.bbox.extents[2], higher_is_better=True, ratio=SIZE_MARGIN)
    if selector == "shortest":
        return _keep_tier(candidates, lambda o: o.bbox.extents[2], higher_is_better=False, ratio=SIZE_MARGIN)
    if selector == "left":
        return _keep_tier(candidates, lambda o: _xy(o)[1], higher_is_better=True)
    if selector == "right":
        return _keep_tier(candidates, lambda o: _xy(o)[1], higher_is_better=False)
    distance = lambda o: float(np.linalg.norm(_xy(o) - robot_xy))  # noqa: E731
    if selector in ("front", "nearest"):
        return _keep_tier(candidates, distance, higher_is_better=False)
    if selector in ("back", "farthest"):
        return _keep_tier(candidates, distance, higher_is_better=True)
    if selector == "middle":
        if len(candidates) < 3:
            return candidates
        ordered = sorted(candidates, key=lambda o: -_xy(o)[1])
        if len(ordered) % 2 == 0:
            mid = len(ordered) // 2
            return ordered[mid - 1: mid + 1]
        middle = ordered[len(ordered) // 2]
        neighbours = ordered[len(ordered) // 2 - 1: len(ordered) // 2 + 2]
        ys = [_xy(o)[1] for o in neighbours]
        if ys[0] - ys[1] < POSITION_TOLERANCE_M or ys[1] - ys[2] < POSITION_TOLERANCE_M:
            return neighbours
        return [middle]
    return candidates


_SELECTOR_ORDER = ("largest", "smallest", "tallest", "shortest", "left", "right",
                   "middle", "front", "back", "nearest", "farthest")


def _not_found(message: str, phrase: str, scene: SceneGraph) -> ObjectNotFound:
    """The refusal. Always lists *everything* in the scene, never a filter's survivors."""
    visible = describe_scene(scene)
    return ObjectNotFound(
        f"{message}; currently visible: {visible}",
        phrase=phrase,
        visible=tuple(describe(o) for o in scene.objects.values()),
    )


# Why a qualifier eliminated every candidate. Measured (logs/e2e/
# run_c_default_phase8.txt, 2026-09-24): "pick the blue cube" answered only
# "cannot find 'blue cube'; currently visible: red block", which reads as if
# the red block were the only thing on the table. The reason now names what
# the class word did match and why it was rejected: "the only block is the red
# block", before the full list of what is in view.


def _article(noun: str) -> str:
    return "an" if noun[:1] in "aeiou" else "a"


def _plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return f"{noun}es"
    return f"{noun}s"


def _join_and(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _the(objects: Sequence[ObjectHypothesis]) -> str:
    """"the red block" / "the red block and the green block" (told apart when alike)."""
    return _join_and([f"the {d}" for d in describe_candidates(list(objects))])


def _class_noun(head: str, matched: Sequence[ObjectHypothesis]) -> str:
    """The class the matched objects share ("block" for "cube"), else the spoken word."""
    labels = {_label_of(o) for o in matched}
    return labels.pop() if len(labels) == 1 and "" not in labels else head


def _why_no_class(head: str, scene: SceneGraph, exclude: set[str]) -> str:
    """Nothing of class ``head`` is left: none in view, or only the excluded one."""
    excluded = [o for o in scene.objects.values() if o.track_id in exclude]
    matched = _class_filter(head, excluded) if excluded else []
    if matched:
        noun = _class_noun(head, matched)
        return f"the only {noun} in view is {_the(matched)}, the object being moved"
    return f"there is no {head} in view"


def _why_no_colour(
    colour: str, before: Sequence[ObjectHypothesis], head: str | None
) -> str:
    """The colour filter emptied ``before``: say what the class word *did* match."""
    if head is None:
        return f"nothing in view is {colour}"
    noun = _class_noun(head, before)
    if len(before) == 1:
        return f"the only {noun} is {_the(before)}"
    return f"the {_plural(noun)} are {_the(before)}, none of them {colour}"


_RELATION_WORDS = {
    "next_to": "next to", "left_of": "to the left of", "right_of": "to the right of",
    "on": "on", "in": "in", "under": "under", "in_front_of": "in front of",
    "behind": "behind",
}


def _why_no_relation(
    relation: str, candidates: Sequence[ObjectHypothesis], refs: Sequence[ObjectHypothesis]
) -> str:
    words = _RELATION_WORDS.get(relation, relation.replace("_", " "))
    verb = "is" if len(candidates) == 1 else "are"
    ref_names = _join_or([f"the {describe(r)}" for r in refs]) or "that"
    return f"{_the(candidates)} {verb} not {words} {ref_names}"


def _ambiguous(phrase: str, candidates: list[ObjectHypothesis]) -> AmbiguousReference:
    ordered = sorted(candidates, key=lambda o: -_xy(o)[1])  # left to right, as seen
    descriptions = describe_candidates(ordered)
    shown = phrase or "that"
    return AmbiguousReference(
        f"{shown!r} could mean {_join_or([f'the {d}' for d in descriptions])}; which one?",
        phrase=phrase,
        candidates=tuple(descriptions),
        track_ids=tuple(o.track_id for o in ordered),
    )


def _candidates(
    phrase: str,
    scene: SceneGraph,
    memory: Any,
    robot_xy: np.ndarray,
    allowed: set[str] | None,
    exclude: set[str],
    depth: int,
) -> list[ObjectHypothesis]:
    """Every object ``phrase`` could mean after filters and selectors.

    Raises ObjectNotFound when a constraint eliminates everything. Returns a
    list (possibly several) so relation references can use all of them.
    """
    if depth > _MAX_RELATION_DEPTH:
        raise _not_found(f"cannot follow {phrase!r}: too many nested relations", phrase, scene)

    pool = [
        o for o in scene.objects.values()
        if o.track_id not in exclude and (allowed is None or o.track_id in allowed)
    ]
    raw = str(phrase).strip()
    pool_ids = {o.track_id for o in pool}
    if raw in scene.objects and raw in pool_ids:
        return [scene.objects[raw]]
    # A track id inside a phrase ("red block obj_004", an option offered when
    # descriptions could not tell objects apart) is exact. Only id-shaped words
    # count: an id that happens to be a plain word ("box") must not hijack
    # "the can on the box".
    named_ids = [w for w in raw.split() if w in pool_ids and not w.isalpha()]
    if len(named_ids) == 1:
        return [scene.objects[named_ids[0]]]

    text = _normalise(raw)
    if text in _PRONOUN_PHRASES:
        if memory is not None:
            spoken = text if memory.is_pronoun(text) else "it"
            resolved = memory.resolve_reference(spoken)
            if resolved is not None:
                if resolved in exclude:
                    raise _not_found(f"{text!r} is the object being moved", raw, scene)
                if resolved in scene.objects:
                    return [scene.objects[resolved]]
                raise _not_found(
                    f"what {text!r} referred to ({resolved}) is no longer visible", raw, scene
                )
        # Nothing to refer back to: "it" is then any object, which is one
        # object (fine) or a question (never a guess).
        return pool

    query = parse_referent(text)
    class_match = _find_class(query, scene)
    candidates = list(pool)

    if class_match.head is None and class_match.unknown:
        unknown = " ".join(class_match.unknown)
        raise _not_found(
            f"cannot find {raw!r}: nothing in view is {_article(unknown)} {unknown}", raw, scene
        )
    if class_match.unknown:
        _log.debug("ignoring unrecognised modifier(s) %s in %r", class_match.unknown, raw)
    if class_match.head is not None:
        candidates = _class_filter(class_match.head, candidates)
        if not candidates:
            raise _not_found(
                f"cannot find {raw!r}: {_why_no_class(class_match.head, scene, exclude)}",
                raw, scene,
            )

    for colour in query.colours:
        before = candidates
        candidates = _colour_filter(colour, candidates)
        if not candidates:
            raise _not_found(
                f"cannot find {raw!r}: {_why_no_colour(colour, before, class_match.head)}",
                raw, scene,
            )

    tie_break_refs: list[ObjectHypothesis] = []
    surface_ref = (
        query.relation_ref is not None
        and _strip_articles(query.relation_ref) in _SURFACE_WORDS
        and not any(_label_of(o) == _strip_articles(query.relation_ref) for o in scene.objects.values())
    )
    if surface_ref:
        if query.relation == "on":
            stacked = {s for s, predicate, _ in _relations(scene) if predicate in ("on_top_of", "inside")}
            candidates = [o for o in candidates if o.track_id not in stacked] or candidates
    elif query.relation is not None and query.relation_ref is not None:
        refs = _candidates(query.relation_ref, scene, memory, robot_xy, None, set(), depth + 1)
        triples = set(_relations(scene))
        satisfied = [
            o for o in candidates
            if any(r.track_id != o.track_id and _satisfies(query.relation, o, r, triples, robot_xy)
                   for r in refs)
        ]
        if not satisfied:
            raise _not_found(
                f"cannot find {raw!r}: {_why_no_relation(query.relation, candidates, refs)}",
                raw, scene,
            )
        candidates = satisfied
        if query.relation in ("next_to", "left_of", "right_of", "in_front_of", "behind"):
            tie_break_refs = refs

    for selector in sorted(query.selectors, key=_SELECTOR_ORDER.index):
        candidates = _apply_selector(selector, candidates, robot_xy)

    if query.ref_selector is not None and query.ref_phrase is not None:
        refs = _candidates(query.ref_phrase, scene, memory, robot_xy, None, set(), depth + 1)
        candidates = [o for o in candidates if o.track_id not in {r.track_id for r in refs}] or candidates
        candidates = _keep_tier(
            candidates,
            lambda o: min(float(np.linalg.norm(_xy(o) - _xy(r))) for r in refs),
            higher_is_better=query.ref_selector == "farthest",
        )

    # "The block next to the can" with two blocks both within reach of the
    # can: the clearly closer one is what a person means.
    if len(candidates) > 1 and tie_break_refs:
        candidates = _keep_tier(
            candidates,
            lambda o: min(float(np.linalg.norm(_xy(o) - _xy(r))) for r in tie_break_refs
                          if r.track_id != o.track_id),
            higher_is_better=False,
        )
    return candidates


def resolve_reference(
    phrase: str,
    scene: SceneGraph,
    *,
    memory: Any = None,
    robot_xy: tuple[float, float] | Sequence[float] = (0.0, 0.0),
    exclude_ids: Iterable[str] = (),
    allowed_ids: Iterable[str] | None = None,
) -> ObjectHypothesis:
    """Resolve a spoken referent to exactly one object in ``scene``.

    ``memory`` (optional, a :class:`~mfw.memory.working_memory.WorkingMemory`)
    resolves pronouns. ``robot_xy`` is the robot base in the scene's frame,
    the origin for nearest/farthest/front/back. ``exclude_ids`` removes objects
    that cannot be meant (the held object, for a place destination);
    ``allowed_ids`` restricts the choice (answering "which one?" among the
    offered options) while relations may still refer to anything in view.

    Raises :class:`ObjectNotFound` or :class:`AmbiguousReference`; never guesses.
    """
    if phrase is None or not str(phrase).strip():
        raise ObjectNotFound("no object was named", phrase="", visible=())
    xy = np.asarray(tuple(robot_xy)[:2], dtype=np.float64)
    allowed = None if allowed_ids is None else set(allowed_ids)
    candidates = _candidates(str(phrase), scene, memory, xy, allowed, set(exclude_ids), 0)
    if not candidates:
        raise _not_found(f"cannot find {str(phrase).strip()!r}", str(phrase), scene)
    if len(candidates) > 1:
        raise _ambiguous(_normalise(str(phrase)), candidates)
    chosen = candidates[0]
    _log.debug("grounded %r -> %s (%s)", phrase, chosen.track_id, describe(chosen))
    return chosen


# ----------------------------------------------------------------------
# answering "which one do you mean?"
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ClarificationChoice:
    """How an answer to a clarification question was understood.

    ``kind`` is ``"choice"`` (``index`` names the option), ``"cancel"`` or
    ``"unknown"`` (ask again).
    """

    kind: str
    index: int | None = None


def _canonical_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for word in _normalise(text).split():
        if word in _FILLER or word in GENERIC_NOUNS:
            continue
        word = _COLOUR_SYNONYMS.get(word, word)
        singular = _singular(word)
        groups = _groups_for(word) or _groups_for(singular)
        if groups:
            tokens.add("class:" + "|".join(sorted(groups[0])))
        else:
            tokens.add(singular)
    return tokens


def interpret_clarification(
    answer: str,
    options: Sequence[str],
    *,
    track_ids: Sequence[str] = (),
    scene: SceneGraph | None = None,
    robot_xy: tuple[float, float] | Sequence[float] = (0.0, 0.0),
) -> ClarificationChoice:
    """Match an operator's answer to one of the offered options.

    Understands cancellation ("cancel", "never mind"), ordinals ("the first
    one", "2", "the last one"), and descriptions ("the red one", "the can",
    "the one on the left"). With ``scene`` and ``track_ids`` the answer is
    grounded with :func:`resolve_reference` restricted to the options, so every
    qualifier the grammar knows works here too; otherwise it falls back to word
    overlap with the option texts.
    """
    text = _normalise(answer)
    if not text or not options:
        return ClarificationChoice("unknown")

    words = text.split()
    harmless = _CANCEL_WORDS | _FILLER | {"them", "it", "thanks", "thank", "you", "all"}
    if any(re.search(rf"\b{p}\b", text) for p in _CANCEL_PHRASES) or (
        set(words) & _CANCEL_WORDS and set(words) <= harmless
    ):
        return ClarificationChoice("cancel")

    # Exact option text.
    for index, option in enumerate(options):
        if _normalise(option) in (text, _strip_articles(text)):
            return ClarificationChoice("choice", index)

    # Ordinals: the answer must be *only* an ordinal ("the second one").
    content = [w for w in words if w not in _FILLER and w not in ("one", "option", "number")]
    if len(content) == 1:
        word = content[0]
        index: int | None = None
        if word in _ORDINALS:
            index = _ORDINALS[word]
        elif word.isdigit():
            index = int(word) - 1
        elif word in _NUMBER_WORDS and ("number" in words or "option" in words):
            index = _NUMBER_WORDS[word]
        if index is not None:
            if index == -1:
                return ClarificationChoice("choice", len(options) - 1)
            if 0 <= index < len(options):
                return ClarificationChoice("choice", index)
            return ClarificationChoice("unknown")
    if words in (["one"], ["the", "one"]):
        return ClarificationChoice("unknown")

    if scene is not None and track_ids and len(track_ids) == len(options):
        try:
            chosen = resolve_reference(
                text, scene, robot_xy=robot_xy, allowed_ids=[t for t in track_ids if t in scene.objects]
            )
        except (ObjectNotFound, AmbiguousReference):
            chosen = None
        if chosen is not None and chosen.track_id in track_ids:
            return ClarificationChoice("choice", list(track_ids).index(chosen.track_id))

    wanted = _canonical_tokens(text)
    if not wanted:
        return ClarificationChoice("unknown")
    matches = [i for i, option in enumerate(options) if wanted <= _canonical_tokens(option)]
    if len(matches) == 1:
        return ClarificationChoice("choice", matches[0])
    return ClarificationChoice("unknown")


def substitute_referent(clause: str, phrase: str, replacement: str) -> str | None:
    """Replace the LAST occurrence of ``phrase`` in ``clause`` with ``replacement``.

    Used to re-run a command once "which one?" is answered: "pick the object"
    with phrase "object" and replacement "obj_002" becomes "pick obj_002". The
    last occurrence, because a command's referent is at its end ("place the
    can next to the can" names the destination last). A preceding article is
    consumed so the result reads cleanly. Returns ``None`` when the phrase is
    not in the clause (an LLM parser may have paraphrased it).
    """
    if not phrase or not clause:
        return None
    norm_clause = _normalise(clause)
    norm_phrase = _normalise(phrase)
    if not norm_phrase:
        return None
    pattern = re.compile(
        rf"(?:\b(?:the|a|an|that|this)\s+)?\b{re.escape(norm_phrase)}\b"
    )
    matches = list(pattern.finditer(norm_clause))
    if not matches:
        return None
    last = matches[-1]
    return f"{norm_clause[: last.start()]}{replacement}{norm_clause[last.end():]}".strip()

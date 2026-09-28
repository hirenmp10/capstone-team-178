"""The detector service on :5558: Florence-2 (the simulation's model) or the scripted fake.

    # Jetson (the MVP): Florence-2-base ONNX FP16, frames from robot_server on the same host
    python3 scripts/serve_detector.py --backend florence-onnx --jetson 127.0.0.1:5560
    # laptop debug fallback: the same, pointed at the Jetson's robot_server
    py -3.12 scripts/serve_detector.py --backend florence-onnx --jetson 192.168.1.50:5560 --port 5558
    # PyTorch Florence (transformers >= 4.56, florence-community checkpoint)
    py -3.12 scripts/serve_detector.py --backend florence --jetson 192.168.1.50:5560
    # the fake lane
    py -3.12 scripts/serve_detector.py --backend scripted --scene "marker:0.18,0.05 bowl:0.15,-0.12"
    py -3.12 scripts/serve_detector.py --fake --print-homography   # == hardware_fake.yaml

Same wire protocol as ``jetson/detector_service.py`` (server, scripted
backend, frame source, pipeline and voter all live there, mfw-free), so the
orchestrator reaches it by ``hardware.detector_host/port`` alone. This script
must also stay mfw-free and Python 3.10 compatible: it runs on the Jetson.

Frames (model backends). Only from ``robot_server.py``'s ``get_frame``
(``--jetson HOST:PORT``, default the local robot server): the robot server
owns the webcam, a second process opening it is the EBUSY trap. A frame older
than ``--max-frame-age`` (0.5 s) is refused, a request that carries ``jpeg``
is refused, and an object is reported only when 2 of the last 3 distinct
frames saw it (``--vote-window/--vote-required``; normally 2 inferences per
request, 1 while recent frames are still in the vote, 3 when the first two
disagree) -- and, because consecutive frames of a still table give the same
answer, those frames must include both caption orders (the ONNX backend
reverses the label order on alternate frames; ``--fixed-caption-order``
turns both off). Measured over the TCP protocol on 6 exterior sim renders
(2026-09-27): with both orders required, the 8-label hardware.yaml
vocabulary found 17/60 present objects with 4 phantoms in 12 requests
(alternating orders alone, earlier run: 22/60 with 17 phantoms); a
colour-worded 5-label vocabulary found 36/42 (12 phantoms, 6 of them on the
cluttered 10-object render); only the objects on the table: 36/42, 2.

Labels. The laptop sends ``hardware.labels``. Each label has a
:class:`LabelRule` (defaults in :data:`DEFAULT_LABEL_RULES`, overridable with
``--label-config FILE``): the phrase put in the grounding caption, synonyms
Florence may echo back, and sanity limits (minimum score, minimum box side,
maximum share of the image). Grounding draws a box for every phrase it is
given, present or not; those limits, the corner rule and the frame vote are
what keep phantom objects out.

Recorded traps:

* Florence echoes a *phrase*, not the label, and sometimes a shortened one
  ("a soup" for "a soup can", "a blue" for "a blue can"; measured on sim
  renders). :func:`match_label` maps it back by whole words, accepts a
  shortened phrase only when it is the start of exactly one configured
  label, and never maps one phrase to two labels.
* Florence-2 emits no scores. The ONNX backend scores each box from its own
  decoder (``jetson/florence_onnx.py``); the PyTorch backend stamps the fixed
  ``--confidence`` on every box, so for it the per-label ``min_score`` is a
  no-op and only the geometric filters and the vote apply.
* ``microsoft/Florence-2-*`` are remote-code repos broken on transformers >=
  4.50; the default PyTorch checkpoint is the official conversion
  ``florence-community/Florence-2-base`` (native class, transformers >= 4.56).
  The remote-code path is still tried as a fallback.
* Beam search (the checkpoint's ``num_beams: 3``): on 6 exterior sim renders
  (ONNX FP16, 2026-09-27) it returned the same phrases as greedy with boxes
  within 1 px, at 1.0-1.4x the time; greedy stays the default for both
  backends (``--num-beams``).
* torch/transformers/onnxruntime are imported on first use only. Importing
  this script (the tests do) must stay cheap.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jetson.detector_service import (  # noqa: E402 - must follow sys.path setup
    DetectorServer,
    FramePipeline,
    FrameVoter,
    LabelRule,
    RobotFrameSource,
    ScriptedDetector,
    SyntheticPinhole,
    add_common_arguments,
    box_iou,
    build_scripted,
    format_homography,
)
from jetson.florence_onnx import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    GROUNDING_TASK,
    FlorenceOnnx,
    ProviderFallbackError,
)

__all__ = [
    "GROUNDING_TASK",
    "DEFAULT_MODEL",
    "DetectorServer",
    "ScriptedDetector",
    "FlorenceDetector",
    "FlorenceOnnxDetector",
    "LabelRule",
    "DEFAULT_LABEL_RULES",
    "load_label_rules",
    "JetsonFrameSource",
    "build_caption",
    "build_prompt",
    "match_label",
    "grounding_to_objects",
    "sanity_filter",
    "build_parser",
    "build_backend",
    "build_frame_provider",
    "build_server",
    "warm_backend",
    "main",
]

_log = logging.getLogger("serve_detector")

DEFAULT_MODEL = "florence-community/Florence-2-base"
REMOTE_CODE_MODEL = "microsoft/Florence-2-base"
NATIVE_MODEL_HINT = DEFAULT_MODEL  # kept for older callers
DEFAULT_MAX_NEW_TOKENS = 128
_WORD_RE = re.compile(r"[a-z0-9]+")
_ARTICLES = frozenset({"a", "an", "the"})

JetsonFrameSource = RobotFrameSource
"""Old name: frames from robot_server's get_frame (now mfw-free, in detector_service)."""


# ---------------------------------------------------------------------------
# Label rules
# ---------------------------------------------------------------------------


# Recommended MVP vocabulary (fixed overhead C270, 720p, objects <= 35 mm across
# the grip plus the placed destinations). Areas are shares of the whole frame:
# with the camera ~0.6 m above the table a 15 cm bowl is ~8 %; 0.15 / 0.30 leave
# room for a closer mount while rejecting the "whole table" / "whole image"
# phantoms measured on sim renders (40-100 % of the frame).
#
# Wordings measured one at a time on 6 exterior sim renders (2026-09-27, FP16
# ONNX, greedy; found = present objects boxed after the sanity filter,
# phantoms = kept boxes on something else): "pen" found the marker 2/3 with
# 1 phantom where "marker" found 1/3 with 17; colour + noun works ("red
# block" 5/6, "green box" 5/6) where the bare noun does not ("block" 0/6,
# "box" 1/6); "banana", "bowl", "dish" 3/3. "ball" (no ball in any render)
# drew 8 phantoms in 6 images. So the marker is asked for as "a pen", and a
# team's real objects should get colour prompts via --label-config (e.g.
# block -> "red block").
DEFAULT_LABEL_RULES: dict[str, LabelRule] = {
    "marker": LabelRule(prompt="pen", synonyms=("marker pen", "felt tip pen", "whiteboard marker", "sharpie")),
    "banana": LabelRule(synonyms=("yellow banana",)),
    "red block": LabelRule(synonyms=("red cube", "red toy block")),
    "block": LabelRule(synonyms=("wooden block", "toy block", "building block")),
    "cube": LabelRule(synonyms=("toy cube", "small cube", "wooden cube")),
    "ball": LabelRule(synonyms=("small ball", "toy ball", "tennis ball", "sphere")),
    "bowl": LabelRule(synonyms=("small bowl", "dish"), max_area_frac=0.30),
    "box": LabelRule(synonyms=("small box", "cardboard box", "carton"), max_area_frac=0.30),
    "bin": LabelRule(synonyms=("small bin", "container", "tub", "basket"), max_area_frac=0.30),
}
DEFAULT_RULE = LabelRule()


def _norm_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def load_label_rules(path: str | Path | None, base: Mapping[str, LabelRule] | None = None) -> dict[str, LabelRule]:
    """``base`` (default :data:`DEFAULT_LABEL_RULES`) overlaid with a JSON/YAML file.

    File shape: ``{"labels": {"marker": {"prompt": "marker pen", "synonyms":
    ["pen"], "min_score": 0.6, "min_side_px": 4, "max_area_frac": 0.1}}}``
    (the top-level ``labels`` key is optional). A listed field replaces the
    base value; unlisted fields keep it.
    """
    rules = {_norm_label(k): v for k, v in (DEFAULT_LABEL_RULES if base is None else base).items()}
    if path is None:
        return rules
    text = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        import yaml  # noqa: PLC0415

        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    entries = raw.get("labels", raw) if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: expected a mapping of label -> rule")
    allowed = {"prompt", "synonyms", "min_score", "min_side_px", "max_area_frac"}
    for label, spec in entries.items():
        spec = dict(spec or {})
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"{path}: label {label!r} has unknown field(s) {sorted(unknown)}")
        if "synonyms" in spec:
            spec["synonyms"] = tuple(str(s) for s in spec["synonyms"] or ())
        for key in ("min_score", "min_side_px", "max_area_frac"):
            if key in spec:
                spec[key] = float(spec[key])
        key = _norm_label(label)
        rules[key] = replace(rules.get(key, DEFAULT_RULE), **spec)
    return rules


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a model)
# ---------------------------------------------------------------------------


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("es") and word[:-2].endswith(("x", "ch", "sh", "ss")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    """Lower-case whole words, articles dropped, crude singular (``boxes`` -> ``box``)."""
    return [_singular(w) for w in _WORD_RE.findall(str(text).lower()) if w not in _ARTICLES]


def build_caption(labels: Sequence[str], rules: Mapping[str, LabelRule] | None = None) -> str:
    """The grounding caption for a vocabulary: ``"a marker. a bowl."``.

    Each label contributes its rule's ``prompt`` (default the label). Bare
    nouns without an article ground noticeably worse on Florence-2-base.
    """
    phrases: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = _norm_label(raw)
        if not label:
            continue
        rule = (rules or {}).get(label)
        phrase = _norm_label(rule.prompt) if rule is not None and rule.prompt else label
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase if phrase.startswith(("a ", "an ", "the ")) else f"a {phrase}")
    return " ".join(f"{p}." for p in phrases)


def build_prompt(labels: Sequence[str], rules: Mapping[str, LabelRule] | None = None) -> str:
    """The Florence-2 processor prompt: the task token plus :func:`build_caption`."""
    return GROUNDING_TASK + build_caption(labels, rules)


def _match(phrase: str, labels: Sequence[str], rules: Mapping[str, LabelRule] | None) -> tuple[str, str] | None:
    """``(label, "full" | "prefix")`` or ``None``; see :func:`match_label`."""
    words = _words(phrase)
    if not words:
        return None
    word_set = set(words)
    # (words matched, own name) per label: a longer name wins ("cardboard box"
    # over "box"); at equal length a label's own name or prompt beats another
    # label's synonym ("a block" is ``block`` even when ``cube`` lists "block").
    full: dict[str, tuple[int, int]] = {}
    prefix: set[str] = set()
    for raw in labels:
        label = _norm_label(raw)
        if not label:
            continue
        rule = (rules or {}).get(label, DEFAULT_RULE)
        own = {_norm_label(label)} | ({_norm_label(rule.prompt)} if rule.prompt else set())
        for name in rule.names(label):
            name_words = _words(name)
            if not name_words:
                continue
            if set(name_words) <= word_set:
                rank = (len(name_words), 1 if _norm_label(name) in own else 0)
                full[label] = max(full.get(label, (0, 0)), rank)
            elif len(words) < len(name_words) and name_words[: len(words)] == words:
                prefix.add(label)
    if full:
        best = max(full.values())
        winners = [label for label, n in full.items() if n == best]
        if len(winners) == 1:
            return winners[0], "full"
        _log.debug("phrase %r names %s equally; mapped to none", phrase, winners)
        return None
    if len(prefix) == 1:
        return next(iter(prefix)), "prefix"
    if prefix:
        _log.debug("shortened phrase %r could be any of %s; mapped to none", phrase, sorted(prefix))
    return None


def match_label(phrase: str, labels: Sequence[str], rules: Mapping[str, LabelRule] | None = None) -> str | None:
    """Map a grounded phrase back to the one requested label it names, or ``None``.

    * whole-word match against the label and its rule's prompt/synonyms
      (``"a whiteboard marker"`` -> ``marker``; ``"candle"`` never matches
      ``can``); the name with the most words wins, so ``"cardboard box"``
      beats ``"box"``; at equal length a label's own name beats another
      label's synonym (``"a block"`` -> ``block`` although ``cube`` lists
      "block"); a remaining tie between two labels maps to none;
    * otherwise a shortened phrase (``"a soup"``) maps to the label it is the
      start of -- only when exactly one configured label starts that way
      (``"a blue"`` is ``blue can`` unless ``blue block`` is also asked for).
    """
    hit = _match(phrase, labels, rules)
    return hit[0] if hit else None


def grounding_to_objects(
    parsed: Mapping[str, Any],
    labels: Sequence[str],
    confidence: float,
    min_score: float,
    image_size: tuple[int, int],
    rules: Mapping[str, LabelRule] | None = None,
    scores: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Post-processed ``{"bboxes", "labels"}`` -> wire objects (no sanity filter; see :func:`sanity_filter`).

    Boxes are clipped to the image and degenerate ones dropped; a phrase that
    names none of the requested labels is dropped too. ``scores`` (one per
    box) replaces the fixed ``confidence`` when the backend has real ones.
    Each object also carries ``"match"`` (``full``/``prefix``), which
    :func:`sanity_filter` uses and the wire drops.
    """
    if scores is None and float(confidence) < float(min_score):
        return []
    width, height = (int(v) for v in image_size)
    boxes = parsed.get("bboxes") or []
    phrases = parsed.get("labels") or []
    out: list[dict[str, Any]] = []
    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        hit = _match(str(phrase), labels, rules)
        if hit is None:
            _log.debug("dropping grounded phrase %r: names no requested label", phrase)
            continue
        score = float(scores[i]) if scores is not None else float(confidence)
        if score < float(min_score):
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = sorted((max(0.0, x0), min(float(width - 1), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(height - 1), y1)))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        out.append({
            "label": hit[0],
            "confidence": score,
            "bbox_px": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
            "match": hit[1],
            "phrase": str(phrase),
        })
    return out


def sanity_filter(
    objects: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
    rules: Mapping[str, LabelRule] | None = None,
    keep_corner_boxes: bool = False,
    duplicate_iou: float = 0.85,
    edge_px: float = 2.0,
    use_scores: bool = True,
    duplicate_margin: float = 0.05,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Drop boxes no real demo object makes; ``(kept, [(dropped, reason)])``.

    Per label (:class:`LabelRule`): score below ``min_score``, a side
    thinner than ``min_side_px``, more than ``max_area_frac`` of the image.
    Globally: a box in an image *corner* (touching a vertical and a
    horizontal border) -- the fixed overhead camera is mounted to see the
    whole table, so a real object never sits there, while grounding puts
    absent objects in dark corners (measured); ``keep_corner_boxes`` turns it
    off. Last, one box claimed by two labels (IoU >= ``duplicate_iou``;
    Florence gave the bowl's box to "a soup" and to an absent "ball") keeps
    the claim that is stronger by at least ``duplicate_margin`` in score, or
    by a full-word match over a shortened phrase at equal-ish scores; when
    neither is clearly stronger, **neither** is kept -- on the sim renders
    the grounding scores of right and wrong claims were both 0.93-1.00, and
    a wrong label on a real object is worse than a miss. ``use_scores=False`` (a backend whose scores are a stamped
    constant) skips the per-label ``min_score``.
    """
    width, height = float(image_size[0]), float(image_size[1])
    image_area = max(1.0, width * height)
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    for obj in objects:
        obj = dict(obj)
        label = _norm_label(obj.get("label", ""))
        rule = (rules or {}).get(label, DEFAULT_RULE)
        x0, y0, x1, y1 = (float(v) for v in obj["bbox_px"])
        w, h = x1 - x0, y1 - y0
        score = float(obj.get("confidence", 0.0))
        if use_scores and score < rule.min_score:
            dropped.append((obj, f"score {score:.2f} < {rule.min_score:.2f}"))
        elif min(w, h) < rule.min_side_px:
            dropped.append((obj, f"side {min(w, h):.0f} px < {rule.min_side_px:.0f}"))
        elif w * h / image_area > rule.max_area_frac:
            dropped.append((obj, f"covers {100 * w * h / image_area:.0f}% of the image > {100 * rule.max_area_frac:.0f}%"))
        elif not keep_corner_boxes and (x0 <= edge_px or x1 >= width - 1 - edge_px) and (
            y0 <= edge_px or y1 >= height - 1 - edge_px
        ):
            dropped.append((obj, "in an image corner"))
        else:
            kept.append(obj)

    def score_of(o: Mapping[str, Any]) -> float:
        return float(o.get("confidence", 0.0)) if use_scores else 0.0

    def full(o: Mapping[str, Any]) -> bool:
        return o.get("match", "full") == "full"

    order = sorted(range(len(kept)), key=lambda i: (score_of(kept[i]), full(kept[i])), reverse=True)
    losers: dict[int, str] = {}
    for rank, i in enumerate(order):
        if i in losers:
            continue
        for j in order[rank + 1:]:
            if j in losers or box_iou(kept[i]["bbox_px"], kept[j]["bbox_px"]) < duplicate_iou:
                continue
            if kept[i]["label"] == kept[j]["label"]:
                losers[j] = "repeat of a stronger box of the same label"
                continue
            margin = score_of(kept[i]) - score_of(kept[j])
            if margin >= duplicate_margin or (full(kept[i]) and not full(kept[j])):
                losers[j] = f"same box as {kept[i]['label']!r} (stronger claim)"
                continue
            losers[i] = f"same box as {kept[j]['label']!r}, neither claim clearly stronger"
            losers[j] = f"same box as {kept[i]['label']!r}, neither claim clearly stronger"
            break
    final = [o for i, o in enumerate(kept) if i not in losers]
    dropped.extend((kept[i], why) for i, why in losers.items())
    return final, dropped


def _wire(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": o["label"], "confidence": float(o["confidence"]), "bbox_px": list(o["bbox_px"])} for o in objects]


# ---------------------------------------------------------------------------
# Florence-2 ONNX backend (the Jetson default)
# ---------------------------------------------------------------------------


class FlorenceOnnxDetector:
    """Florence-2-base FP16 ONNX (``jetson/florence_onnx.py``) behind the 5558 protocol.

    ``model`` may be injected (tests pass a fake with ``generate``); by
    default a :class:`FlorenceOnnx` is built, which loads nothing until
    :meth:`warm_up` or the first :meth:`detect`.

    ``vary_caption_order`` (default on) reverses the caption's label order
    on every other call and tags each call's boxes with that order
    (``last_view``); :func:`build_server` then makes the frame vote require
    both orders, so a box must survive two different captions: Florence's
    phantom assignments depend on the caption's order more than real ones
    do, while consecutive frames of a still scene give the same answer
    every time (greedy decoding is deterministic).
    """

    name = "florence-onnx"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        precision: str = "fp16",
        device: str = "cuda",
        allow_cpu: bool = False,
        num_beams: int = 1,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        rules: Mapping[str, LabelRule] | None = None,
        keep_corner_boxes: bool = False,
        vary_caption_order: bool = True,
        model: Any = None,
    ) -> None:
        self.model = model if model is not None else FlorenceOnnx(
            model_dir=model_dir, precision=precision, device=device, allow_cpu=allow_cpu,
            num_beams=num_beams, max_new_tokens=max_new_tokens)
        self.rules = dict(DEFAULT_LABEL_RULES if rules is None else rules)
        self.keep_corner_boxes = bool(keep_corner_boxes)
        self.vary_caption_order = bool(vary_caption_order)
        self.calls = 0
        self.last_view: str | None = None
        """Caption order of the last call; the frame vote needs both orders when they alternate."""
        self.last_caption = ""
        self.last_latency_s: float | None = None
        self.last_generation: Any = None
        self.last_dropped: list[tuple[dict[str, Any], str]] = []

    def detect(self, rgb: NDArray[np.uint8] | None, labels: Sequence[str], min_score: float = 0.0) -> list[dict[str, Any]]:
        """Ground every requested label in one RGB frame; sanity-filtered wire objects."""
        if rgb is None:
            raise RuntimeError("FlorenceOnnxDetector needs a frame (the server gets it from robot_server)")
        vocabulary = [_norm_label(l) for l in labels if _norm_label(l)]
        if not vocabulary:
            return []
        reverse = self.vary_caption_order and self.calls % 2 == 1
        if reverse:
            vocabulary = list(reversed(vocabulary))
        self.calls += 1
        self.last_view = "reversed" if reverse else "forward"
        caption = build_caption(vocabulary, self.rules)
        self.last_caption = caption
        started = time.perf_counter()
        generation = self.model.generate(rgb, caption)
        self.last_latency_s = time.perf_counter() - started
        self.last_generation = generation
        size = (int(rgb.shape[1]), int(rgb.shape[0]))
        parsed = {"bboxes": [b.box for b in generation.boxes], "labels": [b.phrase for b in generation.boxes]}
        candidates = grounding_to_objects(parsed, vocabulary, 1.0, 0.0, size, self.rules,
                                          scores=[b.score for b in generation.boxes])
        kept, dropped = sanity_filter(candidates, size, self.rules, self.keep_corner_boxes)
        self.last_dropped = dropped
        for obj, why in dropped:
            _log.debug("dropped %s %s (%r): %s", obj["label"], obj["bbox_px"], obj.get("phrase"), why)
        if getattr(generation, "truncated", False):
            _log.warning("Florence hit max_new_tokens (%s) before EOS: trailing labels may be missing; "
                         "raise --max-new-tokens or ask for fewer labels", getattr(self.model, "max_new_tokens", "?"))
        return _wire([o for o in kept if float(o["confidence"]) >= float(min_score)])

    def warm_up(self, size: tuple[int, int] = (640, 480)) -> float:
        """Load the sessions and run one throwaway inference (checks the providers); seconds taken."""
        started = time.perf_counter()
        load = getattr(self.model, "load", None)
        if load is not None:
            load()
        self.detect(np.zeros((int(size[1]), int(size[0]), 3), dtype=np.uint8), ["object"])
        return time.perf_counter() - started


# ---------------------------------------------------------------------------
# Florence-2 PyTorch backend
# ---------------------------------------------------------------------------


class FlorenceDetector:
    """Florence-2 phrase grounding with PyTorch/transformers behind the 5558 protocol.

    Nothing heavy happens at construction; the first :meth:`detect` loads the
    model (``torch``/``transformers`` imported there) onto CUDA when available,
    fp16 on the GPU and fp32 on the CPU. ``detect`` is serialised by the
    server, so one model instance is enough.
    """

    name = "florence"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        confidence: float = 0.6,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = 1,
        rules: Mapping[str, LabelRule] | None = None,
        keep_corner_boxes: bool = False,
    ) -> None:
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.model_name = str(model)
        self.requested_device = device
        self.confidence = float(confidence)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.rules = dict(DEFAULT_LABEL_RULES if rules is None else rules)
        self.keep_corner_boxes = bool(keep_corner_boxes)
        self.loader: str | None = None
        """``"native"`` or ``"remote_code"`` once loaded."""
        self.device: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._dtype: Any = None
        self.last_latency_s: float | None = None
        self.last_dropped: list[tuple[dict[str, Any], str]] = []

    # -- loading -----------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "torch is not importable in this interpreter; the Florence backend needs "
                "torch + transformers (or use --backend florence-onnx / scripted)"
            ) from exc
        device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        started = time.perf_counter()
        model, processor, loader = self._load(device, dtype)
        self._model, self._processor, self._dtype = model, processor, dtype
        self.device, self.loader = device, loader
        _log.info(
            "Florence-2 ready: %s via %s on %s (%s) in %.1f s",
            self.model_name, loader, device, str(dtype).replace("torch.", ""), time.perf_counter() - started,
        )

    def _load(self, device: str, dtype: Any) -> tuple[Any, Any, str]:
        """Native transformers class first, ``trust_remote_code`` second."""
        import transformers  # noqa: PLC0415

        native_error: Exception | None = None
        native_cls = getattr(transformers, "Florence2ForConditionalGeneration", None)
        if native_cls is not None:
            try:
                model = native_cls.from_pretrained(self.model_name, torch_dtype=dtype).to(device).eval()
                processor = transformers.AutoProcessor.from_pretrained(self.model_name)
                return model, processor, "native"
            except Exception as exc:  # noqa: BLE001 - any load failure means "try the other path"
                native_error = exc
                _log.info("native Florence2 class could not load %s (%s); trying trust_remote_code",
                          self.model_name, f"{type(exc).__name__}: {exc}"[:200])
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, trust_remote_code=True
            ).to(device).eval()
            processor = transformers.AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"could not load {self.model_name!r} with transformers {transformers.__version__}: "
                f"native: {native_error!r}; remote_code: {type(exc).__name__}: {exc}. "
                f"On transformers >= 4.56 use --model {DEFAULT_MODEL}; on older versions "
                f"use --model {REMOTE_CODE_MODEL}."
            ) from exc
        return model, processor, "remote_code"

    # -- inference -----------------------------------------------------------

    def detect(
        self,
        rgb: NDArray[np.uint8] | None,
        labels: Sequence[str],
        min_score: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Ground every requested label in one RGB frame."""
        if rgb is None:
            raise RuntimeError(
                "FlorenceDetector needs a frame: the server gets it from robot_server "
                "(--jetson HOST:PORT)"
            )
        vocabulary = [_norm_label(l) for l in labels if _norm_label(l)]
        if not vocabulary or self.confidence < float(min_score):
            return []
        self._ensure_model()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.fromarray(np.ascontiguousarray(rgb))
        prompt = build_prompt(vocabulary, self.rules)
        started = time.perf_counter()
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, self._dtype)
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=GROUNDING_TASK, image_size=(image.width, image.height)
        )
        self.last_latency_s = time.perf_counter() - started
        grounding = parsed.get(GROUNDING_TASK, parsed) if isinstance(parsed, Mapping) else {}
        size = (image.width, image.height)
        candidates = grounding_to_objects(grounding, vocabulary, self.confidence, min_score, size, self.rules)
        kept, self.last_dropped = sanity_filter(candidates, size, self.rules, self.keep_corner_boxes,
                                                use_scores=False)
        _log.debug("Florence grounded %d box(es) for %s in %.0f ms",
                   len(kept), vocabulary, self.last_latency_s * 1000.0)
        return _wire(kept)

    def warm_up(self, size: tuple[int, int] = (64, 64)) -> float:
        """Load the model and run one throwaway inference; returns the seconds it took.

        Review P3: the model used to load lazily inside the *first* ``detect``,
        behind the server's lock, so the orchestrator's first observe waited
        10-40 s and its 5 s timeout reported a running service as dead. Called
        once from :func:`main` before the socket accepts anything. The dummy
        frame never reaches the frame vote (the pipeline is not involved).
        """
        started = time.perf_counter()
        self._ensure_model()
        self.detect(np.zeros((int(size[1]), int(size[0]), 3), dtype=np.uint8), ["object"], min_score=0.0)
        return time.perf_counter() - started


def warm_backend(backend: Any) -> float | None:
    """Warm a backend that supports it (Florence); ``None`` for one that does not.

    A warm-up failure is logged and swallowed: the server still starts, and
    the first ``detect`` then reports the real error to the orchestrator
    rather than the service refusing to come up at all -- except
    :class:`ProviderFallbackError`, which is re-raised: a detector silently
    running on the CPU must not start at all.
    """
    warm = getattr(backend, "warm_up", None)
    if warm is None:
        return None
    _log.info("warming the %s backend (loading the model before accepting requests)...",
              getattr(backend, "name", "detector"))
    try:
        seconds = float(warm())
    except ProviderFallbackError:
        raise
    except (RuntimeError, OSError, ImportError, ValueError) as exc:
        _log.error("backend warm-up failed (the first detect will report it): %s", exc)
        return None
    _log.info("backend warm in %.1f s", seconds)
    return seconds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--backend", choices=("florence-onnx", "florence", "scripted"), default="florence-onnx",
                        help="florence-onnx (default; the Jetson), florence (PyTorch), scripted (the fake lane)")
    parser.add_argument("--fake", action="store_true", help="alias for --backend scripted")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"(florence) checkpoint; {REMOTE_CODE_MODEL} for transformers < 4.56")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="(florence-onnx) onnx-community/Florence-2-base checkout (tokenizer.json + onnx/)")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16", help="(florence-onnx)")
    parser.add_argument("--device", default=None, choices=("cuda", "cpu"),
                        help="florence-onnx: cuda unless given (CUDA is required unless --allow-cpu); "
                             "florence: cuda if available")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="(florence-onnx) accept a CPU fallback instead of refusing to start")
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="(florence) score stamped on every PyTorch box (that backend emits no scores)")
    parser.add_argument("--num-beams", "--beams", dest="num_beams", type=int, default=1,
                        help="1 = greedy (default, the validated setting); the checkpoint's own default is 3")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--label-config", default=None, metavar="FILE",
                        help="JSON/YAML per-label rules (prompt, synonyms, min_score, min_side_px, "
                             "max_area_frac) over the built-in MVP table")
    parser.add_argument("--fixed-caption-order", action="store_true",
                        help="(florence-onnx) do not reverse the caption on alternate frames")
    parser.add_argument("--keep-corner-boxes", action="store_true",
                        help="do not drop boxes in an image corner (only for a camera that does not see "
                             "the whole table)")
    parser.add_argument("--jetson", default="127.0.0.1:5560", metavar="HOST[:PORT]",
                        help="robot_server.py serving get_frame -- the only frame source of the model "
                             "backends (default: the local robot server)")
    parser.add_argument("--frame-timeout", type=float, default=2.0, help="get_frame reply timeout, s")
    parser.add_argument("--max-frame-age", type=float, default=0.5,
                        help="refuse a frame captured longer ago than this, s")
    parser.add_argument("--vote-window", type=int, default=3, help="frames in the vote")
    parser.add_argument("--vote-required", type=int, default=2, help="frames that must agree")
    parser.add_argument("--vote-iou", type=float, default=0.3, help="IoU for two frames' boxes to agree")
    parser.add_argument("--history-max-age", type=float, default=2.0,
                        help="frames from earlier requests younger than this still vote, s")
    parser.add_argument("--request-budget", type=float, default=4.0,
                        help="stop grabbing extra frames past this, s (the laptop's detect timeout is 5 s)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip loading the model before serving; the first detect then waits for it")
    return parser


def _rules_from(args: argparse.Namespace) -> dict[str, LabelRule]:
    return load_label_rules(getattr(args, "label_config", None))


def build_backend(args: argparse.Namespace) -> Any:
    """The detector the parsed arguments ask for (no model is loaded here)."""
    if args.fake or args.backend == "scripted":
        return build_scripted(args.scene, args.world_from, args.hide_above)
    rules = _rules_from(args)
    if args.backend == "florence":
        return FlorenceDetector(model=args.model, device=args.device, confidence=args.confidence,
                                max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, rules=rules,
                                keep_corner_boxes=args.keep_corner_boxes)
    return FlorenceOnnxDetector(model_dir=args.model_dir, precision=args.precision,
                                device=args.device or "cuda", allow_cpu=args.allow_cpu,
                                num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, rules=rules,
                                keep_corner_boxes=args.keep_corner_boxes,
                                vary_caption_order=not args.fixed_caption_order)


def _is_scripted(args: argparse.Namespace) -> bool:
    return bool(args.fake or args.backend == "scripted")


def build_frame_provider(args: argparse.Namespace) -> tuple[Any, Any]:
    """``(frame source, owner to close)``: robot_server's get_frame for the model backends, nothing for scripted."""
    if _is_scripted(args):
        return None, None
    host, _, port_text = str(args.jetson).partition(":")
    source = RobotFrameSource(host or "127.0.0.1", int(port_text or 5560), timeout_s=args.frame_timeout,
                              max_age_s=args.max_frame_age)
    return source, source


def build_server(args: argparse.Namespace, backend: Any = None) -> tuple[DetectorServer, Any]:
    """Everything ``main`` does short of blocking: ``(server, frame_owner)``.

    The server is *not* bound yet; call ``serve_forever`` or
    ``serve_in_thread(port=0)`` (tests) on it. ``backend`` overrides the one
    the arguments name (tests inject a fake model this way).
    """
    backend = build_backend(args) if backend is None else backend
    source, owner = build_frame_provider(args)
    if source is None:
        return DetectorServer(args.host, args.port, backend), None
    views = 2 if getattr(backend, "vary_caption_order", False) and args.vote_required >= 2 else 1
    pipeline = FramePipeline(
        backend,
        source,
        FrameVoter(args.vote_window, args.vote_required, args.vote_iou, required_views=views),
        max_frame_age_s=args.max_frame_age,
        history_max_age_s=args.history_max_age,
        request_budget_s=args.request_budget,
    )
    return DetectorServer(args.host, args.port, backend, pipeline=pipeline), owner


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    if args.print_homography:
        print(format_homography(SyntheticPinhole().homography_pixel_to_table()))
        return 0

    server, owner = build_server(args)
    backend_name = server.backend_name
    if server.pipeline is not None:
        _log.info("frames from robot_server %s (max age %.2f s, vote %d of %d frames)",
                  args.jetson, args.max_frame_age, args.vote_required, args.vote_window)
    if not args.no_warmup:
        try:
            warm_backend(server.backend)
        except ProviderFallbackError as exc:
            _log.critical("refusing to serve: %s", exc)
            if owner is not None:
                owner.close()
            return 3
    _log.info("serving %s on %s:%d", backend_name, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if owner is not None:
            owner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

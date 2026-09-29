"""Florence-2-base on ONNX Runtime: the Jetson detector's model (no torch, no ``mfw``).

The same weights the simulation lane's Florence-2 uses (onnx-community's
export of ``microsoft/Florence-2-base``), in FP16, driven by a decode loop
written here because onnxruntime has no ``generate``:

    pixel_values (1,3,768,768) -> vision_encoder        -> image features (1,577,768)
    prompt ids                  -> embed_tokens          -> prompt embeddings
    [image features; prompt]    -> encoder_model         -> encoder hidden states
    decoder_model_merged (KV cache; ``use_cache_branch`` False on step 0)
        -> greedy (default) or beam search, forced BOS at step 0, stop at EOS,
           ``no_repeat_ngram_size`` 3 (the checkpoint's generation_config)
    ``<loc_k>`` tokens -> pixel boxes, ``(k + 0.5) * size / 1000``

Only ``<CAPTION_TO_PHRASE_GROUNDING>`` is used by the detector service: the
configured labels become the caption (``"a marker. a bowl."``) and every
grounded phrase comes back with its boxes.

Scores. Florence emits no detection scores, so each box gets one from the
decoder itself: for each of its four ``<loc_k>`` tokens, the probability mass
the decoder put on bins ``k - w .. k + w`` (``w = loc_window``, 2 by
default, so a coordinate split between neighbouring bins is not mistaken for
doubt), and the box score is the geometric mean of the four. It is a
*coordinate* confidence, not a calibrated objectness score; the detector
service uses it only as one of its per-label sanity filters.

Recorded traps (all measured on the laptop, 2026-09-26, ORT 1.23.2):

* ``decoder_model_merged_fp16.onnx`` is rejected by ORT ("Subgraph output
  (logits) is an outer scope value being returned directly"): the fp16
  converter renamed the If-branch outputs' producers to
  ``graph_output_cast_i`` but left the branch outputs named ``logits`` /
  ``present.*`` and typed float32. :func:`fix_fp16_merged_decoder` repoints
  each branch output at the If node's i-th output name and types it
  float16, in memory. No weight changes, the file on disk is untouched.
* On the CPU EP the fp16 vision encoder fails to initialise inside
  ``SimplifiedLayerNormFusion``; that optimiser is disabled on CPU.
* onnxruntime-gpu built for CUDA 13 **silently** fell back to CPU on driver
  577.03: the session loads, runs, and is 10x slower. So CUDA is *required*
  unless ``allow_cpu``: the available providers are checked at load, every
  session's providers are checked at load **and again after the first
  inference**, and a session that is not on ``CUDAExecutionProvider`` raises
  :class:`ProviderFallbackError`, which also poisons every later call. On the
  Jetson install onnxruntime-gpu from the Jetson AI Lab jp6/cu126 index,
  never a CUDA 13 build.
* FP16 == FP32 on 12 sim renders: 72/72 grounded boxes IoU >= 0.8 (min
  0.985), greedy decoding. Beam search (``num_beams`` 3, the checkpoint's
  default) on 6 of them: the same phrases as greedy, boxes within 1 px,
  1.0-1.4x the decode time -- greedy stays the default.

Python 3.10 compatible; numpy at import, everything else (onnxruntime, onnx,
tokenizers, PIL) imported on first use so the pure helpers are testable
anywhere.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GROUNDING_TASK",
    "TASK_PROMPTS",
    "DEFAULT_MODEL_DIR",
    "ProviderFallbackError",
    "GroundedBox",
    "Generation",
    "ImageFeatures",
    "FlorenceOnnx",
    "fix_fp16_merged_decoder",
    "check_session_providers",
    "banned_ngram_tokens",
    "search",
    "parse_token_boxes",
    "loc_to_pixels",
    "preprocess",
    "model_paths",
]

_log = logging.getLogger("jetson.florence_onnx")

GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_PROMPTS: dict[str, str] = {
    "<OD>": "Locate the objects with category name in the image.",
    GROUNDING_TASK: "Locate the phrases in the caption: {input}",
}

IMAGE_SIZE = 768
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BOS, PAD, EOS, DECODER_START = 0, 1, 2, 2
SPECIAL_IDS = frozenset({BOS, PAD, EOS})
N_LAYERS, N_HEADS, HEAD_DIM = 6, 12, 64
NUM_LOC_BINS = 1000
FLOAT16_ENUM = 10
"""``onnx.TensorProto.FLOAT16``; a literal so the fix is testable without onnx."""

DEFAULT_MODEL_DIR = Path(os.environ.get("FLORENCE_ONNX_DIR", str(Path.home() / "jetson_models" / "florence")))
"""``onnx-community/Florence-2-base`` checkout: ``tokenizer.json`` + ``onnx/*.onnx``."""

MODEL_PARTS = ("vision_encoder", "embed_tokens", "encoder_model", "decoder_model_merged")
CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"


class ProviderFallbackError(RuntimeError):
    """A session is not on the CUDA execution provider although CUDA was required."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without onnxruntime)
# ---------------------------------------------------------------------------


def model_paths(model_dir: str | Path, precision: str = "fp16") -> dict[str, Path]:
    """The four ONNX parts and the tokenizer for ``precision``; raises if any is missing."""
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
    root = Path(model_dir).expanduser()
    suffix = "_fp16" if precision == "fp16" else ""
    paths = {part: root / "onnx" / f"{part}{suffix}.onnx" for part in MODEL_PARTS}
    paths["tokenizer"] = root / "tokenizer.json"
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Florence-2 ONNX files missing: " + ", ".join(missing)
            + ". Download onnx-community/Florence-2-base (tokenizer.json + onnx/"
            f"{{{','.join(MODEL_PARTS)}}}{suffix}.onnx) into {root} or pass --model-dir"
        )
    return paths


def fix_fp16_merged_decoder(model: Any, float16_enum: int = FLOAT16_ENUM) -> int:
    """Repair the fp16 merged decoder's If-branch outputs in place; returns how many were renamed.

    ``model`` is an ``onnx.ModelProto`` (only ``graph.node[*].op_type``,
    ``.attribute[*].g`` and ``.output`` are touched, so a duck-typed stand-in
    works in tests). A branch output whose name no node in that branch
    produces, while the If node's i-th output name *is* produced there, is
    renamed to it and typed float16 -- the only shape of the defect seen.
    """
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "If":
            continue
        for attr in node.attribute:
            branch = getattr(attr, "g", None)
            if branch is None or not len(getattr(branch, "output", ())):
                continue
            produced = {name for n in branch.node for name in n.output}
            for i, out in enumerate(branch.output):
                if i >= len(node.output):
                    break
                if out.name not in produced and node.output[i] in produced:
                    out.name = node.output[i]
                    out.type.tensor_type.elem_type = int(float16_enum)
                    fixed += 1
    return fixed


def check_session_providers(sessions: Mapping[str, Any], require_cuda: bool, when: str = "load") -> dict[str, str]:
    """``{name: first provider}``; raises :class:`ProviderFallbackError` on a CPU session when CUDA is required."""
    providers = {name: str((sess.get_providers() or ["?"])[0]) for name, sess in sessions.items()}
    if require_cuda:
        off = {name: p for name, p in providers.items() if p != CUDA_EP}
        if off:
            raise ProviderFallbackError(
                f"Florence ONNX session(s) not on CUDA after {when}: {off}. onnxruntime fell back to "
                "the CPU (a CUDA-13 build on a CUDA-12 driver does this silently, as do missing "
                "cuDNN/cuBLAS libraries). Install onnxruntime-gpu for CUDA 12 (Jetson: the Jetson AI Lab "
                "jp6/cu126 index), or pass --allow-cpu to run slowly on purpose."
            )
    return providers


def banned_ngram_tokens(tokens: Sequence[int], n: int) -> set[int]:
    """Tokens that would repeat an ``n``-gram already in ``tokens`` (HF ``no_repeat_ngram_size``)."""
    if n <= 0 or len(tokens) + 1 < n:
        return set()
    if n == 1:
        return set(int(t) for t in tokens)
    prefix = tuple(tokens[len(tokens) - (n - 1):])
    banned: set[int] = set()
    for j in range(len(tokens) - n + 1):
        if tuple(tokens[j:j + n - 1]) == prefix:
            banned.add(int(tokens[j + n - 1]))
    return banned


def _log_softmax(row: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return np.full_like(row, -np.inf)
    m = float(finite.max())
    shifted = row - m
    return shifted - math.log(float(np.exp(shifted[np.isfinite(shifted)]).sum()))


@dataclass
class _Hypothesis:
    tokens: list[int]
    score: float
    token_logprobs: list[float] = field(default_factory=list)
    loc_masses: list[float | None] = field(default_factory=list)


def search(
    step: Callable[[list[list[int]], list[int]], NDArray[np.floating]],
    num_beams: int = 1,
    max_new_tokens: int = 128,
    no_repeat_ngram_size: int = 3,
    loc_ids: tuple[int, int] | None = None,
    loc_window: int = 2,
    length_penalty: float = 1.0,
) -> tuple[list[int], list[float], list[float | None], bool]:
    """Greedy (``num_beams == 1``) or beam search over a decoder ``step``.

    ``step(beam_tokens, parent_idx)`` returns next-token logits ``(B, V)`` for
    the ``B`` alive beams; ``parent_idx[i]`` is the beam of the previous step
    that beam ``i`` extends (the caller reorders its KV cache with it; step 0
    passes ``[0] * B``). Forced BOS at step 0, EOS ends a hypothesis, early
    stopping once ``num_beams`` hypotheses finished (the checkpoint's
    ``early_stopping: true``). With one beam this is exactly greedy argmax.

    Returns ``(tokens, token_logprobs, loc_masses, truncated)`` for the best
    hypothesis: ``tokens`` starts with the decoder start token and the two
    lists have one entry per *generated* token (``len(tokens) - 1``), so
    ``loc_masses[i]`` belongs to ``tokens[i + 1]``: the probability mass
    within ``loc_window`` bins when it is a ``<loc_k>`` token (``loc_ids =
    (first_id, count)``), else ``None``.
    """
    beams = max(1, int(num_beams))
    alive = [_Hypothesis([DECODER_START], 0.0)]
    finished: list[tuple[float, _Hypothesis]] = []
    parents = [0]
    truncated = True
    for step_index in range(int(max_new_tokens)):
        logits = np.asarray(step([h.tokens for h in alive], parents), dtype=np.float64)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        vocab = logits.shape[-1]
        candidates: list[tuple[float, int, int, float, float | None]] = []
        for b, hyp in enumerate(alive):
            row = logits[b].copy()
            for t in banned_ngram_tokens(hyp.tokens, no_repeat_ngram_size):
                if 0 <= t < vocab:
                    row[t] = -np.inf
            if step_index == 0:
                forced = np.full_like(row, -np.inf)
                forced[BOS] = 0.0
                row = forced
            logp = _log_softmax(row)
            k = min(vocab, 2 * beams)
            top = np.argpartition(-logp, k - 1)[:k]
            for t in top:
                lp = float(logp[t])
                if not np.isfinite(lp):
                    continue
                mass: float | None = None
                if loc_ids is not None and loc_ids[0] <= int(t) < loc_ids[0] + loc_ids[1]:
                    lo = max(loc_ids[0], int(t) - loc_window)
                    hi = min(loc_ids[0] + loc_ids[1], int(t) + loc_window + 1)
                    mass = float(np.exp(logp[lo:hi]).sum())
                candidates.append((hyp.score + lp, b, int(t), lp, mass))
        if not candidates:
            break
        candidates.sort(key=lambda c: -c[0])
        next_alive: list[_Hypothesis] = []
        next_parents: list[int] = []
        for rank, (total, b, t, lp, mass) in enumerate(candidates):
            parent = alive[b]
            if t == EOS:
                if rank >= beams:
                    continue
                done = _Hypothesis(parent.tokens + [t], total, parent.token_logprobs + [lp],
                                   parent.loc_masses + [mass])
                gen_len = max(1, len(done.tokens) - 1)
                finished.append((total / (gen_len ** float(length_penalty)), done))
            else:
                next_alive.append(_Hypothesis(parent.tokens + [t], total, parent.token_logprobs + [lp],
                                              parent.loc_masses + [mass]))
                next_parents.append(b)
            if len(next_alive) == beams:
                break
        if len(finished) >= beams or not next_alive:
            truncated = False
            break
        alive, parents = next_alive, next_parents
    if truncated or not finished:
        for hyp in alive:
            gen_len = max(1, len(hyp.tokens) - 1)
            finished.append((hyp.score / (gen_len ** float(length_penalty)), hyp))
    best = max(finished, key=lambda item: item[0])[1]
    ended = bool(best.tokens and best.tokens[-1] == EOS)
    return best.tokens, best.token_logprobs, best.loc_masses, (not ended)


def loc_to_pixels(bins: Sequence[int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Four ``<loc_k>`` bins -> pixel ``(x0, y0, x1, y1)``: ``(k + 0.5) * size / 1000``, sorted."""
    width, height = float(size[0]), float(size[1])
    x0, y0, x1, y1 = (int(b) for b in bins)
    xs = sorted(((x0 + 0.5) * width / NUM_LOC_BINS, (x1 + 0.5) * width / NUM_LOC_BINS))
    ys = sorted(((y0 + 0.5) * height / NUM_LOC_BINS, (y1 + 0.5) * height / NUM_LOC_BINS))
    return (xs[0], ys[0], xs[1], ys[1])


@dataclass(frozen=True)
class GroundedBox:
    """One grounded phrase's box in pixels, with its decoder-derived score."""

    phrase: str
    box: tuple[float, float, float, float]
    score: float
    """Geometric mean of the four coordinates' windowed probability masses."""
    coord_masses: tuple[float, ...] = ()


def parse_token_boxes(
    tokens: Sequence[int],
    loc_masses: Sequence[float | None],
    loc_ids: tuple[int, int],
    decode: Callable[[list[int]], str],
    size: tuple[int, int],
) -> list[GroundedBox]:
    """Token stream -> ``[GroundedBox]`` (HF ``phrase_grounding`` post-processing, plus scores).

    A phrase is the text tokens before a run of ``<loc_k>`` tokens; every
    complete group of four in the run is one box for that phrase (a phrase
    with 8 loc tokens is two boxes). Specials are skipped; a trailing
    incomplete group is dropped. ``loc_masses[i]`` belongs to ``tokens[i]``.
    """
    first, count = int(loc_ids[0]), int(loc_ids[1])
    out: list[GroundedBox] = []
    phrase_ids: list[int] = []
    run: list[tuple[int, float]] = []

    def flush() -> None:
        nonlocal phrase_ids, run
        if run:
            phrase = decode(phrase_ids).strip() if phrase_ids else ""
            for g in range(0, len(run) - 3, 4):
                group = run[g:g + 4]
                masses = tuple(max(1e-12, float(m)) for _, m in group)
                score = float(np.exp(np.mean(np.log(masses))))
                if phrase:
                    out.append(GroundedBox(phrase, loc_to_pixels([k for k, _ in group], size), score, masses))
            phrase_ids = []
        run = []

    for i, tok in enumerate(tokens):
        tok = int(tok)
        if tok in SPECIAL_IDS:
            continue
        if first <= tok < first + count:
            mass = loc_masses[i] if i < len(loc_masses) and loc_masses[i] is not None else 1.0
            run.append((tok - first, float(mass)))
        else:
            if run:
                flush()
            phrase_ids.append(tok)
    flush()
    return out


def preprocess(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """RGB ``(H, W, 3)`` uint8 -> ``(1, 3, 768, 768)`` float32 (bicubic, ImageNet mean/std)."""
    from PIL import Image  # noqa: PLC0415

    image = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    x = (np.asarray(image, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class ImageFeatures:
    """The vision encoder's output for one frame (reused by every prompt on that frame)."""

    features: NDArray[np.float32]
    size: tuple[int, int]
    """The frame's ``(width, height)`` in pixels."""
    seconds: float = 0.0


@dataclass
class Generation:
    """One decode: raw text, tokens, parsed boxes and where the time went."""

    text: str
    tokens: list[int]
    boxes: list[GroundedBox]
    truncated: bool
    """The token budget ran out before EOS: trailing phrases may be missing."""
    timings: dict[str, float]
    features: ImageFeatures | None = None


class FlorenceOnnx:
    """Florence-2 ONNX sessions plus the decode loop. Not thread-safe; the server serialises calls.

    Nothing heavy happens at construction; :meth:`load` (or the first
    :meth:`generate`) creates the sessions.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        precision: str = "fp16",
        device: str = "cuda",
        allow_cpu: bool = False,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 3,
        loc_window: int = 2,
        ort_module: Any = None,
    ) -> None:
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be cuda or cpu, got {device!r}")
        if int(num_beams) < 1 or int(max_new_tokens) < 1:
            raise ValueError("num_beams and max_new_tokens must be >= 1")
        self.model_dir = Path(model_dir).expanduser()
        self.precision = precision
        self.requested_device = device
        self.allow_cpu = bool(allow_cpu)
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.loc_window = int(loc_window)
        self._ort = ort_module
        self.sessions: dict[str, Any] = {}
        self.providers: dict[str, str] = {}
        self.device = device
        self.load_s: float | None = None
        self.inferences = 0
        self._tokenizer: Any = None
        self._loc_ids: tuple[int, int] = (0, 0)
        self._decoder_outputs: list[str] = []
        self._fatal: Exception | None = None

    @property
    def require_cuda(self) -> bool:
        return self.requested_device == "cuda" and not self.allow_cpu

    @property
    def loaded(self) -> bool:
        return bool(self.sessions)

    # -- loading -----------------------------------------------------------

    def _import_ort(self) -> Any:
        if self._ort is None:
            try:
                import onnxruntime as ort  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is not importable: install onnxruntime-gpu for CUDA 12 "
                    "(Jetson: pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126)"
                ) from exc
            self._ort = ort
        return self._ort

    def load(self) -> None:
        """Create the four sessions and the tokenizer (idempotent)."""
        if self._fatal is not None:
            raise self._fatal
        if self.sessions:
            return
        paths = model_paths(self.model_dir, self.precision)
        ort = self._import_ort()
        want_cuda = self.requested_device == "cuda"
        if want_cuda and os.name == "nt" and hasattr(ort, "preload_dlls"):
            try:  # Windows wheels find the nvidia-*-cu12 pip DLLs this way
                ort.preload_dlls(directory="")
            except Exception as exc:  # noqa: BLE001 - PATH may still provide them
                _log.debug("preload_dlls failed: %s", exc)
        available = list(ort.get_available_providers())
        if want_cuda and CUDA_EP not in available:
            message = (f"onnxruntime {getattr(ort, '__version__', '?')} offers no CUDA provider "
                       f"(available: {available}); this is the CPU-only or wrong-CUDA build")
            if self.require_cuda:
                self._fatal = ProviderFallbackError(message + ". Install onnxruntime-gpu for CUDA 12 or pass --allow-cpu.")
                raise self._fatal
            _log.warning("%s; --allow-cpu given, running on the CPU (slow)", message)
            want_cuda = False
        on_cpu = not want_cuda
        providers: list[Any] = (
            [(CUDA_EP, {"arena_extend_strategy": "kSameAsRequested"}), CPU_EP] if want_cuda else [CPU_EP]
        )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        disabled = ["SimplifiedLayerNormFusion"] if on_cpu else None
        started = time.perf_counter()
        sessions: dict[str, Any] = {}
        for part in ("vision_encoder", "embed_tokens", "encoder_model"):
            sessions[part] = ort.InferenceSession(str(paths[part]), options, providers=providers,
                                                  disabled_optimizers=disabled)
        decoder_source: Any = str(paths["decoder_model_merged"])
        if self.precision == "fp16":
            try:
                import onnx  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError("the fp16 merged decoder needs the onnx package for its in-memory fix "
                                   "(pip install onnx)") from exc
            proto = onnx.load(decoder_source)
            renamed = fix_fp16_merged_decoder(proto, onnx.TensorProto.FLOAT16)
            _log.info("fp16 merged decoder: %d If-branch output(s) repointed in memory", renamed)
            decoder_source = proto.SerializeToString()
        sessions["decoder_model_merged"] = ort.InferenceSession(decoder_source, options, providers=providers,
                                                                disabled_optimizers=disabled)
        self.load_s = time.perf_counter() - started
        if self.require_cuda:
            # onnxruntime's Python ``run`` catches an EP failure (e.g. a cuDNN
            # sub-library that will not load), calls ``set_providers([CPU])``
            # and retries: the silent fallback, measured on this laptop. Off.
            for sess in sessions.values():
                if hasattr(sess, "disable_fallback"):
                    sess.disable_fallback()
        try:
            self.providers = check_session_providers(sessions, self.require_cuda, "load")
        except ProviderFallbackError as exc:
            self._fatal = exc
            raise
        self.device = "cuda" if all(p == CUDA_EP for p in self.providers.values()) else "cpu"
        if self.device == "cpu" and self.requested_device == "cuda":
            _log.warning("Florence ONNX runs on the CPU (--allow-cpu): %s", self.providers)
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        first = self._tokenizer.token_to_id("<loc_0>")
        last = self._tokenizer.token_to_id(f"<loc_{NUM_LOC_BINS - 1}>")
        if first is None or last is None or last - first != NUM_LOC_BINS - 1:
            raise RuntimeError("tokenizer.json has no contiguous <loc_0>..<loc_999> tokens")
        self._loc_ids = (int(first), NUM_LOC_BINS)
        self._decoder_outputs = [o.name for o in sessions["decoder_model_merged"].get_outputs()]
        self.sessions = sessions
        _log.info("Florence-2 ONNX %s loaded from %s in %.1f s; providers %s",
                  self.precision, self.model_dir, self.load_s, self.providers)

    # -- inference -----------------------------------------------------------

    def prompt_ids(self, text: str, task: str = GROUNDING_TASK) -> NDArray[np.int64]:
        template = TASK_PROMPTS[task]
        prompt = template.format(input=text) if "{input}" in template else template
        return np.array([self._tokenizer.encode(prompt).ids], dtype=np.int64)

    def _embed(self, ids: NDArray[np.int64]) -> NDArray[np.float32]:
        return np.asarray(self.sessions["embed_tokens"].run(None, {"input_ids": ids})[0], dtype=np.float32)

    def encode_image(self, rgb: NDArray[np.uint8]) -> ImageFeatures:
        """Run the vision encoder once; the result can serve several prompts (grounding, then region checks)."""
        if self._fatal is not None:
            raise self._fatal
        self.load()
        t0 = time.perf_counter()
        pixels = preprocess(rgb)
        features = np.asarray(self.sessions["vision_encoder"].run(None, {"pixel_values": pixels})[0], np.float32)
        return ImageFeatures(features, (int(rgb.shape[1]), int(rgb.shape[0])), time.perf_counter() - t0)

    def generate(
        self,
        rgb: NDArray[np.uint8] | None,
        text: str,
        task: str = GROUNDING_TASK,
        features: ImageFeatures | None = None,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
    ) -> Generation:
        """Run ``task`` with ``text`` on one frame (or on ``features`` already encoded from it)."""
        if self._fatal is not None:
            raise self._fatal
        self.load()
        ort = self._ort
        t0 = time.perf_counter()
        if features is None:
            if rgb is None:
                raise ValueError("generate needs rgb or features")
            features = self.encode_image(rgb)
        width, height = features.size
        ids = self.prompt_ids(text, task)
        t2 = time.perf_counter()
        enc_in = np.concatenate([features.features, self._embed(ids)], axis=1)
        enc_mask = np.ones(enc_in.shape[:2], dtype=np.int64)
        enc_out = np.asarray(self.sessions["encoder_model"].run(
            None, {"inputs_embeds": enc_in, "attention_mask": enc_mask})[0], np.float32)
        t3 = time.perf_counter()

        beams = self.num_beams if num_beams is None else max(1, int(num_beams))
        dev = self.device
        decoder = self.sessions["decoder_model_merged"]
        enc_out_v = ort.OrtValue.ortvalue_from_numpy(np.repeat(enc_out, beams, axis=0), dev, 0)
        enc_mask_v = ort.OrtValue.ortvalue_from_numpy(np.repeat(enc_mask, beams, axis=0), dev, 0)
        empty = np.zeros((beams, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
        empty_v = ort.OrtValue.ortvalue_from_numpy(empty, dev, 0)
        dec_names = [f"past_key_values.{i}.decoder.{kv}" for i in range(N_LAYERS) for kv in ("key", "value")]
        enc_names = [f"past_key_values.{i}.encoder.{kv}" for i in range(N_LAYERS) for kv in ("key", "value")]
        state: dict[str, Any] = {"dec": {n: empty_v for n in dec_names}, "enc": None, "calls": 0}

        def step(beam_tokens: list[list[int]], parents: list[int]) -> NDArray[np.float32]:
            first_call = state["calls"] == 0
            alive = len(beam_tokens)
            # Fewer live hypotheses than beams (steps 0-1): pad with the first so
            # the batch keeps its shape; the padded rows' logits are discarded.
            order = [int(p) for p in parents] + [int(parents[0])] * (beams - alive)
            if not first_call and beams > 1 and order != list(range(beams)):
                index = np.asarray(order, dtype=np.int64)
                state["dec"] = {n: ort.OrtValue.ortvalue_from_numpy(
                    np.ascontiguousarray(v.numpy()[index]), dev, 0) for n, v in state["dec"].items()}
            last = np.array([[toks[-1]] for toks in beam_tokens] + [[beam_tokens[0][-1]]] * (beams - alive),
                            dtype=np.int64)
            io = decoder.io_binding()
            io.bind_ortvalue_input("inputs_embeds", ort.OrtValue.ortvalue_from_numpy(self._embed(last), dev, 0))
            io.bind_ortvalue_input("encoder_hidden_states", enc_out_v)
            io.bind_ortvalue_input("encoder_attention_mask", enc_mask_v)
            io.bind_cpu_input("use_cache_branch", np.array([not first_call]))
            for name in dec_names:
                io.bind_ortvalue_input(name, state["dec"][name])
            for name in enc_names:
                io.bind_ortvalue_input(name, state["enc"][name] if state["enc"] is not None else empty_v)
            io.bind_output(self._decoder_outputs[0], "cpu")
            for name in self._decoder_outputs[1:]:
                io.bind_output(name, dev)
            decoder.run_with_iobinding(io)
            outs = dict(zip(self._decoder_outputs, io.get_outputs()))
            state["dec"] = {n: outs[n.replace("past_key_values", "present")] for n in dec_names}
            if state["enc"] is None:
                state["enc"] = {n: outs[n.replace("past_key_values", "present")] for n in enc_names}
            state["calls"] += 1
            logits = outs[self._decoder_outputs[0]].numpy()[:, -1, :]
            return logits[:alive]

        budget = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        tokens, _logprobs, masses, truncated = search(
            step, beams, budget, self.no_repeat_ngram_size, self._loc_ids, self.loc_window)
        t4 = time.perf_counter()
        self.inferences += 1
        if self.inferences == 1:
            try:
                self.providers = check_session_providers(self.sessions, self.require_cuda, "the first inference")
            except ProviderFallbackError as exc:
                self._fatal = exc
                raise
        text_out = self._tokenizer.decode(tokens, skip_special_tokens=False)
        for special in ("</s>", "<s>", "<pad>"):
            text_out = text_out.replace(special, "")
        # ``search`` gives one mass per *generated* token; ``tokens`` starts
        # with the decoder start token, which has none.
        boxes = parse_token_boxes(tokens, [None] + list(masses), self._loc_ids,
                                  lambda ids_: self._tokenizer.decode(ids_, skip_special_tokens=True),
                                  (width, height))
        timings = {"vision_s": features.seconds, "encoder_s": t3 - t2, "decode_s": t4 - t3,
                   "total_s": t4 - t0, "new_tokens": float(len(tokens) - 1)}
        return Generation(text_out, list(tokens), boxes, truncated, timings, features)

"""Hardware MVP, detector stream: Florence-2 ONNX on the Jetson behind the 5558 protocol.

No onnxruntime, no GPU, no webcam. Every fake here models the physical
failure it stands in for:

* **Silent CPU fallback.** A fake onnxruntime whose sessions load on CUDA and
  quietly move to the CPU on their first run (what a CUDA-13 wheel on a
  CUDA-12 driver did on the laptop) -- the backend must refuse, and stay
  refused; ``--allow-cpu`` is the only way through.
* **Grounding hallucinates.** A fake Florence that draws a box for every
  phrase it is given -- absent objects too, somewhere different on every
  frame, with scores as high as real ones -- and misses a real object on
  some frames. Only the frame vote, the per-label sanity filter and the
  two-caption-order rule stand between that and the scene graph.
* **The camera stalls / replies late / repeats a frame.** A fake
  robot_server frame source whose frames age, whose replies arrive late and
  whose ``seq`` does not advance when the camera is slower than the detector.
* **The decoder's tokens.** A scripted decoder session drives the real
  greedy / beam loop, KV-cache plumbing and ``<loc_k>`` post-processing.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jetson import florence_onnx as fo
from jetson.detector_service import (
    DetectorServer,
    Frame,
    FramePipeline,
    FrameUnavailable,
    FrameVoter,
    RobotFrameSource,
    StaleFrameError,
    box_iou,
    build_parser as detector_service_parser,
)

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script(name: str) -> types.ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}_fb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sd():
    return _load_script("serve_detector")


# ======================================================================
# Fake onnxruntime + tokenizer (the scripted decoder)
# ======================================================================

LOC0 = 1000
VOCAB = LOC0 + 1000 + 50
WORDS = {100: "a", 101: " marker", 102: " bowl", 103: ".", 104: " pen"}


class FakeOrtValue:
    def __init__(self, array: np.ndarray, device: str = "cpu") -> None:
        self.array = np.asarray(array)
        self.device = device

    @staticmethod
    def ortvalue_from_numpy(array: np.ndarray, device: str = "cpu", index: int = 0) -> "FakeOrtValue":
        return FakeOrtValue(np.array(array), device)

    def numpy(self) -> np.ndarray:
        return self.array


class FakeIo:
    def __init__(self) -> None:
        self.inputs: dict[str, Any] = {}
        self.outputs: list[str] = []
        self.result: list[FakeOrtValue] = []

    def bind_ortvalue_input(self, name: str, value: FakeOrtValue) -> None:
        self.inputs[name] = value.numpy()

    def bind_cpu_input(self, name: str, value: np.ndarray) -> None:
        self.inputs[name] = np.asarray(value)

    def bind_output(self, name: str, device: str = "cpu") -> None:
        self.outputs.append(name)

    def get_outputs(self) -> list[FakeOrtValue]:
        return self.result


class FakeSession:
    """One ONNX part. ``fallback_on_run``: the first run moves it to the CPU.

    That is onnxruntime's own behaviour when an EP fails mid-run: its Python
    ``run`` catches the error, calls ``set_providers([CPU])`` and retries --
    unless ``disable_fallback()`` was called, in which case the run raises.
    ``has_disable_fallback=False`` models a build without that switch, where
    only the post-inference provider check can catch it.
    """

    def __init__(self, part: str, providers: list[str], scripts: dict[int, list[int]] | None,
                 fallback_on_run: bool = False, has_disable_fallback: bool = True,
                 disabled_optimizers: Any = None, split: dict[int, tuple[int, float]] | None = None) -> None:
        self.part = part
        self.split = split or {}
        """step -> (runner-up token, its logit): the decoder hesitating between two tokens."""
        self.providers = [p if isinstance(p, str) else p[0] for p in providers]
        self.scripts = scripts or {}
        self.fallback_on_run = fallback_on_run
        self.fallback_disabled = False
        self.disabled_optimizers = disabled_optimizers
        self.runs = 0
        self.step = 0
        if has_disable_fallback:
            self.disable_fallback = self._disable_fallback  # type: ignore[method-assign]

    def _disable_fallback(self) -> None:
        self.fallback_disabled = True

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def _ran(self) -> None:
        self.runs += 1
        if self.fallback_on_run and self.providers[0] != fo.CPU_EP:
            if self.fallback_disabled:
                raise RuntimeError("CUDA failure 35: CUDA driver version is insufficient for CUDA runtime version")
            self.providers = [fo.CPU_EP]

    def get_outputs(self) -> list[Any]:
        names = ["logits"] + [f"present.{i}.{kind}.{kv}" for i in range(fo.N_LAYERS)
                              for kind in ("decoder", "encoder") for kv in ("key", "value")]
        return [types.SimpleNamespace(name=n) for n in names]

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self._ran()
        if self.part == "vision_encoder":
            assert feeds["pixel_values"].shape == (1, 3, fo.IMAGE_SIZE, fo.IMAGE_SIZE)
            return [np.zeros((1, 4, 8), dtype=np.float32)]
        if self.part == "embed_tokens":
            ids = feeds["input_ids"]
            return [np.zeros(ids.shape + (8,), dtype=np.float32)]
        if self.part == "encoder_model":
            return [np.asarray(feeds["inputs_embeds"], dtype=np.float32)]
        raise AssertionError(self.part)

    def io_binding(self) -> FakeIo:
        return FakeIo()

    def run_with_iobinding(self, io: FakeIo) -> None:
        self._ran()
        if not bool(np.asarray(io.inputs["use_cache_branch"]).reshape(-1)[0]):
            self.step = 0
        batch = io.inputs["inputs_embeds"].shape[0]
        logits = np.full((batch, 1, VOCAB), -20.0, dtype=np.float32)
        for b in range(batch):
            script = self.scripts.get(b, self.scripts.get(0, [2]))
            token = script[min(self.step, len(script) - 1)]
            logits[b, 0, token] = 20.0
            if self.step in self.split:
                alt, value = self.split[self.step]
                logits[b, 0, alt] = value
        self.step += 1
        past = io.inputs["past_key_values.0.decoder.key"]
        grown = np.zeros((batch, fo.N_HEADS, past.shape[2] + 1, 1), dtype=np.float32)
        out = [FakeOrtValue(logits)]
        for name in io.outputs[1:]:
            out.append(FakeOrtValue(grown))
        io.result = out


class FakeOrt:
    """Just enough of the ``onnxruntime`` module for :class:`FlorenceOnnx`."""

    __version__ = "1.23.2-fake"
    OrtValue = FakeOrtValue

    class SessionOptions:
        log_severity_level = 2

    def __init__(self, available: tuple[str, ...] = (fo.CUDA_EP, fo.CPU_EP), fallback_on_run: bool = False,
                 has_disable_fallback: bool = True, script: list[int] | None = None,
                 split: dict[int, tuple[int, float]] | None = None) -> None:
        self.split = split
        self.available = list(available)
        self.fallback_on_run = fallback_on_run
        self.has_disable_fallback = has_disable_fallback
        self.script = script if script is not None else _MARKER_SCRIPT
        self.sessions: dict[str, FakeSession] = {}

    def get_available_providers(self) -> list[str]:
        return list(self.available)

    def InferenceSession(self, source: Any, options: Any, providers: Any = None,  # noqa: N802 - ORT's name
                         disabled_optimizers: Any = None) -> FakeSession:
        text = source if isinstance(source, str) else "decoder_model_merged"
        part = next(p for p in fo.MODEL_PARTS if p in text)
        usable = [p for p in providers if (p if isinstance(p, str) else p[0]) in self.available]
        sess = FakeSession(part, usable, {0: self.script} if part == "decoder_model_merged" else None,
                           fallback_on_run=self.fallback_on_run, has_disable_fallback=self.has_disable_fallback,
                           disabled_optimizers=disabled_optimizers,
                           split=self.split if part == "decoder_model_merged" else None)
        self.sessions[part] = sess
        return sess


class FakeTokenizer:
    def token_to_id(self, token: str) -> int | None:
        if token.startswith("<loc_") and token.endswith(">"):
            return LOC0 + int(token[5:-1])
        return None

    def encode(self, text: str) -> Any:
        return types.SimpleNamespace(ids=[0] + [100 + (len(w) % 5) for w in text.split()] + [2])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (0, 1, 2):
                if not skip_special_tokens:
                    out.append({0: "<s>", 1: "<pad>", 2: "</s>"}[i])
            elif LOC0 <= i < LOC0 + 1000:
                out.append(f"<loc_{i - LOC0}>")
            else:
                out.append(WORDS.get(i, f"<{i}>"))
        return "".join(out)


def _loc(k: int) -> int:
    return LOC0 + k


# "a marker" + one box, EOS
_MARKER_SCRIPT = [0, 100, 101, _loc(100), _loc(200), _loc(300), _loc(400), 2]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "onnx").mkdir()
    for part in fo.MODEL_PARTS:
        for suffix in ("", "_fp16"):
            (tmp_path / "onnx" / f"{part}{suffix}.onnx").write_bytes(b"not a real model")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_tokenizers(monkeypatch):
    module = types.ModuleType("tokenizers")

    class Tokenizer:
        @staticmethod
        def from_file(path: str) -> FakeTokenizer:
            return FakeTokenizer()

    module.Tokenizer = Tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    return module


def _florence(model_dir: Path, ort: FakeOrt, **kw: Any) -> fo.FlorenceOnnx:
    return fo.FlorenceOnnx(model_dir=model_dir, precision="fp32", ort_module=ort, **kw)


RGB = np.full((480, 640, 3), 90, dtype=np.uint8)


# ======================================================================
# florence_onnx: pure helpers
# ======================================================================


class TestPureHelpers:
    def test_loc_bins_map_to_pixel_centres_and_sort(self):
        box = fo.loc_to_pixels([300, 400, 100, 200], (640, 480))
        assert box == pytest.approx((100.5 * 0.64, 200.5 * 0.48, 300.5 * 0.64, 400.5 * 0.48))
        assert fo.loc_to_pixels([0, 0, 999, 999], (1000, 1000)) == pytest.approx((0.5, 0.5, 999.5, 999.5))

    def test_parse_tokens_two_boxes_per_phrase_and_incomplete_group_dropped(self):
        tok = FakeTokenizer()
        tokens = [2, 0, 100, 101, _loc(10), _loc(20), _loc(30), _loc(40), _loc(50), _loc(60), _loc(70), _loc(80),
                  100, 102, _loc(1), _loc(2), _loc(3), 2]
        masses = [None, None, None, None, 0.9, 0.8, 0.9, 0.8, 1.0, 1.0, 1.0, 1.0, None, None, 0.5, 0.5, 0.5, None]
        boxes = fo.parse_token_boxes(tokens, masses, (LOC0, 1000), lambda ids: tok.decode(ids), (1000, 1000))
        assert [b.phrase for b in boxes] == ["a marker", "a marker"]  # the bowl's 3 loc tokens are no box
        assert boxes[0].box == pytest.approx((10.5, 20.5, 30.5, 40.5))
        assert boxes[0].score == pytest.approx((0.9 * 0.8 * 0.9 * 0.8) ** 0.25)
        assert boxes[1].score == pytest.approx(1.0)

    def test_no_repeat_ngram(self):
        assert fo.banned_ngram_tokens([5, 6, 7, 5, 6], 3) == {7}
        assert fo.banned_ngram_tokens([5, 6], 3) == set()
        assert fo.banned_ngram_tokens([1, 2, 3], 0) == set()
        assert fo.banned_ngram_tokens([1, 2, 1], 1) == {1, 2}

    def test_greedy_forces_bos_follows_argmax_and_stops_at_eos(self):
        script = [9, 5, 6, 2]  # step 0 wants 9, but BOS is forced

        def step(beams, parents):
            n = len(beams[0]) - 1
            logits = np.full((len(beams), 12), -5.0)
            logits[:, script[n]] = 5.0
            return logits

        tokens, logprobs, masses, truncated = fo.search(step, num_beams=1, max_new_tokens=10)
        assert tokens == [fo.DECODER_START, fo.BOS, 5, 6, fo.EOS]
        assert not truncated and len(logprobs) == 4 and masses == [None] * 4

    def test_token_budget_marks_the_output_truncated(self):
        def step(beams, parents):
            logits = np.zeros((len(beams), 12))
            logits[:, 7] = 5.0  # never EOS; the n-gram rule bans a second "7 7 7"
            logits[:, 8] = 4.0
            return logits

        tokens, _, _, truncated = fo.search(step, num_beams=1, max_new_tokens=5, no_repeat_ngram_size=3)
        assert truncated and tokens == [fo.DECODER_START, fo.BOS, 7, 7, 7, 8]

    def test_beam_search_beats_greedy_on_a_garden_path(self):
        """Greedy takes the likelier first token into a dead end; 3 beams find the better sequence."""
        a, b, c, d = 3, 4, 5, 6

        def step(beams, parents):
            rows = []
            for toks in beams:
                row = np.full(10, -30.0)
                last = toks[-1]
                if len(toks) == 1:
                    row[fo.BOS] = 0.0
                elif last == fo.BOS:
                    row[a], row[b] = np.log(0.6), np.log(0.4)
                elif last == a:
                    row[c], row[d], row[7], row[8] = [np.log(0.25)] * 4  # flat: no good continuation
                elif last == b:
                    row[fo.EOS] = np.log(0.99)
                else:
                    row[fo.EOS] = 0.0
                rows.append(row)
            return np.stack(rows)

        greedy, _, _, _ = fo.search(step, num_beams=1, max_new_tokens=6)
        beam, _, _, _ = fo.search(step, num_beams=3, max_new_tokens=6)
        assert greedy[2] == a and beam[2] == b and beam[-1] == fo.EOS

    def test_loc_window_mass_counts_neighbouring_bins(self):
        def step(beams, parents):
            n = len(beams[0]) - 1
            row = np.full(40, -50.0)
            if n == 0:
                row[fo.BOS] = 0.0
            elif n == 1:  # the coordinate is split between bins 20 and 21: not doubt
                row[20], row[21] = np.log(0.5), np.log(0.5)
            else:
                row[fo.EOS] = 0.0
            return row[None, :].repeat(len(beams), axis=0)

        _, _, masses, _ = fo.search(step, loc_ids=(10, 20), loc_window=2)
        assert masses[1] == pytest.approx(1.0)
        _, _, masses, _ = fo.search(step, loc_ids=(10, 20), loc_window=0)
        assert masses[1] == pytest.approx(0.5)

    def test_fp16_decoder_fix_repoints_only_the_broken_branch_outputs(self):
        def vi(name: str) -> Any:
            return types.SimpleNamespace(name=name, type=types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(elem_type=1)))

        broken = types.SimpleNamespace(node=[types.SimpleNamespace(output=["graph_output_cast_0"]),
                                             types.SimpleNamespace(output=["graph_output_cast_1"])],
                                       output=[vi("logits"), vi("present.0.decoder.key")])
        healthy = types.SimpleNamespace(node=[types.SimpleNamespace(output=["logits_x"])], output=[vi("logits_x")])
        if_node = types.SimpleNamespace(op_type="If", output=["graph_output_cast_0", "graph_output_cast_1"],
                                        attribute=[types.SimpleNamespace(g=broken), types.SimpleNamespace(g=healthy)])
        other = types.SimpleNamespace(op_type="MatMul", output=["y"], attribute=[])
        model = types.SimpleNamespace(graph=types.SimpleNamespace(node=[other, if_node]))
        assert fo.fix_fp16_merged_decoder(model) == 2
        assert [o.name for o in broken.output] == ["graph_output_cast_0", "graph_output_cast_1"]
        assert all(o.type.tensor_type.elem_type == fo.FLOAT16_ENUM for o in broken.output)
        assert healthy.output[0].name == "logits_x" and healthy.output[0].type.tensor_type.elem_type == 1
        assert fo.fix_fp16_merged_decoder(model) == 0  # idempotent

    def test_model_paths_name_every_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="decoder_model_merged_fp16.onnx"):
            fo.model_paths(tmp_path, "fp16")
        with pytest.raises(ValueError):
            fo.model_paths(tmp_path, "int8")


# ======================================================================
# florence_onnx: the model class against the fake runtime
# ======================================================================


class TestFlorenceOnnxRuntime:
    def test_greedy_generation_end_to_end_on_cuda(self, model_dir, fake_tokenizers):
        ort = FakeOrt()
        model = _florence(model_dir, ort)
        gen = model.generate(RGB, "a marker.")
        assert model.providers == {p: fo.CUDA_EP for p in fo.MODEL_PARTS}
        assert all(s.fallback_disabled for s in ort.sessions.values())
        assert all(s.disabled_optimizers is None for s in ort.sessions.values())  # the fusion is off on CPU only
        (box,) = gen.boxes
        assert box.phrase == "a marker" and not gen.truncated
        assert box.box == pytest.approx((100.5 * 0.64, 200.5 * 0.48, 300.5 * 0.64, 400.5 * 0.48))
        assert box.score > 0.99
        assert gen.text.startswith("a marker<loc_100>")
        assert set(gen.timings) >= {"vision_s", "encoder_s", "decode_s", "total_s"}

    def test_box_score_comes_from_its_own_four_coordinates(self, model_dir, fake_tokenizers):
        """The decoder hesitates on the LAST coordinate (x1 vs a bin 10 away): the score must show it.

        Pins the off-by-one between generated tokens and their masses: shifted
        by one token, the doubt would land on the EOS token and the box would
        score 1.0.
        """
        ort = FakeOrt(split={6: (_loc(410), 19.9)})  # step 6 emits <loc_400>
        (box,) = _florence(model_dir, ort).generate(RGB, "a marker.").boxes
        p400 = 1.0 / (1.0 + np.exp(-0.1))
        assert box.coord_masses[3] == pytest.approx(p400, abs=1e-3)
        assert box.coord_masses[:3] == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
        assert box.score == pytest.approx(p400 ** 0.25, abs=1e-3)

    def test_beam_search_runs_through_the_kv_cache_plumbing(self, model_dir, fake_tokenizers):
        model = _florence(model_dir, FakeOrt(), num_beams=3)
        gen = model.generate(RGB, "a marker.")
        assert [b.phrase for b in gen.boxes] == ["a marker"] and not gen.truncated

    def test_max_new_tokens_truncates(self, model_dir, fake_tokenizers):
        gen = _florence(model_dir, FakeOrt(), max_new_tokens=4).generate(RGB, "a marker.")
        assert gen.truncated and gen.boxes == []  # 2 of the 4 loc tokens: no box, and it says so

    def test_no_cuda_provider_refuses_to_load_and_stays_refused(self, model_dir, fake_tokenizers):
        ort = FakeOrt(available=(fo.CPU_EP,))  # the CPU-only / wrong-CUDA wheel
        model = _florence(model_dir, ort)
        with pytest.raises(fo.ProviderFallbackError, match="no CUDA provider"):
            model.load()
        with pytest.raises(fo.ProviderFallbackError):
            model.generate(RGB, "a marker.")
        assert not ort.sessions  # nothing was built on the CPU behind our back

    def test_silent_fallback_on_first_run_is_caught_after_the_first_inference(self, model_dir, fake_tokenizers):
        ort = FakeOrt(fallback_on_run=True, has_disable_fallback=False)
        model = _florence(model_dir, ort)
        model.load()
        assert set(model.providers.values()) == {fo.CUDA_EP}  # looked fine at load
        with pytest.raises(fo.ProviderFallbackError, match="after the first inference"):
            model.generate(RGB, "a marker.")
        with pytest.raises(fo.ProviderFallbackError):  # poisoned: never "works" slowly later
            model.generate(RGB, "a marker.")

    def test_fallback_with_the_switch_disabled_raises_instead_of_running_on_cpu(self, model_dir, fake_tokenizers):
        model = _florence(model_dir, FakeOrt(fallback_on_run=True, has_disable_fallback=True))
        with pytest.raises(RuntimeError, match="CUDA failure"):
            model.generate(RGB, "a marker.")

    def test_allow_cpu_runs_on_the_cpu_on_purpose(self, model_dir, fake_tokenizers):
        ort = FakeOrt(available=(fo.CPU_EP,))
        model = _florence(model_dir, ort, allow_cpu=True)
        gen = model.generate(RGB, "a marker.")
        assert model.device == "cpu" and [b.phrase for b in gen.boxes] == ["a marker"]
        assert all(s.disabled_optimizers == ["SimplifiedLayerNormFusion"] for s in ort.sessions.values())

    def test_importing_the_backend_loads_no_runtime(self):
        code = ("import sys; sys.path.insert(0, %r); import jetson.florence_onnx, jetson.detector_service\n"
                "print(','.join(k for k in ('onnxruntime', 'onnx', 'torch', 'tokenizers', 'mfw', 'zmq') "
                "if k in sys.modules))" % str(REPO_ROOT))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""

    @pytest.mark.parametrize("rel", ["jetson/florence_onnx.py", "jetson/detector_service.py",
                                     "scripts/serve_detector.py"])
    def test_jetson_files_parse_as_python_310_and_never_import_mfw(self, rel):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "mfw" for a in node.names), rel
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "mfw", rel


# ======================================================================
# serve_detector: label matching and the sanity filter
# ======================================================================


class TestMatchLabel:
    def test_shortened_phrase_maps_only_when_unambiguous(self, sd):
        assert sd.match_label("a soup", ["soup can", "bowl"]) == "soup can"
        assert sd.match_label("a blue", ["blue can", "bowl"]) == "blue can"
        assert sd.match_label("a blue", ["blue can", "blue block"]) is None

    def test_whole_words_only(self, sd):
        assert sd.match_label("a candle", ["can"]) is None
        assert sd.match_label("a can", ["can"]) == "can"
        assert sd.match_label("bowls", ["bowl"]) == "bowl"
        assert sd.match_label("a boxes", ["box"]) == "box"

    def test_never_one_phrase_to_two_labels(self, sd):
        rules = {"cup": sd.LabelRule(synonyms=("container",)), "bin": sd.LabelRule(synonyms=("container",))}
        assert sd.match_label("a container", ["cup", "bin"], rules) is None

    def test_own_name_beats_another_labels_synonym(self, sd):
        # hardware.yaml asks for cube AND block; a team config lists "block" as a cube synonym too
        rules = dict(sd.DEFAULT_LABEL_RULES, cube=sd.LabelRule(synonyms=("block", "toy cube")))
        labels = ["marker", "cube", "ball", "banana", "block", "bowl", "bin", "box"]
        assert sd.match_label("a block", labels, rules) == "block"
        assert sd.match_label("a block", ["cube", "bowl"], rules) == "cube"  # without block, the synonym counts
        assert sd.match_label("a cardboard box", labels, rules) == "box"
        assert sd.match_label("a red block", ["red block", "block"], rules) == "red block"
        assert sd.match_label("a block", labels, sd.DEFAULT_LABEL_RULES) == "block"

    def test_synonyms_and_prompts_come_from_the_config(self, sd, tmp_path):
        cfg = tmp_path / "labels.json"
        cfg.write_text(json.dumps({"labels": {"marker": {"prompt": "black pen", "synonyms": ["sharpie", "texta"]},
                                              "mug": {"synonyms": ["cup"], "max_area_frac": 0.05}}}),
                       encoding="utf-8")
        rules = sd.load_label_rules(cfg)
        assert sd.match_label("a texta", ["marker"], rules) == "marker"
        assert sd.match_label("a black pen", ["marker"], rules) == "marker"
        assert sd.match_label("a cup", ["mug", "marker"], rules) == "mug"
        assert rules["mug"].max_area_frac == 0.05 and rules["bowl"].max_area_frac == 0.30  # defaults kept
        assert sd.build_caption(["marker", "mug"], rules) == "a black pen. a mug."
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"marker": {"colour": "red"}}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown field"):
            sd.load_label_rules(bad)

    def test_default_marker_is_asked_for_as_a_pen(self, sd):
        # measured: "pen" found 2/3 markers with 1 phantom, "marker" 1/3 with 17
        assert sd.build_caption(["marker", "bowl"], sd.DEFAULT_LABEL_RULES) == "a pen. a bowl."
        assert sd.match_label("a pen", ["marker", "bowl"], sd.DEFAULT_LABEL_RULES) == "marker"


class TestSanityFilter:
    SIZE = (640, 480)

    def _obj(self, label: str, box: list[int], score: float = 0.95, match: str = "full") -> dict[str, Any]:
        return {"label": label, "confidence": score, "bbox_px": box, "match": match}

    def test_per_label_limits(self, sd):
        rules = {"marker": sd.LabelRule(min_score=0.6), "bowl": sd.LabelRule(max_area_frac=0.30)}
        objs = [
            self._obj("marker", [100, 100, 110, 160], 0.55),   # low decoder confidence
            self._obj("marker", [200, 100, 202, 160]),         # 2 px sliver
            self._obj("marker", [10, 10, 400, 300]),           # "the whole table"
            self._obj("bowl", [150, 150, 350, 330]),           # 23 %: a close bowl is allowed
            self._obj("marker", [300, 200, 320, 260]),
        ]
        kept, dropped = sd.sanity_filter(objs, self.SIZE, rules)
        assert [o["bbox_px"] for o in kept] == [[150, 150, 350, 330], [300, 200, 320, 260]]
        reasons = " | ".join(why for _, why in dropped)
        assert "score 0.55" in reasons and "side 2 px" in reasons and "of the image" in reasons

    def test_corner_boxes_are_dropped_unless_asked(self, sd):
        objs = [self._obj("bin", [0, 400, 120, 479]), self._obj("bin", [0, 200, 60, 260])]
        kept, dropped = sd.sanity_filter(objs, self.SIZE, {})
        assert [o["bbox_px"] for o in kept] == [[0, 200, 60, 260]] and dropped[0][1] == "in an image corner"
        kept, _ = sd.sanity_filter(objs, self.SIZE, {}, keep_corner_boxes=True)
        assert len(kept) == 2

    def test_one_box_two_labels(self, sd):
        # clearly stronger claim wins
        kept, _ = sd.sanity_filter([self._obj("bowl", [200, 280, 256, 325], 0.99),
                                    self._obj("ball", [201, 281, 256, 324], 0.80)], self.SIZE, {})
        assert [o["label"] for o in kept] == ["bowl"]
        # a full-word claim beats a shortened phrase at equal scores
        kept, _ = sd.sanity_filter([self._obj("soup can", [200, 280, 256, 325], 0.97, "prefix"),
                                    self._obj("bowl", [200, 280, 256, 325], 0.97)], self.SIZE, {})
        assert [o["label"] for o in kept] == ["bowl"]
        # neither clearly stronger: both go (a wrong label is worse than a miss)
        kept, dropped = sd.sanity_filter([self._obj("bowl", [200, 280, 256, 325], 0.97),
                                          self._obj("ball", [200, 280, 256, 325], 0.95)], self.SIZE, {})
        assert kept == [] and len(dropped) == 2
        # same label twice: one box
        kept, _ = sd.sanity_filter([self._obj("bowl", [200, 280, 256, 325], 0.9),
                                    self._obj("bowl", [200, 280, 257, 325], 0.8)], self.SIZE, {})
        assert len(kept) == 1 and kept[0]["confidence"] == 0.9


# ======================================================================
# The hallucinating Florence + frames + vote
# ======================================================================


class HallucinatingFlorence:
    """Grounding as measured: every phrase gets a box, present or not.

    ``scene``: label -> true box. A present object comes back jittered by a
    few pixels (the frame's sensor noise), except on the frames listed in
    ``misses[label]``. An absent label gets a box too, at a place that
    depends on the caption's order -- consecutive frames of a still scene
    give the *same* phantom for the same caption (greedy decoding is
    deterministic), a different caption moves it -- and its score is as
    high as a real object's (0.93-1.00 on the sim renders).
    ``hallucinate_stably``: the phantom does not move with the caption order
    either (Florence naming a similar-looking object that is present).
    The frame index travels in the pixel at (0, 0).
    """

    max_new_tokens = 128

    def __init__(self, scene: dict[str, list[float]], misses: dict[str, set[int]] | None = None,
                 hallucinate_stably: tuple[str, ...] = ()) -> None:
        self.scene = scene
        self.misses = misses or {}
        self.stable = set(hallucinate_stably)
        self.captions: list[str] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(self, rgb: np.ndarray, caption: str) -> Any:
        self.captions.append(caption)
        frame = int(rgb[0, 0, 0])
        phrases = [p.strip() for p in caption.split(".") if p.strip()]
        order_key = sum((i + 1) * len(p) for i, p in enumerate(phrases))
        boxes = []
        for i, phrase in enumerate(phrases):
            label = phrase[2:] if phrase.startswith("a ") else phrase
            if label in self.scene:
                if frame in self.misses.get(label, set()):
                    continue
                jitter = (frame % 3) - 1
                box = tuple(v + jitter for v in self.scene[label])
                score = 0.97
            else:
                seed = sum(map(ord, label)) if label in self.stable else 7 * i + order_key
                x = 40 + (seed * 37) % 500
                y = 60 + (seed * 53) % 350
                box = (x, y, x + 30, y + 30)
                score = 0.96
            boxes.append(fo.GroundedBox(phrase, tuple(float(v) for v in box), score))
        return fo.Generation(text="", tokens=[], boxes=boxes, truncated=False, timings={})


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(dt, 1e-3)


class CameraFrames:
    """robot_server's latest frame, as a :class:`RobotFrameSource` would hand it over.

    ``fps``: a new ``seq`` only every 1/fps s (a camera slower than the
    detector repeats a frame); ``stall_after``: from that ``seq`` on the
    camera stops and the last frame just ages; ``drop``: grabs (0-based) that
    fail like a timed-out get_frame; ``inference_s``: how far the fake clock
    moves per grab, standing in for the model's run time.
    """

    def __init__(self, clock: FakeClock, fps: float = 30.0, stall_after: int | None = None,
                 drop: set[int] | None = None, advance_s: float = 0.25) -> None:
        self.clock = clock
        self.fps = fps
        self.t0 = clock.t
        self.stall_after = stall_after
        self.drop = drop or set()
        self.advance_s = advance_s
        self.grabs = 0

    def grab(self) -> Frame:
        n = self.grabs
        self.grabs += 1
        self.clock.t += self.advance_s
        if n in self.drop:
            raise FrameUnavailable("'get_frame' to tcp://127.0.0.1:5560 got no reply within 2.0 s")
        seq = int((self.clock.t - self.t0) * self.fps)
        if self.stall_after is not None:
            seq = min(seq, self.stall_after)
        t_capture = self.t0 + seq / self.fps
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[0, 0, 0] = seq % 256
        return Frame(rgb=rgb, seq=seq, t_capture=t_capture, received=self.clock.t)


SCENE = {"marker": [300.0, 200.0, 318.0, 262.0], "bowl": [150.0, 300.0, 250.0, 380.0]}
VOCAB_WITH_ABSENT = ["marker", "bowl", "ball", "banana"]


def _pipeline(sd, model: Any, source: Any, clock: FakeClock, views: int = 2, **kw: Any) -> FramePipeline:
    backend = sd.FlorenceOnnxDetector(model=model, rules={})
    return FramePipeline(backend, source, FrameVoter(3, 2, 0.3, required_views=views),
                         clock=clock, sleep=clock.sleep, **kw)


class TestFrameVote:
    def test_phantoms_never_reported_real_objects_survive_a_missed_frame(self, sd):
        clock = FakeClock()
        model = HallucinatingFlorence(SCENE, misses={"marker": {7}})
        pipe = _pipeline(sd, model, CameraFrames(clock), clock)
        reported: list[set[str]] = []
        for _ in range(6):
            reply = pipe.detect(VOCAB_WITH_ABSENT)
            reported.append({o["label"] for o in reply["objects"]})
            clock.t += 3.0  # the next observe, after the history has aged out
        assert all(r == {"marker", "bowl"} for r in reported), reported
        # both caption orders were used, which is what defeats the phantoms
        assert {c.split(".")[0] for c in model.captions} == {"a ball", "a marker"}  # sorted, then reversed

    def test_single_frame_would_have_reported_the_phantoms(self, sd):
        """Control: without the vote the same fake does fool the detector (the fake can show the failure)."""
        backend = sd.FlorenceOnnxDetector(model=HallucinatingFlorence(SCENE), rules={})
        labels = {o["label"] for o in backend.detect(np.zeros((480, 640, 3), np.uint8), VOCAB_WITH_ABSENT)}
        assert {"ball", "banana"} <= labels

    def test_a_phantom_stable_in_one_caption_order_needs_the_other_order(self, sd):
        """Two frames of a still scene with the same caption agree on a phantom; two orders do not."""
        clock = FakeClock()
        # views=1: the old rule (2 of 3 frames). Frames 1 and 3 share a caption order.
        voter = FrameVoter(3, 2, 0.3, required_views=1)
        phantom = {"label": "ball", "confidence": 0.96, "bbox_px": [100, 100, 130, 130]}
        voter.record(1, [phantom], clock(), "forward")
        voter.record(2, [], clock(), "reversed")
        voter.record(3, [phantom], clock(), "forward")
        assert [d["label"] for d in voter.decide()[0]] == ["ball"]
        strict = FrameVoter(3, 2, 0.3, required_views=2)
        for fid, dets, view in ((1, [phantom], "forward"), (2, [], "reversed"), (3, [phantom], "forward")):
            strict.record(fid, dets, clock(), view)
        confirmed, undecided = strict.decide()
        assert confirmed == [] and undecided == set()  # window full: a miss, never a phantom
        with pytest.raises(ValueError):
            FrameVoter(3, 2, required_views=3)

    def test_a_similar_object_named_by_both_orders_is_a_known_limit(self, sd):
        """Measured on the cluttered YCB render: 'marker' was put on the drill in both orders.

        The vote cannot catch a phantom that is stable across captions; the
        test pins that honestly so nobody believes it does (keep the table to
        the demo objects).
        """
        clock = FakeClock()
        model = HallucinatingFlorence(SCENE, hallucinate_stably=("banana",))
        pipe = _pipeline(sd, model, CameraFrames(clock), clock)
        assert "banana" in {o["label"] for o in pipe.detect(VOCAB_WITH_ABSENT)["objects"]}

    def test_the_same_frame_never_votes_twice(self, sd):
        """A camera slower than the detector hands back the same seq: that is one frame, not two."""
        clock = FakeClock()
        source = CameraFrames(clock, fps=2.0, advance_s=0.001)  # a 2 fps camera: fresh, but the same seq
        pipe = _pipeline(sd, HallucinatingFlorence(SCENE), source, clock, new_frame_wait_s=0.2)
        reply = pipe.detect(["marker", "bowl"])
        assert reply["objects"] == []  # one distinct frame: undecided, so not reported
        assert reply["frames"]["voted_over"] == [0] and reply["frames"]["inferences"] == 1
        assert source.grabs > 3  # it did ask again, and got the same frame back each time
        assert set(reply["frames"]["undecided"]) == {"marker", "bowl"}

    def test_stalled_camera_is_refused_through_the_wire(self, sd):
        clock = FakeClock()
        source = CameraFrames(clock, stall_after=3, advance_s=0.3)
        backend = sd.FlorenceOnnxDetector(model=HallucinatingFlorence(SCENE), rules={})
        pipe = FramePipeline(backend, source, FrameVoter(3, 2, required_views=2), clock=clock, sleep=clock.sleep)
        server = DetectorServer("127.0.0.1", 0, backend, pipeline=pipe)
        for _ in range(3):
            source.grab()  # the camera runs, then freezes at seq 3 while time goes on
        reply = server.handle_request({"cmd": "detect", "labels": ["marker"]})
        assert reply["ok"] is False and reply["error"].startswith("StaleFrameError")
        assert "old" in reply["error"]

    def test_dropped_extra_frame_decides_with_what_it_has(self, sd):
        clock = FakeClock()
        source = CameraFrames(clock, drop={1})
        pipe = _pipeline(sd, HallucinatingFlorence(SCENE), source, clock)
        reply = pipe.detect(["marker", "bowl", "ball"])
        assert reply["objects"] == [] and reply["frames"]["inferences"] == 1  # a miss, not a phantom
        clock.t += 3.0
        reply = pipe.detect(["marker", "bowl", "ball"])  # the camera is back
        assert {o["label"] for o in reply["objects"]} == {"marker", "bowl"}

    def test_recent_frames_carry_over_and_a_new_vocabulary_resets(self, sd):
        clock = FakeClock()
        model = HallucinatingFlorence(SCENE)
        pipe = _pipeline(sd, model, CameraFrames(clock), clock)
        first = pipe.detect(["marker", "bowl"])
        assert first["frames"]["inferences"] == 2
        second = pipe.detect(["marker", "bowl"])  # within history_max_age: one new frame is enough
        assert second["frames"]["inferences"] == 1 and {o["label"] for o in second["objects"]} == {"marker", "bowl"}
        third = pipe.detect(["bowl"])
        assert third["frames"]["inferences"] == 2 and len(third["frames"]["voted_over"]) == 2

    def test_an_object_moved_between_detects_is_never_reported_at_its_old_place(self, sd):
        """Review finding: two frames from the previous request outvoted the fresh one.

        The marker is seen at A, then within history_max_age_s (1 s later: a
        lift/place on the arm) it is at B. The reply must say B (or nothing),
        never A; and when it was carried out of view it must not be reported.
        """
        clock = FakeClock()
        scene = dict(SCENE)
        model = HallucinatingFlorence(scene)
        pipe = _pipeline(sd, model, CameraFrames(clock), clock)
        first = pipe.detect(["marker", "bowl"])
        old_box = next(o["bbox_px"] for o in first["objects"] if o["label"] == "marker")
        clock.t += 1.0
        scene["marker"] = [480.0, 60.0, 498.0, 122.0]  # moved well clear of the old box
        second = pipe.detect(["marker", "bowl"])
        markers = [o["bbox_px"] for o in second["objects"] if o["label"] == "marker"]
        assert markers and all(box_iou(b, old_box) < 0.3 for b in markers), (markers, old_box)
        assert box_iou(markers[0], scene["marker"]) > 0.5
        assert second["frames"]["inferences"] == 2  # history disagreed, so it looked again
        clock.t += 1.0
        del scene["marker"]  # carried out of the camera's view (the fake draws nothing for it now)
        model.scene = scene
        third = pipe.detect(["marker", "bowl"])
        assert "marker" not in {o["label"] for o in third["objects"]}
        assert "bowl" in {o["label"] for o in third["objects"]}

    def test_history_alone_never_confirms(self, sd):
        """Pure voter: two old frames agree, the current one shows the object elsewhere."""
        voter = FrameVoter(3, 2, 0.3)
        old = {"label": "marker", "confidence": 0.9, "bbox_px": [100, 100, 140, 120]}
        new = {"label": "marker", "confidence": 0.9, "bbox_px": [300, 300, 340, 320]}
        voter.record(1, [old], 0.0)
        voter.record(2, [old], 0.1)
        voter.record(3, [new], 1.2)
        confirmed, undecided = voter.decide(current={3})
        assert confirmed == [] and undecided == {"marker"}  # look again, never report the old box
        assert [d["bbox_px"] for d in voter.decide()[0]] == [old["bbox_px"]]  # legacy call: all current
        voter.record(4, [new], 1.3)  # evicts frame 1
        confirmed, _ = voter.decide(current={3, 4})
        assert [d["bbox_px"] for d in confirmed] == [new["bbox_px"]]

    def test_request_budget_stops_early(self, sd):
        class SlowModel(HallucinatingFlorence):
            def generate(self, rgb, caption):
                time.sleep(0.05)
                return super().generate(rgb, caption)

        clock = FakeClock()
        pipe = _pipeline(sd, SlowModel(SCENE), CameraFrames(clock), clock, request_budget_s=0.26)
        reply = pipe.detect(["marker"])
        assert reply["frames"]["inferences"] == 1 and reply["objects"] == []

    def test_warm_up_on_a_black_frame_never_votes(self, sd):
        clock = FakeClock()
        model = HallucinatingFlorence(SCENE)
        backend = sd.FlorenceOnnxDetector(model=model, rules={})
        pipe = FramePipeline(backend, CameraFrames(clock), FrameVoter(3, 2, required_views=2),
                             clock=clock, sleep=clock.sleep)
        assert sd.warm_backend(backend) is not None and model.loaded
        assert len(pipe.voter) == 0
        reply = pipe.detect(["marker", "bowl"])
        assert all(fid >= 0 for fid in reply["frames"]["voted_over"])
        assert {o["label"] for o in reply["objects"]} == {"marker", "bowl"}


# ======================================================================
# RobotFrameSource: ages, late replies, clock offset
# ======================================================================


def _jpeg(w: int = 64, h: int = 48) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


class TestRobotFrameSource:
    def test_age_from_robot_server_and_late_reply_counts_as_staleness(self):
        clock = FakeClock()
        sent: list[dict[str, Any]] = []

        def rpc(endpoint, data, timeout_s):
            sent.append({"endpoint": endpoint, **data})
            clock.t += 0.40  # the reply is late: WiFi, a busy Jetson
            if endpoint == "ping":
                return {"ok": True, "t": 5.0}  # the Jetson's clock is nowhere near ours: irrelevant
            return {"jpeg": _jpeg(), "seq": 11, "age_s": 0.20, "t_capture": 5.0, "width": 64, "height": 48}

        source = RobotFrameSource("192.168.1.50", 5560, rpc=rpc, clock=clock, max_age_s=0.5)
        source.grab()
        sent.clear()
        frame = source.grab()
        assert [s["endpoint"] for s in sent] == ["get_frame"]  # age_s seen: no more clock pings
        assert sent[0]["max_age_s"] == 0.5
        assert frame.seq == 11 and frame.size == (64, 48)
        assert clock() - frame.t_capture == pytest.approx(0.60)  # 0.2 s on the Jetson + 0.4 s late
        pipe = FramePipeline(object(), source, clock=clock, sleep=clock.sleep, max_frame_age_s=0.5)
        with pytest.raises(StaleFrameError, match="0.60 s old"):
            pipe.detect(["marker"])

    def test_robot_servers_own_stale_refusal_is_a_stale_frame(self):
        def rpc(endpoint, data, timeout_s):
            return {"error": "RuntimeError: stale camera frame: the latest is 1.30 s old (limit 0.50 s, 12 failed reads since)"}

        with pytest.raises(StaleFrameError, match="1.30 s old"):
            RobotFrameSource("127.0.0.1", 5560, rpc=rpc).grab()

    def test_other_failures_are_frame_unavailable(self):
        replies = iter([{"error": "RuntimeError: no camera on this server"}, {"seq": 1}, {"jpeg": b"junk", "seq": 2}])
        source = RobotFrameSource("127.0.0.1", 5560, rpc=lambda e, d, t: next(replies))
        for match in ("no camera", "no JPEG", "undecodable"):
            with pytest.raises(FrameUnavailable, match=match) as info:
                source.grab()
            assert not isinstance(info.value, StaleFrameError)

    def test_without_age_the_clock_offset_uses_the_fastest_ping(self):
        """An older robot_server: t_capture on the Jetson clock (37 s ahead); the first ping is slow one way."""
        clock = FakeClock(1000.0)
        offset = 37.0
        legs = iter([(1.3, 0.0), (0.01, 0.01), (0.02, 0.02), (0.01, 0.01), (0.03, 0.03)])

        def rpc(endpoint, data, timeout_s):
            if endpoint == "ping":
                there, back = next(legs)
                clock.t += there
                t = clock.t + offset
                clock.t += back
                return {"ok": True, "t": t}
            clock.t += 0.01
            return {"jpeg": _jpeg(), "seq": 3, "t_capture": clock.t + offset - 0.1}

        source = RobotFrameSource("192.168.1.50", 5560, rpc=rpc, clock=clock)
        frame = source.grab()
        assert source.offset_s == pytest.approx(offset, abs=0.011)
        assert clock() - frame.t_capture == pytest.approx(0.1, abs=0.02)

    def test_same_host_needs_no_ping(self):
        calls = []

        def rpc(endpoint, data, timeout_s):
            calls.append(endpoint)
            return {"jpeg": _jpeg(), "seq": 1, "t_capture": time.time()}

        RobotFrameSource("127.0.0.1", 5560, rpc=rpc).grab()
        assert calls == ["get_frame"]


# ======================================================================
# The service end to end: TCP, CLI, refusal to start on the CPU
# ======================================================================


class TestService:
    def test_tcp_detect_through_serve_detector_build_server(self, sd, monkeypatch):
        from mfw.hardware.detector import DetectorError, RemoteDetector

        made: dict[str, Any] = {}

        class LiveCamera:
            """A 30 fps camera on the real clock (the server's pipeline uses time.time)."""

            def __init__(self) -> None:
                self.t0 = time.time()

            def grab(self) -> Frame:
                seq = int((time.time() - self.t0) * 30.0)
                rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                rgb[0, 0, 0] = seq % 256
                return Frame(rgb=rgb, seq=seq, t_capture=self.t0 + seq / 30.0, received=time.time())

        def fake_source(host, port, timeout_s=2.0, max_age_s=None):
            made.update(host=host, port=port, max_age_s=max_age_s)
            return LiveCamera()

        monkeypatch.setattr(sd, "RobotFrameSource", fake_source)
        args = sd.build_parser().parse_args(["--port", "0", "--jetson", "10.0.0.7:5561", "--max-frame-age", "0.8"])
        backend = sd.FlorenceOnnxDetector(model=HallucinatingFlorence(SCENE), rules={})
        server, _owner = sd.build_server(args, backend=backend)
        assert made == {"host": "10.0.0.7", "port": 5561, "max_age_s": 0.8}
        assert server.pipeline.voter.required_views == 2
        _, port = server.serve_in_thread(port=0)
        try:
            detector = RemoteDetector("127.0.0.1", port, labels=VOCAB_WITH_ABSENT, timeout_s=5.0)
            assert detector.connect(retries=1)["backend"] == "florence-onnx"
            labels = {o["label"] for o in detector.detect(None)}
            assert labels == {"marker", "bowl"}
            assert detector.last_reply["frames"]["inferences"] >= 2
            with pytest.raises(DetectorError, match="only from robot_server"):
                detector._request({"cmd": "detect", "labels": ["marker"], "jpeg": "AAAA"})
        finally:
            server.stop()

    def test_main_refuses_to_serve_on_a_silent_cpu_fallback(self, sd, model_dir, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt(available=(fo.CPU_EP,)))
        with caplog.at_level("CRITICAL", logger="serve_detector"):
            code = sd.main(["--model-dir", str(model_dir), "--precision", "fp32", "--port", "0",
                            "--jetson", "127.0.0.1:1"])
        assert code == 3 and "refusing to serve" in caplog.text

    def test_cli_defaults(self, sd):
        args = sd.build_parser().parse_args([])
        assert (args.backend, args.max_new_tokens, args.num_beams, args.max_frame_age) == ("florence-onnx", 128, 1, 0.5)
        assert (args.jetson, args.vote_window, args.vote_required) == ("127.0.0.1:5560", 3, 2)
        assert sd.DEFAULT_MODEL == "florence-community/Florence-2-base"
        onnx = sd.build_backend(args)
        assert onnx.model.max_new_tokens == 128 and onnx.model.num_beams == 1 and onnx.model.require_cuda
        assert sd.build_backend(sd.build_parser().parse_args(["--allow-cpu"])).model.require_cuda is False

    def test_nanoowl_is_never_the_default(self):
        parser = detector_service_parser()
        assert parser.parse_args([]).backend == "scripted"
        assert "nanoowl is NOT the sim model; do not deploy it" in " ".join(parser.format_help().split())

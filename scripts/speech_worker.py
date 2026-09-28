"""Speech recognition worker. Runs in its OWN interpreter, never Isaac Sim's.

Emits one JSON object per line on stdout: ``{"text": ..., "confidence": ...}``.

Why a separate process: speech backends load either CTranslate2 (faster-whisper)
or torch (Canary). CTranslate2 crashes when loaded inside Isaac Sim's interpreter
on Windows and takes the simulator with it, and torch has no business sharing a
process with the renderer. A crash here costs one transcription instead of the
session.

Four engines, same output protocol, chosen with ``--asr``:

* ``whisper``  -- faster-whisper. Light, CPU-friendly, returns a per-segment
  confidence that the caller's low-confidence gate can act on.
* ``canary``   -- NVIDIA Canary-Qwen-2.5B (``nvidia/canary-qwen-2.5b``, CC-BY-4.0).
  Substantially more accurate, needs a GPU and the NeMo toolkit from git trunk.
  It exposes **no per-utterance confidence**, so it reports a fixed 1.0 and the
  caller's confidence gate is effectively inert -- see ``CANARY_CONFIDENCE``.
* ``parakeet`` -- NVIDIA Parakeet-TDT-0.6B-v2, INT8, through sherpa-onnx on the
  CPU. The CPU-only fallback for the Jetson Orin Nano (1-1.5 GB RSS where NeMo
  Canary needs > 8 GB) at a published +0.09 pp WER cost, and the only engine
  here that can report a real per-utterance confidence.
* ``canary-gguf`` -- the SAME Canary-Qwen-2.5B, quantised to GGUF Q4_K_M and run
  by transcribe.cpp (``transcribe_cpp`` Python binding, v0.2.4). No torch, no
  NeMo; ~3 GB of GPU memory. Measured on the laptop it matched bf16 Canary on
  30/30 commands, and it is the engine the Jetson Orin Nano runs (CUDA, sm_87).
  Like Canary it has no per-utterance confidence and reports a fixed 1.0.

Launch with a system Python, not ``python.bat``:

    python scripts/speech_worker.py --asr whisper --model base.en
    python scripts/speech_worker.py --asr canary --device cuda
    python scripts/speech_worker.py --asr parakeet --serve --host 0.0.0.0
    python scripts/speech_worker.py --asr canary-gguf --gguf canary-qwen-2.5b-Q4_K_M.gguf \\
        --serve --host 0.0.0.0 --port 5556 --mic 0
    python scripts/speech_worker.py --stdin            # typed lines, no audio
    python scripts/speech_worker.py --asr whisper --model small.en --wav cmd.wav

``--wav`` is the offline audio path: it transcribes one or more WAV files with
the chosen engine and exits, so the ASR half of the voice pipeline can be run
end to end without a microphone (``scripts/tts_to_wav.py`` synthesises a command
WAV with Windows SAPI). Before it existed the only microphone-free mode was
``--stdin``, which echoes typed text and never touches an ASR engine -- so no
run, and no test, had ever pushed audio through a model.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 16000

#: Canary produces no usable per-utterance confidence: SALM.generate returns
#: token ids, not scores. Reporting 1.0 is honest about that -- it is a constant,
#: not a measurement -- but it does mean the caller's low-confidence gate cannot
#: filter Canary output. Use --asr whisper if that gate matters to you.
CANARY_CONFIDENCE = 1.0

#: The k2-fsa release asset for Parakeet. sherpa-onnx ships the NeMo transducer
#: pre-exported and INT8-quantised, so nothing is converted locally -- the four
#: files below are the whole model. Kept under the user's cache rather than the
#: repo because the archive is ~600 MB.
PARAKEET_MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PARAKEET_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{PARAKEET_MODEL_NAME}.tar.bz2"
)
PARAKEET_DEFAULT_MODEL_DIR = Path.home() / ".cache" / "mfw" / PARAKEET_MODEL_NAME
PARAKEET_REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


#: Where protocol lines go. Replaced by :func:`serve_forever` with a socket
#: writer so the identical mic loop feeds either a pipe (one-shot subprocess) or a
#: TCP client (persistent server) with no branching in the capture code.
_SINK = None


def _write_line(payload: dict) -> None:
    line = json.dumps(payload) + "\n"
    if _SINK is None:
        sys.stdout.write(line)
        sys.stdout.flush()
    else:
        _SINK(line)


def emit(text: str, confidence: float) -> None:
    """Write one transcript line and flush.

    Flushing matters: the parent reads line-by-line, and a buffered worker looks
    like a silent one.
    """
    _write_line({"text": text, "confidence": confidence})


def emit_event(event: str, **fields) -> None:
    """Write one control event.

    Status travels on the *same* stream as transcripts. Loading a 2.5B model takes
    ~35 s, and without a machine-readable "ready" the operator has no idea when
    the microphone is actually live -- they speak into a model that is still
    initialising and conclude the system is broken.
    """
    _write_line({"event": event, **fields})


def normalise_audio(np, samples, target_peak: float = 0.55):
    """Scale an utterance to a consistent loudness before transcription.

    Measured on this machine, the laptop mic array delivers speech peaking around
    0.01-0.05 -- roughly 20-50x quieter than the recordings ASR models are trained
    on. Feeding that in raw produces mel features far below the expected range,
    and a speech-LLM responds to weak features by inventing fluent text
    ("I'm not going to be able to do that") instead of returning nothing.

    Peak normalisation is the standard fix and costs microseconds. A floor on the
    divisor stops near-silent input being amplified into pure noise, which would
    trade one hallucination source for another.
    """
    peak = float(np.abs(samples).max())
    if peak < 1e-4:
        return samples
    return samples * (target_peak / max(peak, 0.02))


def log(message: str) -> None:
    """Diagnostics go to stderr; stdout carries the transcript + event protocol."""
    print(message, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------


class WhisperEngine:
    """faster-whisper. Accepts a float32 array directly."""

    name = "whisper"

    def __init__(self, model: str, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        log(f"whisper ready (model={model}, device={device}, compute={compute_type})")

    def transcribe(self, samples) -> list[tuple[str, float]]:
        import numpy as np

        samples = normalise_audio(np, samples)
        segments, _info = self._model.transcribe(samples, language="en", vad_filter=True)
        results = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                # avg_logprob is a log probability; exponentiating gives a usable
                # 0-1 confidence for the caller's threshold.
                results.append((text, float(min(1.0, max(0.0, 2.718281828**segment.avg_logprob)))))
        return results


class CanaryEngine:
    """NVIDIA Canary-Qwen-2.5B via NeMo's SALM.

    SALM takes **file paths**, not arrays, so each utterance is written to a
    temporary 16 kHz mono WAV. That is the model's documented input contract, not
    an inefficiency worth engineering around at one utterance per few seconds.
    """

    name = "canary"
    MODEL_ID = "nvidia/canary-qwen-2.5b"

    def __init__(self, device: str, max_new_tokens: int = 128) -> None:
        try:
            from nemo.collections.speechlm2.models import SALM
        except ImportError as exc:
            raise SystemExit(
                "NeMo is not installed in this interpreter, so Canary cannot load.\n"
                f"  {exc}\n\n"
                "Canary-Qwen-2.5B needs the NeMo *trunk* build:\n"
                '    pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"\n\n'
                "Install it into a DEDICATED venv. Installing NeMo into an existing\n"
                "environment can replace a working torch build, and on this machine\n"
                "that would break the CUDA 13 / Blackwell (sm_120) setup the GPU needs.\n\n"
                "Or use the lighter engine:  --asr whisper"
            ) from exc

        import torch

        self._max_new_tokens = max_new_tokens
        log(f"loading {self.MODEL_ID} (this downloads ~5 GB on first run)...")
        self._model = SALM.from_pretrained(self.MODEL_ID)

        if device == "cuda":
            if not torch.cuda.is_available():
                log("CUDA requested but unavailable; falling back to CPU (much slower)")
                device = "cpu"
            else:
                # Cast BEFORE the transfer, not after.
                #
                # `.cuda().to(bfloat16)` looks equivalent and is not: it sends the
                # full fp32 model across PCIe first, and torch's caching allocator
                # keeps those freed fp32 blocks in its reserved pool afterwards.
                # Measured that way, nvidia-smi reported 10.7 GB for a model whose
                # live weights are ~5 GB. Casting on the host halves both the
                # transfer and the peak.
                self._model = self._model.to(torch.bfloat16).cuda()
                # bf16 -- chosen for memory, not speed.
                #
                # Speed is not the reason: bf16 measured 0.39 s vs fp32's 0.41 s, a
                # 5% gain that would not justify the cost. The cost is real -- the
                # cast also drops the audio front-end (STFT and mel filterbank) to
                # bf16, where reduced mantissa precision degrades the features the
                # encoder sees, and a speech-LLM answers degraded features with
                # fluent invention rather than silence.
                #
                # Memory is the reason, and it is a hard constraint. The benchmark
                # scene (configs/benchmark.yaml: office interior, five furniture
                # assets, nine YCB meshes) measures 16.3 GB of VRAM on this 24.4 GB
                # card. fp32 Canary needs ~10 GB on top of that and does not fit;
                # bf16 needs ~5 GB and leaves about 3 GB spare.
                #
                # So the precision risk is accepted deliberately. If hallucinated
                # transcripts appear, move ASR off the GPU (--asr whisper --device
                # cpu, which also restores a real confidence score) rather than
                # returning to fp32 while Isaac Sim needs the same card.
                log("using bfloat16 (~5 GB, leaves headroom for Isaac Sim)")
        self._model.eval()
        log(f"canary ready (device={device})")

    def transcribe(self, samples) -> list[tuple[str, float]]:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utterance.wav"
            # 16-bit PCM mono at 16 kHz, the documented input format.
            pcm = normalise_audio(np, samples)
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(pcm.tobytes())

            prompt = [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {self._model.audio_locator_tag}",
                    "audio": [str(path)],
                }
            ]
            answer_ids = self._model.generate(
                prompts=[prompt], max_new_tokens=self._max_new_tokens
            )

        text = self._model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        return [(text, CANARY_CONFIDENCE)] if text else []


class ParakeetEngine:
    """NVIDIA Parakeet-TDT-0.6B-v2 (INT8) via sherpa-onnx's offline transducer.

    The CPU fallback for the Jetson. (The earlier claim that Canary cannot be
    quantised for the Jetson was wrong: see ``canary-gguf``, the same
    checkpoint at Q4_K_M in ~3 GB, which is the Jetson's primary engine.) This
    model runs on four Cortex-A78 cores at a real-time factor around 0.09
    (published for the A76) -- 0.3-1.6 s per spoken command -- inside
    1-1.5 GB. The published accuracy cost against Canary is +0.09 pp WER on
    LibriSpeech, which is far below the noise a cheap USB microphone adds.

    Confidence: a transducer emits a log-probability per token, and sherpa-onnx
    exposes them on the result as ``ys_log_probs`` in recent builds. When present,
    ``exp(mean)`` gives a 0-1 figure the caller's low-confidence gate can act on
    -- which Canary never offered. Whether the installed binding exposes it is
    checked at runtime rather than assumed; if it does not, 1.0 is reported, and
    that constant is documented as such rather than dressed up as a measurement.

    No model is downloaded here. Loading a model is the operator's decision on a
    board with a 7.6 GB budget, and an unexpected 600 MB fetch inside a service
    start is exactly the kind of surprise that makes a demo fail.
    """

    name = "parakeet"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SystemExit(
                "sherpa-onnx is not installed in this interpreter, so Parakeet cannot load.\n"
                f"  {exc}\n\n"
                "Install it into THIS interpreter (it is a small CPU-only wheel):\n"
                "    python -m pip install sherpa-onnx\n\n"
                "Or use another engine:  --asr whisper"
            ) from exc

        self.model_dir = Path(model_dir or PARAKEET_DEFAULT_MODEL_DIR).expanduser()
        missing = [name for name in PARAKEET_REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise SystemExit(
                f"Parakeet model files missing from {self.model_dir}: {', '.join(missing)}\n\n"
                + self.download_hint(self.model_dir)
            )

        log(
            f"loading {PARAKEET_MODEL_NAME} from {self.model_dir} "
            f"(threads={num_threads}, provider={provider})..."
        )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.model_dir / "encoder.int8.onnx"),
            decoder=str(self.model_dir / "decoder.int8.onnx"),
            joiner=str(self.model_dir / "joiner.int8.onnx"),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider=provider,
        )
        log(f"parakeet ready (provider={provider}, threads={num_threads})")

    @staticmethod
    def download_hint(model_dir: str | Path | None = None) -> str:
        """How to fetch the model by hand. Nothing here downloads anything."""
        target = Path(model_dir or PARAKEET_DEFAULT_MODEL_DIR).expanduser()
        return (
            f"Download the k2-fsa release asset {PARAKEET_MODEL_NAME}.tar.bz2 and unpack\n"
            f"it so that {target} contains {', '.join(PARAKEET_REQUIRED_FILES)}:\n"
            f"    mkdir -p {target.parent}\n"
            f"    curl -L -o {PARAKEET_MODEL_NAME}.tar.bz2 {PARAKEET_RELEASE_URL}\n"
            f"    tar xjf {PARAKEET_MODEL_NAME}.tar.bz2 -C {target.parent}\n"
            "Or point --model-dir at an existing copy."
        )

    def transcribe(self, samples) -> list[tuple[str, float]]:
        """Decode one float32, 16 kHz, mono utterance."""
        import numpy as np

        samples = np.ascontiguousarray(normalise_audio(np, samples), dtype=np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result

        text = (result.text or "").strip()
        if not text:
            return []
        return [(text, self.confidence_of(result))]

    @staticmethod
    def confidence_of(result) -> float:
        """``exp(mean(ys_log_probs))`` when the binding exposes it, else 1.0.

        Clamped to [0, 1] so a binding that reports linear probabilities under the
        same name cannot produce a confidence above one.
        """
        log_probs = getattr(result, "ys_log_probs", None)
        if log_probs is None:
            return 1.0
        values = [float(v) for v in log_probs]
        if not values:
            return 1.0
        return float(min(1.0, max(0.0, math.exp(sum(values) / len(values)))))


#: transcribe.cpp's context window for the Canary LLM half. 1024 tokens covers
#: the audio tokens of a spoken command (the worker caps an utterance at
#: ``--chunk-seconds``, 4 s by default) plus the transcript, and keeps the KV
#: cache small on the Jetson's shared 8 GB.
CANARY_GGUF_N_CTX = 1024

#: Length of the start-up warm-up call. The first run on a fresh process pays
#: for kernel compilation/loading (measured on the laptop: 55 s for the first
#: file, ~0.12 s afterwards), and that cost must land before "ready", not on
#: the operator's first spoken command.
CANARY_GGUF_WARMUP_SECONDS = 1.0


def _transcribe_cpp_hint(exc: BaseException) -> str:
    return (
        "transcribe.cpp could not be loaded, so --asr canary-gguf cannot start.\n"
        f"  {type(exc).__name__}: {exc}\n\n"
        "Needs BOTH:\n"
        "  1. the Python binding in THIS interpreter:\n"
        "       python -m pip install transcribe-cpp==0.2.4\n"
        "  2. the native library it drives. Point TRANSCRIBE_LIBRARY at the built\n"
        "     SHARED library (cmake -DTRANSCRIBE_BUILD_SHARED=ON; the ggml backend\n"
        "     modules must sit next to it):\n"
        "       Linux/Jetson:  export TRANSCRIBE_LIBRARY=<build>/bin/libtranscribe.so\n"
        "                      (or <build>/src/libtranscribe.so)\n"
        "       Windows:       set TRANSCRIBE_LIBRARY=<dir>\\transcribe.dll\n"
        "     The binding and the library must be the same base version (0.2.4)."
    )


class TranscribeCppEngine:
    """Canary-Qwen-2.5B as GGUF (Q4_K_M) through transcribe.cpp.

    The Jetson engine. Canary's bf16 NeMo build needs > 8 GB and torch; the
    GGUF build runs in ~3 GB on ggml's CUDA backend and, measured on 30 spoken
    commands on the laptop, produced the same normalised transcript as bf16 on
    all 30 (29/30 exact against the reference, both mishearing "can" as "kin").

    Backend policy -- fail loudly rather than slowly: the CUDA backend is
    required. A missing/CPU-only native build would still "work", at many
    seconds per command on the Jetson's A78 cores, and the operator would
    discover that as a robot that ignores them. ``allow_cpu=True`` is the
    explicit opt-in for development machines without a GPU.

    The model is loaded once and one session is kept for the life of the
    process; a warm-up call runs at start-up so the first real command does
    not pay the first-call compile cost.

    Confidence: like :class:`CanaryEngine`, none is available, so the fixed
    :data:`CANARY_CONFIDENCE` is reported -- which keeps the caller's
    confirmation step for voice motion commands in force.
    """

    name = "canary-gguf"

    def __init__(
        self,
        gguf_path: str | Path,
        n_ctx: int = CANARY_GGUF_N_CTX,
        allow_cpu: bool = False,
        n_threads: int = 0,
        warmup_seconds: float = CANARY_GGUF_WARMUP_SECONDS,
    ) -> None:
        import time as _time

        import numpy as np

        self.gguf_path = Path(gguf_path).expanduser()
        if not self.gguf_path.is_file():
            raise SystemExit(
                f"--gguf {self.gguf_path} does not exist.\n"
                "Expected the Canary-Qwen-2.5B GGUF, e.g. canary-qwen-2.5b-Q4_K_M.gguf."
            )
        try:
            import transcribe_cpp as tc
        except Exception as exc:  # ImportError, or TranscribeError from the DLL load
            raise SystemExit(_transcribe_cpp_hint(exc)) from exc
        self._tc = tc

        if self.cuda_available(tc):
            backend = "cuda"
        elif allow_cpu:
            backend = "cpu"
            log("WARNING: transcribe.cpp has no CUDA backend here; running Canary GGUF "
                "on the CPU because --allow-cpu was given (expect seconds per command)")
        else:
            raise SystemExit(self.no_cuda_message(tc))

        log(f"loading {self.gguf_path.name} with transcribe.cpp "
            f"({self._native_version(tc)}, backend={backend}, n_ctx={n_ctx})...")
        started = _time.perf_counter()
        self._model = tc.Model(str(self.gguf_path), backend=backend)
        loaded_backend = str(getattr(self._model, "backend", backend) or backend).lower()
        if backend == "cuda" and "cuda" not in loaded_backend:
            # Asked for CUDA, got something else: the same silent slowdown the
            # check above exists to prevent.
            self._model.close()
            raise SystemExit(
                f"transcribe.cpp loaded the model on {loaded_backend!r}, not CUDA.\n"
                + self.no_cuda_message(tc)
            )
        self.backend = loaded_backend
        self._session = self._model.session(n_threads=n_threads, n_ctx=n_ctx)
        self.load_seconds = _time.perf_counter() - started
        self.max_samples = self._max_samples()

        # One warm-up run: a quiet tone, output discarded.
        started = _time.perf_counter()
        count = max(1, int(warmup_seconds * SAMPLE_RATE))
        t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
        warm = (0.05 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        self._session.run(warm)
        self.warmup_seconds = _time.perf_counter() - started
        log(
            f"canary-gguf ready (backend={self.backend}, load {self.load_seconds:.1f} s, "
            f"warm-up {self.warmup_seconds:.1f} s"
            + (f", max audio {self.max_samples / SAMPLE_RATE:.1f} s" if self.max_samples else "")
            + ")"
        )

    @staticmethod
    def _native_version(tc) -> str:
        try:
            return f"native {tc.native_version()}"
        except Exception:  # noqa: BLE001 - version is cosmetic
            return "native version unknown"

    @staticmethod
    def cuda_available(tc) -> bool:
        """Whether the loaded native library can run on a CUDA device."""
        try:
            if tc.backend_available("cuda"):
                return True
        except Exception:  # noqa: BLE001, S110 - fall through to the device list
            pass
        try:
            return any(str(dev.kind).lower() == "cuda" for dev in tc.backends())
        except Exception:  # noqa: BLE001 - no device list means no CUDA
            return False

    @staticmethod
    def no_cuda_message(tc) -> str:
        try:
            devices = ", ".join(f"{d.name} ({d.kind})" for d in tc.backends()) or "none"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            devices = f"unknown ({exc})"
        return (
            "transcribe.cpp has NO CUDA backend in this process -- refusing to run\n"
            "Canary GGUF on the CPU (seconds per command; the robot would appear\n"
            "to ignore speech).\n"
            f"  devices the native library registered: {devices}\n\n"
            "Fix: point TRANSCRIBE_LIBRARY at a CUDA build of transcribe.cpp v0.2.4\n"
            "(Jetson: cmake -DTRANSCRIBE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87\n"
            "-DTRANSCRIBE_BUILD_SHARED=ON), with ggml-cuda next to the library and the\n"
            "CUDA runtime on the loader path. Without TRANSCRIBE_LIBRARY the binding\n"
            "loads a pip-installed native provider, which need not have CUDA (the\n"
            "v0.2.4 aarch64 release asset is CPU/Vulkan only).\n"
            "Or pass --allow-cpu to run on the CPU deliberately."
        )

    def _max_samples(self) -> int | None:
        # ``Session.limits`` is a property in transcribe_cpp 0.2.4; tolerate a
        # method form too so a binding bump does not silently drop the guard.
        try:
            limits = self._session.limits
            if callable(limits):
                limits = limits()
            limit_ms = int(limits.effective_max_audio_ms)
        except Exception:  # noqa: BLE001 - no limits means no guard, not no engine
            return None
        return limit_ms * SAMPLE_RATE // 1000 if limit_ms > 0 else None

    def transcribe(self, samples) -> list[tuple[str, float]]:
        """Decode one float32, 16 kHz, mono utterance."""
        import numpy as np

        pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return []
        if self.max_samples is not None and pcm.size > self.max_samples:
            raise ValueError(
                f"utterance is {pcm.size / SAMPLE_RATE:.1f} s but this session's n_ctx "
                f"allows {self.max_samples / SAMPLE_RATE:.1f} s; lower --chunk-seconds "
                "or raise --n-ctx"
            )
        pcm = np.clip(normalise_audio(np, pcm), -1.0, 1.0)
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        result = self._session.run(pcm)
        text = (getattr(result, "text", "") or "").strip()
        return [(text, CANARY_CONFIDENCE)] if text else []

    def close(self) -> None:
        for handle in (getattr(self, "_session", None), getattr(self, "_model", None)):
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # noqa: BLE001, S110 - best-effort release
                    pass


def build_engine(
    asr: str,
    model: str = "base.en",
    device: str = "cpu",
    model_dir: str | Path | None = None,
    gguf: str | Path | None = None,
    n_ctx: int = CANARY_GGUF_N_CTX,
    allow_cpu: bool = False,
):
    """Construct the engine named by ``--asr``.

    ``model`` is whisper's size name; ``model_dir`` is Parakeet's export folder;
    ``device`` is honoured by whisper and Canary and selects Parakeet's
    onnxruntime provider (``cuda`` only if a GPU build of sherpa-onnx is installed
    -- the CPU build is the one measured for the Jetson budget). ``gguf``,
    ``n_ctx`` and ``allow_cpu`` belong to ``canary-gguf``, which picks its own
    backend (CUDA, or CPU only when ``allow_cpu``) and ignores ``device``.
    """
    if asr == "whisper":
        return WhisperEngine(model, device)
    if asr == "canary":
        return CanaryEngine(device)
    if asr == "parakeet":
        provider = "cuda" if device == "cuda" else "cpu"
        return ParakeetEngine(model_dir=model_dir, num_threads=4, provider=provider)
    if asr == "canary-gguf":
        if not gguf:
            raise SystemExit("--asr canary-gguf needs --gguf PATH (canary-qwen-2.5b-Q4_K_M.gguf)")
        return TranscribeCppEngine(gguf, n_ctx=n_ctx, allow_cpu=allow_cpu)
    raise SystemExit(f"unknown ASR engine {asr!r}")


# ----------------------------------------------------------------------
# capture loops
# ----------------------------------------------------------------------


#: Rates tried, after the device's own default, when a microphone refuses
#: 16 kHz. All are >= 16 kHz: capturing below the model rate and upsampling
#: cannot restore the missing band.
FALLBACK_CAPTURE_RATES = (48000, 44100, 32000, 24000, 22050)

#: Output samples computed per block by the NumPy resampler (bounds memory).
_RESAMPLE_BLOCK = 8192


def _resample_poly_numpy(np, x, up: int, down: int):
    """Polyphase FIR resampling by ``up/down`` in NumPy -- the same filter and
    alignment as ``scipy.signal.resample_poly`` (Kaiser beta 5 windowed sinc,
    ``10 * max(up, down)`` half-length, gain ``up``, output ``ceil(n*up/down)``).

    The fallback for interpreters without SciPy (the Jetson image need not
    carry it). The anti-alias low-pass is the point: plain decimation or
    linear interpolation folds 8-24 kHz room noise and fan whine into the
    speech band, which a speech-LLM hears as words.
    """
    n = int(x.shape[0])
    n_out = -(-n * up // down)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    n_taps = 2 * half_len + 1
    cutoff = 1.0 / max_rate
    k = np.arange(n_taps) - half_len
    h = cutoff * np.sinc(cutoff * k) * np.kaiser(n_taps, 5.0)
    h = h * (up / h.sum())

    taps_per_phase = -(-n_taps // up)
    tap_offsets = up * np.arange(taps_per_phase)
    out = np.empty(n_out, dtype=np.float64)
    xd = np.asarray(x, dtype=np.float64)
    for start in range(0, n_out, _RESAMPLE_BLOCK):
        stop = min(n_out, start + _RESAMPLE_BLOCK)
        j = np.arange(start, stop) * down + half_len  # index in the upsampled, filtered signal
        kk = (j % up)[:, None] + tap_offsets[None, :]  # filter taps that hit real samples
        ii = (j[:, None] - kk) // up  # the input samples they hit
        valid = (kk < n_taps) & (ii >= 0) & (ii < n)
        taps = np.where(valid, h[np.minimum(kk, n_taps - 1)], 0.0)
        values = np.where(valid, xd[np.clip(ii, 0, n - 1)], 0.0)
        out[start:stop] = (taps * values).sum(axis=1)
    return out


def resample_audio(samples, from_rate: int, to_rate: int = SAMPLE_RATE, use_scipy: bool | None = None):
    """Band-limited resampling of mono audio to ``to_rate`` (float32 out).

    ``scipy.signal.resample_poly`` when SciPy imports (``use_scipy=None``),
    else the identical NumPy polyphase filter. ``use_scipy`` forces a path
    (tests use it to exercise both).
    """
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    from_rate, to_rate = int(from_rate), int(to_rate)
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError(f"sample rates must be positive (got {from_rate} -> {to_rate})")
    if from_rate == to_rate or data.size == 0:
        return np.ascontiguousarray(data, dtype=np.float32)
    g = math.gcd(from_rate, to_rate)
    up, down = to_rate // g, from_rate // g

    result = None
    if use_scipy is not False:
        try:
            from scipy.signal import resample_poly
        except ImportError:
            if use_scipy:
                raise
        else:
            result = resample_poly(data.astype(np.float64), up, down)
    if result is None:
        result = _resample_poly_numpy(np, data, up, down)
    return np.ascontiguousarray(result, dtype=np.float32)


class CaptureRateError(RuntimeError):
    """No sample rate usable for 16 kHz ASR was accepted by the input device."""


def choose_capture_rate(sd, device=None) -> int:
    """The rate to open ``device`` at: 16 kHz if it accepts it, else its native rate.

    Many microphones refuse 16 kHz outright -- WASAPI entries on Windows, and
    on the Jetson a USB mic opened as a raw ALSA ``hw:`` device that only
    clocks 48 or 44.1 kHz. Before, that ended the worker at start-up; now the
    device is opened at a rate it does accept and each utterance is resampled
    to 16 kHz (:func:`resample_audio`) before it reaches the model.

    Raises :class:`CaptureRateError` listing every refusal when nothing works.
    """
    refusals: list[str] = []

    def accepts(rate: int) -> bool:
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=rate, dtype="float32")
            return True
        except Exception as exc:  # noqa: BLE001 - PortAudio raises its own types
            refusals.append(f"{rate} Hz: {exc}")
            return False

    if accepts(SAMPLE_RATE):
        return SAMPLE_RATE

    candidates: list[int] = []
    try:
        info = sd.query_devices(device, "input") if device is None else sd.query_devices(device)
        native = round(float(info["default_samplerate"]))
        candidates.append(native)
    except Exception:  # noqa: BLE001, S110 - a missing rate falls back to the list
        pass
    candidates.extend(FALLBACK_CAPTURE_RATES)
    seen = {SAMPLE_RATE}
    for rate in candidates:
        if rate in seen or rate < SAMPLE_RATE:
            continue
        seen.add(rate)
        if accepts(rate):
            return rate
    raise CaptureRateError("; ".join(refusals) or "no rate accepted")


def load_wav(path: str | Path):
    """Read a PCM WAV file as float32 mono at :data:`SAMPLE_RATE`.

    Every engine here takes 16 kHz mono float32 in [-1, 1]; a SAPI or phone
    recording is typically 22.05/44.1 kHz and may be stereo. Channels are
    averaged and the signal is resampled with the same band-limited polyphase
    filter the live microphone path uses (:func:`resample_audio`), so a
    ``--wav`` run at 44.1 kHz exercises exactly what a 44.1 kHz microphone
    does. 8-bit (unsigned), 16-, 24- and 32-bit PCM are accepted; compressed
    WAV variants (float, ADPCM) are rejected by :mod:`wave` itself.
    """
    import numpy as np

    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path} is not a PCM WAV file: {exc}") from exc

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        data = values.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE and data.size:
        data = resample_audio(data, rate, SAMPLE_RATE)

    return np.ascontiguousarray(data, dtype=np.float32)


def run_wav_mode(
    paths: list[str],
    asr: str,
    model_name: str,
    device: str,
    model_dir: str | None = None,
    engine_options: dict | None = None,
) -> int:
    """Transcribe WAV files through the real engine and emit protocol lines.

    Emits ``{"event": "file", ...}`` before each file and the usual transcript
    lines after it, so the output can be fed straight to anything that reads the
    live protocol. Returns 2 if a file is missing or unreadable (checked before
    the model loads, because loading Canary costs ~35 s).
    """
    audio = []
    for name in paths:
        try:
            audio.append((name, load_wav(name)))
        except (OSError, ValueError) as exc:
            log(f"cannot read {name}: {exc}")
            return 2

    engine = build_engine(asr, model_name, device, model_dir, **(engine_options or {}))
    for name, samples in audio:
        emit_event("file", path=str(name), seconds=round(len(samples) / SAMPLE_RATE, 2))
        results = engine.transcribe(samples)
        if not results:
            emit_event("empty")
        for text, confidence in results:
            emit(text, confidence)
    return 0


def run_stdin_mode() -> int:
    """Echo typed lines as transcripts, for testing the pipeline without audio."""
    for line in sys.stdin:
        line = line.strip()
        if line:
            emit(line, 1.0)
    return 0


def list_microphones() -> int:
    """Print the available input devices with their indices."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        print(f"sounddevice is not installed here: {exc}", file=sys.stderr)
        return 2

    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    print("Input devices (use the index with --mic):\n")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        marker = "  <- current default" if index == default_in else ""
        print(
            f"  [{index:>2}] {device['name'][:58]:<58} "
            f"ch={device['max_input_channels']} {int(device['default_samplerate'])}Hz{marker}"
        )
    print("\nA near-field USB headset or condenser mic gives far better results than")
    print("a laptop array: more signal, less room, and no aggressive noise suppression.")
    return 0


def scan_microphones(seconds: float = 1.5) -> int:
    """Sample every input device and report which ones actually hear you.

    A dead reading on one device says nothing about *why*: wrong index, a combo
    jack that registered headphones-only, or a Windows mute all look identical.
    Recording from every device while the operator talks continuously answers it
    directly -- whichever shows signal is the one to use, and if none do, the
    problem is upstream of this program.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        print(f"audio dependencies missing: {exc}", file=sys.stderr)
        return 2

    devices = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
    print(f"Testing {len(devices)} input devices, {seconds:.1f}s each.")
    print("KEEP TALKING CONTINUOUSLY until it finishes.\n")

    results = []
    for index, device in devices:
        name = device["name"][:44]
        try:
            rate = choose_capture_rate(sd, index)
            block = int(0.1 * rate)
            with sd.InputStream(
                device=index, samplerate=rate, channels=1,
                dtype="float32", blocksize=block
            ) as stream:
                frames = []
                for _ in range(int(seconds / 0.1)):
                    data, _ = stream.read(block)
                    frames.append(data.reshape(-1))
            peak = float(np.abs(np.concatenate(frames)).max())
            results.append((peak, index, name))
            verdict = "GOOD" if peak > 0.25 else ("weak" if peak > 0.04 else "silent")
            rate_note = "" if rate == SAMPLE_RATE else f"  (at {rate} Hz, resampled)"
            print(f"  [{index:>2}] {name:<44} peak={peak:.4f}  {verdict}{rate_note}")
        except Exception as exc:
            print(f"  [{index:>2}] {name:<44} unusable ({str(exc)[:28]})")

    print()
    usable = [r for r in results if r[0] > 0.04]
    if usable:
        # Loudest is NOT best. A laptop array applies automatic gain control and
        # beamforming, so it often peaks *higher* than a headset boom mic while
        # delivering worse audio for ASR -- the AGC pumps, the beamformer chases
        # room reflections, and the noise suppressor removes speech detail. A
        # near-field headset at 0.5 beats an array at 0.9 every time, so prefer a
        # headset whenever one is actually working.
        def is_headset(name: str) -> bool:
            lowered = name.lower()
            return any(
                word in lowered
                for word in ("headset", "headphone", "hyperx", "boom", "usb audio")
            )

        headsets = sorted((r for r in usable if is_headset(r[2])), reverse=True)
        loudest = sorted(usable, reverse=True)[0]

        if headsets:
            peak, index, name = headsets[0]
            print(f"RECOMMENDED: index {index} ({name.strip()}) at peak {peak:.3f}")
            print("  Near-field headset: no beamforming, no auto gain, no noise suppression.")
            if loudest[1] != index:
                print(
                    f"  (Index {loudest[1]} peaked higher at {loudest[0]:.3f}, but it is an "
                    "array mic -\n   its AGC and noise suppression degrade ASR accuracy.)"
                )
        else:
            peak, index, name = loudest
            print(f"RECOMMENDED: index {index} ({name.strip()}) at peak {peak:.3f}")
            print("  No headset detected. An array mic works but transcribes less reliably.")
        print(f"\nUse it:  --mic {index}")
    else:
        print("No device picked up any audio. The cause is outside this program:")
        print("  1. Windows may have the mic muted or its level at 0")
        print("     Settings > System > Sound > Input > (device) > Properties")
        print("  2. Privacy may be blocking microphone access")
        print("     Settings > Privacy & security > Microphone > let apps access")
        print("  3. A 3.5mm combo jack may have registered the headset as")
        print("     headphones-only. Unplug, replug, and pick 'Headset' if asked.")
        print("  4. A USB headset would appear under its own name, not 'Realtek'.")
    return 0


def check_microphone(seconds: float = 6.0, capture_rate: int | None = None) -> int:
    """Report live input levels so a mic can be positioned before it matters.

    Placement and gain dominate ASR accuracy far more than model choice, and both
    are invisible until something has already been mis-transcribed. This makes
    them visible in advance.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        print(f"audio dependencies missing: {exc}", file=sys.stderr)
        return 2

    if capture_rate is None:
        capture_rate = resolve_default_capture_rate(sd)
    print(f"Speak normally for {seconds:.0f} seconds. Watching input level...\n")
    block = int(0.1 * capture_rate)
    peak_overall = 0.0

    with sd.InputStream(samplerate=capture_rate, channels=1, dtype="float32",
                        blocksize=block) as stream:
        for _ in range(int(seconds / 0.1)):
            data, _ = stream.read(block)
            level = float(np.abs(data).max())
            peak_overall = max(peak_overall, level)
            bars = int(min(level, 1.0) * 50)
            verdict = "GOOD" if level > 0.15 else ("weak" if level > 0.04 else "")
            print(f"\r  [{'#' * bars:<50}] {level:.3f} {verdict:<5}", end="", flush=True)

    print(f"\n\n  peak level: {peak_overall:.3f}")
    if peak_overall > 0.25:
        print("  VERDICT: good signal - Canary should transcribe accurately.")
    elif peak_overall > 0.08:
        print("  VERDICT: usable but quiet. Move closer, or raise the Windows input level.")
    else:
        print("  VERDICT: too quiet. Expect hallucinated transcripts.")
        print("  Fix: use a near-field mic, move closer, and raise the Windows input level")
        print("       (Settings > System > Sound > Input > Properties).")
    return 0


def calibrate_noise_floor(
    sd, np, seconds: float = 2.0, block_seconds: float = 0.05, capture_rate: int = SAMPLE_RATE
):
    """Measure the room's noise floor and derive a speech threshold from it.

    A fixed threshold cannot work: a quiet desktop mic idles near 0.0005 while a
    laptop array next to a fan can sit above 0.01. Guess low and every fan gust
    trips the recorder; guess high and quiet speech is ignored.

    That matters far more with Canary than with a classical recogniser. Canary is
    a speech-*LLM*: handed noise, it does not return empty, it generates fluent
    plausible text -- "I'm not going to be able to do that" -- which then reaches
    the robot as a command. Silence that never gets transcribed is the only real
    defence, so the gate is calibrated to the actual room.

    Levels are mean absolute amplitudes per ``block_seconds``, which do not
    depend on the sample rate, so a threshold measured at a device's native
    ``capture_rate`` gates that same stream correctly.

    Returns ``(threshold, ambient_median, ambient_peak)``.
    """
    block = max(1, int(block_seconds * capture_rate))
    samples = sd.rec(int(seconds * capture_rate), samplerate=capture_rate, channels=1,
                     dtype="float32")
    sd.wait()
    flat = samples.reshape(-1)

    levels = np.array(
        [np.abs(flat[i : i + block]).mean() for i in range(0, len(flat) - block, block)]
    )
    if levels.size == 0:
        return 0.005, 0.0, 0.0

    ambient = float(np.median(levels))
    peak = float(levels.max())

    # Speech must clear the *peak* of ambient with headroom, not merely its median:
    # noise is bursty, and a threshold set on the median is tripped by every gust.
    # The absolute floor stops a silent studio from producing a hair-trigger.
    threshold = max(peak * 2.5, ambient * 6.0, 0.006)
    return threshold, ambient, peak


def record_utterance(
    sd,
    np,
    silence_threshold: float,
    max_seconds: float,
    hangover_seconds: float = 0.6,
    min_speech_seconds: float = 0.45,
    block_seconds: float = 0.05,
    pre_roll_seconds: float = 0.5,
    capture_rate: int = SAMPLE_RATE,
):
    """Record one utterance, stopping as soon as the speaker stops.

    This is the single biggest latency win available, and it has nothing to do
    with the GPU. A fixed-window ``sd.rec(4s)`` always costs four seconds even
    though "pick up the can" takes about 1.2 to say -- the remaining 2.8 is dead
    air the operator waits through on every command.

    Here the stream is read in 50 ms blocks: silence before speech is discarded,
    recording ends after ``hangover_seconds`` of quiet, and ``max_seconds`` caps a
    runaway. Typical commands finish in ~1.5 s instead of a flat 4.

    ``hangover_seconds`` exists because natural speech contains gaps -- the pause
    in "pick up ... the can" would otherwise truncate the utterance mid-phrase.

    ``pre_roll_seconds`` is what makes the *start* of a command survive. Speech
    ramps up: the plosive that opens "pick" carries very little energy, so the
    first blocks fall under the threshold and, without a lookback buffer, get
    discarded -- turning "pick up the can" into "up the can", which then fails to
    parse. A rolling buffer of recent audio is kept at all times and prepended the
    moment speech is detected, so the onset is never lost.

    ``capture_rate`` is the rate the device was opened at (see
    :func:`choose_capture_rate`). Gating runs on the raw stream; the finished
    utterance is resampled to :data:`SAMPLE_RATE` once, as a whole, so there
    are no filter edges at 50 ms block boundaries.

    Returns 16 kHz float32 samples, or ``None`` if nothing was said.
    """
    from collections import deque

    block = max(1, int(block_seconds * capture_rate))
    collected = []
    # Always-on lookback so the attack of the first word is available when the
    # threshold finally trips.
    pre_roll = deque(maxlen=max(1, int(pre_roll_seconds / block_seconds)))
    speech_blocks = 0
    silence_run = 0.0
    started = False
    elapsed = 0.0

    with sd.InputStream(
        samplerate=capture_rate, channels=1, dtype="float32", blocksize=block
    ) as stream:
        while elapsed < max_seconds:
            data, _overflowed = stream.read(block)
            samples = data.reshape(-1)
            elapsed += block_seconds
            level = float(np.abs(samples).mean())

            if level >= silence_threshold:
                if not started:
                    started = True
                    # Recover the onset that was below threshold. Without this the
                    # first word is clipped and the transcript is a fragment.
                    collected.extend(pre_roll)
                    pre_roll.clear()
                    emit_event("speech_started")
                collected.append(samples)
                speech_blocks += 1
                silence_run = 0.0
            elif started:
                # Keep the trailing quiet: cutting at the first silent block
                # clips word endings, which the model then mis-transcribes.
                collected.append(samples)
                silence_run += block_seconds
                if silence_run >= hangover_seconds:
                    break
            else:
                # Not speaking yet: remember this audio in case the next block
                # turns out to be the start of a word.
                pre_roll.append(samples)

    if not started or speech_blocks * block_seconds < min_speech_seconds:
        return None
    utterance = np.concatenate(collected)
    if capture_rate != SAMPLE_RATE:
        utterance = resample_audio(utterance, capture_rate, SAMPLE_RATE)
    return utterance


def run_microphone_mode(
    asr: str,
    model_name: str,
    device: str,
    chunk_seconds: float,
    silence_threshold: float,
    model_dir: str | None = None,
    engine_options: dict | None = None,
    capture_rate: int | None = None,
) -> int:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        log(
            "Audio dependencies are missing. Install them into THIS interpreter "
            "(not Isaac Sim's):\n"
            "    python -m pip install sounddevice\n"
            f"  ({exc})"
        )
        return 2

    emit_event("loading", engine=asr)
    engine = build_engine(asr, model_name, device, model_dir, **(engine_options or {}))
    if capture_rate is None:
        capture_rate = resolve_default_capture_rate(sd)

    # The operator's cue that the microphone is genuinely live.
    emit_event("ready", engine=asr, chunk_seconds=chunk_seconds, warm=False,
               capture_rate=capture_rate)
    log("listening - speak a command")

    _capture_loop(sd, np, engine, chunk_seconds, silence_threshold, capture_rate=capture_rate)
    return 0


def resolve_default_capture_rate(sd) -> int:
    """:func:`choose_capture_rate` for the default input device.

    Without ``--mic`` nothing was checked at start-up, so this runs just before
    capture. If even that probe fails, 16 kHz is kept -- the stream then fails
    to open inside the capture loop with PortAudio's own error, as before.
    """
    try:
        rate = choose_capture_rate(sd, None)
    except CaptureRateError as exc:
        log(f"default input device accepted no usable rate ({exc}); trying {SAMPLE_RATE} Hz")
        return SAMPLE_RATE
    if rate != SAMPLE_RATE:
        log(f"default input device refuses {SAMPLE_RATE} Hz; capturing at {rate} Hz "
            f"and resampling to {SAMPLE_RATE} Hz")
    return rate


def _capture_loop(
    sd,
    np,
    engine,
    chunk_seconds: float,
    silence_threshold: float,
    capture_rate: int = SAMPLE_RATE,
) -> None:
    """Record, transcribe, emit -- forever.

    Shared verbatim by the one-shot subprocess and the persistent server: they
    differ only in where ``_write_line`` sends bytes, so there is exactly one
    capture implementation to reason about and test.

    ``silence_threshold <= 0`` means "calibrate to this room".
    """
    import time as _time

    if silence_threshold <= 0:
        emit_event("calibrating")
        log("measuring the noise floor - stay quiet for 2 seconds...")
        silence_threshold, ambient, peak = calibrate_noise_floor(
            sd, np, capture_rate=capture_rate
        )
        emit_event(
            "calibrated",
            threshold=round(silence_threshold, 5),
            ambient=round(ambient, 5),
            peak=round(peak, 5),
        )
        log(
            f"noise floor: median {ambient:.5f}, peak {peak:.5f} "
            f"-> speech threshold {silence_threshold:.5f}"
        )

    while True:
        emit_event("listening")
        try:
            samples = record_utterance(
                sd, np, silence_threshold=silence_threshold, max_seconds=chunk_seconds,
                capture_rate=capture_rate,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            log(f"audio capture failed: {exc}")
            emit_event("error", message=f"audio capture failed: {exc}")
            return

        # Nothing was said: loop straight back rather than asking the model to
        # transcribe room noise, which reliably hallucinates short phrases.
        if samples is None:
            continue

        # Second gate: the captured audio must clearly exceed the threshold, not
        # merely have brushed it. Canary invents fluent sentences from noise, so a
        # marginal clip is worse than no clip -- it becomes a robot command.
        speech_level = float(np.abs(samples).mean())
        if speech_level < silence_threshold * 1.3:
            emit_event("too_quiet", level=round(speech_level, 5),
                       threshold=round(silence_threshold, 5))
            continue

        started_at = _time.perf_counter()
        emit_event("thinking", seconds=round(len(samples) / SAMPLE_RATE, 2),
                   level=round(speech_level, 5))
        try:
            results = engine.transcribe(samples)
            emit_event("timing", inference_s=round(_time.perf_counter() - started_at, 2))
            if not results:
                emit_event("empty")
            for text, confidence in results:
                emit(text, confidence)
        except Exception as exc:
            # One bad utterance must not end the session.
            log(f"transcription failed: {type(exc).__name__}: {exc}")
            emit_event("error", message=f"{type(exc).__name__}: {exc}")


#: Exit status when the serve port is already taken (distinct from 2, which
#: means a missing dependency or bad input).
EXIT_PORT_IN_USE = 3


class PortInUseError(OSError):
    """The ``--serve`` port is held by another process."""


def port_in_use_message(host: str, port: int, exc: BaseException) -> str:
    """Why a second speech server must not start, and what to do instead."""
    return (
        f"speech server NOT started: {host}:{port} is already in use ({exc}).\n"
        "Another speech server is probably running already -- possibly a duplicate\n"
        "spawned by a venv launcher stub, which previously loaded a SECOND Canary\n"
        "(~5.5 GB more VRAM) and took the microphone away from the process that\n"
        "was actually serving transcripts.\n"
        "  - to use the running server: run_assistant.py --voice --voice-server "
        f"{host}:{port}\n"
        f"  - to find the owner:        netstat -ano | findstr :{port}   (Windows)\n"
        f"                              ss -ltnp | grep :{port}           (Linux)\n"
        "  - or pick another port with --port N.\n"
        "Nothing was loaded and the microphone was not opened."
    )


def bind_server_socket(host: str, port: int):
    """Bind (but do not yet listen on) the ``--serve`` port, exclusively.

    Called BEFORE the model loads and before any microphone handling, so a
    duplicate process fails in milliseconds instead of loading a second model
    and opening the same microphone. Listening starts only once the model is
    resident (:func:`serve_forever`): a bound, non-listening port refuses
    connections, so a client's "is it up?" probe keeps meaning "ready".

    Windows: ``SO_REUSEADDR`` there lets a second socket bind a port another
    socket already holds -- exactly what must not happen -- so the socket asks
    for ``SO_EXCLUSIVEADDRUSE`` instead.

    POSIX: no ``SO_REUSEADDR`` at all. The port is claimed *before* listening,
    and Linux lets two ``SO_REUSEADDR`` sockets bind the same port as long as
    neither is listening yet (socket(7)) -- measured on GitHub's ubuntu-latest
    runner, where the duplicate's bind succeeded. Without the option the
    second bind fails with EADDRINUSE immediately. The cost: a restart within
    ~60 s of a crash that left server-side TIME_WAIT connections is refused
    too, with the same "already in use" message -- wait, or pick another port.

    Raises :class:`PortInUseError` with an operator-facing message.
    """
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                server.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        server.bind((host, port))
    except OSError as exc:
        server.close()
        raise PortInUseError(port_in_use_message(host, port, exc)) from exc
    return server


def serve_forever(
    host: str,
    port: int,
    asr: str,
    model_name: str,
    device: str,
    chunk_seconds: float,
    silence_threshold: float,
    model_dir: str | None = None,
    server_socket=None,
    engine_options: dict | None = None,
    capture_rate: int | None = None,
) -> int:
    """Load the model once and serve transcripts to clients over TCP.

    Measured on this machine, ``SALM.from_pretrained`` costs 35.1 s of disk and
    single-threaded CPU work plus 3.7 s of PCIe transfer -- roughly 90% of it with
    the GPU idle, and no CUDA setting touches it. The only way to stop paying it is
    to stop repeating it, which is exactly what Ollama does by keeping models
    resident.

    So: load once here, then every assistant launch attaches in milliseconds. The
    microphone is a single exclusive device, so one client is served at a time;
    when it disconnects the loop returns to waiting with the model still warm.

    ``server_socket`` is the port :func:`main` already bound (see
    :func:`bind_server_socket`); when omitted it is bound here, still before
    the model loads.
    """
    global _SINK

    server = server_socket
    if server is None:
        try:
            server = bind_server_socket(host, port)
        except PortInUseError as exc:
            log(str(exc))
            return EXIT_PORT_IN_USE

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        server.close()
        log(f"audio dependencies missing: {exc}")
        return 2

    try:
        log(f"port {host}:{port} reserved; loading {asr} model (one time)...")
        engine = build_engine(asr, model_name, device, model_dir, **(engine_options or {}))
        if capture_rate is None:
            capture_rate = resolve_default_capture_rate(sd)
        log("model resident - clients now attach instantly")
        try:
            server.listen(1)
        except OSError as exc:
            server.close()
            log(port_in_use_message(host, port, exc))
            return EXIT_PORT_IN_USE
    except BaseException:
        server.close()
        raise
    log(f"speech server listening on {host}:{port} (Ctrl+C to stop)")

    try:
        while True:
            log("waiting for a client...")
            conn, addr = server.accept()
            log(f"client connected from {addr}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def sink(line: str, _conn=conn) -> None:
                _conn.sendall(line.encode("utf-8"))

            _SINK = sink
            try:
                # The model is already loaded, so "ready" is immediate -- which is
                # the entire point of this mode.
                emit_event("ready", engine=asr, chunk_seconds=chunk_seconds, warm=True,
                           capture_rate=capture_rate)
                _capture_loop(sd, np, engine, chunk_seconds, silence_threshold,
                              capture_rate=capture_rate)
            except (ConnectionError, OSError) as exc:
                log(f"client disconnected: {exc}")
            finally:
                _SINK = None
                try:
                    conn.close()
                except OSError:
                    pass
                # Deliberately NOT unloading the model: the next client is the
                # reason this process exists.
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speech recognition worker")
    parser.add_argument(
        "--asr", default="whisper", choices=["whisper", "canary", "parakeet", "canary-gguf"]
    )
    parser.add_argument(
        "--model", default="base.en", help="whisper model (ignored for the other engines)"
    )
    parser.add_argument(
        "--gguf",
        default=None,
        metavar="PATH",
        help="canary-gguf only: the Canary-Qwen-2.5B GGUF (e.g. canary-qwen-2.5b-Q4_K_M.gguf). "
        "transcribe.cpp's native library is found through TRANSCRIBE_LIBRARY",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=CANARY_GGUF_N_CTX,
        help=f"canary-gguf only: session context in tokens (default {CANARY_GGUF_N_CTX})",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="canary-gguf only: run on the CPU when transcribe.cpp has no CUDA backend "
        "(default: refuse and exit, because CPU decoding is seconds per command)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="parakeet only: folder holding the sherpa-onnx export "
        f"(default {PARAKEET_DEFAULT_MODEL_DIR}); never downloaded automatically",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.0,
        help="speech detection threshold. 0 (the default) calibrates to your room's "
        "measured noise floor, which is what stops Canary hallucinating sentences "
        "from fan noise. Pass a value only to override.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read typed lines instead of audio (pipeline test mode)",
    )
    parser.add_argument(
        "--wav",
        action="append",
        default=None,
        metavar="PATH",
        help="transcribe this WAV file (repeatable) with --asr and exit; no microphone "
        "needed. Any PCM rate/channel count is converted to 16 kHz mono",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="stay resident and serve clients over TCP, so the model loads once",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--mic", type=int, default=None, help="input device index (see --list-mics)"
    )
    parser.add_argument("--list-mics", action="store_true", help="list input devices and exit")
    parser.add_argument(
        "--scan-mics",
        action="store_true",
        help="record briefly from EVERY input device to find which one hears you",
    )
    parser.add_argument(
        "--check-mic",
        action="store_true",
        help="show a live input-level meter so you can position the mic, then exit",
    )
    args = parser.parse_args(argv)
    if args.asr == "canary-gguf" and not args.gguf and not (args.stdin or args.list_mics):
        parser.error("--asr canary-gguf needs --gguf PATH")
    args.engine_options = engine_options_from_args(args)

    if args.list_mics:
        return list_microphones()

    # Before any microphone handling: offline files need no audio device.
    if args.wav:
        return run_wav_mode(
            args.wav, args.asr, args.model, args.device, args.model_dir,
            engine_options=args.engine_options,
        )

    # Before --mic validation: the whole point is to find a working index when
    # the one you picked is silent.
    if args.scan_mics:
        return scan_microphones()

    # --serve: claim the port FIRST -- before sounddevice, the --mic check, the
    # model load and the microphone. A duplicate process (a venv launcher stub
    # once produced two) must fail here in milliseconds, not load a second
    # Canary and steal the microphone from the process that serves clients.
    server_socket = None
    serving = args.serve and not (args.stdin or args.check_mic)
    if serving:
        try:
            server_socket = bind_server_socket(args.host, args.port)
        except PortInUseError as exc:
            log(str(exc))
            return EXIT_PORT_IN_USE

    try:
        return _run_selected_mode(args, server_socket)
    finally:
        if server_socket is not None:
            # serve_forever closes it on every path; this covers the modes that
            # return before reaching it (a bad --mic, missing sounddevice).
            try:
                server_socket.close()
            except OSError:
                pass


def engine_options_from_args(args) -> dict:
    """The engine-specific keyword arguments :func:`build_engine` takes."""
    if getattr(args, "asr", None) != "canary-gguf":
        return {}
    return {"gguf": args.gguf, "n_ctx": args.n_ctx, "allow_cpu": args.allow_cpu}


def _run_selected_mode(args, server_socket) -> int:
    """Everything after argument parsing and the --serve port claim."""
    engine_options = getattr(args, "engine_options", None) or engine_options_from_args(args)
    capture_rate = None
    if args.mic is not None:
        import sounddevice as sd

        # Fail here, with an explanation, rather than deep inside the capture
        # loop. The same physical mic is listed once per audio API, and the WASAPI
        # entry refuses 16 kHz outright -- it will not resample in shared mode.
        # Such a device is now opened at a rate it accepts (its native rate) and
        # every utterance is resampled to 16 kHz; only a device that accepts no
        # usable rate at all stops the worker.
        try:
            capture_rate = choose_capture_rate(sd, args.mic)
        except CaptureRateError as exc:
            name = "unknown"
            try:
                name = sd.query_devices(args.mic)["name"]
            except Exception:
                pass
            print(
                f"Input device {args.mic} ({name}) cannot record mono audio at "
                f"{SAMPLE_RATE} Hz or at any rate that can be resampled to it.\n"
                f"  {exc}\n\n"
                "The same microphone is usually listed several times, once per audio\n"
                "API; another entry for the same device normally works.\n"
                "Run --list-mics and try another index for the same name.",
                file=sys.stderr,
            )
            return 2

        # Set the *input* half only; leaving output untouched avoids hijacking
        # playback, which is what carries the assistant's spoken replies.
        sd.default.device = (args.mic, sd.default.device[1])
        log(f"using input device {args.mic}: {sd.query_devices(args.mic)['name']}")
        if capture_rate != SAMPLE_RATE:
            log(f"input device {args.mic} refuses {SAMPLE_RATE} Hz; capturing at "
                f"{capture_rate} Hz and resampling each utterance to {SAMPLE_RATE} Hz")

    if args.check_mic:
        return check_microphone(capture_rate=capture_rate)

    if args.stdin:
        return run_stdin_mode()
    if args.serve:
        return serve_forever(
            args.host,
            args.port,
            args.asr,
            args.model,
            args.device,
            args.chunk_seconds,
            args.silence_threshold,
            args.model_dir,
            server_socket=server_socket,
            engine_options=engine_options,
            capture_rate=capture_rate,
        )
    return run_microphone_mode(
        args.asr,
        args.model,
        args.device,
        args.chunk_seconds,
        args.silence_threshold,
        args.model_dir,
        engine_options=engine_options,
        capture_rate=capture_rate,
    )


if __name__ == "__main__":
    raise SystemExit(main())

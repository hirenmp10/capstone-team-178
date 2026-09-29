"""speech_worker.py --asr canary-gguf: Canary-Qwen-2.5B GGUF via transcribe.cpp.

No model and no native library: ``transcribe_cpp`` is replaced by a stub that
behaves like the 0.2.4 binding where it matters -- it rejects anything but a
1-D, C-contiguous float32 buffer (as ``_pcm_to_carray`` does), its first run
is the slow one (kernel JIT: 55 s measured on the laptop), a CPU-only build
registers no CUDA device, and ``Session.limits`` is a property whose
``effective_max_audio_ms`` depends on ``n_ctx``.

The real model was run through this engine separately (30 recorded commands,
29/30 exact -- see the stream report); these tests pin the behaviour around it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.phase8

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "speech_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("speech_worker_gguf_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _StubLimits:
    def __init__(self, n_ctx: int):
        self.effective_n_ctx = n_ctx
        # Measured on the real Q4_K_M: n_ctx 1024 -> 58 880 ms, 512 -> 17 920 ms.
        self.effective_max_audio_ms = {512: 17920, 1024: 58880}.get(n_ctx, 3253760)
        self.max_kv_bytes = 0


class _StubTranscribeCpp(types.ModuleType):
    """A transcribe_cpp stand-in. ``cuda`` False models a CPU-only native build."""

    def __init__(self, cuda: bool = True, loads_on: str | None = None, text: str = "Pick up the marker."):
        super().__init__("transcribe_cpp")
        self.cuda = cuda
        self.loads_on = loads_on  # the backend Model reports, if not the requested one
        self.text = text
        self.models: list = []
        self.runs: list = []  # (kind, samples) in call order
        self.run_seconds: list = []  # simulated wall time per call

    # module-level API
    def native_version(self):
        return "0.2.4"

    def backend_available(self, backend):
        return backend == "cpu" or (backend == "cuda" and self.cuda)

    def backends(self):
        devices = [types.SimpleNamespace(name="CPU", kind="cpu")]
        if self.cuda:
            devices.insert(0, types.SimpleNamespace(name="CUDA0", kind="cuda"))
        return devices

    def Model(self, path, backend="auto", device=None):
        if backend == "cuda" and not self.cuda:
            raise RuntimeError("backend error: CUDA requested but not registered")
        model = _StubModel(self, path, backend)
        self.models.append(model)
        return model


class _StubModel:
    def __init__(self, tc, path, backend):
        self.tc, self.path = tc, path
        self.backend = tc.loads_on or ("cuda0" if backend == "cuda" else backend)
        self.sessions: list = []
        self.closed = False

    def session(self, n_threads=0, kv_type="auto", n_ctx=0):
        session = _StubSession(self.tc, n_ctx)
        self.sessions.append(session)
        return session

    def close(self):
        self.closed = True


class _StubSession:
    def __init__(self, tc, n_ctx):
        self.tc, self.n_ctx = tc, n_ctx
        self.closed = False

    @property
    def limits(self):  # a property in 0.2.4, not a method
        return _StubLimits(self.n_ctx)

    def run(self, pcm):
        # The binding's own input contract (transcribe_cpp._pcm_to_carray).
        if not isinstance(pcm, np.ndarray) or pcm.dtype != np.float32:
            raise TypeError(f"PCM must be float32 (got {getattr(pcm, 'dtype', type(pcm))})")
        if pcm.ndim != 1:
            raise ValueError(f"PCM buffer must be 1-D (got shape {pcm.shape})")
        if not pcm.flags["C_CONTIGUOUS"]:
            raise ValueError("PCM buffer must be C-contiguous")
        # First run on a fresh process compiles kernels: slow. Recorded, not slept.
        self.tc.run_seconds.append(55.0 if not self.tc.runs else 0.12)
        self.tc.runs.append(pcm.copy())
        text = self.tc.text if float(np.abs(pcm).max()) > 0.1 else ""
        return types.SimpleNamespace(text=text)

    def close(self):
        self.closed = True


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "canary-qwen-2.5b-Q4_K_M.gguf"
    path.write_bytes(b"GGUF\x03\x00\x00\x00")
    return path


def _install(monkeypatch, stub):
    monkeypatch.setitem(sys.modules, "transcribe_cpp", stub)
    return stub


def _speech(seconds=1.5, rate=16000, amplitude=0.02, dtype=np.float64):
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * 300 * t)).astype(dtype)


class TestBackendPolicy:
    def test_cuda_is_used_when_the_native_build_has_it(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp(cuda=True))
        engine = worker.TranscribeCppEngine(gguf)
        assert engine.backend == "cuda0"
        assert stub.models[0].path == str(gguf)

    def test_cpu_only_build_is_refused_loudly(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp(cuda=False))
        with pytest.raises(SystemExit) as info:
            worker.TranscribeCppEngine(gguf)
        message = str(info.value)
        assert "NO CUDA backend" in message and "--allow-cpu" in message
        assert "TRANSCRIBE_LIBRARY" in message and "CMAKE_CUDA_ARCHITECTURES=87" in message
        assert "TRANSCRIBE_BUILD_SHARED=ON" in message  # the binding needs a shared library
        assert "CPU (cpu)" in message  # what the library did register
        assert stub.models == [], "nothing may be loaded before refusing"

    def test_allow_cpu_is_the_explicit_opt_in(self, monkeypatch, gguf, capsys):
        worker = _load_worker()
        _install(monkeypatch, _StubTranscribeCpp(cuda=False))
        engine = worker.TranscribeCppEngine(gguf, allow_cpu=True)
        assert engine.backend == "cpu"
        assert "WARNING" in capsys.readouterr().err

    def test_a_model_that_lands_off_the_gpu_is_refused(self, monkeypatch, gguf):
        """CUDA was registered but the load fell back: the same silent slowdown."""
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp(cuda=True, loads_on="cpu"))
        with pytest.raises(SystemExit, match="not CUDA"):
            worker.TranscribeCppEngine(gguf)
        assert stub.models[0].closed

    def test_missing_binding_names_transcribe_library(self, monkeypatch, gguf):
        worker = _load_worker()
        monkeypatch.setitem(sys.modules, "transcribe_cpp", None)  # import fails
        with pytest.raises(SystemExit) as info:
            worker.TranscribeCppEngine(gguf)
        assert "TRANSCRIBE_LIBRARY" in str(info.value)
        assert "transcribe-cpp==0.2.4" in str(info.value)
        assert "TRANSCRIBE_BUILD_SHARED=ON" in str(info.value)

    def test_missing_gguf_fails_before_the_library_is_touched(self, monkeypatch, tmp_path):
        worker = _load_worker()
        monkeypatch.setitem(sys.modules, "transcribe_cpp", None)
        with pytest.raises(SystemExit, match="does not exist"):
            worker.TranscribeCppEngine(tmp_path / "absent.gguf")


class TestLoadAndWarmUp:
    def test_loads_once_and_warms_up_exactly_once_before_serving(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        engine = worker.TranscribeCppEngine(gguf)

        assert len(stub.models) == 1 and len(stub.models[0].sessions) == 1
        assert len(stub.runs) == 1, "exactly one warm-up call at start-up"
        warm = stub.runs[0]
        assert warm.dtype == np.float32 and len(warm) == worker.SAMPLE_RATE  # 1 s at 16 kHz

        engine.transcribe(_speech())
        engine.transcribe(_speech())
        # Still one model/session; the slow first-call cost landed on the warm-up.
        assert len(stub.models) == 1 and len(stub.models[0].sessions) == 1
        assert len(stub.runs) == 3
        assert stub.run_seconds[0] == 55.0 and max(stub.run_seconds[1:]) < 1.0

    def test_n_ctx_reaches_the_session_and_sets_the_length_guard(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        engine = worker.TranscribeCppEngine(gguf, n_ctx=512)
        assert stub.models[0].sessions[0].n_ctx == 512
        assert engine.max_samples == 17920 * 16

        with pytest.raises(ValueError, match="--n-ctx"):
            engine.transcribe(_speech(seconds=20.0))
        assert len(stub.runs) == 1, "an over-long utterance must not reach the model"

    def test_default_n_ctx_is_1024(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        worker.TranscribeCppEngine(gguf)
        assert worker.CANARY_GGUF_N_CTX == 1024
        assert stub.models[0].sessions[0].n_ctx == 1024


class TestTranscribe:
    def test_audio_reaches_the_model_as_16k_mono_float32(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        engine = worker.TranscribeCppEngine(gguf)
        # A (frames, 1) float64 capture, the shape sounddevice hands back.
        column = _speech(seconds=1.5, dtype=np.float64).reshape(-1, 1)

        result = engine.transcribe(column)

        sent = stub.runs[-1]
        assert sent.dtype == np.float32 and sent.ndim == 1 and sent.flags["C_CONTIGUOUS"]
        assert len(sent) == int(1.5 * 16000)
        # Quiet laptop-mic level is normalised up, as for the other engines.
        assert float(np.abs(sent).max()) == pytest.approx(0.55, abs=0.01)
        assert result == [("Pick up the marker.", worker.CANARY_CONFIDENCE)]
        assert worker.CANARY_CONFIDENCE == 1.0

    def test_empty_transcript_and_empty_audio_give_no_result(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp(text="   "))
        engine = worker.TranscribeCppEngine(gguf)
        assert engine.transcribe(_speech()) == []
        assert engine.transcribe(np.zeros(0, np.float32)) == []
        assert len(stub.runs) == 2  # warm-up + one real call; empty audio never sent

    def test_close_releases_session_then_model(self, monkeypatch, gguf):
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        engine = worker.TranscribeCppEngine(gguf)
        engine.close()
        assert stub.models[0].sessions[0].closed and stub.models[0].closed


class TestCommandLine:
    def test_canary_gguf_requires_gguf(self, capsys):
        worker = _load_worker()
        with pytest.raises(SystemExit) as info:
            worker.main(["--asr", "canary-gguf", "--wav", "x.wav"])
        assert info.value.code == 2
        assert "--gguf" in capsys.readouterr().err

    def test_wav_mode_end_to_end_with_a_44k_stereo_file(self, monkeypatch, gguf, tmp_path, capsys):
        """--asr canary-gguf --gguf --n-ctx --wav: the same push-JSON lines as
        the other engines, audio resampled to 16 kHz before the model sees it."""
        worker = _load_worker()
        stub = _install(monkeypatch, _StubTranscribeCpp())
        rate = 44100
        mono = (0.3 * 32767 * np.sin(2 * np.pi * 300 * np.arange(int(rate * 1.2)) / rate)).astype("<i2")
        with wave.open(str(tmp_path / "cmd.wav"), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(np.repeat(mono[:, None], 2, axis=1).tobytes())

        code = worker.main(["--asr", "canary-gguf", "--gguf", str(gguf), "--n-ctx", "1024",
                            "--wav", str(tmp_path / "cmd.wav")])

        assert code == 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
        assert {"text": "Pick up the marker.", "confidence": 1.0} in lines
        assert any(line.get("event") == "file" for line in lines)
        sent = stub.runs[-1]
        assert len(sent) == pytest.approx(1.2 * 16000, abs=2)
        assert stub.models[0].sessions[0].n_ctx == 1024

    def test_allow_cpu_flag_reaches_the_engine(self, monkeypatch, gguf, tmp_path, capsys):
        worker = _load_worker()
        _install(monkeypatch, _StubTranscribeCpp(cuda=False))
        wav = tmp_path / "c.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((_speech(amplitude=0.3) * 32767).astype("<i2").tobytes())

        with pytest.raises(SystemExit, match="NO CUDA backend"):
            worker.main(["--asr", "canary-gguf", "--gguf", str(gguf), "--wav", str(wav)])
        assert worker.main(["--asr", "canary-gguf", "--gguf", str(gguf), "--allow-cpu",
                            "--wav", str(wav)]) == 0
        assert '"confidence": 1.0' in capsys.readouterr().out

    def test_serve_passes_engine_options_through(self, monkeypatch, gguf):
        """--serve builds the engine with --gguf/--n-ctx/--allow-cpu intact."""
        worker = _load_worker()
        seen: dict = {}

        class _Stop(Exception):
            pass

        def build_engine(*args, **kwargs):
            seen.update(kwargs)
            raise _Stop

        monkeypatch.setattr(worker, "build_engine", build_engine)
        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            default=types.SimpleNamespace(device=(None, None))))
        with pytest.raises(_Stop):
            worker.main(["--serve", "--host", "127.0.0.1", "--port", "0",
                         "--asr", "canary-gguf", "--gguf", str(gguf), "--n-ctx", "768",
                         "--allow-cpu"])
        assert seen == {"gguf": str(gguf), "n_ctx": 768, "allow_cpu": True}

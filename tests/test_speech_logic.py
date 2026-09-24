"""Phase 8 tests: voice front-end.

No Isaac Sim, no microphone. The subprocess recogniser is exercised against the
real worker script in ``--stdin`` mode, so the process boundary, the JSON protocol
and the queue behaviour are all genuinely tested -- only the audio capture is
absent.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from mfw.language.speech import (
    ScriptedRecognizer,
    SpeechError,
    TextRecognizer,
    Transcript,
    VoiceCommandLoop,
    WhisperSubprocessRecognizer,
)

pytestmark = pytest.mark.phase8

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "speech_worker.py"


class TestTextRecognizer:
    def test_reads_lines_as_transcripts(self):
        recognizer = TextRecognizer(stream=io.StringIO("pick the can\nplace it\n"), prompt="")
        assert recognizer.listen().text == "pick the can"
        assert recognizer.listen().text == "place it"
        assert recognizer.listen() is None

    def test_skips_blank_lines(self):
        recognizer = TextRecognizer(stream=io.StringIO("\n"), prompt="")
        assert recognizer.listen() is None


class TestScriptedRecognizer:
    def test_replays_in_order_then_ends(self):
        recognizer = ScriptedRecognizer(["observe", "pick the can"])
        assert recognizer.listen().text == "observe"
        assert recognizer.listen().text == "pick the can"
        assert recognizer.listen() is None


class TestVoiceCommandLoop:
    def test_executes_each_command_once(self):
        handled: list[str] = []
        loop = VoiceCommandLoop(
            ScriptedRecognizer(["observe", "pick the can", "place it"]), handled.append
        )
        assert loop.run() == 3
        assert handled == ["observe", "pick the can", "place it"]

    def test_commands_are_strictly_sequential(self):
        """Overlapping commands on a physical arm are unsafe.

        Records entry and exit so an interleaving would be visible as nesting.
        """
        events: list[str] = []

        def handler(text: str) -> None:
            events.append(f"start:{text}")
            events.append(f"end:{text}")

        VoiceCommandLoop(ScriptedRecognizer(["a", "b"]), handler).run()
        assert events == ["start:a", "end:a", "start:b", "end:b"]

    def test_stop_word_ends_the_loop_without_executing_it(self):
        handled: list[str] = []
        loop = VoiceCommandLoop(
            ScriptedRecognizer(["observe", "quit", "pick the can"]), handled.append
        )
        assert loop.run() == 1
        assert handled == ["observe"]

    def test_low_confidence_speech_is_not_executed(self):
        """A misheard manipulation command moves a real arm."""

        class Unsure(ScriptedRecognizer):
            def listen(self, timeout_s=None):
                if not self._remaining:
                    return None
                return Transcript(self._remaining.pop(0), 0.2, "test")

        handled: list[str] = []
        flagged: list[Transcript] = []
        loop = VoiceCommandLoop(
            Unsure(["pick the can"]),
            handled.append,
            min_confidence=0.5,
            on_low_confidence=flagged.append,
        )
        assert loop.run() == 0
        assert handled == []
        assert len(flagged) == 1 and flagged[0].text == "pick the can"

    def test_max_commands_is_respected(self):
        handled: list[str] = []
        loop = VoiceCommandLoop(ScriptedRecognizer(["a", "b", "c"]), handled.append)
        assert loop.run(max_commands=2) == 2
        assert handled == ["a", "b"]

    def test_recognizer_is_stopped_even_if_the_handler_raises(self):
        """A crashing command must still release the audio device."""

        class Tracking(ScriptedRecognizer):
            def __init__(self, utterances):
                super().__init__(utterances)
                self.stopped = False

            def stop(self):
                self.stopped = True

        recognizer = Tracking(["boom"])

        def explode(_text):
            raise RuntimeError("handler failed")

        loop = VoiceCommandLoop(recognizer, explode)
        with pytest.raises(RuntimeError):
            loop.run()
        assert recognizer.stopped

    def test_transcripts_iterator_does_not_execute(self):
        loop = VoiceCommandLoop(ScriptedRecognizer(["a", "b"]), lambda _t: pytest.fail("executed"))
        assert [t.text for t in loop.transcripts()] == ["a", "b"]


class TestSubprocessIsolation:
    def test_worker_script_exists(self):
        assert WORKER.is_file(), f"speech worker missing at {WORKER}"

    def test_missing_worker_is_reported_clearly(self):
        recognizer = WhisperSubprocessRecognizer(worker_script=REPO_ROOT / "no_such_worker.py")
        with pytest.raises(SpeechError, match="not found"):
            recognizer.start()

    def test_transcribes_through_a_real_subprocess(self):
        """Exercises the actual process boundary and JSON protocol.

        Uses the worker's ``--stdin`` mode so no audio stack is needed, but the
        subprocess, the pipe and the line protocol are all real -- which is the
        part that has to work, since speech backends crash Isaac Sim's interpreter
        on Windows.
        """
        recognizer = WhisperSubprocessRecognizer(
            worker_script=WORKER, python_executable=sys.executable
        )
        # Route the worker into stdin mode by extending the command it is launched with.
        original_start = recognizer.start

        import subprocess
        import threading

        def start_in_stdin_mode() -> None:
            recognizer._process = subprocess.Popen(
                [sys.executable, str(WORKER), "--stdin"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            recognizer._stop.clear()
            recognizer._reader = threading.Thread(target=recognizer._read_loop, daemon=True)
            recognizer._reader.start()

        recognizer.start = start_in_stdin_mode  # type: ignore[method-assign]
        recognizer.start()
        try:
            assert recognizer._process is not None
            recognizer._process.stdin.write("pick the can\n")
            recognizer._process.stdin.flush()

            transcript = recognizer.listen(timeout_s=15.0)
            assert transcript is not None, "no transcript came back from the worker"
            assert transcript.text == "pick the can"
            assert transcript.source == "whisper"
        finally:
            recognizer.start = original_start  # type: ignore[method-assign]
            recognizer.stop()

    def test_non_json_worker_output_is_ignored(self):
        """A worker printing diagnostics must not break the caller."""
        recognizer = WhisperSubprocessRecognizer(worker_script=WORKER)

        class FakeStdout:
            def __iter__(self):
                return iter(["loading model...\n", '{"text": "observe", "confidence": 0.9}\n'])

        class FakeProcess:
            stdout = FakeStdout()

        recognizer._process = FakeProcess()  # type: ignore[assignment]
        recognizer._read_loop()

        assert recognizer._queue.qsize() == 1
        assert recognizer._queue.get().text == "observe"


# ---------------------------------------------------------------------------
# TcpSpeechRecognizer against a real in-thread socket server
# ---------------------------------------------------------------------------


class _LineServer:
    """A loopback TCP server that plays one script per accepted connection.

    Each script is a list of byte strings sent in order; after the last one the
    connection is closed (the server-side EOF the client must notice). With
    ``keep_listening=False`` the listening socket is closed after the final
    script too, so reconnection is refused.
    """

    def __init__(self, scripts: list[list[bytes]], keep_listening: bool = True) -> None:
        import socket
        import threading

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self.port = self._listener.getsockname()[1]
        self.accepted = 0
        self._scripts = list(scripts)
        self._keep_listening = keep_listening
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import time

        while self._scripts:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self.accepted += 1
            script = self._scripts.pop(0)
            with conn:
                for chunk in script:
                    conn.sendall(chunk)
                    time.sleep(0.05)
        if not self._keep_listening:
            self._listener.close()

    def close(self) -> None:
        try:
            self._listener.close()
        except OSError:
            pass


def _within(seconds: float, fn):
    """Run ``fn`` on a daemon thread; fail (rather than hang) if it blocks.

    A regression of the disconnect bug is an indefinite block, which must show
    up as a failure, not as a suite that never finishes.
    """
    import threading

    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), f"still blocked after {seconds} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class TestTcpSpeechRecognizer:
    def test_a_line_split_across_sends_is_reassembled(self):
        from mfw.language.speech import TcpSpeechRecognizer

        server = _LineServer([[b'{"text": "pick up', b' the can", "confidence": 0.9}\n']],
                             keep_listening=True)
        recognizer = TcpSpeechRecognizer(port=server.port, reconnect_attempts=0)
        recognizer.start()
        try:
            transcript = recognizer.listen(timeout_s=5.0)
        finally:
            recognizer.stop()
            server.close()
        assert transcript == Transcript("pick up the can", 0.9, "speech-server")

    def test_server_disconnect_raises_instead_of_hanging(self):
        """The old client blocked forever in listen(None) once the server closed.

        Transcripts that arrived before the close are still delivered first.
        """
        import threading
        import time

        from mfw.language.speech import TcpSpeechRecognizer

        server = _LineServer([[b'{"text": "observe", "confidence": 1.0}\n']],
                             keep_listening=False)
        recognizer = TcpSpeechRecognizer(
            port=server.port, reconnect_attempts=1, reconnect_backoff_s=0.05,
            connect_timeout_s=1.0,
        )
        recognizer.start()
        outcome: dict = {}

        def blocking_listen() -> None:
            try:
                outcome["first"] = recognizer.listen(timeout_s=None)
                outcome["second"] = recognizer.listen(timeout_s=None)
            except SpeechError as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=blocking_listen, daemon=True)
        started = time.monotonic()
        worker.start()
        worker.join(timeout=10.0)
        recognizer.stop()
        server.close()

        assert not worker.is_alive(), "listen(None) is still blocked after the server closed"
        assert time.monotonic() - started < 5.0
        assert outcome["first"].text == "observe"
        assert "second" not in outcome
        message = str(outcome["error"])
        assert "lost the speech server" in message
        assert "closed the connection" in message
        assert "speech_worker.py --serve" in message

    def test_reconnects_when_the_server_is_back(self):
        """A resident server returns to accept() after a drop; the client resumes."""
        from mfw.language.speech import TcpSpeechRecognizer

        server = _LineServer(
            [
                [b'{"text": "pick up the can", "confidence": 1.0}\n'],
                [b'{"text": "place it", "confidence": 1.0}\n'],
            ],
            keep_listening=True,
        )
        events: list[str] = []
        recognizer = TcpSpeechRecognizer(
            port=server.port, reconnect_attempts=3, reconnect_backoff_s=0.05
        )
        recognizer.on_event = lambda name, fields: events.append(name)
        recognizer.start()
        try:
            first = recognizer.listen(timeout_s=5.0)
            second = recognizer.listen(timeout_s=5.0)
        finally:
            recognizer.stop()
            server.close()

        assert (first.text, second.text) == ("pick up the can", "place it")
        assert server.accepted == 2
        assert "reconnected" in events

    def test_drain_cannot_hide_a_disconnect(self):
        """VoiceCommandLoop drains after every command; the drop must survive that."""
        import time

        from mfw.language.speech import TcpSpeechRecognizer

        server = _LineServer([[b'{"text": "observe", "confidence": 1.0}\n']],
                             keep_listening=False)
        recognizer = TcpSpeechRecognizer(port=server.port, reconnect_attempts=0)
        recognizer.start()
        try:
            deadline = time.monotonic() + 5.0
            while recognizer.connected and time.monotonic() < deadline:
                time.sleep(0.02)
            recognizer.drain()
            with pytest.raises(SpeechError, match="lost the speech server"):
                _within(10.0, lambda: recognizer.listen(timeout_s=None))
        finally:
            recognizer.stop()
            server.close()

    def test_voice_loop_ends_with_an_error_when_the_server_dies(self):
        """End to end: the loop that used to hang now reports and stops."""
        from mfw.language.speech import TcpSpeechRecognizer

        server = _LineServer([[b'{"text": "observe", "confidence": 1.0}\n']],
                             keep_listening=False)
        handled: list[str] = []
        recognizer = TcpSpeechRecognizer(
            port=server.port, reconnect_attempts=1, reconnect_backoff_s=0.05,
            connect_timeout_s=1.0,
        )
        loop = VoiceCommandLoop(recognizer, handled.append)
        try:
            with pytest.raises(SpeechError, match="lost the speech server"):
                _within(10.0, loop.run)
        finally:
            server.close()
        assert handled == ["observe"]


# ---------------------------------------------------------------------------
# speech_worker.py --wav: offline audio path (no model is loaded)
# ---------------------------------------------------------------------------


def _load_worker():
    import importlib.util

    spec = importlib.util.spec_from_file_location("speech_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, samples, rate: int, channels: int = 1, width: int = 2) -> None:
    import wave

    import numpy as np

    data = np.asarray(samples, dtype=np.float64)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    if width == 1:
        raw = np.clip(data * 127 + 128, 0, 255).astype(np.uint8).tobytes()
    elif width == 2:
        raw = np.clip(data * 32767, -32768, 32767).astype("<i2").tobytes()
    else:
        raw = np.clip(data * 2147483647, -2147483648, 2147483647).astype("<i4").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(raw)


class TestWavInput:
    def test_resamples_stereo_44k_to_16k_mono_float32(self, tmp_path):
        import numpy as np

        worker = _load_worker()
        rate, seconds, tone = 44100, 0.5, 440.0
        t = np.arange(int(rate * seconds)) / rate
        _write_wav(tmp_path / "a.wav", 0.5 * np.sin(2 * np.pi * tone * t), rate, channels=2)

        samples = worker.load_wav(tmp_path / "a.wav")

        assert samples.dtype == np.float32 and samples.ndim == 1
        assert len(samples) == pytest.approx(worker.SAMPLE_RATE * seconds, abs=2)
        assert float(np.abs(samples).max()) == pytest.approx(0.5, abs=0.01)
        # The tone survives resampling: count zero crossings (2 per cycle).
        crossings = np.count_nonzero(np.diff(np.signbit(samples)))
        assert crossings == pytest.approx(2 * tone * seconds, abs=3)

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_reads_8_16_and_32_bit_pcm(self, tmp_path, width):
        import numpy as np

        worker = _load_worker()
        t = np.arange(1600) / 16000
        _write_wav(tmp_path / "b.wav", 0.25 * np.sin(2 * np.pi * 200 * t), 16000, width=width)

        samples = worker.load_wav(tmp_path / "b.wav")
        assert len(samples) == 1600
        assert float(np.abs(samples).max()) == pytest.approx(0.25, abs=0.02)

    def test_rejects_a_file_that_is_not_pcm_wav(self, tmp_path):
        worker = _load_worker()
        bogus = tmp_path / "c.wav"
        bogus.write_bytes(b"ID3\x03not a wave file at all")
        with pytest.raises(ValueError, match="PCM WAV"):
            worker.load_wav(bogus)

    def test_wav_mode_feeds_the_engine_and_emits_protocol_lines(
        self, tmp_path, monkeypatch, capsys
    ):
        """The --wav plumbing with a stand-in engine: no ASR model is loaded.

        Asserts what the worker did with the file (16 kHz float32 into
        ``transcribe``, one JSON line per result) -- not recognition quality.
        """
        import json

        import numpy as np

        worker = _load_worker()
        seen: list = []

        class RecordingEngine:
            def transcribe(self, samples):
                seen.append(samples)
                return [("pick up the red block", 0.87)]

        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: RecordingEngine())
        t = np.arange(22050) / 22050
        _write_wav(tmp_path / "cmd.wav", 0.3 * np.sin(2 * np.pi * 300 * t), 22050)

        assert worker.main(["--wav", str(tmp_path / "cmd.wav")]) == 0

        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
        transcripts = [line for line in lines if "text" in line]
        assert transcripts == [{"text": "pick up the red block", "confidence": 0.87}]
        assert any(line.get("event") == "file" for line in lines)
        assert len(seen) == 1 and seen[0].dtype == np.float32
        assert len(seen[0]) == pytest.approx(16000, abs=2)

    def test_missing_wav_is_a_clear_error(self, tmp_path, monkeypatch, capsys):
        worker = _load_worker()
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: None)
        assert worker.main(["--wav", str(tmp_path / "nope.wav")]) == 2
        assert "nope.wav" in capsys.readouterr().err


class TestServeSpeechVenvPath:
    def _load(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "serve_speech_under_test", REPO_ROOT / "scripts" / "serve_speech.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_default_is_derived_from_the_home_directory(self, monkeypatch):
        monkeypatch.delenv("MFW_NEMO_SITE_PACKAGES", raising=False)
        module = self._load()
        resolved = module.resolve_site_packages(None)
        assert resolved == module.default_site_packages()
        assert resolved.is_relative_to(Path.home() / "canary-venv")
        if sys.platform == "win32":
            # Identical to the constant it replaced on the development machine.
            assert resolved == Path.home() / "canary-venv" / "Lib" / "site-packages"

    def test_environment_variable_and_flag_override_it(self, monkeypatch, tmp_path):
        module = self._load()
        monkeypatch.setenv("MFW_NEMO_SITE_PACKAGES", str(tmp_path / "env"))
        assert module.resolve_site_packages(None) == tmp_path / "env"
        assert module.resolve_site_packages(str(tmp_path / "flag")) == tmp_path / "flag"

    def test_missing_directory_fails_with_the_fix(self, tmp_path, capsys):
        module = self._load()
        assert module.main(["--nemo-site-packages", str(tmp_path / "absent"), "--stdin"]) == 2
        err = capsys.readouterr().err
        assert "MFW_NEMO_SITE_PACKAGES" in err and "absent" in err


# ---------------------------------------------------------------------------
# serve_speech.py: --device cuda with a torch that sees no GPU
# ---------------------------------------------------------------------------


def _load_serve_speech():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "serve_speech_cuda_under_test", REPO_ROOT / "scripts" / "serve_speech.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_torch(available: bool):
    import types

    torch = types.ModuleType("torch")
    torch.__version__ = "2.12.0+cu130"
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.cuda = types.SimpleNamespace(is_available=lambda: available)
    return torch


class TestServeSpeechCudaGuard:
    @pytest.fixture
    def worker_calls(self, monkeypatch):
        """A stand-in speech_worker module: records the args, loads nothing."""
        import types

        calls: list = []
        fake = types.ModuleType("speech_worker")
        fake.main = lambda argv: calls.append(list(argv)) or 0
        monkeypatch.setitem(sys.modules, "speech_worker", fake)
        monkeypatch.setattr(sys, "path", list(sys.path))
        return calls

    def test_cuda_requested_but_unavailable_exits_with_the_working_command(
        self, tmp_path, monkeypatch, capsys, worker_calls
    ):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
        module = _load_serve_speech()
        code = module.main(["--nemo-site-packages", str(tmp_path), "--asr", "canary",
                            "--device", "cuda", "--mic", "1", "--serve"])

        assert code != 0 and code == module.EXIT_NO_CUDA
        assert worker_calls == [], "the model must not start loading"
        err = capsys.readouterr().err
        assert "2.12.0+cu130" in err and "CUDA unavailable" in err
        assert "cu128" in err and "577.03" in err
        assert str(module.canary_venv_python()) in err
        assert "speech_worker.py" in err and "--device cuda" in err

    def test_equals_form_is_recognised_too(self, tmp_path, monkeypatch, capsys, worker_calls):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
        module = _load_serve_speech()
        assert module.main(["--nemo-site-packages", str(tmp_path), "--asr=canary",
                            "--device=cuda", "--serve"]) == module.EXIT_NO_CUDA
        assert worker_calls == []

    def test_cuda_available_passes_through(self, tmp_path, monkeypatch, worker_calls):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
        module = _load_serve_speech()
        rest = ["--asr", "canary", "--device", "cuda", "--serve"]
        assert module.main(["--nemo-site-packages", str(tmp_path), *rest]) == 0
        assert worker_calls == [rest]

    @pytest.mark.parametrize(
        "rest",
        [
            ["--asr", "canary", "--device", "cpu", "--serve"],
            ["--asr", "whisper", "--device", "cuda", "--serve"],  # CTranslate2, not torch
        ],
    )
    def test_no_torch_check_when_torch_is_not_what_runs_on_the_gpu(
        self, tmp_path, monkeypatch, worker_calls, rest
    ):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
        module = _load_serve_speech()
        assert module.main(["--nemo-site-packages", str(tmp_path), *rest]) == 0
        assert worker_calls == [rest]


# ---------------------------------------------------------------------------
# speech_worker.py --serve: the port is claimed before the model and the mic
# ---------------------------------------------------------------------------


@pytest.fixture
def occupied_port():
    """A listening socket owned by another thread, like a running speech server."""
    import socket
    import threading

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    stop = threading.Event()

    def serve() -> None:
        holder.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = holder.accept()
                conn.close()
            except OSError:
                continue

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield holder.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=2.0)
        holder.close()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Sentinel(Exception):
    pass


def _fake_sounddevice(opened: list):
    import types

    sd = types.ModuleType("sounddevice")

    def refuse(*args, **kwargs):
        opened.append("microphone")
        raise AssertionError("the microphone must not be touched")

    sd.check_input_settings = refuse
    sd.InputStream = refuse
    sd.rec = refuse
    sd.query_devices = lambda *a, **k: {"name": "fake mic"}
    sd.default = types.SimpleNamespace(device=(None, None))
    return sd


class TestServePortClaimedFirst:
    def test_port_in_use_exits_before_loading_the_model_or_the_mic(
        self, occupied_port, monkeypatch, capsys
    ):
        worker = _load_worker()
        loaded: list = []
        opened: list = []
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: loaded.append(a))
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice(opened))

        code = worker.main(["--serve", "--asr", "canary", "--device", "cuda",
                            "--host", "127.0.0.1", "--port", str(occupied_port), "--mic", "1"])

        assert code == worker.EXIT_PORT_IN_USE and code != 0
        assert loaded == [], "a duplicate process must not load a second model"
        assert opened == [], "a duplicate process must not open the microphone"
        err = capsys.readouterr().err
        assert f"127.0.0.1:{occupied_port} is already in use" in err
        assert "--voice-server" in err and "Nothing was loaded" in err

    def test_serve_forever_alone_also_claims_the_port_first(
        self, occupied_port, monkeypatch
    ):
        worker = _load_worker()
        loaded: list = []
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: loaded.append(a))
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice([]))
        code = worker.serve_forever("127.0.0.1", occupied_port, "whisper", "base.en",
                                    "cpu", 4.0, 0.0)
        assert code == worker.EXIT_PORT_IN_USE and loaded == []

    def test_free_port_is_bound_before_the_model_loads_and_listens_only_after(
        self, monkeypatch
    ):
        import socket

        worker = _load_worker()
        port = _free_port()
        observed: dict = {}

        def build_engine(*args, **kwargs):
            # The port is already ours: a second bind must fail ...
            try:
                worker.bind_server_socket("127.0.0.1", port).close()
                observed["second_bind"] = "succeeded"
            except worker.PortInUseError:
                observed["second_bind"] = "refused"
            # ... but nothing is listening yet, so a client probe is refused and
            # "is the server up?" still means "is the model loaded?".
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                observed["probe"] = "accepted"
            except OSError:
                observed["probe"] = "refused"
            raise _Sentinel

        monkeypatch.setattr(worker, "build_engine", build_engine)
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice([]))
        with pytest.raises(_Sentinel):
            worker.main(["--serve", "--host", "127.0.0.1", "--port", str(port)])

        assert observed == {"second_bind": "refused", "probe": "refused"}
        # Released on the way out: the port can be bound again.
        worker.bind_server_socket("127.0.0.1", port).close()

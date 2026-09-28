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


# ---------------------------------------------------------------------------
# speech_worker.py: band-limited resampling (native-rate mics, --wav files)
# ---------------------------------------------------------------------------


def _dominant_hz(samples, rate: int) -> float:
    import numpy as np

    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1.0 / rate)[int(np.argmax(spectrum))])


def _tone(hz: float, rate: int, seconds: float, amplitude: float = 0.5):
    import numpy as np

    t = np.arange(round(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TestResampler:
    @pytest.mark.parametrize("use_scipy", [False, True], ids=["numpy", "scipy"])
    @pytest.mark.parametrize("rate", [48000, 44100, 22050, 32000])
    def test_tone_frequency_and_length_are_kept(self, rate, use_scipy):
        import numpy as np

        if use_scipy:
            pytest.importorskip("scipy.signal")
        worker = _load_worker()
        x = _tone(440.0, rate, 1.3)

        y = worker.resample_audio(x, rate, 16000, use_scipy=use_scipy)

        assert y.dtype == np.float32 and y.ndim == 1
        # resample_poly's length contract: ceil(n * up / down).
        assert len(y) == -(-len(x) * 16000 // rate)
        assert _dominant_hz(y, 16000) == pytest.approx(440.0, abs=2.0)
        # Amplitude survives in the steady-state middle (edges ramp by design).
        middle = y[len(y) // 4 : 3 * len(y) // 4]
        assert float(np.abs(middle).max()) == pytest.approx(0.5, abs=0.01)

    def test_numpy_fallback_matches_scipy(self):
        import numpy as np

        pytest.importorskip("scipy.signal")
        worker = _load_worker()
        rng = np.random.default_rng(7)
        x = rng.normal(0, 0.1, 44100).astype(np.float32)
        a = worker.resample_audio(x, 44100, 16000, use_scipy=False)
        b = worker.resample_audio(x, 44100, 16000, use_scipy=True)
        assert a.shape == b.shape
        assert float(np.abs(a - b).max()) < 1e-5

    @pytest.mark.parametrize("use_scipy", [False, True], ids=["numpy", "scipy"])
    def test_out_of_band_noise_is_filtered_not_folded_into_speech(self, use_scipy):
        """A 12 kHz whine at 48 kHz must not come out as a 4 kHz tone.

        That is exactly what decimation or linear interpolation does (12 kHz
        aliases to 16 - 12 = 4 kHz, inside the speech band), and a speech-LLM
        reads such tones as phonemes.
        """
        import numpy as np

        if use_scipy:
            pytest.importorskip("scipy.signal")
        worker = _load_worker()
        whine = _tone(12000.0, 48000, 1.0)

        y = worker.resample_audio(whine, 48000, 16000, use_scipy=use_scipy)
        naive = whine[::3]  # plain decimation, for contrast

        middle = slice(len(y) // 4, 3 * len(y) // 4)
        assert float(np.sqrt(np.mean(y[middle] ** 2))) < 0.01 * 0.354  # > 40 dB down
        assert float(np.sqrt(np.mean(naive[middle] ** 2))) > 0.3  # the failure mode

    def test_same_rate_and_empty_input_pass_through(self):
        import numpy as np

        worker = _load_worker()
        x = _tone(300.0, 16000, 0.1)
        assert np.array_equal(worker.resample_audio(x, 16000, 16000), x)
        assert worker.resample_audio(np.zeros(0, np.float32), 48000, 16000).size == 0
        with pytest.raises(ValueError):
            worker.resample_audio(x, 0, 16000)

    def test_load_wav_uses_the_band_limited_resampler(self, tmp_path):
        """A 48 kHz WAV carrying an out-of-band whine loads as clean 16 kHz audio."""
        import numpy as np

        worker = _load_worker()
        mix = _tone(300.0, 48000, 1.0, 0.3) + _tone(12000.0, 48000, 1.0, 0.3)
        _write_wav(tmp_path / "w.wav", mix, 48000)
        samples = worker.load_wav(tmp_path / "w.wav")
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        freqs = np.fft.rfftfreq(len(samples), 1 / 16000)
        at_300 = spectrum[np.argmin(np.abs(freqs - 300))]
        at_4k = spectrum[np.argmin(np.abs(freqs - 4000))]  # where 12 kHz would alias
        assert at_4k < 0.01 * at_300


# ---------------------------------------------------------------------------
# speech_worker.py: a microphone that refuses 16 kHz is captured natively
# ---------------------------------------------------------------------------


class _NativeRateMic:
    """A stand-in ``sounddevice`` for a mic that only clocks its native rate.

    Models what the hardware does, not what the worker hopes for:

    * 16 kHz is refused both by ``check_input_settings`` and when a stream is
      actually opened (PortAudio's "Invalid sample rate [PaErrorCode -9997]").
    * The reported ``default_samplerate`` can be wrong (``reported_rate``).
    * Each opened stream plays the next entry of ``sessions``: ``"speech"``
      (room noise, a spoken burst -- a tone -- with one DROPPED block in the
      middle flagged as an overflow, then silence), ``"unplugged"`` (the open
      fails like a pulled USB cable) or ``"interrupt"`` (Ctrl+C).
    * When a speech stream runs out, ``read`` raises like a dead device.
    """

    def __init__(self, native_rate=48000, accepted=None, reported_rate=None,
                 tone_hz=300.0, sessions=("speech", "unplugged")):
        import types

        self.native_rate = native_rate
        self.accepted = set(accepted) if accepted is not None else {native_rate}
        self.reported_rate = reported_rate or native_rate
        self.tone_hz = tone_hz
        self.sessions = list(sessions)
        self.opened_rates: list = []
        self.rec_rates: list = []
        self.default = types.SimpleNamespace(device=(None, None))

    # -- sounddevice surface -------------------------------------------------
    def check_input_settings(self, device=None, channels=1, samplerate=None, dtype=None):
        if int(samplerate) not in self.accepted:
            raise RuntimeError("Error opening InputStream: Invalid sample rate [PaErrorCode -9997]")

    def query_devices(self, device=None, kind=None):
        return {"name": "USB PnP Sound Device", "max_input_channels": 1,
                "default_samplerate": float(self.reported_rate)}

    def _noise(self, count, seed):
        import numpy as np

        return np.random.default_rng(seed).normal(0.0, 0.002, count).astype(np.float32)

    def rec(self, frames, samplerate=None, channels=1, dtype="float32"):
        self.check_input_settings(samplerate=samplerate)
        self.rec_rates.append(samplerate)
        return self._noise(int(frames), 1).reshape(-1, 1)

    def wait(self):
        return None

    def InputStream(self, samplerate=None, channels=1, dtype="float32", blocksize=None,
                    device=None):
        self.check_input_settings(samplerate=samplerate)
        kind = self.sessions.pop(0) if self.sessions else "unplugged"
        if kind == "unplugged":
            raise RuntimeError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
        if kind == "interrupt":
            raise KeyboardInterrupt
        self.opened_rates.append(samplerate)
        return _ScriptedStream(self._speech_signal(int(samplerate)), int(blocksize))

    # -- the recording ----------------------------------------------------------
    LEAD_S, SPEECH_S, TAIL_S = 0.6, 1.0, 1.0

    def _speech_signal(self, rate):
        import numpy as np

        lead, speech, tail = (round(s * rate) for s in (self.LEAD_S, self.SPEECH_S, self.TAIL_S))
        t = np.arange(speech) / rate
        voiced = 0.3 * np.sin(2 * np.pi * self.tone_hz * t).astype(np.float32)
        voiced += self._noise(speech, 3)
        # One lost block mid-word: the driver overran and the samples are gone.
        block = int(0.05 * rate)
        drop = lead + speech // 2
        signal = np.concatenate([self._noise(lead, 2), voiced, self._noise(tail, 4)])
        signal[drop : drop + block] = 0.0
        return signal, (drop, drop + block)


class _ScriptedStream:
    def __init__(self, scripted, blocksize):
        (self.signal, self.dropped) = scripted
        self.blocksize = blocksize
        self.pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        if self.pos + frames > len(self.signal):
            raise RuntimeError("Error reading stream: Unanticipated host error [PaErrorCode -9999]")
        chunk = self.signal[self.pos : self.pos + frames]
        overflowed = self.pos <= self.dropped[0] < self.pos + frames
        self.pos += frames
        return chunk.reshape(-1, 1).copy(), overflowed


class _RecordingEngine:
    def __init__(self):
        self.seen: list = []

    def transcribe(self, samples):
        self.seen.append(samples)
        return [("pick up the marker", 1.0)]


def _protocol_lines(text: str) -> list:
    import json

    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestNativeRateCapture:
    def test_choose_rate_prefers_16k_when_accepted(self):
        worker = _load_worker()
        mic = _NativeRateMic(accepted={16000, 48000})
        assert worker.choose_capture_rate(mic, 3) == 16000

    def test_choose_rate_falls_back_to_the_native_rate(self):
        worker = _load_worker()
        assert worker.choose_capture_rate(_NativeRateMic(native_rate=44100), 3) == 44100

    def test_a_misreported_default_rate_still_finds_one_that_opens(self):
        worker = _load_worker()
        mic = _NativeRateMic(native_rate=48000, accepted={48000}, reported_rate=44100)
        assert worker.choose_capture_rate(mic, None) == 48000

    def test_a_device_that_accepts_nothing_usable_is_reported(self):
        worker = _load_worker()
        mic = _NativeRateMic(accepted={8000})  # below 16 kHz: never used
        with pytest.raises(worker.CaptureRateError, match="16000 Hz"):
            worker.choose_capture_rate(mic, 3)

    @pytest.mark.parametrize("native_rate", [48000, 44100])
    def test_mic_mode_captures_natively_and_hands_the_model_16k(
        self, native_rate, monkeypatch, capsys
    ):
        """End to end through main(): refusal -> native capture -> 16 kHz to the engine."""
        import numpy as np

        worker = _load_worker()
        mic = _NativeRateMic(native_rate=native_rate)
        engine = _RecordingEngine()
        monkeypatch.setitem(sys.modules, "sounddevice", mic)
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: engine)

        assert worker.main(["--mic", "2", "--asr", "whisper"]) == 0

        out = capsys.readouterr()
        lines = _protocol_lines(out.out)
        ready = [line for line in lines if line.get("event") == "ready"]
        assert ready and ready[0]["capture_rate"] == native_rate
        assert mic.opened_rates == [native_rate] and mic.rec_rates == [native_rate]
        assert f"capturing at {native_rate} Hz" in out.err
        assert {"text": "pick up the marker", "confidence": 1.0} in lines
        # The unplug that followed is reported, not a crash.
        assert any(line.get("event") == "error" for line in lines)

        assert len(engine.seen) == 1
        audio = engine.seen[0]
        assert audio.dtype == np.float32 and audio.ndim == 1
        # 0.5 s pre-roll + 1.0 s speech + 0.6 s hangover, now at 16 kHz.
        assert len(audio) == pytest.approx(2.1 * 16000, abs=800)
        assert _dominant_hz(audio, 16000) == pytest.approx(300.0, abs=3.0)

    def test_mic_that_accepts_16k_is_not_resampled(self, monkeypatch, capsys):
        import numpy as np

        worker = _load_worker()
        mic = _NativeRateMic(native_rate=16000)
        engine = _RecordingEngine()
        monkeypatch.setitem(sys.modules, "sounddevice", mic)
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: engine)
        calls: list = []
        real = worker.resample_audio
        monkeypatch.setattr(worker, "resample_audio", lambda *a, **k: calls.append(a) or real(*a, **k))

        assert worker.main(["--mic", "0"]) == 0
        capsys.readouterr()
        assert mic.opened_rates == [16000] and calls == []
        assert len(engine.seen) == 1 and engine.seen[0].dtype == np.float32

    def test_a_mic_with_no_usable_rate_exits_2_before_the_model_loads(
        self, monkeypatch, capsys
    ):
        worker = _load_worker()
        loaded: list = []
        monkeypatch.setitem(sys.modules, "sounddevice", _NativeRateMic(accepted={8000}))
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: loaded.append(a))
        assert worker.main(["--mic", "5"]) == 2
        err = capsys.readouterr().err
        assert "Input device 5" in err and "--list-mics" in err
        assert loaded == []

    def test_served_client_gets_resampled_transcripts_over_tcp(self, monkeypatch):
        """--serve with a 48 kHz-only default mic: a client connects, receives
        "ready" (capture_rate 48000) and the transcript, and the server survives
        the disconnect (its model stays resident) until it is interrupted.

        Also pins the TCP path itself: serve_forever sets TCP_NODELAY on the
        accepted connection, which needs ``socket`` at module scope.
        """
        import json
        import socket
        import threading

        worker = _load_worker()
        mic = _NativeRateMic(native_rate=48000, sessions=("speech", "unplugged", "interrupt"))
        engine = _RecordingEngine()
        monkeypatch.setitem(sys.modules, "sounddevice", mic)
        monkeypatch.setattr(worker, "build_engine", lambda *a, **k: engine)

        server = worker.bind_server_socket("127.0.0.1", 0)
        port = server.getsockname()[1]
        result: dict = {}

        def run():
            try:
                result["code"] = worker.serve_forever(
                    "127.0.0.1", port, "canary-gguf", "base.en", "cpu", 4.0, 0.0,
                    server_socket=server,
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                result["error"] = repr(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def session() -> list:
            for _ in range(100):
                try:
                    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
                    break
                except OSError:
                    threading.Event().wait(0.05)  # not listening until the model is "loaded"
            else:
                raise AssertionError("server never started listening")
            data = b""
            with conn:
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            return [json.loads(line) for line in data.decode().splitlines() if line]

        first = session()
        ready = [line for line in first if line.get("event") == "ready"]
        assert ready and ready[0]["capture_rate"] == 48000 and ready[0]["warm"] is True
        assert {"text": "pick up the marker", "confidence": 1.0} in first
        assert len(engine.seen) == 1 and len(engine.seen[0]) == pytest.approx(33600, abs=800)

        session()  # second client: the operator hits Ctrl+C while it is attached
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert result == {"code": 0}

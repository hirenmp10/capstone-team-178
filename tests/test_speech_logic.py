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

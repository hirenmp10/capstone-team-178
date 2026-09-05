"""Speech recognition front-end.

Pure stdlib. This module must never import Isaac Sim -- **nor any speech backend
in-process**.

Speech recognition runs in a **subprocess**, always. This is not defensive
structuring: CTranslate2 (which backs faster-whisper) crashes the process when
loaded inside Isaac Sim's interpreter on Windows, and it takes the simulator down
with it. A crash in a subprocess costs one transcription; a crash in-process costs
the whole session and whatever the robot was holding.

The front-end only produces *text*. It performs no parsing, no grounding and no
execution -- the transcript goes to the intent parser exactly as if it had been
typed, so voice adds an input path without adding a second command pipeline that
could drift from the text one.
"""

from __future__ import annotations

import abc
import json
import socket
import subprocess
import sys
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

__spoken_feedback_doc__ = """Spoken feedback uses Windows SAPI via PowerShell so it
needs no extra install; see :func:`speak`."""

from mfw.core.errors import MfwError
from mfw.utils.logging import get_logger

__all__ = [
    "speak",
    "ISpeechRecognizer",
    "TextRecognizer",
    "ScriptedRecognizer",
    "WhisperSubprocessRecognizer",
    "TcpSpeechRecognizer",
    "Transcript",
    "VoiceCommandLoop",
    "SpeechError",
]

_log = get_logger("language.speech")


class SpeechError(MfwError):
    """Speech recognition failed."""


@dataclass(frozen=True)
class Transcript:
    """One recognised utterance."""

    text: str
    confidence: float = 1.0
    source: str = "text"

    def to_log(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "source": self.source}


def speak(text: str, enabled: bool = True, blocking: bool = False) -> None:
    """Say ``text`` aloud using the platform's built-in speech synthesiser.

    Windows SAPI via PowerShell, which ships with the OS -- no pip install, and
    nothing extra in the venv. Falls back silently on other platforms or if the
    synthesiser is unavailable: losing spoken feedback must never interrupt the
    robot, since this is a convenience channel, not a control path.

    Non-blocking by default so announcing "ready" does not delay the microphone
    opening.
    """
    if not enabled or not text:
        return

    try:
        if sys.platform == "win32":
            # Single-quote escaping for PowerShell string literals.
            safe = text.replace("'", "''")
            command = (
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak('{safe}')"
            )
            args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        elif sys.platform == "darwin":
            args = ["say", text]
        else:
            args = ["spd-say", text]

        if blocking:
            subprocess.run(args, capture_output=True, timeout=30)
        else:
            subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception as exc:  # pragma: no cover - platform dependent
        _log.debug("Speech synthesis unavailable: %s", exc)


class ISpeechRecognizer(abc.ABC):
    """Produces transcripts. An interface so voice is swappable and testable."""

    @abc.abstractmethod
    def listen(self, timeout_s: float | None = None) -> Transcript | None:
        """Return the next utterance, or ``None`` on timeout."""

    def start(self) -> None:
        """Begin capturing. Optional."""

    def stop(self) -> None:
        """Stop capturing and release resources. Optional."""


class TextRecognizer(ISpeechRecognizer):
    """Reads typed lines from a stream.

    The default input path, and the reason the whole language stack is testable
    without a microphone: typed and spoken commands take the same route from here
    on, so there is only ever one command pipeline to verify.
    """

    def __init__(self, stream: Any = None, prompt: str = "> ") -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._prompt = prompt

    def listen(self, timeout_s: float | None = None) -> Transcript | None:
        if self._prompt and self._stream is sys.stdin:
            print(self._prompt, end="", flush=True)
        line = self._stream.readline()
        if not line:
            return None
        line = line.strip()
        return Transcript(line, 1.0, "text") if line else None


class ScriptedRecognizer(ISpeechRecognizer):
    """Replays a fixed list of utterances. For tests and demos."""

    def __init__(self, utterances: list[str]) -> None:
        self._remaining = list(utterances)

    def listen(self, timeout_s: float | None = None) -> Transcript | None:
        if not self._remaining:
            return None
        return Transcript(self._remaining.pop(0), 1.0, "scripted")


class _JsonLineRecognizer(ISpeechRecognizer):
    """Shared plumbing for recognisers fed by a stream of JSON lines.

    The subprocess and TCP recognisers differ only in *where* the bytes come
    from; the protocol -- transcripts interleaved with status events -- is
    identical. Keeping one parser means a protocol change cannot leave the two
    transports subtly out of step.
    """

    def __init__(self, queue_size: int = 16, source: str = "speech") -> None:
        self._queue: queue.Queue[Transcript] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.source = source
        #: Called with (event_name, fields) for worker status: loading, ready,
        #: listening, thinking, silence, error.
        self.on_event: Callable[[str, dict[str, Any]], None] | None = None

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # A worker that prints diagnostics must not break the caller.
            _log.debug("Ignoring non-JSON line from the speech worker: %r", line[:120])
            return

        if "event" in payload:
            if self.on_event is not None:
                try:
                    self.on_event(str(payload["event"]), payload)
                except Exception:  # pragma: no cover - callback is caller code
                    _log.debug("on_event callback raised", exc_info=True)
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            return
        try:
            confidence = float(payload.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8

        self._offer(Transcript(text=text, confidence=confidence, source=self.source))

    def _offer(self, transcript: Transcript) -> None:
        try:
            self._queue.put_nowait(transcript)
        except queue.Full:
            # Prefer the newest speech: an operator saying "stop" wants it acted
            # on now, not after a backlog drains.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(transcript)
            except queue.Empty:  # pragma: no cover - race
                pass

    def listen(self, timeout_s: float | None = None) -> Transcript | None:
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Discard buffered transcripts. Returns how many were dropped.

        Called after a command finishes. A pick takes tens of seconds, and the
        microphone keeps listening throughout -- so anything said while the arm
        was moving (including thinking aloud, or repeating the command because
        nothing seemed to happen) is sitting in the queue when it ends.

        Executing that backlog is actively dangerous: observed in practice, one
        pick was followed by three queued "place" commands firing back to back,
        two of them against a gripper that was already empty. Speech uttered
        before the robot finished cannot have accounted for the result, so it is
        dropped rather than obeyed.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped


class TcpSpeechRecognizer(_JsonLineRecognizer):
    """Client for a resident speech server (``speech_worker.py --serve``).

    The model stays loaded in that process, so attaching costs milliseconds
    instead of the ~39 s a cold ``SALM.from_pretrained`` takes. Same protocol as
    the subprocess transport, so nothing downstream changes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        queue_size: int = 16,
        source: str = "speech-server",
        connect_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(queue_size=queue_size, source=source)
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        except OSError as exc:
            raise SpeechError(
                f"could not reach the speech server at {self.host}:{self.port}: {exc}\n"
                "Start it first, in its own terminal:\n"
                "    <voice-python> scripts/speech_worker.py --serve --asr canary --device cuda"
            ) from exc

        # No timeout once connected: the socket blocks between utterances, which
        # is normal, not a stall.
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._sock is not None
        buffer = ""
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            # TCP is a stream: a recv can split a line or deliver several, so
            # reassemble on newlines rather than assuming message boundaries.
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                self._handle_line(line)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class WhisperSubprocessRecognizer(ISpeechRecognizer):
    """Whisper transcription in an isolated subprocess.

    The worker is launched with an interpreter of the caller's choosing --
    deliberately *not* Isaac Sim's. It streams one JSON object per line on stdout,
    so a malformed or crashed worker degrades to "no transcript" rather than
    corrupting the caller.

    The worker script is expected to accept ``--model`` and ``--device`` and to
    emit ``{"text": ..., "confidence": ...}`` lines. It is kept external because a
    speech backend has no business being importable from the simulator's process.
    """

    def __init__(
        self,
        worker_script: str | Path,
        python_executable: str = sys.executable,
        model: str = "base.en",
        device: str = "cpu",
        queue_size: int = 16,
        asr: str = "whisper",
        extra_args: tuple[str, ...] = (),
    ) -> None:
        """``python_executable`` must NOT be Isaac Sim's interpreter.

        It defaults to ``sys.executable`` for standalone use, but when this runs
        inside the simulator that default *is* Isaac's python -- the one
        environment where the speech backend crashes the process. Callers running
        under Isaac must pass a system interpreter explicitly.
        """
        self.worker_script = Path(worker_script)
        self.python_executable = python_executable
        self.model = model
        self.device = device
        self.asr = asr
        self.extra_args = tuple(extra_args)
        self._process: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[Transcript] = queue.Queue(maxsize=queue_size)
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stop = threading.Event()
        #: Called with (event_name, fields) for worker status: loading, ready,
        #: listening, thinking, silence, error. Lets the caller show the operator
        #: when the microphone is actually live.
        self.on_event: Callable[[str, dict[str, Any]], None] | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.worker_script.is_file():
            raise SpeechError(f"speech worker script not found: {self.worker_script}")

        command = [
            self.python_executable,
            str(self.worker_script),
            "--asr",
            self.asr,
            "--device",
            self.device,
        ]
        # --model names a Whisper checkpoint. Canary's weights are fixed by its
        # model id, so passing it there would print a misleading "base.en" in the
        # launch log and invite the reader to think Whisper was loading.
        if self.asr == "whisper":
            command += ["--model", self.model]
        command += list(self.extra_args)
        _log.info("Starting speech worker: %s", " ".join(command))
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise SpeechError(f"could not start the speech worker: {exc}") from exc

        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        # Draining stderr is mandatory, not tidiness. The pipe has a fixed OS
        # buffer (~64 KB); once it fills, the worker blocks on its next write and
        # the whole speech pipeline deadlocks with no error. NeMo logs prolifically
        # -- model init alone can exceed that -- so an undrained stderr would hang
        # the session partway through.
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()

    def _drain_stderr(self) -> None:
        """Consume the worker's stderr so its pipe can never fill."""
        assert self._process is not None
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            if self._stop.is_set():
                return
            line = line.rstrip()
            if line:
                _log.debug("speech worker: %s", line[:200])

    def _read_loop(self) -> None:
        """Drain the worker's stdout into the queue."""
        assert self._process is not None
        stdout = self._process.stdout
        if stdout is None:
            return

        for line in stdout:
            if self._stop.is_set():
                return
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # A worker that prints diagnostics must not break the caller.
                _log.debug("Ignoring non-JSON line from the speech worker: %r", line[:120])
                continue

            # Status events share the stream with transcripts; route them to the
            # caller instead of treating them as speech.
            if "event" in payload:
                if self.on_event is not None:
                    try:
                        self.on_event(str(payload["event"]), payload)
                    except Exception:  # pragma: no cover - callback is caller code
                        _log.debug("on_event callback raised", exc_info=True)
                continue

            try:
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue
                transcript = Transcript(
                    text=text,
                    confidence=float(payload.get("confidence", 0.8)),
                    source=self.asr,
                )
            except (TypeError, ValueError):
                _log.debug("Malformed transcript payload: %r", line[:120])
                continue

            try:
                self._queue.put_nowait(transcript)
            except queue.Full:
                # Prefer the newest speech: an operator saying "stop" wants it
                # acted on now, not after a backlog drains.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(transcript)
                except queue.Empty:  # pragma: no cover - race
                    pass

    def listen(self, timeout_s: float | None = None) -> Transcript | None:
        if self._process is None:
            self.start()
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self._process.kill()
            self._process = None


class VoiceCommandLoop:
    """Feeds transcripts to a handler, one command at a time.

    Strictly sequential: the next utterance is not read until the current command
    has finished. Overlapping commands on a physical arm are unsafe, and the
    atomicity guarantee only means anything if commands cannot interleave.

    Recognised speech below ``min_confidence`` is surfaced for confirmation rather
    than executed. Acting on a misheard manipulation command moves a real arm.
    """

    def __init__(
        self,
        recognizer: ISpeechRecognizer,
        handle: Callable[[str], Any],
        min_confidence: float = 0.5,
        on_low_confidence: Callable[[Transcript], None] | None = None,
        stop_words: tuple[str, ...] = ("quit", "exit", "shutdown"),
        confirm: Callable[[str], bool] | None = None,
        on_dropped: Callable[[int], None] | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.handle = handle
        self.min_confidence = min_confidence
        self.on_low_confidence = on_low_confidence
        #: Asked before executing. Returning False cancels the command. This is
        #: the only defence available when the recogniser reports no confidence
        #: -- Canary always claims 1.0, so the confidence gate cannot filter it.
        self.confirm = confirm
        self.on_dropped = on_dropped
        self.stop_words = tuple(word.lower() for word in stop_words)
        self._running = False

    def run(self, max_commands: int | None = None, timeout_s: float | None = None) -> int:
        """Process commands until the stream ends or a stop word is heard.

        Returns the number of commands executed.
        """
        self.recognizer.start()
        self._running = True
        executed = 0

        try:
            while self._running:
                if max_commands is not None and executed >= max_commands:
                    break

                transcript = self.recognizer.listen(timeout_s)
                if transcript is None:
                    break

                if transcript.text.lower().strip() in self.stop_words:
                    _log.info("Stop word heard: %r", transcript.text)
                    break

                if transcript.confidence < self.min_confidence:
                    _log.info(
                        "Low-confidence transcript (%.2f): %r -- not executing",
                        transcript.confidence,
                        transcript.text,
                    )
                    if self.on_low_confidence is not None:
                        self.on_low_confidence(transcript)
                    continue

                if self.confirm is not None and not self.confirm(transcript.text):
                    # Declined: drop anything said while we were asking, so a
                    # spoken "no" cannot itself become the next command.
                    drain = getattr(self.recognizer, "drain", None)
                    if drain is not None:
                        drain()
                    continue

                # Blocking by design: one command completes before the next is read.
                self.handle(transcript.text)
                executed += 1

                # Drop whatever was said while the robot was busy. See
                # _JsonLineRecognizer.drain for why obeying it is unsafe.
                drain = getattr(self.recognizer, "drain", None)
                if drain is not None:
                    dropped = drain()
                    if dropped and self.on_dropped is not None:
                        self.on_dropped(dropped)
        finally:
            self._running = False
            self.recognizer.stop()

        return executed

    def stop(self) -> None:
        self._running = False

    def transcripts(self, timeout_s: float | None = None) -> Iterator[Transcript]:
        """Iterate transcripts without executing them. For diagnostics."""
        self.recognizer.start()
        try:
            while True:
                transcript = self.recognizer.listen(timeout_s)
                if transcript is None:
                    return
                yield transcript
        finally:
            self.recognizer.stop()

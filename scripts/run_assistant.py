"""Run the manipulation assistant.

Must be launched with Isaac Sim's interpreter:

    ..\\python.bat manipulation_framework\\scripts\\run_assistant.py --demo
    ..\\python.bat manipulation_framework\\scripts\\run_assistant.py --gui
    ..\\python.bat manipulation_framework\\scripts\\run_assistant.py --gui --interactive
    ..\\python.bat manipulation_framework\\scripts\\run_assistant.py -c "pick up the can"

Modes:
  --demo         run a scripted sequence that exercises the whole pipeline
  --interactive  type commands at a prompt
  -c/--command   run one or more commands and exit
  --gui          show the Isaac Sim window (default is headless)

Hardware lane (no Isaac Sim; run with a plain ``py -3.12``):

    py -3.12 scripts/run_assistant.py --fake-hardware -c "pick up the marker"
    py -3.12 scripts/run_assistant.py --fake-hardware --demo
    py -3.12 scripts/run_assistant.py --hardware --jetson 192.168.1.50 --interactive
    py -3.12 scripts/run_assistant.py --hardware --jetson 192.168.1.50 --demo

  --hardware        drive the real arm through the Jetson robot server
                    (configs/hardware.yaml unless --config is given)
  --fake-hardware   the same lane against loopback fakes; autostarts
                    jetson/robot_server.py --driver fake and
                    jetson/detector_service.py --backend scripted
  --jetson HOST[:PORT], --detector-server HOST:PORT, --llm-server HOST:PORT
                    point the three services at another machine
  --demo            on the hardware lane: scan the room -> what do you see ->
                    pick up the marker -> put it in the bowl -> go home
                    (hardware.demo_script in the YAML overrides it)
  --home-confirmed  pre-confirm the first home for a run with no terminal
                    (see below)

First home (real arm only). After every robot_server start the servo bridge
knows no position, and the bring-up `home` attaches every servo AT home at
full speed. Against a real driver in that state, --hardware prints the
hand-pose instructions (arm at home, E-stop in reach, hands clear), discards
anything typed during bring-up and waits for the word `home`; Ctrl-C or end
of input aborts before anything moves. It asks the same way on a second run
against the same robot_server when the servos are detached (the server's
5 s host timeout after the previous session, a cleared estop): that home
re-attaches AT the last pulsed pose at full speed first, so the arm is posed
there, not at home. With no terminal (stdin not a TTY: piped, scheduled, a
service unit) it refuses and exits 1 unless --home-confirmed says a person
already hand-posed the arm. Only the fake lane, and a real arm whose servos
are still attached at a known position, never prompt.

Exit status: 0 when every -c/--demo command succeeded, 1 when any failed (or
the first home was refused), 130 on Ctrl-C. On the hardware lane the run stops at the first failure unless
--keep-going; the sim lane runs every command, as it always has.

``sys.argv`` is cleared before the simulator starts: ``SimulationApp`` parses argv
itself and exits on flags it does not recognise, so our own options must not still
be there when it initialises.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Human-readable engine names. "canary" alone reads like a third-party model
#: when it is in fact NVIDIA's own ASR family -- worth spelling out on screen so
#: nobody has to wonder whether the NVIDIA model actually loaded.
ENGINE_LABELS = {
    "canary": "NVIDIA Canary-Qwen-2.5B  (nvidia/canary-qwen-2.5b)",
    "whisper": "OpenAI Whisper via faster-whisper",
}

DEMO_SCRIPT = [
    "what do you see",
    "open the gripper",
    "pick up the block",
    "move up 5 cm",
    "place it",
    "go home",
]

#: The hardware MVP demo (``--demo`` with ``--hardware``/``--fake-hardware``),
#: in the words of configs/hardware.yaml's labels: marker (19 mm across the
#: jaw) into the bowl. "can" is avoided on purpose -- Canary hears "kin".
#: ``hardware.demo_script`` in the YAML replaces it without a code change.
HARDWARE_DEMO_SCRIPT = [
    "scan the room",
    "what do you see",
    "pick up the marker",
    "put it in the bowl",
    "go home",
]

#: Whole-clause phrases that mean "scan the room" on the hardware lane but
#: that the rule parser maps elsewhere ("scan the table" -> observe) or not at
#: all ("look around"). Rewritten before parsing, only when scan_scene is
#: registered, so the sim lane's parse of the same words is untouched.
_SCAN_PHRASE = re.compile(
    r"(?:please )?(?:(?:can|could|will) you )?"
    r"(?:(?:have a |take a )?look around(?: the (?:room|table|scene))?"
    r"|scan (?:the )?(?:room|table|scene|area|workspace)"
    r"|look at the table)"
    r"(?: for me)?(?: please)?"
)


def route_hardware_phrase(utterance: str) -> str:
    """``"look around"`` / ``"scan the table"`` -> ``"scan the room"``; else unchanged."""
    text = re.sub(r"[^a-z0-9\s]", " ", str(utterance).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return "scan the room" if _SCAN_PHRASE.fullmatch(text) else utterance


def route_clauses(
    clauses_of: Callable[[str], list],
    run_clause: Callable[[str], Any],
    route: Callable[[str], str] = route_hardware_phrase,
) -> tuple[Callable[[str], list], Callable[[str], Any]]:
    """``(clauses_of, run_clause)`` that apply ``route`` to every clause first."""
    return (
        lambda utterance: [route(c) for c in clauses_of(utterance)],
        lambda clause: run_clause(route(clause)),
    )


def format_outcome(utterance: str, outcome, show_remaining: bool = True) -> list[str]:
    """What the robot understood, decided and did for one clause, as lines."""
    status = "OK " if outcome.ok else "-- "
    lines = [f"{status} \"{utterance}\""]
    ran_as = str(getattr(outcome, "utterance", "") or "").strip()
    if ran_as and ran_as != utterance.strip():
        # A clarification re-ran the clause against a named track.
        lines.append(f"    ran as   : {ran_as}")
    lines.append(f"    skill    : {outcome.skill or '(not understood)'}")
    if outcome.params:
        lines.append(f"    params   : {outcome.params}")
    lines.append(f"    result   : {outcome.message or '(no message)'}")

    if outcome.needs_clarification and outcome.clarification_options:
        from mfw.language.grounding import clarification_question

        lines.append(f"    ambiguous: {', '.join(outcome.clarification_options)}")
        lines.append(f"    question : {clarification_question(outcome.clarification_options)}")
    if show_remaining and outcome.remaining_clauses:
        lines.append(f"    not run  : {' / '.join(outcome.remaining_clauses)}")

    if outcome.action_graph:
        lines.append(f"    pipeline : {' -> '.join(outcome.action_graph)}")

    # Surface the perceived scene, since it is the whole basis for the decision.
    if outcome.result is not None and outcome.result.data.get("objects"):
        objects = outcome.result.data["objects"]
        if isinstance(objects, list):
            lines.append("    perceived:")
            for obj in objects:
                lines.append(
                    f"       {obj['label']:<8} at {obj['position']} "
                    f"size {obj['size']} conf {obj['confidence']}"
                )

    if outcome.attempts > 1:
        lines.append(f"    attempts : {outcome.attempts}")
    lines.append(f"    took     : {outcome.duration_s:.2f}s")
    return lines


def _print_outcome(utterance: str, outcome) -> None:
    """Show what the robot understood, decided and did."""
    print("\n" + "\n".join(format_outcome(utterance, outcome)))


class CommandReport:
    """Every clause one utterance ran, in order, and the clauses it never ran.

    ``Assistant.command`` returns only the LAST clause's outcome, so "move the
    red block onto the green box" used to print the place half alone: the pick
    -- its skill, params and result -- was invisible. The CLI therefore runs the
    clauses itself (the same split ``Assistant.clauses`` gives, the same stop at
    the first clause that is not ok) and keeps every outcome.
    """

    def __init__(self, utterance: str, steps: list, not_run: tuple = ()) -> None:
        self.utterance = utterance
        self.steps = list(steps)  # [(clause, CommandOutcome), ...]
        self.not_run = tuple(not_run)

    @property
    def final(self):
        """The last outcome that ran (what a spoken reply should report)."""
        return self.steps[-1][1]

    @property
    def ok(self) -> bool:
        return bool(self.steps) and not self.not_run and all(o.ok for _, o in self.steps)

    @property
    def compound(self) -> bool:
        return len(self.steps) + len(self.not_run) > 1

    @property
    def duration_s(self) -> float:
        return float(sum(o.duration_s for _, o in self.steps))


def run_command(
    utterance: str,
    clauses_of: Callable[[str], list],
    run_clause: Callable[[str], Any],
) -> CommandReport:
    """Run ``utterance`` clause by clause, keeping every outcome.

    ``clauses_of`` is ``Assistant.clauses``; ``run_clause`` is
    ``Assistant.command`` or a ``command_with_clarification`` wrapper. A single
    clause is handed over whole, exactly as before.
    """
    clauses = [c for c in clauses_of(utterance) if c]
    if len(clauses) <= 1:
        outcome = run_clause(utterance)
        return CommandReport(utterance, [(utterance, outcome)], tuple(outcome.remaining_clauses))
    steps: list = []
    for index, clause in enumerate(clauses):
        outcome = run_clause(clause)
        steps.append((clause, outcome))
        if not outcome.ok:
            not_run = tuple(outcome.remaining_clauses) + tuple(clauses[index + 1:])
            return CommandReport(utterance, steps, not_run)
    return CommandReport(utterance, steps, ())


def format_report(report: CommandReport) -> str:
    """Every clause's skill, params and result, then the clauses that did not run."""
    if not report.compound:
        clause, outcome = report.steps[0]
        return "\n".join(format_outcome(report.utterance, outcome))

    total = len(report.steps) + len(report.not_run)
    succeeded = sum(1 for _, o in report.steps if o.ok)
    failed = len(report.steps) - succeeded
    lines = [
        f"{'OK ' if report.ok else '-- '} \"{report.utterance}\"",
        f"    clauses  : {total} -> {succeeded} ok, {failed} failed, "
        f"{len(report.not_run)} not run",
    ]
    for number, (clause, outcome) in enumerate(report.steps, start=1):
        clause_lines = format_outcome(clause, outcome, show_remaining=False)
        lines.append(f"  [{number}/{total}] {clause_lines[0]}")
        lines.extend(clause_lines[1:])
    for number, clause in enumerate(report.not_run, start=len(report.steps) + 1):
        lines.append(f"  [{number}/{total}] NOT RUN \"{clause}\"")
    if report.not_run:
        lines.append(
            f"    not run  : {' / '.join(report.not_run)}  "
            f"(stopped at clause {len(report.steps)} of {total})"
        )
    lines.append(f"    total    : {report.duration_s:.2f}s")
    return "\n".join(lines)


def _print_report(report: CommandReport) -> None:
    print("\n" + format_report(report))


def run_script(
    commands: list[str],
    run_one: Callable[[str], CommandReport],
    keep_going: bool = False,
    emit: Callable[[CommandReport], None] = _print_report,
) -> tuple[list[CommandReport], tuple[str, ...]]:
    """Run scripted (-c / --demo) commands in order; ``(reports, not_run)``.

    Stops at the first command that is not ok unless ``keep_going``: "put it
    in the bowl" after a failed pick has nothing to put, and on the real arm
    every further motion is one more thing to go wrong unobserved.
    """
    reports: list[CommandReport] = []
    for index, utterance in enumerate(commands):
        report = run_one(utterance)
        emit(report)
        reports.append(report)
        if not report.ok and not keep_going:
            return reports, tuple(commands[index + 1:])
    return reports, ()


def script_summary(reports: list[CommandReport], not_run: tuple[str, ...]) -> str:
    """One line for the end of a scripted run; names the first failure."""
    failed = [r for r in reports if not r.ok]
    line = (
        f"commands : {len(reports) + len(not_run)} -> {len(reports) - len(failed)} ok, "
        f"{len(failed)} failed, {len(not_run)} not run"
    )
    if failed:
        first = failed[0]
        line += f'\n  first failure: "{first.utterance}" -> {first.final.message or "(no message)"}'
    if not_run:
        line += f"\n  not run  : {' / '.join(not_run)}  (pass --keep-going to run them anyway)"
    return line


def script_exit_code(reports: list[CommandReport], not_run: tuple[str, ...]) -> int:
    """0 only when every scripted command ran and succeeded."""
    return 0 if not not_run and all(r.ok for r in reports) else 1


def lane_exit_code(hardware_mode: bool, reports: list[CommandReport], not_run: tuple[str, ...]) -> int:
    """The process exit status after a scripted run.

    Hardware lanes (``--hardware``/``--fake-hardware``): :func:`script_exit_code`,
    so a failed demo step is a non-zero exit. The sim lane keeps HEAD's
    behaviour -- a scripted run exits 0 whatever its commands did (review:
    the simulation's behaviour must not change); read its summary instead.
    """
    return script_exit_code(reports, not_run) if hardware_mode else 0


def _listen_for_reply(recognizer: Any, attempts: int = 3) -> str | None:
    """The next non-empty thing the operator says, or ``None``.

    Raises whatever the recogniser raises, so the caller can fall back to the
    keyboard when voice is unavailable.
    """
    for _ in range(attempts):
        reply = recognizer.listen()
        if reply is None:
            continue
        heard = str(getattr(reply, "text", reply)).strip()
        if heard:
            print(f'  heard: "{heard}"', flush=True)
            return heard
    return None


def _typed_ask(question: str) -> str | None:
    """Ask at the keyboard; ``None`` on end of input."""
    print(f"  {question}", flush=True)
    try:
        return input("  which? ").strip()
    except (EOFError, KeyboardInterrupt):
        return None


#: Interpreters to try for the speech worker, best first. Deliberately excludes
#: Isaac Sim's own python: CTranslate2 (the faster-whisper backend) crashes when
#: loaded inside it on Windows and takes the simulator with it.
_VOICE_PYTHON_CANDIDATES = (
    # Dedicated NeMo/Canary environment first: it is the only one with the
    # speech-LLM stack, and it is deliberately isolated so installing NeMo could
    # not replace the CUDA torch build the rest of the system depends on.
    str(Path.home() / "canary-venv" / "Scripts" / "python.exe"),
    str(Path.home() / "canary-venv" / "bin" / "python"),
    str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe"),
    "python",
)


def _find_voice_python(explicit: str | None, asr: str) -> str:
    """Locate an interpreter that can actually run the speech worker."""
    import shutil
    import subprocess

    needed = {
        "whisper": ("faster_whisper", "sounddevice"),
        "canary": ("nemo", "sounddevice"),
    }[asr]
    probe_source = "import " + ", ".join(needed)

    candidates = [explicit] if explicit else list(_VOICE_PYTHON_CANDIDATES)
    checked = []

    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if not resolved:
            checked.append(f"{candidate} (not found)")
            continue
        try:
            probe = subprocess.run(
                [resolved, "-c", probe_source], capture_output=True, timeout=180
            )
        except Exception as exc:
            checked.append(f"{resolved} ({exc})")
            continue
        if probe.returncode == 0:
            return resolved
        checked.append(f"{resolved} (missing one of: {', '.join(needed)})")

    install = (
        'pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" sounddevice'
        if asr == "canary"
        else "pip install faster-whisper sounddevice"
    )
    extra = (
        "\nInstall NeMo into a DEDICATED venv. Dropping it into an existing\n"
        "environment can replace a working torch build, and on this machine that\n"
        "would break the CUDA 13 / Blackwell (sm_120) setup the GPU needs.\n"
        if asr == "canary"
        else ""
    )
    raise SystemExit(
        f"No interpreter with the {asr} dependencies was found.\n"
        "Checked:\n  " + "\n  ".join(checked) + "\n\n"
        "Install into a SYSTEM python (never Isaac Sim's):\n"
        f"    <python.exe> -m {install}\n" + extra + "\nthen pass it explicitly:\n"
        "    --voice-python <path-to-python.exe>"
    )


def _server_is_up(host: str, port: int, timeout: float = 1.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _autostart_speech_server(
    voice_python: str, worker: Path, host: str, port: int, asr: str, device: str,
    mic: int | None, wait_seconds: float = 300.0,
) -> bool:
    """Launch the resident speech server and wait for it to accept connections.

    Requiring the operator to keep a second terminal alive is friction with no
    upside: the server's value is that the model stays loaded, and that holds
    whether a human started it or we did. Launched in its own console window so
    its loading progress stays visible and, crucially, so it **outlives this
    process** -- the next run then attaches in milliseconds instead of paying the
    ~39 s load again.
    """
    import subprocess

    command = [
        voice_python, str(worker), "--serve",
        "--asr", asr, "--device", device,
        "--host", host, "--port", str(port),
    ]
    if mic is not None:
        command += ["--mic", str(mic)]

    print(f"\n  Speech server not running. Starting it on {device}...")
    print(f"    {' '.join(command)}")

    creationflags = 0
    if sys.platform == "win32":
        # Its own console: the model load prints there, and the process is not
        # killed when this one exits.
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    try:
        subprocess.Popen(command, creationflags=creationflags, close_fds=True)
    except OSError as exc:
        print(f"  Could not launch the speech server: {exc}")
        return False

    if wait_seconds <= 0:
        # Caller will wait later. Returning now lets the model load in parallel
        # with Isaac Sim's own startup rather than after it.
        print("  Loading in its own window while Isaac Sim starts...")
        return True

    print(f"  Loading the model (up to {wait_seconds:.0f}s on first run)", end="", flush=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _server_is_up(host, port):
            print(" ready.")
            return True
        print(".", end="", flush=True)
        time.sleep(2.0)

    print("\n  Speech server did not come up in time. Check its console window.")
    return False


def _robot_server_is_up(host: str, port: int, timeout: float = 1.0) -> bool:
    """ZMQ ping to a robot server. A TCP connect is not enough: ZeroMQ accepts
    the connection whatever is (or is not) behind the port."""
    try:
        import zmq
    except ImportError:
        return False
    try:
        import msgpack  # noqa: F401
    except ImportError:
        return False
    from mfw.hardware.zmq_rpc import RpcError, ZmqRpcClient

    client = ZmqRpcClient(host, port, request_timeout_s=timeout)
    try:
        return bool(client.ping(retries=1).get("ok"))
    except RpcError:
        return False
    finally:
        client.close()


def _parse_host_port(spec: str, default_host: str, default_port: int) -> tuple[str, int]:
    host, _, port_text = str(spec).partition(":")
    return host or default_host, int(port_text or default_port)


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


#: The scripted scene both fake services agree on: the robot server owns it
#: (objects move when the fake jaw closes on them) and the detector reads it.
#: The cube sits 37 cm out, beyond the workspace, so the demo also shows the
#: "out of reach" report (it is never picked or planned around).
FAKE_SCENE = "marker:0.18,0.05 bowl:0.15,-0.12 cube:0.20,0.31"
#: The Uno sketch's watchdog (servo_bridge.ino), mirrored by the spawned fake.
FAKE_WATCHDOG_MS = 500

#: Printed at startup on the hardware lane (reviews P4 / HS-4). The assistant is
#: single-threaded and blocks inside each motion RPC: a typed/spoken "stop" and
#: Ctrl-C take effect only when that RPC returns -- between trajectory chunks, or
#: after the boot homing move -- never inside one. Only the servo power switch
#: stops the arm mid-motion.
STOP_NOTICE = (
    "software stop (typed/spoken 'stop', Ctrl-C) acts only between chunks "
    "(each <= trajectory_chunk_s = {chunk:.1f} s) and after the boot homing move; "
    "the +6 V switch is the real mid-motion stop"
)

#: Receive timeout for the one-shot estop sent when Ctrl-C arrives before the
#: arm's own stop channel exists. Short: the operator is waiting on it.
ONE_SHOT_ESTOP_TIMEOUT_S = 2.0


def _terminate_spawned(procs: list) -> None:
    """Terminate the fake services this run started. Idempotent; never raises.

    Registered with ``atexit`` the moment they are spawned *and* called from
    ``main``'s ``finally``, so an exception between the spawn and the ``try``
    (the LLM setup, a banner print) or a second Ctrl-C cannot orphan them.
    """
    import subprocess

    for proc in list(procs):
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except OSError:
            pass
    procs.clear()


def _one_shot_estop(
    host: str,
    port: int,
    client_factory: "Callable[..., Any] | None" = None,
    timeout_s: float = ONE_SHOT_ESTOP_TIMEOUT_S,
) -> bool:
    """Send ``estop`` on a fresh connection to the robot server; never raises.

    For Ctrl-C before the runtime's arm (and its dedicated stop channel) exists
    -- mid bring-up, e.g. during the boot homing move. The robot server answers
    ``estop`` from any client while another waits on a motion, so a new
    connection is as good as the dedicated channel. ``client_factory`` defaults
    to :class:`~mfw.hardware.jetson_client.JetsonClient` (tests pass a fake).
    """
    client = None
    try:
        if client_factory is None:
            from mfw.hardware.jetson_client import JetsonClient as client_factory
        client = client_factory(host, port, request_timeout_s=timeout_s)
        reply = client.estop()
    except Exception as exc:  # noqa: BLE001 - a stop that fails is reported, not raised
        print(f"  one-shot estop to tcp://{host}:{port} failed: {exc}", flush=True)
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
    return not (isinstance(reply, dict) and reply.get("estopped") is False)


def _send_interrupt_estop(
    runtime: Any,
    fallback_estop: "Callable[[], bool] | None" = None,
    fallback_label: str = "the robot server",
) -> bool | None:
    """The estop half of the Ctrl-C handler. ``None`` when nothing was sent.

    With a built arm: ``runtime.emergency_stop()`` on its dedicated channel,
    then once more through ``fallback_estop`` if that was not acknowledged.
    Before the arm exists (bring-up: Assistant construction, boot homing):
    ``fallback_estop`` -- a one-shot estop to the configured Jetson.
    """
    stop = getattr(runtime, "emergency_stop", None) if runtime is not None else None
    arm_ready = stop is not None and getattr(runtime, "robot", True) is not None
    if arm_ready:
        print("\n  Ctrl-C: sending estop on the arm's dedicated stop channel "
              "(servos detach; the arm goes limp)...", flush=True)
        try:
            acknowledged = bool(stop())
        except Exception as exc:  # noqa: BLE001 - never mask the interrupt itself
            acknowledged = False
            print(f"  estop failed: {exc}", flush=True)
        if not acknowledged and fallback_estop is not None:
            print(f"  retrying once on a fresh connection to {fallback_label}...", flush=True)
            acknowledged = bool(fallback_estop())
    elif fallback_estop is not None:
        print("\n  Ctrl-C during bring-up (the arm may be homing): sending a one-shot "
              f"estop to {fallback_label} (servos detach; the arm goes limp)...", flush=True)
        acknowledged = bool(fallback_estop())
    else:
        return None
    print("  estop acknowledged." if acknowledged else
          "  estop NOT acknowledged -- cut the +6 V servo supply now.", flush=True)
    return acknowledged


def _install_estop_on_sigint(
    get_runtime: "Callable[[], Any]",
    fallback_estop: "Callable[[], bool] | None" = None,
    fallback_label: str = "the robot server",
) -> Any:
    """Make Ctrl-C estop the arm (hardware lane) before the process exits.

    The handler sends ``estop`` (see :func:`_send_interrupt_estop`), then
    raises ``KeyboardInterrupt`` so the normal shutdown path runs. A second
    Ctrl-C gets Python's default behaviour. Returns the previous handler.
    """
    import signal

    def handler(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        _send_interrupt_estop(get_runtime(), fallback_estop, fallback_label)
        raise KeyboardInterrupt

    return signal.signal(signal.SIGINT, handler)


class HardwareStopGuard:
    """Ctrl-C -> estop for the whole hardware session, bring-up included (HS-4).

    ``Assistant(...)`` connects, builds the arm and runs the boot homing move
    before it returns, so a handler that looked only at the finished
    ``assistant`` found nothing during bring-up and sent no stop at all. The
    guard keeps a handle on the half-built Assistant (:meth:`construct`) so the
    handler reaches its runtime's dedicated stop channel as soon as the arm
    exists, and falls back to a one-shot estop to the configured Jetson before
    that.
    """

    def __init__(
        self,
        jetson_host: str,
        jetson_port: int,
        client_factory: "Callable[..., Any] | None" = None,
        timeout_s: float = ONE_SHOT_ESTOP_TIMEOUT_S,
    ) -> None:
        self.jetson_host = str(jetson_host)
        self.jetson_port = int(jetson_port)
        self.client_factory = client_factory
        self.timeout_s = float(timeout_s)
        self.assistant: Any = None
        self._pending: Any = None
        self._previous: Any = None
        self._installed = False

    @property
    def bringing_up(self) -> bool:
        """True until :meth:`construct` returned."""
        return self.assistant is None

    def runtime(self) -> Any:
        target = self.assistant if self.assistant is not None else self._pending
        return getattr(target, "runtime", None) if target is not None else None

    def one_shot_estop(self) -> bool:
        return _one_shot_estop(
            self.jetson_host, self.jetson_port, self.client_factory, self.timeout_s
        )

    def install(self) -> None:
        self._previous = _install_estop_on_sigint(
            self.runtime,
            fallback_estop=self.one_shot_estop,
            fallback_label=f"tcp://{self.jetson_host}:{self.jetson_port}",
        )
        self._installed = True

    def construct(self, assistant_cls: Any, **kwargs: Any) -> Any:
        """``assistant_cls(**kwargs)``, visible to the handler while it runs."""
        obj = assistant_cls.__new__(assistant_cls)
        self._pending = obj
        obj.__init__(**kwargs)
        self.assistant = obj
        self._pending = None
        return obj

    def release_partial(self) -> None:
        """Close the runtime of an Assistant whose construction did not finish."""
        if self.assistant is not None or self._pending is None:
            return
        runtime = getattr(self._pending, "runtime", None)
        self._pending = None
        close = getattr(runtime, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - shutting down anyway
            print(f"  (closing the half-built runtime failed: {exc})", flush=True)

    def restore(self) -> None:
        if self._installed:
            import signal

            if self._previous is not None:
                signal.signal(signal.SIGINT, self._previous)
            self._installed = False


def _autostart_fake_hardware(
    robot_host: str, robot_port: int, detector_host: str, detector_port: int,
    wait_seconds: float = 30.0,
) -> list:
    """Launch the two fake services this interpreter can run, if they are not up.

    Returns the Popen handles so the caller can stop what it started. Both
    run in this interpreter (``sys.executable``): they need only numpy, pyzmq
    and msgpack, all of which are already required to talk to them.
    """
    import subprocess

    spawned = []
    jobs = []
    if not _robot_server_is_up(robot_host, robot_port):
        jobs.append((
            "robot server",
            # --fake-watchdog-ms: the fake detaches itself after 500 ms without a
            # write, exactly like the Uno, so the fake lane proves the server's
            # keepalive rather than assuming it (review P1).
            [sys.executable, str(REPO_ROOT / "jetson" / "robot_server.py"),
             "--driver", "fake", "--fake-camera", "--fake-world", FAKE_SCENE,
             "--fake-watchdog-ms", str(FAKE_WATCHDOG_MS),
             "--host", robot_host, "--port", str(robot_port)],
            lambda: _robot_server_is_up(robot_host, robot_port),
        ))
    if not _server_is_up(detector_host, detector_port):
        jobs.append((
            "detector service",
            [sys.executable, str(REPO_ROOT / "jetson" / "detector_service.py"),
             "--backend", "scripted", "--scene", FAKE_SCENE,
             "--world-from", f"{robot_host}:{robot_port}",
             "--host", detector_host, "--port", str(detector_port)],
            lambda: _server_is_up(detector_host, detector_port),
        ))
    for name, command, probe in jobs:
        print(f"\n  Fake {name} not running. Starting it...")
        print(f"    {' '.join(command)}")
        try:
            spawned.append(subprocess.Popen(command, close_fds=True))
        except OSError as exc:
            print(f"  Could not launch the fake {name}: {exc}")
            continue
        deadline = time.time() + wait_seconds
        while time.time() < deadline and not probe():
            time.sleep(0.25)
        print(f"  fake {name} {'ready' if probe() else 'did not come up in time'}.")
    return spawned


def _find_llm_python(user_override: str | None = None) -> str:
    if user_override:
        return user_override
    env_override = os.environ.get("LLM_PYTHON")
    if env_override and Path(env_override).is_file():
        return env_override
    candidates = [
        Path.home() / "ml311" / "Scripts" / "python.exe",
        Path.home() / "qwen-venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable



def _apply_llm_mode(assistant: Any, mode: str) -> None:
    """Set the LLM parser's policy (``--llm-mode``) on a built Assistant.

    ``Assistant`` constructs ``LlmIntentParser`` itself; the planner holds the
    same parser object, so setting ``mode`` here changes what every command
    uses. Parsing has not started yet (construction only runs ``observe``).
    """
    from mfw.language.intent_parser import LlmIntentParser

    parser = getattr(assistant, "parser", None)
    if isinstance(parser, LlmIntentParser):
        parser.mode = mode
        print(f"  LLM parser mode   : {mode}"
              + (" (rules first; LLM only when the rules refuse)" if mode == "hybrid" else ""))


def _autostart_llm_server(
    llm_python: str,
    worker: Path,
    host: str,
    port: int,
    model: str,
    device: str,
    wait_seconds: float = 60.0,
) -> bool:
    import subprocess

    command = [
        llm_python,
        str(worker),
        "--model", model,
        "--device", device,
        "--host", host,
        "--port", str(port),
    ]
    print(f"\n  LLM server not running. Starting it on {device}...")
    print(f"    {' '.join(command)}")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    try:
        subprocess.Popen(command, creationflags=creationflags, close_fds=True)
    except OSError as exc:
        print(f"  Could not launch LLM server: {exc}")
        return False

    print("  Loading Qwen model in background...", end="", flush=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _server_is_up(host, port):
            print(" ready.")
            return True
        print(".", end="", flush=True)
        time.sleep(2.0)
    print("\n  LLM server did not come up in time.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the manipulation assistant")
    parser.add_argument(
        "--config",
        default=None,
        help="framework YAML (default: configs/default.yaml; configs/hardware.yaml under "
        "--hardware; configs/hardware_fake.yaml under --fake-hardware)",
    )
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="hardware lane: no Isaac Sim; drive the arm through the Jetson robot "
        "server and perceive through the detector service (backend: hardware)",
    )
    parser.add_argument(
        "--fake-hardware",
        action="store_true",
        help="hardware lane against loopback fakes (robot_server.py --driver fake, "
        "detector_service.py --backend scripted), autostarted if not running",
    )
    parser.add_argument(
        "--jetson",
        default=None,
        metavar="HOST[:PORT]",
        help="robot server address; overrides hardware.jetson_host/jetson_port",
    )
    parser.add_argument(
        "--detector-server",
        default=None,
        metavar="HOST:PORT",
        help="detector service address; overrides hardware.detector_host/detector_port",
    )
    parser.add_argument(
        "--llm-server",
        default=None,
        metavar="HOST:PORT",
        help="resident LLM worker for --llm (default 127.0.0.1:5557). A non-loopback "
        "host is never autostarted.",
    )
    parser.add_argument(
        "--groot",
        action="store_true",
        help="route motor skills (pick/place/move) to the GR00T VLA. Requires the "
        "policy server to be running. Perception, memory and safety stay classical.",
    )
    parser.add_argument(
        "--groot-skills",
        default="pick",
        help="comma-separated skills to hand to GR00T (default: pick)",
    )
    parser.add_argument(
        "--gpu-physics",
        action="store_true",
        help="run PhysX rigid-body dynamics and broadphase on the GPU. Rendering is "
        "always GPU. Worth it for scenes with hundreds of bodies; for a handful, "
        "kernel-launch and sync overhead usually makes it SLOWER than CPU.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the scripted demo (hardware lane: scan the room, what do you see, pick up "
        "the marker, put it in the bowl, go home)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="with -c/--demo on the hardware lane: run every command even after one fails "
        "(the exit status is still 1). Default there: stop at the first failure. The sim "
        "lane always runs every command",
    )
    parser.add_argument(
        "--home-confirmed",
        action="store_true",
        help="hardware lane: a person has hand-posed the arm (at home after a robot_server "
        "start; at its last pose when the servos are limp), E-stop in reach, hands clear, so the "
        "first home -- a full-speed attach -- may be sent without the typed 'home' prompt. "
        "Required when stdin is not a terminal (scripted or service runs); without it such a run "
        "refuses before moving anything",
    )
    parser.add_argument("--interactive", action="store_true", help="prompt for commands")
    parser.add_argument(
        "-c", "--command", action="append", default=[], help="run a command (repeatable)"
    )
    parser.add_argument("--voice", action="store_true", help="listen for spoken commands")
    parser.add_argument(
        "--asr",
        default="whisper",
        choices=["whisper", "canary"],
        help="speech engine. 'canary' is NVIDIA Canary-Qwen-2.5B (needs NeMo + GPU); "
        "'whisper' is lighter and reports a usable confidence score",
    )
    parser.add_argument(
        "--voice-python",
        default=None,
        help="interpreter for the speech worker. MUST NOT be Isaac's python: "
        "CTranslate2 crashes inside it and takes the simulator down. "
        "Auto-detected if omitted.",
    )
    parser.add_argument("--voice-model", default="base.en", help="whisper model")
    parser.add_argument(
        "--voice-mic",
        type=int,
        default=None,
        help="input device index for an auto-started speech server (see "
        "speech_worker.py --list-mics)",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="do not launch the speech server automatically if it is not running",
    )
    parser.add_argument(
        "--voice-device", default="cpu", choices=["cpu", "cuda"], help="whisper device"
    )
    parser.add_argument(
        "--voice-server",
        default=None,
        metavar="HOST:PORT",
        help="attach to a resident speech server instead of spawning a worker. "
        "The model stays loaded there, so startup drops from ~39s to milliseconds. "
        "Use 'default' for 127.0.0.1:5556.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="execute motion commands without asking. Off by default because "
        "Canary reports no confidence, so a misheard command cannot be filtered.",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="disable spoken replies (they are on by default in voice mode)",
    )
    parser.add_argument(
        "--voice-confidence",
        type=float,
        default=0.5,
        help="below this, a transcript is reported but NOT executed",
    )
    parser.add_argument(
        "--llm",
        default=None,
        metavar="MODEL_ID",
        help="activate the LLM intent parser with a HuggingFace model ID. "
        "Example: --llm qwen (shorthand for Qwen/Qwen2.5-3B-Instruct) or a "
        "full HF repo id. When omitted the default rule-based parser is used.",
    )
    parser.add_argument(
        "--llm-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device for the LLM intent model (default: cuda)",
    )
    parser.add_argument(
        "--llm-mode",
        default="hybrid",
        choices=["hybrid", "llm-first"],
        help="with --llm: 'hybrid' (default) runs the rule parser first and asks the LLM "
        "only when the rule parser refuses; the LLM never overrides a rule parse. "
        "'llm-first' asks the LLM first and falls back to the rules.",
    )
    args = parser.parse_args()

    hardware_mode = bool(args.hardware or args.fake_hardware)
    if args.config is None:
        default_yaml = (
            "hardware_fake.yaml" if args.fake_hardware
            else "hardware.yaml" if args.hardware
            else "default.yaml"
        )
        args.config = str(REPO_ROOT / "configs" / default_yaml)
    if hardware_mode and args.gui:
        print("\n  NOTE: --gui is ignored under --hardware; there is no simulator window to show.")

    # Is a resident speech server already listening? Answer this FIRST, because
    # it decides whether this process needs local ASR dependencies at all.
    _host, _port = "127.0.0.1", 5556
    server_already_up = False
    if args.voice and args.voice_server:
        spec = "127.0.0.1:5556" if args.voice_server == "default" else args.voice_server
        _host, _, _port_text = spec.partition(":")
        _host = _host or "127.0.0.1"
        _port = int(_port_text or 5556)
        server_already_up = _server_is_up(_host, _port)

    # Resolve the speech interpreter *before* booting the simulator: probing takes
    # a second, and a missing dependency should fail immediately rather than after
    # a 60 s Isaac Sim startup.
    #
    # Skipped entirely when a server is already up. In that case the model lives
    # in the server's process and this one only sends audio over TCP, so demanding
    # a local nemo/faster-whisper install blocks a perfectly valid setup -- a
    # server on another host, or a client machine with no ASR install at all.
    # It also fails for a confusing reason: --asr selects which dependency set is
    # probed, so connecting to a running Canary server without repeating
    # `--asr canary` reports missing *whisper* packages.
    # A speech server on another machine (the Jetson, :5556) is never started
    # from here -- launching a local worker bound to the Jetson's address can
    # only fail, and probing for a local ASR install would demand NeMo on a
    # laptop that needs none. The voice block below waits for it instead.
    remote_voice_server = bool(args.voice and args.voice_server and not _is_loopback(_host))
    if remote_voice_server and not server_already_up:
        print(f"\n  Speech server {_host}:{_port} is not answering yet (remote: not autostarted;"
              " start it there, e.g. jetson_mode.sh conversation).")

    voice_python = None
    if args.voice and not server_already_up and not remote_voice_server:
        # Needed either to spawn a one-shot worker, or to auto-start the resident
        # server when it is not already up.
        voice_python = _find_voice_python(args.voice_python, args.asr)

    if args.voice and args.voice_server and not args.no_autostart and not remote_voice_server:
        if not server_already_up:
            # Started before Isaac Sim so the ~39 s model load overlaps Isaac's
            # own ~60 s startup instead of running after it.
            _autostart_speech_server(
                voice_python=voice_python,
                worker=REPO_ROOT / "scripts" / "speech_worker.py",
                host=_host,
                port=_port,
                asr=args.asr,
                device=args.voice_device,
                mic=args.voice_mic,
                wait_seconds=0.0,  # do not block here; we re-check after Isaac boots
            )
        else:
            print(f"\n  Speech server already running at {_host}:{_port} (model warm).")

    # Must happen before SimulationApp is constructed.
    sys.argv = [sys.argv[0]]

    from mfw.config.schema import load_config

    overrides: dict = {
        "simulation": {"headless": not args.gui},
        "physics": {"gpu_dynamics": args.gpu_physics},
    }
    if args.groot:
        # Only the named motor skills are handed over. observe/stop/go_home stay
        # classical because a VLA emits motion, not a scene graph -- and
        # Gr00tExecutor.supports() declines them anyway.
        skills = [s.strip() for s in args.groot_skills.split(",") if s.strip()]
        overrides["gr00t"] = {"enabled": True, "use_mock_server": False}
        overrides["executor_overrides"] = {s: "gr00t" for s in skills}

    if hardware_mode:
        overrides["backend"] = "hardware"
        hw_overrides: dict = {}
        if args.jetson:
            host, port = _parse_host_port(args.jetson, "127.0.0.1", 5560)
            hw_overrides.update({"jetson_host": host, "jetson_port": port})
        if args.detector_server:
            host, port = _parse_host_port(args.detector_server, "127.0.0.1", 5558)
            hw_overrides.update({"detector_host": host, "detector_port": port})
        if args.fake_hardware:
            hw_overrides["arm_bridge"] = "fake"
        if hw_overrides:
            overrides["hardware"] = hw_overrides

    config = load_config(args.config, overrides=overrides)

    spawned_fakes: list = []
    if hardware_mode:
        hw = config.hardware
        if args.fake_hardware:
            spawned_fakes = _autostart_fake_hardware(
                hw.jetson_host, hw.jetson_port, hw.detector_host, hw.detector_port
            )
            if spawned_fakes:
                import atexit

                atexit.register(_terminate_spawned, spawned_fakes)
                print(f"  fakes    : spawned PIDs {[p.pid for p in spawned_fakes]} (terminated on exit)")
        print("=" * 72)
        print("  Manipulation Framework - hardware lane (no Isaac Sim)")
        print(f"  config   : {args.config}")
        print(f"  robot    : tcp://{hw.jetson_host}:{hw.jetson_port}  (bridge: {hw.arm_bridge})")
        print(f"  detector : {hw.detector_host}:{hw.detector_port}")
        print(f"  speech   : {_host}:{_port}" if args.voice else "  speech   : off (pass --voice)")
        print(f"  clock    : {'virtual (fake_clock)' if hw.fake_clock else 'wall'}")
        print(f"  backend  : {config.default_executor}")
        print(f"  labels   : {', '.join(hw.labels)}")
        print(f"  STOP     : {STOP_NOTICE.format(chunk=hw.trajectory_chunk_s)}")
        print("  1st home : " + (
            "pre-confirmed (--home-confirmed): the arm must already be hand-posed at home"
            if args.home_confirmed else
            "a real arm with no known position or limp servos asks you to type 'home' first"
        ))
        print("=" * 72)
    else:
        print("=" * 72)
        print("  Manipulation Framework - starting Isaac Sim")
        print(f"  config   : {args.config}")
        print(f"  display  : {'GUI window' if args.gui else 'headless'}")
        print(f"  rendering: GPU (RTX {config.simulation.renderer})")
        print(f"  physx    : {'GPU dynamics + GPU broadphase' if args.gpu_physics else 'CPU (TGS solver)'}")
        if args.groot:
            routed = ", ".join(f"{k}->{v}" for k, v in config.executor_overrides.items())
            print(f"  backend  : classical + GR00T ({routed})")
            print(f"  policy   : {config.gr00t.host}:{config.gr00t.port}")
        else:
            print(f"  backend  : {config.default_executor}")
        print(f"  objects  : {', '.join(o.name for o in config.scene.objects)}")
        print("=" * 72)
        print("\nFirst launch takes ~40-60s (loading Isaac Sim and fetching the Franka USD).\n")

    from mfw.assistant import Assistant, classify_confirmation

    # ---------------------------------------------------------------------------
    # Optional LLM intent parser (activated only when --llm is supplied)
    # ---------------------------------------------------------------------------
    #: Shorthand aliases so the user can type --llm qwen instead of the full
    #: HuggingFace repo id.
    _LLM_ALIASES = {
        "qwen": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    llm_complete = None
    if args.llm:
        model_id = _LLM_ALIASES.get(args.llm.lower(), args.llm)
        llm_host, llm_port = _parse_host_port(args.llm_server or "127.0.0.1:5557", "127.0.0.1", 5557)
        print(f"\n  LLM intent parser : {model_id} (device={args.llm_device}) at {llm_host}:{llm_port}")

        if not _server_is_up(llm_host, llm_port) and _is_loopback(llm_host):
            llm_py = _find_llm_python()
            _autostart_llm_server(
                llm_python=llm_py,
                worker=REPO_ROOT / "scripts" / "llm_worker.py",
                host=llm_host,
                port=llm_port,
                model=model_id,
                device=args.llm_device,
            )

        if _server_is_up(llm_host, llm_port):
            def llm_complete(prompt: str) -> str:
                import socket
                with socket.create_connection((llm_host, llm_port), timeout=60.0) as sock:
                    sock.sendall(json.dumps({"prompt": prompt}).encode("utf-8") + b"\n")
                    reader = sock.makefile("r", encoding="utf-8")
                    line = reader.readline()
                    data = json.loads(line)
                    if not data.get("ok"):
                        raise RuntimeError(data.get("error", "LLM inference failed"))
                    return data["text"]

            print(f"  Connected to resident LLM worker at {llm_host}:{llm_port} ({model_id})")
        else:
            print("  WARNING: LLM worker is unreachable; using rule-based parser.")
            llm_complete = None

    assistant = None
    exit_code = 0
    stop_guard: HardwareStopGuard | None = None

    def _route_if_hardware(clauses_of: Callable[[str], list], run_clause: Callable[[str], Any]):
        """The hardware lane's "look around" -> "scan the room" rewrite (see :func:`route_clauses`)."""
        if hardware_mode and assistant.runtime.skills.has("scan_scene"):
            return route_clauses(clauses_of, run_clause)
        return clauses_of, run_clause
    #: Caught below only on the hardware lane (an empty tuple catches nothing),
    #: so the sim lane never imports the hardware runtime.
    home_refused: tuple = ()
    previous_home_gate: Any = None
    if hardware_mode:
        # The first home after a robot_server start jumps a real arm to home at
        # full speed; the runtime asks the operator (TTY) or needs
        # --home-confirmed. Assistant builds the runtime itself, so the gate is
        # installed as the module default and restored in the finally below.
        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused, set_first_home_gate

        home_refused = (FirstHomeRefused,)
        previous_home_gate = set_first_home_gate(FirstHomeGate(confirmed=args.home_confirmed))
        # Ctrl-C estops first -- during bring-up too (HS-4) -- then exits through
        # the finally below.
        stop_guard = HardwareStopGuard(config.hardware.jetson_host, config.hardware.jetson_port)
        stop_guard.install()
    try:
        # Inside the try on purpose: if HardwareRuntime.build() raises (unreachable
        # Jetson, limits outside the pulse map, empty homography) the finally below
        # still terminates the fakes this run spawned instead of orphaning them.
        if stop_guard is not None:
            assistant = stop_guard.construct(Assistant, config=config, llm_complete=llm_complete)
        else:
            assistant = Assistant(config=config, llm_complete=llm_complete)
        if llm_complete is not None:
            _apply_llm_mode(assistant, args.llm_mode)
        state = assistant.describe()
        print("\n" + "-" * 72)
        print("  Robot ready")
        print(f"  skills   : {', '.join(state['skills'])}")
        print(f"  backends : {', '.join(state['backends'])}")
        print(f"  sees     : {len(state['objects'])} object(s)")
        for obj in state["objects"]:
            print(f"       {obj['label']:<8} at {obj['position']}")
        print("-" * 72)

        commands = list(args.command)
        if args.demo or (not commands and not args.interactive and not args.voice):
            if hardware_mode:
                commands = list(config.hardware.demo_script) or list(HARDWARE_DEMO_SCRIPT)
                print(f"\n  demo     : {' -> '.join(commands)}")
            else:
                commands = DEMO_SCRIPT

        if commands:
            reports, not_run = run_script(
                commands,
                lambda utterance: run_command(
                    utterance, *_route_if_hardware(assistant.clauses, assistant.command)
                ),
                # Stop-at-first-failure is a hardware-lane safety rule (every
                # further motion of a real arm is one more unobserved risk).
                # The sim lane keeps running every command, as it always did.
                keep_going=args.keep_going or not hardware_mode,
            )
            print("\n" + script_summary(reports, not_run))
            exit_code = lane_exit_code(hardware_mode, reports, not_run)

        if args.voice:
            from mfw.language.speech import (
                SpeechError,
                TcpSpeechRecognizer,
                VoiceCommandLoop,
                WhisperSubprocessRecognizer,
                speak,
            )

            talk = not args.no_speak
            worker = REPO_ROOT / "scripts" / "speech_worker.py"

            using_server = bool(args.voice_server)
            if using_server:
                spec = "127.0.0.1:5556" if args.voice_server == "default" else args.voice_server
                host, _, port_text = spec.partition(":")
                host = host or "127.0.0.1"
                port = int(port_text or 5556)

                # It may still be loading in its own window; Isaac's startup has
                # likely covered most of that, but wait out the remainder rather
                # than failing on a race.
                if not _server_is_up(host, port):
                    print("\n  Waiting for the speech server to finish loading", end="", flush=True)
                    deadline = time.time() + 300.0
                    while time.time() < deadline and not _server_is_up(host, port):
                        print(".", end="", flush=True)
                        time.sleep(2.0)
                    print(" ready." if _server_is_up(host, port) else " timed out.")

                recognizer = TcpSpeechRecognizer(host=host, port=port)
                transport = f"resident server at {host}:{port} (model already loaded)"
            else:
                recognizer = WhisperSubprocessRecognizer(
                    worker_script=worker,
                    python_executable=voice_python,
                    asr=args.asr,
                    model=args.voice_model,
                    device=args.voice_device,
                )
                transport = f"subprocess ({voice_python})"

            label = ENGINE_LABELS.get(args.asr, args.asr)
            print("\n" + "=" * 72)
            print("  VOICE MODE")
            print(f"  speech model  : {label}")
            # With a resident server, the model's device is whatever that process
            # chose at load time. Echoing this client's --voice-device would be a
            # plain lie: it said "cpu" while the server ran on CUDA.
            print(f"  device        : {'set by the server' if using_server else args.voice_device}")
            print(f"  transport     : {transport}")
            if args.asr == "canary":
                print("  NOTE: Canary reports no confidence, so the low-confidence")
                print("        gate cannot filter its output - every command executes.")
            print("=" * 72)

            def on_worker_event(event: str, fields: dict) -> None:
                """Surface worker status so the operator knows when to speak."""
                if event == "loading":
                    engine = fields.get("engine", args.asr)
                    print(f"\n  Loading {ENGINE_LABELS.get(engine, engine)}", flush=True)
                    print("  Reading ~4.8 GB from disk and building the model.", flush=True)
                    print("  Expect 60-150s. This is disk + CPU work, not GPU.", flush=True)
                elif event == "ready":
                    seconds = fields.get("chunk_seconds", 4)
                    print("\n" + "*" * 72)
                    print("  MICROPHONE IS LIVE - SPEAK NOW")
                    print(f"  You have {seconds:.0f} seconds per command. Speak clearly.")
                    print('  Try: "scan the room"  /  "pick up the marker"  /  "put it in the bowl"'
                          if hardware_mode else
                          '  Try: "what do you see"  /  "pick up the can"  /  "place it"')
                    print("  Say 'quit' to stop.")
                    print("*" * 72, flush=True)
                    speak("Robot ready. Give me your command.", talk)
                elif event == "calibrating":
                    print("\n  Measuring your room's noise floor - STAY QUIET (2s)...", flush=True)
                elif event == "calibrated":
                    print(
                        f"  noise floor: ambient {fields.get('ambient')}, "
                        f"peak {fields.get('peak')} -> threshold {fields.get('threshold')}",
                        flush=True,
                    )
                elif event == "too_quiet":
                    print(
                        f" too quiet ({fields.get('level')} < {fields.get('threshold')}), ignored",
                        flush=True,
                    )
                elif event == "listening":
                    print("\n>>> listening...", end="", flush=True)
                elif event == "speech_started":
                    print(" [speech detected]", end="", flush=True)
                elif event == "thinking":
                    print(f" got {fields.get('seconds')}s, transcribing...", end="", flush=True)
                elif event == "timing":
                    print(f" ({fields.get('inference_s')}s)", flush=True)
                elif event == "empty":
                    print("    (no speech recognised)", flush=True)
                elif event == "error":
                    print(f"    worker error: {fields.get('message')}", flush=True)

            recognizer.on_event = on_worker_event

            def on_unsure(transcript) -> None:
                print(
                    f'\n  ?? heard "{transcript.text}" '
                    f"(confidence {transcript.confidence:.2f}) - NOT executing"
                )
                speak("Sorry, I did not catch that.", talk)

            def ask_by_voice(question: str) -> str | None:
                """Speak the question and take the next utterance as the answer."""
                print(f"\n  {question}", flush=True)
                speak(question, talk, blocking=True)
                try:
                    return _listen_for_reply(recognizer)
                except Exception as exc:  # noqa: BLE001 - fall back to keyboard
                    print(f"  (voice answer unavailable: {exc})", flush=True)
                    return _typed_ask(question)

            def tell(text: str) -> None:
                print(f"  {text}", flush=True)
                speak(text, talk)

            def handle_spoken(text: str) -> None:
                print(f'\n  [heard] "{text}"')
                report = run_command(
                    text,
                    *_route_if_hardware(
                        assistant.clauses,
                        lambda clause: assistant.command_with_clarification(
                            clause, ask_by_voice, notify=tell
                        ),
                    ),
                )
                _print_report(report)
                outcome = report.final
                # Spoken confirmation, so you can keep your eyes on the robot
                # rather than the console.
                speak(outcome.message or ("Done." if outcome.ok else "I could not do that."), talk)

            def confirm_command(text: str) -> bool:
                """Ask before moving.

                This is the safety gate Canary cannot provide: it reports a
                constant confidence of 1.0, so a misheard command is
                indistinguishable from a clear one and would otherwise execute
                unchallenged. Your own log showed "Place the can on" acted upon
                as a truncated fragment.

                Whether to ask is decided by the PARSED skill of every clause
                (``Assistant.needs_confirmation``), not by keywords: the audit
                found the old word list let "get/fetch the X", "set it",
                "shift/nudge", "turn", "survey", "look at", "return home",
                "squeeze" and "let go" through unconfirmed.
                """
                if hardware_mode:
                    # "look around" moves the base once routed to scan_scene.
                    text = route_hardware_phrase(text)
                if args.no_confirm or not assistant.needs_confirmation(text):
                    return True

                print(f'\n  CONFIRM: "{text}"')
                speak(f"Did you say {text}?", talk)
                # Confirm by VOICE, falling back to the keyboard.
                #
                # input() blocks the whole loop, and in voice mode that is not a
                # pause -- it is a hang with the microphone still live. The
                # recogniser keeps capturing and transcribing, and every result
                # is discarded because nothing can consume it. Observed as the
                # session going deaf after the very first command, while the log
                # cheerfully reported "transcribing... (0.55s)" for each phrase
                # it was throwing away.
                #
                # Hands-free operation cannot depend on a keyboard.
                print("  Say YES or NO   (or press Enter=yes / n=no)", flush=True)
                for _ in range(3):
                    try:
                        heard = _listen_for_reply(recognizer, attempts=1)
                    except Exception as exc:  # noqa: BLE001 - fall back to keyboard
                        print(f"  (voice confirmation unavailable: {exc})", flush=True)
                        break
                    if heard is None:
                        continue
                    # Whole words: "I know" is not "no", "yesterday" is not "yes".
                    verdict = classify_confirmation(heard)
                    if verdict is True:
                        return True
                    if verdict is False:
                        print("  Cancelled.")
                        speak("Cancelled.", talk)
                        return False
                    print("  That was neither yes nor no.", flush=True)

                try:
                    answer = input("  Execute? [Enter=yes / n=no] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return False

                if answer in ("", "y", "yes"):
                    return True
                print("  Cancelled.")
                speak("Cancelled.", talk)
                return False

            def report_dropped(count: int) -> None:
                print(f"  (discarded {count} phrase(s) heard while the robot was busy)")

            loop = VoiceCommandLoop(
                recognizer,
                handle_spoken,
                min_confidence=args.voice_confidence,
                on_low_confidence=on_unsure,
                confirm=confirm_command,
                on_dropped=report_dropped,
            )
            try:
                spoken = loop.run()
                print(f"\nExecuted {spoken} spoken command(s).")
            except SpeechError as exc:
                # Print it here rather than letting it propagate. Isaac's
                # fastShutdown tears the process down before an unhandled
                # traceback reaches the terminal, so the session appears to exit
                # silently for no reason -- which is exactly what a missing speech
                # server looked like.
                print("\n" + "!" * 72)
                print("  VOICE SETUP FAILED")
                print("!" * 72)
                print(f"  {exc}\n")
                if using_server:
                    print("  Start the speech server FIRST, in a separate terminal:\n")
                    print("    cd manipulation_framework")
                    print("    <voice-venv>\\Scripts\\python.exe "
                          "scripts\\speech_worker.py \\")
                    print("        --serve --asr canary --device cuda --mic 1\n")
                    print("  Wait for 'speech server listening', then re-run this command.")
                    print("  Or drop --voice-server to spawn a worker here (slower startup).")
                print("!" * 72)

        if args.interactive:
            print("\n" + "=" * 72)
            print("  Interactive mode. Try:")
            if hardware_mode:
                print('    "scan the room"        "what do you see"     "pick up the marker"')
                print('    "put it in the bowl"   "open the gripper"    "go home"')
            else:
                print('    "what do you see"      "pick up the can"     "place it"')
                print('    "move left 5 cm"       "open the gripper"    "go home"')
            print("  Type 'quit' to exit.")
            print("=" * 72)
            while True:
                try:
                    utterance = input("\nrobot> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not utterance:
                    continue
                if utterance.lower() in ("quit", "exit", "q"):
                    break
                # An ambiguous referent is answered at the same prompt: "Which
                # one do you mean: the red block, the blue can or the green
                # box?" -> "the red one" re-runs the command on that object;
                # "cancel" / "never mind" abandons it.
                _print_report(
                    run_command(
                        utterance,
                        *_route_if_hardware(
                            assistant.clauses,
                            lambda clause: assistant.command_with_clarification(
                                clause, _typed_ask, notify=lambda text: print(f"  {text}")
                            ),
                        ),
                    )
                )

        print(f"\nEvent log: {assistant.runtime.events.path}")
        print("Done.")
    except KeyboardInterrupt:
        # The SIGINT handler already sent the estop on the hardware lane.
        if stop_guard is not None and stop_guard.bringing_up:
            print("\nInterrupted during bring-up (connecting / boot homing). An estop was "
                  "sent to the robot server -- the line above says whether it was "
                  "acknowledged; if it was not, cut the +6 V servo supply. An "
                  "acknowledged estop keeps the servos detached until clear_estop "
                  "(`scripts/calibrate_servos.py --jetson <ip>:5560` -> `clear`).")
        else:
            print("\nInterrupted.")
        exit_code = 130
    except home_refused as exc:
        # Nothing was sent to the arm: say why and how to proceed, no traceback.
        print("\n" + "!" * 72)
        print("  FIRST HOME NOT SENT")
        print("!" * 72)
        print(f"  {exc}")
        print("!" * 72)
        exit_code = 1
    finally:
        if previous_home_gate is not None:
            from mfw.hardware.runtime import set_first_home_gate

            set_first_home_gate(previous_home_gate)
        if assistant is not None:
            assistant.close()
        elif stop_guard is not None:
            # Construction did not finish: release its sockets and heartbeat.
            stop_guard.release_partial()
        # Only what this run started; a fake server the operator launched by
        # hand is theirs to keep.
        _terminate_spawned(spawned_fakes)
        if stop_guard is not None:
            stop_guard.restore()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

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

``sys.argv`` is cleared before the simulator starts: ``SimulationApp`` parses argv
itself and exits on flags it does not recognise, so our own options must not still
be there when it initialises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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


def _print_outcome(utterance: str, outcome) -> None:
    """Show what the robot understood, decided and did."""
    status = "OK " if outcome.ok else "-- "
    print(f"\n{status} \"{utterance}\"")
    print(f"    skill    : {outcome.skill or '(not understood)'}")
    if outcome.params:
        print(f"    params   : {outcome.params}")
    print(f"    result   : {outcome.message or '(no message)'}")

    if outcome.needs_clarification and outcome.clarification_options:
        print(f"    ambiguous: {', '.join(outcome.clarification_options)}")

    if outcome.action_graph:
        print(f"    pipeline : {' -> '.join(outcome.action_graph)}")

    # Surface the perceived scene, since it is the whole basis for the decision.
    if outcome.result is not None and outcome.result.data.get("objects"):
        objects = outcome.result.data["objects"]
        if isinstance(objects, list):
            print("    perceived:")
            for obj in objects:
                print(
                    f"       {obj['label']:<8} at {obj['position']} "
                    f"size {obj['size']} conf {obj['confidence']}"
                )

    if outcome.attempts > 1:
        print(f"    attempts : {outcome.attempts}")
    print(f"    took     : {outcome.duration_s:.2f}s")


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
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
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
    parser.add_argument("--demo", action="store_true", help="run the scripted demo")
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
    args = parser.parse_args()

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
    voice_python = None
    if args.voice and not server_already_up:
        # Needed either to spawn a one-shot worker, or to auto-start the resident
        # server when it is not already up.
        voice_python = _find_voice_python(args.voice_python, args.asr)

    if args.voice and args.voice_server and not args.no_autostart:
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

    config = load_config(args.config, overrides=overrides)

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

    from mfw.assistant import Assistant

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
        llm_host, llm_port = "127.0.0.1", 5557
        print(f"\n  LLM intent parser : {model_id} (device={args.llm_device})")

        if not _server_is_up(llm_host, llm_port):
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

    assistant = Assistant(config=config, llm_complete=llm_complete)
    try:
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
            commands = DEMO_SCRIPT

        for utterance in commands:
            _print_outcome(utterance, assistant.command(utterance))

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
                    print('  Try: "what do you see"  /  "pick up the can"  /  "place it"')
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

            def handle_spoken(text: str) -> None:
                print(f'\n  [heard] "{text}"')
                outcome = assistant.command(text)
                _print_outcome(text, outcome)
                # Spoken confirmation, so you can keep your eyes on the robot
                # rather than the console.
                speak(outcome.message or ("Done." if outcome.ok else "I could not do that."), talk)

            #: Commands that physically move the robot. Only these need
            #: confirming -- asking before "what do you see" would be pure
            #: friction, since observing changes nothing in the world.
            MOTION_WORDS = (
                "pick", "place", "move", "go home", "rotate", "open", "close",
                "grab", "take", "put", "lift", "drop", "release",
            )

            def needs_confirmation(text: str) -> bool:
                lowered = text.lower()
                return any(word in lowered for word in MOTION_WORDS)

            def confirm_command(text: str) -> bool:
                """Ask before moving.

                This is the safety gate Canary cannot provide: it reports a
                constant confidence of 1.0, so a misheard command is
                indistinguishable from a clear one and would otherwise execute
                unchallenged. Your own log showed "Place the can on" acted upon
                as a truncated fragment.
                """
                if args.no_confirm or not needs_confirmation(text):
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
                        reply = recognizer.listen()
                    except Exception as exc:  # noqa: BLE001 - fall back to keyboard
                        print(f"  (voice confirmation unavailable: {exc})", flush=True)
                        break
                    if reply is None:
                        continue
                    heard = str(getattr(reply, "text", reply)).strip().lower()
                    if not heard:
                        continue
                    print(f'  heard: "{heard}"', flush=True)
                    if any(w in heard for w in ("yes", "yeah", "yep", "correct", "go ahead")):
                        return True
                    if any(w in heard for w in ("no", "nope", "cancel", "stop", "wrong")):
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
                _print_outcome(utterance, assistant.command(utterance))

        print(f"\nEvent log: {assistant.runtime.events.path}")
        print("Done.")
    finally:
        assistant.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

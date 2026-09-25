# Multimodal Conversational Robotic Arm for Natural Language-Based Object Manipulation

**Capstone Project — Team 178**

---

## Overview

A Franka Emika Panda arm in NVIDIA Isaac Sim 5.1 takes typed (or transcribed spoken) English
commands such as "pick the red object" or "put it in the bowl", finds the named object with its
two simulated RGB-D cameras, grasps it through real PhysX contact, and places it where it was told.
The language layer resolves colour, size, position and synonyms, asks "which one?" when a reference
is ambiguous, and refuses honestly when an object is absent or too wide to grasp. A GR00T N1.7
vision-language-action policy can be swapped in for the pick skill as an optional, out-of-process
backend; it runs but does not yet complete tasks (see [GR00T / VLA status](#gr00t--vla-status)).

This branch (`capstone-completion`) contains the **simulation lane only**. A physical-hardware lane
is in development and is not part of this branch; the `--hardware`, `--fake-hardware`, `--jetson`
and `--detector-server` flags that `run_assistant.py --help` shows belong to it and do not work here.

## Status at a glance

Legend: **IMPLEMENTED (verified)** = code present and exercised in a recorded run or test on
2026-09-24/25 · **IMPLEMENTED (not verified)** = code present, not exercised in that record ·
**NOT CURRENTLY OPERATIONAL** = runs but does not achieve the task · **PLANNED** = not in the code.
Evidence files are listed in [SYSTEM_VERIFICATION.md](SYSTEM_VERIFICATION.md).

| Capability | Status | Evidence |
|---|---|---|
| Text commands (`-c`) | IMPLEMENTED (verified) | Default and benchmark scene runs, `logs/e2e/phase8_default.txt`, `phase8_benchmark_rerun.txt` |
| Interactive prompt (`--interactive`) | IMPLEMENTED (not verified) | Same parser and planner as `-c`, plus an answer-at-the-prompt clarification loop; no interactive session in the record |
| Rule-based intent parser (default) | IMPLEMENTED (verified) | Used by every recorded run; `"dance"` -> "could not understand" |
| Voice / ASR on recorded audio | IMPLEMENTED (verified) | 5 synthesised WAV commands: Canary-Qwen-2.5B 5/5 exact, faster-whisper `small.en` 5/5; Canary transcripts executed in Isaac (`logs/e2e/voice_transcripts.txt`) |
| Voice with a live microphone | IMPLEMENTED (not verified) | No live-mic session recorded (microphone muted at OS level) |
| LLM intent parser (`--llm qwen`, Qwen2.5-3B-Instruct) | IMPLEMENTED (not verified) | Falls back to the rule parser on any failure; model not present on the test machine, path never exercised |
| Grounding: colour | IMPLEMENTED (verified) | "pick the red object" -> red block; "pick the yellow object" -> banana |
| Grounding: size | IMPLEMENTED (verified) | "pick the smallest object" -> block; "pick the large object" -> box (then refused as too wide) |
| Grounding: position | IMPLEMENTED (verified) | "pick the object on the right" -> block |
| Grounding: relations ("the block next to the can") | IMPLEMENTED (not verified) | Unit tests only (`tests/test_grounding.py`); not exercised in Isaac |
| Grounding: synonyms | IMPLEMENTED (verified) | "pick the blue cube" -> "the only block is the red block" |
| Grounding: ambiguity -> question | IMPLEMENTED (verified) | "pick the object" (3 in view) -> "Which one do you mean: ...?", no motion |
| Perception (Isaac instance segmentation + depth) | IMPLEMENTED (verified) | First "what do you see" names all objects with correct colours on both scenes |
| Pick | IMPLEMENTED (verified) | Benchmark: 5/5 lifted by ground truth; default scene picks |
| Place: in / on / next to / direction / back | IMPLEMENTED (verified) | in bowl 25 mm; on box 25 mm; next to 13-45 mm; 10 cm left 42 mm; back 22-28 mm |
| Go home | IMPLEMENTED (verified) | `go home` -> "at home posture" in the default, benchmark, voice and GR00T runs |
| Safety checks: workspace limit, trajectory abort, too-wide refusal, GR00T action filter | IMPLEMENTED (verified) | Isaac tests `test_workspace_violation_is_rejected`, `test_stop_aborts_execution`; refusal of the 74 mm box; GR00T run 0 clamped / 0 rejected; filter unit tests in `tests/test_gr00t_logic.py` |
| Safety checks: path-deviation and stall aborts | IMPLEMENTED (not verified) | In `mfw/controllers/joint_controller.py`; no dedicated test or recorded trigger |
| Stop / emergency stop during a running skill | PLANNED | Currently processed between commands only |
| Manipulation benchmark (`benchmark_manipulation.py --place`) | IMPLEMENTED (verified) | 6/6 perceived, 5/5 lifted, 5/5 released; 4/5 within the 50 mm place tolerance (soup can settled 70 mm away, reported as a miss by Place); `logs/benchmark/20260924_121023_manipulation.json` |
| GR00T N1.7 bridge and inference | IMPLEMENTED (verified) | Server loads the checkpoint, inference round trips, actions move the arm through the safety filter |
| GR00T N1.7 task success | NOT CURRENTLY OPERATIONAL | Zero-shot pick 0/2 (the planner retried each failed pick 3 times); failure reported honestly |
| CI | IMPLEMENTED (verified) | `.github/workflows/tests.yml` runs the pure-logic suite on Ubuntu and Windows; commit `38cfee9` passed both jobs (GitHub Actions runs 36095172081 and 36095172011) |
| Fine-tuned GR00T, learned detector | PLANNED | See [Future work](#future-work) |

## Architecture

```
 text (-c / --interactive)      voice (speech worker, own interpreter, TCP)
              \                  /
               v                v
        intent parser  (rule-based default | optional LLM worker on :5557)
               |
               v
        task planner + state machine  <-->  working memory (held object, last referent)
               |         one atomic skill per clause, then WAIT
               v
        executor per skill:  classical (default)  |  GR00T (optional, out of process, ZeroMQ :5555)
               |
               v
        skills: observe, pick, place, go_home, move_relative, ...
               |        grounding resolves "the red one" against the scene graph
               v
        grasp synthesis -> motion (Lula RRT + straight-line Cartesian) -> joint controller
               |
               v
        Isaac Sim 5.1 / PhysX  (Franka Panda, two RGB-D cameras)
               |
               v
        perception: instance segmentation + depth -> per-object point clouds
                    -> pose, size, colour -> tracker -> scene graph  (feeds planner and skills)
```

Design details, the state machine and the recovery policy are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

**Hardware**
- NVIDIA RTX GPU. Recorded runs used an RTX 5090 Laptop GPU (24 GB). Isaac + GR00T together
  peaked at 9,976 MiB; Canary ASR in bf16 uses about 5.8 GB.
- NVIDIA driver: recorded runs used **577.03**. Driver **610.88** crashed Isaac Sim in
  `rtx.scenedb` on the development laptop (9/9 runs); see [Troubleshooting](#troubleshooting).

**Software**
- Windows 11 and Isaac Sim **5.1.0 standalone**. Isaac scripts run under Isaac's own Python 3.11
  (`..\python.bat`), so the repository must sit **inside** the Isaac Sim folder, e.g.
  `isaac-sim-standalone-5.1.0-windows-x86_64\capstone-team-178\`.
- Python **3.12** (`py -3.12`) for the pure-logic tests and `scripts/tts_to_wav.py`; speech and
  GR00T each use their own interpreter (see [HANDOFF.md](HANDOFF.md)).
- Optional: a speech environment (faster-whisper, or NeMo for Canary in a `canary-venv`).
- Optional: a separate Python 3.12 GR00T environment (`groot_env`) with the Isaac-GR00T package
  and the `nvidia/GR00T-N1.7-3B` checkpoint in the HuggingFace cache.

`requirements.txt` lists what each interpreter needs; only block A installs with `pip`. Blocks C
and G refer to files that are not on this branch, and for Canary follow [Voice mode](#running) below
(the canary venv's own python), not block E.

## Installation & setup

```powershell
cd C:\isaac-sim-standalone-5.1.0-windows-x86_64
git clone https://github.com/hirenmp10/capstone-team-178.git
cd capstone-team-178
git checkout capstone-completion

# Pure-logic test dependencies (Python 3.12). Install nothing into Isaac's Python.
py -3.12 -m pip install -r requirements.txt
```

Isaac Sim needs no extra packages.

Optional environments:
- **Whisper**: `pip install faster-whisper sounddevice` into a system Python (never Isaac's).
- **Canary**: NeMo (from git) in `$env:USERPROFILE\canary-venv`, with a CUDA torch that works on your
  driver (on driver 577.03 that was the venv's torch 2.11+cu128).
- **LLM parser**: `torch`, `transformers`, `accelerate` in a separate interpreter, plus the
  `Qwen/Qwen2.5-3B-Instruct` weights.
- **GR00T**: see [GR00T / VLA status](#gr00t--vla-status).

## Running

All commands run from the repository folder. Usage strings inside some scripts still show an older
`manipulation_framework\scripts\...` path; ignore them and run from the repository folder as shown
here.

**Default scene** (`configs/default.yaml`: red block, blue can, green box):

```powershell
..\python.bat scripts\run_assistant.py -c "what do you see" -c "pick the red object" -c "place it back" -c "go home"
```

**Benchmark scene** (`configs/benchmark.yaml`: YCB soup can, foam brick, pudding box, banana, marker,
bowl, sorting bin, office furniture):

```powershell
..\python.bat scripts\run_assistant.py --config configs/benchmark.yaml --gui -c "what do you see" -c "pick up the marker" -c "put it in the bowl" -c "go home"
```

- `-c "<command>"` runs one command; repeat it to run several in order, then exit.
- `--interactive` opens a prompt for commands, after any `-c` commands have run.
- With none of `-c`, `--interactive` or `--voice`, a built-in demo sequence runs and moves the arm
  (what do you see, open the gripper, pick up the block, move up 5 cm, place it, go home);
  `--demo` forces it.
- `--gui` shows the Isaac Sim window; without it the run is headless.
- Each run writes an event log to `logs/<timestamp>_<id>/events.jsonl`.

**Manipulation benchmark** (pick every object, then put it back where it was picked):

```powershell
..\python.bat scripts\benchmark_manipulation.py --config configs/benchmark.yaml --place
```

Success is judged from simulator ground truth (the object rises more than 3 cm, then is released
and rests), not from the skill's own report. Results go to
`logs/benchmark/<timestamp>_manipulation.json` and `.csv` (config SHA-256, git commit, per-object
outcome, timings), with the latest copy in `logs/manipulation_benchmark.json` and a text log in
`logs/manipulation_benchmark.txt`. `--place-on bowl` places on a destination instead; `--only` limits
the objects.

**Voice mode** (IMPLEMENTED, not verified with a live microphone). Start the speech server in its own
interpreter, then attach the assistant to it:

```powershell
# 1. Canary on the GPU, using the canary venv's own python (see Troubleshooting)
& "$env:USERPROFILE\canary-venv\Scripts\python.exe" scripts\speech_worker.py --serve --asr canary --device cuda --mic 1

# 2. The assistant, attached to the server on 127.0.0.1:5556
..\python.bat scripts\run_assistant.py --voice --voice-server default --asr canary
```

- List input devices with `speech_worker.py --list-mics`. On the development machine only MME
  devices (`--mic 1`, `--mic 2`) delivered audio; DirectSound devices returned silence.
- Motion commands heard by voice are confirmed before they run (`--no-confirm` disables this),
  because Canary reports no per-utterance confidence.
- Whisper alternative: `--asr whisper` (CPU by default; reports a confidence score). The default
  model is `base.en`, which was not verified; the recorded 5/5 used `small.en`
  (`speech_worker.py --asr whisper --model small.en`, or `run_assistant.py --voice --asr whisper
  --voice-model small.en` when the assistant spawns its own worker).

**Offline ASR check** (verified path; no microphone needed):

```powershell
py -3.12 scripts\tts_to_wav.py "pick up the marker" logs\wav\pick.wav
& "$env:USERPROFILE\canary-venv\Scripts\python.exe" scripts\speech_worker.py --asr canary --device cuda --wav logs\wav\pick.wav
```

## Example commands

All of these were run in Isaac Sim and their outcome recorded (2026-09-24/25).

| Group | Command (scene) | Recorded outcome |
|---|---|---|
| See | `what do you see` (default) | "observed 3 object(s): red block, blue can, green box" |
| See | `what do you see` (benchmark) | red brick, grey box, grey marker, red can, yellow banana, red bowl |
| Pick by name | `pick up the marker` (benchmark) | held |
| Pick by name | `pick the blue can` (default) | held |
| Pick by colour | `pick the red object` (default) | the red block |
| Pick by colour | `pick the yellow object` (benchmark) | the banana |
| Pick by size | `pick the smallest object` (default) | the block |
| Pick by position | `pick the object on the right` (default) | the block |
| Place in | `put it in the bowl` (benchmark) | placed in the red bowl, settled 25 mm |
| Place next to | `place it next to the green box` (default) | settled 45 mm from target |
| Place direction | `put it on the left` (default) | placed 10 cm left of the pick point, 42 mm from target |
| Place back | `place it back` | 22-28 mm from where it was picked |
| Multi-step | `move the red block onto the green box` (default) | 2 clauses, both OK, settled 25 mm |
| Multi-step | `move the brick next to the bowl` (benchmark) | 2 clauses, both OK, settled 13 mm |
| Clarification | `pick the object` (default, 3 objects) | "Which one do you mean: the green box, the blue can or the red block?" |
| Clarification | `pick the red object` (benchmark, 3 red objects) | asks which one |
| Refusal | `pick the blue cube` (default) | "cannot find 'blue cube': the only block is the red block; currently visible: ..." |
| Refusal | `pick the large object` (default) | box is 74 mm at its narrowest, gripper spans 78 mm: too wide to grasp safely |
| Refusal | `pick the purple dragon` (default) | nothing in view is a dragon |
| Refusal | `dance` | "could not understand 'dance'" |
| Go home | `go home` | at home posture |

## GR00T / VLA status

STATUS (2026-09-25, commit 8c397b3):

| Item | State |
|---|---|
| Code | `mfw/gr00t_bridge/` (executor, observation builder, safety filter, ZeroMQ/msgpack client), `scripts/groot_server.py`, `scripts/start_groot_server.ps1`. Tests `test_gr00t_logic.py`, `test_gr00t_zmq.py`, `test_gr00t_honesty.py` validate the bridge, not model quality. |
| Config | Disabled by default (`gr00t.enabled: false`, `default_executor: classical`). Enabled per run with `--groot --groot-skills pick`. |
| Checkpoint | `nvidia/GR00T-N1.7-3B` (6.5 GB) + `nvidia/Cosmos-Reason2-2B` backbone (4.6 GB) in the local HuggingFace cache of the development machine. Not fine-tuned: the `oxe_droid` embodiment was trained on DROID data, never on this robot or scene. |
| Environment | Separate Python 3.12 venv (`groot_env`, torch 2.9.0+cu128). Needs `PYTHONPYCACHEPREFIX`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `GROOT_PATCH_MISTRAL=1` and a local snapshot path, scoped to the server process only. `start_groot_server.ps1` does this. |
| Inference | Executes. Checkpoint loaded, "Serving on 127.0.0.1:5555" after 72 s; Isaac connected with 0 import errors. |
| Robot integration | The policy's actions drive the simulated Franka through the safety filter: net TCP travel 0.234 m (blue can) and 0.224 m (red block), largest single step 4.2 mm, 0 clamped, 0 rejected. GPU peak 9,976 MiB. |
| End to end | **0/2 zero-shot picks.** The arm did not approach the targets (drifted away from the can; moved toward the block's side but never descended) and the policy never commanded the gripper to close (peak 0.006). The executor reports "no object grasped" and a following "place it back" is refused. The planner retried each failed pick 3 times (36 iteration events per command). DROID frame conventions vs the Isaac world frame are not verified. |

Launch (two terminals, from the repository folder):

```powershell
# 1. Policy server (GR00T venv, environment scoped to this process)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_groot_server.ps1
#    -DryRun prints the resolved command and environment without starting anything

# 2. Isaac with pick routed to GR00T (other skills stay classical)
..\python.bat scripts\run_assistant.py --groot --groot-skills pick -c "what do you see" -c "pick up the blue can"
```

If the server is unreachable the assistant logs a warning and continues with the classical backend.

## Testing

**Pure-logic suite** (no GPU, no Isaac; Python 3.12, run from the repository root):

```powershell
py -3.12 -m pytest tests -q -m "not isaac" -p no:cacheprovider
```

Recorded result: **1069 passed, 3 skipped** (the 3 skips need files that are not on this branch).

**Isaac integration suite** (99 tests, about 6 minutes):

```powershell
..\python.bat scripts\run_isaac_tests.py
```

Recorded result: **99/99 passed**; report in `logs/isaac_test_report.txt` (the previous report is kept
as `.prev.txt`). Do not run the Isaac tests with plain `pytest`: `SimulationApp` parses `sys.argv`
and exits on pytest's flags, and Isaac swallows stdout, so the runner calls `pytest.main`
in-process and writes the report to a file.

**CI**: `.github/workflows/tests.yml` runs the pure-logic suite on Ubuntu and Windows with Python
3.12 on every push and pull request; commit `38cfee9` passed on both. The Isaac suite cannot run on hosted
runners.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Isaac Sim crashes in `rtx.scenedb` on start-up or during a run | NVIDIA driver 610.88 (9/9 crashes on the development laptop). Use a 577.03 / 592.27-class driver. |
| Canary loads on the CPU, or `serve_speech.py` exits with status 4 | The base Python's torch is a CUDA 13 build that sees no GPU on driver 577.03. Run `speech_worker.py` with `$env:USERPROFILE\canary-venv\Scripts\python.exe` (torch 2.11+cu128). |
| Speech server hears nothing | The microphone is muted at the OS level, or a DirectSound device was chosen. Use an MME device (`--mic 1` / `--mic 2`); `speech_worker.py --scan-mics` and `--check-mic` help find one. |
| Second speech server exits immediately / port 5556 in use | Intended: the server claims its port exclusively before loading a model, so a duplicate cannot load a second copy or open the mic. Stop the other server first. |
| Isaac fails to import extensions after GR00T work | GR00T's environment variables (especially `PYTHONPYCACHEPREFIX`) leaked into the Isaac shell. Start the server only through `start_groot_server.ps1`, which sets them for the child process alone, and never export them in the shell that runs `python.bat`. |
| First "what do you see" reported black objects | Fixed on this branch: start-up now observes until object colours agree across frames, and every observe renders 8 frames before reading the cameras. If it recurs, check that `perception.render_frames_before_capture` has not been lowered. |
| A script "does nothing" and exits 0 | Isaac kills the process with exit code 0 on some USD errors and discards `print` output. Check the log files the scripts write under `logs/`. |

## Known limitations

- Perception uses Isaac's ground-truth instance segmentation and semantic labels plus depth, not a
  learned detector. Pose, size and colour are estimated from the segmented depth points; single-view
  bias can offset a large box's centre by about 5 cm. This would not carry over to a real camera as is.
- Stop and emergency stop are processed between commands; they cannot interrupt a skill that is
  already executing.
- A long object gripped far from its centre can miss the place tolerance (50 mm). Place reports the
  miss: in one benchmark run the banana, gripped 74 mm off-centre, landed 95 mm from target and the
  next pick found no path; the re-run succeeded. In the recorded `--place` benchmark the soup can
  settled 70 mm from where it was picked and Place reported it as a miss (4/5 within tolerance).
- "Next to" placement picks a free side; observed accuracy 13-45 mm.
- Objects wider than 68 mm at their narrowest are refused (78 mm grasp limit minus a 10 mm margin).
- GR00T zero-shot task success is 0/2; the LLM parser and live microphone use are not verified.
- Evidence logs under `logs/` are git-ignored; they exist on the verification machine only.

## Future work

- Fine-tune GR00T N1.7 on this embodiment and scene, and verify its frame conventions against Isaac.
- Replace ground-truth segmentation with a learned detector.
- Validate a live-microphone voice session end to end.
- Validate the LLM intent parser (install the model, measure accuracy against the rule parser).
- Let stop / emergency stop interrupt a running skill.

## Repository structure

```
.github/workflows/tests.yml   CI: pure-logic suite (Ubuntu + Windows, Python 3.12)
configs/
  default.yaml                base config and default scene (red block, blue can, green box)
  benchmark.yaml              office scene with YCB objects (extends default.yaml)
  assets.yaml                 measured asset catalogue
  benchmark_*.yaml, test_*_moved.yaml   diagnostic scene variants
mfw/                          the framework package
  assistant.py                top-level wiring; clause splitting, clarification, confirmation
  config/                     typed, strict YAML schema (schema.py)
  core/                       data types, interfaces, error taxonomy
  simulation/                 Isaac app lifecycle, scene builder, runtime bring-up, scene validation
  robot/                      Franka wrapper and TCP (franka.py)
  physics/                    PhysX materials, contact-based grasp verification, colliders
  vision/                     cameras, point-cloud geometry, tracking, scene graph, VisionManager
  grasp/                      grasp candidate generation and scoring
  motion/                     Lula RRT + Cartesian planner, trajectory timing
  controllers/                joint trajectory controller
  skills/                     atomic skills and the classical executor
  planner/                    task planner and state machine
  memory/                     working memory and pronoun resolution
  language/                   intent parser, grounding, speech front end
  gr00t_bridge/               GR00T executor, observation builder, safety filter, clients, mock server
  isaaclab_ext/               empty placeholder
  utils/                      transforms, logging
scripts/
  run_assistant.py            main entry point (text, interactive, voice, --groot)
  benchmark_manipulation.py   ground-truth pick/place benchmark
  run_isaac_tests.py          Isaac test runner
  speech_worker.py, serve_speech.py, tts_to_wav.py   speech tools
  llm_worker.py               resident LLM worker for --llm
  groot_server.py, start_groot_server.ps1, start_groot_server.bat   GR00T policy server
  validate_scene.py, render_cameras.py, measure_assets.py, capture_grasp.py, probe_contact_gap.py
tests/                        pure-logic tests + test_phase*_isaac.py integration tests
renders/                      camera captures from earlier development (not part of the dated record)
requirements.txt              per-interpreter dependency recipes
ARCHITECTURE.md               design and rationale
SYSTEM_VERIFICATION.md        dated verification record
HANDOFF.md                    operator notes
```

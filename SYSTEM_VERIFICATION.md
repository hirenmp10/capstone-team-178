# Multimodal Conversational Robotic Arm for Natural Language-Based Object Manipulation
## Verification Record

**Capstone Project — Team 178**

This is a dated record of what was run and observed on 2026-09-24 and 2026-09-25 for the simulation
lane (branch `capstone-completion`). It replaces earlier summary tables whose figures could not be
traced to a run. Anything not listed here is not claimed as verified; see
[Not verified](#6-not-verified).

---

## 1. Environment

| Item | Value |
|---|---|
| Machine | Developer laptop, NVIDIA RTX 5090 Laptop GPU (24 GB) |
| Driver | NVIDIA 577.03 |
| OS | Windows 11 |
| Simulator | Isaac Sim 5.1.0 standalone (Python 3.11 via `..\python.bat`) |
| Test interpreter | Python 3.12 (`py -3.12`) with numpy, scipy, pyyaml, pyzmq, msgpack, msgpack-numpy, opencv-python, pytest |
| Code | Commit `8c397b3` unless noted (base `main` = `7cfb504`; `9071905` and `8c397b3` on top). `38cfee9` later changed only `scripts/speech_worker.py` (port claim on Linux) and a test skip on non-Windows runners. |
| Robot / scenes | Franka Panda; `configs/default.yaml` (red block, blue can, green box) and `configs/benchmark.yaml` (YCB soup can, foam brick, pudding box, banana, marker; bowl and sorting bin as destinations; office furniture) |

**Evidence location.** Paths below are relative to the repository root of the checkout that produced
them. `logs/` is listed in `.gitignore`, so these files are on the verification machine and are not
part of a clone unless added explicitly. Two GR00T logs from 2026-09-24 are in the separate
development checkout `manipulation_framework/` and are marked as such.

---

## 2. Test suites

| Suite | Command | Result | Evidence |
|---|---|---|---|
| Pure logic (no GPU, no Isaac) | `py -3.12 -m pytest tests -q -m "not isaac" -p no:cacheprovider` (from the repo root) | **1069 passed, 3 skipped** (the skips need hardware-lane files that are not on this branch) | Console output; also stated in the `8c397b3` and `38cfee9` commit messages |
| Isaac integration | `..\python.bat scripts\run_isaac_tests.py` | **99/99 passed**, about 6 min | `logs/isaac_test_report.txt` (99 of 99 selected tests PASSED, none FAILED; the report has no final pytest summary line because Isaac teardown ends the process); `logs/e2e/isaac_suite.txt` |
| CI | `.github/workflows/tests.yml`: pure-logic suite on ubuntu-latest and windows-latest, Python 3.12 | `38cfee9` passed: pure-logic (ubuntu-latest, py3.12) and pure-logic (windows-latest, py3.12) both success, on the push run and the PR #1 run (2026-09-25). `8c397b3` had failed on ubuntu only (3 Windows-launcher tests and a Linux port-claim test), fixed in `38cfee9` | GitHub Actions runs 36095172081, 36095172011 |

The Isaac suite must be run through `run_isaac_tests.py`, not plain `pytest`: `SimulationApp` parses
`sys.argv`, and Isaac discards stdout, so the runner calls `pytest.main` in-process and writes the
report to a file.

---

## 3. End-to-end runs in Isaac Sim (text commands via `-c`, rule-based parser)

### 3.1 Default scene — `logs/e2e/phase8_default.txt` (events `logs/20260924_120601_b4199d/events.jsonl`)

| Command | Outcome |
|---|---|
| what do you see | "observed 3 object(s): red block, blue can, green box" (colours correct on the first look) |
| pick the red object | picked the block |
| place it back | placed 25 mm from where it was picked |
| pick the blue cube | refused: "cannot find 'blue cube': the only block is the red block; currently visible: red block, blue can, green box" |
| pick the blue can | held |
| place it next to the green box | placed, settled 45 mm from the target |
| pick the smallest object | the block |
| put it on the left | placed 10 cm left of the pick point (42 mm from target) |
| pick the large object | refused: the box is 74 mm at its narrowest, gripper span 78 mm |
| pick the object (3 present) | "Which one do you mean: the green box, the blue can or the red block?" (no motion) |
| pick the object on the right | the block; placed back 28 mm |
| move the red block onto the green box | 2 clauses, both OK: picked, "placed on the green box (settled 25 mm)" |
| pick the purple dragon | refused: nothing in view is a dragon |
| dance | "could not understand" |

### 3.2 Benchmark scene — `logs/e2e/phase8_benchmark_rerun.txt`

| Command | Outcome |
|---|---|
| what do you see (first) | red brick, grey box, grey marker, red can, yellow banana, red bowl |
| pick up the marker; put it in the bowl | held; placed in the red bowl, settled 25 mm |
| pick the yellow object; place it back | the banana; 22 mm |
| move the brick next to the bowl | both clauses OK, settled 13 mm |

An earlier run (`logs/e2e/phase8_benchmark.txt`) gripped the banana 74 mm from its centre; "place it
back" honestly reported a miss (95 mm > 50 mm tolerance) and the following brick pick found no path.
The re-run above completed all commands. Recorded as a known limitation.

### 3.3 Fresh-clone check

A fresh clone of commit `9071905` (before `8c397b3`): the README benchmark-scene command ran, and the
benchmark lifted 5/5 objects. Lift gains: soup can +118 mm, foam brick +119, pudding box +114,
banana +120, marker +67.

---

## 4. Manipulation benchmark (ground truth, not self-report)

Command: `..\python.bat scripts\benchmark_manipulation.py --config configs/benchmark.yaml --place`

| Measure | Result |
|---|---|
| Perceived | 6/6 |
| Lifted (> 3 cm ground-truth rise) | 5/5 attempted; bowl skipped as a destination (55 mm minimum extent) |
| Placed back, released by ground truth | 5/5 (4 of 5 within the 50 mm tolerance and reported ok by the Place skill itself; the soup can settled 70 mm from where it was picked and was reported as a miss) |

Evidence: `logs/benchmark/20260924_121023_manipulation.json` and `.csv` (timestamp, config SHA-256,
git commit, per-object outcome, timings); console log `logs/e2e/benchmark_place.txt`. The JSON's git
field records `9071905` with uncommitted changes (`dirty: true`), i.e. the working tree that was then
committed as `8c397b3`.

---

## 5. Voice and GR00T

### 5.1 Speech recognition on recorded audio

- Five commands were synthesised with Windows SAPI (`scripts/tts_to_wav.py`) and transcribed with
  `scripts/speech_worker.py --wav`:
  - Canary-Qwen-2.5B (GPU, bf16, about 5.8 GB): **5/5 exact**.
  - faster-whisper `small.en` (CPU): **5/5** (confidence 0.47-0.70).
- The five Canary transcripts were run in Isaac on the benchmark scene
  (`logs/e2e/voice_transcripts.txt`): see; pick up the marker; put it in the bowl (25 mm); "pick the
  red object" -> clarification (three red objects); go home. All as expected.
- Canary needs a CUDA-enabled torch. On driver 577.03 the base Python's torch 2.12+cu130 reports no
  CUDA, so the worker was run with the canary venv's own python (torch 2.11+cu128).
  `scripts/serve_speech.py` exits with a clear message if `--device cuda` is requested without CUDA.
- Canary reports no per-utterance confidence (fixed 1.0); Whisper does.
- On this machine DirectSound microphones return digital silence; MME devices (`--mic 1`, `--mic 2`)
  are the ones to use.

### 5.2 GR00T N1.7

**2026-09-24, first real-model run** (development checkout: `manipulation_framework/logs/e2e/groot_server2.txt`,
`run_f_groot.txt`): the server loaded the checkpoint in 66 s; Isaac connected over ZeroMQ; one
inference round trip; 8 action steps passed the safety filter (0 clamped); GPU peak 9.4 GB. The
zero-shot pick did not grasp the can, and the executor at that time **falsely reported success**.
Fixed in `8c397b3`: a pick succeeds only with a held object, and targets are resolved before the
policy runs.

**2026-09-25, re-run after the fix** (`logs/e2e/groot3_server.txt`, `logs/e2e/groot3_isaac.txt`,
events `logs/20260925_100116_ba3c2e/events.jsonl`, GPU trace `logs/e2e/groot3_gpu.csv`):

- `scripts/start_groot_server.ps1` (defaults): checkpoint loaded, "Serving on 127.0.0.1:5555" after 72 s.
- Isaac, default scene, `--groot --groot-skills pick`: connected, 0 import errors (the server's
  environment stayed scoped to the server process).
- "pick up the blue can" -> **FAILED, reported honestly**: "policy ran 12 iteration(s) / 96 step(s)
  (0 clamped, 0 rejected); no object grasped: the gripper reports no grasp (width 80.0 mm) (target:
  blue can)". "place it back" -> refused (not holding anything).
- "pick the red object" -> grounding resolved the red block before the policy ran; same honest failure.
- The planner retried each failed GR00T pick (FAILED -> REPLAN, up to `motion.max_replan_attempts` = 3):
  each command produced 3 attempts x 12 iterations = 36 `gr00t.iteration` events (72 in the run's
  `events.jsonl`).
- The policy did move the simulated arm: net TCP travel 0.234 m (blue can) and 0.224 m (red block),
  largest single step 4.2 mm, all within the safety filter (0 clamped, 0 rejected). It did not approach
  the target: for the blue can (y +0.12, z 0.47) the TCP drifted to y -0.20 and stayed about 0.62 m
  high; for the red block it moved toward the block's side but never descended. The policy's gripper
  output peaked at 0.006 (never commanded closing).
- GPU peak 9,976 MiB with Isaac and GR00T resident.

**Status:** inference executes and its actions drive the simulated arm through the safety filter;
zero-shot task success **0/2**; failures are reported honestly. The checkpoint is not fine-tuned for
this robot or scene (the `oxe_droid` embodiment was trained on DROID data).

---

## 6. Not verified

| Item | State |
|---|---|
| Live microphone voice session | Not run end to end (the microphone was muted at the OS level). Only recorded-audio ASR is verified. |
| LLM intent parser (`--llm qwen`, Qwen2.5-3B-Instruct) | Not exercised; the model is not present on the test machine. Falls back to the rule parser. |
| GR00T task success | 0/2 zero-shot; DROID frame conventions vs the Isaac world frame not verified. |
| Grounding by relation to another object ("the block next to the can") | Unit tests only; not exercised in Isaac. |
| Answering a clarification question interactively | Unit tests only; recorded runs used `-c`, where the question is printed. |
| Stop / emergency stop during a running skill | Not supported: processed between commands only. |

## 7. Known limitations observed

- Perception uses Isaac ground-truth instance segmentation + depth (not a learned detector); colour,
  size and pose are estimated from the segmented depth points; single-view bias can offset a large
  box's centre by about 5 cm.
- Long objects gripped far off-centre can miss the 50 mm place tolerance (reported by Place).
- "Next to" accuracy observed 13-45 mm.
- NVIDIA driver 610.88 crashed Isaac in `rtx.scenedb` on this laptop (9/9 runs); 577.03 / 592.27-class
  drivers are required.

## 8. Earlier camera captures

`renders/` holds camera captures (RGB, depth, segmentation; `renders/pipeline_verification/`) from
earlier development. They are not part of this dated record.

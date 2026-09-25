# Handoff: operator notes for the simulation lane

Practical notes for whoever runs or continues this branch (`capstone-completion`). How to install
and run everything is in [README.md](README.md); the dated run record is in
[SYSTEM_VERIFICATION.md](SYSTEM_VERIFICATION.md); design is in [ARCHITECTURE.md](ARCHITECTURE.md).
The physical-hardware lane is in development and is not part of this branch.

## Current state (2026-09-25)

- **Default scene** (red block, blue can, green box): see, pick by name / colour / size / position,
  place back / next to / on / in a direction, multi-step "move X onto Y", clarification and refusals
  all ran as expected (`logs/e2e/phase8_default.txt`).
- **Benchmark scene**: the object set was curated in `9071905` to soup can, foam brick, pudding box,
  banana and marker (plus bowl and sorting bin as destinations). The earlier set's mug, cracker box,
  mustard bottle and drill were removed; no result for them is in the current record. Benchmark
  `--place`: 6/6 perceived, 5/5 lifted, 5/5 released by ground truth; 4/5 within the 50 mm place
  tolerance (the soup can settled 70 mm away and Place reported it as a miss).
- **Voice**: ASR verified on synthesised audio (Canary 5/5, Whisper `small.en` 5/5) and the
  transcripts executed in Isaac. A live-microphone session has not been verified.
- **LLM parser**: present, not exercised (model not installed on the test machine).
- **GR00T**: inference runs and moves the arm through the safety filter; zero-shot pick 0/2, reported
  honestly. Disabled by default.
- **Tests**: pure logic 1069 passed / 3 skipped; Isaac 99/99.

The older "2 of 8 objects pick; object height is the cause" diagnosis in earlier versions of this
file is superseded; the results above were obtained after the fixes in `9071905` (IK-branch
variation on replanning, grasp width plausibility, go-home retreat ladder, place aiming the held
object) and the curated object set.

## Interpreters on the development machine

Keep these separate. Loading the wrong stack into Isaac's process is the most common way to break it.

| Purpose | Interpreter |
|---|---|
| Isaac Sim, assistant, benchmark, Isaac tests | `..\python.bat` (Isaac's Python 3.11). Install nothing into it. |
| Pure-logic tests, `tts_to_wav.py` | `py -3.12` |
| Canary ASR (GPU) | `$env:USERPROFILE\canary-venv\Scripts\python.exe` running `scripts\speech_worker.py` (torch 2.11+cu128) |
| Whisper ASR | any system Python with `faster-whisper` (never Isaac's: CTranslate2 crashes inside it) |
| GR00T server | `$env:USERPROFILE\groot_env` (Python 3.12, torch 2.9.0+cu128), launched only via `scripts\start_groot_server.ps1` |
| LLM worker | a separate interpreter with torch + transformers + accelerate (not set up on the test machine) |

## GR00T procedure

1. Check the resolved command and environment without starting anything:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_groot_server.ps1 -DryRun`
2. Start the server the same way without `-DryRun` (or `scripts\start_groot_server.bat`). Wait for
   "Serving on 127.0.0.1:5555" (72 s in the recorded run).
3. In a **different** shell: `..\python.bat scripts\run_assistant.py --groot --groot-skills pick -c "..."`.

The launcher resolves `nvidia/GR00T-N1.7-3B` to a local snapshot in the HuggingFace cache and sets
`PYTHONPYCACHEPREFIX`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `GROOT_PATCH_MISTRAL=1` on the
server process only. Never set these in the shell that runs `python.bat`: an exported
`PYTHONPYCACHEPREFIX` made Isaac write `.pyc` files into the prefix and fail to import extensions.
Parameters: `-GrootEnv`, `-GrootPython`, `-Model <snapshot dir>`, `-HfCache`, `-Port`, `-Device`.

To see what the policy did, read the `gr00t.iteration` events in the run's `events.jsonl`: TCP at the
start and end of each iteration, largest single step, gripper output, clamped/rejected counts, goal
verdict. The comment above `gr00t.enabled` in `configs/default.yaml` ("has never completed a live
inference") predates the 2026-09-24/25 runs.

## Colour warm-up

The first "what do you see" used to report "black block, black can, black box". Cause: headless,
`SimulationContext.step` does not render, so the RGB buffer is stale while depth and segmentation are
fine. Now:
- every `observe()` renders `perception.render_frames_before_capture` (8, measured) frames first;
- a frame that is still unlit contributes geometry but no colour;
- a track holds a known colour until frames agree;
- at start-up the assistant observes twice (tracks need two frames to be confirmed), then up to three
  more times until colours have settled, and logs a warning if they have not.

Do not lower `render_frames_before_capture` without measuring; dark objects are the ones that go
wrong, and the whole-image brightness does not reveal it.

## Isaac quirks

- Isaac discards `print` output and on some USD errors exits with code 0 and no traceback. The
  scripts write their results to files under `logs/`; read those, not the console.
- `SimulationApp` parses `sys.argv`. `run_assistant.py` clears it before start-up; Isaac tests must go
  through `scripts/run_isaac_tests.py`, never plain `pytest`.
- Usage strings inside some scripts still show an older `..\python.bat manipulation_framework\scripts\...`
  path; ignore them and run from the repository folder as in the README.
- `run_isaac_tests.py --help` / `--collect-only` do not overwrite the report; a real run moves the old
  report to `logs/isaac_test_report.prev.txt` first.
- NVIDIA driver 610.88 crashed Isaac in `rtx.scenedb` 9/9 times on the development laptop; 577.03 is
  what the recorded runs used.

## Speech notes

- Use MME microphones (`--mic 1` / `--mic 2`); DirectSound devices return digital silence here.
  `speech_worker.py --list-mics`, `--scan-mics`, `--check-mic` help find the right one.
- `serve_speech.py` (base interpreter + NeMo appended from the canary venv) refuses `--device cuda`
  when its torch sees no GPU (exit status 4) and prints the working command. On driver 577.03, run
  `speech_worker.py` with the canary venv's python instead.
- The speech server claims port 5556 exclusively before loading a model, so a duplicate exits at once
  instead of loading a second Canary and holding a dead microphone stream.
- Canary gives no per-utterance confidence (fixed 1.0), so voice motion commands are confirmed before
  they run unless `--no-confirm` is passed.

## Measurement discipline

- Single runs are not measurements. RRT is randomised; the same configuration has given different
  pick results on different runs. Compare several runs before attributing a change in outcome to a
  code change.
- Judge manipulation by ground truth (`benchmark_manipulation.py`), not by the skill's own message.
  Every run saves `logs/benchmark/<timestamp>_manipulation.json/.csv` with the config hash and git
  commit.
- Example of why: in one benchmark-scene run the banana was gripped 74 mm off-centre, "place it back"
  missed by 95 mm (tolerance 50 mm) and the next pick found no path; the re-run was clean.
- There is no `--repeat` option on the benchmark yet; repeat runs by hand.

## Config flags that do nothing

- `perception.support_constrained_height`: **not implemented**. Nothing reads it; `validate()` rejects
  `true`. It stays in the schema only so configs that spell out `false` still load. (Earlier versions
  of this file described it as a working option; that was wrong.)
- `motion.planner`: validated (`rrt` / `rmpflow` / `joint_space`) but not read. The sim lane always
  builds `LulaMotionPlanner` (Lula RRT + straight-line Cartesian). RMPflow is not wired.
- `mfw/isaaclab_ext/` is an empty placeholder.

## Settled, do not re-litigate

- **TCP** = `panda_hand` + `[0, 0, 0.1034]` m; the stock finger frame is 51.5 mm off.
- **Reach is a sphere, not a box.** `scene.robot_reach_m` (0.78 on the benchmark) and
  `robot_min_reach_m` (0.40) are enforced by the layout and the validator; the workspace box is a clamp
  on stray targets, not a reach model.
- **Benchmark table** at x = 0.45, 0.62 x 1.00 m. On the original deeper table no valid layout existed
  for the object footprint.
- **Grasp width**: `max_grasp_width` 0.078, `finger_width_margin` 0.010, so the widest graspable object
  is 68 mm; wider ones are refused.
- **YCB assets** are authored -Y up with centred origins, and most ship without colliders
  (`configs/assets.yaml`, `mfw/physics/collision.py`).
- **Indoors, dome and distant lights do nothing** on the office scene; light comes from
  `scene.workspace_light_intensity`.
- **Camera intrinsics are not square-pixel** (`fx != fy`); forcing `fy = fx` made reconstruction worse
  (see `mfw/vision/camera.py`).

## Perception dead ends (measured; documented in code)

- A cross-view centre estimator improved centre error (cracker box 47 -> 14 mm) and still made picking
  worse; removed (`mfw/vision/manager.py`). A better-centred box is not a more graspable box: grasp
  candidates and pre-grasp standoffs move with it.
- Fusing both cameras' point clouds before fitting inflated a 0.05 m cube to 0.095 m; the best single
  view is used instead.
- `remove_statistical_outliers` is O(n^2) in memory; any new code path that denoises a cloud must cap
  its input at `perception.max_points_for_outlier_filter`.

## Open items

1. GR00T: fine-tune on this embodiment and scene; verify DROID frame conventions against the Isaac
   world frame. Zero-shot, it does not approach the target or close the gripper.
2. Stop / emergency stop cannot interrupt a running skill; they are handled between commands.
3. Replace ground-truth segmentation with a learned detector before any real-camera use.
4. Verify a live-microphone voice session and the LLM parser.
5. Add a repeat option to the benchmark and report spread, not single runs.
6. Long objects gripped far off-centre can miss the place tolerance (reported by Place, not fixed).

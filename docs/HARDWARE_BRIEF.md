# HARDWARE BRIEF — Team 178 capstone, hardware phase

Paste this whole file into a fresh session, or hand it to a teammate. It is the
single source of truth for the hardware phase. Where it disagrees with the
code, **the code wins — read it before asserting anything.** Several confident
claims in this project were wrong until someone checked.

> **2026-09-28: the MVP target changed** (existing hobby arm, one fixed
> overhead C270, Canary GGUF + Qwen2.5-3B GGUF + Florence-2 ONNX on the
> Jetson). For bring-up and the demo follow `docs/MVP_RUNBOOK.md`,
> `jetson/README.md` and `docs/JETSON_MODELS_PLAN.md`; where this brief
> disagrees with them (models, NanoOWL, Parakeet, setup commands), they win.

## 1. Project

- Capstone: *Multimodal Conversational Robotic Arm for Natural Language-Based
  Object Manipulation* (PES University, Team 178: Adyanth, Hiren, Gowtham, Shivanand).
- Repo: `C:\isaac-sim-standalone-5.1.0-windows-x86_64\manipulation_framework`,
  Python package `mfw`. Sim lane runs on Isaac Sim 5.1 via `..\python.bat`
  (Python 3.11). Hardware lane runs on vanilla **Python 3.12** (`py -3.12`) —
  **no Isaac Sim**. All 36 `mfw` modules import cleanly without Isaac.
- Pipeline: voice → ASR → intent parser → perception → ONE atomic skill per
  command (pick / place / move / …) → wait. The task planner never chains
  skills; atomicity is enforced by the state-machine graph, not convention.
- Deadline: hardware demo in **under 4 weeks** from 2026-09-18.

## 2. Decisions already made (do not relitigate)

1. **The arm the team owns** (5-DOF Arduino-Uno-class hobby kit) runs the
   **scripted/classical pipeline**. GR00T does **not** run on hardware: the arm's
   PWM servos have no position readback (no `state`, no teleop dataset), Isaac-
   GR00T N1.7 has **no LoRA** and a documented 40 GB fine-tune minimum vs. the
   laptop's 24 GB, Windows is unsupported (WSL2 only), and the deadline is 4
   weeks vs. NVIDIA's 8–12-week path. GR00T stays honest as sim + live bridge
   against `MockPolicy` (`mfw/gr00t_bridge/`).
2. **The AI models run ON the Jetson, quantized.** ASR and vision migrate to
   the Jetson; the intent LLM is optional and migrates last; Canary-Qwen-2.5B
   and GR00T are retired from the demo path (they cannot fit 8 GB shared RAM).
3. **Arduino Uno stays as the servo bridge** (USB-CDC). PCA9685 over I²C is the
   documented alternative if the Uno is a CH340 clone.
4. `mfw` runs on the **laptop** as orchestrator through Stage 3; Stage 4
   (stretch) moves it onto the Jetson — a deployment change, not a code change.

## 3. Hardware truth

| Item | Fact |
|---|---|
| Arm | Techno-Tirupati 5-DOF PLA kit. 4 arm DOF: base yaw + shoulder/elbow/wrist pitch (coplanar) + gripper. Reach ~20–30 cm. Payload ~100–200 g. |
| Servos | 3 × MG996R (stall 1.4 A genuine / 2.5 A clone @ 6 V), 2 × MG90S (~0.7 A). PWM, **no feedback**. Owned: 1 MG996R, 1 SG90 (SG90 → spare; gripper gets MG90S). |
| Jetson | Orin Nano 8 GB dev kit. Linux sees **7 620 MB**. Swap/zram cannot back GPU allocs. Header: 3.3 V logic, 3 PWM pins, 5 V pins 0.5 A total. **No mic jack** → USB mic. |
| Laptop | Windows 11, RTX 5090 Laptop 24 GB (95 W). `py -3.12` has torch/transformers/pyzmq + (now) msgpack, msgpack-numpy, opencv. |
| Camera | one USB webcam, overhead, fixed. No depth, no wrist cam. |
| Objects | marker, banana (toy), box (small, light), cube (foam), bowl + bin (destinations only). Labels match sim so the language layer is untouched. |

## 4. Architecture

```
JETSON ORIN NANO 8 GB (JetPack 6.2.2, headless, 25 W)          LAPTOP (orchestrator, py3.12, no Isaac)
speech_worker.py --asr parakeet --host 0.0.0.0 :5556  ◄──TCP──  run_assistant.py --hardware
  Parakeet-TDT-0.6B-v2 INT8, sherpa-onnx (CPU), USB mic          --voice-server <jetson>:5556
jetson/detector_service.py (NanoOWL OWL-ViT B/32) :5558 ◄──TCP── --detector-server <jetson>:5558
  detections JSON (label, conf, bbox_px) — never video           mfw/hardware/: RemoteArm, PlanarPerception,
jetson/robot_server.py (ZMQ/msgpack) :5560           ◄──ZMQ──    JointSpacePlanner, TopDownGraspGenerator,
  ├─ UnoSerialDriver ── /dev/ttyACM0 ──► Arduino Uno              RemoteController, WallClock, HardwareRuntime
  │    servo_bridge.ino: Servo lib, 20 ms interp, watchdog       RuleBasedIntentParser (default, 0 GB)
  └─ camera: get_frame (calibration + Florence fallback)         optional: --llm-server <jetson>:5557
[optional] llama-server Qwen2.5-1.5B Q4_K_M + jetson/llm_worker_llamacpp.py :5557
Servos: separate 6 V ≥5 A supply ── E-stop switch in +6 V ── ONE common ground to the Uno
```

Perception without depth: overhead camera → pixel → table XY by homography
(`exterior_camera.homography`, 9 numbers from `scripts/calibrate_table.py
--touch`); z and extents from `hardware.object_sizes`. Grasp: top-down at the
object centre (no wrist roll). Grasp verification: perception-based, because
PWM servos cannot report a stall. The verdict is carried / occluded / resting /
knocked / unknown (`mfw/physics/contact.py`): a lifted object stays visible and
*grows* in the image, so a carry is confirmed by pixel growth (plus the lift
prediction when `pose_measured`), never by "object gone" (review F1). A pick
that cannot be confirmed stops with the jaw closed for a human to check.

## 5. Service contracts (all exist or are specified; verified against code)

- **Speech 5556** (`scripts/speech_worker.py`): TCP, `listen(1)`, model persists.
  **Microphone is captured on the server side** → the mic plugs into whichever
  machine runs the speech server. Server pushes newline JSON: first
  `{"event":"ready",...}`, then optional events (`calibrating, calibrated,
  listening, speech_started, too_quiet, thinking, timing, empty, error`);
  transcripts are `{"text": str, "confidence": float}`. Client:
  `mfw/language/speech.py` `TcpSpeechRecognizer`; host/port via
  `run_assistant.py --voice-server HOST:PORT`; pass `--no-autostart` when the
  server is remote. Engines: `--asr canary|whisper|parakeet` (parakeet is new).
- **LLM 5557** (`scripts/llm_worker.py`): newline JSON `{"prompt"}` →
  `{"ok","text","duration_s"}`; one connection per prompt, 60 s timeout; any
  failure or unknown skill → rule-based parser. Host via `--llm-server` (new;
  previously hardcoded `127.0.0.1:5557` at `run_assistant.py:473`).
- **Detector 5558** (new; same protocol on Jetson NanoOWL and laptop Florence):
  `{"cmd":"detect","labels":[...],"min_score":0.1}` →
  `{"ok":true,"t":…,"width":…,"height":…,"objects":[{"label","confidence","bbox_px":[x0,y0,x1,y1]}]}`;
  `{"cmd":"ping"}` → `{"ok":true,"backend":"nanoowl|florence|scripted"}`.
- **Robot 5560** (`jetson/robot_server.py`, ZMQ ROUTER on the server, REQ in
  `mfw/hardware/zmq_rpc.py`; msgpack+msgpack_numpy, framing as
  `mfw/gr00t_bridge/zmq_client.py`; a motion worker thread lets `estop` answer
  mid-trajectory): `ping`, `get_state` →
  `{q:[4] rad, gripper_width, moving, estopped, t}` (commanded values),
  `get_frame` → JPEG, `set_gripper{width}`, `follow_trajectory{waypoints f64[N,4], dt}`
  (blocks, clamps limits, stretches time for `max_joint_velocity`), `home`,
  `estop`/`clear_estop`, `set_torque{enabled}`, `get/set_calibration`, and
  `get_fake_world` (only with `--driver fake --fake-world`). Servo calibration
  lives **on the Jetson** (`jetson/robot_config.yaml`), never in `configs/`.
- **Uno serial** (115200, ASCII lines): `P<us>,<us>,<us>,<us>,<us>\n` or
  `T<us>×5,<ms>\n` → `OK\n`; `?\n` → `Q…`; `D\n` detach all; `K\n` → `OK\n`
  (keepalive: refreshes the watchdog only). Uno interpolates every 20 ms and
  **detaches all servos after 500 ms without a P/T/K frame**; the Jetson driver
  sends `K` every 150 ms while idle, `?` reports the Uno's attached flag, and a
  re-attach slews from the last pulsed position (reviews P1/HS-1). `get_state`
  also returns `attached`, `bridge_q` and `bridge_gripper_width`.

## 6. Jetson memory budget and quality targets (measure on day 1 with `tegrastats`)

| Tenant | GB |
|---|---|
| Headless OS | 0.6–0.7 (GNOME adds +0.8 — disable it) |
| Parakeet-TDT-0.6B-v2 INT8 (sherpa-onnx, CPU) | 1.0–1.5 (est.) |
| NanoOWL OWL-ViT B/32 TRT FP16 | ~2.0 (est., unpublished) |
| Qwen2.5-1.5B Q4_K_M llama-server | ~1.6 (est.) |
| ASR + vision | ~4.5–5.2 → 2.4–3.1 GB headroom |
| ASR + vision + LLM | ~6.1–6.9 → 0.7–1.5 GB headroom; only headless, loaded once at boot largest-first, only if ≥ 1 GB free after ASR+vision |

Targets ("performance dropped merely"):
- ASR: Parakeet vs Canary WER +0.09/+0.09/+0.42 pp (clean/other/Open-ASR). Bench: 50 live commands with servos moving → ≥ 95 % object words, ≤ 2 s, RSS ≤ 1.5 GB.
- Intent: rule-based by default (identical to today). Optional Qwen2.5-1.5B Q4: ≥ 90 % agreement with the rule parser on 30 demo utterances + 10 paraphrases, ≤ 2.5 s.
- Vision: NanoOWL — each object in ≥ 95 % of 50 saved frames, ≤ 1 false positive/50, centroid ≤ 1.5 cm, ≤ 500 ms/frame. Weak labels: "marker" (thin), bin/box/cube wording → synonyms + per-label thresholds + 3-frame voting.

## 7. Jetson setup (day 1)

JetPack **6.2.2 / L4T r36.5** (not 7.x, not r36.4.7) on **NVMe** · headless
(`systemctl set-default multi-user.target`) · `systemctl disable nvzramconfig` ·
8 GB swap file on NVMe · `nvpmodel -m 1` (25 W) · static IP · `ufw allow 5556
5558 5560` · `jetson-containers` + `dustynv/nanoowl:r36.4.0` · `pip install
sherpa-onnx pyserial pyzmq msgpack msgpack-numpy opencv-python` · lock webcam
exposure/WB (`v4l2-ctl`), disable USB autosuspend · `usermod -aG dialout` ·
record `tegrastats` idle. **Do not upgrade JetPack mid-project.**

## 8. Parts (order day 1)

2 × MG996R · 2 × MG90S · genuine Arduino Uno R3/R4 (CH340 clones do not
enumerate on JetPack 6; if clone → PCA9685) · regulated 6 V DC ≥ 5 A (8–10 A
preferred) · 1000 µF/16 V cap · inline SPST ≥ 10 A DC switch (E-stop) · 1080p
USB webcam · close-talk USB microphone · NVMe SSD if not present · clamp,
jumpers, bus bar · foam cube, toy banana, small light box, bowl, bin · printed
A4 checkerboard. ≈ ₹9.6k (≈ ₹5.9k if Uno + NVMe already owned).

## 9. Wiring (Option B) and smoke test

Jetson on its own 19 V adapter; **nothing from the arm touches header 5 V pins**.
Uno USB-B → Jetson USB-A (Uno and webcam on different USB stacks). 6 V (+) →
E-stop → 1000 µF → servo bus (+); 6 V (−) → servo bus (−); **one wire** servo
bus (−) → Uno GND (never switch/fuse the ground). Signals: D3 base, D5 shoulder,
D6 elbow, D9 wrist, D10 gripper. Cutting +6 V makes the arm go limp and drop —
pad the workspace.

**Stopping the arm: limitations (reviews P4 / HS-4).** The +6 V switch is the
only stop that works *mid-motion*. The software stops -- the spoken or typed
`emergency stop` / `stop`, and Ctrl-C in `run_assistant.py` -- are real but
land **between trajectory chunks** (`hardware.trajectory_chunk_s`, 1 s by
default): the assistant is single-threaded, so a command or a signal handler
runs only when the current chunk's `follow_trajectory` returns. The estop
itself travels on a dedicated second ZMQ socket (`RemoteArm.estop_client`,
opened with a <= 2 s ping, falling back to the main socket with a warning), so
it is never queued behind a blocked chunk once it is sent. `run_assistant.py`
prints this at startup on the hardware lane and installs a Ctrl-C handler that
estops before exiting. Do not demo, document or teach the spoken stop as a
mid-motion stop; a student's hand goes to the switch. Estop detaches the servos
(the arm goes limp and drops what it holds); the next motion re-attaches
slowly from the last pulsed pose.

**Placeholder elbow range (reviews GEO-1 / HS-2).** Until S5 measures the
elbow, `hardware.arm` and `jetson/robot_config.yaml` both limit it to +/-90 deg,
all the placeholder 600..2400 us pulse map can command. Top-down grasps then
reach only ~0.145-0.20 m from the base axis (75 deg approaches ~0.17-0.225 m);
nearer objects are refused as unreachable instead of being executed 5-9 cm off
by a silently clamped servo. `HardwareRuntime` refuses to build when the laptop
limits are wider than the Jetson's pulse map; fix it with `calibrate_servos.py`.

Smoke, linkages off horns: S1 supply 5.9–6.1 V · S2 `/dev/ttyACM0` present ·
S3 `P1500,…` → `OK` · S4 one MG90S 1200→1800→1500, bus > 5.5 V · S5 add servos
one at a time, record end-stop µs (check no MG996R is continuous-rotation) ·
S6 5-joint slow sweep 2 min watching `dmesg -w` — a Jetson reboot here is
power/ground, not code.

## 10. File map (hardware lane) — matches the tree as of 2026-09-21

```
jetson/                              runs ON the Jetson; never imports mfw
  __init__.py
  robot_server.py                    ZMQ ROUTER + msgpack :5560; drivers uno|pca9685|fake; CameraGrabber;
                                     --fake-world (objects attach when the jaw closes); get_fake_world
  detector_service.py                TCP-JSON :5558; NanoOwlDetector (container) + ScriptedDetector
                                     (SyntheticPinhole; --scene, --world-from, --print-homography)
  llm_worker_llamacpp.py             5557 shim over llama-server (json_schema, 14-skill enum); --selftest
  robot_config.yaml                  servo calibration: pulse<->angle, limits, home, channel map
  README.md                          flash, Uno, run lines, systemd order, smoke tests S1-S6
  arduino/servo_bridge/servo_bridge.ino   P/T/?/D at 115200; 20 ms interp; 500 ms watchdog; MIN_US/MAX_US
mfw/hardware/                        laptop side; never imports isaacsim/omni/pxr/carb/torch/transformers
  __init__.py  clock.py (WallClock)  zmq_rpc.py  jetson_client.py (JetsonClient)  kinematics.py (PlanarKinematics)
  remote_arm.py (RemoteArm)  controller.py (RemoteController)  remote_camera.py (RemoteCamera)
  detector.py (RemoteDetector)  perception.py (HomographyPoseEstimator, PlanarPerception)
  grasp.py (TopDownGraspGenerator)  planner.py (JointSpacePlanner)  runtime.py (HardwareRuntime, HARDWARE_SKILLS)
scripts/
  run_assistant.py                   --hardware --jetson --detector-server --llm-server --fake-hardware
  serve_detector.py                  laptop detector :5558: --backend florence | --fake/--backend scripted
  calibrate_servos.py                REPL over JetsonClient: read move pulse jaw torque zero limit gripper home save ...
  calibrate_table.py                 >= 4 clicked points, typed XY or --touch, RANSAC -> exterior_camera.homography
  speech_worker.py                   --asr whisper|canary|parakeet (ParakeetEngine, --model-dir)
configs/
  hardware.yaml                      backend: hardware; planar_4dof; hardware.arm (MEASURE); homography: [] until calibrated
  hardware_fake.yaml                 loopback hosts, fake_clock, the scripted detector's exact homography
tests/  (marker phase9, all under py -3.12, in-thread servers only)
  test_hardware_config.py (94)  test_hardware_kinematics.py (35)  test_hardware_grasp_planner.py (41)
  test_hardware_perception.py (46)  test_hardware_bridge.py (41)  test_hardware_language.py (77)
  test_hardware_e2e.py (13)
docs/HARDWARE_BRIEF.md               this file
```

Verification (all exit 0 today; the first two need no hardware at all):

```
py -3.12 -m pytest tests -q -m "not isaac" -p no:cacheprovider        # 1886 passed (2026-09-24, after the review fix pass)
py -3.12 scripts/run_assistant.py --fake-hardware -c "what do you see" -c "pick up the marker" -c "place it in the bowl"
py -3.12 scripts/serve_detector.py --fake --print-homography          # == configs/hardware_fake.yaml
py -3.12 jetson/llm_worker_llamacpp.py --selftest
py -3.12 scripts/{run_assistant,serve_detector,calibrate_servos,calibrate_table}.py --help
```

Modified: `mfw/config/schema.py` (`backend`, `hardware`, `robot.kinematics`,
`robot.gripper_feedback`, camera intrinsics/homography, `grasp.recentre_from_standoff`,
`grasp.verify_min_displacement`, `motion.planner: joint_space`), `mfw/assistant.py`
(`_build_runtime`), `mfw/skills/base.py` (`grasp_generator`), `mfw/skills/primitives.py`
(`Pick._generate`, standoff gate, verify args), `mfw/physics/contact.py`
(feedback-free verification), `mfw/motion/planner.py` (`interpolate_pose`),
`mfw/language/intent_parser.py` (3 regex fixes), `scripts/speech_worker.py`
(`ParakeetEngine`), `scripts/run_assistant.py` (`--hardware --jetson
--detector-server --llm-server --fake-hardware`), `pytest.ini`, `requirements.txt`.

## 11. Stages and gates

| Stage | What | Gate (else roll back by host/port) |
|---|---|---|
| 0 · days 1–3 | Parts ordered; Jetson flashed; arm wired + smoke S1–S6; laptop fake e2e (`--fake-hardware`) green; rule-parser regex fixes; all models still on laptop. Day-1 measurements: `tegrastats` idle, NanoOWL frame latency + RSS, Parakeet CPU latency. | fake e2e green; 2-min sweep clean; measurements recorded. **Demo-of-record.** |
| 1 · wk 1–2 | ASR → Jetson (`ParakeetEngine`, USB mic on Jetson, `--voice-server <jetson>:5556 --no-autostart`). | 50 live commands: ≥ 95 % object words, ≤ 2 s, RSS ≤ 1.5 GB. |
| 2 · wk 2–3 | Vision → Jetson (NanoOWL 5558); servo calibration; camera homography `--touch`; first real voice pick. | 6 objects ≥ 95 %/50 frames, ≤ 1 FP/50, ≤ 1.5 cm, ≤ 500 ms; 20 live picks ≥ 80 %; ≥ 2 GB free. |
| 3 · wk 3 (opt) | LLM → Jetson only if ≥ 1 GB free (`--llm-server` first, against the laptop worker). | ≥ 90 % agreement with rule parser, ≤ 2.5 s, ≥ 1 GB headroom. Fail → omit `--llm`. |
| 4 · wk 4 (stretch) | `run_assistant.py --hardware` on the Jetson itself. | 10-min continuous rehearsal, all processes resident. |
| wk 4 | Buffer; 10-trial benchmark per object → CSV; demo rehearsed twice; video; slides. | |

## 12. Work split and status (as of 2026-09-28)

Code citations are `file:line` at local commit 7046082. Status words: **DONE IN CODE** = written and passing tests on the laptop against
fakes, stubs or real models on the laptop GPU (never on the Jetson, the Uno or
the arm); **PENDING HARDWARE** = needs the Jetson, the arm, the C270 or the mic;
**CHANGED** = the original assignment was superseded. Nothing below has run on
real hardware. The local lane is committed on the local branch `capstone-work`
(7046082); no remote branch contains it.

- **Adyanth** — integration lead, speech, LLM, gate owner.
  - DONE IN CODE: `HardwareRuntime`, the `run_assistant.py` hardware flags and
    the fake end-to-end lane (`--fake-hardware --demo`: 5/5 commands, exit 0);
    the hardware and fake-hardware lanes exit 1 on a failed command
    (`lane_exit_code`, `scripts/run_assistant.py:298`).
  - CHANGED: `ParakeetEngine` is superseded by `TranscribeCppEngine`
    (`--asr canary-gguf`, `scripts/speech_worker.py:416`): the same
    Canary-Qwen-2.5B as GGUF Q4_K_M, with native-rate capture and resampling.
    Laptop run: 29/30 recorded WAV commands transcribed exactly; tests in
    `tests/test_speech_gguf.py` and `tests/test_speech_logic.py::TestResampler`.
  - DONE IN CODE: the 5557 shim (`jetson/llm_worker_llamacpp.py`, default
    `--skills hardware`) tested against a real `llama-server`
    running `qwen2.5-3b-instruct-q4_k_m.gguf` on the laptop GPU; hybrid mode
    (rules first) is what `run_assistant.py` uses. Accuracy figures and their
    caveats: `docs/JETSON_MODELS_PLAN.md` section 0.
  - PENDING HARDWARE: day-1 `tegrastats` (`scripts/tegrastats_log.sh`) and the
    memory go/no-go for conversation mode (`docs/MVP_RUNBOOK.md` section 11);
    live-mic voice on the Jetson; Jetson builds of transcribe.cpp/llama.cpp.
  - DONE: GR00T slide (`docs/slides/groot-bridge.html`) updated to the
    2026-09-25 live-inference result (0/2 zero-shot picks, simulation only).
- **Hiren** — Jetson, electronics, Uno bridge (safety owner).
  - DONE IN CODE (local lane): sketch overlong-line fix (a long line gets one
    reply and runs nothing), resync when a resent frame is also late,
    refusal of a `measured: false` calibration without
    `--allow-placeholder-calibration`, `jetson/serial_smoke.py` for S3 (waits
    out the DTR reset), `Pca9685Driver` tests. The sketch is executed on the
    laptop with g++ and a mock Arduino core (`tests/test_mvp_bridge.py`),
    not flashed.
  - PENDING HARDWARE: JetPack flash, `lsusb` genuine-Uno check, flashing
    `servo_bridge.ino`, power/E-stop/common-ground wiring, bench S1-S6.
  - See the integration note below: PR #2 carries a different bridge.
- **Gowtham** — mechanics, kinematics, calibration.
  - DONE IN CODE: camera calibration tool `scripts/calibrate_camera.py`
    (checkerboard intrinsics, `pose` via `solvePnP`, `check`), measured only
    on synthetic C270-class renders (`tests/test_camera_calibration.py`);
    `calibrate_table.py --touch` now holds the arm with a bounded heartbeat.
  - PENDING HARDWARE: assembly, link lengths and limits (all three copies:
    `configs/hardware.yaml`, `jetson/robot_config.yaml`, the README
    `--arm-geometry` line), S5 end-stops, `calibrate_servos.py`, the real
    camera run that sets `exterior_camera.pose_measured: true`, object sizes.
- **Shivanand** — vision, skills, evaluation, demo.
  - CHANGED: NanoOWL is superseded by Florence-2-base FP16 ONNX
    (`scripts/serve_detector.py --backend florence-onnx`), evaluated on Isaac
    renders only (`tests/test_florence_backend.py`); the frame vote now needs
    a frame from the current request.
  - CHANGED: scan-then-act with a wrist camera became the fixed-camera sweep
    for the MVP (`FixedCameraScan`, `mfw/skills/primitives.py:1077`,
    registered as `scan_scene` in `mfw/hardware/runtime.py:73`); objects
    outside the workspace are reported as "out of reach".
  - PENDING HARDWARE: label list and prompts from real C270 frames
    (`/etc/mfw/detector_labels.json`), demo objects that fit the 45 mm jaw,
    the 10-trial benchmark CSV, video and report section.

**Integration note (teammate PR #2).** GitHub PR #2 (branch
`hiren/robot-server`, opened 2026-09-25, open against `main`) contains its own
`jetson/robot_server.py` and a sketch at `firmware/servo_bridge/servo_bridge.ino`
(protocol in its `docs/hardware/PROTOCOL.md`). Its serial protocol is
space-separated: `P 1500 ...`, `T <ms> <pulses>`, `W <ms>`, `S` (status, also a
keepalive) and `V` (version). The local lane's `jetson/arduino/servo_bridge/`
sketch uses comma frames (`P1500,...`, `T<pulses>,<ms>`), `K` keepalive, `?` ->
`Q...` status and `S<nonce>` resync echo, which `UnoSerialDriver._resync`
depends on. The two cannot be mixed: flash the sketch that matches the
`robot_server.py` you run. The team must pick one before bench day.

Daily 15-min sync; run the fake e2e test before every push.

## 13. Rules

1. **Read the code before asserting behaviour.** Cite `file:line`.
2. Never power servos from the Jetson 5 V rail or the Uno 5 V pin. One common ground. E-stop in the +6 V lead only.
3. Load GPU models **once at boot, largest first**; never lazily per utterance (NvMap error 12).
4. Do not upgrade JetPack mid-project; 6.2.2 locks the toolchain.
5. `gr00t.use_mock_server` stays `false` over LAN — the mock transport is pickle (remote code execution). Never expose the mock server beyond loopback.
6. Every stage is reversible by host/port; the laptop-hosted path remains the demo-of-record until a gate passes.
7. Numbers marked (est.) in the plan are unpublished for the Orin Nano — replace them with day-1 measurements and say so.
8. Sim lane must stay green: `..\python.bat scripts\run_isaac_tests.py` (never plain pytest for Isaac tests).
9. **The Uno watchdog is fed by the Jetson, not by motion.** Never disable the
   robot server's keepalive (`K` frames) or raise `WATCHDOG_MS` to hide a gap.
   If `get_state` reports `attached: false` and you did not command it, the arm
   went limp: the next motion re-attaches slowly, but check what it dropped
   first.
10. **Boot is a full-speed jump; later re-attaches are one slow move.** After
    every robot_server start (the port open resets the Uno) the bridge knows
    no position, so the first frame attaches every servo AT home instantly.
    That frame is the first `home` request -- sent when
    `run_assistant.py --hardware` connects, possibly minutes after the start,
    but only after its FIRST HOME gate (`FirstHomeGate`,
    `mfw/hardware/runtime.py`): the operator types `home` at a terminal, or the
    run passes `--home-confirmed`; with neither it refuses before moving.
    **Hand-pose the arm at home at that prompt.** The gate also runs when the
    position is known but the servos are limp (the server's host timeout
    after the previous run), because that home re-attaches AT the last pulsed
    pose first; the prompt then asks for the arm at that pose. Once a position is
    known, a re-attach after estop / torque off / watchdog is one slow move
    (`reattach_s` 1 s, `home_move_s` 2.5 s), including a jaw-only command
    (every frame carries all five channels). The jump nothing can remove is
    from where gravity left the arm to its last pulsed pose, so hand-pose it
    near home before clearing an estop. `calibrate_servos.py torque on`
    re-energises at the LAST PULSED pose (an arm moved by hand while limp
    jumps back) and asks for `yes` first.
11. **Limits must fit the pulse map.** `robot_config.yaml` limits outside the
    pulse range are refused at load/save; `hardware.arm` limits wider than the
    Jetson's map make `HardwareRuntime` refuse to build. Change both files
    together after S5, never one. `clamped: true` on a reply means a servo
    could not reach the plan: stop and recalibrate, do not ignore it.
12. **The software stop lands between chunks** (<= `trajectory_chunk_s`); the
    +6 V switch is the only mid-motion stop (section 9).
13. **A feedback-less pick that cannot confirm the grasp stops with the jaw
    closed.** Look at the gripper, then say "open the gripper". Keep
    `exterior_camera.pose_measured: false` until `scripts/calibrate_camera.py`
    (`intrinsics`, then `pose`) has measured it; while it is false the grasp
    verdict can **never** be "carried" (every real pick ends unconfirmed) and
    the pinhole refinement is off.

## 14. Verified-by-code anchors (so nobody re-derives them wrong)

Hardware `IRobot` surface is 13 methods (`joint_names, config, get_state, tcp_pose,
forward_kinematics, inverse_kinematics, get_arm_joint_positions, set_arm_joint_targets,
open_gripper, close_gripper, get_gripper_width, get_gripper_state, go_home_immediate`).
Skills also call `planner.plan_with_retries / exclude_from_collision / include_from_collision`
(not on `IMotionPlanner`). `Pick` calls `generate_grasp_candidates` directly
(`primitives.py:518,571`). `RobotConfig.validate` requires 6 joints unless
`kinematics: planar_4dof`. `Assistant` builds the runtime at `assistant.py:60`.
Speech server captures the mic itself (`speech_worker.py:556-620,680`). LLM host
was hardcoded at `run_assistant.py:473`. Rule parser quirks: "drop it in the bowl"
hit the gripper matcher before place; "take a picture" → pick; "stay" → wait.

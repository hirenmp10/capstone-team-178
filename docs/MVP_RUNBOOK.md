# MVP runbook -- from parts on the desk to the demo

Team 178 (PES capstone). Target decided 2026-09-27: the team's **existing
Techno-Tirupati PLA hobby arm** (4 arm joints + gripper, PWM servos with **no
position feedback**) driven by an Arduino Uno from a Jetson Orin Nano 8 GB,
**one fixed overhead Logitech C270**, and the simulation's own models,
quantized, on the Jetson. The laptop (`py -3.12`) orchestrates with `mfw`.

> **Nothing in this runbook has been executed on real hardware yet.** Every
> step marked **UNTESTED ON HARDWARE** is written from the code and from
> fakes; expect to correct it on the bench and write down what changed. What
> *is* proven, on the laptop only:
> - the whole demo flow on the fake lane (`py -3.12 scripts/run_assistant.py --fake-hardware --demo`: 5/5 commands ok, 2026-09-28);
> - the three models on the laptop GPU (Canary Q4_K_M = bf16 on 30/30 commands; Qwen Q4_K_M hybrid parser 61/62 on the dev set, 31/31 held-out, 34/34 fresh after the 2026-09-28 language changes -- the one dev miss, "drop the can next to the bowl", is a testset label that now disagrees with the parser by design (it is a pick-then-place transfer); Florence FP16 = FP32 on 72/72 boxes) -- `docs/JETSON_MODELS_PLAN.md` section 0;
> - every safety path (placeholder guard, serial resync, host timeout, stale frames) against fakes that model the physical failure, in the pure test suite.

Deeper references: `jetson/README.md` (Jetson install, builds, systemd,
smoke tests), `docs/JETSON_MODELS_PLAN.md` (models and memory),
`docs/HARDWARE_BRIEF.md` (contracts and background; older, the code wins).

---

## 1. Parts on the desk

**This table is the single source of truth for the MVP parts** (servo per
joint, servo supply, E-stop). `docs/HARDWARE_BRIEF.md` and `jetson/README.md`
point here; where anything else disagrees, this table wins.

| Part | Role | Note |
|---|---|---|
| DS3218 20 kg-cm | shoulder | **Buy the 180-degree version.** A 270-degree DS3218 maps 500..2500 us to 270 deg; the placeholder map assumes 600..2400 us = +-90 deg. Either works after calibration (section 6), but the placeholder band would move it further than expected. |
| MG996R x2 | base, elbow | stall 1.4 A (genuine) to 2.5 A (clone) at 6 V |
| MG90S x2 | wrist, gripper | never an SG90 on the gripper (strips) |
| Arduino Uno R3, genuine (`lsusb` shows `2341:`) | servo bridge | a CH340 clone does not enumerate on JetPack 6 -> PCA9685 board (backup driver, no watchdog) |
| Regulated ~6 V DC, >= 5 A (8-10 A preferred) | servo supply | a 6 V supply, or a 5 V 10 A SMPS trimmed up to 5.8-6.0 V (the 0.3 V above 5.5 V is the sag budget: the bus must stay > 5.5 V during a full-speed home, S4 and section 11). Never the Jetson's or the Uno's 5 V |
| Latching E-stop / SPST switch >= 10 A DC | the only mid-motion stop | in the servo supply's +V (+6 V) lead only, never the ground |
| 1000 uF / 16 V electrolytic | across the servo bus | observe polarity |
| Optional 7.5-10 A blade fuse | +6 V lead | after the switch |
| 18 AWG wire, terminal block / bus bar | power bus | servo leads are 22-26 AWG: keep them short |
| Logitech C270 + clamp stand | the one overhead camera | 720p, fixed focus; served at 640x480 |
| Close-talk USB microphone | speech on the Jetson | the Jetson has no mic jack |
| Jetson Orin Nano 8 GB + NVMe SSD | models + bridge | JetPack 6.2.2 / r36.5 |
| Demo objects | picked | **<= 35 mm across the grip, <= 50 g**: whiteboard marker (19 mm), 25 mm wooden block, toy banana <= 30 mm thick, 30 mm foam ball |
| Bowl (<= 15 cm wide), small box, bin | destinations | placed INTO/ONTO, never picked |
| A4 print of the checkerboard | camera calibration | `calibrate_camera.py make-board`, printed at 100 % |
| Multimeter, calipers, ruler, tape | measuring | current clamp if available (gripper stall current) |

## 2. Wiring -- UNTESTED ON HARDWARE

```
 6 V PSU (+) --[E-stop]--[fuse]--+--------------------------+---- servo V+ (red) x5
                                 |                          |
                              1000 uF (+)                   |
 6 V PSU (-) --------------------+--(-) cap ----------------+---- servo GND (brown/black) x5
                                 |
                                 +---- ONE wire ----> Uno GND

 Uno D3  -> base_yaw signal      Uno D5  -> shoulder_pitch signal
 Uno D6  -> elbow_pitch signal   Uno D9  -> wrist_pitch signal
 Uno D10 -> gripper signal       Uno USB-B -> Jetson USB-A (not the webcam's port group)
 Jetson on its own 19 V adapter. C270 and USB mic on the Jetson.
```
- One common ground between the servo bus and the Uno, as **one** wire; never
  switch or fuse the ground. Nothing from the arm touches the Jetson's header.
- The E-stop cuts +6 V only: the arm goes limp and **drops what it holds** --
  pad the workspace.
- PCA9685 backup: servo bus V+ to the board's V+ terminal, SDA/SCL/3.3 V/GND to
  the Jetson's I2C header (bus 7, address 0x40), channels 0-4 in the order above.

## 3. Pre-power-on safety checklist (before EVERY power-on)

Two people: one operates, one keeps a hand on the E-stop.

1. E-stop within reach of the second person and **open** (off) before plugging anything in.
2. Supply measured with no load (S1), one pass condition per supply: 5.9-6.1 V from a 6 V supply; 5.8-6.0 V from a trimmed 5 V SMPS. Below 5.8 V there is no room for the load sag S4 allows (bus > 5.5 V). Polarity of the capacitor and of every servo plug checked.
3. Exactly one ground wire servo bus (-) -> Uno GND; no arm wire on any Jetson pin.
4. Workspace clear of hands, cables and anything that must not be knocked over; foam or a towel under the reach circle.
5. The arm **hand-posed at home** (upper arm vertical, forearm folded ~35 deg back, jaws up; `robot.home_joint_positions`) **right before the first `home`**: after every robot_server start the bridge knows no position, and the first `home` request attaches all servos AT home at full speed. That request is `calibrate_servos.py` -> `home`, or `run_assistant.py --hardware` connecting (HardwareRuntime homes on startup), which can be minutes after `jetson_mode.sh conversation` while an unpowered arm sags. **run_assistant does not send it silently:** against a real driver with no known position it prints the hand-pose checklist (arm at home, E-stop in reach, hands clear), discards anything typed during bring-up, and waits for you to type **`home`** (a bare Enter is asked again); Ctrl-C or end of input aborts before anything moves (Ctrl-C also sends an estop: clear it with `calibrate_servos.py` -> `clear` before the next run, or the next run refuses and says so). Pose the arm at that prompt, not earlier. **A second run on the same robot_server prompts too**: 5 s after the previous run's last request the server's host timeout detaches the servos and the arm goes limp, and the next `home` re-attaches every servo AT ITS LAST POSE at full speed before moving slowly home -- a sagged arm snaps back there. The checklist then says "DETACHED (limp)" and prints that last pose: hand-pose the arm at that pose (not at home). A run with no terminal (piped input, a scheduled task, a service unit) refuses and exits 1 unless `--home-confirmed` is given -- pass it only when a person has just hand-posed the arm and is at the E-stop. Only the fake lane, and a real arm whose servos are still attached at a known position (a run started within the host timeout), never prompt.
6. `jetson/robot_config.yaml` says `measured: true`, or robot_server is started with `--allow-placeholder-calibration` (first bring-up only).
7. The laptop can reach the Jetson (`ping`), and nobody is about to start a second robot_server.
8. After power-on: watch the first `home`. If anything strains, buzzes or heats, hit the E-stop, then investigate.

## 4. Jetson and Uno software -- UNTESTED ON HARDWARE

Follow `jetson/README.md` sections 1-5a: flash JetPack 6.2.2 (never r36.4.7),
headless, MAXN_SUPER, packages and the venv, llama.cpp and transcribe.cpp
built from source for sm_87, onnxruntime-gpu from the jp6/cu126 index, the
model files, then flash `servo_bridge.ino` with arduino-cli. Record idle
memory with `scripts/tegrastats_log.sh 1000 60 idle`.

## 5. Smoke tests S1-S6, linkages OFF -- UNTESTED ON HARDWARE

`jetson/README.md` section 8. S3 is now `python3 jetson/serial_smoke.py
--serial /dev/ttyACM0` (the old `printf` lost its frame in the Uno's reset
window). S4-S6 run robot_server with `--allow-placeholder-calibration`; S5
(end-stops) adds `--placeholder-band-us 600,2400`. Write every end-stop pulse
down: section 6 needs them.

## 6. Servo calibration, in this order -- UNTESTED ON HARDWARE

1. **Pulse ranges (S5, linkages off).** For each servo, `pulse <joint> <us>`
   in 50 us steps toward each end until it strains; the last quiet pulse is its
   end-stop. Put them into `jetson/robot_config.yaml` `pulse_min_us` /
   `pulse_max_us`, and tighten `MIN_US/MAX_US` in `servo_bridge.ino` **and**
   `SKETCH_ARM_PULSE_US` / `SKETCH_GRIPPER_PULSE_US` in `jetson/robot_server.py`
   together (a test pins them); re-flash.
2. **Mount the linkages at the servo centre.** `pulse <joint> 1500`, then fit
   each horn so the joint sits at its reference: base pointing straight ahead
   (+x), upper arm vertical, forearm in line with the upper arm, wrist in line
   with the forearm. This keeps every zero offset small. **Gripper:** at
   1500 us the placeholder gripper map (open 1000 us = 45 mm, closed 1900 us =
   0 mm) is about 20 mm *half-open*, not closed: fit the gripper horn with the
   jaws ~20 mm apart, or leave it off until step 6. A horn fitted "jaws
   closed" at 1500 us drives 400 us past contact on the first close and stalls
   the MG90S.
3. Restart robot_server (placeholder flag, default band) with the arm
   hand-posed at home. On the laptop:
   `py -3.12 scripts/calibrate_servos.py --jetson <jetson-ip>:5560`, first command `home`.
4. `zero <joint>` for each joint, after driving it to its reference with `move`/`pulse`.
   Do zeros **before** limits (a zero after a limit shifts that limit).
5. `limit <joint> lower|upper`: drive each joint slowly to a few degrees inside
   where it would hit something (another link, the table, the stand) and record.
   The elbow folds far one way and barely the other; set its window where it
   really moves (`angle_at_min_deg`/`angle_at_max_deg` or `zero_offset_rad`).
6. `gripper open <mm>` and `gripper closed <mm>` with calipers (open should be ~45 mm).
7. Pose the parked position you want (clear of the table and of the camera's
   view) and `home set` (it refuses a home with a link below the table).
8. `velocity 0.5` for the first real moves, then `measured yes`, then `save`.
9. **Mirror to the laptop** (`configs/hardware.yaml`): the limits into
   `hardware.arm.joint_lower/joint_upper`, home into
   `robot.home_joint_positions`; measure the link lengths (shoulder axis
   height, upper arm, forearm, wrist axis -> jaw midpoint) into `hardware.arm`
   and into `MFW_ARM_GEOMETRY` in `/etc/mfw/mfw.env`. The laptop refuses to
   build when its limits are wider than the Jetson's map, and warns when they
   are narrower.
10. Restart robot_server **without** `--allow-placeholder-calibration`.
11. Gripper squeeze: hold a marker for a minute and measure the MG90S current
    and temperature. A grasp closes to the planned width minus 4 mm
    (`hardware.grasp_close_margin_m`); lower it (1-2 mm) if it sits near stall.

## 7. Camera: mount and calibrate -- UNTESTED ON HARDWARE

**Mount.** Clamp the C270 about 0.55-0.60 m above the table, looking straight
down at the workspace in front of the base (the placeholder config assumes
`position [0.18, 0, 0.60]`, `look_at [0.18, 0, 0]`). The parked arm must not
cover the reach band (14-28 cm from the base). Lock exposure and white balance
(`jetson/README.md` section 1). robot_server owns the camera; everything else
uses `get_frame`.

**Calibrate** (laptop, `py -3.12 scripts/calibrate_camera.py ...`; writes
`configs/hardware.yaml` `exterior_camera` with a `.bak` first):
1. `make-board --out board.png`; print A4 at 100 % (no fit-to-page); measure
   the 100 mm bar top right. If it is off, pass `--square-mm <measured>` to
   every later step. Tape the sheet to something flat and rigid.
2. `intrinsics --jetson <jetson-ip>:5560`: show the board in >= 15 still,
   different poses (tilted up to ~40 deg, reaching every image corner). It
   refuses to save above 1.0 px RMS and resets `pose_measured` to false.
3. Park the arm out of view, lay the sheet flat on the table, then
   `pose --jetson <jetson-ip>:5560 --touch`: frames first, then jog the closed
   jaw tip onto the 4 outer corners (needs step 6's calibration). Fallback:
   `--origin X Y --yaw DEG` measured with a ruler from the base (5 mm or 3 deg
   of ruler error moved synthetic objects 5-14 mm and left 2/5 carries
   unconfirmed). It writes position/look_at/up, a workspace homography and
   `pose_measured: true`.
4. `check --jetson <jetson-ip>:5560 --origin X Y --yaw DEG` writes
   `camera_check.png` (robot grid, workspace, reach band) and exits 1 above
   3 mm corner error.
5. Redo 3-4 whenever the stand is bumped; redo 2 if the resolution changes.

Without `pose_measured: true` no real pick can be confirmed: it ends "cannot
confirm the grasp" with the jaw closed (say "open the gripper").

**Detector labels.** Write `/etc/mfw/detector_labels.json` from live frames of
the real objects and set `MFW_DETECTOR_EXTRA="--label-config /etc/mfw/detector_labels.json"`.
Colour + noun grounds far better than a bare noun on sim renders ('red block'
5/6 vs 'block' 0/6; 'green box' 5/6 vs 'box' 1/6; the marker as 'pen'). Start from:
```json
{"labels": {
  "marker": {"prompt": "pen", "synonyms": ["marker pen", "whiteboard marker"]},
  "block":  {"prompt": "red block", "synonyms": ["red cube", "wooden block"]},
  "banana": {"synonyms": ["yellow banana"]},
  "bowl":   {"synonyms": ["dish"], "max_area_frac": 0.30},
  "box":    {"prompt": "green box", "synonyms": ["small box"], "max_area_frac": 0.30}}}
```
and keep `hardware.labels` in `configs/hardware.yaml` to **only the objects on
the table** (it ships as the demo table: `[marker, block, bowl, box]`; add a
label only when that object is there): Florence draws a box for every label it
is asked for, present or not (through the voted service the old 8-label list
found 17/60 objects vs 36/42 for only the objects present; in raw
single-frame output 'ball' with no ball present made 8-12 phantoms per 6
images, which the frame vote removed; a 'banana' phantom did survive it).

## 8. Table calibration -- UNTESTED ON HARDWARE

The camera `pose` step already writes the homography. Two cross-checks:
- **Table height**: the base must sit on the table plane (`perception.ground_plane_z: 0.0`).
  If the base is on a riser, pass `--table-z` to `pose`/`check` and set `ground_plane_z`.
- **`scripts/calibrate_table.py --jetson <jetson-ip>:5560 --config configs/hardware.yaml --touch`**
  re-measures only the homography by touching points with the jaw (it has a
  heartbeat, so waiting at its prompt does not relax the arm -- but only while
  the tip is at least 5 mm above the model's table and for at most 90 s
  without a typed command; below 5 mm or after that the arm relaxes within
  5 s, `ok` still records the commanded pose, and the next move re-attaches
  it slowly. The jaw closes to 2 mm, never onto its end stop). Use it if
  the checkerboard pose keeps failing; `calibrate_camera.py check` then reports
  how far the homography and the camera model disagree.

## 9. First bench moves at reduced speed -- UNTESTED ON HARDWARE

1. robot_server with the measured calibration, `max_joint_velocity` 0.5 rad/s (step 6.8).
2. `calibrate_servos.py`: `home`, `move base_yaw 20`, `move base_yaw -20`,
   `home`, then each pitch joint +-10 deg, `jaw 30`, `jaw 10`. Watch for `clamped`.
3. `py -3.12 scripts/run_assistant.py --hardware --jetson <jetson-ip> --detector-server <jetson-ip>:5558 -c "what do you see"`
   -- check every object is reported where it is (distances from the base).
   After a robot_server restart it first stops at the FIRST HOME prompt:
   hand-pose the arm at home, then type `home`. On later runs the servos are
   limp (host timeout) and the prompt shows the last pose instead: pose it there.
4. `-c "scan the room"`: the base turns slowly (0.35 rad/s) toward each object and back home.
5. One object, typed: `-c "pick up the marker" -c "open the gripper" -c "go home"`.
   Hand on the E-stop. The run stops at the first failed command (`--keep-going` overrides).
6. S6-style soak: 10 picks and places of one object before raising
   `max_joint_velocity` (up to 1.2 after a clean 2-minute sweep).

## 10. The demo -- UNTESTED ON HARDWARE

**Setup.** Objects 14-20 cm from the base axis (where top-down grasps reach
on the placeholder geometry), inside the camera view; long objects lying
pointing at the base or across it (a diagonal one is refused); the bowl at
about 15 cm forward, 12 cm right. Never name anything "can" (the ASR hears "kin").

**Start.** Hand on the E-stop, then on the Jetson
`sudo scripts/jetson_mode.sh conversation` and `scripts/jetson_mode.sh status`
(all five units active). Start `scripts/tegrastats_log.sh 500 0 demo`.
Start run_assistant (below). Its startup sends the first `home`, which
attaches every servo AT home at full speed (robot_server's own start attaches
nothing), so it stops at a **FIRST HOME** checklist first: **hand-pose the arm
at home at that prompt**, check the E-stop is in reach and hands are clear,
then type `home` and press Enter. If an earlier run ended more than 5 s ago the
servos are limp and the checklist asks for the arm at its printed last pose
instead. For a run with no terminal add `--home-confirmed` (section 3 item 5).

**Typed run (the scripted demo):**
```
py -3.12 scripts/run_assistant.py --hardware --jetson <jetson-ip> --detector-server <jetson-ip>:5558 --demo
```
= scan the room -> what do you see -> pick up the marker -> put it in the bowl
-> go home (`hardware.demo_script` in `configs/hardware.yaml`). Exit status 0
only when all five succeed. "look around" / "scan the table" are routed to
the scan on the hardware lane.

**Voice run:**
```
py -3.12 scripts/run_assistant.py --hardware --jetson <jetson-ip> --detector-server <jetson-ip>:5558 \
    --voice --asr canary --voice-server <jetson-ip>:5556 --llm qwen --llm-server <jetson-ip>:5557
```
The parser is **hybrid** by default (rules first; Qwen only when the rules
refuse). Motion commands are confirmed by voice first (Canary reports no
confidence); keep that on for the demo. A remote speech server is never
autostarted from the laptop: it is waited for. The spoken "stop" lands only
between motion chunks (<= 1 s); **the E-stop is the stop**.

## 11. What to measure (write it down every session)

| What | How | Target (plan go/no-go) |
|---|---|---|
| Memory per mode | `scripts/tegrastats_log.sh` in build mode, conversation idle, during the demo; `jetson_mode.sh status` MemAvailable | >= 0.4 GB free in conversation mode, no NvMap errors through 20 cycles |
| GPU really used | `GR3D_FREQ` in tegrastats during a transcription / parse / detect | non-zero |
| Speech latency and accuracy | 50 live commands: `journalctl -u mfw-speech` timings; object words right / 50 | <= 2 s end of speech -> text; >= 95 % object words |
| Parse | the `intent source=rule|llm` log line; time when source=llm | <= 2.5 s per command |
| Detect | 50 frames per object: found / phantoms (`serve_detector` log per request) | each demo object in >= 95 % of frames; <= 1.5 s per detect |
| Command time | `took` per command in the run_assistant report; `logs/<stamp>/events.jsonl` | -- |
| Pick success | 10 trials per object: position, verdict printed, what an observer saw | report honestly; "carried" must match the observer |
| Gripper hold | MG90S current and temperature over 60 s holding the marker | well below stall; not hot to touch |
| Supply | bus voltage during a full-speed home | > 5.5 V |

## 12. Fallbacks

Each can be applied alone; stop the Jetson unit it replaces to free memory
(`sudo systemctl stop mfw-speech`, `mfw-llama mfw-llm-shim`, or `mfw-detector`).
robot_server always stays on the Jetson (it owns the USB serial and the webcam).

- **Speech on the laptop** (verified on the laptop in --wav and --serve mode, 2026-09-27):
  a venv with `transcribe-cpp==0.2.4 numpy sounddevice`, then (PowerShell)
  `$env:TRANSCRIBE_LIBRARY = "$env:USERPROFILE\jetson_tools\transcribe.cpp\cuda\transcribe-native-windows-x86_64-cuda\transcribe.dll"`
  and `<venv>\Scripts\python.exe scripts\speech_worker.py --asr canary-gguf --gguf $env:USERPROFILE\jetson_models\canary\canary-qwen-2.5b-Q4_K_M.gguf --serve --host 127.0.0.1 --port 5556 --mic N`
  (`--list-mics` for N; a mic that refuses 16 kHz is captured at 48 kHz and resampled).
  Laptop: `--voice-server 127.0.0.1:5556`.
- **LLM on the laptop** (verified on the laptop, 2026-09-26/27):
  `$env:USERPROFILE\jetson_tools\llama.cpp\cuda12.4\llama-server.exe -m $env:USERPROFILE\jetson_models\qwen\qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -c 2048 -np 1 --host 127.0.0.1 --port 8080`
  (the laptop check used build b11200 with one 2048-token slot)
  and `py -3.12 jetson\llm_worker_llamacpp.py --host 127.0.0.1 --port 5557 --llama-url http://127.0.0.1:8080 --skills hardware`.
  Laptop: `--llm qwen --llm-server 127.0.0.1:5557`.
- **Detector on the laptop** (verified on the laptop against a stub robot_server, 2026-09-27):
  in `$env:USERPROFILE\jetson_check_venv` (onnxruntime-gpu 1.23.2, CUDA 12 build) with
  `site-packages\nvidia\{cudnn,cublas,cuda_runtime,cuda_nvrtc}\bin` on `PATH`:
  `python scripts\serve_detector.py --backend florence-onnx --model-dir $env:USERPROFILE\jetson_models\florence --precision fp16 --jetson <jetson-ip>:5560 --host 127.0.0.1 --port 5558 [--label-config FILE]`.
  Laptop: `--detector-server 127.0.0.1:5558`. Frames still come from the Jetson's camera.
- **Rule parser only**: leave out `--llm`. Rules alone: 57/62 on the dev set
  with the hardware skills; paraphrases are where the LLM helps.
- **Typed instead of spoken**: `--interactive` or `-c ...`.
- **Qwen on the Jetson's CPU**: `LLAMA_ARGS="-ngl 0 ..."` (same file; estimated 4-7 s per parse).
- **Everything on the laptop except robot_server**: all three above at once;
  the Jetson then only runs `mfw-robot`.

## 13. Things that will surprise a first demo attempt

- `--fake-hardware` starts fresh fakes and refuses if something already listens on
  5560/5558 (a leftover fake from an earlier run or a test made the demo fail
  intermittently when it was reused silently). Stop the leftover, or pass
  `--reuse-fakes` to use it on purpose; an estopped fake or a server with a real
  driver is always refused.
- With `--allow-placeholder-calibration` picks stop with "Jetson clamped a
  target" (the laptop's limits are wider than the 1000..2000 us band; measured
  on the fake lane). Calibrate first (section 6).
- Top-down grasps reach only about 14.5-20 cm from the base on the placeholder
  geometry; nearer or farther objects are refused as unreachable or "out of
  reach" (by name), never attempted wrong.
- There is no position feedback: "max joint error 0.000 rad" in a report is the
  commanded pose, not a measurement. Watch the arm.
- The report's `pipeline` line for "scan the room" still prints the sim scan's
  stages ("sweep viewpoints -> observe at each -> merge tracks"); the hardware
  scan looks from the parked pose, then sweeps the base visibly.
- Florence-2 sometimes grounds a label on the arm itself ('banana' landed on
  the sim Franka in 4/4 requests when no banana was present): ask only for
  objects that are on the table. A phantom drawn in the same place on every
  frame passes both the service's vote and the scan's 2-of-3 check: "scan the
  room" then reports it and turns the base toward it (pinned in
  `tests/test_mvp_flow.py`, `test_a_consistent_phantom_is_claimed_and_swept_to_a_known_limit`).
  A phantom seen in one frame only is not reported.
- The detector's frame vote carries frames over between requests less than
  2 s apart, but every reported box must include a frame grabbed by the
  current request: an object moved since the last request is reported where
  it is now, or not at all (never at its old place).
- **The scan's "seen in 2 of 3 looks" is not independent evidence.** The
  three looks are `hardware.scan.frame_gap_s` (0.15 s) apart, inside that 2 s
  history, so a look can confirm an object with one new frame plus the
  previous look's last frame: a glare or phantom on three consecutive frames
  is reported as seen (pinned against the real frame vote in
  `tests/test_mvp_flow.py`, `TestScanIndependence`). Two knobs make the looks
  independent, both shown working there and neither on by default because the
  extra detector time is unmeasured on the Jetson: `--history-max-age 0` in
  `MFW_DETECTOR_EXTRA` (every detect then votes on its own frames only: at
  least two Florence inferences per detect, everywhere, not just in the scan),
  or `hardware.scan.frame_gap_s` above 2.0 s (adds about 4 s to every scan).
  Try the first if phantoms show up in scans; watch the detect time.
- The first `home` after every robot_server start, every E-stop recovery and
  every host-timeout detach (the previous run ended) can jump the arm:
  hand-pose it right before, every time (at home after a server start, at its
  last pose when the servos are limp). run_assistant --hardware asks for that
  at its FIRST HOME prompt (type `home`; or refuses without a terminal unless
  `--home-confirmed`); `calibrate_servos.py` asks too.

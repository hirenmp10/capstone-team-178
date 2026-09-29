# Jetson Orin Nano 8 GB -- MVP bring-up and run book

Everything in this directory runs **on the Jetson** (Python 3.10, JetPack 6)
and never imports `mfw`. The laptop (`py -3.12`, `scripts/run_assistant.py
--hardware`) orchestrates; the Jetson hosts the arm bridge, the one overhead
webcam and the three quantized models.

> **Status (2026-09-28): nothing in this file has been executed on a Jetson,
> an Uno, the arm or the C270.** Every command below is **UNTESTED ON
> HARDWARE** unless a line says otherwise. What *has* run: every service's
> code path against fakes and stubs (pure suite), the three models on the
> laptop GPU (`docs/JETSON_MODELS_PLAN.md` section 0), and the whole flow on
> the fake lane (`py -3.12 scripts/run_assistant.py --fake-hardware --demo`).
> The team's step-by-step from parts to demo is `docs/MVP_RUNBOOK.md`.

| Port | Process | What |
|---|---|---|
| 5560 | `jetson/robot_server.py` | arm bridge (Uno over USB serial) + the webcam (`get_frame`) |
| 5556 | `scripts/speech_worker.py --asr canary-gguf` | Canary-Qwen-2.5B GGUF Q4_K_M via transcribe.cpp, mic on the Jetson |
| 5557 | `jetson/llm_worker_llamacpp.py` | shim in front of llama-server (Qwen2.5-3B-Instruct Q4_K_M) |
| 8080 | `llama-server` | loopback only; the laptop never talks to it directly |
| 5558 | `scripts/serve_detector.py --backend florence-onnx` | Florence-2-base ONNX FP16; frames come from 5560 |

---

## 1. Flash and configure the OS (day 1)

- SDK Manager -> **JetPack 6.2.2 / L4T r36.5** onto the **NVMe SSD**. Check
  `head -1 /etc/nv_tegra_release`: it must say `R36 (release), REVISION: 5.x`.
  **Never R36.4.7** (NVIDIA-acknowledged memory regression: models over ~1 GB
  fail to load). Fallback: 6.2.1 / r36.4.4. Do not move to JetPack 7.x, and
  do not upgrade JetPack mid-project.
- Headless: `sudo systemctl set-default multi-user.target` (the desktop costs ~0.8 GB).
- `sudo systemctl disable nvzramconfig`; add an 8 GB swap file on the NVMe
  (`fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`, plus `/etc/fstab`).
  Swap only helps the CPU side: it never backs a GPU allocation.
- Power mode: list the modes with `sudo nvpmodel -q --verbose` and select
  **MAXN_SUPER** with `sudo nvpmodel -m <its id>` (the id differs between
  board configs; MAXN_SUPER only exists when the board was flashed with the
  *Super* configuration). Run `sudo jetson_clocks` before any timing run.
- Static IP; firewall: `sudo ufw allow 5556,5557,5558,5560/tcp` (8080 stays closed).
- Serial access: `sudo usermod -aG dialout,audio,video $USER`, log out and in.
  If `/dev/ttyACM0` appears and vanishes: `sudo systemctl disable --now brltty`.
- **Uno check**: `lsusb` must show `2341:` (genuine Arduino). A CH340 clone
  (`1a86:7523`) does not enumerate reliably on JetPack 6 -> use a PCA9685
  board and `--driver pca9685` (section 5c).
- Webcam (Logitech C270, fixed focus): plug it into a different USB port
  group from the Uno. Lock exposure and white balance once the lighting is
  final (control names differ between kernels; list them first):
  `v4l2-ctl -d /dev/video0 --list-ctrls`, then e.g.
  `v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c white_balance_automatic=0`.
  Disable USB autosuspend: add `usbcore.autosuspend=-1` to the kernel command
  line in `/boot/extlinux/extlinux.conf`.
- Record idle memory: `scripts/tegrastats_log.sh 1000 60 idle` (target: <= 0.7 GB used, headless).

## 2. Packages and the Python environment

```
sudo apt update
sudo apt install -y build-essential cmake git python3.10-venv python3-pip \
    libportaudio2 libopenblas-dev v4l-utils i2c-tools curl
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc   # nvcc for the builds
nvcc --version                                                                    # CUDA 12.6 on JetPack 6.2.x

git clone <the team's repo> ~/mfw && cd ~/mfw && git checkout capstone-work
python3 -m venv ~/mfw-venv && . ~/mfw-venv/bin/activate
pip install --timeout 120 numpy pyzmq msgpack msgpack-numpy pyserial opencv-python pyyaml \
    sounddevice smbus2 pillow tokenizers onnx huggingface_hub
# onnxruntime-gpu ONLY from the Jetson AI Lab index for JetPack 6 / CUDA 12.6 --
# never a generic/CUDA-13 wheel (on the laptop a CUDA-13 build silently ran on the CPU):
pip install --timeout 120 onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python3 -c "import onnxruntime as o; print(o.get_available_providers())"         # must list CUDAExecutionProvider
# transcribe.cpp's Python binding (the native library comes from the source build in 3b):
pip install --timeout 120 transcribe-cpp==0.2.4 || pip install --timeout 120 --no-deps transcribe-cpp==0.2.4
```
`/etc/mfw/mfw.env` must then say `MFW_PYTHON=/home/jetson/mfw-venv/bin/python`
(the example file says `/usr/bin/python3`). The `--no-deps` fallback is for
the case where `transcribe-cpp-native==0.2.4.*` has no aarch64 wheel
(unverified either way).

## 3. Source builds (sm_87) -- in build mode

Stop every model first so the compiler gets the whole 8 GB:
`sudo scripts/jetson_mode.sh build`. Use `-j4`, never `-j6` (host OOM).

**3a. llama.cpp** (llama-server for Qwen):
```
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
git checkout 81bc6b83f   # build 11200: the version the laptop check ran and LLAMA_ARGS is tested against
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-bench
build/bin/llama-bench -m ~/models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -fa 1   # smoke; GR3D_FREQ must be > 0
```

**3b. transcribe.cpp v0.2.4** (Canary GGUF):
```
git clone https://github.com/handy-computer/transcribe.cpp ~/transcribe.cpp && cd ~/transcribe.cpp
git checkout v0.2.4
cmake -B build -DTRANSCRIBE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DTRANSCRIBE_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
find build -name 'libtranscribe.so'          # build/bin/ or build/src/; the ggml backend libs must sit beside it
```
`-DTRANSCRIBE_BUILD_SHARED=ON` is required: the Python binding dlopens
`libtranscribe.so`. Point `TRANSCRIBE_LIBRARY` in `/etc/mfw/mfw.env` at the
path `find` printed. Without it the binding may load a CPU-only pip library,
and the worker then **refuses to start** ("NO CUDA backend") instead of
running at seconds per command.

## 4. Model files (`~/models`, the paths `mfw.env.example` expects)

The exact files validated on the laptop are in `$env:USERPROFILE\jetson_models\`;
copying them is the safest route (`scp -r` each folder), otherwise download:

| Model | File(s) | Size (bytes) | Source |
|---|---|---|---|
| Qwen2.5-3B-Instruct Q4_K_M | `qwen/qwen2.5-3b-instruct-q4_k_m.gguf` | 2,104,932,768 | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Canary-Qwen-2.5B Q4_K_M | `canary/canary-qwen-2.5b-Q4_K_M.gguf` | 1,737,575,808 (sha256 `db5162229d6fa22597d06a613bd9b543eddb3ee02e6afc5e759120fde02bebf7`) | `handy-computer/canary-qwen-2.5b-gguf` |
| Florence-2-base ONNX FP16 | `florence/tokenizer.json` + `florence/onnx/{vision_encoder,embed_tokens,encoder_model,decoder_model_merged}_fp16.onnx` | 183,972,269 / 78,780,353 / 86,747,414 / 194,546,759 | `onnx-community/Florence-2-base` |

```
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ~/models/qwen
huggingface-cli download handy-computer/canary-qwen-2.5b-gguf canary-qwen-2.5b-Q4_K_M.gguf --local-dir ~/models/canary
huggingface-cli download onnx-community/Florence-2-base tokenizer.json onnx/vision_encoder_fp16.onnx \
    onnx/embed_tokens_fp16.onnx onnx/encoder_model_fp16.onnx onnx/decoder_model_merged_fp16.onnx --local-dir ~/models/florence
sha256sum ~/models/canary/canary-qwen-2.5b-Q4_K_M.gguf
```
The FP16 merged decoder's broken output names are fixed **in memory** at load
(`jetson/florence_onnx.py`); the file on disk is never modified.

## 5. The arm bridge

### 5a. Flash the Uno (arduino-cli; no IDE needed)

```
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=$HOME/bin sh
~/bin/arduino-cli core update-index && ~/bin/arduino-cli core install arduino:avr
~/bin/arduino-cli lib install Servo
~/bin/arduino-cli compile --fqbn arduino:avr:uno ~/mfw/jetson/arduino/servo_bridge
~/bin/arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyACM0 ~/mfw/jetson/arduino/servo_bridge
```
(An Uno R4 Minima would be `--fqbn arduino:renesas_uno:minima` after
`core install arduino:renesas_uno`; untested with this sketch.) Stop
robot_server before uploading (only one process may own the port).
**Re-flash after every change to the sketch**: robot_server needs the `S`
sync echo, the 7-field `Q` reply and the whole-line overlong discard.

Pins: **D3 base_yaw, D5 shoulder_pitch, D6 elbow_pitch, D9 wrist_pitch, D10
gripper**. 115200 baud. Servos stay detached until the first `P`/`T` frame;
500 ms watchdog; per-joint `MIN_US/MAX_US` clamps (600..2400 us arm,
900..2100 us gripper) -- tighten them after S5 *and* mirror them in
`SKETCH_ARM_PULSE_US` / `SKETCH_GRIPPER_PULSE_US` in `robot_server.py`
(`tests/test_hardware_bridge.py` pins the two together).

Servos (the parts list is `docs/MVP_RUNBOOK.md` section 1): shoulder DS3218
20 kg-cm (180-degree version), base + elbow MG996R, wrist + gripper MG90S
(never an SG90 on the gripper).

Power: a separate regulated **~6 V >= 5 A** supply (8-10 A preferred; a 5 V
10 A SMPS trimmed to 5.8-6.0 V is acceptable) -> **E-stop switch** in the
+V lead -> 1000 uF across the servo bus; **one** ground wire from the servo
bus (-) to an Uno GND; nothing from the arm on the Jetson's 5 V pins. Full
wiring and the pre-power-on checklist: `docs/MVP_RUNBOOK.md` sections 2 and 3.

### 5b. robot_server run line (the camera is part of it)

```
. ~/mfw-venv/bin/activate && cd ~/mfw
python3 jetson/robot_server.py --config jetson/robot_config.yaml --driver uno --serial /dev/ttyACM0 \
    --host 0.0.0.0 --port 5560 --arm-geometry 0.07,0.0,0.105,0.10,0.09 --require-camera \
    [--allow-placeholder-calibration [--placeholder-band-us 600,2400]]
```
- **Camera**: robot_server is the **only** process that opens `/dev/video0`
  (`robot_config.yaml` `camera:` 640x480 @ 30 fps, JPEG 85). It is on by
  default; `--require-camera` makes a failed open fatal (the error asks
  whether another process holds the device; `sudo fuser -v /dev/video0` names it). `get_frame` returns `{jpeg, width, height,
  t_capture, age_s, seq, t_server}`; `age_s` is measured on the Jetson's own
  monotonic clock, so no clock sync is needed, and frames older than 1.0 s
  (`max_age_s` in the request) are refused. The detector,
  `calibrate_camera.py` and `calibrate_table.py` all take frames through it.
  Never enable `detector_service.py`'s own `CameraLoop` next to it.
- **Placeholder guard**: `robot_config.yaml` ships `measured: false`. With
  `--driver uno`/`pca9685` the server **exits 2** on it. For the FIRST bench
  bring-up only, add `--allow-placeholder-calibration`: every pulse stays in
  1000..2000 us, speed is capped at 0.35 rad/s, re-attach takes >= 2 s and
  the home move >= 4 s. For the S5 end-stop search with the **linkages off**,
  widen with `--placeholder-band-us 600,2400`. After calibration
  (`calibrate_servos.py`: zero, limits, gripper, `home set`, `measured yes`,
  `save`), restart **without** the flag. The band is never written to the file.
  While the flag is on, picks will stop with "Jetson clamped a target"
  (measured on the fake lane: the laptop's `hardware.arm` limits are wider
  than the band). That is intended: calibrate first.
- `--arm-geometry` (base_height, shoulder_offset, upper_arm, forearm, tool in
  metres) drives the home-below-table check. The value above is the
  placeholder `hardware.arm`; replace it with the measured links (the same
  numbers as `hardware.arm` in the laptop's `configs/hardware.yaml`).
- No hardware yet: `python3 jetson/robot_server.py --driver fake --fake-camera --fake-world "marker:0.18,0.05 bowl:0.15,-0.12"`.

### 5c. What the bridge guarantees (and what you must do for it)

- **Hand-pose the arm to home before the first `home` after EVERY server
  start.** Opening the serial port resets the Uno (DTR), so every start -- and
  any USB re-enumeration or brownout -- is a power-up: the bridge knows no
  position and its first frame attaches every servo *at* its target,
  instantly. The server refuses every motion except `home` until that first
  attach (`get_state` -> `bridge_position_known: false`); the boot `home` is a
  full-speed jump to `home_q`, logged as one. Neither laptop tool sends it
  silently: `run_assistant.py --hardware` stops at a FIRST HOME checklist,
  discards anything typed during bring-up and waits for the word `home`
  (Ctrl-C / end of input aborts before anything moves; with no terminal it
  refuses unless `--home-confirmed`), and `calibrate_servos.py` asks for `yes`
  (`--yes` in scripts). The same checklist appears on a later run against a
  running server once the servos are limp (the server detaches them 5 s after
  the previous run's last request): that `home` re-attaches every servo AT ITS
  LAST POSE at full speed before moving slowly home, so pose the arm at the
  last pose it prints, not at home. The placeholder home is `[0, 0, -0.6109, 0.1745]`
  rad = 1500/1500/1150/1600 us: upper arm vertical, forearm folded 35 deg
  back, jaws up-and-back, TCP about (-0.095, 0, 0.338) m -- behind the base and
  out of the camera's workspace. S5 measurements replace it.
- **Keepalive bounded by the laptop's liveness.** The Uno detaches after
  500 ms without a frame; the server sends `K` every 150 ms while attached and
  idle. After `host_timeout_s` (5 s) with no client request and no motion,
  the server **detaches** (the arm goes limp and drops what it holds). The
  assistant, `calibrate_servos.py` and `calibrate_table.py --touch` send a
  `get_state` heartbeat every second, so only a dead or frozen laptop trips it.
- **Every detach is reported** (`detach_count`, `last_detach_reason`,
  `reattached: true`); an unrequested detach stops the current motion.
- **Slow re-attach once a position is known**: after estop / torque off /
  watchdog / host timeout, the first motion re-energises with one `T` frame
  over `reattach_s` (1 s; `home` `home_move_s` 2.5 s). The one jump nothing
  removes is from where gravity left the arm to its last pulsed pose:
  hand-pose it there (or near home and run `home`) before clearing an estop.
- **Torque on re-energises at the LAST PULSED pose**: an arm moved by hand
  while limp jumps back at full speed. `calibrate_servos.py` warns and asks
  for `yes` (`--script` runs need `--yes`).
- **A failed estop still relaxes the arm**: `D` is retried 3 times; if it is
  never confirmed the keepalive has already stopped and the Uno's watchdog
  drops the arm within 500 ms. The +6 V switch is the mid-motion stop.
  The estop reply never asks the Uno `?` after its `D` attempts, so it
  carries `bridge_probed: false` and `bridge_q: null`; `get_state` asks.
- **Serial resync**: after any failed exchange the host sends `S<nonce>` and
  reads until the echo *before* it sends the next frame; after 3 failures in
  a row it declares the servos detached. A line of 64 bytes or more is
  discarded whole by the sketch and gets exactly one `ERR line too long`.
- **Pulse-map rule**: a `robot_config.yaml` whose pulse range reaches past the
  sketch's clamps, or whose radian limits map outside its own pulse range, is
  refused at load and on `save` (the sketch clamps silently and still answers
  `OK`). `follow_trajectory` replies `clamped: true` whenever a pulse was clamped.
- **PCA9685 backup** (`--driver pca9685 --i2c-bus 7`, board at 0x40): it has
  **no hardware watchdog** -- the last pulse keeps running while powered. The
  only bound is the host timeout, which writes full-off on every channel; the
  server refuses `--host-timeout-s 0` with it, and a detach that cannot be
  confirmed says to **cut the +6 V supply**. Keep a hand on the switch.

## 6. The conversation mode (systemd)

Unit files are in `jetson/systemd/`; `scripts/jetson_mode.sh` installs and switches them.
```
sudo scripts/jetson_mode.sh install         # units -> /etc/systemd/system, /etc/mfw/mfw.env from the example (once)
sudoedit /etc/mfw/mfw.env                   # MFW_PYTHON, model paths, TRANSCRIBE_LIBRARY, MFW_MIC, MFW_ARM_GEOMETRY, *_EXTRA
# hand on the E-stop, then:
sudo scripts/jetson_mode.sh conversation    # robot -> llama-server -> shim -> speech -> detector, each after a cache drop
scripts/jetson_mode.sh status               # unit states, MemAvailable, one tegrastats line
journalctl -u 'mfw-*' -b --no-pager | tail -n 80
sudo scripts/jetson_mode.sh build           # stop everything (nothing resident)
```
- Order: robot_server first (no model; the detector needs its frames), then
  llama-server (largest contiguous allocation), shim, speech, detector. Each
  unit drops the page cache and compacts memory before it starts, and its
  `ExecStartPost` waits until the model is actually serving.
- **Nothing starts at boot on purpose**: after robot_server starts, its first
  `home` request attaches every servo AT home at full speed, and
  `run_assistant.py --hardware` sends that home when it connects -- after its
  FIRST HOME prompt (hand-pose the arm there, E-stop in reach, hands clear,
  then type `home`; on a later run with limp servos, at the last pose it
  prints). A run with no terminal (a script, a scheduled task, a future
  service unit for the laptop side) refuses and exits 1 unless it passes
  `--home-confirmed`, which asserts that a person has just hand-posed the arm
  and is at the E-stop. None of the units here runs run_assistant.
- `mfw.env` hooks: `MFW_ROBOT_EXTRA="--require-camera"` (add
  `--allow-placeholder-calibration` for the first bring-up only);
  `MFW_DETECTOR_EXTRA="--label-config /etc/mfw/detector_labels.json"` for the
  per-object colour prompts (`docs/MVP_RUNBOOK.md` section 7);
  `MFW_MIC` from `python3 scripts/speech_worker.py --list-mics`.
- Speech makes one warm-up call before it reports ready. On a cold board the
  first CUDA kernel compile can take a minute or more (55 s on the laptop's
  first-ever call); the unit allows 240 s.
- Any `NvMapMemAllocInternalTagged error 12` or cudaMalloc out-of-memory in
  the journal: **reboot**, do not retry. `signal: killed` is the memory
  budget, not a bug.

Running a service by hand (debugging), with the same arguments as the units:
```
llama-server -m ~/models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -c 2048 -np 1 -ub 128 -fa on -lm none --host 127.0.0.1 --port 8080
python3 jetson/llm_worker_llamacpp.py --host 0.0.0.0 --port 5557 --llama-url http://127.0.0.1:8080 --skills hardware
python3 scripts/speech_worker.py --asr canary-gguf --gguf ~/models/canary/canary-qwen-2.5b-Q4_K_M.gguf --n-ctx 1024 --serve --host 0.0.0.0 --port 5556 --mic N
python3 scripts/serve_detector.py --backend florence-onnx --model-dir ~/models/florence --precision fp16 --jetson 127.0.0.1:5560 --host 0.0.0.0 --port 5558 [--label-config FILE]
```
Self-checks: `python3 jetson/llm_worker_llamacpp.py --selftest` (in-thread fake
llama-server); `python3 scripts/speech_worker.py --asr canary-gguf --gguf ... --wav cmd.wav`
(offline transcription); the detector refuses to start (exit 3) when
onnxruntime has no CUDA provider (`--allow-cpu` only for a deliberately slow run).

## 7. tegrastats logging

```
scripts/tegrastats_log.sh 1000 60 idle          # build mode, nothing resident
scripts/tegrastats_log.sh 1000 120 conv_idle    # conversation mode, models loaded, nobody talking
scripts/tegrastats_log.sh 500 0 demo            # during the demo, until Ctrl-C
```
Logs go to `logs/tegrastats_<stamp>_<label>.txt` with a header naming the
L4T release and which `mfw-*` units were up. `GR3D_FREQ` must be non-zero
while a model runs (a zero means a silent CPU fallback).

## 8. Smoke tests S1-S6 (linkages OFF the horns)

| # | Check | Pass |
|---|---|---|
| S1 | Servo supply under no load, at the bus, E-stop closed | 5.9-6.1 V from a 6 V supply, or 5.8-6.0 V from a trimmed 5 V SMPS (room for the S4 sag to > 5.5 V); E-stop open -> 0 V |
| S2 | `ls /dev/ttyACM0`; `lsusb` shows `2341:` (PCA9685: `i2cdetect -y -r 7` shows `40`) | present |
| S3 | robot_server stopped; `python3 jetson/serial_smoke.py --serial /dev/ttyACM0` | `S3 PASS` (5 exchanges: `?` fresh, `P` attach, `?` echo `,1,1`, `D`, `?` `,0,1`) |
| S4 | robot_server with `--allow-placeholder-calibration`; laptop `calibrate_servos.py`: `pulse wrist_pitch 1200`, `1800`, `1500` | bus stays > 5.5 V |
| S5 | Add servos one at a time (`--placeholder-band-us 600,2400`); `pulse <joint> <us>` in 50 us steps to each end-stop; record the last pulse before it strains | no stall; no continuous-rotation servo |
| S6 | 5-joint slow sweep for 2 min with `dmesg -w` and the supply meter | no Jetson reboot, no webcam drop |

**Why S3 is a script now**: `printf 'P1500,...' > /dev/ttyACM0` opens the
port, which resets the Uno, and writes while the bootloader is still running
(~1-2 s): the frame is lost and nothing answers. `serial_smoke.py` opens the
port once at 115200, waits 2 s (`--reset-wait-s`), drains the boot output and
checks each reply. `--pulses` sets the P frame (default 1500 x5).

Then, from the laptop: `py -3.12 scripts/calibrate_servos.py --jetson <ip>:5560`
(first command after a server start: `home`, with the arm hand-posed there;
it asks for `yes`),
followed by the camera and table calibration in `docs/MVP_RUNBOOK.md`.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| robot_server exits 2, "measured: false" | placeholder calibration; see 5b |
| "could not open camera /dev/video0" | another process has it: `sudo fuser -v /dev/video0` (a stray detector_service CameraLoop, a `v4l2-ctl --stream`, a viewer) |
| "flash the current servo_bridge.ino" | old sketch without `S` sync / 7-field `Q`: re-flash (5a) |
| speech exits "NO CUDA backend" | `TRANSCRIBE_LIBRARY` unset or pointing at a CPU build |
| detector exits 3 "refusing to serve" | onnxruntime-gpu not from the jp6/cu126 index |
| detector "StaleFrameError" | the camera stalled or robot_server is overloaded; frames older than 0.5 s are refused |
| arm goes limp after 5 s at a prompt | host timeout: the tool you use has no heartbeat, or the laptop froze |
| `signal: killed` / NvMap error 12 | memory budget: reboot; `docs/JETSON_MODELS_PLAN.md` section 2 has the fallbacks |

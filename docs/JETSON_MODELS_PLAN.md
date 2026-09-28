# Running the simulation's models on the Jetson Orin Nano 8 GB — quantization plan

Status: PLAN, revised 2026-09-26 after a verified research pass (4 researchers + 4 skeptics + a memory
budget); **laptop results added 2026-09-28 (section 0)**. **No model has run on the Jetson yet; nothing
below has been measured on an Orin Nano.**

MVP decision (user, 2026-09-27): the team's existing hobby arm with **one fixed overhead C270** and **no
wrist camera**; "scan the room" looks from the parked pose with the fixed camera, then sweeps the base
visibly toward what it found and reports it (`FixedCameraScan`), and real eye-in-hand scanning (section 3) moves to a later SO-101 arm. GR00T is
excluded from the hardware MVP. The team's step-by-step is `docs/MVP_RUNBOOK.md`.
Labels: **PUB** = published by the model owner or vendor, **COM** = community/runtime-author measurement,
**EST** = our estimate (arithmetic in the research output). Day-1 `tegrastats` replaces every EST.

Requirement (user, 2026-09-26): the Jetson runs **the same models the simulation uses**, quantized so accuracy
drops only slightly — not smaller substitutes — and the arm **scans its surroundings first, builds a scene
map, then acts on spoken commands**.

---

## 0. Laptop results (RTX 5090 Laptop, Windows, 2026-09-26..28) -- the same files the Jetson will load

Measured on the laptop GPU, never on an Orin Nano: they show the **quantized files are accurate and the
code paths work**, not how fast they run or whether they fit on the Jetson. Raw evidence is in the
session scratchpad `jetson_checks/{canary,speech,qwen,language,florence,detector}`.

**Canary-Qwen-2.5B GGUF Q4_K_M** (transcribe.cpp 0.2.4 Python binding, CUDA backend):

| Check | Result |
|---|---|
| Q4_K_M vs NeMo bf16, 30 recorded demo commands | normalised transcripts **identical on 30/30**; both 29/30 exact vs the reference, WER 0.75 %, command/attribute words 41/41, object words 28/29. The one miss in *both*: "can" -> "kin" (so demo commands avoid "can") |
| Through `speech_worker.py --asr canary-gguf --wav` | 29/30 exact, 30/30 the same text as the direct binding; 3/3 native-rate WAVs (48 kHz mono, 44.1 kHz stereo, 22.05 kHz) right after the worker's resampling |
| Latency, warm CUDA JIT cache | load 3.2 s, warm-up 0.6 s, per command median 0.18 s, max 0.44 s (n = 30) |
| First-ever call | 55 s (CUDA kernel JIT) -> the worker now makes one warm-up call before "ready" |
| GPU memory | peak 3,062 MiB on the laptop (includes the CUDA context) |
| Guards | a CPU-only native library is refused loudly (exit 1, no transcript); n_ctx 1024 = 58.9 s of audio |

**Qwen2.5-3B-Instruct GGUF Q4_K_M** (llama-server b11200, CUDA 12.4, one 2048-token slot, behind the
:5557 shim, through the framework's own `LlmIntentParser`; correct = skill and parameters right):

| Utterance set | HEAD code (legacy prompt, LLM first) | New prompt, LLM first | New prompt, **hybrid** (default) | Rule parser alone |
|---|---|---|---|---|
| Dev set, 62 (the set the prompt was written against) | 48/62 (77.4 %) | 61/62 | 61/62; **62/62** with the hardware skill enum | 56/62 (57/62 hardware); HEAD rules 48/62 (49/62 hardware) |
| Held-out, 31 (written 2026-09-26 and scored *before* the rule-parser changes; those changes then fixed 3 of its items, and the rules run first in hybrid, so it is not fully clean either) | 18/31 (58.1 %) | 30/31 | **31/31** | 26/31; HEAD rules 23/31 |
| Fresh, 34 (written 2026-09-27; two prompt rules were then added for its errors, and the rule parser went from 17 to 22 on it, so it is *not* a clean held-out set) | 18/34 (52.9 %) | 32/34 | **33/34** | 22/34; HEAD rules 14/34 |
| Independent review sample, 14 (written by the reviewer 2026-09-28, in no set, never tuned against; real shim + llama-server, hardware skills) | -- | -- | **14/14** (rules 10, LLM 4/4) | 10/14 |

HEAD rule-parser figures (48/62, 23/31, 14/34) are from the review's scoring of `git archive 9c8d523 mfw`;
the earlier "52/62" was an intermediate working-tree parser. The 14-utterance review sample is the
cleanest number here; keep an untouched set for any future claim. After the 2026-09-28 fixes (negation
refused, a stop word wins, an LLM intent needs a word asking for its skill) a replay of the recorded Qwen
replies through the new parser gives the same hybrid scores (62/63, 31/31, 33/34; offline, no model run).

- The first check on 2026-09-26 (older harness) scored the model 46/62 = 74.2 % with 35/48 = 72.9 %
  agreement where the rules were right (gate: >= 90 %): a FAIL. Its error classes, each now targeted by
  a prompt rule: the destination replaced by "it" ("put it in the box" -> target "it"); targets invented
  from the visible-objects list ("grab the block" -> "red block"); millimetres not converted ("move up
  20 mm" -> 20); the clockwise sign; "drop it" / "let go" / "hold still" mis-mapped; "raise the arm" ->
  rotate_wrist.
- **Raw model** agreement where the rules are right, after the prompt fix: dev 54/56 = 96.4 %, held-out
  22/26 = 84.6 %, fresh 19/22 = 86.4 %. So the model alone still misses the 90 % gate on unseen
  wording. The **hybrid** mode passes that gate by construction (the rules run first, an LLM answer
  never overrides a rule parse, and the LLM is asked only when the rules refuse), which means the gate
  no longer measures the model. The gate that does: **the LLM's accuracy on the utterances the rules
  refuse**, on an untouched set. So far: 4/4 on the independent 14-utterance review sample; on the
  (contaminated) held-out and fresh sets the LLM answered 5 of 31 and 11 of 34 utterances. Hybrid is
  the default (`run_assistant.py --llm-mode hybrid`); since 2026-09-28 an LLM intent that moves the arm
  also needs a word in the utterance asking for that skill (and a spoken direction for a relative move).
- Remaining raw-model errors: the sign of "turn the gripper 45 degrees to the left"; "look at the can" ->
  observe (should refuse); "drop the eraser in the bowl" -> place (the rules make the same mistake, so
  hybrid keeps it: 1 of the 34 fresh utterances).
- Latency when the LLM is called: median 0.16-0.19 s, max 0.50 s (laptop GPU, prompt cache warm);
  prompt 4.8-5.3 k characters, fixed part first so the cache reuses it.

**Florence-2-base ONNX FP16** (`onnx-community/Florence-2-base`, onnxruntime-gpu 1.23.2 CUDA 12 build,
greedy decoding):

| Check | Result |
|---|---|
| FP16 vs FP32 on 12 sim renders (OD + phrase grounding) | **72/72 boxes matched at IoU >= 0.8**; output tokens identical in 22 of 24 image-task runs (the other two: min IoU 0.985 and 0.999); all four sessions on CUDA |
| Needed to get there | onnxruntime-gpu for **CUDA 12** (a CUDA 13 build silently ran on the CPU, ~10x slower; the service now refuses to start then, exit 3); the venv's nvidia cudnn/cublas DLL dirs on PATH; the fp16 merged decoder's output names fixed in memory at load; `SimplifiedLayerNormFusion` disabled on CPU only |
| Beams 3 vs greedy | same phrases, boxes within 1 px, 1.0-1.4x the time, so greedy stays the default |
| In-process latency, 204 inferences | median 228 ms, p90 424 ms, max 1.1 s; load 4.1 s, warm-up 6.7 s |
| Over TCP with the MVP safeguards (both caption orders, 2-of-3 frame vote), 6 exterior renders x 2 | The old 8-label hardware.yaml vocabulary: 17/60 objects found, 4 phantoms (hardware.yaml now ships the 4 demo labels; the 8-label list lives in hardware_fake.yaml). Colour-worded 5 labels: 36/42, 12 phantoms. Only the objects present: 36/42, 2 phantoms. First request median 0.6-1.3 s, repeats 0.3-0.4 s |

Detection limits (sim renders only; real C270 frames are unmeasured): Florence-2-base misses some objects
outright (a soup can 0/4 on YCB renders; the default scene 13/13); bare nouns ground poorly ("block"
0/6, "box" 1/6) while colour + noun works ("red block" 5/6, "green box" 5/6); every absent label becomes
a phantom on a similar object or on the arm ("banana" on the Franka in 4/4 requests; "ball" 8-12
phantoms per 6 images in raw, single-frame model output -- the service's frame vote removed them in the
final voted run, where "banana" was the phantom that survived). Hence: short, colour-worded label lists of **only the objects on the table**
(`--label-config`, `docs/MVP_RUNBOOK.md` section 7). The earlier `match_label` bug that dropped
shortened phrases ("a soup", "a blue") is fixed: a shortened phrase is accepted only when it fits
exactly one label.

---

## 1. Model by model (same checkpoints, quantized)

| Sim model | Jetson form | Size | Accuracy cost | Memory resident (EST) | Latency (EST) | Status |
|---|---|---|---|---|---|---|
| **Canary-Qwen-2.5B** (ASR; bf16 5.08 GB, 5.8 GB measured on the laptop) | GGUF of the same checkpoint, **Q4_K_M**, in `transcribe.cpp` v0.2.4 built from source with CUDA for sm_87, called in-process from `speech_worker.py` via its Python binding | **1.74 GB** PUB (Q8_0 2.80 GB) | LibriSpeech test-clean WER **1.63 %** for every quant vs **1.61 %** for the NeMo reference, ≈ +0.02 pp COM (runtime author; flagged "verified: false"). Noisy/far-field speech unmeasured | 2.3–2.8 GB (Q8_0: 3.4–3.9) | 0.5–1.4 s per 2–4 s command | Expected, **unverified**: `transcribe.cpp` has never run on any Jetson; its aarch64 release is CPU/Vulkan only |
| **Qwen2.5-3B-Instruct** (intent parser) | Qwen's **official** `qwen2.5-3b-instruct-q4_k_m.gguf` in `llama-server` (llama.cpp source build pinned at build 11200 / commit 81bc6b83f, sm_87): `-ngl 99 -c 2048 -np 1 -ub 128 -fa on -lm none` (this build rejects the old `--no-mmap` and exits 1; `-lm none` started and served on the laptop, 2026-09-28), behind the existing shim on :5557 | **2.1 GB** PUB (Q5_K_M 2.44 GB) | No published Qwen2.5-3B GGUF score. Proxy: Qwen2 Int4 lost 2.0–3.7 IFEval points PUB. Bounded by the JSON schema (one of 14 skills) and the rule-parser fallback | 2.5–2.8 GB | ~1.0 s warm per command (prompt cache), ~1.7 s first command | Expected, unverified (no Qwen2.5-3B llama.cpp run on an Orin Nano) |
| **Florence-2-base** (detector; the sim itself uses ground truth; YOLO ruled out) | `florence-community/Florence-2-base` (the official transformers conversion of the same weights) in **FP16**, PyTorch from the Jetson AI Lab jp6/cu126 wheels, transformers ≥ 4.56 native class, via `scripts/serve_detector.py` on :5558 | 0.46 GB weights EST | FP16 ≈ FP32 (< 0.5 mAP EST). No laptop baseline exists yet | 1.5–2.0 GB (+0.46 GB while loading) | 0.5–1.4 s per detect | Expected, unverified |
| Florence-2-base, **the deployed MVP path** | `onnx-community/Florence-2-base` fp16 ONNX on onnxruntime-gpu, own decode loop | 0.54 GB PUB | as above | 0.9–1.3 GB | similar | **Default** (`serve_detector.py --backend florence-onnx`); FP16 = FP32 on the laptop (section 0); Jetson unmeasured |
| **GR00T N1.7-3B** (VLA) | The same checkpoint post-training-quantized with **FoldQuantVLA "W4A4 + o/d INT8"** as 7 TensorRT 10.3 engines built **on the Nano**, served by a **custom loader we must write** (FoldQuant's own `serve.py` first loads the full bf16 policy onto the GPU and would run out of memory) | ~3.45 GB weights EST | LIBERO fine-tuned: bf16 95.8 → W4A4 94.9 PUB (desktop GPUs) | 5.0–6.0 GB (**alone**) | ~0.24–0.31 s per 40-step chunk (3–4 Hz) | **Fits alone only, unproven** — nothing published on an 8 GB Nano; upstream bf16 (6.9 GB of weights, NVIDIA "16 GB+") does not fit |

Correction to the first version of this plan: FoldQuant's "15 to 20 GB free" is **disk space** for the
engine build, not memory. GR00T is therefore not ruled out on the Jetson — but only alone, only with the
custom loader, and only if its engines build within the Nano's memory (no workstation fallback: TensorRT
engines are device-specific).

**GR00T, honestly:** (a) *Runs on the Jetson?* Probably, alone, after the loader work. (b) *Moves this arm?*
Yes mechanically through an adapter — its DROID embodiment expects an exterior + wrist camera, end-effector
pose, gripper position and a 7-joint Franka state; the adapter would synthesize the pose from forward
kinematics of the commanded servo angles, pad the joints, and project its motions through 4-joint IK —
but not meaningfully. (c) *Completes a pick?* No: ~0 % zero-shot EST (0/2 even on the simulated Franka it
was built for). A pick needs a new-embodiment fine-tune: 50–100 demonstrations on this arm plus a rented
40 GB+ GPU, which does not fit the deadline. Deliverable: "GR00T runs on the Jetson and moves the arm" as a
separate VLA-mode demonstration; the voice pick-and-place runs in Conversation mode.

Fallback ASR (not the same model): faster-whisper small.en, test-clean WER 3.05 % PUB vs Canary 1.60 %.

## 2. Memory budget and modes (7.6 GB visible; swap cannot back GPU memory)

Base: headless OS 0.6–0.7 GB + robot_server/shim 0.3–0.5 GB → 6.4–6.7 GB left for models (EST).

| Mode | Resident | Total (EST) | Verdict |
|---|---|---|---|
| **C — Conversation** (default) | Canary Q4_K_M + Qwen2.5-3B Q4_K_M + Florence-2-base FP16 + robot_server | **7.2–8.9 GB** | **MARGINAL** — day-1 `tegrastats` decides |
| C-lite, step 1 | Florence on the fp16 ONNX path instead of PyTorch | 6.6–8.2 GB | Likely (needs the ONNX decode loop — being written in the laptop check) |
| C-lite, step 2 | Qwen on the CPU (`-ngl 0`, same file) | −0.2–0.4 GB | Parse slows to 4–7 s |
| C-lite, last resort | Qwen Q3_K_M (1.72 GB) | −0.4 GB | Larger accuracy loss |
| Debug | Qwen + Florence, typed commands (no ASR) | 4.9–6.0 GB | Fits |
| **V — VLA** | GR00T W4A4 engines alone + robot_server; speech/LLM/detector stopped | 5.9–7.2 GB | Fits if the loader works and the engines build |
| B — Build | nothing resident | — | For building llama.cpp, transcribe.cpp (`-j4`, not `-j6`) and the GR00T engines one at a time |
| Not possible | Canary Q8_0 + Qwen + Florence; GR00T + any other model; Florence-2-large with both others | 8.1–11 GB | No |

Rules:
- Each mode is a systemd target; switch modes by restarting services (`systemctl isolate`) or by reboot —
  **never load and unload models per command** (repeated large allocations trigger
  `NvMapMemAllocInternalTagged error 12`).
- Before **each** model service starts: `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory`
  (a community report on this exact board shows both load orders failing without it). Load order: llama-server,
  then speech, then detector.
- Any NvMap error 12 or cudaMalloc out-of-memory: reboot, do not retry.
- Canary Q8_0 only if `tegrastats` shows ≥ 1.5 GB free with Q4_K_M loaded.
- In Mode V there is no voice: stop comes from the hardware E-stop, the robot_server watchdog, or the laptop.

## 3. Scan-then-act (DEFERRED: not in the MVP)

**MVP (2026-09-27):** no wrist camera. The fixed overhead C270 already sees the table, so "scan the
room" / "look around" / "scan the table" runs `FixedCameraScan`: several detector frames from the parked
pose (an object counts when seen in >= 2 of 3), then a slow visible base sweep toward what was found and
back home, and a report with directions, distances and what is out of reach (`hardware.scan` in
`configs/hardware.yaml`). The eye-in-hand design below is kept for the later SO-101 arm.

The arm cannot scan with a fixed overhead camera, so the camera moves onto the arm.

**Hardware:** a UVC USB camera on the forearm/wrist (eye-in-hand), cable tied along the arm. A Logitech C270
with its stand removed (~75 g) is the affordable option; it adds load at the end of the arm, so the elbow
also gets the 20 kg-cm DS3218. "Room" means **what the camera can see from the arm's base**; only objects
within reach (~15–20 cm radius with the current placeholder geometry) can be picked.

**Behaviour:**
1. On start-up, or on "scan the room / look around / scan the table": the arm visits 5–7 fixed scan poses
   (base yaw sweep, e.g. −75° … +75°, camera tilted down), waits ~0.5 s to settle, captures a frame.
2. Florence-2 detects objects in each frame. Each detection's pixel is projected to the table plane using the
   camera pose from forward kinematics of the commanded joints + a hand-eye calibration.
3. Detections from all views are merged (same label, < 3 cm apart) into a **scene map**: label, colour,
   size, table position, number of views that saw it.
4. "What do you see" answers from the map. "Pick the red block" is resolved against the map with the
   simulation's grounding (colour / size / position / relations / "which one do you mean?").
5. Before grasping, the arm moves above the object and **re-detects it close up** (corrects PLA backlash and
   servo error), grasps top-down, lifts, and checks the object is gone from its old spot.
6. After every place, the map entry is updated; a new scan runs on request or when an object is not where
   the map says.

Scan cost: 5–7 frames × Florence latency + settling ≈ 5–15 s EST.

**Code to add (hardware lane):** `mfw/hardware/eye_in_hand.py` (camera pose from FK + hand-eye extrinsic;
pixel → table-plane point); a `ScanScene` hardware skill + scene-map merge (reuse `ObjectTracker`); close-up
re-detection in the hardware pick; intent phrases "scan the room", "look around", "scan the table" → `scan`;
`scripts/calibrate_hand_eye.py` (ArUco sheet, several arm poses); camera intrinsics calibration (none exists
yet — `pose_measured: false` currently makes every real pick end "cannot confirm the grasp");
`configs/hardware.yaml` `wrist_camera` intrinsics/extrinsic and `scan.poses`.

## 4. Laptop work (no Jetson needed)

Status 2026-09-28: items 1-3 done on the laptop (results in section 0; Florence's PyTorch path and real
webcam frames not run). Item 4's code changes are in the working tree: speech `--asr canary-gguf` with
native-rate capture; `serve_detector.py --backend florence-onnx` as the default, taking frames only from
robot_server; the shim's enum restricted to the hardware skills; NanoOWL marked "do not deploy";
`jetson/systemd/` conversation/build targets and `scripts/jetson_mode.sh`. The `compare_*.py` evaluation
scripts and all GR00T items were not done (GR00T is out of the MVP).

1. **Canary GGUF:** transcribe the test WAVs with `transcribe.cpp` (Windows CUDA build) using Q4_K_M, and
   with NeMo bf16 (`canary-venv` python, `speech_worker.py --asr canary --device cuda --wav`). Gate: every
   command word identical. *(Running now.)*
2. **Qwen GGUF:** `llama-server` + the shim on :5557; replay the operator utterances through the framework's
   LLM parser; compare with the rule parser. Gate: ≥ 90 % agreement where the rule parser is right, and ≥ rule
   on paraphrases. *(Running now.)*
3. **Florence:** fp16 vs fp32 ONNX on sim renders *(running now)*; then the PyTorch FP16
   `florence-community/Florence-2-base` path through `serve_detector.py`, and real webcam frames once a camera
   exists.
4. **Code changes:**
   - `scripts/speech_worker.py`: `TranscribeCppEngine` (`--asr canary-gguf --gguf PATH --n-ctx 1024`), CUDA
     device required, same :5556 protocol; record at the mic's native rate and resample.
   - `scripts/serve_detector.py`: default model → `florence-community/Florence-2-base` (the
     `microsoft/...` remote code breaks on transformers ≥ 4.50); `--max-new-tokens` default 128 (1024 can take
     15–25 s on the Nano and blow the 5 s timeout).
   - `jetson/llm_worker_llamacpp.py`: docstring → Qwen2.5-3B; restrict the skill enum to the skills the
     hardware runtime registers.
   - `jetson/detector_service.py`: NanoOWL marked "not the sim model; do not deploy"; one detector on :5558.
   - `jetson/systemd/` targets (conversation / vla / build) + `scripts/jetson_mode.sh`.
   - Evaluation: `scripts/compare_asr.py`, `compare_intent.py`, `compare_detector.py`.
   - GR00T (only if Mode V is pursued): `scripts/groot_server_jetson.py` (meta-device skeleton, load only the
     non-quantized tensors, TensorRT engines), a logged DROID adapter in `mfw/gr00t_bridge/`, FoldQuant
     calibration + ONNX export on the laptop, reference actions for on-board verification.

## 5. Jetson work (day 1–3)

1. **JetPack 6.2.2 / L4T r36.5** on the NVMe SSD. Check `cat /etc/nv_tegra_release`: **never R36.4.7**
   (NVIDIA-acknowledged memory regression: models over ~1 GB fail to load). Fallback 6.2.1 / r36.4.4.
   Do not move to JetPack 7.2 (breaks FoldQuant, drops the Super power modes).
2. Headless (`systemctl set-default multi-user.target`), `nvzramconfig` disabled, 8 GB swap file on NVMe,
   MAXN_SUPER if offered (`sudo nvpmodel -q --verbose`), `jetson_clocks` for timing runs.
3. Toolchain: `build-essential cmake git libopenblas-dev python3.10-venv`, CUDA on PATH, `libcudss` from
   NVIDIA's apt repo (needed by the jp6 torch wheels).
4. llama.cpp: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release &&
   cmake --build build -j4 --target llama-server llama-bench`; smoke `llama-bench -m qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -fa 1`.
5. transcribe.cpp v0.2.4: `cmake -B build -DTRANSCRIBE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`;
   smoke `build/bin/transcribe-cli -m canary-qwen-2.5b-Q4_K_M.gguf cmd.wav` — `GR3D_FREQ` in `tegrastats`
   must be non-zero (no silent CPU fallback).
6. Conversation venv (Python 3.10): `torch==2.11.0 torchvision==0.26.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126`,
   then `transformers>=4.56 pillow opencv-python pyzmq msgpack msgpack-numpy sounddevice numpy pyyaml pyserial`.
7. Start Mode C by hand (drop caches before each), then install the systemd targets.

**Go/no-go on the Jetson:**

| Check | Pass |
|---|---|
| Canary Q4_K_M | command words ≥ 95 % on 50 live commands with servos moving; ≤ 2 s end of speech → text; matches the laptop NeMo transcripts |
| Qwen2.5-3B Q4_K_M | ≥ 90 % agreement with the rule parser where it is right; ≤ 2.5 s per command |
| Florence-2-base FP16 | each demo object found in ≥ 95 % of 50 frames; ≤ 1.5 s per detect |
| Mode C resident | ≥ 0.4 GB free and no NvMap errors through 20 full voice → parse → scan → pick cycles |
| Mode V (optional) | engines build; loader stays < 6.9 GB; chunk ≤ 0.35 s; actions match laptop reference |

## 6. Still uncertain

- Whether `transcribe.cpp` builds and runs Canary with CUDA on sm_87 (no Jetson run anywhere).
- Whether the three conversation models co-reside (per-process CUDA context cost is the largest unknown).
- Canary Q4_K_M accuracy on far-field USB-mic speech (parity is shown only on LibriSpeech test-clean).
- Qwen2.5-3B Q4_K_M accuracy on this parser's commands (being measured on the laptop now).
- All latencies (bandwidth/FLOP-scaled estimates; a non-Super board is 1.5–1.7× slower).
- GR00T: engine build within Nano memory, the custom loader, and any useful motion through the adapter.

Key sources: github.com/handy-computer/transcribe.cpp (docs/models/canary-qwen-2.5b.md, releases) ·
huggingface.co/handy-computer/canary-qwen-2.5b-gguf · huggingface.co/nvidia/canary-qwen-2.5b ·
huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF · github.com/ggml-org/llama.cpp (server README, discussions
5059 and 16706) · huggingface.co/florence-community · huggingface.co/onnx-community/Florence-2-base ·
github.com/NVIDIA/Isaac-GR00T scripts/deployment/README.md · arxiv.org/html/2609.24433 (FoldQuantVLA) ·
github.com/cair-vinuni/FoldQuantVLA docs/deploy · forums.developer.nvidia.com (r36.4.7 memory issue).
Full research output: session scratchpad `models_synthesis.txt`.

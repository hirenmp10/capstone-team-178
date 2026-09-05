# Multimodal Conversational Robotic Arm for Natural Language-Based Object Manipulation
## System Verification & Experimental Evaluation Report

**Capstone Project — Team 178**

### 1. Executive Summary & System Scope

This repository contains the verified implementation and experimental evaluation of the **Multimodal Conversational Robotic Arm for Natural Language-Based Object Manipulation** on **NVIDIA Isaac Sim 5.1 + PhysX**.

### Verified System Capabilities
* **Active & Verified in Current Deployment:**
  - Perception-driven 6-DoF object pose estimation via exterior and wrist cameras (no hardcoded coordinates, no simulator ground-truth reads).
  - Physics-constrained grasp candidate generation, surface normal scoring, and contact verification.
  - Collision-aware motion planning via Lula RRT and RMPflow on the Franka Emika Panda arm.
  - End-to-end atomic skill execution (`Pick`, `Place`, `Observe`, `Move`, `Go Home`).
  - Resident LLM Natural Language Intent Parser powered by live GPU inference (`Qwen/Qwen2.5-3B-Instruct`).
  - NVIDIA GR00T N1.7 ZeroMQ/msgpack IPC bridge, data contract validation, and deterministic safety filter.
  - 349 pure-logic unit/regression tests (100% passing, ~8 s execution).
  - 99 Isaac Sim integration test gates (100% passing).
  - Visual verification evidence captured directly from Isaac Sim rendering.

* **Modular Extension Boundaries (Future Milestones):**
  - Jetson Orin Nano edge deployment.
  - Sim-to-Real physical Franka hardware deployment.
  - Live GR00T policy checkpoint inference.
  - NVIDIA Canary-Qwen-2.5B ASR deployment (Whisper currently operational as fallback).

---

## 2. Quantitative Verification Results

| Component | Metric | Result | Status |
|---|---|---|---|
| **Pure Logic Test Suite** | Test count & pass rate | **349 / 349 passing (100%)** in 7.57s | **GREEN** |
| **Isaac Sim Integration Suite** | Test count & pass rate | **99 / 99 passing (100%)** | **GREEN** |
| **Franka TCP Calibration** | Distance error relative to true fingertip midpoint | **0.0 mm** (corrected 51.5 mm stock offset) | **VERIFIED** |
| **Object Height Gating** | Height threshold for pick reliability on YCB benchmark | **< 80 mm** (Banana 39mm, Foam brick 51mm) | **CHARACTERIZED** |
| **Qwen 2.5-3B Intent Parsing** | Cold-start latency | **~3.2 s** | **VERIFIED** |
| **Qwen 2.5-3B Intent Parsing** | Warm GPU inference latency | **0.26 s – 0.38 s** | **VERIFIED** |
| **Qwen 2.5-3B Intent Extraction** | Semantic entity extraction accuracy on tested variations | **100%** (action, target, destination, spatial relation) | **VERIFIED** |
| **GR00T ZeroMQ/msgpack Bridge** | Roundtrip IPC latency | **< 1.0 ms** | **VERIFIED** |
| **GR00T Action Safety Filter** | Step delta clamping & joint limit violation rejection | **100%** containment | **VERIFIED** |

---

## 3. Visual Evidence Registry

Direct camera captures from Isaac Sim 5.1 are organized in [`renders/pipeline_verification/`](renders/pipeline_verification/):

| Stage | Camera | File Path | Verification Event |
|---|---|---|---|
| **Stage 1** | Exterior (overhead diagonal) | [`renders/pipeline_verification/stage1_initial_exterior.png`](renders/pipeline_verification/stage1_initial_exterior.png) | Workspace setup with Franka arm, red block, and green box on tabletop |
| **Stage 1** | Wrist (eye-in-hand) | [`renders/pipeline_verification/stage1_initial_wrist.png`](renders/pipeline_verification/stage1_initial_wrist.png) | In-hand camera perspective of tabletop targets prior to action |
| **Stage 2** | Exterior | [`renders/pipeline_verification/stage2_block_held_exterior.png`](renders/pipeline_verification/stage2_block_held_exterior.png) | Franka arm lifting block in stable grasp; ground-truth height gain > 30 mm |
| **Stage 2** | Wrist | [`renders/pipeline_verification/stage2_block_held_wrist.png`](renders/pipeline_verification/stage2_block_held_wrist.png) | Eye-in-hand view showing block securely held between parallel gripper fingers |
| **Stage 3** | Exterior | [`renders/pipeline_verification/stage3_block_placed_exterior.png`](renders/pipeline_verification/stage3_block_placed_exterior.png) | Franka arm opening gripper and depositing block inside container |
| **Stage 3** | Wrist | [`renders/pipeline_verification/stage3_block_placed_wrist.png`](renders/pipeline_verification/stage3_block_placed_wrist.png) | Eye-in-hand view looking down into container with deposited block |

---

## 4. End-to-End LLM → Robot Execution Flow

Natural language commands are parsed live by resident `Qwen/Qwen2.5-3B-Instruct`:

```text
User Input: "grab the block and put it inside the box"
  │
  ▼ [LLM Intent Parser - Qwen 2.5-3B on CUDA] (latency: 320ms)
Structured Intent:
  {
    "action": "pick",
    "target": "block",
    "destination": "box",
    "spatial_relation": "in"
  }
  │
  ▼ [TaskPlanner & Working Memory]
Decomposed Subtasks:
  1. Pick(target="block")
  2. Place(target="block", destination="box", relation="in")
  │
  ▼ [Perception Layer]
Camera Deprojection → Depth Point Cloud → Filter → Instance Pose & OBB Estimation
  │
  ▼ [Grasp & Motion Generation]
Physics-Constrained Grasp Synthesis → Lula RRT Collision-Free Trajectory
  │
  ▼ [PhysX & Controller Execution]
Franka Arm Executes Pick → Lifts > 30mm → Executes Place → Releases Inside Box
  │
  ▼ [Visual Verification]
Post-Action Camera Re-check Confirms Object Inside Container
```

---

## 5. Verification Commands

### A. Run Pure-Logic Regression Suite (349 tests, no GPU required)
```powershell
cd manipulation_framework
$env:PYTHONPATH = (Get-Location).Path
..\python.bat -m pytest tests/ -q -m "not isaac"
```

### B. Run Isaac Sim Integration Gates (99 tests)
```powershell
cd ..
.\python.bat manipulation_framework\scripts\run_isaac_tests.py
```

### C. Run Interactive Assistant with Live Qwen LLM
```powershell
.\python.bat manipulation_framework\scripts\run_assistant.py --gui --llm qwen --interactive
```

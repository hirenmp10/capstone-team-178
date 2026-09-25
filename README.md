# Multimodal Conversational Robotic Arm for Natural Language-Based Object Manipulation

**Capstone Project — Team 178**  
Modular perception-driven manipulation framework on **NVIDIA Isaac Sim 5.1 + PhysX**, featuring multimodal sensory perception, resident LLM cognitive intent parsing (**Qwen 2.5-3B-Instruct**), physics-constrained grasp synthesis, collision-aware motion planning, and an **NVIDIA GR00T N1.7** policy backend.

---

## System Overview

This framework enables a 7-DoF Franka Emika Panda manipulator to autonomously comprehend natural-language instructions, perceive its environment via multi-view RGB-D vision, synthesize stable grasps, and plan collision-free motions in real-time.

```
 Natural Language / Voice
          │
          ▼
┌───────────────────────────┐
│   Resident LLM Parser     │  Qwen 2.5-3B-Instruct (GPU inference ~300ms)
│   (Intent Extraction)     │  Extracts: Action, Target, Destination, Spatial Relation
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│   Task Planner & Memory   │ ◄──► │  Dual Camera Perception   │
│   (Working Memory + SM)   │      │  Exterior + Wrist RGB-D   │
└─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │
              ▼                                  │
┌───────────────────────────┐                    │
│      Skill Registry       │                    │
│   (Pick, Place, Move...)  │                    │
└──────┬─────────────┬──────┘                    │
       │             │                           │
       ▼             ▼                           │
┌──────────────┐ ┌──────────────┐                │
│ Classical    │ │ GR00T N1.7   │                │
│ Pipeline     │ │ VLA Policy   │                │
└──────┬───────┘ └──────┬───────┘                │
       │                │                        │
       ▼                ▼                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  PhysX 5 / Isaac Sim 5.1                     │
│               Franka Emika Panda (7-DoF Arm)                 │
└──────────────────────────────────────────────────────────────┘
```

### Key Pillars
1. **Multimodal Perception**:
   - Dual RGB-D cameras (overhead exterior scene-wide view + eye-in-hand wrist camera).
   - Point cloud deprojection, ground/support plane removal, statistical outlier filtering, and PCA-based 6-DoF pose estimation with oriented bounding boxes (OBB).
   - Temporal tracking across frames generating a live `SceneGraph` with spatial relationships (`inside`, `on_top_of`, `next_to`).
   - **Zero hardcoded coordinates** — all manipulation targets are derived strictly from live perception.
2. **Cognitive Language Understanding**:
   - Live resident GPU inference with `Qwen/Qwen2.5-3B-Instruct`.
   - Structured JSON intent extraction with entity grounding and spatial relation resolution.
3. **Physics-Constrained Grasp Synthesis**:
   - Generates anti-podal grasp candidates aligned with object principal inertia axes.
   - Evaluates grasps using surface normal alignment, aperture constraints, gripper orientation, and approach reachability.
4. **Collision-Aware Motion Planning**:
   - Global path planning via Lula RRT and reactive execution with RMPflow.
   - Accurate Tool Center Point (TCP) calibration: fingertip midpoint `[0, 0, 0.1034] m` on `panda_hand` (correcting stock 51.5 mm offset).
5. **VLA Foundation Policy Bridge**:
   - Out-of-process ZeroMQ + msgpack IPC bridge to NVIDIA GR00T N1.7 (`oxe_droid_relative_eef_relative_joint`).
   - Deterministic `ActionSafetyFilter` enforcing step delta clamping and joint limits.

---

## Verification & Test Status

All engineering gates are fully implemented and verified:

| Test Suite | Tests Passing | Scope / Environment |
|---|---|---|
| **Pure Logic Regression Suite** | **349 / 349 (100%)** | Plain interpreter (~8 s, no GPU, no simulator required) |
| **Isaac Sim Integration Suite** | **99 / 99 (100%)** | Isaac Sim 5.1 + PhysX |
| **Dynamic Manipulation Benchmarks** | **Verified** | Dynamic object relocations, lift height verification (>30 mm gain) |

Detailed verification metrics and visual evidence captures are documented in [SYSTEM_VERIFICATION.md](SYSTEM_VERIFICATION.md).

---

## Repository Structure

```
manipulation_framework/
├── configs/                  # Strictly-typed YAML configuration files
│   ├── default.yaml          # Base robot, camera, planning, physics parameters
│   ├── benchmark.yaml        # YCB benchmark object test scene
│   └── assets.yaml           # Measured asset catalog (meshes, masses, colliders)
├── mfw/                      # Core Manipulation Framework package
│   ├── config/               # Schema definitions and strict config loader
│   ├── controllers/          # Joint trajectory controller & target writer
│   ├── core/                 # Interfaces, data types, error hierarchy
│   ├── gr00t_bridge/         # ZeroMQ IPC client/server & safety filter
│   ├── grasp/                # Grasp candidate generation & scoring
│   ├── language/             # Qwen intent parser & speech front-end
│   ├── memory/               # Working memory & spatial scene graph
│   ├── motion/               # Lula RRT, RMPflow & trajectory interpolation
│   ├── physics/              # PhysX material, contact & collision bindings
│   ├── planner/              # Task planner & finite state machine
│   ├── robot/                # Franka Panda kinematic abstraction & TCP
│   ├── simulation/           # Isaac Sim app lifecycle & scene builder
│   ├── skills/               # Atomic skills (Pick, Place, Observe, Move, Home)
│   ├── utils/                # 3D transforms, SE(3) math, structured logging
│   └── vision/               # Camera drivers, point cloud filters, pose estimation
├── renders/                  # Camera captures & verification evidence
│   └── pipeline_verification/# Exterior & wrist camera captures at each stage
├── scripts/                  # CLI entry points and background workers
│   ├── run_assistant.py      # Main interactive CLI assistant
│   ├── run_isaac_tests.py    # Isaac Sim test runner
│   ├── llm_worker.py         # Dedicated worker for Qwen 2.5-3B-Instruct
│   ├── speech_worker.py      # Dedicated worker for microphone & ASR
│   ├── groot_server.py       # ZeroMQ server for GR00T policy inference
│   ├── validate_scene.py     # 9-point physical asset & reachability validator
│   └── benchmark_manipulation.py # Automated manipulation benchmark runner
├── tests/                    # 349 pure-logic unit tests + 99 Isaac Sim integration tests
├── ARCHITECTURE.md           # Comprehensive architectural specification & design rules
├── HANDOFF.md                # Benchmark analysis & measured physical signals
├── SYSTEM_VERIFICATION.md    # Experimental evaluation & verification evidence
└── requirements.txt          # Package dependencies
```

---

## Quick Start & Running the Framework

### 1. Run Pure-Logic Regression Tests (No GPU / No Simulator)
The pure logic tests validate geometry, config schemas, grasp scoring, motion kinematics, LLM prompt templates, and state machines in seconds:

```bash
cd manipulation_framework
python -m pytest tests/ -q -m "not isaac"
```

### 2. Run Isaac Sim Integration Tests
Runs the full 99-test suite inside the Isaac Sim environment:

```bash
python manipulation_framework/scripts/run_isaac_tests.py
```

### 3. Run the Interactive Assistant with Live Qwen LLM
Launches Isaac Sim with Franka Panda, activates live camera perception, boots the resident Qwen 2.5-3B LLM, and waits for user commands:

```bash
python manipulation_framework/scripts/run_assistant.py --gui --llm qwen --interactive
```

Example commands to try in interactive mode:
```text
> "what do you see"
> "grab the block and put it inside the box"
> "pick up the red block"
> "place it in the green box"
> "move left 5 cm"
> "go home"
```

### 4. Scene & Asset Physical Validation
Verifies physical properties (colliders, materials, center of mass, reachability, dimensions) across all objects:

```bash
python manipulation_framework/scripts/validate_scene.py --gui --hold 30
```

| Check | Failure Mode Prevented |
|---|---|
| `collision_mesh` | Object falls through the table; reads as perception failure |
| `physics_material` | Default 0.5 friction; object slips from parallel fingers |
| `mass` | Unrealistic inertia; flies off on contact or resists movement |
| `center_of_mass` | Topples immediately upon gripper release |
| `semantic_label` | Segmentation returns background; perception blinded |
| `dimensions` | Mesh authoring scale errors; verified against physical ground-truth |
| `reachable` | Ensures objects lie inside kinematic dexterous workspace |
| `no_penetration` | Prevents interpenetrating spawns and explosive physics repulsion |
| `stable_at_rest` | Zero physics drift before manipulation sequence initiates |

### 5. Run the Franka Panda on the Benchmark Scene
The benchmark scene (`configs/benchmark.yaml`) is an office table with scanned YCB objects that the Franka picks reliably — marker, banana, pudding box, soup can and foam brick — plus a bowl and a sorting bin as destinations.

The repository must sit **inside** the Isaac Sim folder (`isaac-sim-standalone-5.1.0-windows-x86_64\<repo>`), because `..\python.bat` is Isaac Sim's own Python launcher. Run from the repository folder:

```bash
..\python.bat scripts\run_assistant.py --config configs/benchmark.yaml --gui -c "what do you see" -c "pick up the marker" -c "put it in the bowl" -c "go home"
```

Each `-c` runs one command, in order. Drop `--gui` to run headless; use `--interactive` instead of `-c` to type commands one at a time.

Benchmark every object — pick it, then place it back where it was picked:

```bash
..\python.bat scripts\benchmark_manipulation.py --config configs/benchmark.yaml --place
```

Success is judged by simulator ground truth (the object must rise more than 3 cm, then come to rest after release), not by the skill's own report. Every run writes `logs/benchmark/<timestamp>_manipulation.json` and `.csv`.

---

## Key Design Principles

1. **Strict Process Isolation for Neural Models**:
   - `Qwen/Qwen2.5-3B-Instruct` and speech engines run in isolated resident worker processes communicating over standard TCP/ZeroMQ sockets. This protects simulator real-time dynamics from GPU memory contention and library conflicts.
2. **Atomic Skill Execution**:
   - Complex natural-language tasks are decomposed into atomic, verifiable primitives (`Pick`, `Place`). Skills never directly invoke other skills; coordination is maintained strictly by the `TaskPlanner` and `WorkingMemory`.
3. **Physics-Driven Manipulation**:
   - No teleportation, coordinate hacking, or artificial parent constraints. Grasps succeed solely through frictional contact forces modeled in PhysX.
4. **Calibrated Tool Center Point**:
   - Franka fingertip midpoint TCP is derived from `panda_hand` at `[0, 0, 0.1034] m`, eliminating the 51.5 mm error present in standard finger-prim configurations.

# Manipulation Framework (`mfw`)

Modular robotic manipulation on Isaac Sim 5.1 + PhysX, with a GR00T N1.7 policy
backend. Franka Panda, wrist + exterior cameras, pure physics grasping, atomic
natural-language skills.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, the three places it
corrects the original brief, and a table of the bugs the test gates caught.

## Status

All eight phases implemented and gated.

| Suite | Tests | Needs |
|---|---|---|
| Pure logic | **349** (~10 s) | No simulator, no GPU |
| Isaac integration | **99** | Isaac Sim 5.1 |

| Phase | Scope |
|---|---|
| 1 | Robot, cameras, physics, TCP, IK, calibration |
| 2 | Perception: 6-DoF pose, tracking, scene graph |
| 3 | Lula RRT + Cartesian planning, trajectory execution |
| 4 | Grasp generation and scoring |
| 5 | `Pick` — physics grasp, lift, verify, hold |
| 6 | `Place` — pose synthesis, release, retreat |
| 7 | Language, memory, planner, state machine, GR00T bridge |
| 8 | Speech front-end |

## Running the tests

The pure-logic suite needs **no simulator and no GPU** — `core/`, `utils/`,
`config/`, and the geometry, grasp, language and memory layers import no Isaac:

```bash
cd manipulation_framework
set PYTHONPATH=%CD%
..\python.bat -m pytest tests/ -q -m "not isaac"
```

The Isaac suite must go through the runner script:

```bash
..\python.bat manipulation_framework\scripts\run_isaac_tests.py
```

> Do **not** use `python.bat -m pytest` for Isaac tests. `SimulationApp` parses
> `sys.argv`, so pytest's own flags reach omni.kit as "Ill formed parameter" and
> tear the app down mid-fixture — leaving a truncated report and a misleading exit
> code 0. The runner clears argv and writes results to
> `logs/isaac_test_report.txt` as they happen.

Phase gates: `-m phase1` … `-m phase8`. Don't start a phase until the previous is green.

## Quick start

```python
from mfw.assistant import Assistant

with Assistant(config_path="configs/default.yaml") as robot:
    print(robot.command("what do you see").result.data["objects"])

    robot.command("pick up the can")     # picks, holds, WAITS
    robot.command("place it")            # "it" = the held object
    robot.command("move left 5 cm")      # moves left only
```

Every command is atomic: one utterance, one action, then the robot waits. "Pick
the can" never places, and `pick up the can and put it in the box` executes only
the pick.

## Configuration

Every tunable lives in [`configs/default.yaml`](configs/default.yaml). Loading is
strict — an unknown key raises at startup rather than silently keeping a default.
If you find yourself editing a number inside `mfw/`, it belongs in config.

```python
from mfw.config.schema import load_config
cfg = load_config("configs/default.yaml", overrides={"simulation": {"headless": False}})
```

## Benchmark environment

[`configs/benchmark.yaml`](configs/benchmark.yaml) is an office/lab scene built
from real scanned meshes rather than coloured primitives. It inherits
`default.yaml` via `extends:` and restates only what differs, so the tuned
physics stays in one place.

```bash
# Build it, settle it, and prove every object is usable. Exits non-zero on failure.
..\python.bat manipulation_framework\scripts\validate_scene.py

# Watch it
..\python.bat manipulation_framework\scripts\validate_scene.py --gui --hold 30

# A randomised layout (reproducible from the seed)
..\python.bat manipulation_framework\scripts\validate_scene.py --randomize --seed 7

# Run the assistant in it
..\python.bat manipulation_framework\scripts\run_assistant.py --config configs/benchmark.yaml --gui
```

**Objects** come from the YCB benchmark set — the object set Open X-Embodiment
and most grasping papers evaluate on, so results are comparable with published
work. The scene deliberately spans a difficulty range: a 15 g marker and an
895 g drill impose opposite demands on grip force, the mug is a
handle-and-cavity problem, and the banana is curved. Success on a uniform object
set is not a measurement.

**No branded assets were fabricated.** Requests for Bisleri, Coca-Cola and the
like are served by real scanned objects of similar geometry, and every
substitution is recorded in `substitutes:` and printed in the validation report.

**The catalogue** ([`configs/assets.yaml`](configs/assets.yaml)) is the single
source of truth for each object's mesh, mass, true dimensions, collider strategy
and semantic label. Scenes refer to an asset by name and inherit all of it.
`size_m` is *measured from the meshes* by
[`scripts/measure_assets.py`](scripts/measure_assets.py), not copied from the YCB
spec sheet — the two disagree (the published banana width is 36 mm; the scanned
mesh's bounding box is 74 mm, because a banana is curved).

### What the validator checks, and why

Every check corresponds to a failure that is silent at build time:

| Check | What its absence looks like |
|---|---|
| `collision_mesh` | Object falls through the table; reads as a perception failure |
| `physics_material` | Default 0.5 friction; object slides out of the fingers |
| `mass` | Flies off on contact, or refuses to move |
| `center_of_mass` | Topples the instant it is released |
| `semantic_label` | Segmentation returns background — perception genuinely cannot see it |
| `dimensions` | **The one the others miss.** See below |
| `reachable` | Every command about it fails at IK, far from the cause |
| `no_penetration` | Interpenetrating spawns get driven apart violently |
| `stable_at_rest` | Drift invalidates every before/after measurement |

The `dimensions` check exists because of a real bug this scene shipped with.
`scale` defaulted to `0.05` — right for a primitive cuboid, catastrophic for a
mesh authored at true scale. Every YCB object spawned at 5% size. Each one still
had a collider, the correct mass, and rested stably on the table, so **all eight
other checks passed** — while a 66 mm can covered two pixels of camera image and
perception reported an empty table. Nothing that inspects physics can catch a
scale error; only comparing rendered size against a known ground truth can.

Two related traps, both measured rather than assumed:

* **The YCB set is authored with up along −Y**, and Isaac's world is Z-up. Left
  uncorrected, cans lie down and roll and the bowl balances on its rim — which
  reads as unstable physics. The *sign* of the correction cannot be recovered
  from bounding boxes, since extents are magnitudes and ±90° give identical
  numbers; it was settled by dropping a brick over the bowl and checking whether
  it landed inside.
* **Seventeen of the twenty-one YCB assets ship without colliders.** Only the
  four under `Axis_Aligned_Physics` have them pre-authored. `PhysxCollisionAPI`
  on the root prim is a no-op for a referenced mesh — the geometry is on
  descendant `Mesh` prims — so [`mfw/physics/collision.py`](mfw/physics/collision.py)
  authors them there and treats zero colliders as an error rather than a warning.

## GR00T backend (optional)

The policy **always** runs in a separate process: Isaac Sim 5.1 ships Python
3.11.13 and GR00T N1.7 requires `>=3.12,<3.13`, so they can never share an
interpreter.

```bash
# Mock policy: no torch, no checkpoint. Exercises the whole integration.
py -3.12 -m mfw.gr00t_bridge.server --mock

# Real checkpoint
py -3.12 -m mfw.gr00t_bridge.server --checkpoint nvidia/GR00T-N1.7-3B
```

Then set `gr00t.enabled: true` and route individual skills:

```yaml
gr00t:
  enabled: true
executor_overrides:
  pick: gr00t        # classical still handles observe, stop, go_home, ...
```

> **Platform caveat.** The server is configured for Windows Python 3.12, which
> NVIDIA does not support: `flash-attn` and `deepspeed` are Linux-gated, so
> attention falls back to eager, and Blackwell (sm_120) needs CUDA 12.8+ wheels.
> The transport is host-agnostic, so switching to WSL2 Ubuntu-22.04 (glibc 2.35 —
> exactly flash-attn's minimum) is a `gr00t.host` change and nothing more.

## Voice (optional)

```bash
python -m pip install faster-whisper sounddevice   # into a SYSTEM python
python scripts/speech_worker.py --model base.en
```

Speech runs in a subprocess by design: CTranslate2 crashes inside Isaac Sim's
interpreter on Windows and takes the simulator with it. Transcripts join the same
path as typed text, so there is only one command pipeline.

## Five things to know before editing

1. **`SimulationApp` must be constructed before any `omni.*` / `isaacsim.*`
   import.** Every Isaac-touching module here imports Isaac *inside functions*
   for this reason.
2. **The TCP is not `panda_rightfinger`.** That frame is 51.5 mm from the real
   grasp point (measured via Lula FK). Use `robot.tcp_pose()`.
3. **Never write an object's pose.** Manipulation is contact forces only — no
   parenting, no teleporting, no fake attachment. `is_grasping` comes from
   evidence, not from having issued a close command.
4. **A skill never calls another skill.** Skills receive a `SkillContext` that
   deliberately excludes the registry, so they *cannot*.
5. **Only `vision/` touches cameras; only `controllers/` writes joint targets.**

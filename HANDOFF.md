# Handoff: benchmark-scene manipulation

Self-contained brief for continuing this work. Read it before changing anything
in `mfw/vision/`, `mfw/grasp/`, or `configs/benchmark.yaml`.

## The one-line state

The office benchmark scene is **built, perceptually correct, and not
manipulable**: 9/9 objects detected with correct labels, colours, sizes and
language grounding, but only **2 of 8** graspable objects can actually be
picked. The default 3-primitive scene works completely and is gated by 99 Isaac
tests.

**The two that lift are the two SHORTEST objects (banana 39 mm, foam_brick
51 mm). Everything over 80 mm tall fails, at every position tested.** Object
height is the variable -- not position, not clutter, not reach. See THE SIGNAL
below.

## How to run things

```bash
# Everything below is run from the workspace root (e.g. C:\Hiren\Capstone)

# Pure-logic tests (~8 s, no GPU) -- 349 passing
.\python.bat -m pytest manipulation_framework/tests/ -q -m "not isaac"

# Isaac tests (~10 min) -- 99 passing. Do NOT use pytest directly for these;
# SimulationApp parses sys.argv and dies on pytest's flags.
.\python.bat manipulation_framework\scripts\run_isaac_tests.py

# Scene validation: 9 physical checks per object + camera exposure
.\python.bat manipulation_framework\scripts\validate_scene.py [--gui --hold 300]

# THE metric. Ground-truth height gain > 30 mm, which no reporting bug can fake.
.\python.bat manipulation_framework\scripts\benchmark_manipulation.py
#   -> logs/manipulation_benchmark.txt and .json

# Camera renders (RGB, depth, segmentation for both cameras)
.\python.bat manipulation_framework\scripts\render_cameras.py

# The working demo, on the default scene
.\python.bat manipulation_framework\scripts\run_assistant.py --gui --interactive
```

Isaac swallows Python `stdout` (routed through `omni.kit.app` at info level) and
kills the process on USD coding errors with **exit code 0 and no traceback**.
Every script here therefore writes results to a file as well as printing them.
When something "silently does nothing", add file-write checkpoints and bisect.

## READ THIS FIRST: measurement discipline

**Single runs are not measurements.** The same unchanged configuration produced
3/8 and then 2/8. Several conclusions earlier in this work were drawn by
comparing one run against one run, and were therefore invalid -- including the
attribution of a regression to a change that turned out not to cause it.

Before drawing any conclusion from a pick-rate delta, run the benchmark **at
least 5 times** and compare distributions, or make the run deterministic. The
per-object failure *reasons* also change between runs (the mug has failed three
different ways with no code change between two of them), so treat those as
samples, not facts.

This is the single most important thing to fix before any further grasp work.

## Dead ends -- measured, do not repeat

Four hypotheses for the perception/grasp errors were tested and are wrong. Each
is documented in-code at the relevant site.

1. **Mask halo on the support surface.** The `points_above_plane` filter existed
   but was aimed at `ground_plane_z = 0.0` (the floor) while objects rest on a
   table at 0.40 m, so it had never removed anything. Fixed to use the real
   support height. Soup can centre error: 71.3 mm before, **71.4 mm after**. No
   effect. (Kept -- it is correct in principle -- but it fixes nothing here.)

2. **Square-pixel intrinsics.** `vertical_aperture` (2.453) genuinely disagrees
   with the resolution aspect ratio, which implies 2.922 for 640x480 at
   `horizontal_aperture` 3.896, making `fy` 19% larger than `fx`. Forcing
   `fy = fx` made reconstruction **worse**, adding up to 43 mm of vertical error
   to objects whose Z had been accurate to 5 mm. Isaac's renderer uses the
   configured vertical aperture. **Reverted.**

3. **Cross-view centre fusion.** Taking the horizontal centre from the midpoint
   of both cameras' trimmed extents. Genuinely improved *accuracy* -- cracker box
   centre error 47 mm -> 14 mm -- and still made the robot worse. **Reverted.**
   A better-centred box is not a better *graspable* box: grasp candidates are
   generated from the pose and filtered on it, so moving a centre relocates every
   pre-grasp standoff.

4. **Support-constrained height.** Deriving height from the support plane and the
   observed top face. Fixes a real error (mustard bottle 191 mm true, 156 mm
   measured) but changes which grasp candidates exist. Left in behind
   `perception.support_constrained_height`, **default off**.

Also note: the first version of (3) called `remove_statistical_outliers` on an
uncapped point union. That function is O(n^2) in memory and a single instance
reaches 30,000 points (~7 GB). It hung a run for 16 minutes. `_fit_single_view`
caps its input at `max_points_for_outlier_filter` for exactly this reason -- any
new code path that denoises a cloud must do the same, or use percentiles.

## RESOLVED: the spurious SafetyViolations

`JointTrajectoryController._assert_within_workspace` ran FK on **every waypoint**
of a joint-space trajectory. The violating coordinate was therefore never a grasp
target -- it was an intermediate configuration of an RRT path. That explained the
whole symptom set: x pinned at the boundary (the check fires the instant a path
crosses it), y arbitrary and sign-flipping between runs (**RRT is randomised**),
and the run-to-run instability that invalidated earlier comparisons.

Fixed: goals are checked with zero margin, intermediate waypoints with
`motion.workspace_transit_margin` (0.25 m). The mug stopped throwing
SafetyViolation and now reports its real failure.

**This did not change the pick rate** (still 2/8) -- it removed a spurious
failure that was *masking* a real one. That is progress in diagnosis, not in
capability, and the two should not be confused.

## RESOLVED BY EXPERIMENT: clutter is NOT the cause

`configs/benchmark_single.yaml` runs the identical scene with **one object** --
the cracker box, at its usual pose, no other objects present, so no obstacles
beyond the table.

It fails identically, at the byte-identical coordinate:

```
TCP in transit [-0.202, -0.526, 0.599]
```

Same value as the nine-object run. So for this object the failure is
**deterministic**, not the RRT randomness seen elsewhere, and **obstacle clutter
plays no part**. Reruns of the single-object config are the cheapest way to test
any future planner change.

IK is warm-started at every call site (`grasp/scorer.py`, `motion/planner.py`,
`controllers/joint_controller.py` all pass `seed=`), so the *goal configuration*
is not flipped. The path to it is the problem.

## THE SIGNAL: object HEIGHT, not position

Tested and **rejected**: clutter (`benchmark_single.yaml`, fails identically with
one object) and inner reach (`benchmark_reachtest.yaml`, the cracker box fails at
x=0.52, the exact spot where the foam brick lifts reliably). Position is not the
variable. Neither is distance from the base -- the mug has the *shortest*
pre-grasp standoff of any object (0.656 m) and fails, while the banana at 0.776 m
lifts.

Height separates them perfectly:

| lifts      | height | fails       | height |
|------------|--------|-------------|--------|
| banana     | 39 mm  | mug         | 81 mm  |
| foam_brick | 51 mm  | soup_can    | 102 mm |
|            |        | marker      | 121 mm |
|            |        | drill       | 187 mm |
|            |        | cracker_box | 213 mm |

Under 60 mm works. Over 80 mm fails. No exceptions, across every position tested.
This is the sharpest signal in the whole dataset and explains why relocating
objects never helped.

**Start here.** Likely places, in order:

1. `GraspScorer._fingers_clear_support` and `support_clearance()` in
   `mfw/grasp/generator.py` -- clearance is computed against the support height,
   and a tall object's grasp sits far above it. Check the sign and whether tall
   objects are being rejected for the wrong reason.
2. `_offsets_along()` samples grasp points at +/-0.3 x extent along the *free*
   axis. For an upright tall object that axis is Z, so candidates spread up and
   down its height -- verify they are not landing above the object's top or
   below the table.
3. Ordering of `planner.exclude_from_collision(target_track_id)` in
   `mfw/skills/primitives.py:504` relative to `update_collision_world()`. If the
   world is rebuilt AFTER the exclusion, the target is re-enabled as an obstacle
   and a tall object blocks its own descent corridor -- which would hurt tall
   objects far more than flat ones.

**Cheapest decisive test:** lay the cracker box on its side (`quat` giving a
90-degree rotation, so its 213 mm axis is horizontal and its height becomes
72 mm) at the same position. If it then lifts, height is confirmed as causal
and (3) is the most likely mechanism.

## Superseded leads (resolved, kept for context)

### Two planner bugs -- A fixed, B outstanding


The six failures are not one problem. Separate them before touching anything:

**A. RRT returns a valid but wildly indirect path** (cracker_box). It plans
successfully and the path sweeps the TCP to x = -0.202, behind the robot's own
base, to reach an object 0.33 m in front of it. There is **no path shortcutting
or smoothing** anywhere in `mfw/motion/planner.py` -- the raw RRT tree path is
executed as returned, and raw RRT paths are notoriously meandering.
*Fix:* add a collision-checked shortcut pass -- repeatedly try to replace
sub-paths with straight-line joint interpolation, keeping the replacement when
the planner's world reports it collision-free. Verify with
`benchmark_single.yaml`, which is deterministic.

**B. RRT returns NO path** (mug, marker, drill, and cracker_box in some runs).
Measured: raising `max_planning_time_s` 5 -> 15 and `max_replan_attempts` 3 -> 6
found **zero** additional paths and doubled runtime. Reverted. More search is not
the answer; something is making the pre-grasp standoff genuinely unreachable.
Suspect the inner reach limit -- see the x-coordinate correlation above.

Do NOT conflate A and B. They need different fixes, and the earlier attempt to
treat "planner problem" as one thing is why the budget increase was wasted.

## Superseded lead (resolved, kept for context)


State after the transit-margin fix. Six failures, and **four are now "no path to
the pregrasp standoff"** -- the spurious SafetyViolations that were masking them
are gone.

Tried and measured: `max_planning_time_s` 5 -> 15 and `max_replan_attempts`
3 -> 6. **Zero additional paths found**, runtime doubled (mustard bottle
16.7 s -> 32.6 s). Reverted. This is decisive: RRT is not returning poor paths
that more search would refine, it is returning none. More budget is not the fix
and neither is path smoothing.

Two candidate causes, neither yet tested:

1. **Obstacle clutter.** Nine perceived objects become nine inflated
   `VisualCuboid` obstacles in the arm's annulus, against three in the default
   scene that works. The pre-grasp standoff sits 100 mm above a target whose
   *neighbours* are still obstacles, so the approach corridor may simply be
   closed. The target itself IS correctly excluded
   (`planner.exclude_from_collision`, called from `mfw/skills/primitives.py`).
   **Decisive test:** build a benchmark variant with a single object and
   re-measure. If picks succeed, clutter is confirmed and the fix is either
   smaller `obstacle_inflation`, obstacles only within some radius of the path,
   or fewer objects in the default layout.

2. **Objects too close to the base.** The table moved to x=0.45 to solve the
   reach problem, and objects now sit at x = 0.26-0.52. Several are near
   `robot_min_reach_m` (0.40), where the arm must fold sharply and has little
   orientation freedom. Note foam_brick (x=0.522) and banana (x=0.467) -- the two
   that DO lift -- are the two furthest out. **That correlation is suspicious and
   is the first thing to check.**

Do the single-object test first; it separates (1) from (2) in one run.

## Superseded lead (resolved, kept for context)


With the margin in place, the cracker box still aborts -- at
`TCP in transit [-0.202, -0.526, 0.599]`. **x = -0.202 is behind the robot's own
base.** The RRT is routing the arm through a grossly indirect path to reach an
object 0.33 m in front of it. That is a planner problem, not a grasp or
perception one, and it is the next thing to chase: look at
`mfw/motion/planner.py`, the Lula RRT obstacle set (the office furniture and room
are now obstacles, which they were not in the default scene), and whether the
planner is being handed a seed configuration that sends it the long way round.

Widening the box further is NOT the fix -- it was already widened twice and the
violation simply moved with it. The path itself is wrong.

## Older lead (superseded, kept for context)


Two of the six failures are a `SafetyViolation` whose TCP target is nowhere near
the object:

```
mug at (0.279, 0.184)   ->  TCP target [0.048, -0.467, 0.719]
                        ->  TCP target [0.050, +0.497, 0.565]   (next run, y flipped)
cracker_box at (0.298, -0.144) -> TCP target [0.048, -0.480, 0.621]
```

The x coordinate is pinned at ~0.05, which is exactly `scene.workspace_min[0]`.
The y is arbitrary and changes sign between runs on a stationary object.

That is not a grasp-synthesis error -- those coordinates are not near the object
in any run, and are inconsistent with each other. It looks like a target being
**clamped to the workspace boundary and then validated against that boundary**.
Find where the TCP target is produced and clamped; `mfw/controllers/`,
`mfw/motion/`, and the `place`/`pick` skills are the places to look. Widening
`workspace_min` does not help -- it was already lowered 0.12 -> 0.05 and the
violation simply moved with it.

## Remaining failures, by cause

| object | failure | cause |
|---|---|---|
| foam_brick | **LIFTS** +118 mm | working |
| banana | **LIFTS** +70 mm | working |
| mug | SafetyViolation / no path / lifted | **nondeterministic**; see live lead |
| cracker_box | SafetyViolation | see live lead |
| soup_can | closes on air | centre error 71 mm on a 68 mm object |
| mustard_bottle | closes on air | centre error ~24 mm |
| drill | closes on air | centre error ~22 mm |
| marker | no path to pre-grasp | planning; its centre is accurate to 3 mm |
| bowl | skipped, correctly | 161 mm destination, not a target |

The soup can is the sharpest unexplained case: its cloud has the **right size**
(72x74x101 for a 68x68x102 can) but sits 71 mm away, on the **far side of the can
from the camera that observed it** -- geometrically impossible for a real
surface. Errors correlate perfectly with visibility: exterior-camera-only
objects show 71/53/43/24 mm, two-camera objects show 22/22/18/14/3 mm.

## What is solid and should not be re-litigated

- **Reach is a sphere, not a box.** `scene.robot_reach_m` (0.78) and
  `robot_min_reach_m` (0.40) are enforced in `ObjectPlacer` and the validator.
  The workspace box's far corner is 1.27 m from a base that reaches 0.855 m; five
  of nine objects were originally out of range and every one passed the box test.
- **The table sits at x=0.45, 0.62x1.00.** A floor-mounted Franka reaching across
  the original 0.8 m-deep table at 0.55 m could serve only ~0.11 m2 of it against
  0.17 m2 of object footprint -- no valid layout existed.
- **Grasp width**: `max_grasp_width` 0.078, `finger_width_margin` 0.010. The old
  0.075/0.012 gave a 63 mm ceiling on an 80 mm gripper and rejected the 65 mm
  soup can. This fix is what made the mug graspable.
- **YCB assets** are authored **-Y up**, origins centred, and 17 of 21 ship
  without colliders. See `configs/assets.yaml` and `mfw/physics/collision.py`.
  The upright sign was settled by dropping a brick over the bowl (lands inside at
  -90 degrees, perches on an inverted dome at +90) -- bounding boxes cannot
  distinguish the two.
- **Indoors, dome and distant lights do nothing** -- they are a sky and a sun and
  the office has a ceiling. Turning both off made the scene *brighter*. All light
  comes from `scene.workspace_light_intensity`.
- **`observe()` must render first.** `SimulationContext.step` only renders when
  windowed; without `render_frames_before_capture` (8, measured) RGB comes back
  black while depth and segmentation stay correct, so only colour-qualified
  references break.

## Suggested order of work

1. Make the benchmark repeatable (`--repeat N`, report mean and spread). Nothing
   below is measurable until this exists.
2. Trace the clamped TCP target. Two failures, and it is nondeterministic, which
   makes it the most likely single cause of run-to-run instability.
3. Only then return to centre accuracy for the three "closes on air" objects --
   and judge any change by pick rate over N runs, never by centre error alone.
   Dead end (3) is the proof that those two metrics can move in opposite
   directions.

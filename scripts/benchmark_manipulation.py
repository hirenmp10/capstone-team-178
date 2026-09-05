"""Attempt a real pick (and optionally a place) on every object in a scene.

    ..\\python.bat manipulation_framework\\scripts\\benchmark_manipulation.py

Success is ground-truth height gain, not the skill's own return value. Nothing
in this framework writes an object pose, so the only mechanism that can raise an
object is genuine fingertip contact and friction -- which makes a >3 cm rise the
one claim that cannot be faked by a bug in the reporting path.

This exists because passing the Isaac test suite does *not* mean manipulation
works in a given scene. Those tests run against the three coloured primitives in
``configs/default.yaml``. A benchmark scene of scanned meshes, with a room and
furniture for the planner to avoid, is a different problem, and the honest way
to find out which objects the robot can actually handle is to try all of them.

Per object it reports: whether a grasp was synthesised, whether the arm reached
it, whether it left the table, and whether it was still held afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LIFT_THRESHOLD_M = 0.03


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--place", action="store_true", help="also attempt a place after each pick")
    parser.add_argument("--only", default="", help="comma-separated object names to try")
    parser.add_argument("--report", default=str(_ROOT / "logs" / "manipulation_benchmark.json"))
    parser.add_argument("--log", default=str(_ROOT / "logs" / "manipulation_benchmark.txt"))
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]

    from mfw.config.schema import load_config

    overrides: dict = {"logging": {"console": False}}
    if args.gui:
        overrides["simulation"] = {"headless": False}
    config = load_config(args.config, overrides=overrides)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from mfw.simulation.runtime import Runtime

    runtime = Runtime(config)
    results: list[dict] = []
    try:
        runtime.build()

        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        targets = [
            obj
            for obj in runtime.scene_config.objects
            if not wanted or obj.name in wanted
        ]

        emit(f"scene: {args.config}")
        emit(f"attempting {len(targets)} object(s); success = ground-truth rise "
             f">{LIFT_THRESHOLD_M * 1000:.0f} mm")
        emit()

        initial_poses = _capture_initial_poses(runtime)

        for obj in targets:
            results.append(_attempt(runtime, obj, args.place, emit))
            _reset_between_attempts(runtime, initial_poses, emit)

        emit()
        _summarise(results, emit)

        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
        emit(f"JSON report: {args.report}")

        attempted = [r for r in results if not r.get("skipped")]
        lifted = sum(1 for r in attempted if r.get("lifted"))
        return 0 if lifted == len(attempted) else 1
    finally:
        runtime.close()


def _attempt(runtime, obj, do_place: bool, emit) -> dict:
    """One pick attempt, judged against ground truth."""
    import numpy as np

    name = obj.name
    prim_path = runtime.scene_builder.object_prim_paths.get(name, "")
    record: dict = {"object": name, "asset": obj.asset, "prim_path": prim_path}

    height_before = _truth_height(prim_path)
    record["height_before"] = height_before

    # Perception decides what to grasp. The scene config is used only to know
    # which objects exist and to read ground truth afterwards -- the skill is
    # given a *label*, never a pose.
    try:
        # Twice, and it is the *second* graph that counts: a track is confirmed
        # only on its second sighting (Track.is_confirmed requires hits >= 2),
        # so the first observation of a fresh scene is always empty. Keeping the
        # first one made the very first object of a run look unperceivable while
        # every later one worked, purely because earlier attempts had warmed the
        # tracker.
        runtime.vision.observe()
        scene = runtime.vision.observe()
    except Exception as exc:  # noqa: BLE001
        record.update(perceived=False, lifted=False, error=f"observe failed: {exc}")
        emit(f"  {name:16} PERCEPTION FAILED: {exc}")
        return record

    spec = runtime.asset_registry.objects.get(obj.asset)
    label = spec.semantic if spec is not None else name

    # Containers are destinations, not targets. The YCB bowl is 161 mm across
    # against an 80 mm gripper -- it is *correct* for the pick to fail, so
    # counting it as a failure would understate the real rate.
    record["graspable"] = bool(spec is None or spec.graspable)
    if not record["graspable"]:
        record.update(lifted=False, skipped=True, perceived=True)
        emit(f"  {name:16} SKIPPED  not graspable by this embodiment "
             f"(min extent {min(spec.size_m) * 1000:.0f} mm) -- it is a destination")
        return record
    matches = scene.by_label(label)
    record["perceived"] = bool(matches)
    if not matches:
        record.update(lifted=False, error=f"{label!r} not in the scene graph")
        emit(f"  {name:16} NOT PERCEIVED as {label!r}")
        return record

    started = time.perf_counter()
    try:
        result = runtime.skills.execute("pick", {"target": label})
        record["skill_ok"] = bool(result.ok)
        record["skill_message"] = str(result.message)
    except Exception as exc:  # noqa: BLE001 - an exception is a result, not a crash
        record.update(skill_ok=False, skill_message=f"{type(exc).__name__}: {exc}", lifted=False)
        emit(f"  {name:16} RAISED {type(exc).__name__}: {exc}")
        return record
    record["duration_s"] = round(time.perf_counter() - started, 1)

    height_after = _truth_height(prim_path)
    gain = height_after - height_before
    record["height_after"] = height_after
    record["gain_m"] = gain
    record["lifted"] = bool(gain > LIFT_THRESHOLD_M)

    width = runtime.robot.get_gripper_width()
    record["gripper_width"] = float(width)
    record["still_held"] = bool(
        runtime.memory.get_held_object() is not None
        and width > runtime.config.robot.gripper_closed_width + 0.004
    )

    verdict = "LIFTED " if record["lifted"] else "no lift"
    emit(
        f"  {name:16} {verdict} gain={gain * 1000:+7.1f} mm  "
        f"skill_ok={str(record['skill_ok']):5} held={str(record['still_held']):5} "
        f"{record['duration_s']:5.1f}s  {record['skill_message'][:60]}"
    )
    if not record["lifted"] and record["skill_ok"]:
        # The dangerous case: the skill reported success while the object never
        # moved. Worth calling out explicitly rather than leaving in a table.
        emit(f"  {'':16} ^^ skill reported success but the object did not rise")

    if do_place and record["still_held"]:
        try:
            place = runtime.skills.execute("place", {"target": "it"})
            record["place_ok"] = bool(place.ok)
            record["place_message"] = str(place.message)
            emit(f"  {'':16} place: ok={place.ok} {str(place.message)[:60]}")
        except Exception as exc:  # noqa: BLE001
            record["place_ok"] = False
            record["place_message"] = f"{type(exc).__name__}: {exc}"
            emit(f"  {'':16} place RAISED {type(exc).__name__}: {exc}")

    return record


def _capture_initial_poses(runtime) -> dict[str, tuple]:
    """Record every object's spawn pose so attempts can be made independent."""
    import numpy as np
    from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

    poses: dict[str, tuple] = {}
    for name, prim_path in runtime.scene_builder.object_prim_paths.items():
        position, quat = get_world_pose(prim_path)
        poses[name] = (np.array(position, dtype=np.float64), np.array(quat, dtype=np.float64))
    return poses


def _reset_between_attempts(runtime, initial_poses: dict, emit) -> None:
    """Put the arm home and every object back where it started.

    Isolating attempts matters more than it looks. Resetting only the robot
    leaves objects wherever the last attempt shoved them: measured here, the
    mustard bottle lifted 115.7 mm when tried first and failed outright when
    tried after a soup-can attempt, with two objects recording a ~52 mm drop
    because they had been knocked to the floor.

    Object poses are restored individually rather than through
    ``World.reset()``. The wholesale reset is the obvious tool and it is wrong
    here -- it also re-initialises the articulation and the physics views that
    the motion planner holds references to, after which *every* target failed
    with "no path to the pregrasp standoff", including one that had lifted
    cleanly twice. Writing poses back is narrower and leaves the planner alone.

    Writing object poses is legitimate in a measurement harness and nowhere
    else: this is the same authority the scene builder has at spawn time. No
    runtime code path can reach it.
    """
    import numpy as np  # noqa: PLC0415

    try:
        runtime.robot.open_gripper()
        runtime.sim.step(30)
        runtime.memory.set_held_object(None)
        runtime.robot.go_home_immediate()

        world = runtime.sim.world
        for name, (position, quat) in initial_poses.items():
            body = world.scene.get_object(name)
            if body is None:
                continue
            body.set_world_pose(position=position, orientation=quat)
            # Zero the velocities too, or the object keeps the momentum the
            # failed attempt gave it and slides off the moment physics resumes.
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))

        runtime.sim.settle()
    except Exception as exc:  # noqa: BLE001
        emit(f"  (reset failed: {type(exc).__name__}: {exc})")


def _truth_height(prim_path: str) -> float:
    """Ground-truth Z. Legitimate only in a measurement harness.

    Perception, planning and skills have no accessor like this; that is what
    makes "never assume object positions" structural rather than aspirational.
    """
    if not prim_path:
        return float("nan")
    from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

    position, _ = get_world_pose(prim_path)
    return float(position[2])


def _summarise(results: list[dict], emit) -> None:
    attempted = [r for r in results if not r.get("skipped")]
    lifted = [r for r in attempted if r.get("lifted")]
    perceived = [r for r in results if r.get("perceived")]
    skipped = [r for r in results if r.get("skipped")]
    false_success = [r for r in results if r.get("skill_ok") and not r.get("lifted")]

    emit("=" * 72)
    emit(f"perceived : {len(perceived)}/{len(results)}")
    emit(f"lifted    : {len(lifted)}/{len(attempted)} attempted"
         + (f"  ({len(skipped)} skipped as non-graspable)" if skipped else ""))
    if false_success:
        emit(f"REPORTED SUCCESS WITHOUT LIFTING: {[r['object'] for r in false_success]}")
    failed = [r["object"] for r in attempted if not r.get("lifted")]
    if failed:
        emit(f"failed    : {failed}")


if __name__ == "__main__":
    raise SystemExit(main())

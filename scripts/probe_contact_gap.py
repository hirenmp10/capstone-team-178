"""Measure where the visible gripper-to-object gap actually comes from.

    ..\\python.bat manipulation_framework\\scripts\\probe_contact_gap.py

Two candidate causes, and tightening the wrong one is what makes the gap look
immovable:

1. **Contact offsets.** ``apply_mesh_colliders`` authors ``contact_offset`` and
   ``rest_offset`` on *scene objects*. Nothing in ``mfw/robot/`` authors them on
   the robot, so the Franka's fingers keep whatever ``franka.usd`` shipped.
   PhysX separates a pair at ``restOffset_A + restOffset_B``, so an untouched
   finger sets the floor no matter how far the object side is lowered.

2. **Collider vs visual mesh.** A collider larger than the mesh it approximates
   holds the *rendered* surfaces apart while the colliders themselves touch. No
   offset change can close that.

This prints both, for the fingers and for one target object, so the next edit is
aimed at whichever one is actually responsible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: PhysX's own defaults, used when a collider authors no opinion. An attribute
#: that is absent is not zero -- reporting it as zero would hide the cause.
PHYSX_DEFAULT_CONTACT_OFFSET = 0.02
PHYSX_DEFAULT_REST_OFFSET = 0.0


def _offsets(stage, prim_path: str, emit) -> None:
    """Report authored contact/rest offsets on every collider under a prim."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        emit(f"  {prim_path}: MISSING")
        return

    stack, found = [root], 0
    while stack:
        prim = stack.pop()
        stack.extend(prim.GetAllChildren())
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        found += 1
        api = PhysxSchema.PhysxCollisionAPI(prim)
        c_attr = api.GetContactOffsetAttr()
        r_attr = api.GetRestOffsetAttr()

        def _read(attr, default):
            if not attr or not attr.HasAuthoredValue():
                return default, "PhysX default"
            value = attr.Get()
            if value is None or float(value) < 0.0:
                return default, "authored -1 (means: use default)"
            return float(value), "authored"

        contact, c_src = _read(c_attr, PHYSX_DEFAULT_CONTACT_OFFSET)
        rest, r_src = _read(r_attr, PHYSX_DEFAULT_REST_OFFSET)

        approx = ""
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            a = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            if a and a.Get():
                approx = f"  approx={a.Get()}"

        emit(
            f"  {prim.GetPath()}\n"
            f"      contact {contact * 1000:7.3f} mm ({c_src})\n"
            f"      rest    {rest * 1000:7.3f} mm ({r_src}){approx}"
        )
    if found == 0:
        emit(f"  {prim_path}: no colliders found beneath it")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--target", default="marker")
    args = parser.parse_args()

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    from mfw.assistant import Assistant  # noqa: PLC0415

    with Assistant(config_path=args.config) as robot:
        stage = robot.runtime.sim.world.stage
        cfg = robot.runtime.config

        emit("=" * 68)
        emit("CONFIGURED (applied to scene objects only)")
        emit("=" * 68)
        emit(f"  physics.contact_offset : {cfg.physics.contact_offset * 1000:.3f} mm")
        emit(f"  physics.rest_offset    : {cfg.physics.rest_offset * 1000:.3f} mm")
        emit("")

        emit("=" * 68)
        emit("ROBOT FINGERS  <-- nothing in mfw/robot/ authors offsets here")
        emit("=" * 68)
        for finger in (cfg.robot.left_finger_prim, cfg.robot.right_finger_prim):
            emit(f"{finger}:")
            _offsets(stage, f"{cfg.robot.prim_path}/{finger}", emit)
        emit("panda_hand:")
        _offsets(stage, f"{cfg.robot.prim_path}/{cfg.robot.tcp_parent_prim}", emit)
        emit("")

        emit("=" * 68)
        emit(f"TARGET OBJECT: {args.target}")
        emit("=" * 68)
        scene = robot.runtime.vision.observe()
        target = next((o for o in scene.objects.values() if o.label == args.target), None)
        if target is None:
            emit(f"  {args.target!r} not visible; objects seen: "
                 f"{sorted(o.label for o in scene.objects.values())}")
        else:
            emit(f"  extents: {[round(float(v) * 1000, 1) for v in target.bbox.extents]} mm")
            for spec in cfg.scene.objects:
                if spec.name == args.target:
                    _offsets(stage, f"/World/{spec.name}", emit)
                    break

        emit("")
        emit("=" * 68)
        emit("WORST-CASE RENDERED SEPARATION")
        emit("=" * 68)
        emit("  PhysX separates a contacting pair at restOffset_A + restOffset_B.")
        emit("  Read the two rest values above and add them: that is the floor")
        emit("  the fingers can reach, before any collider-vs-visual mismatch.")

    out = _ROOT / "renders" / "contact_gap_probe.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

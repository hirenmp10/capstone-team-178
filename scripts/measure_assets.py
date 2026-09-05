"""Measure the true dimensions of every asset in the catalogue.

    ..\\python.bat manipulation_framework\\scripts\\measure_assets.py

This is where ``size_m`` in ``configs/assets.yaml`` comes from. Those values are
measured from the actual USD, not copied from the YCB spec sheet, because the
two disagree in ways that matter:

* The published figure is often a nominal cross-section. The YCB banana is
  listed at 36 mm wide; the scanned mesh's bounding box is 74 mm, because a
  banana is curved and the box has to contain the curve.
* The shipped asset is not always the same variant as the published entry. The
  large clamp measures 165 x 122 mm against a published 165 x 213 mm.

Extents are reported in the **upright** frame -- after each asset's
``upright_quat`` is applied -- because the YCB set is authored Y-up and Isaac's
world is Z-up. Recording raw bounds would leave every consumer to remember the
swap, and the ones that forgot would be silently wrong about which dimension the
gripper has to span.

Pass ``--check`` to compare the catalogue against the meshes and exit non-zero
on a mismatch, which is how a stale ``size_m`` gets caught.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="", help="catalogue path (default: configs/assets.yaml)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a catalogue size_m disagrees with the mesh",
    )
    parser.add_argument("--tolerance", type=float, default=0.10, help="relative tolerance for --check")
    parser.add_argument(
        "--log",
        default=str(_ROOT / "logs" / "asset_measurements.txt"),
        help="write the table here as well as to stdout",
    )
    args = parser.parse_args()

    # SimulationApp parses sys.argv and dies on flags it does not recognise.
    sys.argv = [sys.argv[0]]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        return _measure(args)
    finally:
        app.close()


def _measure(args) -> int:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Usd, UsdGeom

    from mfw.simulation.asset_registry import AssetRegistry

    registry = AssetRegistry.load(args.assets or None)
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("Isaac assets root unreachable", flush=True)
        return 2

    # A World rather than an in-memory stage: add_reference_to_stage targets
    # the *current* stage, and only a World reliably establishes one.
    world = World()
    stage = world.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    lines: list[str] = []

    def emit(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    emit(f"{'asset':20} {'measured upright (mm)':28} {'catalogue (mm)':28} {'worst':>7}")
    emit("-" * 88)

    failures: list[str] = []
    for name, spec in sorted(registry.objects.items()):
        prim_path = f"/Measure/{name}"
        add_reference_to_stage(usd_path=assets_root + spec.usd, prim_path=prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            emit(f"{name:20} FAILED to resolve {spec.usd}")
            failures.append(name)
            continue

        # Apply the asset's upright correction before measuring, so the numbers
        # are in the frame the scene will actually spawn it in.
        quat = spec.upright_quat
        rotation = Gf.Matrix4d().SetRotate(
            Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])))
        )
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.MakeMatrixXform().Set(rotation)

        cache.Clear()
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            emit(f"{name:20} EMPTY BOUNDS -- the reference resolved but has no geometry")
            failures.append(name)
            continue

        measured = np.array([box.GetMax()[i] - box.GetMin()[i] for i in range(3)])
        catalogue = np.asarray(spec.size_m, dtype=np.float64)
        worst = float(np.max(np.abs(np.sort(measured) / np.sort(catalogue) - 1.0)))

        flag = ""
        if worst > args.tolerance:
            flag = "  <-- MISMATCH"
            failures.append(name)

        emit(
            f"{name:20} {str(np.round(measured * 1000, 1).tolist()):28} "
            f"{str(np.round(catalogue * 1000, 1).tolist()):28} {worst * 100:6.0f}%{flag}"
        )

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWritten to {log_path}", flush=True)

    if args.check and failures:
        print(
            f"\n{len(failures)} asset(s) disagree with the catalogue by more than "
            f"{args.tolerance * 100:.0f}%: {sorted(failures)}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

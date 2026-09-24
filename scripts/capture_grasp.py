"""Photograph the gripper while it is actually holding something.

    ..\\python.bat manipulation_framework\\scripts\\capture_grasp.py --target marker

Writes RGB / depth / segmentation PNGs from both cameras *mid-grasp*, which is
the only way to judge the finger-to-object gap. ``render_cameras.py`` renders a
settled scene with the arm at home and can never show it.

Why this exists: a grasp that lifts the object 115 mm is unambiguously real, and
can still look fake, because PhysX holds surfaces apart by ``rest_offset`` and
starts generating contacts at ``contact_offset``. Those are millimetres, and
whether that reads as "gripped" or "floating" depends on the object's size. The
argument is not settleable from logs, so this takes the picture instead.
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
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--target", default="marker", help="object label to grasp")
    parser.add_argument("--out", default="renders/grasp", help="output directory")
    parser.add_argument(
        "--lift",
        default="move up 10 cm",
        help="command run after the pick so the object is clear of the table",
    )
    args = parser.parse_args()

    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from mfw.assistant import Assistant  # noqa: PLC0415 - must follow sys.path setup
    from render_cameras import _write_frame  # noqa: PLC0415

    log_lines: list[str] = []

    def emit(text: str) -> None:
        print(text, flush=True)
        log_lines.append(text)

    with Assistant(config_path=args.config) as robot:
        emit(robot.command("what do you see").message)

        picked = robot.command(f"pick up the {args.target}")
        emit(f"pick: {picked.message}")
        if not picked.ok:
            emit("nothing is being held; a photograph of an empty gripper proves nothing")
            return 1

        lifted = robot.command(args.lift)
        emit(f"lift: {lifted.message}")

        # Let the renderer converge before capturing. RTX needs frames to
        # resolve, and a shot taken immediately comes back black -- the same
        # warm-up that makes the first observe() of every run return 0 objects.
        robot.runtime.sim.render_step(12)

        width = robot.runtime.robot.get_gripper_width()
        emit(f"gripper width while holding: {width * 1000:.1f} mm")

        scene = robot.runtime.vision.observe()
        held = next(
            (o for o in scene.objects.values() if o.label == args.target), None
        )
        if held is not None:
            extents = [round(float(v) * 1000, 1) for v in held.bbox.extents]
            emit(f"{args.target} measured while held: {extents} mm")
            # The gripped axis is the one the fingers span, which is the
            # extent closest to the finger separation -- not the smallest.
            # Using min() here reported 13.9 mm of gap on a banana the fingers
            # were touching, because it measured the 14.5 mm thickness while
            # the jaws were closed on the 41.5 mm width.
            gripped = min(held.bbox.extents, key=lambda e: abs(float(e) - width))
            emit(
                f"  gripped axis {float(gripped) * 1000:.1f} mm -> gap per finger "
                f"{(width - float(gripped)) * 1000 / 2:.2f} mm"
            )

        written: list[Path] = []
        for name, camera in robot.runtime.cameras.items():
            frame = camera.capture()
            written += _write_frame(frame, f"holding_{args.target}_{name}", out_dir, emit)

        emit("")
        emit(f"{len(written)} image(s) written to {out_dir}")

    (out_dir / "grasp_report.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

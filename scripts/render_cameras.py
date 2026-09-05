"""Render what the robot's cameras actually see.

    ..\\python.bat manipulation_framework\\scripts\\render_cameras.py \\
        --config manipulation_framework/configs/benchmark.yaml --out renders/

Writes RGB, depth and instance-segmentation PNGs for every configured camera.

The GUI's perspective viewport is an *authoring* view -- wide, well-placed, and
nothing like the robot's input. Perception runs off these two cameras at
640x480, and their limits are what actually decide whether an object is
detectable: the 19 mm marker clears the 60-point minimum only because it sits in
the wrist camera's coverage. Looking at the perspective view and concluding "the
scene is fine" is how you end up debugging a detector that was never the problem.

The segmentation render doubles as ground truth for the label pipeline: an
object that is invisible here cannot be perceived, no matter what the physics
says about it.
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
    parser.add_argument("--config", default=str(_ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--out", default=str(_ROOT / "renders"))
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settle-steps", type=int, default=120, help="physics steps before rendering"
    )
    args = parser.parse_args()

    # SimulationApp parses sys.argv and dies on flags it does not recognise.
    sys.argv = [sys.argv[0]]

    from mfw.config.schema import load_config

    overrides: dict = {"simulation": {"headless": True}, "logging": {"console": False}}
    if args.randomize:
        overrides["scene"] = {"randomization": {"enabled": True, "seed": args.seed}}

    config = load_config(args.config, overrides=overrides)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def emit(text: str) -> None:
        print(text, flush=True)
        log_lines.append(text)

    from mfw.simulation.runtime import Runtime

    runtime = Runtime(config)
    try:
        runtime.build()
        runtime.sim.step(args.settle_steps)

        # Two observations: a track is only confirmed on its second sighting
        # (Track.is_confirmed requires hits >= 2), so a single observe() always
        # yields an empty scene graph.
        runtime.vision.observe()
        runtime.sim.step(5)
        scene = runtime.vision.observe()

        emit(f"perception confirmed {len(scene.objects)} objects")
        for obj in sorted(scene.objects.values(), key=lambda o: o.pose.position[0]):
            extents = [round(float(v) * 1000, 1) for v in obj.bbox.extents]
            emit(
                f"  {obj.label:10} conf={obj.confidence:.2f} pts={obj.num_points:5} "
                f"pos={[round(float(v), 3) for v in obj.pose.position]} size_mm={extents}"
            )

        written: list[Path] = []
        for name, camera in runtime.cameras.items():
            frame = camera.capture()
            written += _write_frame(frame, name, out_dir, emit)

        emit("")
        emit(f"{len(written)} image(s) written to {out_dir}")
        (out_dir / "render_report.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return 0
    finally:
        runtime.close()


def _write_frame(frame, name: str, out_dir: Path, emit) -> list[Path]:
    """Write RGB, depth and segmentation PNGs for one camera frame."""
    import numpy as np

    written: list[Path] = []

    if frame.rgb is not None:
        path = out_dir / f"{name}_rgb.png"
        _save_png(np.asarray(frame.rgb, dtype=np.uint8), path)
        written.append(path)
        emit(f"{name}: rgb {frame.rgb.shape} -> {path.name}")

    if frame.depth is not None:
        depth = np.asarray(frame.depth, dtype=np.float32)
        finite = np.isfinite(depth)
        if finite.any():
            near, far = float(depth[finite].min()), float(depth[finite].max())
            # Near = white, far = dark, so the objects of interest are the
            # bright ones. Misses render black rather than as a huge value.
            normalised = np.zeros(depth.shape, dtype=np.float32)
            span = max(far - near, 1e-6)
            normalised[finite] = 1.0 - (depth[finite] - near) / span
            grey = (normalised * 255.0).astype(np.uint8)
            path = out_dir / f"{name}_depth.png"
            _save_png(np.dstack([grey, grey, grey]), path)
            written.append(path)
            emit(f"{name}: depth {near:.3f}-{far:.3f} m -> {path.name}")

    if frame.segmentation is not None:
        seg = np.asarray(frame.segmentation, dtype=np.int32)
        path = out_dir / f"{name}_segmentation.png"
        _save_png(_colourise_segmentation(seg), path)
        written.append(path)
        labels = {
            sid: frame.seg_id_to_label.get(sid, "?")
            for sid in np.unique(seg).tolist()
        }
        visible = {k: v for k, v in labels.items() if str(v) not in ("BACKGROUND", "UNLABELLED")}
        emit(f"{name}: segmentation -> {path.name}  ({len(visible)} labelled regions)")

    return written


def _colourise_segmentation(seg):
    """Give each instance id a distinct, stable colour.

    Hash-derived rather than a palette lookup: the number of instances is not
    known ahead of time, and a wrapping palette would give two adjacent objects
    the same colour, which is precisely the confusion the render exists to rule
    out.
    """
    import numpy as np

    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for seg_id in np.unique(seg):
        if seg_id <= 1:
            continue  # 0 = BACKGROUND, 1 = UNLABELLED: leave black
        rng = np.random.default_rng(int(seg_id) * 9781 + 17)
        # Bias away from black so every region reads against the background.
        colour = (rng.integers(60, 256, size=3)).astype(np.uint8)
        out[seg == seg_id] = colour
    return out


def _save_png(rgb, path: Path) -> None:
    """Write an HxWx3 uint8 array as a PNG.

    PIL is what Isaac's own replicator writers use, so it is present; the
    zlib/struct path is a dependency-free fallback rather than a rewrite of PNG
    encoding for its own sake.
    """
    try:
        from PIL import Image  # noqa: PLC0415

        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        pass

    import struct  # noqa: PLC0415
    import zlib  # noqa: PLC0415

    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


if __name__ == "__main__":
    raise SystemExit(main())

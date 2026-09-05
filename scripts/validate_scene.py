"""Build a scene and prove every object in it is usable.

    ..\\python.bat manipulation_framework\\scripts\\validate_scene.py \\
        --config manipulation_framework/configs/benchmark.yaml

Exits non-zero if any check fails, so this can gate a run: there is no point
launching a two-hour data collection on a scene where the drill has no collider.

Add ``--gui`` to watch it, ``--randomize --seed N`` to validate a randomised
layout, and ``--report out.json`` to keep the machine-readable result.
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


def _emit(text: str, path: Path | None) -> None:
    """Write a line to stdout and, if given, to a text file.

    Isaac routes Python's ``stdout`` through ``omni.kit.app`` at info level,
    which the default log configuration discards. A ``print`` that works
    interactively therefore vanishes when the run is redirected -- so anything
    that matters goes to a real file as well.
    """
    print(text, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--gui", action="store_true", help="run with the Isaac Sim viewport")
    parser.add_argument("--randomize", action="store_true", help="randomise object placement")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settle-steps", type=int, default=120, help="physics steps before measuring"
    )
    parser.add_argument(
        "--observe-steps", type=int, default=120, help="steps over which drift is measured"
    )
    parser.add_argument("--report", default="", help="write the JSON report here")
    parser.add_argument(
        "--log",
        default=str(_ROOT / "logs" / "scene_validation.txt"),
        help="write the human-readable report here (stdout alone is unreliable under Isaac)",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=0,
        help="with --gui, keep the viewport open this many wall-clock seconds",
    )
    args = parser.parse_args()

    # SimulationApp parses sys.argv and dies on flags it does not recognise, so
    # the CLI is cleared before it starts. This is the same trap that killed the
    # Isaac test runner; see scripts/run_isaac_tests.py.
    sys.argv = [sys.argv[0]]

    from mfw.config.schema import load_config

    overrides: dict = {}
    if args.randomize:
        overrides = {"scene": {"randomization": {"enabled": True, "seed": args.seed}}}
    if args.gui:
        overrides.setdefault("simulation", {})["headless"] = False

    log_path: Path | None = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    config = load_config(args.config, overrides=overrides)
    _emit(f"config: {args.config}", log_path)
    _emit(f"scene:  environment={config.scene.environment or 'none'} "
          f"objects={len(config.scene.objects)} furniture={len(config.scene.furniture)}", log_path)

    from mfw.simulation.runtime import Runtime
    from mfw.simulation.validation import SceneValidator

    # Built through Runtime rather than SceneBuilder alone, so the scene is
    # validated exactly as the framework assembles it -- and so the cameras
    # exist, which the exposure check needs. A scene can be geometrically
    # perfect and still render black.
    _emit("starting simulator and building scene...", log_path)
    runtime = Runtime(config)
    try:
        runtime.build()
        builder = runtime.scene_builder
        _emit(
            f"built: {len(builder.object_prim_paths)} objects, "
            f"{len(builder.furniture_prim_paths)} furniture, "
            f"{len(runtime.cameras)} cameras",
            log_path,
        )

        validator = SceneValidator(
            runtime.sim,
            builder,
            registry=runtime.asset_registry,
            cameras=runtime.cameras,
            approach_offset=config.grasp.approach_offset,
        )
        report = validator.validate(
            settle_steps=args.settle_steps, observe_steps=args.observe_steps
        )
        sim = runtime.sim

        _emit("\n" + report.summary(), log_path)

        if args.report:
            Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            _emit(f"\nJSON report written to {args.report}", log_path)
        if log_path is not None:
            print(f"Report also written to {log_path}", flush=True)

        if args.gui and args.hold > 0:
            # Wall-clock, not simulated time. Stepping `hold / physics_dt` times
            # is the obvious reading and it is wrong by whatever the render rate
            # happens to be: at 120 Hz physics with rendering on, a requested
            # 900 s held the window open for 37 real minutes. Someone asking to
            # keep a viewport open means seconds they can wait through.
            deadline = time.monotonic() + args.hold
            print(f"\nHolding the viewport for {args.hold}s (Ctrl-C to exit early)...", flush=True)
            try:
                while time.monotonic() < deadline:
                    sim.step(10)
            except KeyboardInterrupt:
                print("Closing.", flush=True)

        return 0 if report.passed else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

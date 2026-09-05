"""Shared test fixtures.

The ``runtime`` fixture is **session-scoped and shared by every Isaac test
module**. Isaac Sim's runtime is a process-wide singleton: a second
``SimulationApp`` corrupts the first rather than failing cleanly, so per-module
fixtures would break as soon as a second phase suite existed.

Because the simulator is shared, tests mutate common state. ``restore_scene``
returns the robot home and re-settles the objects between tests so ordering
cannot silently couple them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfw.config.schema import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def config():
    """Framework config used by all Isaac tests."""
    return load_config(
        DEFAULT_CONFIG,
        overrides={"simulation": {"headless": True}, "logging": {"console": False}},
    )


@pytest.fixture(scope="session")
def runtime(config):
    """One Isaac Sim instance for the entire session."""
    from mfw.simulation.runtime import Runtime

    rt = Runtime(config)
    rt.build()
    yield rt
    rt.close()


@pytest.fixture
def restore_scene(runtime):
    """Reset the robot *and the objects* before a test that needs a clean start.

    Resetting the objects matters once picks are involved: a successful pick moves
    things, and without restoration each later test inherits a scene that previous
    tests rearranged -- which shows up as mysterious, order-dependent failures
    when an object has been carried somewhere unreachable.

    This writes object poses directly. That is legitimate **test setup** and is
    the one place it happens; the framework itself never writes an object pose,
    because doing so would fake the manipulation it is supposed to demonstrate.

    Requested explicitly rather than autouse: settling costs ~1 s and read-only
    assertions do not need it.
    """
    import numpy as np

    runtime.robot.go_home_immediate()
    runtime.robot.open_gripper()

    scene = runtime.sim.world.scene
    for spec in runtime.config.scene.objects:
        obj = scene.get_object(spec.name)
        if obj is None:
            continue
        obj.set_world_pose(
            position=np.array(spec.position, dtype=float),
            orientation=np.array(spec.quat, dtype=float),
        )
        # Zero the velocities too: a body restored mid-flight keeps its momentum
        # and immediately slides away from where it was just put.
        if hasattr(obj, "set_linear_velocity"):
            obj.set_linear_velocity(np.zeros(3))
        if hasattr(obj, "set_angular_velocity"):
            obj.set_angular_velocity(np.zeros(3))

    runtime.sim.settle()
    runtime.sim.render_step(2)
    return runtime


@pytest.fixture
def vision(runtime, config):
    """A VisionManager wired to the runtime's cameras."""
    from mfw.vision.manager import VisionManager

    return VisionManager(
        sim=runtime.sim,
        cameras=runtime.cameras,
        config=config.perception,
        event_logger=runtime.events,
    )


def ground_truth_pose(runtime, object_name: str):
    """Ground-truth world position of a spawned object.

    Legitimate in tests, which need something to check perception *against*.
    Runtime code has no accessor that returns this.
    """
    import numpy as np
    from isaacsim.core.utils.xforms import get_world_pose

    path = runtime.scene_builder.object_prim_paths[object_name]
    position, quat = get_world_pose(path)
    return np.asarray(position, dtype=float), np.asarray(quat, dtype=float)

"""Isaac Sim application bootstrap.

This is the *only* module allowed to construct a ``SimulationApp``, and it must
run before any ``omni.*`` or ``isaacsim.*`` import anywhere in the process.
Isaac Sim loads its extension plugins during ``SimulationApp.__init__``; an
earlier Isaac import binds against plugins that do not exist yet and fails in
ways that look unrelated to the real cause.

Every other Isaac-touching module in this framework therefore imports Isaac
*inside functions*, never at module scope, so that importing e.g.
``mfw.robot.franka`` for a docstring or a type never detonates.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any

from mfw.config.schema import SimulationConfig
from mfw.core.errors import SimulationError
from mfw.utils.logging import get_logger

__all__ = ["SimulationContext", "get_simulation_app", "is_app_running"]

_log = get_logger("simulation.app")

_APP: Any = None
_LOCK = threading.Lock()


def is_app_running() -> bool:
    """Whether a SimulationApp exists in this process."""
    return _APP is not None


def get_simulation_app() -> Any:
    """Return the live SimulationApp, or raise if bootstrap has not run."""
    if _APP is None:
        raise SimulationError(
            "SimulationApp has not been created. Construct a SimulationContext "
            "before importing or using any Isaac Sim functionality."
        )
    return _APP


#: Kit channels that emit a warning per simulation step. They are not reporting
#: a fault -- the simulation manager logs one line every time a transform is
#: queried at a timestamp it has no surrounding samples for, which is every step
#: once a run is long enough -- but at two lines per step they bury every real
#: message. A 500-second run produced tens of thousands of them, and the pick
#: outcomes they were hiding are the only reason anyone reads this log.
_NOISY_CHANNELS = ("isaacsim.core.simulation_manager.plugin",)


def _quieten_noisy_channels() -> None:
    """Raise the log threshold on channels that spam once per step.

    Best effort by design. The channel-level API has moved between Kit
    releases, so every attempt is guarded: failing to silence a warning is a
    cosmetic problem, while an exception here would take down a simulator that
    was otherwise fine.
    """
    try:
        import omni.log  # noqa: PLC0415 - only exists once Kit has started

        log = omni.log.get_log()
        for channel in _NOISY_CHANNELS:
            log.set_channel_level(channel, omni.log.Level.ERROR)
        _log.debug("Quietened %d noisy Kit log channel(s)", len(_NOISY_CHANNELS))
        return
    except Exception as exc:  # pragma: no cover - depends on the Kit build
        _log.debug("omni.log channel filter unavailable (%s); trying carb", exc)

    try:
        import carb.settings  # noqa: PLC0415

        settings = carb.settings.get_settings()
        for channel in _NOISY_CHANNELS:
            settings.set(f"/log/channels/{channel}/level", "Error")
    except Exception as exc:  # pragma: no cover
        _log.debug("Could not quieten Kit log channels: %s", exc)


class SimulationContext:
    """Owns the SimulationApp and the physics ``World``.

    Only one may exist per process: Isaac Sim's runtime is a process-wide
    singleton, so a second instance corrupts the first rather than failing
    cleanly. That is also why Isaac-dependent tests are marked ``isaac`` and run
    serially.

    Use as a context manager so the app is always shut down, including on error::

        with SimulationContext(cfg) as sim:
            sim.reset()
            sim.step(60)
    """

    def __init__(self, config: SimulationConfig, extra_app_kwargs: dict[str, Any] | None = None) -> None:
        global _APP

        with _LOCK:
            if _APP is not None:
                raise SimulationError(
                    "A SimulationApp already exists in this process. Isaac Sim's runtime "
                    "is a process-wide singleton; reuse the existing SimulationContext."
                )

            config.validate()
            self.config = config

            from isaacsim import SimulationApp  # noqa: PLC0415 - must follow config validation

            app_kwargs: dict[str, Any] = {
                "headless": config.headless,
                "renderer": config.renderer,
            }
            if extra_app_kwargs:
                app_kwargs.update(extra_app_kwargs)

            _log.info("Starting Isaac Sim (headless=%s, renderer=%s)", config.headless, config.renderer)
            _APP = SimulationApp(app_kwargs)
            self._app = _APP
            _quieten_noisy_channels()

        atexit.register(self._safe_close)

        # Safe only now that the app exists and its extensions are loaded.
        from isaacsim.core.api import World  # noqa: PLC0415

        self._world = World(
            physics_dt=config.physics_dt,
            rendering_dt=config.rendering_dt,
            stage_units_in_meters=config.stage_units_in_meters,
        )
        self._closed = False
        self._step_index = 0
        _log.info(
            "World ready (physics_dt=%.6f, rendering_dt=%.6f)", config.physics_dt, config.rendering_dt
        )

    @property
    def app(self) -> Any:
        return self._app

    @property
    def world(self) -> Any:
        """The Isaac ``World``. Scene construction and stepping go through it."""
        return self._world

    @property
    def step_index(self) -> int:
        """Physics steps executed since construction. Stamped onto every observation."""
        return self._step_index

    @property
    def sim_time(self) -> float:
        """Simulated seconds elapsed.

        Derived from the step count rather than wall time so that logs replay
        identically regardless of how fast the machine ran.
        """
        return self._step_index * self.config.physics_dt

    def reset(self) -> None:
        """Reset the world and initialise physics handles.

        Must be called after all prims are added and before reading any
        articulation state: Isaac only creates the physics views on reset, so
        joint queries before this return empty or stale data.
        """
        self._world.reset()
        self._step_index = 0

    def step(self, count: int = 1, render: bool | None = None) -> None:
        """Advance physics ``count`` steps.

        Rendering defaults to on when not headless. Stepping without rendering
        is much faster and is what the settle loops use, but cameras will not
        produce new frames, so any step that precedes a capture must render.
        """
        do_render = (not self.config.headless) if render is None else render
        for _ in range(count):
            self._world.step(render=do_render)
            self._step_index += 1

    def settle(self, steps: int | None = None) -> None:
        """Run physics until the scene is at rest.

        Objects spawn with a small gap above their support and must fall and
        stop before perception runs; measuring a still-falling object yields a
        pose that is wrong by the time the arm arrives.
        """
        # Rendering follows the headless flag, so objects are visibly seen to fall
        # and come to rest in the GUI rather than the window freezing.
        self.step(steps if steps is not None else self.config.settle_steps)

    def render_step(self, count: int = 1) -> None:
        """Advance with rendering forced on, to refresh camera products."""
        self.step(count, render=True)

    def close(self) -> None:
        global _APP
        if self._closed:
            return
        self._closed = True
        _log.info("Shutting down Isaac Sim after %d steps", self._step_index)
        try:
            self._app.close()
        finally:
            with _LOCK:
                _APP = None

    def _safe_close(self) -> None:  # pragma: no cover - atexit path
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "SimulationContext":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

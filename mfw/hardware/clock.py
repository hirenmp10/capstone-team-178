"""Wall-clock stand-in for the simulator's stepping seam.

Pure stdlib. This module must never import Isaac Sim, torch or transformers.

Outside :mod:`mfw.simulation` the framework touches the simulator through a
tiny surface (audited by grepping ``sim.`` in the skills, the joint controller
and the vision manager): ``step``, ``render_step``, ``settle``, ``reset``,
``close``, ``step_index``, ``sim_time`` and ``config.physics_dt``. Every one
of those is really a statement about *time* -- "let the arm arrive", "let
the scene settle", "how stale is this observation" -- so on hardware the
seam is satisfied by a clock.

Two modes:

* **real** (default): ``step(n)`` sleeps ``n * physics_dt`` seconds. That is
  what turns ``settle_steps_after_motion`` into servo settling time and makes
  a ``Wait`` skill actually wait. ``sim_time`` is wall time since construction
  (or the last ``reset``), *not* steps times dt: on hardware the world keeps
  moving while the laptop is blocked in a detector round trip, and the
  freshness check in :meth:`mfw.vision.manager.PerceptionManager.require_fresh_scene`
  must see that time pass even though nobody stepped.
* **fast** (``hardware.fake_clock: true``): nothing sleeps and time is
  virtual, ``sim_time = steps * physics_dt``. Deterministic and instant, which
  is what the in-process end-to-end test wants.

``render_step`` is a no-op in both modes: the hardware camera is live, so
there is no render product to refresh, and sleeping there would only slow
every observe. Rendering never existed on this lane, so it is not mimicked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mfw.core.errors import ConfigurationError
from mfw.utils.logging import get_logger

__all__ = ["WallClock", "WallClockConfig"]

_log = get_logger("hardware.clock")


@dataclass(frozen=True)
class WallClockConfig:
    """The two numbers the rest of the framework reads off ``sim.config``.

    ``physics_dt`` is the duration of one ``step``; the joint controller turns
    ``motion.interpolation_dt`` into a step count with it. ``settle_steps`` is
    what :meth:`WallClock.settle` sleeps for. ``headless`` is carried only so
    code that branches on it keeps working; there is nothing to render.
    """

    physics_dt: float
    settle_steps: int
    headless: bool = True

    def validate(self) -> None:
        if not self.physics_dt > 0.0:
            raise ConfigurationError(f"WallClock physics_dt must be > 0, got {self.physics_dt}")
        if self.settle_steps < 0:
            raise ConfigurationError(f"WallClock settle_steps must be >= 0, got {self.settle_steps}")


class WallClock:
    """Duck-types the stepping half of ``mfw.simulation.app.Simulation``.

    Not an ``ISimulation`` subclass: the hardware lane has no world, stage or
    app, and pretending otherwise would only move the failure from import
    time to the first attribute access.
    """

    def __init__(self, physics_dt: float, settle_steps: int, fast: bool = False) -> None:
        self.config = WallClockConfig(physics_dt=float(physics_dt), settle_steps=int(settle_steps))
        self.config.validate()
        self.fast = bool(fast)
        self._steps = 0
        self._t0 = time.monotonic()
        self._closed = False
        _log.debug(
            "WallClock ready (physics_dt=%.4f, settle_steps=%d, fast=%s)",
            self.config.physics_dt, self.config.settle_steps, self.fast,
        )

    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        """Seconds elapsed: wall time since construction/reset, or virtual when fast."""
        if self.fast:
            return self._steps * self.config.physics_dt
        return time.monotonic() - self._t0

    @property
    def step_index(self) -> int:
        """``int(sim_time / physics_dt)``: the step the current instant belongs to.

        In fast mode this is exactly the number of steps taken. In real mode it
        keeps counting while nobody steps, which is the behaviour the
        observation-age bookkeeping wants.
        """
        if self.fast:
            return self._steps
        return int(self.sim_time / self.config.physics_dt)

    def reset(self) -> None:
        """Re-zero the clock. On hardware nothing else needs resetting."""
        self._steps = 0
        self._t0 = time.monotonic()

    def step(self, count: int = 1, render: bool | None = None) -> None:
        """Advance ``count`` steps: sleep ``count * physics_dt`` (real) or just count (fast).

        ``render`` is accepted for signature compatibility and ignored.
        """
        n = int(count)
        if n <= 0:
            return
        self._steps += n
        if not self.fast:
            time.sleep(n * self.config.physics_dt)

    def settle(self, steps: int | None = None) -> None:
        """Wait ``settle_steps`` (or ``steps``) for the servos to physically arrive."""
        self.step(self.config.settle_steps if steps is None else int(steps))

    def render_step(self, count: int = 1) -> None:
        """No-op: there is no renderer and the camera is live."""
        return None

    def close(self) -> None:
        """No-op apart from a log line; kept so shutdown code is backend-agnostic."""
        if self._closed:
            return
        self._closed = True
        _log.debug("WallClock closed after %d steps (%.2f s)", self._steps, self.sim_time)

"""Skill infrastructure: the execution context and the atomic-skill base class.

Pure stdlib + NumPy at module scope; the concrete skills reach Isaac only through
the injected components. This module must never import Isaac Sim.

**The atomicity rule.** A skill does exactly what it was asked and then returns.
It never invokes another skill. ``Pick`` picks and holds; it does not place.
``Place`` places; it does not go home. Composition is the planner's job, and the
planner only ever composes what the user actually asked for.

That rule is enforced structurally: a skill receives a :class:`SkillContext`
holding perception, planning, control and memory -- but *not* the registry, so it
has no way to reach another skill even if its implementation wanted to.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any

from mfw.config.schema import FrameworkConfig
from mfw.core.errors import MfwError, SafetyViolation
from mfw.core.interfaces import ISkill
from mfw.core.types import SkillResult, SkillStatus
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["SkillContext", "Skill"]

_log = get_logger("skills")


@dataclass
class SkillContext:
    """Everything a skill is allowed to touch.

    Deliberately excludes the skill registry: without a handle on other skills,
    a skill *cannot* chain into one, so atomicity is a property of the wiring
    rather than of everyone remembering the rule.
    """

    sim: Any
    robot: Any
    vision: Any
    planner: Any
    controller: Any
    grasp_scorer: Any
    memory: Any
    config: FrameworkConfig
    events: EventLogger | None = None
    support_height: float = 0.0
    grasp_generator: Any = None
    """Optional :class:`~mfw.core.interfaces.IGraspGenerator`. ``None`` keeps
    the sim lane's direct OBB synthesis; the hardware lane injects a top-down
    generator because its depthless perception has no real box to enumerate."""

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(event, payload)


class Skill(ISkill):
    """Base class handling timing, logging, validation and error translation.

    Subclasses implement :meth:`_run` and return a :class:`SkillResult`. Anything
    raised is converted into a failed result rather than propagating, so a single
    bad command can never take down the session -- except
    :class:`SafetyViolation`, which is re-raised because a breached safety limit
    must not be swallowed and silently retried.
    """

    #: Skill name used by the planner, the registry and the logs.
    skill_name: str = "unnamed"

    def __init__(self, context: SkillContext) -> None:
        self.ctx = context

    @property
    def name(self) -> str:
        return self.skill_name

    def validate(self, params: dict[str, Any]) -> str | None:
        """Return an error string if the parameters are unusable, else ``None``.

        Runs before any motion, so an impossible command costs nothing and the
        user gets an immediate, specific reason.
        """
        return None

    def execute(self, params: dict[str, Any]) -> SkillResult:
        """Validate, run, and return exactly one result. Then the robot WAITS."""
        started = time.perf_counter()
        params = dict(params or {})

        error = self.validate(params)
        if error is not None:
            result = SkillResult(
                skill_name=self.name,
                status=SkillStatus.INFEASIBLE,
                message=error,
                duration_s=time.perf_counter() - started,
            )
            self._finish(result, params)
            return result

        try:
            result = self._run(params)
        except SafetyViolation:
            # Never swallowed: a safety breach must reach the operator.
            self.ctx.emit("skill.safety_violation", {"skill": self.name, "params": params})
            raise
        except MfwError as exc:
            result = SkillResult(
                skill_name=self.name,
                status=SkillStatus.FAILED,
                message=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - started,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("Unexpected error in skill %s", self.name)
            result = SkillResult(
                skill_name=self.name,
                status=SkillStatus.FAILED,
                message=f"unexpected {type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - started,
            )

        if result.duration_s == 0.0:
            result.duration_s = time.perf_counter() - started
        self._finish(result, params)
        return result

    def _finish(self, result: SkillResult, params: dict[str, Any]) -> None:
        """Log the outcome and record it in memory."""
        self.ctx.emit("skill.result", {"params": params, **result.to_log()})
        if self.ctx.memory is not None:
            self.ctx.memory.record_result(result)
        _log.info(
            "%s -> %s (%.2fs)%s",
            self.name,
            result.status.value,
            result.duration_s,
            f": {result.message}" if result.message else "",
        )

    @abc.abstractmethod
    def _run(self, params: dict[str, Any]) -> SkillResult:
        """Perform the action. Must not invoke another skill."""

    # ------------------------------------------------------------------
    # helpers shared by concrete skills
    # ------------------------------------------------------------------

    def _ok(self, message: str = "", **data: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name, status=SkillStatus.SUCCESS, message=message, data=data
        )

    def _fail(self, message: str, **data: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name, status=SkillStatus.FAILED, message=message, data=data
        )

    def _infeasible(self, message: str, **data: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name, status=SkillStatus.INFEASIBLE, message=message, data=data
        )

    def _aborted(self, message: str, **data: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name, status=SkillStatus.ABORTED, message=message, data=data
        )

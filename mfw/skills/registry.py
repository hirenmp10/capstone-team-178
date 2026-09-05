"""Skill registry and the classical execution backend.

Pure stdlib + NumPy. This module must never import Isaac Sim.

The registry owns the skill instances; individual skills do not, which is what
makes the atomicity rule structural: a skill has no handle on the registry, so it
cannot reach another skill.

``ClassicalExecutor`` and ``Gr00tExecutor`` both implement
:class:`~mfw.core.interfaces.ISkillExecutor` and are **peers**, selected per-skill
by configuration. GR00T is not a stage between the planner and the motion
planner: it emits relative end-effector action chunks, so it *is* a motion
generator, and wiring it in series would be a category error.
"""

from __future__ import annotations

from typing import Any

from mfw.core.errors import SkillNotFound
from mfw.core.interfaces import ISkill, ISkillExecutor
from mfw.core.types import SkillResult, SkillStatus
from mfw.skills.base import Skill, SkillContext
from mfw.skills.primitives import ALL_SKILLS
from mfw.utils.logging import get_logger

__all__ = ["SkillRegistry", "ClassicalExecutor"]

_log = get_logger("skills.registry")


class SkillRegistry:
    """Holds one instance of every skill, keyed by name."""

    def __init__(self, context: SkillContext, skill_classes: tuple[type[Skill], ...] = ALL_SKILLS):
        self._context = context
        self._skills: dict[str, Skill] = {}
        for cls in skill_classes:
            skill = cls(context)
            if skill.name in self._skills:
                raise ValueError(f"duplicate skill name {skill.name!r}")
            self._skills[skill.name] = skill
        _log.info("Registered %d skills: %s", len(self._skills), sorted(self._skills))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> ISkill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFound(
                f"unknown skill {name!r}; available: {', '.join(self.names)}"
            ) from None

    def execute(self, name: str, params: dict[str, Any] | None = None) -> SkillResult:
        """Execute exactly one skill and return. The robot then WAITS."""
        return self.get(name).execute(params or {})


class ClassicalExecutor(ISkillExecutor):
    """Deterministic backend: perception, grasp synthesis, RRT, PhysX.

    The default. It is fully functional with no learned model, no checkpoint and
    no GPU inference server, which is what makes the system testable end to end
    before GR00T exists.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    @property
    def backend_name(self) -> str:
        return "classical"

    def supports(self, skill_name: str) -> bool:
        return self._registry.has(skill_name)

    def execute(self, skill_name: str, params: dict[str, Any]) -> SkillResult:
        if not self.supports(skill_name):
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.INFEASIBLE,
                message=f"classical backend does not implement {skill_name!r}",
            )
        return self._registry.execute(skill_name, params)

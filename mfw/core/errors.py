"""Framework exception hierarchy.

Pure stdlib. This module must never import Isaac Sim.

Distinct types exist so the state machine can decide recovery policy from the
exception class alone: a :class:`PlanningError` is worth replanning, a
:class:`PerceptionError` is worth re-observing, and a :class:`SafetyViolation`
is never retried.
"""

from __future__ import annotations

__all__ = [
    "MfwError",
    "ConfigurationError",
    "SimulationError",
    "PerceptionError",
    "ObjectNotFound",
    "AmbiguousReference",
    "KinematicsError",
    "UnreachableTarget",
    "PlanningError",
    "GraspError",
    "NoGraspFound",
    "ExecutionError",
    "GraspVerificationFailed",
    "SafetyViolation",
    "PolicyError",
    "SkillNotFound",
]


class MfwError(Exception):
    """Base class for every framework error."""


class ConfigurationError(MfwError):
    """Invalid or inconsistent configuration."""


class SimulationError(MfwError):
    """Simulator failed to start, step, or provide a required asset."""


class PerceptionError(MfwError):
    """Perception could not produce a usable observation.

    Recovery: re-observe, possibly after moving the camera.
    """


class ObjectNotFound(PerceptionError):
    """A referenced object is not present in the current scene graph.

    Expected and non-fatal: objects may legitimately be absent, occluded, or
    removed. The correct response is to report to the user and WAIT.

    ``phrase`` and ``visible`` are optional structured context (the referent
    that failed, and descriptions such as ``"red block"`` of what *is* in
    view). Keyword-only with defaults, so ``ObjectNotFound("no teapot")`` keeps
    working everywhere it is already raised.
    """

    def __init__(
        self,
        message: str = "",
        *,
        phrase: str | None = None,
        visible: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(message)
        self.phrase = phrase
        self.visible: tuple[str, ...] = tuple(visible)


class AmbiguousReference(PerceptionError):
    """A phrase matched several objects and cannot be resolved.

    Recovery: ask the user which one, never guess.

    ``candidates`` are human descriptions ("red block on the left") and
    ``track_ids`` the objects they name, index-aligned, so the question the
    planner asks and the answer it re-runs with refer to the same objects.
    Both default to empty: the one-argument form raised by older code still
    works, and the planner falls back to listing candidates itself.
    """

    def __init__(
        self,
        message: str = "",
        *,
        phrase: str | None = None,
        candidates: tuple[str, ...] | list[str] = (),
        track_ids: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(message)
        if track_ids and candidates and len(track_ids) != len(candidates):
            raise ValueError("candidates and track_ids must be index-aligned")
        self.phrase = phrase
        self.candidates: tuple[str, ...] = tuple(candidates)
        self.track_ids: tuple[str, ...] = tuple(track_ids)


class KinematicsError(MfwError):
    """IK or FK failed."""


class UnreachableTarget(KinematicsError):
    """Target pose has no valid IK solution within joint limits."""


class PlanningError(MfwError):
    """Motion planning failed to find a collision-free path.

    Recovery: replan, up to ``motion.max_replan_attempts``.
    """


class GraspError(MfwError):
    """Grasp synthesis or scoring failed."""


class NoGraspFound(GraspError):
    """No candidate survived width, reachability and collision filtering."""


class ExecutionError(MfwError):
    """Trajectory execution failed or exceeded tracking tolerance."""


class GraspVerificationFailed(ExecutionError):
    """The gripper closed but contact forces show nothing is held.

    Raised on a closure that caught air, or on an object dropped during lift.
    Detected from physics, never inferred from having issued a close command.
    """


class SafetyViolation(MfwError):
    """A commanded motion breached a safety limit.

    Never retried automatically. Covers workspace bounds, velocity limits, and
    policy action deltas exceeding their configured clamps.
    """


class PolicyError(MfwError):
    """The out-of-process policy server is unreachable or returned a bad response."""


class SkillNotFound(MfwError):
    """The requested skill is not registered."""

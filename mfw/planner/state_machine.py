"""The command state machine.

Pure stdlib. This module must never import Isaac Sim.

    IDLE -> WAIT_FOR_COMMAND -> PARSE -> OBSERVE -> PLAN -> EXECUTE
         -> VERIFY -> COMPLETE -> WAIT_FOR_COMMAND

with recovery ``FAILED -> REPLAN -> OBSERVE`` and a ``CLARIFY`` branch for
ambiguous references.

**Atomicity is a property of the graph.** ``COMPLETE`` has exactly one outgoing
edge, back to ``WAIT_FOR_COMMAND``. There is no transition from the completion of
one skill to the start of another, so no command can trigger a follow-up action
even if some future skill implementation tried to request one.

Recovery policy is decided by exception type, which is why the error hierarchy in
``mfw.core.errors`` is granular: a planning failure is worth replanning, a
perception failure is worth re-observing, and a safety violation is never retried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mfw.core.errors import (
    AmbiguousReference,
    ObjectNotFound,
    PerceptionError,
    PlanningError,
    SafetyViolation,
)
from mfw.core.types import SkillResult, SkillStatus
from mfw.utils.logging import get_logger

__all__ = ["State", "StateMachine", "Transition"]

_log = get_logger("planner.state_machine")


class State(str, Enum):
    IDLE = "idle"
    WAIT_FOR_COMMAND = "wait_for_command"
    PARSE = "parse"
    CLARIFY = "clarify"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"
    REPLAN = "replan"
    ABORTED = "aborted"


#: Legal transitions. Note that COMPLETE leads only back to WAIT_FOR_COMMAND:
#: that single edge is what makes every command atomic.
TRANSITIONS: dict[State, tuple[State, ...]] = {
    State.IDLE: (State.WAIT_FOR_COMMAND,),
    State.WAIT_FOR_COMMAND: (State.PARSE, State.IDLE),
    State.PARSE: (State.OBSERVE, State.CLARIFY, State.FAILED),
    State.CLARIFY: (State.WAIT_FOR_COMMAND,),
    State.OBSERVE: (State.PLAN, State.FAILED),
    State.PLAN: (State.EXECUTE, State.FAILED),
    State.EXECUTE: (State.VERIFY, State.FAILED, State.ABORTED),
    State.VERIFY: (State.COMPLETE, State.FAILED),
    State.COMPLETE: (State.WAIT_FOR_COMMAND,),
    State.FAILED: (State.REPLAN, State.WAIT_FOR_COMMAND),
    State.REPLAN: (State.OBSERVE, State.WAIT_FOR_COMMAND),
    State.ABORTED: (State.WAIT_FOR_COMMAND,),
}


@dataclass(frozen=True)
class Transition:
    """One state change, for the trace."""

    from_state: State
    to_state: State
    reason: str = ""

    def to_log(self) -> dict[str, Any]:
        return {"from": self.from_state.value, "to": self.to_state.value, "reason": self.reason}


class IllegalTransition(RuntimeError):
    """Attempted a transition the graph does not allow."""


@dataclass
class StateMachine:
    """Tracks the current state and enforces the transition graph.

    The graph is enforced rather than advisory: an illegal transition raises. A
    controller that can silently jump from ``EXECUTE`` to ``EXECUTE`` is a
    controller that can chain actions, and that is the one thing this design must
    prevent.
    """

    state: State = State.IDLE
    history: list[Transition] = field(default_factory=list)
    max_history: int = 500

    def to(self, target: State, reason: str = "") -> None:
        """Transition, or raise :class:`IllegalTransition`."""
        allowed = TRANSITIONS.get(self.state, ())
        if target not in allowed:
            raise IllegalTransition(
                f"cannot go from {self.state.value} to {target.value}; "
                f"allowed: {[s.value for s in allowed]}"
            )
        _log.debug("%s -> %s%s", self.state.value, target.value, f" ({reason})" if reason else "")
        self.history.append(Transition(self.state, target, reason))
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]
        self.state = target

    def can(self, target: State) -> bool:
        return target in TRANSITIONS.get(self.state, ())

    def reset_to_waiting(self, reason: str = "") -> None:
        """Return to WAIT_FOR_COMMAND from wherever we are.

        Used after a command finishes or gives up. Walks the graph legally where
        possible so the trace stays truthful about how the run actually went.
        """
        if self.state is State.WAIT_FOR_COMMAND:
            return
        if self.can(State.WAIT_FOR_COMMAND):
            self.to(State.WAIT_FOR_COMMAND, reason)
            return
        # Route through FAILED, which is always able to return to waiting.
        if self.can(State.FAILED):
            self.to(State.FAILED, reason or "abandoning command")
            self.to(State.WAIT_FOR_COMMAND, "ready for the next command")
            return
        self.history.append(Transition(self.state, State.WAIT_FOR_COMMAND, reason or "forced"))
        self.state = State.WAIT_FOR_COMMAND

    def trace(self) -> list[dict[str, Any]]:
        return [t.to_log() for t in self.history]


def recovery_for(error: Exception) -> State:
    """Decide the next state from the type of failure.

    Deliberately driven by exception class rather than by inspecting messages, so
    recovery policy is a property of the error taxonomy and cannot drift as
    wording changes.
    """
    if isinstance(error, SafetyViolation):
        # Never retried. A breached limit means the command itself was unsafe.
        return State.ABORTED
    if isinstance(error, AmbiguousReference):
        # Ask; never guess which object was meant.
        return State.CLARIFY
    if isinstance(error, ObjectNotFound):
        # Nothing to retry against: the object genuinely is not there.
        return State.FAILED
    if isinstance(error, (PlanningError, PerceptionError)):
        # Both are worth another attempt from a fresh observation: RRT is
        # randomised, and a viewpoint change can resolve an occlusion.
        return State.REPLAN
    return State.FAILED


def status_to_state(status: SkillStatus) -> State:
    """Map a skill's terminal status onto the machine's vocabulary."""
    if status is SkillStatus.SUCCESS:
        return State.COMPLETE
    if status is SkillStatus.ABORTED:
        return State.ABORTED
    return State.FAILED

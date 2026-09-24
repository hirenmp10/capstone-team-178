"""Task planner: utterance in, one executed skill out.

Pure stdlib + NumPy. This module must never import Isaac Sim.

The planner is the *only* component that composes anything, and what it composes
is deliberately minimal: it drives the state machine through one atomic skill per
command and then waits. It does not expand "pick the can" into a pick-and-place,
and there is no code path in which one command's completion begins another.

The internal stages of a skill (observe, plan grasp, approach, close, verify,
lift) are the *action graph* for that command. They are steps within one atomic
action, not separate commands, and they are reported as such so the operator can
see what the robot decided.

Recovery policy is decided by the *kind* of failure (``recovery_for``), never by
retrying blindly. The audit of 2026-09-24 measured the old behaviour: an
ambiguous "can" (two in view) was retried three times and the operator was
never asked which one; "pick up the teapot" was retried three times against an
object that does not exist. Both arrived as a FAILED ``SkillResult`` because
``Skill.execute`` converts every non-safety ``MfwError`` into a result, so the
planner's exception branches were unreachable. Reference failures are now
recognised on both paths: raised, or reported by a skill result (its
``data["error_type"]``, or the ``"<ErrorClass>: ..."`` prefix that
``Skill.execute`` writes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mfw.core.errors import AmbiguousReference, MfwError, ObjectNotFound, SafetyViolation
from mfw.core.interfaces import IIntentParser, ISkillExecutor
from mfw.core.types import SkillResult, SkillStatus
from mfw.language.intent_parser import UnparsedCommand
from mfw.memory.working_memory import CommandRecord, WorkingMemory
from mfw.planner.state_machine import State, StateMachine, recovery_for
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["TaskPlanner", "CommandOutcome", "ACTION_GRAPHS"]

_log = get_logger("planner.task")


#: The internal stages of each skill, for explaining a decision to the operator.
#: Documentation of the pipeline, not a script: the skills own their control flow,
#: and nothing here is executed.
ACTION_GRAPHS: dict[str, tuple[str, ...]] = {
    "pick": (
        "observe",
        "locate target",
        "generate grasp candidates",
        "score and choose best grasp",
        "plan to pregrasp",
        "approach in a straight line",
        "close gripper (force limited)",
        "verify grasp from contact evidence",
        "lift",
        "hold and WAIT",
    ),
    "place": (
        "observe",
        "locate destination",
        "synthesise release pose",
        "transit above destination",
        "lower",
        "open gripper",
        "retreat",
        "confirm by perception",
        "WAIT",
    ),
    "move_relative": ("compute Cartesian delta", "plan straight line", "execute", "WAIT"),
    "move_to": ("observe", "plan collision-free path", "execute", "WAIT"),
    "look_at": ("observe", "plan viewing pose", "execute", "re-observe", "WAIT"),
    "open_gripper": ("open fingers", "WAIT"),
    "close_gripper": ("close fingers (force limited)", "report stall", "WAIT"),
    "observe": ("capture", "detect", "estimate poses", "track", "build scene graph", "WAIT"),
    "scan_scene": ("sweep viewpoints", "observe at each", "merge tracks", "WAIT"),
    "go_home": ("plan to home posture", "execute", "WAIT"),
    "rotate_wrist": ("plan joint rotation", "execute", "WAIT"),
    "wait": ("idle", "WAIT"),
    "stop": ("halt motion", "hold position"),
    "emergency_stop": ("halt immediately", "zero velocities", "hold position"),
}


@dataclass
class CommandOutcome:
    """Everything that happened in response to one utterance."""

    utterance: str
    skill: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    result: SkillResult | None = None
    message: str = ""
    needs_clarification: bool = False
    clarification_options: tuple[str, ...] = ()
    clarification_track_ids: tuple[str, ...] = ()
    """Index-aligned with ``clarification_options`` when known: the object each
    option names, so the answer can be re-run against exactly that object."""
    clarification_phrase: str = ""
    """The referent that was ambiguous, as the command spelled it ("object")."""
    remaining_clauses: tuple[str, ...] = ()
    """Clauses of the same utterance not yet run because this one stopped
    (set by the Assistant for conjoined and move-X-to-Y commands)."""
    action_graph: tuple[str, ...] = ()
    state_trace: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    def to_log(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "skill": self.skill,
            "params": self.params,
            "message": self.message,
            "needs_clarification": self.needs_clarification,
            "clarification_options": list(self.clarification_options),
            "clarification_track_ids": list(self.clarification_track_ids),
            "remaining_clauses": list(self.remaining_clauses),
            "attempts": self.attempts,
            "duration_s": self.duration_s,
            "result": self.result.to_log() if self.result else None,
            "state_trace": self.state_trace,
        }


class TaskPlanner:
    """Drives one atomic skill per command, then waits."""

    def __init__(
        self,
        parser: IIntentParser,
        executors: dict[str, ISkillExecutor],
        memory: WorkingMemory,
        config: Any,
        vision: Any = None,
        events: EventLogger | None = None,
    ) -> None:
        if not executors:
            raise ValueError("TaskPlanner requires at least one executor")
        self.parser = parser
        self.executors = executors
        self.memory = memory
        self.config = config
        self.vision = vision
        self.events = events
        self.machine = StateMachine()
        self.machine.to(State.WAIT_FOR_COMMAND, "ready")

    # ------------------------------------------------------------------

    def handle(self, utterance: str) -> CommandOutcome:
        """Parse and execute exactly one command, then return to waiting."""
        started = time.perf_counter()
        outcome = CommandOutcome(utterance=utterance)
        trace_start = len(self.machine.history)

        if self.machine.state is not State.WAIT_FOR_COMMAND:
            self.machine.reset_to_waiting("new command arrived")

        try:
            self.machine.to(State.PARSE, "parsing")
            intent = self.parser.parse(utterance, self._context())
            outcome.skill = intent.skill
            outcome.params = dict(intent.params)
            outcome.action_graph = ACTION_GRAPHS.get(intent.skill, ())
        except UnparsedCommand as exc:
            outcome.message = str(exc)
            self.machine.to(State.FAILED, "unparsed")
            self.machine.to(State.WAIT_FOR_COMMAND, "ready")
            outcome.state_trace = self.machine.trace()[trace_start:]
            outcome.duration_s = time.perf_counter() - started
            self._emit(outcome)
            return outcome

        record = CommandRecord(utterance=utterance, skill=intent.skill, params=outcome.params)
        self.memory.record_command(record)

        outcome = self._execute_with_recovery(intent, outcome, trace_start)
        outcome.duration_s = time.perf_counter() - started
        self._emit(outcome)
        return outcome

    def _execute_with_recovery(
        self, intent: Any, outcome: CommandOutcome, trace_start: int
    ) -> CommandOutcome:
        """Run the skill, retrying on recoverable failures only."""
        executor = self._executor_for(intent.skill)
        max_attempts = max(1, self.config.motion.max_replan_attempts)

        for attempt in range(1, max_attempts + 1):
            outcome.attempts = attempt
            try:
                if self.machine.state is State.PARSE:
                    self.machine.to(State.OBSERVE, "perceiving")
                elif self.machine.state is State.REPLAN:
                    self.machine.to(State.OBSERVE, f"retry {attempt}")

                self.machine.to(State.PLAN, "planning")
                self.machine.to(State.EXECUTE, f"executing {intent.skill}")

                result = executor.execute(intent.skill, dict(intent.params))
                outcome.result = result
                outcome.message = result.message

                if result.status is SkillStatus.SUCCESS:
                    self.machine.to(State.VERIFY, "verifying")
                    self.machine.to(State.COMPLETE, "verified")
                    # The single edge out of COMPLETE. No follow-up action.
                    self.machine.to(State.WAIT_FOR_COMMAND, "awaiting the next command")
                    self._remember_reference(intent, result)
                    outcome.state_trace = self.machine.trace()[trace_start:]
                    return outcome

                reference_error = _reference_error(result)
                if reference_error == "AmbiguousReference":
                    return self._clarify(
                        outcome, intent, trace_start, reason="ambiguous reference",
                        candidates=tuple(result.data.get("candidates") or ()),
                        track_ids=tuple(result.data.get("track_ids") or ()),
                    )
                if reference_error == "ObjectNotFound":
                    # Nothing to retry against: the object is not in view.
                    self.machine.to(State.FAILED, "object not found")
                    self.machine.to(State.WAIT_FOR_COMMAND, "ready")
                    outcome.state_trace = self.machine.trace()[trace_start:]
                    return outcome

                if result.status is SkillStatus.ABORTED:
                    self.machine.to(State.ABORTED, "aborted")
                    self.machine.to(State.WAIT_FOR_COMMAND, "ready")
                    outcome.state_trace = self.machine.trace()[trace_start:]
                    return outcome

                # INFEASIBLE is a considered refusal, not bad luck: retrying an
                # object that is too wide for the gripper cannot help.
                # A skill that has already changed the world irreversibly (a
                # Place that released the object, then measured a miss) says
                # retryable=False: a retry could only answer "not holding
                # anything" and replace the measured failure.
                not_retryable = isinstance(result.data, dict) and result.data.get("retryable") is False
                if result.status is SkillStatus.INFEASIBLE or not_retryable or attempt >= max_attempts:
                    self.machine.to(State.FAILED, result.status.value)
                    self.machine.to(State.WAIT_FOR_COMMAND, "ready")
                    outcome.state_trace = self.machine.trace()[trace_start:]
                    return outcome

                self.machine.to(State.FAILED, "will retry")
                self.machine.to(State.REPLAN, f"attempt {attempt + 1}")

            except SafetyViolation as exc:
                # Never retried, and never swallowed. EXECUTE -> ABORTED is the
                # graph's own word for it; the trace used to say FAILED.
                outcome.message = f"safety violation: {exc}"
                if self.machine.can(State.ABORTED):
                    self.machine.to(State.ABORTED, "safety violation")
                self.machine.reset_to_waiting("safety violation")
                outcome.state_trace = self.machine.trace()[trace_start:]
                _log.error("Safety violation handling %r: %s", outcome.utterance, exc)
                return outcome

            except AmbiguousReference as exc:
                outcome.message = str(exc)
                return self._clarify(
                    outcome, intent, trace_start, reason="ambiguous reference",
                    candidates=exc.candidates, track_ids=exc.track_ids,
                    phrase=exc.phrase,
                )

            except ObjectNotFound as exc:
                outcome.message = f"{type(exc).__name__}: {exc}"
                self.machine.to(State.FAILED, "object not found")
                self.machine.to(State.WAIT_FOR_COMMAND, "ready")
                outcome.state_trace = self.machine.trace()[trace_start:]
                return outcome

            except MfwError as exc:
                outcome.message = f"{type(exc).__name__}: {exc}"
                # Recovery policy comes from the exception's class, so it cannot
                # drift as error wording changes.
                target = recovery_for(exc)

                if target is State.REPLAN and attempt < max_attempts:
                    # EXECUTE -> FAILED -> REPLAN, then the loop re-enters at
                    # OBSERVE so the retry works from a fresh observation.
                    self.machine.to(State.FAILED, f"{type(exc).__name__}, will retry")
                    self.machine.to(State.REPLAN, f"attempt {attempt + 1}")
                    continue

                self.machine.reset_to_waiting(target.value)
                outcome.state_trace = self.machine.trace()[trace_start:]
                return outcome

        self.machine.reset_to_waiting("attempts exhausted")
        outcome.state_trace = self.machine.trace()[trace_start:]
        return outcome

    # ------------------------------------------------------------------

    def _executor_for(self, skill_name: str) -> ISkillExecutor:
        """Select the backend for this skill.

        Falls back to the classical executor when a configured backend cannot run
        the skill, rather than failing: a GR00T checkpoint that does not cover
        ``open_gripper`` should not make the gripper unusable.
        """
        preferred = self.config.executor_for(skill_name)
        executor = self.executors.get(preferred)
        if executor is not None and executor.supports(skill_name):
            return executor

        for name, candidate in self.executors.items():
            if candidate.supports(skill_name):
                if preferred != name:
                    _log.info(
                        "Backend %r cannot run %r; using %r", preferred, skill_name, name
                    )
                return candidate

        raise MfwError(f"no backend can execute {skill_name!r}")

    def _remember_reference(self, intent: Any, result: SkillResult) -> None:
        """Record which object a successful command referred to.

        This is what lets the *next* command say "it" -- the referent chain is
        built from what actually succeeded, not from what was requested.
        """
        track_id = result.data.get("track_id")
        if track_id:
            self.memory.note_reference(str(track_id))

    def _clarify(
        self,
        outcome: CommandOutcome,
        intent: Any,
        trace_start: int,
        *,
        reason: str,
        candidates: tuple[str, ...] = (),
        track_ids: tuple[str, ...] = (),
        phrase: str | None = None,
    ) -> CommandOutcome:
        """EXECUTE -> CLARIFY -> WAIT with the options to offer. Never retried.

        Options come from the exception when it carried them; otherwise they are
        recomputed by grounding the same phrase against the scene the skill just
        observed, so the question always names real, distinguishable objects.
        """
        outcome.needs_clarification = True
        target = str(intent.params.get("target") or "")
        outcome.clarification_phrase = phrase or target
        if not candidates or not track_ids:
            candidates, track_ids = self._ground_candidates(target)
        if not candidates:
            candidates = self._clarification_options(intent)
            track_ids = ()
        outcome.clarification_options = tuple(candidates)
        outcome.clarification_track_ids = tuple(track_ids)
        if not outcome.message:
            outcome.message = f"{target!r} is ambiguous"
        self.machine.to(State.CLARIFY, reason)
        self.machine.to(State.WAIT_FOR_COMMAND, "awaiting clarification")
        outcome.state_trace = self.machine.trace()[trace_start:]
        return outcome

    def _ground_candidates(self, phrase: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Descriptions and track ids ``phrase`` could mean in the current scene."""
        from mfw.language.grounding import resolve_reference

        scene = self.memory.current_scene
        if not phrase or scene is None:
            return (), ()
        try:
            resolve_reference(phrase, scene, memory=self.memory, robot_xy=self._robot_xy())
        except AmbiguousReference as exc:
            return exc.candidates, exc.track_ids
        except MfwError:
            return (), ()
        return (), ()

    def _robot_xy(self) -> tuple[float, float]:
        robot = getattr(self.config, "robot", None)
        base = getattr(robot, "base_position", (0.0, 0.0, 0.0))
        return float(base[0]), float(base[1])

    def preview(self, utterance: str) -> Any | None:
        """Parse ``utterance`` without executing it or touching the state machine.

        ``None`` when it does not parse. Used to decide, before anything moves,
        whether a spoken command needs confirming.
        """
        try:
            return self.parser.parse(utterance, self._context())
        except UnparsedCommand:
            return None

    def _clarification_options(self, intent: Any) -> tuple[str, ...]:
        """Candidate objects to offer when a reference is ambiguous (legacy form)."""
        target = intent.params.get("target")
        if not target:
            return ()
        scene = self.memory.current_scene
        candidates = self.memory.candidates_for(str(target))
        if scene is None:
            return tuple(candidates)
        return tuple(
            f"{track_id} ({scene.objects[track_id].label})"
            for track_id in candidates
            if track_id in scene.objects
        )

    def _context(self) -> dict[str, Any]:
        """Grounding context handed to the parser."""
        scene = self.memory.current_scene
        return {
            "visible_objects": (
                sorted({obj.label for obj in scene.objects.values() if obj.label})
                if scene is not None
                else []
            ),
            "held_object": self.memory.get_held_object(),
            "state": self.machine.state.value,
        }

    def describe_action_graph(self, skill_name: str) -> tuple[str, ...]:
        """The internal stages of a skill, for explaining a decision."""
        return ACTION_GRAPHS.get(skill_name, ())

    def _emit(self, outcome: CommandOutcome) -> None:
        if self.events is not None:
            self.events.emit("planner.command", outcome.to_log())


_REFERENCE_ERRORS = ("AmbiguousReference", "ObjectNotFound")


def _reference_error(result: SkillResult) -> str | None:
    """Which reference failure, if any, a non-successful skill result reports.

    ``Skill.execute`` turns a raised ``AmbiguousReference``/``ObjectNotFound``
    into a FAILED result whose message is ``"<ErrorClass>: <text>"`` -- written
    by code, not prose, so it is a stable marker. ``data["error_type"]`` is
    preferred when a skill supplies it.
    """
    if result.status is SkillStatus.SUCCESS:
        return None
    declared = result.data.get("error_type") if isinstance(result.data, dict) else None
    if declared in _REFERENCE_ERRORS:
        return str(declared)
    message = result.message or ""
    for name in _REFERENCE_ERRORS:
        if message.startswith(f"{name}:"):
            return name
    return None

"""GR00T execution backend.

Pure stdlib + NumPy at module scope (plus the framework's own pure modules).
This module must never import Isaac Sim, torch, or GR00T -- the model lives in
another process entirely.

``Gr00tExecutor`` is a **peer** of ``ClassicalExecutor``, not a layer above or
below it. Both implement :class:`~mfw.core.interfaces.ISkillExecutor`, and config
picks which one runs each skill.

The closed loop, per skill invocation:

    ground the target -> observe -> build observation (history) -> request an
    action chunk -> execute a *prefix* of it through the safety filter ->
    judge the goal from physical evidence -> re-observe

Only a prefix (``actions_executed_per_chunk`` of ``action_horizon``) is executed
before re-observing. Running all 40 steps open-loop lets prediction error compound
with nothing to correct it; the policy was trained to be queried repeatedly.

Skills the policy cannot express -- ``observe``, ``go_home``, ``stop`` -- are not
faked here. :meth:`supports` returns ``False`` for them so the planner routes them
to the classical backend, which is why a partial policy never makes a whole
capability unavailable.

Honesty rules (measured failures, 2026-09-24)
---------------------------------------------
A live run with the real N1.7 server reported "pick up the blue can" as
SUCCESS after one iteration while the executor itself logged that nothing was
in the hand. The first observation after startup had coloured every object
black, so "blue can" matched nothing, ``target_track_id`` was ``None``, and the
old goal test returned ``True`` for *any* gripper reporting a grasp -- and a
close on air reports one. Hence:

* **Grounding comes first.** The pick target (and a place destination) goes
  through :func:`mfw.language.grounding.resolve_reference`, pronouns via
  working memory, exactly as the classical skills do. ``ObjectNotFound`` /
  ``AmbiguousReference`` become a FAILED result in the planner's reference
  contract *before the policy is queried*: the policy is never run blind for
  a named object.
* **Pick success is physical.** The gripper must report a grasp, the resolved
  object must be within :data:`FINGERTIP_RADIUS_M` of the TCP, and -- on a lane
  with depth (``robot.gripper_feedback``) -- it must have risen at least
  :data:`MIN_LIFT_M` from where it rested. SUCCESS is returned only once the
  object is recorded as held in working memory; anything else is FAILED with
  "policy ran N iteration(s) / M step(s); no object grasped".
* **Place success is a release, not an open gripper.** The gripper must have
  been closed on the held object when the place began (a place whose gripper
  is already open is refused before the policy runs), the policy must have
  commanded an open during this skill, and the gripper must now measure open
  with no grasp. Memory and the classical backend's ``held_grasp`` record are
  cleared together.

Observability
-------------
Every ``gr00t.iteration`` event records the TCP at the start and end of the
iteration, the displacement between them, the largest measured single-step
move, the policy's raw gripper closure and the commanded and measured
gripper widths, clamped/rejected counts, the
instruction, the raw action-chunk statistics (shape and per-dimension min/max
for every key the policy returned) and the goal verdict with its reason. That
is the evidence that GR00T did -- or did not -- move the arm.

Working memory is shared with the classical backend
---------------------------------------------------
With ``--groot --groot-skills pick`` the pick runs here and the following
"place it" runs classically. A successful pick calls ``memory.set_held_object``
and returns ``data["track_id"]`` (which the planner also uses to resolve "it");
a successful place clears it.

Status: this loop has run against fakes and the mock policy
(``tests/test_gr00t_zmq.py``, ``tests/test_gr00t_honesty.py``) and once against
the real checkpoint in Isaac (``logs/e2e/run_f_groot.txt``), where the false
success above was found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mfw.config.schema import FrameworkConfig
from mfw.core.errors import (
    AmbiguousReference,
    ObjectNotFound,
    PolicyError,
    SafetyViolation,
)
from mfw.core.interfaces import ISkillExecutor
from mfw.core.types import SkillResult, SkillStatus
from mfw.gr00t_bridge.observation import ObservationBuilder, gripper_width_from_closure
from mfw.gr00t_bridge.safety import ActionSafetyFilter
from mfw.language.grounding import describe, resolve_reference
from mfw.utils.logging import EventLogger, get_logger

__all__ = [
    "Gr00tExecutor",
    "support_height",
    "FINGERTIP_RADIUS_M",
    "MIN_LIFT_M",
    "chunk_stats",
]

_log = get_logger("gr00t.executor")

#: Skills a vision-language-action policy can plausibly perform. Everything else
#: belongs to the deterministic backend.
POLICY_SKILLS = frozenset({"pick", "place", "move_to", "move_relative", "look_at"})

#: A picked object must be within this distance of the TCP to count as held.
FINGERTIP_RADIUS_M = 0.09

#: On a lane with depth, a picked object must have risen at least this far from
#: where it rested. Closing next to an object leaves it within the fingertip
#: radius; only a lift shows it followed the hand.
MIN_LIFT_M = 0.02

#: The gripper counts as open above this fraction of its full opening.
_OPEN_FRACTION = 0.7

#: Language given to the policy for each skill. GR00T is language-conditioned, so
#: this is a functional input, not a label.
INSTRUCTION_TEMPLATES = {
    "pick": "pick up the {target}",
    "place": "put down the object{destination}",
    "move_to": "move the gripper to the {target}",
    "move_relative": "move the gripper {direction}",
    "look_at": "look at the {target}",
}


def support_height(config: FrameworkConfig) -> float:
    """Height of the surface the arm works on, by the runtimes' own rule.

    Mirrors ``simulation.runtime.Runtime.table_top_height`` (table top when a
    table is spawned, else ``perception.ground_plane_z``) without importing the
    simulation package, which this module must never do.
    """
    if not config.scene.add_table:
        return float(config.perception.ground_plane_z)
    return float(config.scene.table_position[2] + config.scene.table_scale[2] / 2.0)


def _unwrap_action(reply: Any) -> dict[str, Any]:
    """Pull the action dict out of whatever the policy returned.

    ``Gr00tPolicy.get_action`` returns a **tuple** whose first element is the
    action dict, not a bare dict as the docs suggest. Handling both keeps this
    working across the local checkpoint and the ZeroMQ server, which unwraps
    differently.
    """
    if isinstance(reply, dict):
        return reply
    if isinstance(reply, (tuple, list)):
        for item in reply:
            if isinstance(item, dict):
                return item
    raise PolicyError(
        f"could not find an action dict in a {type(reply).__name__} reply"
    )


def _drop_batch(array: np.ndarray) -> np.ndarray:
    """Remove a leading singleton batch axis, leaving (horizon, D).

    The policy emits (B, horizon, D) with B=1 for single-robot inference.
    Squeezing only a *singleton* axis is deliberate: a genuine batch greater
    than one would mean several robots' actions arrived together, and silently
    taking the first would drive this arm with another's commands.
    """
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise PolicyError(
                f"expected batch size 1 from the policy, got {array.shape[0]}"
            )
        return array[0]
    return array


def _rounded(values: Any, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def chunk_stats(chunk: dict[str, Any]) -> dict[str, Any]:
    """Shape and per-dimension min/max of every numeric array in an action chunk.

    Recorded raw -- before unbatching, clamping or thresholding -- so the log
    shows what the policy actually emitted (for example whether
    ``gripper_position`` is a width in metres or a 0..1 closure).
    """
    stats: dict[str, Any] = {}
    for key, value in chunk.items():
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        entry: dict[str, Any] = {"shape": [int(n) for n in array.shape]}
        if array.size:
            flat = array.reshape(-1, array.shape[-1]) if array.ndim >= 1 else array.reshape(1, 1)
            entry["min"] = _rounded(flat.min(axis=0))
            entry["max"] = _rounded(flat.max(axis=0))
        stats[str(key)] = entry
    return stats


@dataclass(frozen=True)
class _Target:
    """A grounded object: the track, how to name it, and where it was."""

    track_id: str
    description: str
    position: tuple[float, float, float]


@dataclass
class _Episode:
    """Per-invocation bookkeeping the goal test needs."""

    skill: str
    target: _Target | None = None
    held_id: str | None = None
    held_description: str = ""
    destination: _Target | None = None
    opened_during_skill: bool = False
    start_gripper_width: float | None = None
    last_reason: str = "the goal was never checked"
    totals: dict[str, int] = field(
        default_factory=lambda: {"executed_steps": 0, "clamped_actions": 0, "rejected_actions": 0}
    )


class Gr00tExecutor(ISkillExecutor):
    """Runs skills by closed-loop querying of the out-of-process policy."""

    def __init__(
        self,
        client: Any,
        robot: Any,
        vision: Any,
        controller: Any,
        cameras: dict[str, Any],
        config: FrameworkConfig,
        memory: Any = None,
        events: EventLogger | None = None,
        skill_context: Any = None,
    ) -> None:
        """``skill_context`` (optional) is the classical backend's
        :class:`~mfw.skills.base.SkillContext`. Its ``held_grasp`` record
        describes how the *classical* pick holds an object; a GR00T pick or
        place invalidates it, so this executor clears it whenever it changes
        what is held.
        """
        self.client = client
        self.robot = robot
        self.vision = vision
        self.controller = controller
        self.cameras = cameras
        self.config = config
        self.memory = memory
        self.events = events
        self.skill_context = skill_context

        self.observation_builder = ObservationBuilder(
            config.gr00t,
            gripper_open_width=config.robot.gripper_open_width,
            gripper_closed_width=config.robot.gripper_closed_width,
        )
        self.safety = ActionSafetyFilter(
            config=config.gr00t,
            workspace_min=np.array(config.scene.workspace_min),
            workspace_max=np.array(config.scene.workspace_max),
            gripper_open_width=config.robot.gripper_open_width,
            gripper_closed_width=config.robot.gripper_closed_width,
            # The workspace box's z-min is the floor; the table top is ~0.4 m
            # above it. Without this the "safe" clamp allowed targets in the table.
            support_height=support_height(config),
        )

    @property
    def backend_name(self) -> str:
        return "gr00t"

    def supports(self, skill_name: str) -> bool:
        """Whether the policy can express this skill.

        Returning ``False`` is the honest answer for ``observe`` or ``stop`` -- a
        VLA emits motion, not a scene graph -- and it lets the planner fall back
        to the classical backend rather than the capability disappearing.
        """
        return skill_name in POLICY_SKILLS

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def execute(self, skill_name: str, params: dict[str, Any]) -> SkillResult:
        started = time.perf_counter()
        params = dict(params or {})

        def refuse(status: SkillStatus, message: str, **data: Any) -> SkillResult:
            self.emit(
                "gr00t.refused",
                {"skill": skill_name, "status": status.value, "message": message,
                 "params": {k: v for k, v in params.items() if isinstance(v, (str, int, float))}},
            )
            return SkillResult(
                skill_name=skill_name,
                status=status,
                message=message,
                duration_s=time.perf_counter() - started,
                data=data,
            )

        if not self.supports(skill_name):
            return refuse(SkillStatus.INFEASIBLE, f"the GR00T backend does not express {skill_name!r}")

        # Preconditions that need no server: refusing before connecting keeps a
        # doomed command from waiting on a policy it will never use.
        episode = _Episode(skill=skill_name)
        if skill_name == "pick":
            problem = self._pick_precondition(params)
            if problem is not None:
                return refuse(SkillStatus.INFEASIBLE, problem)
        elif skill_name == "place":
            problem = self._place_precondition(episode)
            if problem is not None:
                return refuse(SkillStatus.INFEASIBLE, problem)

        # Ground before the policy runs: an object that cannot be named is a
        # question or a refusal, never a blind policy rollout.
        try:
            self._ground(skill_name, params, episode)
        except (AmbiguousReference, ObjectNotFound) as exc:
            return refuse(
                SkillStatus.FAILED, f"{type(exc).__name__}: {exc}", **_reference_error_data(exc)
            )

        if not self.client.is_ready():
            try:
                self.client.connect()
            except PolicyError as exc:
                return refuse(SkillStatus.FAILED, f"policy server unavailable: {exc}")

        instruction = self._instruction(skill_name, params, episode)
        max_iterations = int(params.get("max_iterations", 12))

        try:
            return self._run_closed_loop(skill_name, instruction, max_iterations, started, episode)
        except SafetyViolation:
            # Propagates: the state machine must see a safety breach.
            raise
        except PolicyError as exc:
            totals = episode.totals
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                message=f"policy error: {exc}",
                duration_s=time.perf_counter() - started,
                data=dict(totals),
            )

    # ------------------------------------------------------------------
    # preconditions and grounding
    # ------------------------------------------------------------------

    def _pick_precondition(self, params: dict[str, Any]) -> str | None:
        if not params.get("target"):
            return "pick requires a target"
        if self.memory is None:
            # SUCCESS means "recorded as held"; without memory that is impossible,
            # so refuse before moving rather than after.
            return "a GR00T pick needs working memory to record what it holds"
        if self.memory.get_held_object() is not None:
            return "already holding something; place or release it first"
        return None

    def _place_precondition(self, episode: _Episode) -> str | None:
        if self.memory is None or self.memory.get_held_object() is None:
            # Same precondition as the classical Place. Without it the goal
            # test would have nothing to have released.
            return "not holding anything to place"
        episode.held_id = str(self.memory.get_held_object())
        gripper = self._gripper_state()
        if gripper is None:
            return "a GR00T place needs gripper state to tell a release from an open hand"
        episode.start_gripper_width = float(gripper.width)
        if self._is_open(gripper):
            # Audit P3: an open gripper satisfied the old goal test before the
            # policy moved at all. Nothing is between the fingers to release.
            return (
                f"memory says {episode.held_id} is held, but the gripper is already open "
                f"({gripper.width * 1000:.1f} mm, no grasp); there is nothing to release"
            )
        return None

    def _ground(self, skill_name: str, params: dict[str, Any], episode: _Episode) -> None:
        """Resolve the pick target / place destination; raise reference errors."""
        if skill_name not in ("pick", "place"):
            return
        if skill_name == "place":
            scene = self.vision.require_fresh_scene() if self.vision is not None else None
            if scene is not None and episode.held_id in getattr(scene, "objects", {}):
                episode.held_description = describe(scene.objects[episode.held_id])
            if not params.get("target"):
                return
            episode.destination = self._resolve_looking_again(
                str(params["target"]), scene, exclude=(episode.held_id,)
            )
            return

        scene = self.vision.require_fresh_scene()
        episode.target = self._resolve_looking_again(str(params["target"]), scene, exclude=())

    def _resolve_looking_again(
        self, reference: str, scene: Any, exclude: tuple[str | None, ...]
    ) -> _Target:
        """:meth:`_resolve`, observing once more on ``ObjectNotFound`` before refusing.

        The same rule as the classical skills (``mfw.skills.primitives._observe_and_resolve``):
        a track that aged out while the object was out of view comes back
        unconfirmed, so the first scene can miss an object that is plainly
        there. ``AmbiguousReference`` is a question and is not retried.
        """
        try:
            return self._resolve(reference, scene, exclude)
        except ObjectNotFound as first:
            if self.vision is None or not callable(getattr(self.vision, "observe", None)):
                raise
            _log.debug("reference not found (%s); observing once more before refusing", first)
        return self._resolve(reference, self.vision.observe(), exclude)

    def _resolve(self, reference: str, scene: Any, exclude: tuple[str | None, ...]) -> _Target:
        """Exactly the classical rule: an exact track id as is, else the grounding."""
        if scene is None:
            raise ObjectNotFound(f"cannot find {reference!r}: no scene is perceived", phrase=reference)
        reference = reference.strip()
        excluded = {e for e in exclude if e}
        objects = getattr(scene, "objects", {})
        if reference in objects and reference not in excluded:
            obj = objects[reference]
        else:
            obj = resolve_reference(
                reference,
                scene,
                memory=self.memory,
                robot_xy=self._robot_xy(),
                exclude_ids=excluded,
            )
        position = np.asarray(obj.pose.position, dtype=np.float64)
        return _Target(
            track_id=obj.track_id,
            description=describe(obj),
            position=(float(position[0]), float(position[1]), float(position[2])),
        )

    def _robot_xy(self) -> tuple[float, float]:
        base = getattr(getattr(self.config, "robot", None), "base_position", (0.0, 0.0, 0.0))
        return float(base[0]), float(base[1])

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _run_closed_loop(
        self,
        skill_name: str,
        instruction: str,
        max_iterations: int,
        started: float,
        episode: _Episode,
    ) -> SkillResult:
        # Fresh episode: a history carried over from a previous command would show
        # the policy motion that never happened in this one.
        self._prime_history()
        totals = episode.totals

        for iteration in range(1, max_iterations + 1):
            self._push_frame()
            state = self.robot.get_state()
            observation = self.observation_builder.build(state, instruction)

            chunk = _unwrap_action(self.client.predict(observation))
            record = self._new_iteration_record(skill_name, iteration, instruction, episode, chunk)

            try:
                eef_chunk, gripper_chunk = self._decode_chunk(chunk)
                self._execute_prefix(eef_chunk, gripper_chunk, iteration, episode, record)
            except SafetyViolation as exc:
                record["rejected"] += 1
                totals["rejected_actions"] += 1
                record["safety_violation"] = str(exc)
                self._finish_iteration_record(record, episode, reached=False,
                                              reason=f"safety violation: {exc}")
                raise

            reached, reason = self._evaluate_goal(skill_name, episode)
            episode.last_reason = reason
            self._finish_iteration_record(record, episode, reached=reached, reason=reason)

            if reached:
                data: dict[str, Any] = {
                    "iterations": iteration,
                    **totals,
                    "instruction": instruction,
                    "evidence": reason,
                }
                recorded = self._record_outcome(skill_name, episode)
                if recorded is None:
                    # Unreachable for a well-behaved memory, but the contract is
                    # "SUCCESS only when memory says it is held": check, don't assume.
                    return self._failure(skill_name, iteration, started, episode,
                                         "working memory did not record the held object")
                data.update(recorded)
                self.emit("gr00t.result", {"skill": skill_name, "status": "success",
                                           "reason": reason, **totals})
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.SUCCESS,
                    message=(
                        f"policy completed after {iteration} iteration(s), "
                        f"{totals['executed_steps']} step(s), {totals['clamped_actions']} clamped: "
                        f"{reason}"
                    ),
                    duration_s=time.perf_counter() - started,
                    data=data,
                )

        return self._failure(skill_name, max_iterations, started, episode, episode.last_reason)

    def _failure(
        self, skill_name: str, iterations: int, started: float, episode: _Episode, reason: str
    ) -> SkillResult:
        """The honest non-success result for a loop that ran out of iterations."""
        totals = episode.totals
        ran = (
            f"policy ran {iterations} iteration(s) / {totals['executed_steps']} step(s) "
            f"({totals['clamped_actions']} clamped, {totals['rejected_actions']} rejected)"
        )
        if skill_name == "pick":
            name = episode.target.description if episode.target else "object"
            status = SkillStatus.FAILED
            message = f"{ran}; no object grasped: {reason} (target: {name})"
        elif skill_name == "place":
            name = episode.held_description or episode.held_id or "held object"
            status = SkillStatus.FAILED
            message = f"{ran}; the {name} was not released: {reason}"
        else:
            # Open-ended motions have no perceptual goal; the budget ends them.
            status = SkillStatus.TIMEOUT
            message = (
                f"policy did not reach the goal in {iterations} iterations "
                f"({totals['executed_steps']} steps executed, {totals['clamped_actions']} clamped)"
            )
        self.emit("gr00t.result", {"skill": skill_name, "status": status.value,
                                   "reason": reason, **totals})
        return SkillResult(
            skill_name=skill_name,
            status=status,
            message=message,
            duration_s=time.perf_counter() - started,
            data={"iterations": iterations, **totals, "reason": reason},
        )

    def _decode_chunk(self, chunk: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        # Actions come back batched as (B, horizon, D); drop the batch axis.
        # (1, 40, 9) is what NVIDIA's own server returned for eef_9d.
        eef_chunk = _drop_batch(np.asarray(chunk.get("eef_9d"), dtype=np.float64))
        if eef_chunk.ndim != 2 or eef_chunk.shape[1] != 9:
            raise PolicyError(
                f"expected an (N, 9) eef_9d chunk after unbatching, got "
                f"shape {eef_chunk.shape}"
            )
        gripper_chunk = _drop_batch(
            np.asarray(
                chunk.get("gripper_position", np.zeros((1, eef_chunk.shape[0], 1))),
                dtype=np.float64,
            )
        ).reshape(eef_chunk.shape[0], -1)
        return eef_chunk, gripper_chunk

    def _execute_prefix(
        self,
        eef_chunk: np.ndarray,
        gripper_chunk: np.ndarray,
        iteration: int,
        episode: _Episode,
        record: dict[str, Any],
    ) -> None:
        """Execute a prefix of the chunk, then re-observe (open-loop compounds error)."""
        totals = episode.totals
        prefix = min(self.config.gr00t.actions_executed_per_chunk, eef_chunk.shape[0])

        for step in range(prefix):
            # The policy's gripper_position is DROID closure (0 open .. 1
            # closed); the filter and the robot speak widths.
            closure = float(gripper_chunk[step, 0])
            width_command = gripper_width_from_closure(
                closure, self.config.robot.gripper_open_width, self.config.robot.gripper_closed_width
            )
            if not np.isfinite(width_command):
                raise SafetyViolation(f"policy gripper command is non-finite ({closure!r})")
            current = self.robot.tcp_pose()
            action = self.safety.apply(
                current_tcp=current,
                eef_9d=eef_chunk[step],
                gripper_command=width_command,
                # GR00T's processor denormalises AND converts relative->absolute,
                # so what arrives is a world-frame pose despite the embodiment
                # tag saying "relative". See ActionSafetyFilter.apply.
                absolute=self.config.gr00t.actions_are_absolute,
            )
            record["max_commanded_step_m"] = max(
                record["max_commanded_step_m"], round(float(action["original_translation"]), 5)
            )
            if action.was_clamped:
                record["clamped"] += 1
                totals["clamped_actions"] += 1

            if not self.controller.servo_to_pose(action.target_pose):
                record["rejected"] += 1
                totals["rejected_actions"] += 1
                self.emit(
                    "gr00t.servo_failed",
                    {"iteration": iteration, "step": step,
                     "target": action.target_pose.to_log()},
                )
                break

            command = self._apply_gripper(action.gripper_width, episode)
            after = np.asarray(self.robot.tcp_pose().position, dtype=np.float64)
            moved = float(np.linalg.norm(after - np.asarray(current.position, dtype=np.float64)))
            record["max_step_translation_m"] = max(record["max_step_translation_m"], round(moved, 5))
            record["gripper_closure_commanded"].append(round(closure, 5))
            record["gripper_commanded_m"].append(round(float(action.gripper_width), 5))
            record["gripper_commands"].append(command)
            measured = self._gripper_state()
            record["gripper_measured_m"].append(
                None if measured is None else round(float(measured.width), 5)
            )
            record["steps"] += 1
            totals["executed_steps"] += 1

    def _new_iteration_record(
        self, skill_name: str, iteration: int, instruction: str, episode: _Episode,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        tcp = np.asarray(self.robot.tcp_pose().position, dtype=np.float64)
        return {
            "skill": skill_name,
            "iteration": iteration,
            "instruction": instruction,
            "target_track_id": episode.target.track_id if episode.target else None,
            "held_track_id": episode.held_id,
            "tcp_start": _rounded(tcp),
            "_tcp_start": tcp,
            "steps": 0,
            "clamped": 0,
            "rejected": 0,
            "max_step_translation_m": 0.0,
            "max_commanded_step_m": 0.0,
            "gripper_closure_commanded": [],
            "gripper_commanded_m": [],
            "gripper_commands": [],
            "gripper_measured_m": [],
            "action_chunk": chunk_stats(chunk),
        }

    def _finish_iteration_record(
        self, record: dict[str, Any], episode: _Episode, *, reached: bool, reason: str
    ) -> None:
        start = record.pop("_tcp_start")
        end = np.asarray(self.robot.tcp_pose().position, dtype=np.float64)
        totals = episode.totals
        record.update(
            tcp_end=_rounded(end),
            tcp_displacement_m=round(float(np.linalg.norm(end - start)), 5),
            executed_steps=totals["executed_steps"],
            clamped_actions=totals["clamped_actions"],
            rejected_actions=totals["rejected_actions"],
            goal_reached=bool(reached),
            goal_reason=reason,
        )
        self.emit("gr00t.iteration", record)

    # ------------------------------------------------------------------
    # outcomes
    # ------------------------------------------------------------------

    def _record_outcome(self, skill_name: str, episode: _Episode) -> dict[str, Any] | None:
        """Tell working memory what a successful policy skill changed.

        Returns the fields to merge into ``SkillResult.data``, or ``None`` when
        a pick could not be recorded as held. ``track_id`` is what the
        classical backend reports too, so "it" resolves the same way whichever
        backend ran the pick.
        """
        if skill_name == "pick":
            if episode.target is None or self.memory is None:
                return None  # _evaluate_goal and the precondition refuse both
            held = episode.target.track_id
            self.memory.set_held_object(held)
            self._clear_held_grasp()
            if self.memory.get_held_object() != held:
                return None
            return {"track_id": held, "object": episode.target.description}

        if skill_name == "place":
            released = episode.held_id
            if self.memory is not None:
                self.memory.set_held_object(None)
            self._clear_held_grasp()
            data: dict[str, Any] = {"track_id": released} if released else {}
            if episode.destination is not None:
                data["destination_id"] = episode.destination.track_id
            return data

        return {}

    def _clear_held_grasp(self) -> None:
        """Drop the classical pick's grasp record: it no longer describes the hand."""
        if self.skill_context is not None and hasattr(self.skill_context, "held_grasp"):
            self.skill_context.held_grasp = None

    # ------------------------------------------------------------------
    # sensing helpers
    # ------------------------------------------------------------------

    def _prime_history(self) -> None:
        exterior, wrist = self._capture_pair()
        self.observation_builder.prime(exterior, wrist)

    def _push_frame(self) -> None:
        exterior, wrist = self._capture_pair()
        self.observation_builder.push(exterior, wrist)

    def _capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Grab both camera images the embodiment requires.

        ``oxe_droid`` declares both ``exterior_image_1_left`` and
        ``wrist_image_left``; a missing view is a contract violation, not a
        degraded input, so this raises rather than substituting a blank frame.
        """
        exterior_name = self.config.exterior_camera.name
        wrist_name = self.config.wrist_camera.name

        exterior = self.cameras.get(exterior_name)
        wrist = self.cameras.get(wrist_name)
        if exterior is None or wrist is None:
            raise PolicyError(
                f"the {self.config.gr00t.embodiment_tag} embodiment requires both an "
                f"exterior and a wrist camera; have {sorted(self.cameras)}"
            )
        return exterior.capture().rgb, wrist.capture().rgb

    def _gripper_state(self) -> Any:
        """The robot's gripper state, or ``None`` for a robot that exposes none."""
        getter = getattr(self.robot, "get_gripper_state", None)
        return getter() if callable(getter) else None

    def _is_open(self, gripper: Any) -> bool:
        return (not gripper.is_grasping) and float(gripper.width) > (
            self.config.robot.gripper_open_width * _OPEN_FRACTION
        )

    def _apply_gripper(self, width: float, episode: _Episode | None = None) -> str:
        """Drive the gripper toward the commanded width; return "open" or "close".

        ``width`` is the filtered width converted from the policy's DROID
        closure; the framework exposes open/close, so it is thresholded at the
        midpoint -- closure 0.5, as NVIDIA's DROID client binarises it.
        """
        midpoint = (
            self.config.robot.gripper_open_width + self.config.robot.gripper_closed_width
        ) / 2.0
        if width >= midpoint:
            self.robot.open_gripper()
            if episode is not None:
                episode.opened_during_skill = True
            return "open"
        self.robot.close_gripper()
        return "close"

    def _goal_reached(self, skill_name: str, episode: _Episode | None = None) -> bool:
        """Boolean form of :meth:`_evaluate_goal`.

        Without an episode there is no resolved target, so a pick is never
        reached: an ungrounded "grasping" flag is not success (a close on air
        reports one).
        """
        return self._evaluate_goal(skill_name, episode or _Episode(skill=skill_name))[0]

    def _evaluate_goal(self, skill_name: str, episode: _Episode) -> tuple[bool, str]:
        """Check completion from physical evidence, not the policy's opinion.

        A VLA has no termination signal -- it will emit actions forever -- so
        success must be judged the same way the classical backend judges it.
        Returns ``(reached, reason)``; the reason is logged every iteration and
        becomes the failure message when the budget runs out.
        """
        if skill_name == "pick":
            return self._pick_goal(episode)
        if skill_name == "place":
            return self._place_goal(episode)
        # Open-ended motions have no perceptual goal; the iteration budget ends them.
        return False, "open-ended motion: no perceptual goal"

    def _pick_goal(self, episode: _Episode) -> tuple[bool, str]:
        target = episode.target
        if target is None:
            return False, "no resolved target; a closed gripper alone is not a grasp"
        gripper = self._gripper_state()
        if gripper is None:
            return False, "the robot reports no gripper state"
        if not gripper.is_grasping:
            return False, f"the gripper reports no grasp (width {gripper.width * 1000:.1f} mm)"

        # The object must actually be at the fingertips: measured here, a close
        # onto empty air reported is_grasping while the can had not moved.
        scene = self.vision.observe()
        obj = scene.get(target.track_id)
        if obj is None:
            return False, f"the {target.description} ({target.track_id}) is no longer perceived"
        position = np.asarray(obj.pose.position, dtype=np.float64)
        tcp = np.asarray(self.robot.tcp_pose().position, dtype=np.float64)
        distance = float(np.linalg.norm(position - tcp))
        if distance > FINGERTIP_RADIUS_M:
            return False, (
                f"the {target.description} is {distance * 1000:.0f} mm from the TCP "
                f"(held means within {FINGERTIP_RADIUS_M * 1000:.0f} mm)"
            )
        rise = float(position[2] - target.position[2])
        # Depth lane: the object must have followed the hand upward. The
        # depthless lane perceives a lifted object at the table plane, so a
        # rise cannot be measured there and is not demanded.
        if self.config.robot.gripper_feedback and rise < MIN_LIFT_M:
            return False, (
                f"the {target.description} is at the fingertips but has not been lifted "
                f"(rose {rise * 1000:.0f} mm, need {MIN_LIFT_M * 1000:.0f} mm)"
            )
        return True, (
            f"the {target.description} is held {distance * 1000:.0f} mm from the TCP "
            f"and rose {rise * 1000:.0f} mm"
        )

    def _place_goal(self, episode: _Episode) -> tuple[bool, str]:
        gripper = self._gripper_state()
        if gripper is None:
            return False, "the robot reports no gripper state"
        if not episode.opened_during_skill:
            return False, "the policy has not commanded the gripper open"
        if not self._is_open(gripper):
            return False, (
                f"the gripper is still closed or grasping (width {gripper.width * 1000:.1f} mm, "
                f"grasping={bool(gripper.is_grasping)})"
            )
        start = episode.start_gripper_width
        opened_from = f"{start * 1000:.1f} mm" if start is not None else "closed"
        return True, (
            f"released: the gripper opened from {opened_from} to {gripper.width * 1000:.1f} mm "
            f"and reports no grasp"
        )

    # ------------------------------------------------------------------

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(event, payload)

    def _instruction(
        self, skill_name: str, params: dict[str, Any], episode: _Episode | None = None
    ) -> str:
        """Render the natural-language instruction the policy is conditioned on.

        A grounded object is named by its perceived description ("blue can"),
        not the raw reference: "it" or a track id such as ``obj_004`` means
        nothing to a language-conditioned policy.
        """
        template = INSTRUCTION_TEMPLATES.get(skill_name, skill_name.replace("_", " "))
        target = params.get("target", "object")
        if episode is not None and episode.target is not None:
            target = episode.target.description
        destination = ""
        if skill_name == "place" and params.get("target"):
            name = (
                episode.destination.description
                if episode is not None and episode.destination is not None
                else params["target"]
            )
            destination = f" on the {name}"
        try:
            return template.format(
                target=target,
                direction=params.get("direction", "forward"),
                destination=destination,
            )
        except KeyError:
            return template


def _reference_error_data(exc: AmbiguousReference | ObjectNotFound) -> dict[str, Any]:
    """``SkillResult.data`` for a reference error, in the planner's contract.

    Same fields as ``mfw.skills.base._reference_error_data`` (``error_type``;
    ``candidates``/``track_ids`` for an ambiguity, ``visible`` otherwise) so the
    planner asks "which one?" identically whichever backend was routed.
    Duplicated rather than imported to keep this package off the skills layer.
    """
    data: dict[str, Any] = {"error_type": type(exc).__name__}
    phrase = getattr(exc, "phrase", None)
    if phrase is not None:
        data["phrase"] = phrase
    if isinstance(exc, AmbiguousReference):
        data["candidates"] = list(exc.candidates)
        data["track_ids"] = list(exc.track_ids)
    else:
        data["visible"] = list(getattr(exc, "visible", ()) or ())
    return data

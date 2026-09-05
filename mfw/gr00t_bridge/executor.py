"""GR00T execution backend.

Pure stdlib + NumPy at module scope. This module must never import Isaac Sim,
torch, or GR00T -- the model lives in another process entirely.

``Gr00tExecutor`` is a **peer** of ``ClassicalExecutor``, not a layer above or
below it. Both implement :class:`~mfw.core.interfaces.ISkillExecutor`, and config
picks which one runs each skill.

The closed loop, per skill invocation:

    observe -> build observation (16-frame history) -> request action chunk
    -> execute a *prefix* of the chunk through the safety filter -> re-observe

Only a prefix (``actions_executed_per_chunk`` of ``action_horizon``) is executed
before re-observing. Running all 40 steps open-loop lets prediction error compound
with nothing to correct it; the policy was trained to be queried repeatedly.

Skills the policy cannot express -- ``observe``, ``go_home``, ``stop`` -- are not
faked here. :meth:`supports` returns ``False`` for them so the planner routes them
to the classical backend, which is why a partial policy never makes a whole
capability unavailable.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mfw.config.schema import FrameworkConfig
from mfw.core.errors import PolicyError, SafetyViolation
from mfw.core.interfaces import ISkillExecutor
from mfw.core.types import SkillResult, SkillStatus
from mfw.gr00t_bridge.observation import ObservationBuilder
from mfw.gr00t_bridge.safety import ActionSafetyFilter
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["Gr00tExecutor"]

_log = get_logger("gr00t.executor")

#: Skills a vision-language-action policy can plausibly perform. Everything else
#: belongs to the deterministic backend.
POLICY_SKILLS = frozenset({"pick", "place", "move_to", "move_relative", "look_at"})

#: Language given to the policy for each skill. GR00T is language-conditioned, so
#: this is a functional input, not a label.
INSTRUCTION_TEMPLATES = {
    "pick": "pick up the {target}",
    "place": "put down the object{destination}",
    "move_to": "move the gripper to the {target}",
    "move_relative": "move the gripper {direction}",
    "look_at": "look at the {target}",
}


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
    ) -> None:
        self.client = client
        self.robot = robot
        self.vision = vision
        self.controller = controller
        self.cameras = cameras
        self.config = config
        self.memory = memory
        self.events = events

        self.observation_builder = ObservationBuilder(config.gr00t)
        self.safety = ActionSafetyFilter(
            config=config.gr00t,
            workspace_min=np.array(config.scene.workspace_min),
            workspace_max=np.array(config.scene.workspace_max),
            gripper_open_width=config.robot.gripper_open_width,
            gripper_closed_width=config.robot.gripper_closed_width,
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

    def execute(self, skill_name: str, params: dict[str, Any]) -> SkillResult:
        started = time.perf_counter()

        if not self.supports(skill_name):
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.INFEASIBLE,
                message=f"the GR00T backend does not express {skill_name!r}",
                duration_s=time.perf_counter() - started,
            )

        if not self.client.is_ready():
            try:
                self.client.connect()
            except PolicyError as exc:
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.FAILED,
                    message=f"policy server unavailable: {exc}",
                    duration_s=time.perf_counter() - started,
                )

        instruction = self._instruction(skill_name, params)
        max_iterations = int(params.get("max_iterations", 12))

        # Resolve the target so completion can be checked against the *object*,
        # not merely the gripper.
        target_track_id = None
        if skill_name in ("pick", "place") and params.get("target"):
            scene = self.vision.require_fresh_scene()
            reference = str(params["target"])
            if reference in scene.objects:
                target_track_id = reference
            else:
                matches = scene.by_label(reference)
                if matches:
                    target_track_id = matches[0].track_id

        try:
            return self._run_closed_loop(
                skill_name, instruction, max_iterations, started, target_track_id
            )
        except SafetyViolation:
            # Propagates: the state machine must see a safety breach.
            raise
        except PolicyError as exc:
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                message=f"policy error: {exc}",
                duration_s=time.perf_counter() - started,
            )

    def _run_closed_loop(
        self, skill_name: str, instruction: str, max_iterations: int, started: float,
        target_track_id: str | None = None,
    ) -> SkillResult:
        # Fresh episode: a history carried over from a previous command would show
        # the policy motion that never happened in this one.
        self._prime_history()

        clamped_actions = 0
        executed_steps = 0

        for iteration in range(1, max_iterations + 1):
            self._push_frame()
            state = self.robot.get_state()
            observation = self.observation_builder.build(state, instruction)

            chunk = _unwrap_action(self.client.predict(observation))

            # Actions come back batched as (B, horizon, D); drop the batch axis.
            # Verified against the live checkpoint: eef_9d is (1, 40, 9).
            eef_chunk = _drop_batch(np.asarray(chunk.get("eef_9d"), dtype=np.float64))
            gripper_chunk = _drop_batch(
                np.asarray(
                    chunk.get("gripper_position", np.zeros((1, eef_chunk.shape[0], 1))),
                    dtype=np.float64,
                )
            ).reshape(eef_chunk.shape[0], -1)

            if eef_chunk.ndim != 2 or eef_chunk.shape[1] != 9:
                raise PolicyError(
                    f"expected an (N, 9) eef_9d chunk after unbatching, got "
                    f"shape {eef_chunk.shape}"
                )

            # Execute a prefix, then re-observe. Open-loop execution of the whole
            # 40-step horizon compounds prediction error.
            prefix = min(self.config.gr00t.actions_executed_per_chunk, eef_chunk.shape[0])

            for step in range(prefix):
                current = self.robot.tcp_pose()
                action = self.safety.apply(
                    current_tcp=current,
                    eef_9d=eef_chunk[step],
                    gripper_command=float(gripper_chunk[step, 0]),
                    # GR00T's processor denormalises AND converts relative->absolute,
                    # so what arrives is a world-frame pose despite the embodiment
                    # tag saying "relative". See ActionSafetyFilter.apply.
                    absolute=self.config.gr00t.actions_are_absolute,
                )
                if action.was_clamped:
                    clamped_actions += 1

                if not self.controller.servo_to_pose(action.target_pose):
                    self.emit(
                        "gr00t.servo_failed",
                        {"iteration": iteration, "step": step,
                         "target": action.target_pose.to_log()},
                    )
                    break

                self._apply_gripper(action.gripper_width)
                executed_steps += 1

            self.emit(
                "gr00t.iteration",
                {
                    "skill": skill_name,
                    "iteration": iteration,
                    "instruction": instruction,
                    "executed_steps": executed_steps,
                    "clamped_actions": clamped_actions,
                },
            )

            if self._goal_reached(skill_name, target_track_id):
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.SUCCESS,
                    message=(
                        f"policy completed after {iteration} iteration(s), "
                        f"{executed_steps} step(s), {clamped_actions} clamped"
                    ),
                    duration_s=time.perf_counter() - started,
                    data={
                        "iterations": iteration,
                        "executed_steps": executed_steps,
                        "clamped_actions": clamped_actions,
                        "instruction": instruction,
                    },
                )

        return SkillResult(
            skill_name=skill_name,
            status=SkillStatus.TIMEOUT,
            message=(
                f"policy did not reach the goal in {max_iterations} iterations "
                f"({executed_steps} steps executed, {clamped_actions} clamped)"
            ),
            duration_s=time.perf_counter() - started,
            data={"executed_steps": executed_steps, "clamped_actions": clamped_actions},
        )

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

    def _apply_gripper(self, width: float) -> None:
        """Drive the gripper toward the commanded width.

        The policy emits an absolute width; the framework exposes open/close, so
        the command is thresholded at the midpoint.
        """
        midpoint = (
            self.config.robot.gripper_open_width + self.config.robot.gripper_closed_width
        ) / 2.0
        if width >= midpoint:
            self.robot.open_gripper()
        else:
            self.robot.close_gripper()

    def _goal_reached(self, skill_name: str, target_track_id: str | None = None) -> bool:
        """Check completion from physical evidence, not the policy's opinion.

        A VLA has no termination signal -- it will emit actions forever -- so
        success must be judged the same way the classical backend judges it.

        For ``pick`` that means the gripper stalled on something **and** the
        target object is still near the fingertips. The gripper flag alone is not
        enough: measured here, a close onto empty air reported ``is_grasping``
        and the executor declared success while the can had not moved a
        millimetre. Requiring the object to have followed the hand is what makes
        the claim falsifiable.
        """
        if skill_name == "pick":
            gripper = self.robot.get_gripper_state()
            if not gripper.is_grasping:
                return False
            if target_track_id is None:
                return True

            # The object must actually be at the fingertips.
            scene = self.vision.observe()
            obj = scene.get(target_track_id)
            if obj is None:
                return False
            tcp = self.robot.tcp_pose()
            distance = float(np.linalg.norm(obj.pose.position - tcp.position))
            return distance <= 0.09

        if skill_name == "place":
            gripper = self.robot.get_gripper_state()
            return gripper.width > self.config.robot.gripper_open_width * 0.7

        # Open-ended motions have no perceptual goal; the iteration budget ends them.
        return False

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(event, payload)

    def _instruction(self, skill_name: str, params: dict[str, Any]) -> str:
        """Render the natural-language instruction the policy is conditioned on."""
        template = INSTRUCTION_TEMPLATES.get(skill_name, skill_name.replace("_", " "))
        target = params.get("target", "object")
        destination = f" on the {params['target']}" if skill_name == "place" and params.get("target") else ""
        try:
            return template.format(
                target=target,
                direction=params.get("direction", "forward"),
                destination=destination,
            )
        except KeyError:
            return template

"""Abstract interfaces between framework layers.

Pure stdlib + NumPy. This module must never import Isaac Sim.

Every layer boundary in the architecture is an ABC here. Two consequences that
matter in practice:

* The classical pipeline and the GR00T policy both implement
  :class:`ISkillExecutor`. They are **peers**, selected per-skill by config.
  GR00T is not a stage between the planner and the motion planner -- it emits
  relative end-effector action chunks, so it *is* a motion generator, and
  wiring it as a middle layer would be a category error.
* Anything the simulator provides (perception, control, physics queries) sits
  behind an interface, so the planner, grasp, memory and language layers can be
  unit-tested against fakes with no simulator running.
"""

from __future__ import annotations

import abc
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mfw.core.types import (
    CameraFrame,
    GraspCandidate,
    JointState,
    Pose,
    RobotState,
    SceneGraph,
    SkillResult,
    Trajectory,
)

__all__ = [
    "ICamera",
    "IPerception",
    "IObjectDetector",
    "IPoseEstimator",
    "IGraspGenerator",
    "IGraspScorer",
    "IMotionPlanner",
    "IController",
    "IRobot",
    "IMemory",
    "ISkill",
    "ISkillExecutor",
    "IIntentParser",
    "IPolicyClient",
]


class ICamera(abc.ABC):
    """A calibrated RGB-D + segmentation sensor."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def capture(self) -> CameraFrame:
        """Return the most recent synchronised frame.

        Implementations must not step the simulation; capture reads whatever the
        renderer last produced. Stepping is the simulation layer's job.
        """

    @abc.abstractmethod
    def get_extrinsics(self) -> Pose:
        """Current camera pose in world. Re-read every call: a wrist camera moves."""


class IObjectDetector(abc.ABC):
    """Turns a camera frame into candidate object masks.

    Kept separate from :class:`IPoseEstimator` so the detector can be swapped
    (segmentation-driven, or an open-vocabulary VLM) without touching pose
    estimation.
    """

    @abc.abstractmethod
    def detect(self, frame: CameraFrame) -> list[dict[str, Any]]:
        """Return per-detection dicts with at least ``mask``, ``label``, ``confidence``."""


class IPoseEstimator(abc.ABC):
    """Estimates 6-DoF pose and extents for a detected object."""

    @abc.abstractmethod
    def estimate(
        self, frame: CameraFrame, detection: dict[str, Any]
    ) -> tuple[Pose, NDArray[np.float64]] | None:
        """Return ``(pose, extents)`` in world, or ``None`` if the evidence is too weak."""


class IPerception(abc.ABC):
    """The only component permitted to touch cameras.

    The planner depends on this interface and never on :class:`ICamera`.
    """

    @abc.abstractmethod
    def observe(self) -> SceneGraph:
        """Acquire fresh sensor data and return an updated scene graph.

        Must be safe to call before every action. The framework's rule is that
        no skill may act on a scene graph it did not just request.
        """

    @abc.abstractmethod
    def last_scene_graph(self) -> SceneGraph | None:
        """Most recent result without re-acquiring. May be stale; callers must check age."""


class IGraspGenerator(abc.ABC):
    """Synthesises grasp candidates from perceived geometry."""

    @abc.abstractmethod
    def generate(self, scene: SceneGraph, track_id: str) -> list[GraspCandidate]:
        """Propose grasps for one object. May return an empty list."""


class IGraspScorer(abc.ABC):
    """Ranks grasp candidates. Split from generation so scoring can consider
    reachability and collision, which generation deliberately does not know about."""

    @abc.abstractmethod
    def score(
        self, candidates: Sequence[GraspCandidate], scene: SceneGraph, robot_state: RobotState
    ) -> list[GraspCandidate]:
        """Return candidates with ``score`` populated, best first, infeasible ones dropped."""


class IMotionPlanner(abc.ABC):
    """Online collision-aware planning. No prerecorded trajectories."""

    @abc.abstractmethod
    def plan_to_pose(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Plan to a Cartesian TCP goal. ``None`` means no plan was found."""

    @abc.abstractmethod
    def plan_to_joint(
        self, start: JointState, goal: NDArray[np.float64], scene: SceneGraph
    ) -> Trajectory | None:
        """Plan to an explicit joint configuration."""

    @abc.abstractmethod
    def plan_cartesian_line(
        self, start: JointState, goal_pose: Pose, scene: SceneGraph
    ) -> Trajectory | None:
        """Plan a straight-line TCP motion.

        Distinct from :meth:`plan_to_pose` because grasp approach and lift must
        not curve: a curved approach sweeps the fingers through the object.
        """

    @abc.abstractmethod
    def update_collision_world(self, scene: SceneGraph) -> None:
        """Refresh obstacles from perception. Called before every plan."""


class IController(abc.ABC):
    """Executes trajectories and gripper commands on the articulation.

    The only layer permitted to write joint targets. Nothing in the framework
    may set object poses -- manipulation is via contact forces only.
    """

    @abc.abstractmethod
    def follow_trajectory(self, trajectory: Trajectory) -> bool:
        """Execute to completion. ``False`` if aborted or tracking error exceeded limits."""

    @abc.abstractmethod
    def servo_to_pose(self, target: Pose) -> bool:
        """Single closed-loop Cartesian step, for reactive/VLA control."""

    @abc.abstractmethod
    def hold_position(self) -> None:
        """Command the current configuration as the target, resisting gravity."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Halt motion immediately, keeping the arm actively controlled."""


class IRobot(abc.ABC):
    """Robot abstraction: state, kinematics and gripper.

    The framework talks to this, never to an Isaac ``SingleManipulator``
    directly, so the arm can be swapped without touching skills or planning.
    """

    @property
    @abc.abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Arm joints only, in articulation order. Excludes fingers."""

    @abc.abstractmethod
    def get_state(self) -> RobotState: ...

    @abc.abstractmethod
    def tcp_pose(self) -> Pose:
        """Pose of the true tool centre point: the midpoint between the fingertips.

        Explicitly **not** a finger link. Isaac Sim's stock Franka configures its
        end-effector prim as ``panda_rightfinger``, which is offset from the
        actual grasp point; using it directly makes every grasp miss by that
        offset while the logs still claim success.
        """

    @abc.abstractmethod
    def forward_kinematics(self, joint_positions: NDArray[np.float64]) -> Pose:
        """TCP pose for a hypothetical configuration, without moving the robot."""

    @abc.abstractmethod
    def inverse_kinematics(
        self, target: Pose, seed: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64] | None:
        """Joint solution for a TCP target, or ``None`` if unreachable."""

    @abc.abstractmethod
    def open_gripper(self) -> None: ...

    @abc.abstractmethod
    def close_gripper(self) -> None: ...


class IMemory(abc.ABC):
    """Persistent robot and scene state across commands.

    This is what makes "place it" resolvable: the referent of a pronoun is
    whatever the memory says is currently held or was most recently discussed.
    """

    @abc.abstractmethod
    def update_scene(self, scene: SceneGraph) -> None: ...

    @abc.abstractmethod
    def update_robot(self, state: RobotState) -> None: ...

    @abc.abstractmethod
    def set_held_object(self, track_id: str | None) -> None: ...

    @abc.abstractmethod
    def get_held_object(self) -> str | None: ...

    @abc.abstractmethod
    def resolve_reference(self, phrase: str) -> str | None:
        """Resolve a natural-language referent ("it", "the can") to a ``track_id``."""

    @abc.abstractmethod
    def record_result(self, result: SkillResult) -> None: ...


class ISkill(abc.ABC):
    """One atomic, independently executable action.

    Hard rule: a skill must never invoke another skill. ``Pick`` picks and then
    holds; it does not place. Composition is the planner's job, and the planner
    only composes what the user actually asked for.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def validate(self, params: dict[str, Any]) -> str | None:
        """Return an error string if the parameters are unusable, else ``None``.

        Runs before any motion so infeasible commands fail fast and cheaply.
        """

    @abc.abstractmethod
    def execute(self, params: dict[str, Any]) -> SkillResult: ...


class ISkillExecutor(abc.ABC):
    """A backend that carries out skills.

    Two implementations, chosen per-skill by config:

    * ``ClassicalExecutor`` -- perception, grasp synthesis, RRT/RMPflow, PhysX.
      Deterministic and inspectable; the default.
    * ``Gr00tExecutor`` -- GR00T N1.7 closed-loop, consuming images plus language
      and emitting relative end-effector chunks through a safety filter.

    Both satisfy this interface, so switching backends is configuration rather
    than a refactor.
    """

    @property
    @abc.abstractmethod
    def backend_name(self) -> str: ...

    @abc.abstractmethod
    def supports(self, skill_name: str) -> bool:
        """Whether this backend can execute the named skill."""

    @abc.abstractmethod
    def execute(self, skill_name: str, params: dict[str, Any]) -> SkillResult: ...


@runtime_checkable
class IPolicyClient(Protocol):
    """Transport to an out-of-process policy server.

    GR00T N1.7 requires Python 3.12 while Isaac Sim 5.1 ships 3.11, and its
    CUDA/attention dependencies are Linux-gated. The policy therefore *cannot*
    share the simulator's interpreter -- it is always a separate process, and
    this Protocol is the only thing that crosses that boundary. Keeping it a
    Protocol lets a mock server satisfy it with no inheritance.
    """

    def is_ready(self) -> bool: ...

    def predict(self, observation: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Send one observation, receive an action chunk.

        For the ``oxe_droid`` embodiment the reply contains ``eef_9d``
        (relative), ``gripper_position`` (absolute) and ``joint_position``
        (relative), each with a 40-step horizon.
        """

    def close(self) -> None: ...


class IIntentParser(abc.ABC):
    """Natural language to a structured, *single* intent.

    Must never expand one utterance into a multi-step plan. "Pick the can"
    yields exactly one ``Pick`` intent -- never ``Pick`` followed by ``Place``.
    """

    @abc.abstractmethod
    def parse(self, utterance: str, context: dict[str, Any]) -> dict[str, Any]:
        """Return ``{"skill": str, "params": dict, "confidence": float}``."""

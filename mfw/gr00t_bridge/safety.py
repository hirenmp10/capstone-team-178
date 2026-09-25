"""Safety filter for policy-generated actions.

Pure NumPy. This module must never import Isaac Sim.

A learned policy has no notion of the robot's limits: it emits whatever its
training distribution suggests, and an out-of-distribution observation can produce
an arbitrarily large end-effector delta. Nothing downstream would question it --
IK would happily solve for a pose half a metre away and the controller would drive
there at full speed.

So every action crosses this filter before it can become motion. The distinction
that matters:

* **Clamping** a slightly-too-large delta is correct. The policy's *direction* is
  usually right even when its magnitude overshoots, so limiting the step keeps the
  behaviour while bounding the risk.
* **Rejecting** is correct when the action is not merely large but invalid --
  non-finite values, or a target outside the workspace. Clamping those would
  fabricate a plausible command out of a meaningless one.

The support surface is part of the envelope
-------------------------------------------
``scene.workspace_min`` is a box whose z-min is the *floor* (0.0), while the
table top in ``configs/default.yaml`` sits at z = 0.4. Clamping an absolute
target into that box was described as "safe by construction", but it is not:
a target at z = 0.005 is inside the box and 0.4 m *inside the table*.
``controller.servo_to_pose`` is IK plus one interpolation step with no
collision check, so nothing downstream would have stopped it. Measured with the
old mock (which emitted deltas that the executor read as absolute poses) the
servo targets walked from [0.415, 0, 0.415] to [0.2, 0, 0.005] -- through the
table -- with every step reported as merely "clamped".

So the filter takes the support height and raises the z floor to
``support_height + support_clearance`` (the TCP sits between the fingertip
pads, so a small clearance keeps the pads off the surface while still letting a
top-down grasp close on a 5 cm block). Absolute targets are clamped onto it;
delta targets that cross it raise, exactly like any other workspace breach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import Gr00tConfig
from mfw.core.errors import SafetyViolation
from mfw.core.types import Frame, Pose
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = ["ActionSafetyFilter", "FilteredAction", "DEFAULT_SUPPORT_CLEARANCE_M"]

_log = get_logger("gr00t.safety")

#: Metres the TCP must stay above the support surface. The TCP is the midpoint
#: between the fingertip pads, which extend roughly a centimetre below it, so
#: 15 mm keeps the pads off the table while leaving a top-down grasp on a
#: 50 mm cube (TCP ~25 mm above the table) reachable.
DEFAULT_SUPPORT_CLEARANCE_M = 0.015


@dataclass(frozen=True)
class FilteredAction(dict):
    """Result of filtering: the pose to command, and what was altered."""

    def __init__(
        self,
        target_pose: Pose,
        gripper_width: float,
        translation_clamped: bool,
        rotation_clamped: bool,
        original_translation: float,
        original_rotation: float,
    ):
        super().__init__(
            target_pose=target_pose,
            gripper_width=gripper_width,
            translation_clamped=translation_clamped,
            rotation_clamped=rotation_clamped,
            original_translation=original_translation,
            original_rotation=original_rotation,
        )

    @property
    def target_pose(self) -> Pose:
        return self["target_pose"]

    @property
    def gripper_width(self) -> float:
        return self["gripper_width"]

    @property
    def was_clamped(self) -> bool:
        return self["translation_clamped"] or self["rotation_clamped"]


class ActionSafetyFilter:
    """Converts a policy action (absolute pose or delta) into a bounded absolute target."""

    def __init__(
        self,
        config: Gr00tConfig,
        workspace_min: NDArray[np.float64],
        workspace_max: NDArray[np.float64],
        gripper_open_width: float,
        gripper_closed_width: float,
        support_height: float | None = None,
        support_clearance: float = DEFAULT_SUPPORT_CLEARANCE_M,
    ) -> None:
        """``support_height`` is the z of the surface the arm works on.

        ``None`` keeps the bare workspace box (the historical behaviour, and
        correct only when the box's z-min already sits above the surface).
        """
        config.validate()
        self.config = config
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64).copy()
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.gripper_open_width = float(gripper_open_width)
        self.gripper_closed_width = float(gripper_closed_width)
        self.support_height = None if support_height is None else float(support_height)
        if self.support_height is not None:
            floor = self.support_height + float(support_clearance)
            if floor >= self.workspace_max[2]:
                raise SafetyViolation(
                    f"support floor z={floor:.3f} (surface {self.support_height:.3f} + "
                    f"clearance {support_clearance:.3f}) is at or above workspace z-max "
                    f"{self.workspace_max[2]:.3f}; no policy target could be valid"
                )
            self.workspace_min[2] = max(self.workspace_min[2], floor)

    @property
    def floor_z(self) -> float:
        """Lowest z a policy target may command (workspace z-min or support floor)."""
        return float(self.workspace_min[2])

    def apply(
        self,
        current_tcp: Pose,
        eef_9d: NDArray[np.float64],
        gripper_command: float,
        absolute: bool = True,
    ) -> FilteredAction:
        """Turn one ``eef_9d`` action into a safe TCP target.

        ``absolute=True`` (the default, and what GR00T actually produces) treats
        the action as a **world-frame pose**: ``[x, y, z, rot6d(6)]``. The step is
        then measured as the distance from the current TCP.

        This distinction cost a full debugging cycle and is worth stating plainly.
        The embodiment tag is ``oxe_droid_relative_eef_relative_joint``, so the
        obvious reading is that actions are relative deltas -- but "relative" names
        the *training* representation. NVIDIA's processor performs the
        relative-to-absolute conversion during postprocessing, so ``get_action``
        hands back denormalised absolute poses.

        Composing those onto the current pose as if they were deltas turned an
        11 mm move into a 0.61 m one, which this filter then clamped -- every
        single action, a 100% clamp rate that looked exactly like a policy
        failing on an unfamiliar scene. It was not: the model was emitting a
        perfectly sensible 8 cm reach-and-lift.

        ``absolute=False`` keeps the delta interpretation for policies that do
        emit deltas.
        """
        action = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
        if action.shape[0] != 9:
            raise SafetyViolation(f"expected a 9-element eef_9d action, got {action.shape[0]}")
        if not np.all(np.isfinite(action)):
            # Not clampable: a NaN carries no usable direction.
            raise SafetyViolation("policy action contains non-finite values")

        # Measure the *step* -- how far this action asks the TCP to move -- which
        # for an absolute pose is its distance from where the TCP is now.
        if absolute:
            delta_translation = action[:3] - np.asarray(current_tcp.position, dtype=np.float64)
        else:
            delta_translation = action[:3]
        original_translation = float(np.linalg.norm(delta_translation))

        translation_clamped = False
        if original_translation > self.config.max_relative_translation:
            delta_translation = (
                delta_translation / original_translation * self.config.max_relative_translation
            )
            translation_clamped = True

        try:
            action_rotation = tf.rot6d_to_matrix(action[3:])
        except ValueError as exc:
            raise SafetyViolation(f"policy action has an invalid rotation: {exc}") from exc

        rotation_clamped = False
        if absolute:
            # action_rotation is the *target* orientation. The step is the
            # rotation needed to get there from the current one; clamping the
            # target's own angle-from-identity would be meaningless.
            current_rotation = current_tcp.rotation_matrix()
            step_rotation = action_rotation @ current_rotation.T
            original_rotation = _rotation_angle(step_rotation)
            if original_rotation > self.config.max_relative_rotation:
                limited = _scale_rotation(
                    step_rotation, self.config.max_relative_rotation / original_rotation
                )
                delta_rotation = tf.orthonormalize(limited @ current_rotation)
                rotation_clamped = True
            else:
                delta_rotation = action_rotation
        else:
            delta_rotation = action_rotation
            original_rotation = _rotation_angle(delta_rotation)
            if original_rotation > self.config.max_relative_rotation:
                delta_rotation = _scale_rotation(
                    delta_rotation, self.config.max_relative_rotation / original_rotation
                )
                rotation_clamped = True

        if absolute:
            # The action already names a world-frame pose. Apply the clamped step
            # from the current position, and take the orientation as given --
            # rotation was clamped above against the current orientation.
            position = np.asarray(current_tcp.position, dtype=np.float64) + delta_translation
            target_matrix = np.eye(4)
            target_matrix[:3, :3] = delta_rotation
            target_matrix[:3, 3] = position
        else:
            # Delta semantics: compose in the TCP frame.
            delta = np.eye(4)
            delta[:3, :3] = delta_rotation
            delta[:3, 3] = delta_translation
            target_matrix = current_tcp.as_matrix() @ delta

        target = Pose.from_matrix(target_matrix, Frame.WORLD)

        outside = np.any(target.position < self.workspace_min) or np.any(
            target.position > self.workspace_max
        )
        workspace_clamped = False
        if outside:
            if absolute:
                # Clamp an absolute target onto the envelope rather than aborting.
                #
                # The envelope IS the safety property -- the workspace box with
                # its z-min raised to the support floor (see the module
                # docstring: the bare box contains the table) -- and a point
                # clamped into it stays out of the table -- unlike a *delta*, where clamping
                # invents a direction the policy never chose. Aborting instead is
                # disproportionate for a policy that steers continuously: measured
                # here, a target 4 mm outside X-min killed an entire 30-iteration
                # run on its first step, from a home posture that sits only 30 mm
                # inside the boundary.
                #
                # Persistent clamping still means something is wrong, so the
                # caller counts it and can give up on its own terms.
                clamped_position = np.clip(
                    target.position, self.workspace_min, self.workspace_max
                )
                target = Pose(clamped_position, target.quat, Frame.WORLD)
                workspace_clamped = True
                _log.debug(
                    "Clamped policy target into the workspace: %s",
                    np.round(clamped_position, 3).tolist(),
                )
            else:
                raise SafetyViolation(
                    f"policy target {np.round(target.position, 3).tolist()} is outside the "
                    f"workspace {np.round(self.workspace_min, 3).tolist()} to "
                    f"{np.round(self.workspace_max, 3).tolist()}"
                )

        width = float(
            np.clip(gripper_command, self.gripper_closed_width, self.gripper_open_width)
        )

        if translation_clamped or rotation_clamped:
            _log.debug(
                "Clamped policy action: translation %.4f -> %.4f m, rotation %.3f -> %.3f rad",
                original_translation,
                min(original_translation, self.config.max_relative_translation),
                original_rotation,
                min(original_rotation, self.config.max_relative_rotation),
            )

        return FilteredAction(
            target_pose=target,
            gripper_width=width,
            translation_clamped=translation_clamped or workspace_clamped,
            rotation_clamped=rotation_clamped,
            original_translation=original_translation,
            original_rotation=original_rotation,
        )


def _rotation_angle(rotation: NDArray[np.float64]) -> float:
    """Geodesic angle of a rotation matrix, in radians."""
    trace = float(np.trace(rotation))
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def _scale_rotation(rotation: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    """Scale a rotation toward identity, preserving its axis.

    Interpolates along the geodesic rather than scaling matrix entries: scaling
    the entries of a rotation matrix does not produce a rotation.
    """
    angle = _rotation_angle(rotation)
    if angle < 1e-9:
        return np.eye(3)

    quat = tf.matrix_to_quat(rotation)
    axis = quat[1:]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.eye(3)
    axis = axis / axis_norm

    scaled_angle = angle * factor
    half = scaled_angle / 2.0
    scaled_quat = np.concatenate([[np.cos(half)], axis * np.sin(half)])
    return tf.quat_to_matrix(scaled_quat)

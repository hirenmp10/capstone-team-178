"""Grasp candidate synthesis from perceived geometry.

Pure NumPy. This module must never import Isaac Sim, so grasp geometry is
unit-testable without a simulator.

No grasp pose appears as a literal anywhere. Candidates are derived from the
object's oriented bounding box: the gripper must close across a pair of opposing
faces, so each box axis whose extent the fingers can span gives a family of
antipodal grasps, one per perpendicular approach direction, sampled along the
remaining free axis.

Gripper frame convention (measured from the Franka asset via Lula FK):

* fingers separate along the hand's local **Y** -- ``panda_leftfingertip`` sits at
  ``[0, +0.025, 0.1034]`` and ``panda_rightfingertip`` at ``[0, -0.025, 0.1034]``
  in the hand frame;
* the gripper approaches along its local **+Z**, out of the palm.

So a grasp pose is fully determined by choosing which world direction the
fingers close along (local Y) and which way the hand travels (local Z).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import GraspConfig
from mfw.core.types import BoundingBox3D, Frame, GraspCandidate, ObjectHypothesis, Pose
from mfw.utils import transforms as tf

__all__ = ["generate_grasp_candidates", "build_grasp_pose", "WORLD_DOWN"]

WORLD_DOWN = np.array([0.0, 0.0, -1.0])

_EPS = 1e-9


def build_grasp_pose(
    grasp_point: NDArray[np.float64],
    closing_axis: NDArray[np.float64],
    approach_axis: NDArray[np.float64],
) -> Pose:
    """Construct a TCP pose from a closing direction and an approach direction.

    ``closing_axis`` becomes the gripper's local Y (fingers separate along it) and
    ``approach_axis`` its local Z (direction of travel). The two are
    orthogonalised, since a box axis pair is only exactly perpendicular for a
    perfectly fitted box.
    """
    z_axis = np.asarray(approach_axis, dtype=np.float64)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < _EPS:
        raise ValueError("approach_axis is degenerate")
    z_axis = z_axis / z_norm

    y_axis = np.asarray(closing_axis, dtype=np.float64)
    # Remove any component along the approach direction so the frame is
    # orthonormal without changing which way the fingers close.
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < _EPS:
        raise ValueError("closing_axis is parallel to approach_axis; no valid grasp frame")
    y_axis = y_axis / y_norm

    x_axis = np.cross(y_axis, z_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return Pose(np.asarray(grasp_point, dtype=np.float64), tf.matrix_to_quat(rotation), Frame.WORLD)


def generate_grasp_candidates(
    obj: ObjectHypothesis, config: GraspConfig
) -> list[GraspCandidate]:
    """Enumerate antipodal grasps for one perceived object.

    Returns candidates with ``score`` unset (0.0); ranking is the scorer's job,
    because reachability and collision depend on the robot and the rest of the
    scene, which generation deliberately knows nothing about.
    """
    bbox = obj.bbox
    rotation = bbox.center.rotation_matrix()
    extents = np.asarray(bbox.extents, dtype=np.float64)
    center = np.asarray(bbox.center.position, dtype=np.float64)

    usable_width = config.max_grasp_width - config.finger_width_margin
    candidates: list[GraspCandidate] = []

    for closing_index in range(3):
        width = float(extents[closing_index])
        # The fingers have to span this dimension. Rejecting here, before any
        # IK or collision work, is what keeps generation cheap.
        if not (config.min_grasp_width <= width <= usable_width):
            continue

        closing_axis = rotation[:, closing_index]

        for approach_index in range(3):
            if approach_index == closing_index:
                continue

            free_index = 3 - closing_index - approach_index
            for approach_sign in (1.0, -1.0):
                # The hand travels *toward* the object, so the approach axis
                # points from the standoff position into the object: opposite the
                # face normal it comes in through.
                approach_axis = -approach_sign * rotation[:, approach_index]

                for offset in _offsets_along(extents[free_index], config):
                    grasp_point = center + rotation[:, free_index] * offset
                    try:
                        pose = build_grasp_pose(grasp_point, closing_axis, approach_axis)
                    except ValueError:
                        continue

                    pregrasp_position = grasp_point - approach_axis * config.approach_offset
                    pregrasp = Pose(pregrasp_position, pose.quat, Frame.WORLD)

                    candidates.append(
                        GraspCandidate(
                            pose=pose,
                            pregrasp_pose=pregrasp,
                            approach_axis=approach_axis,
                            width=width,
                            score=0.0,
                            target_track_id=obj.track_id,
                            scores_breakdown={},
                        )
                    )

    return candidates[: config.max_candidates]


def _offsets_along(extent: float, config: GraspConfig) -> list[float]:
    """Sample grasp positions along the axis the gripper does not constrain.

    A tall object can be gripped near its middle or nearer its top; those are
    genuinely different grasps with different collision and stability
    properties. Offsets stay within the middle 60% of the extent so the fingers
    never hang off an end, where they would slip.
    """
    if extent < 0.04:
        # Too short for the choice to matter; one central grasp is enough.
        return [0.0]

    limit = 0.3 * extent
    num_samples = max(1, min(3, int(config.num_orientation_samples // 6) + 1))
    if num_samples == 1:
        return [0.0]
    return list(np.linspace(-limit, limit, num_samples))


def support_clearance(candidate: GraspCandidate, support_height: float) -> float:
    """Vertical clearance between the fingertips at closure and the support surface.

    Negative means the fingers would have to pass through the table to reach the
    grasp. Used by the scorer as a hard rejection, since no amount of other
    quality compensates for driving the hand into the worktop.
    """
    return float(candidate.pose.position[2] - support_height)

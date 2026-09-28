"""Top-down grasp synthesis for a jaw with no wrist roll.

Pure NumPy. This module must never import Isaac Sim, torch or transformers.

The sim generator enumerates antipodal grasps over every box axis because a
7-DoF wrist can meet any of them. This arm cannot: its jaw closes along a
direction fixed by how the gripper is bolted on (``hardware.arm.jaw_axis``,
tangential or radial to the base->object ray), and the only orientation
freedom is the tool pitch in the arm plane. So there is exactly one grasp
family per object -- at its centre, from above, at one of a few pitches -- and
the honest question is not "which grasp" but "does the jaw fit across the
object *as it lies*".

Width is the object's chord along the closing axis through its centre, not
the shadow of the box on that axis. For a 140 x 19 mm marker lying 15 degrees
off the jaw line the shadow is 56 mm (rejected) while the chord the fingers
actually meet is 19.7 mm (fine). The chord is exact for a box.

Reachability pre-filter: the scorer checks IK for the grasp and pregrasp, but
not for the **lift**, and on a 30 cm arm the vertical lift from a far grasp
is what runs out of reach first (measured on the placeholder geometry: a
marker at 19 cm is graspable straight down but cannot be lifted 6 cm without
tilting the tool). Rejecting such candidates here, before the scorer ranks
them, is what turns "pick fails at step 7" into "the 75-degree pitch is
chosen". Without a kinematics model the filter is skipped.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import GraspConfig, HardwareConfig
from mfw.core.interfaces import IGraspGenerator
from mfw.core.types import Frame, GraspCandidate, ObjectHypothesis, Pose, SceneGraph
from mfw.grasp.generator import build_grasp_pose
from mfw.utils.logging import get_logger

__all__ = ["TopDownGraspGenerator", "chord_width"]

_log = get_logger("hardware.grasp")

_EPS = 1e-9
_MIN_GRASP_ABOVE_SUPPORT_M = 0.005
"""Never plan the jaw midpoint closer than this to the table."""


def chord_width(
    extents: Sequence[float], rotation: NDArray[np.float64], direction: NDArray[np.float64]
) -> float:
    """Length of the box's chord through its centre along ``direction`` (world).

    ``rotation`` has the box axes as columns. The chord is the distance between
    the two points where the centre line exits the box.
    """
    half = np.asarray(extents, dtype=np.float64) / 2.0
    u = rotation.T @ (np.asarray(direction, dtype=np.float64) / max(float(np.linalg.norm(direction)), _EPS))
    t = math.inf
    for i in range(3):
        if abs(u[i]) > _EPS:
            t = min(t, half[i] / abs(u[i]))
    return 0.0 if not math.isfinite(t) else float(2.0 * t)


class TopDownGraspGenerator(IGraspGenerator):
    """One candidate per configured pitch, at the object centre, jaw axis fixed."""

    def __init__(
        self,
        config: GraspConfig,
        hardware: HardwareConfig,
        support_height: float = 0.0,
        kinematics: Any | None = None,
    ) -> None:
        config.validate()
        hardware.validate()
        self.config = config
        self.hardware = hardware
        self.support_height = float(support_height)
        self.kinematics = kinematics
        self.jaw_axis = hardware.arm.jaw_axis
        self.pitches_deg: tuple[float, ...] = tuple(float(p) for p in hardware.grasp_pitch_angles_deg)

    # ------------------------------------------------------------------

    @staticmethod
    def _frame(bearing: float, tilt: float, jaw_axis: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(approach, closing)`` for tool tilt ``tilt`` rad from vertical, outward."""
        radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        tangential = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        approach = math.sin(tilt) * radial - math.cos(tilt) * up
        if jaw_axis == "tangential":
            closing = tangential
        else:
            # In the arm plane, perpendicular to the tool, forward when vertical.
            closing = math.cos(tilt) * radial + math.sin(tilt) * up
        return approach, closing

    def _reachable(self, pose: Pose, pregrasp: Pose) -> bool:
        kin = self.kinematics
        if kin is None:
            return True
        if kin.ik(pose) is None or kin.ik(pregrasp) is None:
            return False
        lift = Pose(pose.position + np.array([0.0, 0.0, self.config.lift_height]), pose.quat, Frame.WORLD)
        return kin.ik(lift) is not None

    def candidates_for(self, obj: ObjectHypothesis) -> list[GraspCandidate]:
        """Candidates for one hypothesis, most vertical pitch first, score unset."""
        centre = np.asarray(obj.bbox.center.position, dtype=np.float64)
        rotation = obj.bbox.center.rotation_matrix()
        extents = np.asarray(obj.bbox.extents, dtype=np.float64)
        bearing = math.atan2(float(centre[1]), float(centre[0]))
        usable = self.config.max_grasp_width - self.config.finger_width_margin

        grasp_point = centre.copy()
        grasp_point[2] = max(float(centre[2]), self.support_height + _MIN_GRASP_ABOVE_SUPPORT_M)

        out: list[GraspCandidate] = []
        rejected: dict[str, int] = {}
        for pitch_deg in self.pitches_deg:
            tilt = math.radians(90.0 - pitch_deg)
            approach, closing = self._frame(bearing, tilt, self.jaw_axis)
            width = chord_width(extents, rotation, closing)
            if not (self.config.min_grasp_width <= width <= usable):
                rejected["width"] = rejected.get("width", 0) + 1
                continue
            try:
                pose = build_grasp_pose(grasp_point, closing, approach)
            except ValueError:
                rejected["degenerate"] = rejected.get("degenerate", 0) + 1
                continue
            pregrasp = Pose(grasp_point - approach * self.config.approach_offset, pose.quat, Frame.WORLD)
            if not self._reachable(pose, pregrasp):
                rejected["unreachable_with_lift"] = rejected.get("unreachable_with_lift", 0) + 1
                continue
            out.append(
                GraspCandidate(
                    pose=pose,
                    pregrasp_pose=pregrasp,
                    approach_axis=approach,
                    width=width,
                    score=0.0,
                    target_track_id=obj.track_id,
                    scores_breakdown={"pitch_deg": float(pitch_deg)},
                )
            )
        if rejected:
            _log.debug("Top-down candidates for %s: %d kept, rejected %s", obj.track_id, len(out), rejected)
        return out[: self.config.max_candidates]

    def generate(self, scene: SceneGraph, track_id: str) -> list[GraspCandidate]:
        obj = scene.get(track_id)
        if obj is None:
            return []
        return self.candidates_for(obj)

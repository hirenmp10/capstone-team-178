"""Canonical data types exchanged between framework layers.

Pure NumPy + stdlib. This module must never import Isaac Sim.

These dataclasses are the contract between modules. The planner never touches a
camera; it receives :class:`SceneGraph`. The controller never touches the
perception stack; it receives :class:`Trajectory`. Keeping the seams as frozen
dataclasses is what makes layers independently testable and swappable.

Every pose carries an explicit :class:`Frame`. Silent frame mismatches are the
single most common source of manipulation bugs, so frames are data, not
convention.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from mfw.utils import transforms as tf

__all__ = [
    "Frame",
    "Pose",
    "CameraIntrinsics",
    "CameraFrame",
    "PointCloud",
    "BoundingBox3D",
    "ObjectHypothesis",
    "SceneGraph",
    "GraspCandidate",
    "JointState",
    "Waypoint",
    "Trajectory",
    "GripperState",
    "RobotState",
    "SkillStatus",
    "SkillResult",
]


class Frame(str, Enum):
    """Named reference frames.

    ``str``-valued so frames serialise directly into JSONL logs and config.
    """

    WORLD = "world"
    ROBOT_BASE = "robot_base"
    TCP = "tcp"
    """Tool centre point: the midpoint between the fingertips. NOT a finger link."""
    WRIST_CAMERA = "wrist_camera"
    EXTERIOR_CAMERA = "exterior_camera"
    OBJECT = "object"


@dataclass(frozen=True)
class Pose:
    """A rigid pose in an explicitly named frame.

    ``quat`` is scalar-first ``(w, x, y, z)`` to match Isaac Sim.
    """

    position: NDArray[np.float64]
    quat: NDArray[np.float64]
    frame: Frame = Frame.WORLD

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64).reshape(-1)
        quat = np.asarray(self.quat, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"position must be (3,), got {pos.shape}")
        if quat.shape != (4,):
            raise ValueError(f"quat must be (4,), got {quat.shape}")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            raise ValueError("quat has zero norm")
        # Normalise on construction so downstream code never has to defend
        # against a drifted quaternion, and freeze the arrays: these objects are
        # passed across layers and must not be mutated in place by a consumer.
        pos.setflags(write=False)
        quat = quat / norm
        quat.setflags(write=False)
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "quat", quat)

    @classmethod
    def identity(cls, frame: Frame = Frame.WORLD) -> "Pose":
        return cls(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), frame)

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.float64], frame: Frame = Frame.WORLD) -> "Pose":
        pos, quat = tf.matrix_to_pose(matrix)
        return cls(pos, quat, frame)

    def as_matrix(self) -> NDArray[np.float64]:
        """4x4 homogeneous transform of this pose."""
        return tf.make_transform(self.position, self.quat)

    def rotation_matrix(self) -> NDArray[np.float64]:
        return tf.quat_to_matrix(self.quat)

    def transformed_by(self, parent: "Pose") -> "Pose":
        """Compose: express this pose (given relative to ``parent``) in ``parent``'s frame.

        Used for chains like wrist-camera-in-TCP composed with TCP-in-world.
        """
        return Pose.from_matrix(parent.as_matrix() @ self.as_matrix(), parent.frame)

    def inverse(self) -> "Pose":
        return Pose.from_matrix(tf.invert_transform(self.as_matrix()), self.frame)

    def translation_distance(self, other: "Pose") -> float:
        """Euclidean distance, guarding against cross-frame comparison."""
        self._assert_same_frame(other)
        return float(np.linalg.norm(self.position - other.position))

    def angular_distance(self, other: "Pose") -> float:
        """Geodesic orientation error in radians, guarding against cross-frame comparison."""
        self._assert_same_frame(other)
        return tf.quat_angular_distance(self.quat, other.quat)

    def _assert_same_frame(self, other: "Pose") -> None:
        if self.frame != other.frame:
            raise ValueError(
                f"Cannot compare poses across frames: {self.frame.value} vs {other.frame.value}. "
                "Transform one into the other's frame first."
            )

    def to_eef_9d(self) -> NDArray[np.float64]:
        """Encode as GR00T's ``eef_9d`` state: ``[xyz(3), rot6d(6)]``.

        This is the exact layout declared for ``oxe_droid_relative_eef_relative_joint``
        in GR00T's embodiment config, so it is an external contract.
        """
        return np.concatenate([self.position, tf.matrix_to_rot6d(self.rotation_matrix())])

    @classmethod
    def from_eef_9d(cls, vec: Sequence[float], frame: Frame = Frame.WORLD) -> "Pose":
        """Decode GR00T's ``eef_9d`` back into a pose."""
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.shape != (9,):
            raise ValueError(f"eef_9d must be (9,), got {arr.shape}")
        return cls(arr[:3], tf.matrix_to_quat(tf.rot6d_to_matrix(arr[3:])), frame)

    def to_log(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "quat": self.quat.tolist(),
            "frame": self.frame.value,
        }


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus image size.

    Isaac Sim configures cameras by focal length and aperture in millimetres;
    this is the derived pixel-space form the perception maths actually needs.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> NDArray[np.float64]:
        """The 3x3 ``K`` matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CameraFrame:
    """One synchronised capture from a single camera.

    ``depth`` is metric depth in metres along the camera's optical axis (Z),
    not radial range. ``segmentation`` maps pixels to instance ids, with
    ``seg_id_to_label`` giving the human-readable mapping supplied by the
    renderer.
    """

    camera_name: str
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32] | None
    segmentation: NDArray[np.int32] | None
    seg_id_to_label: Mapping[int, str]
    """Segmentation id -> semantic *class* ("can", "block"). What language
    grounding matches against."""
    seg_id_to_prim: Mapping[int, str]
    """Segmentation id -> USD prim path. Distinguishes two instances of the same
    class, which a class label alone cannot. Used for instance tracking, never
    for locating an object."""
    intrinsics: CameraIntrinsics
    extrinsics: Pose
    """Camera pose in world. Optical convention: +Z forward, +X right, +Y down."""
    sim_time: float
    step_index: int

    def to_log(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "rgb_shape": list(self.rgb.shape),
            "has_depth": self.depth is not None,
            "has_segmentation": self.segmentation is not None,
            "num_seg_labels": len(self.seg_id_to_label),
            "extrinsics": self.extrinsics.to_log(),
            "sim_time": self.sim_time,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class PointCloud:
    """Deprojected points with optional per-point colour and instance id."""

    points: NDArray[np.float64]
    frame: Frame
    colors: NDArray[np.uint8] | None = None
    seg_ids: NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {pts.shape}")
        object.__setattr__(self, "points", pts)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def subset(self, mask: NDArray[np.bool_]) -> "PointCloud":
        return PointCloud(
            points=self.points[mask],
            frame=self.frame,
            colors=None if self.colors is None else self.colors[mask],
            seg_ids=None if self.seg_ids is None else self.seg_ids[mask],
        )


@dataclass(frozen=True)
class BoundingBox3D:
    """Oriented bounding box: centre pose plus full extents along its own axes."""

    center: Pose
    extents: NDArray[np.float64]

    def __post_init__(self) -> None:
        ext = np.asarray(self.extents, dtype=np.float64).reshape(-1)
        if ext.shape != (3,):
            raise ValueError(f"extents must be (3,), got {ext.shape}")
        if np.any(ext < 0.0):
            raise ValueError(f"extents must be non-negative, got {ext.tolist()}")
        object.__setattr__(self, "extents", ext)

    @property
    def volume(self) -> float:
        return float(np.prod(self.extents))

    @property
    def min_extent(self) -> float:
        """Smallest side length; the axis a parallel gripper is most likely to span."""
        return float(np.min(self.extents))

    def to_log(self) -> dict[str, Any]:
        return {"center": self.center.to_log(), "extents": self.extents.tolist()}


@dataclass
class ObjectHypothesis:
    """A perceived object. Never a ground-truth handle.

    ``track_id`` is assigned by the tracker and is stable across frames; it is
    what the memory layer resolves pronouns against. ``label`` is whatever the
    detector produced and may be empty, so nothing downstream may branch on a
    specific label string.
    """

    track_id: str
    label: str
    pose: Pose
    bbox: BoundingBox3D
    confidence: float
    num_points: int
    last_seen_sim_time: float
    last_seen_step: int
    seg_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "pose": self.pose.to_log(),
            "bbox": self.bbox.to_log(),
            "confidence": self.confidence,
            "num_points": self.num_points,
            "last_seen_sim_time": self.last_seen_sim_time,
            "seg_id": self.seg_id,
        }


@dataclass
class SceneGraph:
    """Structured scene handed to the planner. The planner sees only this.

    Deliberately excludes raw images: if the planner could reach pixels it would
    eventually start making perception decisions, which is the coupling this
    architecture is designed to prevent.
    """

    objects: dict[str, ObjectHypothesis]
    sim_time: float
    step_index: int
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    """Triples of ``(subject_track_id, predicate, object_track_id)``, e.g. ``on_top_of``."""

    def get(self, track_id: str) -> ObjectHypothesis | None:
        return self.objects.get(track_id)

    def by_label(self, label: str) -> list[ObjectHypothesis]:
        """Resolve a spoken description to objects, most confident first.

        Handles three forms, in decreasing specificity:

        1. ``"green box"``  -- colour + class, the natural way people disambiguate
        2. ``"box"``        -- class alone
        3. ``"green"``      -- colour alone, when the class is obvious from context

        Matching the class alone would make "the green box" fail whenever the
        perceived class is just "box", which is exactly what happened in practice:
        the operator said "place the can on the green box" and the robot replied
        that it could not find a "green box" while looking straight at one.
        """
        needle = " ".join(label.strip().lower().split())
        if not needle:
            return []

        words = set(needle.split())

        def describes(obj: ObjectHypothesis) -> int:
            """0 = no match; higher = more specific match."""
            obj_class = obj.label.strip().lower()
            obj_color = str(obj.attributes.get("color", "")).strip().lower()

            if needle == obj_class:
                return 3
            if obj_color and needle == f"{obj_color} {obj_class}":
                return 4
            # Word-level: every word must be either the class or the colour, and
            # the class must be present, so "green box" cannot match a green can.
            if words and words <= {obj_class, obj_color} - {""}:
                return 4 if obj_class in words else 1
            if obj_class in words and words <= {obj_class, obj_color, "the", "a"} - {""}:
                return 2
            return 0

        scored = [(describes(o), o) for o in self.objects.values()]
        matched = [(rank, o) for rank, o in scored if rank > 0]
        if not matched:
            return []

        # Keep only the most specific tier: if "green box" matched one object
        # exactly, a looser colour-only match must not make it ambiguous.
        best_rank = max(rank for rank, _ in matched)
        return sorted(
            (o for rank, o in matched if rank == best_rank),
            key=lambda o: o.confidence,
            reverse=True,
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "step_index": self.step_index,
            "num_objects": len(self.objects),
            "objects": [o.to_log() for o in self.objects.values()],
            "relations": [list(r) for r in self.relations],
        }


@dataclass(frozen=True)
class GraspCandidate:
    """A synthesised grasp. Never hardcoded; always generated from geometry.

    ``pose`` is the target **TCP** pose at closure. ``pregrasp_pose`` is offset
    backwards along the approach axis, giving the motion planner a collision-free
    staging point from which the final approach is a straight line.

    ``width`` is the required gripper opening in metres, used to reject grasps
    the hardware cannot span before any planning cost is paid.
    """

    pose: Pose
    pregrasp_pose: Pose
    approach_axis: NDArray[np.float64]
    width: float
    score: float
    target_track_id: str
    scores_breakdown: Mapping[str, float] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_log(),
            "pregrasp_pose": self.pregrasp_pose.to_log(),
            "approach_axis": np.asarray(self.approach_axis).tolist(),
            "width": self.width,
            "score": self.score,
            "target_track_id": self.target_track_id,
            "scores_breakdown": dict(self.scores_breakdown),
        }


@dataclass(frozen=True)
class JointState:
    """Arm joint configuration. Excludes finger joints, which live in :class:`GripperState`."""

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64] | None = None
    efforts: NDArray[np.float64] | None = None
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "positions", pos)

    def to_log(self) -> dict[str, Any]:
        return {
            "positions": self.positions.tolist(),
            "names": list(self.names),
            "velocities": None if self.velocities is None else np.asarray(self.velocities).tolist(),
        }


@dataclass(frozen=True)
class Waypoint:
    """One point on a planned path, in joint space."""

    positions: NDArray[np.float64]
    time_from_start: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", np.asarray(self.positions, dtype=np.float64).reshape(-1))


@dataclass(frozen=True)
class Trajectory:
    """A time-parameterised joint-space path produced by the motion layer.

    Always the output of online planning. Nothing in this framework may
    construct a Trajectory from recorded joint angles.
    """

    waypoints: tuple[Waypoint, ...]
    joint_names: tuple[str, ...]
    planner_name: str
    planning_time_s: float

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def duration(self) -> float:
        return self.waypoints[-1].time_from_start if self.waypoints else 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "num_waypoints": len(self.waypoints),
            "duration": self.duration,
            "planner_name": self.planner_name,
            "planning_time_s": self.planning_time_s,
            "joint_names": list(self.joint_names),
        }


@dataclass(frozen=True)
class GripperState:
    """Finger state, plus whether physics reports an actual grasp.

    ``is_grasping`` must come from contact forces, never from "we commanded a
    close". A commanded close that closed on air reports ``False``.
    """

    width: float
    target_width: float
    is_moving: bool
    is_grasping: bool
    left_contact_force: float = 0.0
    right_contact_force: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "target_width": self.target_width,
            "is_moving": self.is_moving,
            "is_grasping": self.is_grasping,
            "left_contact_force": self.left_contact_force,
            "right_contact_force": self.right_contact_force,
        }


@dataclass(frozen=True)
class RobotState:
    """Complete proprioceptive snapshot at one instant."""

    joint_state: JointState
    tcp_pose: Pose
    gripper: GripperState
    sim_time: float
    step_index: int

    def to_log(self) -> dict[str, Any]:
        return {
            "joint_state": self.joint_state.to_log(),
            "tcp_pose": self.tcp_pose.to_log(),
            "gripper": self.gripper.to_log(),
            "sim_time": self.sim_time,
            "step_index": self.step_index,
        }


class SkillStatus(str, Enum):
    """Terminal status of a skill execution."""

    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"
    """Pre-empted by Stop / EmergencyStop."""
    INFEASIBLE = "infeasible"
    """Rejected before execution: no grasp found, target unreachable, IK failed."""
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Outcome of exactly one atomic skill.

    A skill returns to WAIT after this. Nothing here may trigger another skill:
    that is the atomicity guarantee the whole design rests on.
    """

    skill_name: str
    status: SkillStatus
    message: str = ""
    duration_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    wall_time: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.status is SkillStatus.SUCCESS

    def to_log(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "skill_name": self.skill_name,
            "status": self.status.value,
            "message": self.message,
            "duration_s": self.duration_s,
            "data": self.data,
        }

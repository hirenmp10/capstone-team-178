"""Planar 4-DoF arm kinematics for the hardware lane.

Pure NumPy + stdlib. This module must never import Isaac Sim.

Coordinate System:
    - Base origin: yaw axis at the table plane (Z=0).
    - +Z: points vertically upward from the table.
    - +X: bearing 0 (forward along the base reference).
    - +Y: completing the right-handed Cartesian coordinate system (+Y = +Z x +X).
    - Base frame is the Table frame.

Joint Ordering & Conventions:
    - Joint 0: base_yaw (rotation around +Z axis).
    - Joint 1: shoulder (pitch in the vertical radial plane).
    - Joint 2: elbow (pitch in the vertical radial plane).
    - Joint 3: wrist (pitch in the vertical radial plane).

Zero Pose [0, 0, 0, 0] rad:
    - Arm points straight upward along +Z.
    - Upper arm (L1), forearm (L2), and tool (L3) are collinear along +Z.
    - Positive pitch angles tip the distal links outward and downward toward the table.

TCP Definition & Orientation:
    - Position: fingertip / jaw midpoint (TCP).
    - Local +Z (approach direction): points out of the palm / wrist along the tool link.
      At phi3 = 0 (upright), approach is +Z [0, 0, 1].
      At phi3 = pi (curled straight down), approach is -Z [0, 0, -1] (top-down grasp).
    - Local +Y (jaw closing axis):
      - "tangential": closes across the radial ray (+/- [-sin(q1), cos(q1), 0]).
      - "radial": closes along the radial plane (+/- [cos(phi3)*cos(q1), cos(phi3)*sin(q1), -sin(phi3)]).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mfw.config.schema import HardwareArmConfig
from mfw.core.errors import KinematicsError
from mfw.core.types import Frame, Pose
from mfw.utils import transforms as tf

__all__ = ["PlanarKinematics"]

_TOLERANCE_ORIENTATION = 1e-3
_TOLERANCE_DISTANCE = 1e-6
_TOLERANCE_LIMITS = 1e-7


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the range (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_to_limits(angle: float, lower: float, upper: float) -> float:
    """Normalize angle to fit within [lower, upper] limits if possible."""
    canonical = _wrap_angle(angle)
    if lower - _TOLERANCE_LIMITS <= canonical <= upper + _TOLERANCE_LIMITS:
        return canonical
    if lower - _TOLERANCE_LIMITS <= canonical + 2.0 * math.pi <= upper + _TOLERANCE_LIMITS:
        return canonical + 2.0 * math.pi
    if lower - _TOLERANCE_LIMITS <= canonical - 2.0 * math.pi <= upper + _TOLERANCE_LIMITS:
        return canonical - 2.0 * math.pi
    return canonical


class PlanarKinematics:
    """Closed-form forward and inverse kinematics for a 4-DoF planar hobby arm.

    The arm consists of a base yaw joint followed by three coplanar pitch joints
    (shoulder, elbow, wrist) and a fixed tool link terminating at the TCP.
    """

    def __init__(self, arm: HardwareArmConfig, joint_names: Sequence[str]) -> None:
        """Initialize and validate kinematics geometry and limits.

        Args:
            arm: HardwareArmConfig containing link lengths, limits, and conventions.
            joint_names: Sequence of 4 joint names (base_yaw, shoulder, elbow, wrist).

        Raises:
            KinematicsError: If configuration is missing, malformed, or has invalid dimensions.
        """
        if not isinstance(arm, HardwareArmConfig):
            raise KinematicsError(f"Expected HardwareArmConfig instance, got {type(arm).__name__}")

        if len(joint_names) != 4:
            raise KinematicsError(
                f"PlanarKinematics requires exactly 4 joint names, got {len(joint_names)}"
            )

        expected_joints = ("base_yaw", "shoulder", "elbow", "wrist")
        if tuple(joint_names) != expected_joints:
            raise KinematicsError(
                f"Expected joint names {expected_joints}, got {tuple(joint_names)}"
            )

        self.arm = arm
        self.joint_names = tuple(joint_names)

        # Validate geometry
        if arm.base_height <= 0.0:
            raise KinematicsError(f"base_height must be positive, got {arm.base_height}")
        if arm.shoulder_offset < 0.0:
            raise KinematicsError(f"shoulder_offset must be non-negative, got {arm.shoulder_offset}")
        if arm.upper_arm <= 0.0:
            raise KinematicsError(f"upper_arm length must be positive, got {arm.upper_arm}")
        if arm.forearm <= 0.0:
            raise KinematicsError(f"forearm length must be positive, got {arm.forearm}")
        if arm.tool <= 0.0:
            raise KinematicsError(f"tool length must be positive, got {arm.tool}")

        # Validate limits
        if len(arm.joint_lower) != 4 or len(arm.joint_upper) != 4:
            raise KinematicsError("joint_lower and joint_upper must each contain 4 values")

        for i, (lo, hi) in enumerate(zip(arm.joint_lower, arm.joint_upper)):
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise KinematicsError(f"Joint {i} limits must be finite numbers, got ({lo}, {hi})")
            if lo >= hi:
                raise KinematicsError(f"Joint {i} lower limit ({lo}) must be < upper limit ({hi})")

        if arm.jaw_axis not in ("tangential", "radial"):
            raise KinematicsError(f"Unknown jaw_axis {arm.jaw_axis!r}, must be 'tangential' or 'radial'")

        self.joint_lower = tuple(float(x) for x in arm.joint_lower)
        self.joint_upper = tuple(float(x) for x in arm.joint_upper)

        self.s = float(arm.shoulder_offset)
        self.h = float(arm.base_height)
        self.L1 = float(arm.upper_arm)
        self.L2 = float(arm.forearm)
        self.L3 = float(arm.tool)

    def within_limits(self, q: Sequence[float] | NDArray[np.float64]) -> bool:
        """Check whether joint angles q are within kinematic limits.

        Args:
            q: 4-element sequence of joint angles in radians.

        Returns:
            True if all joints are within configured lower and upper bounds.
        """
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)):
            return False
        return all(
            lo - _TOLERANCE_LIMITS <= val <= hi + _TOLERANCE_LIMITS
            for lo, val, hi in zip(self.joint_lower, arr, self.joint_upper)
        )

    def fk(self, q: Sequence[float] | NDArray[np.float64]) -> Pose:
        """Compute forward kinematics for joint configuration q.

        Args:
            q: 4-element sequence of joint angles [base_yaw, shoulder, elbow, wrist] (rad).

        Returns:
            TCP Pose in Frame.WORLD.

        Raises:
            KinematicsError: If q has invalid length or contains non-finite values (NaN / Inf).
        """
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (4,):
            raise KinematicsError(f"Expected 4 joint values, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise KinematicsError("Joint angles contain non-finite values (NaN or Inf)")

        q1, q2, q3, q4 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

        phi1 = q2
        phi2 = q2 + q3
        phi3 = q2 + q3 + q4

        # Planar radial and vertical offsets
        r = (
            self.s
            + self.L1 * math.sin(phi1)
            + self.L2 * math.sin(phi2)
            + self.L3 * math.sin(phi3)
        )
        z = (
            self.h
            + self.L1 * math.cos(phi1)
            + self.L2 * math.cos(phi2)
            + self.L3 * math.cos(phi3)
        )

        x = r * math.cos(q1)
        y = r * math.sin(q1)

        position = np.array([x, y, z], dtype=np.float64)

        # TCP local +Z (approach direction)
        z_tcp = np.array(
            [
                math.sin(phi3) * math.cos(q1),
                math.sin(phi3) * math.sin(q1),
                math.cos(phi3),
            ],
            dtype=np.float64,
        )

        # TCP local +Y (jaw closing axis) and local +X
        if self.arm.jaw_axis == "tangential":
            y_tcp = np.array([-math.sin(q1), math.cos(q1), 0.0], dtype=np.float64)
            x_tcp = np.cross(y_tcp, z_tcp)
        else:  # "radial"
            x_tcp = np.array([-math.sin(q1), math.cos(q1), 0.0], dtype=np.float64)
            y_tcp = np.cross(z_tcp, x_tcp)

        # Ensure unit vectors
        x_norm = np.linalg.norm(x_tcp)
        y_norm = np.linalg.norm(y_tcp)
        z_norm = np.linalg.norm(z_tcp)

        if x_norm > 1e-9:
            x_tcp = x_tcp / x_norm
        if y_norm > 1e-9:
            y_tcp = y_tcp / y_norm
        if z_norm > 1e-9:
            z_tcp = z_tcp / z_norm

        rotation_matrix = np.stack([x_tcp, y_tcp, z_tcp], axis=1)
        quat = tf.matrix_to_quat(rotation_matrix)

        return Pose(position=position, quat=quat, frame=Frame.WORLD)

    def ik(
        self,
        pose: Pose,
        seed: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64] | None:
        """Compute closed-form inverse kinematics for target TCP pose.

        Considers both position and 3D orientation (approach direction and jaw closing axis).

        Args:
            pose: Target TCP Pose.
            seed: Optional 4-element joint configuration for deterministic tie-breaking.

        Returns:
            4-element NDArray of joint angles [q1, q2, q3, q4] (rad), or None if unreachable.

        Raises:
            KinematicsError: If pose or seed is malformed or contains NaN / Inf.
        """
        if not isinstance(pose, Pose):
            raise KinematicsError(f"Expected Pose instance, got {type(pose).__name__}")

        if not np.all(np.isfinite(pose.position)) or not np.all(np.isfinite(pose.quat)):
            raise KinematicsError("Target pose contains non-finite values (NaN or Inf)")

        seed_arr: NDArray[np.float64] | None = None
        if seed is not None:
            seed_arr = np.asarray(seed, dtype=np.float64).reshape(-1)
            if seed_arr.shape != (4,):
                raise KinematicsError(f"Expected 4 joint seed values, got {seed_arr.shape[0]}")
            if not np.all(np.isfinite(seed_arr)):
                raise KinematicsError("Seed contains non-finite values (NaN or Inf)")

        x, y, z = float(pose.position[0]), float(pose.position[1]), float(pose.position[2])
        r_xy = math.hypot(x, y)

        # Reject targets directly on the yaw axis or behind the base
        if r_xy < 1e-6:
            return None

        q1 = math.atan2(y, x)
        q1_norm = _normalize_to_limits(q1, self.joint_lower[0], self.joint_upper[0])

        # Check joint 1 bounds
        if not (self.joint_lower[0] - _TOLERANCE_LIMITS <= q1_norm <= self.joint_upper[0] + _TOLERANCE_LIMITS):
            return None

        # Extract target rotation matrix and column vectors
        rot = pose.rotation_matrix()
        y_target = rot[:, 1]
        z_target = rot[:, 2]

        r_hat = np.array([math.cos(q1), math.sin(q1), 0.0], dtype=np.float64)
        t_hat = np.array([-math.sin(q1), math.cos(q1), 0.0], dtype=np.float64)

        # 1. Approach direction must lie in the vertical radial plane (perpendicular to t_hat)
        z_tangential = float(np.dot(z_target, t_hat))
        if abs(z_tangential) > _TOLERANCE_ORIENTATION:
            return None

        # 2. Jaw closing axis validation
        if self.arm.jaw_axis == "tangential":
            y_tangential = abs(float(np.dot(y_target, t_hat)))
            if y_tangential < 1.0 - _TOLERANCE_ORIENTATION:
                return None
        else:  # "radial"
            y_tangential = abs(float(np.dot(y_target, t_hat)))
            if y_tangential > _TOLERANCE_ORIENTATION:
                return None

        # Calculate pitch sum phi3 = q2 + q3 + q4
        z_radial = float(np.dot(z_target, r_hat))
        z_z = float(z_target[2])
        phi3 = math.atan2(z_radial, z_z)

        # Compute wrist position by stepping back along approach axis by tool length L3
        wrist_pos = pose.position - self.L3 * z_target
        r_w = float(wrist_pos[0] * math.cos(q1) + wrist_pos[1] * math.sin(q1))
        z_w = float(wrist_pos[2])

        r_local = r_w - self.s
        z_local = z_w - self.h

        d_sq = r_local * r_local + z_local * z_local
        d = math.sqrt(d_sq)

        # Reachability distance check for 2-link (shoulder -> elbow -> wrist)
        d_min = abs(self.L1 - self.L2) - _TOLERANCE_DISTANCE
        d_max = self.L1 + self.L2 + _TOLERANCE_DISTANCE

        if not (d_min <= d <= d_max):
            return None

        if d < 1e-9:
            cos_q3 = -1.0
        else:
            cos_q3 = (d_sq - self.L1 * self.L1 - self.L2 * self.L2) / (2.0 * self.L1 * self.L2)
            cos_q3 = max(-1.0, min(1.0, cos_q3))

        sin_q3_mag = math.sqrt(max(0.0, 1.0 - cos_q3 * cos_q3))

        # Two elbow branches:
        # Branch 0 (elbow-up): q3 <= 0
        # Branch 1 (elbow-down): q3 >= 0
        solutions: list[NDArray[np.float64]] = []
        for sign in (-1.0, 1.0):
            sin_q3 = sign * sin_q3_mag
            q3_sol = math.atan2(sin_q3, cos_q3)
            a = self.L1 + self.L2 * cos_q3
            b = self.L2 * sin_q3
            q2_sol = math.atan2(a * r_local - b * z_local, a * z_local + b * r_local)
            q4_sol = _wrap_angle(phi3 - q2_sol - q3_sol)

            # Normalize to limit intervals if within range
            q1_c = _normalize_to_limits(q1, self.joint_lower[0], self.joint_upper[0])
            q2_c = _normalize_to_limits(q2_sol, self.joint_lower[1], self.joint_upper[1])
            q3_c = _normalize_to_limits(q3_sol, self.joint_lower[2], self.joint_upper[2])
            q4_c = _normalize_to_limits(q4_sol, self.joint_lower[3], self.joint_upper[3])

            solutions.append(np.array([q1_c, q2_c, q3_c, q4_c], dtype=np.float64))

        cand_up = solutions[0]
        cand_down = solutions[1]

        pref_cand = cand_up if self.arm.elbow_up else cand_down
        alt_cand = cand_down if self.arm.elbow_up else cand_up

        pref_valid = self.within_limits(pref_cand)
        alt_valid = self.within_limits(alt_cand)

        if pref_valid and alt_valid:
            if seed_arr is not None:
                # Deterministic tie-break using Euclidean distance in joint space
                dist_pref = float(np.sum((pref_cand - seed_arr) ** 2))
                dist_alt = float(np.sum((alt_cand - seed_arr) ** 2))
                if dist_alt < dist_pref - 1e-9:
                    return alt_cand
                return pref_cand
            return pref_cand
        elif pref_valid:
            return pref_cand
        elif alt_valid:
            return alt_cand
        else:
            return None

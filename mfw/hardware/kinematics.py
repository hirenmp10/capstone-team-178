"""Closed-form kinematics for the hardware lane's planar hobby arm.

Pure NumPy. This module must never import Isaac Sim, torch or transformers:
it runs on the laptop without Isaac and, at Stage 4, on the Jetson.

Why closed form rather than Lula
--------------------------------
The hardware lane has no URDF worth the name and no Lula. The arm is a base
yaw followed by three pitch joints whose axes are all parallel, so every
target reduces to a 2-link problem in one vertical plane. Solving that
analytically is exact, sub-microsecond, and -- unlike a numerical solver -- can
enumerate *both* elbow branches, which is what ``plan_with_retries`` needs when
the preferred branch sweeps through the table.

Frames and conventions (everything else in the lane assumes these)
----------------------------------------------------------------
World frame: origin on the yaw axis at table level (``z = 0``), **+X forward**
(away from the base, toward the workspace), **+Y left**, **+Z up**. This is the
same frame ``MoveRelative.DIRECTIONS`` and the scene graph use, so "move it
left 5 cm" means the same thing in sim and on metal.

``q = [yaw, shoulder, elbow, wrist]`` in radians, the wire order of the Uno
frames and of ``hardware.arm.joint_lower/upper``.

* **yaw** rotates about world Z; ``0`` points the arm plane down +X, positive
  turns it toward +Y (left). The arm plane is spanned by the *radial* unit
  vector ``r = (cos yaw, sin yaw, 0)`` and Z; the *tangential* unit vector is
  ``t = (-sin yaw, cos yaw, 0)`` (equal to +Y at zero yaw).
* Every pitch is measured **from vertical (+Z) toward the radial direction**,
  so a link at pitch ``phi`` points along ``sin(phi)·r + cos(phi)·Z``.
  ``phi = 0`` is straight up, ``pi/2`` is horizontal-forward, ``pi`` is
  straight down.
* **shoulder** ``q[1]``: absolute pitch of the upper arm. ``0`` = upper arm
  vertical, positive = leaning forward.
* **elbow** ``q[2]``: pitch of the forearm *relative to the upper arm*. ``0``
  = straight; positive bends the forearm further forward/down, which is the
  **elbow-up** posture (the elbow sits above the shoulder-to-wrist chord;
  ``sin(elbow) > 0`` is exactly that condition); negative bends it back
  over the shoulder (elbow-down / folded back).
* **wrist** ``q[3]``: pitch of the tool *relative to the forearm*, same sense.
  The absolute tool pitch is therefore ``shoulder + elbow + wrist``; a
  top-down grasp has tool pitch ``pi``.

Geometry: the shoulder axis sits at height ``arm.base_height`` and radial
offset ``arm.shoulder_offset`` from the yaw axis; ``arm.upper_arm`` and
``arm.forearm`` are the link lengths; ``arm.tool`` runs from the wrist axis to
the jaw midpoint, which is the TCP (fingertip midpoint, never a finger).

Under these conventions the placeholder home in ``configs/hardware.yaml``,
``[0, 0, -0.6109, 0.1745]``, is the upper arm vertical with the forearm folded
35 deg back and the jaws up-and-back (TCP about 9.5 cm behind the base axis,
34 cm high; 1500/1500/1150/1600 us on the placeholder pulse map): inside the
default limits, well away from the pulse extremes, clear of the table and out
of the overhead camera's workspace. Its signs are correct for this module;
only its magnitudes are placeholders (the S5 end-stop measurements replace it).

TCP frame (must match ``build_grasp_pose`` in :mod:`mfw.grasp.generator`)
-------------------------------------------------------------------------
Rotation columns are ``[X, Y, Z]`` with **Z = approach / tool direction**
(``sin(pitch)·r + cos(pitch)·Z``), **Y = finger closing axis**, ``X = Y × Z``.
The closing axis is fixed by how the gripper is bolted on (``arm.jaw_axis``):
``"tangential"`` closes along ``t`` (across the ray from the base), ``"radial"``
closes in the arm plane, perpendicular to the tool and pointing forward when
the tool points down. The frame is assembled by calling ``build_grasp_pose``
itself rather than a copy of it, so FK and grasp synthesis cannot drift apart.

Roll about the approach axis is not a degree of freedom, so IK ignores it and
only the approach axis of a requested pose is honoured. Requests whose
approach axis leaves the arm plane by more than ``orientation_tolerance`` are
rejected as unreachable rather than silently re-aimed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mfw.config.schema import ConfigError, HardwareArmConfig
from mfw.core.types import Frame, Pose
from mfw.grasp.generator import build_grasp_pose
from mfw.utils.logging import get_logger

__all__ = ["PlanarKinematics"]

_log = get_logger("hardware.kinematics")

_AXIS_EPS_M = 1e-12
"""Targets closer than this to the yaw axis have no defined yaw; the seed's is kept.

Measured trap: this used to be 1 mm, and for a target 0.77 mm off the axis a
seeded ``ik`` kept the seed's yaw, solved the plane along that radial and
returned a configuration whose TCP sat 1.5 mm from the request -- while still
reporting success. Roughly 0.3% of random in-limit configurations put the TCP
that close to the axis, so the FK-IK round trip flaked with 4 seeds in 10. Off
the axis the yaw is now always ``atan2`` (or its antipode), which is exact;
the seed only decides between those two, so a base swing is bounded by a
quarter turn and never invented for a target that is genuinely on the axis.
"""
_LIMIT_EPS = 1e-9
"""Slack on joint limits so a solution sitting exactly on a limit is accepted."""
_REACH_EPS = 1e-9
"""Slack on the 2-link reach annulus so a fully stretched pose is reachable."""
_DUP_EPS = 1e-9
"""Two branches closer than this in joint space are the same (straight-arm) solution."""


def _wrap_pi(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


class PlanarKinematics:
    """Analytic FK/IK for a yaw + three-coplanar-pitch arm with a fixed tool.

    Stateless apart from the geometry, so one instance can be shared by the
    robot proxy, the grasp scorer and the planner. All methods take and return
    NumPy arrays in the joint order given by ``joint_names``.
    """

    DOF = 4

    def __init__(
        self,
        arm: HardwareArmConfig,
        joint_names: Sequence[str],
        jaw_axis: str | None = None,
    ) -> None:
        arm.validate()
        names = tuple(str(n) for n in joint_names)
        if len(names) != self.DOF:
            raise ConfigError(
                "PlanarKinematics models exactly 4 joints (yaw, shoulder, elbow, wrist); "
                f"got {len(names)} joint names: {list(names)}"
            )
        axis = arm.jaw_axis if jaw_axis is None else str(jaw_axis)
        if axis not in HardwareArmConfig.JAW_AXES:
            raise ConfigError(
                f"jaw_axis must be one of {list(HardwareArmConfig.JAW_AXES)}, got {axis!r}"
            )

        self.arm = arm
        self.joint_names: tuple[str, ...] = names
        self.jaw_axis: str = axis
        self.tool_length: float = float(arm.tool)
        self.lower: NDArray[np.float64] = np.asarray(arm.joint_lower, dtype=np.float64)
        self.upper: NDArray[np.float64] = np.asarray(arm.joint_upper, dtype=np.float64)
        self.lower.setflags(write=False)
        self.upper.setflags(write=False)

        self._h = float(arm.base_height)
        self._s0 = float(arm.shoulder_offset)
        self._l1 = float(arm.upper_arm)
        self._l2 = float(arm.forearm)
        self._l3 = float(arm.tool)

        _log.debug(
            "PlanarKinematics: joints=%s h=%.3f s0=%.3f l1=%.3f l2=%.3f tool=%.3f jaw=%s elbow_up=%s",
            names, self._h, self._s0, self._l1, self._l2, self._l3, axis, arm.elbow_up,
        )

    # ------------------------------------------------------------------
    # helpers

    @property
    def dof(self) -> int:
        """Number of arm joints (always 4)."""
        return self.DOF

    def _q(self, q: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.DOF:
            raise ValueError(f"q must have {self.DOF} elements, got shape {np.shape(q)}")
        return arr

    @staticmethod
    def _plane_axes(yaw: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Radial and tangential unit vectors of the arm plane at ``yaw``."""
        c, s = math.cos(yaw), math.sin(yaw)
        radial = np.array([c, s, 0.0])
        tangential = np.array([-s, c, 0.0])
        return radial, tangential

    def _plane_points(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """In-plane ``(r, z)`` coordinates of shoulder, elbow, wrist and TCP, shape (4, 2)."""
        phi1 = q[1]
        phi2 = phi1 + q[2]
        phi3 = phi2 + q[3]
        shoulder = np.array([self._s0, self._h])
        elbow = shoulder + self._l1 * np.array([math.sin(phi1), math.cos(phi1)])
        wrist = elbow + self._l2 * np.array([math.sin(phi2), math.cos(phi2)])
        tcp = wrist + self._l3 * np.array([math.sin(phi3), math.cos(phi3)])
        return np.stack([shoulder, elbow, wrist, tcp])

    def _closing_axis(
        self, pitch: float, radial: NDArray[np.float64], tangential: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self.jaw_axis == "tangential":
            return tangential
        # In-plane, perpendicular to the tool, i.e. the direction at pitch - 90 deg:
        # forward (+radial) when the tool points straight down.
        return -math.cos(pitch) * radial + math.sin(pitch) * np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # forward kinematics

    def link_points(self, q: ArrayLike) -> NDArray[np.float64]:
        """World positions of ``[base, shoulder, elbow, wrist, tcp]``, shape (5, 3).

        The base point is the yaw axis at table level. Handy for clearance
        checks: any of these dipping below the table plane means the trajectory
        drives the arm into it.
        """
        qv = self._q(q)
        radial, _ = self._plane_axes(qv[0])
        plane = self._plane_points(qv)
        pts = plane[:, :1] * radial[None, :] + plane[:, 1:2] * np.array([[0.0, 0.0, 1.0]])
        return np.vstack([np.zeros((1, 3)), pts])

    def wrist_point(self, q: ArrayLike) -> NDArray[np.float64]:
        """World position of the wrist pitch axis, shape (3,)."""
        return self.link_points(q)[3]

    def fk(self, q: ArrayLike) -> Pose:
        """TCP pose in :attr:`Frame.WORLD` for joint vector ``q``.

        Raises ``ValueError`` if ``q`` does not have four elements. Limits are
        not checked here: FK of an out-of-limit configuration is still a valid
        question (the planner asks it when deciding whether to clamp).
        """
        qv = self._q(q)
        radial, tangential = self._plane_axes(qv[0])
        plane = self._plane_points(qv)
        tcp = plane[3, 0] * radial + np.array([0.0, 0.0, plane[3, 1]])
        pitch = qv[1] + qv[2] + qv[3]
        approach = math.sin(pitch) * radial + np.array([0.0, 0.0, math.cos(pitch)])
        closing = self._closing_axis(pitch, radial, tangential)
        return build_grasp_pose(tcp, closing, approach)

    # ------------------------------------------------------------------
    # limits

    def within_limits(self, q: ArrayLike) -> bool:
        """True when every joint lies inside ``[joint_lower, joint_upper]`` (1e-9 slack)."""
        qv = self._q(q)
        return bool(
            np.all(qv >= self.lower - _LIMIT_EPS) and np.all(qv <= self.upper + _LIMIT_EPS)
        )

    def clamp(self, q: ArrayLike) -> NDArray[np.float64]:
        """Copy of ``q`` with every joint clipped into its limits."""
        return np.clip(self._q(q), self.lower, self.upper)

    def reach_radius(self) -> float:
        """Distance from the yaw axis to the TCP with the arm fully stretched, metres.

        An upper bound on horizontal reach (the true envelope is a torus
        section that also depends on the target height and the tool pitch).
        """
        return self._s0 + self._l1 + self._l2 + self._l3

    # ------------------------------------------------------------------
    # inverse kinematics

    def _solve_yaw(self, position: NDArray[np.float64], seed: NDArray[np.float64] | None) -> float:
        x, y = float(position[0]), float(position[1])
        if math.hypot(x, y) < _AXIS_EPS_M:
            # Exactly on the yaw axis every yaw is equally valid; keeping the
            # seed's avoids a needless base swing. Anywhere else only atan2 (or
            # its antipode, tried by the caller) reaches the target exactly.
            return float(seed[0]) if seed is not None else 0.0
        return math.atan2(y, x)

    def _solve_plane(
        self,
        yaw: float,
        rho: float,
        z: float,
        approach: NDArray[np.float64],
        orientation_tolerance: float,
    ) -> list[NDArray[np.float64]]:
        """2-link closed form in the arm plane at ``yaw``; both elbow branches, preferred first.

        ``rho`` is the *signed* radial coordinate of the TCP along that plane's
        radial axis (negative when the target lies behind the yaw axis).
        """
        radial, tangential = self._plane_axes(yaw)

        # Roll is not available, so only the approach axis is honoured -- and
        # only if it (nearly) lies in the arm plane.
        a_t = float(np.dot(approach, tangential))
        out_of_plane = math.asin(min(1.0, abs(a_t)))
        if out_of_plane > orientation_tolerance:
            return []
        a_r = float(np.dot(approach, radial))
        a_z = float(approach[2])
        if math.hypot(a_r, a_z) < 1e-9:
            return []
        pitch = math.atan2(a_r, a_z)

        # The wrist sits ``tool`` behind the TCP along the (in-plane) tool
        # direction. Using the in-plane direction keeps the TCP position exact
        # even when the requested approach is slightly out of plane.
        wrist_r = rho - self._l3 * math.sin(pitch)
        wrist_z = z - self._l3 * math.cos(pitch)

        dr = wrist_r - self._s0
        dz = wrist_z - self._h
        dist_sq = dr * dr + dz * dz
        dist = math.sqrt(dist_sq)
        if dist > self._l1 + self._l2 + _REACH_EPS or dist < abs(self._l1 - self._l2) - _REACH_EPS:
            return []

        cos_elbow = (dist_sq - self._l1 * self._l1 - self._l2 * self._l2) / (2.0 * self._l1 * self._l2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow_mag = math.acos(cos_elbow)
        chord = math.atan2(dr, dz)

        # Positive elbow == elbow-up (see module docstring).
        signs = (1.0, -1.0) if self.arm.elbow_up else (-1.0, 1.0)
        solutions: list[NDArray[np.float64]] = []
        for sign in signs:
            elbow = sign * elbow_mag
            shoulder = chord - math.atan2(
                self._l2 * math.sin(elbow), self._l1 + self._l2 * math.cos(elbow)
            )
            wrist = _wrap_pi(pitch - shoulder - elbow)
            q = np.array([yaw, shoulder, elbow, wrist], dtype=np.float64)
            if not self.within_limits(q):
                continue
            if any(np.linalg.norm(q - other) < _DUP_EPS for other in solutions):
                continue  # straight arm: both branches coincide
            solutions.append(q)
        return solutions

    def _planar_candidates(
        self,
        pose: Pose,
        seed: NDArray[np.float64] | None,
        orientation_tolerance: float,
    ) -> list[NDArray[np.float64]]:
        """Every in-limit solution for ``pose``: forward yaw first, then the antipodal yaw.

        The shared core of :meth:`ik` and :meth:`both_branches`. A target at
        ``atan2(y, x)`` can also be reached with the base turned half a turn
        and the arm folded back over itself (signed radial coordinate ``-rho``).
        That posture is rarely inside the yaw limits for a workspace target,
        but it *is* where a third of the in-limit configurations put the TCP,
        and an IK that cannot invert its own FK there would make the round-trip
        tests -- and any planner seeded from such a posture -- lie.
        """
        if pose.frame != Frame.WORLD:
            raise ValueError(f"PlanarKinematics expects a WORLD pose, got frame {pose.frame!r}")
        position = np.asarray(pose.position, dtype=np.float64)
        approach = pose.rotation_matrix()[:, 2]

        yaw = self._solve_yaw(position, seed)
        rho = math.hypot(float(position[0]), float(position[1]))
        z = float(position[2])

        solutions = self._solve_plane(yaw, rho, z, approach, orientation_tolerance)
        yaw_back = _wrap_pi(yaw + math.pi)
        if self.lower[0] - _LIMIT_EPS <= yaw_back <= self.upper[0] + _LIMIT_EPS:
            solutions.extend(self._solve_plane(yaw_back, -rho, z, approach, orientation_tolerance))
        return solutions

    def ik(
        self,
        pose: Pose,
        seed: NDArray[np.float64] | None = None,
        orientation_tolerance: float = 0.2,
    ) -> NDArray[np.float64] | None:
        """Joint vector reaching ``pose``'s position with its approach axis, or ``None``.

        * Yaw is ``atan2(y, x)``; only for a target exactly on the axis is the
          seed's yaw (or 0) kept. The antipodal yaw (base turned half a turn,
          arm folded back over itself) is tried second when it lies inside the
          yaw limits, so a seeded call near the axis picks whichever of the two
          exact yaws is closer rather than an inexact compromise.
        * The approach axis may deviate from the arm plane by at most
          ``orientation_tolerance`` radians (default 0.2); roll about it is
          ignored, since the arm has no such joint.
        * The elbow branch follows ``arm.elbow_up``, falling back to the other
          branch when the preferred one violates a limit. When *both* branches
          are valid and a ``seed`` is given, the branch nearer the seed in joint
          space wins, so a following motion does not flip the elbow through the
          table.
        * ``None`` when the target is outside the reach annulus, the approach
          axis is out of plane, or no branch fits the limits.

        Raises ``ValueError`` if ``pose`` is not expressed in ``Frame.WORLD``.
        """
        seed_v = None if seed is None else self._q(seed)
        candidates = self._planar_candidates(pose, seed_v, float(orientation_tolerance))
        if not candidates:
            return None
        if seed_v is not None and len(candidates) > 1:
            candidates.sort(key=lambda q: float(np.linalg.norm(q - seed_v)))
        return candidates[0]

    def both_branches(
        self, pose: Pose, orientation_tolerance: float = 0.2
    ) -> list[NDArray[np.float64]]:
        """Every in-limit IK solution for ``pose``, preferred branch first.

        Usually 0, 1 or 2 entries (both elbow branches at the forward yaw); up
        to 4 when the antipodal yaw is also inside the yaw limits.
        ``plan_with_retries`` walks this list so that a goal configuration the
        planner has already rejected is not solved for again.
        """
        return self._planar_candidates(pose, None, float(orientation_tolerance))

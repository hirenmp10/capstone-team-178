"""Trajectory time-parameterisation and interpolation.

Pure NumPy. This module must never import Isaac Sim.

A planner returns a *geometric* path -- an ordered list of configurations with no
timing. Executing that directly means stepping between waypoints as fast as the
loop runs, which produces velocity spikes at every corner and, with an object in
the gripper, inertial loads that break the grasp. This module assigns timing that
respects velocity and acceleration limits, and resamples the result at the
control period.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mfw.core.types import Trajectory, Waypoint

__all__ = [
    "time_parameterise",
    "resample",
    "densify_path",
    "path_length",
    "max_joint_step",
]

_EPS = 1e-12


def path_length(path: NDArray[np.float64]) -> float:
    """Total joint-space arc length of a path."""
    pts = np.asarray(path, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def max_joint_step(path: NDArray[np.float64]) -> float:
    """Largest single-joint change between consecutive waypoints, in radians.

    Used to decide whether a path needs densifying: a large step means the
    straight line between those two configurations was never collision-checked
    at any intermediate point.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(pts, axis=0))))


def densify_path(
    path: NDArray[np.float64], max_step: float
) -> NDArray[np.float64]:
    """Insert intermediate configurations so no joint moves more than ``max_step``.

    RRT returns a sparse path whose segments can span large joint motions. The
    segment endpoints are collision-free but the straight line between them is
    only verified by the planner's own edge check; densifying gives the
    controller a path it can track smoothly and lets us re-verify clearance at
    a chosen resolution.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    if max_step <= 0.0:
        raise ValueError("max_step must be > 0")

    segments = [pts[0:1]]
    for start, end in zip(pts[:-1], pts[1:]):
        delta = end - start
        steps = int(np.ceil(float(np.max(np.abs(delta))) / max_step))
        steps = max(1, steps)
        # Exclude the start (already emitted) and include the end.
        fractions = np.linspace(0.0, 1.0, steps + 1)[1:]
        segments.append(start + fractions[:, None] * delta)

    return np.vstack(segments)


def time_parameterise(
    path: NDArray[np.float64],
    joint_names: tuple[str, ...],
    max_velocity: float,
    max_acceleration: float,
    planner_name: str = "unknown",
    planning_time_s: float = 0.0,
) -> Trajectory:
    """Assign timing to a geometric path under velocity and acceleration limits.

    Each segment is timed by the slowest joint at ``max_velocity``, then the
    whole trajectory is stretched by a single factor if that timing would demand
    more than ``max_acceleration`` anywhere.

    Uniform stretching rather than per-segment optimal timing is deliberate: a
    time-optimal profile spends most of its life at a limit, so any tracking
    error has no headroom to recover. Manipulation values a trajectory the
    controller can actually follow over one that finishes 15% sooner.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"path must be (N, DOF), got shape {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("cannot parameterise an empty path")
    if pts.shape[1] != len(joint_names):
        raise ValueError(
            f"path has {pts.shape[1]} joints but {len(joint_names)} joint names were given"
        )
    if max_velocity <= 0.0 or max_acceleration <= 0.0:
        raise ValueError("max_velocity and max_acceleration must be > 0")

    if pts.shape[0] == 1:
        return Trajectory(
            waypoints=(Waypoint(pts[0], 0.0),),
            joint_names=tuple(joint_names),
            planner_name=planner_name,
            planning_time_s=planning_time_s,
        )

    deltas = np.diff(pts, axis=0)
    segment_spans = np.max(np.abs(deltas), axis=1)
    # A zero-length segment still needs a nonzero duration or the velocity
    # estimate below divides by zero.
    durations = np.maximum(segment_spans / max_velocity, 1e-4)
    times = np.concatenate([[0.0], np.cumsum(durations)])

    scale = _acceleration_scale(pts, times, max_acceleration)
    if scale > 1.0:
        times = times * scale

    return Trajectory(
        waypoints=tuple(Waypoint(q, float(t)) for q, t in zip(pts, times)),
        joint_names=tuple(joint_names),
        planner_name=planner_name,
        planning_time_s=planning_time_s,
    )


def _acceleration_scale(
    path: NDArray[np.float64], times: NDArray[np.float64], max_acceleration: float
) -> float:
    """Factor by which to stretch time so no joint exceeds ``max_acceleration``.

    Acceleration scales as ``1/t^2``, so stretching time by ``s`` divides
    acceleration by ``s^2`` -- hence the square root.
    """
    if path.shape[0] < 3:
        return 1.0

    dt = np.diff(times)
    velocities = np.diff(path, axis=0) / dt[:, None]
    dt_mid = (dt[:-1] + dt[1:]) / 2.0
    accelerations = np.diff(velocities, axis=0) / dt_mid[:, None]

    peak = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0
    if peak <= max_acceleration or peak <= _EPS:
        return 1.0
    return float(np.sqrt(peak / max_acceleration))


def resample(trajectory: Trajectory, dt: float) -> Trajectory:
    """Resample a trajectory onto a uniform time grid.

    The controller runs at a fixed period, so handing it waypoints on an
    irregular grid means it either interpolates itself or jitters. Linear
    interpolation in joint space is correct here because timing has already been
    chosen to keep the motion within limits.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if len(trajectory) < 2:
        return trajectory

    times = np.array([w.time_from_start for w in trajectory.waypoints])
    positions = np.stack([w.positions for w in trajectory.waypoints])

    # Always include the final waypoint: dropping it would stop the arm short of
    # the goal, which for a grasp approach means closing on empty space.
    num_samples = max(2, int(np.ceil(times[-1] / dt)) + 1)
    sample_times = np.minimum(np.arange(num_samples) * dt, times[-1])
    if sample_times[-1] < times[-1]:
        sample_times = np.append(sample_times, times[-1])

    sampled = np.stack(
        [np.interp(sample_times, times, positions[:, j]) for j in range(positions.shape[1])],
        axis=1,
    )

    return Trajectory(
        waypoints=tuple(Waypoint(q, float(t)) for q, t in zip(sampled, sample_times)),
        joint_names=trajectory.joint_names,
        planner_name=trajectory.planner_name,
        planning_time_s=trajectory.planning_time_s,
    )

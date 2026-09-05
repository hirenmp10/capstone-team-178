"""Phase 3 pure-logic tests: trajectory timing and interpolation.

No Isaac Sim. Timing correctness matters because a trajectory that violates
velocity or acceleration limits produces inertial loads that break a grasp mid-carry.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.motion.trajectory import (
    densify_path,
    max_joint_step,
    path_length,
    resample,
    time_parameterise,
)

pytestmark = pytest.mark.phase3

JOINTS = tuple(f"j{i}" for i in range(7))


def _line_path(steps=10, span=1.0):
    return np.linspace(np.zeros(7), np.full(7, span), steps)


class TestPathMetrics:
    def test_path_length_of_straight_line(self):
        path = _line_path(steps=3, span=1.0)
        assert path_length(path) == pytest.approx(np.linalg.norm(np.full(7, 1.0)))

    def test_single_point_has_no_length(self):
        assert path_length(np.zeros((1, 7))) == 0.0

    def test_max_joint_step(self):
        path = np.array([[0.0] * 7, [0.1] * 6 + [0.5]])
        assert max_joint_step(path) == pytest.approx(0.5)


class TestDensify:
    def test_inserts_intermediate_waypoints(self):
        path = np.array([[0.0] * 7, [1.0] * 7])
        dense = densify_path(path, max_step=0.1)
        assert dense.shape[0] == 11
        assert max_joint_step(dense) <= 0.1 + 1e-9

    def test_preserves_endpoints(self):
        path = np.array([[0.0] * 7, [0.3] * 7, [-0.2] * 7])
        dense = densify_path(path, max_step=0.05)
        assert np.allclose(dense[0], path[0])
        assert np.allclose(dense[-1], path[-1])

    def test_already_fine_path_is_unchanged_in_shape(self):
        path = np.array([[0.0] * 7, [0.01] * 7])
        assert densify_path(path, max_step=0.1).shape[0] == 2

    def test_rejects_bad_step(self):
        with pytest.raises(ValueError):
            densify_path(np.zeros((2, 7)), max_step=0.0)


class TestTimeParameterise:
    def test_respects_velocity_limit(self):
        path = _line_path(steps=5, span=1.0)
        traj = time_parameterise(path, JOINTS, max_velocity=0.5, max_acceleration=100.0)

        times = np.array([w.time_from_start for w in traj.waypoints])
        positions = np.stack([w.positions for w in traj.waypoints])
        velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
        assert np.max(np.abs(velocities)) <= 0.5 + 1e-6

    def test_respects_acceleration_limit(self):
        """The reason a trajectory is stretched rather than run time-optimally."""
        path = np.array([[0.0] * 7, [0.5] * 7, [0.5] * 7, [0.0] * 7])
        traj = time_parameterise(path, JOINTS, max_velocity=2.0, max_acceleration=1.0)

        times = np.array([w.time_from_start for w in traj.waypoints])
        positions = np.stack([w.positions for w in traj.waypoints])
        velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
        dt_mid = (np.diff(times)[:-1] + np.diff(times)[1:]) / 2.0
        accelerations = np.diff(velocities, axis=0) / dt_mid[:, None]
        assert np.max(np.abs(accelerations)) <= 1.0 * 1.05

    def test_times_are_monotonic_and_start_at_zero(self):
        traj = time_parameterise(_line_path(8), JOINTS, 1.0, 2.0)
        times = [w.time_from_start for w in traj.waypoints]
        assert times[0] == 0.0
        assert all(b > a for a, b in zip(times, times[1:]))

    def test_single_waypoint_yields_zero_duration(self):
        traj = time_parameterise(np.zeros((1, 7)), JOINTS, 1.0, 2.0)
        assert len(traj) == 1 and traj.duration == 0.0

    def test_duplicate_waypoints_do_not_divide_by_zero(self):
        """A planner can legitimately emit repeated configurations."""
        path = np.array([[0.0] * 7, [0.0] * 7, [0.1] * 7])
        traj = time_parameterise(path, JOINTS, 1.0, 2.0)
        assert all(np.isfinite(w.time_from_start) for w in traj.waypoints)

    def test_rejects_joint_count_mismatch(self):
        with pytest.raises(ValueError, match="joint names"):
            time_parameterise(np.zeros((3, 5)), JOINTS, 1.0, 2.0)

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match="empty path"):
            time_parameterise(np.zeros((0, 7)), JOINTS, 1.0, 2.0)

    def test_rejects_nonpositive_limits(self):
        with pytest.raises(ValueError):
            time_parameterise(_line_path(3), JOINTS, 0.0, 1.0)

    def test_records_planner_metadata(self):
        traj = time_parameterise(
            _line_path(3), JOINTS, 1.0, 2.0, planner_name="lula_rrt", planning_time_s=0.25
        )
        assert traj.planner_name == "lula_rrt"
        assert traj.planning_time_s == 0.25


class TestResample:
    def test_produces_uniform_grid(self):
        traj = time_parameterise(_line_path(5, span=1.0), JOINTS, 1.0, 100.0)
        dense = resample(traj, dt=0.02)
        times = np.array([w.time_from_start for w in dense.waypoints])
        gaps = np.diff(times)[:-1]  # last gap may be short to land on the end
        assert np.allclose(gaps, 0.02, atol=1e-9)

    def test_always_includes_the_final_waypoint(self):
        """Stopping short of the goal means a grasp closes on empty space."""
        traj = time_parameterise(_line_path(5, span=1.0), JOINTS, 1.0, 100.0)
        dense = resample(traj, dt=0.0301)
        assert dense.waypoints[-1].time_from_start == pytest.approx(traj.duration)
        assert np.allclose(dense.waypoints[-1].positions, traj.waypoints[-1].positions)

    def test_interpolated_values_lie_on_the_path(self):
        traj = time_parameterise(_line_path(3, span=1.0), JOINTS, 1.0, 100.0)
        dense = resample(traj, dt=0.05)
        for waypoint in dense.waypoints:
            # A straight line in joint space: all joints must stay equal.
            assert np.allclose(waypoint.positions, waypoint.positions[0])

    def test_short_trajectory_passes_through(self):
        traj = time_parameterise(np.zeros((1, 7)), JOINTS, 1.0, 2.0)
        assert resample(traj, 0.01) is traj

    def test_rejects_bad_dt(self):
        traj = time_parameterise(_line_path(3), JOINTS, 1.0, 2.0)
        with pytest.raises(ValueError):
            resample(traj, 0.0)

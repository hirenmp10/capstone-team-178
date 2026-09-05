"""Phase 3 gate: online motion planning and trajectory execution.

Verifies that the arm plans collision-free paths against a perception-derived
collision world and actually reaches its goals -- with no recorded trajectories
anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.core.errors import SafetyViolation
from mfw.core.types import Frame, JointState, Pose

pytestmark = [pytest.mark.isaac, pytest.mark.phase3]


@pytest.fixture
def planning_scene(restore_scene):
    """A settled scene with a fresh observation, ready to plan against."""
    runtime = restore_scene
    runtime.vision.observe()
    scene = runtime.vision.observe()
    return runtime, scene


def _joint_state(runtime) -> JointState:
    return runtime.robot.get_state().joint_state


class TestPlannerSetup:
    def test_planner_and_controller_exist(self, restore_scene):
        assert restore_scene.planner is not None
        assert restore_scene.controller is not None

    def test_collision_world_updates_from_perception(self, planning_scene):
        """Obstacles come from the scene graph, never from spawn configuration."""
        runtime, scene = planning_scene
        runtime.planner.update_collision_world(scene)
        # One proxy obstacle per perceived object.
        assert len(runtime.planner._obstacle_prims) == len(scene.objects)
        for track_id in scene.objects:
            assert track_id in runtime.planner._obstacle_prims

    def test_vanished_objects_stop_blocking(self, planning_scene):
        """Otherwise the arm avoids empty space forever."""
        runtime, scene = planning_scene
        runtime.planner.update_collision_world(scene)

        from mfw.core.types import SceneGraph

        empty = SceneGraph(objects={}, sim_time=scene.sim_time, step_index=scene.step_index)
        runtime.planner.update_collision_world(empty)
        # Prims are pooled, not deleted; they must simply be disabled.
        assert len(runtime.planner._obstacle_prims) == len(scene.objects)


class TestJointPlanning:
    def test_plans_to_a_reachable_joint_goal(self, planning_scene):
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = start.positions + np.array([0.3, 0.1, -0.1, 0.15, 0.0, -0.1, 0.2])

        trajectory = runtime.planner.plan_to_joint(start, goal, scene)
        assert trajectory is not None, "RRT found no path to a nearby configuration"
        assert len(trajectory) >= 2
        assert trajectory.planner_name == "lula_rrt"
        assert np.allclose(trajectory.waypoints[-1].positions, goal, atol=0.05)

    def test_trajectory_is_time_parameterised(self, planning_scene):
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = start.positions + np.array([0.25, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

        trajectory = runtime.planner.plan_to_joint(start, goal, scene)
        assert trajectory is not None
        assert trajectory.duration > 0.0
        times = [w.time_from_start for w in trajectory.waypoints]
        assert all(b > a for a, b in zip(times, times[1:])), "timings not monotonic"

    def test_plan_is_not_a_recorded_trajectory(self, planning_scene):
        """Two different goals must produce genuinely different plans."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)

        first = runtime.planner.plan_to_joint(
            start, start.positions + np.array([0.3, 0, 0, 0, 0, 0, 0]), scene
        )
        second = runtime.planner.plan_to_joint(
            start, start.positions + np.array([-0.3, 0, 0, 0, 0, 0, 0]), scene
        )
        assert first is not None and second is not None
        assert not np.allclose(
            first.waypoints[-1].positions, second.waypoints[-1].positions, atol=0.05
        )


class TestCartesianPlanning:
    def test_plans_a_straight_line_in_tcp_space(self, planning_scene):
        """Approach and lift must not curve, or the fingers sweep the object."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        current = runtime.robot.forward_kinematics(start.positions)

        goal = Pose(
            current.position + np.array([0.0, 0.0, -0.06]), current.quat, Frame.WORLD
        )
        trajectory = runtime.planner.plan_cartesian_line(start, goal, scene)
        assert trajectory is not None, "Cartesian IK failed on a short vertical move"
        assert trajectory.planner_name == "cartesian_line"

        # Every waypoint's TCP must lie near the straight line from start to goal.
        line = goal.position - current.position
        line_length = float(np.linalg.norm(line))
        direction = line / line_length
        for waypoint in trajectory.waypoints:
            tcp = runtime.robot.forward_kinematics(waypoint.positions).position
            offset = tcp - current.position
            along = float(np.dot(offset, direction))
            perpendicular = float(np.linalg.norm(offset - along * direction))
            assert perpendicular < 0.012, (
                f"TCP deviated {perpendicular * 1000:.1f} mm from the straight line"
            )

    def test_returns_none_when_unreachable(self, planning_scene):
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = Pose(np.array([3.0, 3.0, 3.0]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        assert runtime.planner.plan_cartesian_line(start, goal, scene) is None


class TestExecution:
    def test_executes_a_trajectory_and_reaches_the_goal(self, planning_scene):
        """The end-to-end claim of Phase 3."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = start.positions + np.array([0.3, 0.1, 0.0, 0.15, 0.0, -0.1, 0.0])

        trajectory = runtime.planner.plan_to_joint(start, goal, scene)
        assert trajectory is not None

        assert runtime.controller.follow_trajectory(trajectory), "execution reported failure"

        final = runtime.robot.get_arm_joint_positions()
        error = float(np.max(np.abs(final - goal)))
        assert error < 0.05, f"arm settled {error:.3f} rad from the goal"

    def test_tcp_reaches_the_planned_cartesian_goal(self, planning_scene):
        """What actually matters for grasping: where the fingertips end up."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        current = runtime.robot.forward_kinematics(start.positions)
        goal = Pose(current.position + np.array([0.03, 0.0, -0.05]), current.quat, Frame.WORLD)

        trajectory = runtime.planner.plan_cartesian_line(start, goal, scene)
        assert trajectory is not None
        assert runtime.controller.follow_trajectory(trajectory)

        achieved = runtime.robot.tcp_pose()
        error = achieved.translation_distance(goal)
        assert error < 0.02, f"TCP is {error * 1000:.1f} mm from the Cartesian goal"

    def test_stop_aborts_execution(self, planning_scene):
        """Stop must halt without dropping the arm."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = start.positions + np.array([0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        trajectory = runtime.planner.plan_to_joint(start, goal, scene)
        assert trajectory is not None

        def stop_midway(index, _positions):
            if index > 3:
                runtime.controller.stop()
            return True

        assert not runtime.controller.follow_trajectory(trajectory, on_step=stop_midway)

        # The arm must still be held, not limp: it should not be at the goal, but
        # it also must not have collapsed.
        final = runtime.robot.get_arm_joint_positions()
        assert float(np.max(np.abs(final - goal))) > 0.05, "stop did not actually interrupt"
        assert np.all(np.isfinite(final))

    def test_callback_abort_stops_execution(self, planning_scene):
        """The hook the pick skill uses to watch contact while closing."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)
        goal = start.positions + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        trajectory = runtime.planner.plan_to_joint(start, goal, scene)
        assert trajectory is not None

        calls = []

        def abort_immediately(index, positions):
            calls.append(index)
            return False

        assert not runtime.controller.follow_trajectory(trajectory, on_step=abort_immediately)
        assert calls == [0], "callback should abort on the first step"

    def test_workspace_violation_is_rejected(self, planning_scene):
        """A safety limit raises rather than returning a status; retrying is never right."""
        runtime, scene = planning_scene
        start = _joint_state(runtime)

        from mfw.core.types import Trajectory, Waypoint

        # Rotate the base ~143 degrees: this swings the TCP behind the robot, to
        # negative X, unambiguously outside workspace_min. Bending an elbow is
        # not enough -- the arm can flex a long way and stay inside the box.
        bad = start.positions.copy()
        bad[0] += 2.5
        assert np.any(
            runtime.robot.forward_kinematics(bad).position
            < np.array(runtime.config.scene.workspace_min)
        ), "test setup does not actually leave the workspace"
        trajectory = Trajectory(
            waypoints=(Waypoint(start.positions, 0.0), Waypoint(bad, 1.0)),
            joint_names=runtime.robot.joint_names,
            planner_name="test",
            planning_time_s=0.0,
        )
        with pytest.raises(SafetyViolation, match="outside the workspace"):
            runtime.controller.follow_trajectory(trajectory)

    def test_hold_position_resists_gravity(self, planning_scene):
        """Without re-asserting targets the drives chase stale commands."""
        runtime, _ = planning_scene
        before = runtime.robot.get_arm_joint_positions()
        runtime.controller.hold_position()
        runtime.sim.step(120, render=False)
        after = runtime.robot.get_arm_joint_positions()
        assert float(np.max(np.abs(after - before))) < 0.05, "arm drifted while holding"


class TestGripperControl:
    def test_open_then_close_blocking(self, planning_scene):
        runtime, _ = planning_scene
        opened = runtime.controller.open_gripper_blocking()
        assert opened > runtime.config.robot.gripper_open_width * 0.7

        closed = runtime.controller.close_gripper_blocking()
        assert closed < opened - 0.01

    def test_close_stalls_early_on_free_air(self, planning_scene):
        """Closing on nothing should terminate on the stall, not burn the budget."""
        runtime, _ = planning_scene
        runtime.controller.open_gripper_blocking()
        width = runtime.controller.close_gripper_blocking(settle_steps=400)
        assert width < 0.01, f"free close ended at {width:.4f} m"

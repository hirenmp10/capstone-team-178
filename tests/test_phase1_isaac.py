"""Phase 1 gate: robot, camera, physics.

Requires a live Isaac Sim. One SimulationApp is shared by the whole module
(session-scoped fixture) because Isaac's runtime is a process-wide singleton --
a second instance corrupts the first rather than failing cleanly.

Run with:
    python.bat -m pytest tests/test_phase1_isaac.py -m isaac -q

Phase 2 does not begin until every test here passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfw.config.schema import load_config
from mfw.core.types import Frame
from mfw.utils import transforms as tf

pytestmark = [pytest.mark.isaac, pytest.mark.phase1]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# The ``runtime`` fixture is session-scoped and lives in conftest.py: Isaac's
# runtime is a process-wide singleton, so every phase suite shares one instance.


class TestRobotBringUp:
    def test_articulation_has_expected_dofs(self, runtime):
        robot = runtime.robot
        assert len(robot.arm_dof_indices) == 7
        assert len(robot.finger_dof_indices) == 2

    def test_arm_dof_indices_match_configured_names(self, runtime):
        """Indices are resolved by name, never assumed from ordering."""
        dof_names = list(runtime.robot.articulation.dof_names)
        for idx, name in zip(runtime.robot.arm_dof_indices, runtime.config.robot.arm_joint_names):
            assert dof_names[idx] == name

    def test_home_posture_reached(self, runtime):
        actual = runtime.robot.get_arm_joint_positions()
        expected = np.array(runtime.config.robot.home_joint_positions)
        assert np.allclose(actual, expected, atol=0.05), (
            f"home posture not held: {np.round(actual, 3).tolist()}"
        )


class TestTcpFrame:
    """The frame bug this framework exists to avoid.

    Isaac's shipped Franka example uses ``panda_rightfinger`` as the end
    effector. These tests pin the TCP to the actual fingertip midpoint and prove
    the difference is large enough to break grasping.
    """

    def test_tcp_is_offset_from_hand_by_configured_amount(self, runtime):
        robot = runtime.robot
        hand = robot.hand_pose()
        tcp = robot.tcp_pose()

        offset_world = tcp.position - hand.position
        offset_local = hand.rotation_matrix().T @ offset_world
        expected = np.array(runtime.config.robot.tcp_offset_from_hand)
        assert np.allclose(offset_local, expected, atol=1e-4), (
            f"TCP offset in hand frame was {np.round(offset_local, 5).tolist()}, "
            f"expected {expected.tolist()}"
        )

    def test_tcp_lies_between_the_fingertips(self, runtime):
        """Ground truth: the TCP must sit at the midpoint of the two fingers,
        not on either one of them."""
        from isaacsim.core.utils.xforms import get_world_pose

        robot = runtime.robot
        cfg = runtime.config.robot
        left, _ = get_world_pose(f"{cfg.prim_path}/{cfg.left_finger_prim}")
        right, _ = get_world_pose(f"{cfg.prim_path}/{cfg.right_finger_prim}")
        midpoint = (np.asarray(left) + np.asarray(right)) / 2.0

        tcp = robot.tcp_pose().position
        lateral = np.linalg.norm(tcp - midpoint)
        assert lateral < 0.06, (
            f"TCP is {lateral * 1000:.1f} mm from the finger midpoint; the TCP frame is wrong"
        )

    def test_tcp_differs_materially_from_the_finger_frame(self, runtime):
        """Regression guard. If someone 'simplifies' tcp_pose() back to the
        finger prim, this fails loudly instead of silently missing every grasp.
        """
        from isaacsim.core.utils.xforms import get_world_pose

        cfg = runtime.config.robot
        finger, _ = get_world_pose(f"{cfg.prim_path}/{cfg.right_finger_prim}")
        error = float(np.linalg.norm(runtime.robot.tcp_pose().position - np.asarray(finger)))
        assert error > 0.03, (
            f"TCP is only {error * 1000:.1f} mm from panda_rightfinger; it appears to be "
            "using the finger frame rather than the fingertip midpoint"
        )

    def test_fk_matches_live_tcp_pose(self, runtime):
        """FK at the current configuration must agree with the measured TCP.

        This is the check that catches a mismatch between the Lula URDF and the
        USD articulation -- the two describe the same robot and must not drift.
        """
        robot = runtime.robot
        fk = robot.forward_kinematics(robot.get_arm_joint_positions())
        live = robot.tcp_pose()

        assert fk.translation_distance(live) < 0.01, (
            f"FK/live TCP disagree by {fk.translation_distance(live) * 1000:.1f} mm"
        )
        assert fk.angular_distance(live) < 0.05


class TestKinematics:
    def test_ik_round_trips_through_fk(self, runtime):
        """IK of an FK-generated pose must return to that pose.

        Uses a perturbed-but-reachable configuration rather than a hand-written
        target, so the test cannot fail merely because a chosen pose was
        unreachable.
        """
        robot = runtime.robot
        seed = robot.get_arm_joint_positions()
        perturbed = seed + np.array([0.15, 0.1, -0.1, 0.12, 0.05, -0.08, 0.2])

        target = robot.forward_kinematics(perturbed)
        solution = robot.inverse_kinematics(target, seed=seed)
        assert solution is not None, "IK failed on a pose generated by FK"

        achieved = robot.forward_kinematics(solution)
        assert achieved.translation_distance(target) < 5e-3
        assert achieved.angular_distance(target) < 0.05

    def test_ik_returns_none_for_unreachable_target(self, runtime):
        """Far outside the workspace must fail cleanly, not raise or return junk."""
        from mfw.core.types import Pose

        target = Pose(np.array([5.0, 5.0, 5.0]), np.array([1.0, 0.0, 0.0, 0.0]), Frame.WORLD)
        assert runtime.robot.inverse_kinematics(target) is None

    def test_ik_target_is_interpreted_as_tcp_not_hand(self, runtime):
        """Guards the TCP->hand conversion inside IK.

        If IK forgot to convert, the achieved TCP would sit ~103 mm from the
        request -- exactly the offset length.
        """
        robot = runtime.robot
        seed = robot.get_arm_joint_positions()
        target = robot.forward_kinematics(seed + np.array([0.1, 0.05, 0.0, 0.1, 0.0, 0.0, 0.0]))

        solution = robot.inverse_kinematics(target, seed=seed)
        assert solution is not None
        error = robot.forward_kinematics(solution).translation_distance(target)
        offset_len = float(np.linalg.norm(runtime.config.robot.tcp_offset_from_hand))
        assert error < offset_len / 4.0, (
            f"IK error {error * 1000:.1f} mm is comparable to the TCP offset "
            f"({offset_len * 1000:.1f} mm); the TCP->hand conversion looks wrong"
        )


class TestCameras:
    def test_both_cameras_exist(self, runtime):
        """oxe_droid requires an exterior *and* a wrist view."""
        assert set(runtime.cameras) == {"wrist_camera", "exterior_camera"}

    @pytest.mark.parametrize("cam_name", ["wrist_camera", "exterior_camera"])
    def test_rgb_is_populated(self, runtime, cam_name):
        frame = runtime.cameras[cam_name].capture()
        width, height = runtime.cameras[cam_name].config.resolution
        assert frame.rgb.shape == (height, width, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.rgb.std() > 1.0, "image is uniform; the camera is likely rendering nothing"

    @pytest.mark.parametrize("cam_name", ["wrist_camera", "exterior_camera"])
    def test_depth_is_metric_and_plausible(self, runtime, cam_name):
        frame = runtime.cameras[cam_name].capture()
        assert frame.depth is not None
        finite = frame.depth[np.isfinite(frame.depth)]
        assert finite.size > 0, "no finite depth samples"
        assert finite.min() > 0.0
        near, far = runtime.cameras[cam_name].config.clipping_range
        assert finite.max() <= far * 1.05, f"depth {finite.max():.2f} exceeds far clip {far}"

    def test_intrinsics_derived_from_lens(self, runtime):
        cam = runtime.cameras["exterior_camera"]
        intr = cam.get_intrinsics()
        width, height = cam.config.resolution

        expected_fx = cam.config.focal_length * width / cam.config.horizontal_aperture
        assert intr.fx == pytest.approx(expected_fx, rel=1e-6)
        assert intr.cx == pytest.approx(width / 2.0)
        assert intr.cy == pytest.approx(height / 2.0)

        k = intr.as_matrix()
        assert k.shape == (3, 3) and k[2, 2] == 1.0

    def test_segmentation_reports_semantic_classes(self, runtime):
        """Objects must carry semantics or perception is structurally blind.

        Checks the resolved *class* names ("can", "block"), not prim paths.
        Isaac's instance annotator reports paths; language grounding needs
        classes, so the camera resolves one to the other.
        """
        frame = runtime.cameras["exterior_camera"].capture()
        assert frame.segmentation is not None
        assert frame.segmentation.shape[:2] == frame.rgb.shape[:2]

        labels = {v.lower() for v in frame.seg_id_to_label.values()}
        expected = {(o.semantic_label or o.name).lower() for o in runtime.config.scene.objects}
        assert expected & labels, (
            f"none of {sorted(expected)} appear in resolved classes {sorted(labels)}"
        )

    def test_segmentation_keeps_instance_identity(self, runtime):
        """Two objects of the same class must remain distinguishable.

        The class label alone cannot separate two identical cans; the prim path
        can, so both mappings are carried.
        """
        frame = runtime.cameras["exterior_camera"].capture()
        object_ids = [
            seg_id
            for seg_id, path in frame.seg_id_to_prim.items()
            if path.startswith("/World/objects/")
        ]
        assert object_ids, "no object instances in the segmentation map"
        paths = {frame.seg_id_to_prim[i] for i in object_ids}
        assert len(paths) == len(object_ids), "distinct instances collapsed to one prim path"

    def test_projection_agrees_with_depth_and_segmentation(self, runtime):
        """End-to-end calibration check, and the regression guard for a real bug.

        Projects each object's ground-truth position through the camera's
        intrinsics and extrinsics, then asserts the depth buffer *and* the
        segmentation mask agree at that exact pixel. Only a correct intrinsics,
        extrinsics and axis-convention chain can satisfy all three at once.

        The bug this pins: Isaac's ``Camera`` interprets pose arguments in its
        own ``camera_axes`` convention (default +X forward), not the USD
        convention (-Z forward) our look-at maths produces. Passing a
        USD-convention quaternion through it silently aimed the camera at the
        horizon while every pose query still echoed back the requested value.
        Only comparing against rendered output catches that.

        Reading ground-truth poses is legitimate *in a test*; runtime code has
        no accessor for them.
        """
        from isaacsim.core.utils.xforms import get_world_pose

        from mfw.utils import transforms as tf_mod

        frame = runtime.cameras["exterior_camera"].capture()
        k = frame.intrinsics.as_matrix()
        world_to_camera = tf_mod.invert_transform(frame.extrinsics.as_matrix())

        checked = 0
        for name, path in runtime.scene_builder.object_prim_paths.items():
            world_position = np.asarray(get_world_pose(path)[0], dtype=np.float64)
            camera_point = (world_to_camera @ np.append(world_position, 1.0))[:3]
            assert camera_point[2] > 0, f"{name} projected behind the camera"

            pixel = k @ camera_point
            u, v = int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))
            if not (0 <= u < frame.intrinsics.width and 0 <= v < frame.intrinsics.height):
                continue

            measured_depth = float(frame.depth[v, u])
            assert np.isfinite(measured_depth), (
                f"{name} projects to ({u},{v}) but depth there is NaN; the camera is "
                "not actually looking where the extrinsics claim"
            )
            # Tolerance covers the offset between an object's centre and its
            # front surface, which is what the depth buffer actually samples.
            assert abs(measured_depth - camera_point[2]) < 0.15, (
                f"{name}: projected depth {camera_point[2]:.3f} vs measured "
                f"{measured_depth:.3f} at pixel ({u},{v})"
            )

            seg_prim = frame.seg_id_to_prim.get(int(frame.segmentation[v, u]), "")
            assert name in seg_prim, (
                f"{name} projects to ({u},{v}) but segmentation reports {seg_prim!r}"
            )
            checked += 1

        assert checked >= 2, f"only {checked} objects were in frame; test is inconclusive"

    def test_wrist_camera_moves_with_the_arm(self, runtime):
        """A parented camera must track the hand.

        If it were placed by world pose instead of a local transform it would
        stay put, and every wrist observation would be taken from a stale
        viewpoint.
        """
        robot = runtime.robot
        before_cam = runtime.cameras["wrist_camera"].get_extrinsics().position.copy()
        before_hand = robot.hand_pose().position.copy()

        target = robot.get_arm_joint_positions() + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.set_arm_joint_targets(target)
        runtime.sim.step(120, render=True)

        after_cam = runtime.cameras["wrist_camera"].get_extrinsics().position
        after_hand = robot.hand_pose().position

        hand_moved = float(np.linalg.norm(after_hand - before_hand))
        cam_moved = float(np.linalg.norm(after_cam - before_cam))
        assert hand_moved > 0.02, "arm did not actually move; test is inconclusive"
        assert cam_moved > 0.5 * hand_moved, (
            f"hand moved {hand_moved * 1000:.0f} mm but wrist camera moved only "
            f"{cam_moved * 1000:.0f} mm; the camera is not parented to the hand"
        )


class TestPointCloud:
    def test_deprojection_produces_world_points_on_the_table(self, runtime):
        """The point cloud must land on real geometry.

        Catches the classic optical-convention error: a flipped Y axis mirrors
        the cloud and puts the table below the floor.
        """
        from mfw.vision.camera import deproject_to_point_cloud

        frame = runtime.cameras["exterior_camera"].capture()
        cloud = deproject_to_point_cloud(frame, runtime.config.perception, to_world=True)

        assert len(cloud) > 1000, f"only {len(cloud)} points deprojected"
        assert cloud.frame is Frame.WORLD

        zs = cloud.points[:, 2]
        assert zs.min() > -0.05, f"points below the floor (min z={zs.min():.3f}); axes look flipped"

        table_top = runtime.config.scene.table_position[2] + runtime.config.scene.table_scale[2] / 2.0
        near_table = np.abs(zs - table_top) < 0.05
        assert near_table.sum() > 200, (
            f"only {near_table.sum()} points near the table top at z={table_top:.3f}"
        )

    def test_invalid_depth_is_dropped_not_clamped(self, runtime):
        """Clamping would invent surfaces at the range limit for the planner."""
        from mfw.vision.camera import deproject_to_point_cloud

        frame = runtime.cameras["exterior_camera"].capture()
        cloud = deproject_to_point_cloud(frame, runtime.config.perception, to_world=False)

        depths = cloud.points[:, 2]
        assert np.all(np.isfinite(cloud.points))
        assert depths.min() > runtime.config.perception.depth_min
        assert depths.max() < runtime.config.perception.depth_max


class TestPhysics:
    def test_objects_settle_and_rest_on_the_table(self, runtime):
        """Pure physics: objects fall and come to rest with no pose writing."""
        from isaacsim.core.utils.xforms import get_world_pose

        runtime.sim.step(120, render=False)
        table_top = runtime.config.scene.table_position[2] + runtime.config.scene.table_scale[2] / 2.0

        for name, path in runtime.scene_builder.object_prim_paths.items():
            position, _ = get_world_pose(path)
            z = float(position[2])
            assert z > table_top - 0.02, f"{name} fell through the table (z={z:.3f})"
            assert z < table_top + 0.3, f"{name} is floating (z={z:.3f})"

    def test_objects_are_at_rest_after_settling(self, runtime):
        """A still-falling object measured by perception is wrong by the time
        the arm arrives, so settling has to actually converge."""
        from isaacsim.core.utils.xforms import get_world_pose

        runtime.sim.step(60, render=False)
        before = {
            name: np.asarray(get_world_pose(path)[0])
            for name, path in runtime.scene_builder.object_prim_paths.items()
        }
        runtime.sim.step(60, render=False)

        for name, path in runtime.scene_builder.object_prim_paths.items():
            drift = float(np.linalg.norm(np.asarray(get_world_pose(path)[0]) - before[name]))
            assert drift < 5e-3, f"{name} still moving after settling (drift {drift * 1000:.1f} mm)"

    def test_gripper_opens_and_closes(self, runtime):
        robot = runtime.robot

        robot.open_gripper()
        runtime.sim.step(120, render=False)
        opened = robot.get_gripper_width()
        assert opened > runtime.config.robot.gripper_open_width * 0.7, (
            f"gripper did not open (width {opened:.4f})"
        )

        robot.close_gripper()
        runtime.sim.step(120, render=False)
        assert robot.get_gripper_width() < opened - 0.01, "gripper did not close"

    def test_free_close_reports_no_grasp(self, runtime):
        """Closing on air must report is_grasping False.

        The whole point of physics-derived grasp detection: a commanded close is
        not evidence of holding anything.
        """
        robot = runtime.robot
        robot.close_gripper()
        runtime.sim.step(150, render=False)
        assert not robot.get_gripper_state().is_grasping

    def test_physx_scene_uses_tgs_solver(self, runtime):
        """TGS and high iteration counts are what make contact grasping stable."""
        from pxr import PhysxSchema, UsdPhysics

        stage = runtime.sim.world.stage
        scene_prim = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
        assert scene_prim is not None, "no physics scene on the stage"

        physx = PhysxSchema.PhysxSceneAPI(scene_prim)
        assert physx.GetSolverTypeAttr().Get() == "TGS"

    def test_object_rigid_bodies_configured(self, runtime):
        from pxr import PhysxSchema

        stage = runtime.sim.world.stage
        cfg = runtime.config.physics
        for name, path in runtime.scene_builder.object_prim_paths.items():
            prim = stage.GetPrimAtPath(path)
            rb = PhysxSchema.PhysxRigidBodyAPI(prim)
            iterations = rb.GetSolverPositionIterationCountAttr().Get()
            assert iterations == cfg.solver_position_iterations, (
                f"{name} has {iterations} position iterations, expected "
                f"{cfg.solver_position_iterations}"
            )

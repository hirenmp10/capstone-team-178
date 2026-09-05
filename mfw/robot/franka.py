"""Franka Panda robot abstraction.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

Why Franka
----------
Two properties of *this* install make it the right choice, both verifiable:

1. ``exts/isaacsim.robot_motion.motion_generation/path_planner_configs/``
   contains exactly one robot -- ``franka``. It is the only arm shipping a Lula
   RRT configuration, and global collision-aware planning is a hard requirement
   here. RMPflow configs exist for 21 robots; RRT does not.
2. GR00T N1.7's only inference-ready manipulation embodiment tag,
   ``oxe_droid_relative_eef_relative_joint``, is trained on DROID -- a Franka
   Panda with a parallel gripper and a wrist camera. Any other arm would require
   finetuning before the policy backend could run at all.

The TCP problem
---------------
Isaac's shipped Franka example sets ``end_effector_prim_path`` to
``panda_rightfinger``. Measured on this install via Lula FK, that frame sits at
``[0, -0.025, 0.0584]`` in the hand frame, while the true fingertip midpoint is
at ``[0, 0, 0.1034]`` -- an error of **51.5 mm**. Planning grasps against the
finger frame drives the gripper 5 cm short and to one side of where it thinks
it is, so grasps close on air while every log line still reports success.

This class therefore derives the TCP from ``panda_hand`` plus a configured
offset, and ``tcp_pose()`` is the only end-effector pose the rest of the
framework is allowed to see.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import PhysicsConfig, RobotConfig
from mfw.core.errors import KinematicsError, SimulationError, UnreachableTarget
from mfw.core.interfaces import IRobot
from mfw.core.types import Frame, GripperState, JointState, Pose, RobotState
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = ["FrankaRobot"]

_log = get_logger("robot.franka")


class FrankaRobot(IRobot):
    """Franka Panda with a derived, correct TCP frame.

    Construct after the stage exists but before ``World.reset()``; call
    :meth:`initialize` after the reset, once physics handles are live.
    """

    def __init__(
        self,
        sim: Any,
        config: RobotConfig,
        physics_config: PhysicsConfig,
    ) -> None:
        self.config = config
        self.physics_config = physics_config
        self._sim = sim
        self._articulation: Any = None
        self._kinematics: Any = None
        self._arm_dof_indices: NDArray[np.int64] | None = None
        self._finger_dof_indices: NDArray[np.int64] | None = None
        self._initialized = False
        self._contact_view: Any = None
        # Tracks the last commanded width. Grasp detection compares actual width
        # against this: fingers that stalled short of the commanded close are
        # holding something.
        self._target_width: float = config.gripper_open_width

        self._spawn()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _spawn(self) -> None:
        """Add the Franka USD to the stage and register the articulation."""
        from isaacsim.core.prims import SingleArticulation  # noqa: PLC0415
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: PLC0415
        from isaacsim.storage.native import get_assets_root_path  # noqa: PLC0415

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise SimulationError(
                "Isaac Sim assets root is unreachable. The Franka USD is served from "
                "the Omniverse content bucket; check network access or configure a "
                "local asset root."
            )

        usd_path = assets_root + self.config.usd_subpath
        _log.info("Loading Franka from %s", usd_path)
        add_reference_to_stage(usd_path=usd_path, prim_path=self.config.prim_path)

        self._articulation = SingleArticulation(
            prim_path=self.config.prim_path,
            name=self.config.name,
            position=np.array(self.config.base_position, dtype=np.float64),
            orientation=np.array(self.config.base_quat, dtype=np.float64),
        )
        self._sim.world.scene.add(self._articulation)

    def initialize(self) -> None:
        """Resolve DOF indices and build the kinematics solver.

        Must run *after* ``World.reset()``: Isaac only creates the physics views
        during reset, so ``dof_names`` is empty before then and every index
        lookup below would silently resolve to nothing.
        """
        if self._initialized:
            return

        dof_names = list(self._articulation.dof_names or [])
        if not dof_names:
            raise SimulationError(
                "Articulation reports no DOFs. initialize() must be called after "
                "SimulationContext.reset(), which creates the physics views."
            )

        self._arm_dof_indices = self._resolve_indices(dof_names, self.config.arm_joint_names, "arm")
        self._finger_dof_indices = self._resolve_indices(
            dof_names, self.config.finger_joint_names, "finger"
        )

        self._build_kinematics()
        self._apply_gripper_force_limits()
        self._build_contact_view()

        self._initialized = True
        _log.info(
            "Franka initialised: %d arm DOFs, %d finger DOFs, TCP offset %s from %s",
            len(self._arm_dof_indices),
            len(self._finger_dof_indices),
            self.config.tcp_offset_from_hand,
            self.config.tcp_parent_prim,
        )

    @staticmethod
    def _resolve_indices(
        dof_names: list[str], wanted: tuple[str, ...], label: str
    ) -> NDArray[np.int64]:
        """Map configured joint names to articulation DOF indices.

        Explicit rather than assuming order: the articulation's DOF ordering is
        an artefact of the USD, and silently mis-indexing joints produces motion
        that looks like a controller bug.
        """
        indices = []
        for name in wanted:
            if name not in dof_names:
                raise SimulationError(
                    f"{label} joint {name!r} not found in articulation. Available DOFs: {dof_names}"
                )
            indices.append(dof_names.index(name))
        return np.array(indices, dtype=np.int64)

    def _build_kinematics(self) -> None:
        """Load the Lula kinematics solver shipped with Isaac Sim for Franka."""
        from isaacsim.robot_motion.motion_generation import (  # noqa: PLC0415
            LulaKinematicsSolver,
            interface_config_loader,
        )

        if self.config.lula_robot_description and self.config.lula_urdf:
            kin_config = {
                "robot_description_path": self.config.lula_robot_description,
                "urdf_path": self.config.lula_urdf,
            }
        else:
            kin_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")

        self._kinematics = LulaKinematicsSolver(**kin_config)
        # Lula works in the robot's base frame; telling it where the base sits
        # lets us hand it world-frame targets directly.
        self._kinematics.set_robot_base_pose(
            robot_position=np.array(self.config.base_position, dtype=np.float64),
            robot_orientation=np.array(self.config.base_quat, dtype=np.float64),
        )

        frames = self._kinematics.get_all_frame_names()
        if self.config.tcp_parent_prim not in frames:
            raise KinematicsError(
                f"TCP parent frame {self.config.tcp_parent_prim!r} is not known to the Lula "
                f"solver. Available frames: {frames}"
            )

    def _apply_gripper_force_limits(self) -> None:
        """Cap the finger drives' force so closing is force-limited, not positional.

        This is what makes pure-physics grasping work. Commanding the fingers to
        a fully closed *position* against a rigid object makes the drive fight
        the contact constraint at unbounded force; the solver resolves the
        conflict by ejecting the object. Capping ``maxForce`` turns the same
        command into a bounded squeeze that settles into a stable grasp.
        """
        from pxr import UsdPhysics  # noqa: PLC0415

        stage = self._sim.world.stage
        for joint_name in self.config.finger_joint_names:
            joint_path = f"{self.config.prim_path}/{self.config.tcp_parent_prim}/{joint_name}"
            prim = stage.GetPrimAtPath(joint_path)
            if not prim or not prim.IsValid():
                prim = self._find_joint_prim(stage, joint_name)
            if prim is None:
                _log.warning("Could not locate drive for finger joint %s", joint_name)
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateMaxForceAttr().Set(float(self.config.gripper_force))

    def _build_contact_view(self) -> None:
        """Create a contact-tracking view over the fingertips, once.

        ``track_contact_forces=True`` is required at construction -- without it
        ``get_net_contact_forces()`` returns nothing and Isaac logs "contact
        forces cannot be retrieved with this API unless the RigidPrim is
        initialized with track_contact_forces=True" on every call.

        Built once here rather than per query for two reasons: constructing a
        physics view inside the verification path logged that warning twice per
        pick, and a view created on the fly has no contact history to report
        anyway. With this in place, contact force becomes a genuine third
        verification signal alongside finger stall and object tracking.
        """
        from isaacsim.core.prims import RigidPrim  # noqa: PLC0415

        try:
            view = RigidPrim(
                prim_paths_expr=f"{self.config.prim_path}/panda_(left|right)finger",
                name=f"{self.config.name}_finger_contacts",
                track_contact_forces=True,
                prepare_contact_sensors=True,
            )
            view.initialize()
            self._contact_view = view
            _log.info("Finger contact tracking enabled")
        except Exception as exc:
            # Not fatal: grasp verification's primary evidence is finger stall
            # plus the object still tracking with the TCP, both of which are
            # independent of contact sensors.
            self._contact_view = None
            _log.info("Finger contact tracking unavailable (%s); using stall + tracking only", exc)

    def net_contact_force(self) -> float | None:
        """Total contact force magnitude on the fingertips, in newtons.

        ``None`` when contact tracking is unavailable -- deliberately distinct
        from ``0.0``, since "no sensor" and "no contact" are different facts and
        conflating them would let a missing sensor read as a confirmed empty
        gripper.
        """
        if not self._initialized or self._contact_view is None:
            return None
        try:
            forces = self._contact_view.get_net_contact_forces()
            if forces is None:
                return None
            return float(np.sum(np.linalg.norm(np.asarray(forces), axis=-1)))
        except Exception:  # pragma: no cover - depends on physics view state
            return None

    def _find_joint_prim(self, stage: Any, joint_name: str) -> Any:
        """Locate a joint prim by name anywhere under the robot."""
        from pxr import UsdPhysics  # noqa: PLC0415

        root = stage.GetPrimAtPath(self.config.prim_path)
        if not root or not root.IsValid():
            return None
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(self.config.prim_path):
                continue
            if prim.GetName() == joint_name and prim.IsA(UsdPhysics.PrismaticJoint):
                return prim
        return None

    def _require_init(self) -> None:
        if not self._initialized:
            raise SimulationError("FrankaRobot.initialize() has not been called")

    # ------------------------------------------------------------------
    # IRobot
    # ------------------------------------------------------------------

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.config.arm_joint_names)

    @property
    def articulation(self) -> Any:
        """Underlying Isaac articulation. For the controller layer only."""
        self._require_init()
        return self._articulation

    @property
    def arm_dof_indices(self) -> NDArray[np.int64]:
        self._require_init()
        assert self._arm_dof_indices is not None
        return self._arm_dof_indices

    @property
    def finger_dof_indices(self) -> NDArray[np.int64]:
        self._require_init()
        assert self._finger_dof_indices is not None
        return self._finger_dof_indices

    def get_arm_joint_positions(self) -> NDArray[np.float64]:
        self._require_init()
        return np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)[
            self.arm_dof_indices
        ]

    def get_state(self) -> RobotState:
        """Full proprioceptive snapshot."""
        self._require_init()
        all_pos = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        all_vel = np.asarray(self._articulation.get_joint_velocities(), dtype=np.float64)

        joint_state = JointState(
            positions=all_pos[self.arm_dof_indices],
            velocities=all_vel[self.arm_dof_indices],
            names=tuple(self.config.arm_joint_names),
        )
        return RobotState(
            joint_state=joint_state,
            tcp_pose=self.tcp_pose(),
            gripper=self.get_gripper_state(),
            sim_time=self._sim.sim_time,
            step_index=self._sim.step_index,
        )

    def tcp_pose(self) -> Pose:
        """World pose of the true tool centre point.

        Reads the live ``panda_hand`` transform from the stage (so it reflects
        actual physics, not a commanded target) and applies the configured
        offset in the hand's local frame.

        Deliberately does not use ``panda_rightfinger``: measured on this
        install, that frame is 51.5 mm from the fingertip midpoint.
        """
        self._require_init()
        from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

        hand_path = f"{self.config.prim_path}/{self.config.tcp_parent_prim}"
        position, quat = get_world_pose(hand_path)

        hand_matrix = tf.make_transform(np.asarray(position), np.asarray(quat))
        offset = np.eye(4)
        offset[:3, 3] = np.asarray(self.config.tcp_offset_from_hand, dtype=np.float64)
        return Pose.from_matrix(hand_matrix @ offset, Frame.WORLD)

    def hand_pose(self) -> Pose:
        """World pose of ``panda_hand`` itself, for IK targets and camera mounting."""
        self._require_init()
        from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

        position, quat = get_world_pose(f"{self.config.prim_path}/{self.config.tcp_parent_prim}")
        return Pose(np.asarray(position), np.asarray(quat), Frame.WORLD)

    def forward_kinematics(self, joint_positions: NDArray[np.float64]) -> Pose:
        """TCP pose for a hypothetical configuration, without moving the robot.

        Used by grasp scoring and by the planner to evaluate candidates, so it
        must not disturb simulation state.
        """
        self._require_init()
        q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self.config.arm_joint_names):
            raise KinematicsError(
                f"forward_kinematics expects {len(self.config.arm_joint_names)} joint values, "
                f"got {q.shape[0]}"
            )

        position, rotation = self._kinematics.compute_forward_kinematics(
            self.config.tcp_parent_prim, q
        )
        hand = np.eye(4)
        hand[:3, :3] = np.asarray(rotation, dtype=np.float64)
        hand[:3, 3] = np.asarray(position, dtype=np.float64)

        offset = np.eye(4)
        offset[:3, 3] = np.asarray(self.config.tcp_offset_from_hand, dtype=np.float64)
        return Pose.from_matrix(hand @ offset, Frame.WORLD)

    def inverse_kinematics(
        self, target: Pose, seed: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64] | None:
        """Joint solution placing the **TCP** at ``target``.

        The target is converted from TCP to ``panda_hand`` before solving, since
        Lula has no frame at the fingertip midpoint. Skipping that conversion is
        the same 51.5 mm error as using the finger frame.
        """
        self._require_init()
        if target.frame is not Frame.WORLD:
            raise KinematicsError(
                f"IK targets must be in the world frame, got {target.frame.value}"
            )

        offset_inv = np.eye(4)
        offset_inv[:3, 3] = -np.asarray(self.config.tcp_offset_from_hand, dtype=np.float64)
        hand_target = target.as_matrix() @ offset_inv
        hand_position, hand_quat = tf.matrix_to_pose(hand_target)

        warm_start = (
            np.asarray(seed, dtype=np.float64)
            if seed is not None
            else self.get_arm_joint_positions()
        )

        solution, success = self._kinematics.compute_inverse_kinematics(
            frame_name=self.config.tcp_parent_prim,
            target_position=hand_position,
            target_orientation=hand_quat,
            warm_start=warm_start,
        )
        if not success:
            return None
        return np.asarray(solution, dtype=np.float64)

    def inverse_kinematics_or_raise(
        self, target: Pose, seed: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        solution = self.inverse_kinematics(target, seed)
        if solution is None:
            raise UnreachableTarget(
                f"No IK solution for TCP target at {np.round(target.position, 4).tolist()}"
            )
        return solution

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def get_gripper_width(self) -> float:
        """Current finger separation in metres.

        The two prismatic joints each travel half the span, so the width is
        their sum.
        """
        self._require_init()
        positions = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        return float(np.sum(positions[self.finger_dof_indices]))

    def get_gripper_state(self) -> GripperState:
        """Finger state including physics-derived grasp detection.

        ``is_grasping`` is computed in :mod:`mfw.physics.contact` from actual
        contact forces. Here it is inferred conservatively from finger geometry:
        the fingers stopped short of closed, which means something is between
        them. A commanded close that caught air closes fully and reports False.
        """
        self._require_init()
        width = self.get_gripper_width()
        velocities = np.asarray(self._articulation.get_joint_velocities(), dtype=np.float64)
        finger_speed = float(np.max(np.abs(velocities[self.finger_dof_indices])))

        # The fingers must have stalled on something *of real size*.
        #
        # A 2 mm margin was far too generous: a close onto empty air settles near
        # 3.7 mm through finger compliance and solver residual, which then read as
        # a successful grasp. Measured here, that made an untouched can report
        # is_grasping=True while its height had not changed by a millimetre.
        #
        # min_graspable_width is the narrowest object worth claiming to hold, so
        # anything below it is the fingers meeting each other, not an object.
        blocked_open = width > (
            self.config.gripper_closed_width + self.config.min_graspable_width
        )
        return GripperState(
            width=width,
            target_width=self._target_width,
            is_moving=finger_speed > 1e-3,
            is_grasping=blocked_open and self._target_width <= self.config.gripper_closed_width + 1e-6,
        )

    def open_gripper(self) -> None:
        """Command the fingers open. Atomic: does not move the arm."""
        self._set_gripper_width(self.config.gripper_open_width)

    def close_gripper(self) -> None:
        """Command a force-limited close.

        Commands the fully-closed position; the drive force cap set in
        :meth:`_apply_gripper_force_limits` turns this into a bounded squeeze
        that stalls against whatever is between the fingers.
        """
        self._set_gripper_width(self.config.gripper_closed_width)

    def _set_gripper_width(self, width: float) -> None:
        self._require_init()
        from isaacsim.core.utils.types import ArticulationAction  # noqa: PLC0415

        width = float(np.clip(width, self.config.gripper_closed_width, self.config.gripper_open_width))
        self._target_width = width
        per_finger = width / 2.0

        self._articulation.apply_action(
            ArticulationAction(
                joint_positions=np.array([per_finger, per_finger], dtype=np.float64),
                joint_indices=self.finger_dof_indices,
            )
        )

    # ------------------------------------------------------------------
    # posture
    # ------------------------------------------------------------------

    def set_arm_joint_targets(self, positions: NDArray[np.float64]) -> None:
        """Command arm joint position targets. Controller layer only."""
        self._require_init()
        from isaacsim.core.utils.types import ArticulationAction  # noqa: PLC0415

        q = np.asarray(positions, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self.arm_dof_indices):
            raise SimulationError(
                f"Expected {len(self.arm_dof_indices)} arm targets, got {q.shape[0]}"
            )
        self._articulation.apply_action(
            ArticulationAction(joint_positions=q, joint_indices=self.arm_dof_indices)
        )

    def go_home_immediate(self) -> None:
        """Snap to the home posture.

        Teleports joints, so it is only legal during scene setup, before any
        manipulation. Never call it while holding an object.
        """
        self._require_init()
        full = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        full[self.arm_dof_indices] = np.asarray(self.config.home_joint_positions, dtype=np.float64)
        full[self.finger_dof_indices] = self.config.gripper_open_width / 2.0
        self._articulation.set_joint_positions(full)
        self._articulation.set_joint_velocities(np.zeros_like(full))
        self._target_width = self.config.gripper_open_width

    def finger_prim_paths(self) -> list[str]:
        """Absolute paths of the finger links, for physics material binding."""
        return [
            f"{self.config.prim_path}/{self.config.left_finger_prim}",
            f"{self.config.prim_path}/{self.config.right_finger_prim}",
        ]

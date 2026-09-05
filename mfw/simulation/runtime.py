"""Runtime assembly: brings up the simulator, scene, robot and cameras together.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

The bring-up order here is not arbitrary and is the part most likely to be got
wrong when wiring Isaac Sim by hand:

1. ``SimulationApp`` -- must precede every Isaac import in the process.
2. Scene prims and the robot USD -- added to the stage while it is still
   editable.
3. ``World.reset()`` -- Isaac creates its physics views and render products
   here. Articulation DOFs and camera annotators do not exist before this point.
4. ``initialize()`` on the robot and cameras -- only now can DOF indices be
   resolved and annotators attached.
5. Settle -- objects spawn slightly above their support and must come to rest
   before anything measures them.

Calling step 4 before step 3 yields an articulation with zero DOFs and cameras
that silently return empty buffers, which is why this sequence is encapsulated
rather than left to each entry point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mfw.config.schema import FrameworkConfig
from mfw.controllers.joint_controller import JointTrajectoryController
from mfw.core.errors import PerceptionError, SimulationError
from mfw.grasp.scorer import GraspScorer
from mfw.memory.working_memory import WorkingMemory
from mfw.motion.planner import LulaMotionPlanner
from mfw.physics.materials import configure_gripper_contacts
from mfw.robot.franka import FrankaRobot
from mfw.simulation.app import SimulationContext
from mfw.simulation.asset_registry import AssetRegistry
from mfw.simulation.layout import randomize_scene
from mfw.simulation.scene import SceneBuilder
from mfw.skills.base import SkillContext
from mfw.skills.registry import ClassicalExecutor, SkillRegistry
from mfw.utils.logging import EventLogger, get_logger
from mfw.vision.camera import Camera
from mfw.vision.manager import VisionManager

__all__ = ["Runtime"]

_log = get_logger("simulation.runtime")


class Runtime:
    """Owns the simulator and every hardware-facing object built on top of it."""

    def __init__(self, config: FrameworkConfig, event_logger: EventLogger | None = None) -> None:
        config.validate()
        self.config = config
        self.events = event_logger or EventLogger(
            log_dir=config.logging.log_dir,
            filename=config.logging.jsonl_filename,
            flush_every=config.logging.flush_every,
            console_level=config.logging.level,
            console=config.logging.console,
        )

        self.sim = SimulationContext(config.simulation)

        # Randomisation is resolved here, once, before anything is spawned. The
        # resulting layout is a plain config that gets logged with the run, so a
        # randomised episode can be replayed exactly -- a randomised run that
        # cannot be reproduced cannot be debugged.
        self.asset_registry = AssetRegistry.load(config.scene.assets_path or None)
        self.scene_config = randomize_scene(config.scene, self.asset_registry)
        if config.scene.randomization.enabled:
            self.events.emit(
                "scene.randomized",
                {
                    "seed": config.scene.randomization.seed,
                    "objects": [
                        {"name": o.name, "asset": o.asset, "position": list(o.position)}
                        for o in self.scene_config.objects
                    ],
                },
            )

        self.scene_builder = SceneBuilder(
            self.sim, self.scene_config, config.physics, registry=self.asset_registry
        )
        self.robot: FrankaRobot | None = None
        self.wrist_camera: Camera | None = None
        self.exterior_camera: Camera | None = None
        self.vision: VisionManager | None = None
        self.planner: LulaMotionPlanner | None = None
        self.controller: JointTrajectoryController | None = None
        self.grasp_scorer: GraspScorer | None = None
        self.memory: WorkingMemory | None = None
        self.skills: SkillRegistry | None = None
        self.executor: ClassicalExecutor | None = None
        self._built = False

    @property
    def table_top_height(self) -> float:
        """Height of the support surface, derived from scene configuration.

        The one place a configured scene dimension is read at runtime, and only
        as a *prior* for grasp clearance -- perception still decides where objects
        are. Without it the scorer would have to assume the floor and would allow
        grasps that drive the hand into the tabletop.
        """
        if not self.config.scene.add_table:
            return float(self.config.perception.ground_plane_z)
        return float(
            self.config.scene.table_position[2] + self.config.scene.table_scale[2] / 2.0
        )

    def build(self) -> None:
        """Run the full bring-up sequence. Safe to call once."""
        if self._built:
            return

        with self.events.timed("runtime.build"):
            self.scene_builder.build()

            self.robot = FrankaRobot(self.sim, self.config.robot, self.config.physics)
            self.wrist_camera = Camera(self.sim, self.config.wrist_camera, self.config.perception)
            self.exterior_camera = Camera(
                self.sim, self.config.exterior_camera, self.config.perception
            )

            # Physics views and render products come into existence here.
            self.sim.reset()

            self.robot.initialize()
            self.wrist_camera.initialize()
            self.exterior_camera.initialize()

            # Needs the finger prims to exist, so it follows robot init.
            configure_gripper_contacts(
                self.sim.world.stage, self.config.physics, self.robot.finger_prim_paths()
            )

            self.robot.go_home_immediate()
            self.sim.settle()
            self._warm_up_cameras()

            # Built after the robot is live: the planner needs the articulation
            # for its path visualiser, and perception needs initialised cameras.
            self.vision = VisionManager(
                sim=self.sim,
                cameras=self.cameras,
                config=self.config.perception,
                event_logger=self.events,
                # Objects rest on the table, not the floor, and the halo filter
                # is only useful when aimed at the surface they actually sit on.
                support_height=self.table_top_height,
            )
            self.planner = LulaMotionPlanner(
                sim=self.sim,
                robot=self.robot,
                config=self.config.motion,
                event_logger=self.events,
            )
            self.controller = JointTrajectoryController(
                sim=self.sim,
                robot=self.robot,
                motion_config=self.config.motion,
                robot_config=self.config.robot,
                workspace_min=np.array(self.config.scene.workspace_min),
                workspace_max=np.array(self.config.scene.workspace_max),
                event_logger=self.events,
            )
            self.grasp_scorer = GraspScorer(
                robot=self.robot,
                config=self.config.grasp,
                support_height=self.table_top_height,
                hand_setback=float(np.linalg.norm(self.config.robot.tcp_offset_from_hand)),
            )
            self.memory = WorkingMemory(self.config.memory)

            # The registry owns the skills; skills receive the context but not the
            # registry, so no skill can reach another. That is what makes
            # atomicity structural rather than a convention.
            self.skills = SkillRegistry(
                SkillContext(
                    sim=self.sim,
                    robot=self.robot,
                    vision=self.vision,
                    planner=self.planner,
                    controller=self.controller,
                    grasp_scorer=self.grasp_scorer,
                    memory=self.memory,
                    config=self.config,
                    events=self.events,
                    support_height=self.table_top_height,
                )
            )
            self.executor = ClassicalExecutor(self.skills)

        self._built = True
        self.events.emit(
            "runtime.ready",
            {
                "objects": list(self.scene_builder.object_prim_paths),
                "arm_dofs": len(self.robot.arm_dof_indices),
                "tcp_pose": self.robot.tcp_pose().to_log(),
            },
        )
        _log.info("Runtime ready")

    _MAX_CAMERA_WARMUP_STEPS = 180

    def _warm_up_cameras(self) -> None:
        """Render until every camera actually yields pixels.

        The render pipeline needs several frames before the first buffer is
        available -- measured on this install, RGB is still empty after 2 render
        steps and populated by ~5. A fixed step count is a guess that silently
        breaks when resolution or renderer changes, so poll instead and fail
        loudly if the pipeline never comes up.
        """
        cameras = list(self.cameras.values())
        if not cameras:
            return

        for step in range(1, self._MAX_CAMERA_WARMUP_STEPS + 1):
            self.sim.render_step(1)
            if all(self._camera_has_data(cam) for cam in cameras):
                _log.info("Cameras produced data after %d render steps", step)
                return

        stalled = [cam.name for cam in cameras if not self._camera_has_data(cam)]
        raise SimulationError(
            f"Cameras {stalled} produced no data after {self._MAX_CAMERA_WARMUP_STEPS} render "
            "steps. The render pipeline failed to start."
        )

    @staticmethod
    def _camera_has_data(camera: Camera) -> bool:
        """Whether a camera is producing *every* annotator it was configured for.

        Checking RGB alone is not enough: the depth and segmentation annotators
        come online a frame or two after colour, so a warm-up that stops at the
        first non-empty RGB buffer hands back frames whose ``depth`` is still
        ``None``. Perception then fails on its very first observation.
        """
        try:
            frame = camera.capture()
        except PerceptionError:
            return False
        if camera.config.enable_depth and frame.depth is None:
            return False
        if camera.config.enable_segmentation and frame.segmentation is None:
            return False
        return True

    @property
    def cameras(self) -> dict[str, Camera]:
        """All cameras by name. The perception layer iterates this."""
        out: dict[str, Camera] = {}
        if self.wrist_camera is not None:
            out[self.wrist_camera.name] = self.wrist_camera
        if self.exterior_camera is not None:
            out[self.exterior_camera.name] = self.exterior_camera
        return out

    def close(self) -> None:
        try:
            self.sim.close()
        finally:
            self.events.close()

    def __enter__(self) -> "Runtime":
        self.build()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

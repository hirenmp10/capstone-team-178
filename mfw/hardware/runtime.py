"""Runtime assembly for the hardware lane: Jetson arm + detector, no simulator.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers; the network clients import pyzmq/msgpack/cv2 lazily.

Mirrors :class:`mfw.simulation.runtime.Runtime` attribute for attribute, because
``Assistant``, ``run_assistant.py`` and ``TaskPlanner`` read the runtime blind:
``config, events, sim, robot, vision, planner, controller, grasp_scorer,
grasp_generator, memory, skills, executor, cameras, table_top_height, build(),
close()`` and context-manager support. Anything the sim runtime exposes that
has no hardware meaning (``scene_builder``, ``wrist_camera``) is deliberately
absent, so a caller that needs it fails at the attribute rather than getting
a plausible fake.

Bring-up order (each step needs the previous one):

1. ``WallClock`` -- the stepping seam every skill and controller uses.
2. ``JetsonClient.connect`` -- proves the robot server answers *before* any
   model, camera or planner is built; a missing server is the most common
   startup failure and should be reported first and plainly.
3. ``PlanarKinematics`` -> ``RemoteArm`` -> ``RemoteCamera``.
4. ``RemoteDetector.connect`` -> ``PlanarPerception``.
5. ``robot.go_home_immediate()`` + settle -- the arm starts from a known
   posture, clear of the camera's view of the table.
6. ``JointSpacePlanner`` -> ``RemoteController`` -> ``GraspScorer`` (reused
   from the sim lane, with ``hand_setback = |tcp_offset_from_hand|``) ->
   ``TopDownGraspGenerator`` -> ``WorkingMemory`` -> ``SkillRegistry`` with
   :data:`HARDWARE_SKILLS` -> ``ClassicalExecutor``.

Refuses to build on an empty ``exterior_camera.homography``: the schema allows
it (``configs/hardware.yaml`` ships it empty on purpose) but an uncalibrated
homography puts every object at a plausible wrong place, and the arm would
then close on air with a confident log.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mfw.config.schema import FrameworkConfig
from mfw.core.errors import ConfigurationError, ExecutionError, PerceptionError
from mfw.grasp.scorer import GraspScorer
from mfw.hardware.clock import WallClock
from mfw.hardware.controller import RemoteController
from mfw.hardware.detector import RemoteDetector
from mfw.hardware.grasp import TopDownGraspGenerator
from mfw.hardware.jetson_client import HEARTBEAT_PERIOD_S, JetsonClient
from mfw.hardware.kinematics import PlanarKinematics
from mfw.hardware.perception import PlanarPerception
from mfw.hardware.planner import JointSpacePlanner
from mfw.hardware.remote_arm import RemoteArm
from mfw.hardware.remote_camera import RemoteCamera
from mfw.memory.working_memory import WorkingMemory
from mfw.skills.base import Skill, SkillContext
from mfw.skills.primitives import ALL_SKILLS, FixedCameraScan, HardwareObserve
from mfw.skills.registry import ClassicalExecutor, SkillRegistry
from mfw.utils.logging import EventLogger, get_logger

__all__ = ["HardwareRuntime", "HARDWARE_SKILLS", "HARDWARE_SKILL_OVERRIDES", "SIM_ONLY_SKILLS"]

_log = get_logger("hardware.runtime")

SIM_ONLY_SKILLS: frozenset[str] = frozenset({"look_at", "rotate_wrist"})
"""Skills that need a wrist camera or a wrist roll joint; this arm has neither."""

HARDWARE_SKILL_OVERRIDES: dict[str, type[Skill]] = {
    # One fixed overhead camera already sees the table: "scan the room" looks
    # (several corroborated frames), sweeps the base visibly toward what it
    # found and reports it with directions. The sim ScanScene's wrist-camera
    # viewpoints have no meaning here.
    "scan_scene": FixedCameraScan,
    # Same scene, reported with where each object is and what is out of reach.
    "observe": HardwareObserve,
}
"""Hardware implementations that replace a sim skill of the same name."""

HARDWARE_SKILLS: tuple[type[Skill], ...] = tuple(
    HARDWARE_SKILL_OVERRIDES.get(cls.skill_name, cls)
    for cls in ALL_SKILLS
    if cls.skill_name not in SIM_ONLY_SKILLS
)
"""The registry's skill classes minus :data:`SIM_ONLY_SKILLS`, with
:data:`HARDWARE_SKILL_OVERRIDES` swapped in by name."""


class HardwareRuntime:
    """Owns the network clients and every framework object built on them."""

    def __init__(self, config: FrameworkConfig, event_logger: EventLogger | None = None) -> None:
        config.validate()
        if config.backend != "hardware":
            raise ConfigurationError(
                f"HardwareRuntime needs backend: hardware, got {config.backend!r}"
            )
        if len(config.exterior_camera.homography) == 0:
            raise ConfigurationError(
                "exterior_camera.homography is empty: the table has not been calibrated. "
                "Run `py -3.12 scripts/calibrate_table.py --jetson <ip>:5560 --config "
                "configs/hardware.yaml --touch` (or `--image frame.png` and type the table "
                "XY of >= 4 clicked points); it writes exterior_camera.homography into the "
                "YAML itself, keeps a .bak, and `--dry-run` only prints the fit. "
                "configs/hardware_fake.yaml carries the scripted detector's exact matrix "
                "(`scripts/serve_detector.py --fake --print-homography`)."
            )
        self.config = config
        self.events = event_logger or EventLogger(
            log_dir=config.logging.log_dir,
            filename=config.logging.jsonl_filename,
            flush_every=config.logging.flush_every,
            console_level=config.logging.level,
            console=config.logging.console,
        )
        hw = config.hardware
        self.sim = WallClock(
            physics_dt=config.simulation.physics_dt,
            settle_steps=config.simulation.settle_steps,
            fast=hw.fake_clock,
        )
        self.client = JetsonClient(
            hw.jetson_host,
            hw.jetson_port,
            request_timeout_s=hw.request_timeout_s,
            trajectory_timeout_margin_s=hw.trajectory_timeout_margin_s,
        )
        self.detector = RemoteDetector(
            hw.detector_host,
            hw.detector_port,
            labels=hw.labels,
            min_score=hw.detection_min_score,
            timeout_s=hw.request_timeout_s,
        )
        self.kinematics = PlanarKinematics(hw.arm, config.robot.arm_joint_names)

        self.robot: RemoteArm | None = None
        self.exterior_camera: RemoteCamera | None = None
        self.vision: PlanarPerception | None = None
        self.planner: JointSpacePlanner | None = None
        self.controller: RemoteController | None = None
        self.grasp_scorer: GraspScorer | None = None
        self.grasp_generator: TopDownGraspGenerator | None = None
        self.memory: WorkingMemory | None = None
        self.skills: SkillRegistry | None = None
        self.executor: ClassicalExecutor | None = None
        self._built = False

    # ------------------------------------------------------------------

    @property
    def table_top_height(self) -> float:
        """Support plane, by the same rule as the sim runtime.

        ``scene.add_table`` is false on this lane, so this is
        ``perception.ground_plane_z``: the robot base frame's z = 0 *is* the
        table the arm is bolted to.
        """
        if not self.config.scene.add_table:
            return float(self.config.perception.ground_plane_z)
        return float(
            self.config.scene.table_position[2] + self.config.scene.table_scale[2] / 2.0
        )

    @property
    def cameras(self) -> dict[str, RemoteCamera]:
        """All cameras by name (one, fixed, overhead)."""
        if self.exterior_camera is None:
            return {}
        return {self.exterior_camera.name: self.exterior_camera}

    def build(self) -> None:
        """Run the bring-up sequence. Safe to call once."""
        if self._built:
            return
        cfg = self.config
        hw = cfg.hardware

        with self.events.timed("runtime.build"):
            try:
                self.client.connect()
            except ExecutionError as exc:
                raise ExecutionError(f"robot server: {exc}") from exc

            try:
                # Refuses (ConfigurationError) when hardware.arm limits exceed the
                # Jetson's pulse map (review HS-2 / GEO-1); opens the dedicated
                # estop channel with a <= 2 s ping and falls back if it cannot;
                # starts the heartbeat that keeps the Jetson's host-liveness
                # bound from relaxing an idle-but-alive arm, and reports every
                # detach this laptop did not ask for as an event.
                self.robot = RemoteArm(
                    self.client,
                    self.kinematics,
                    cfg.robot,
                    self.sim,
                    event_logger=self.events,
                    heartbeat_s=float(getattr(hw, "heartbeat_s", HEARTBEAT_PERIOD_S)),
                )
            except Exception:
                # Not swallowed: re-raised after the socket is released, so a
                # refused bring-up does not leave a REQ socket (and the process)
                # hanging on exit.
                self.client.close()
                raise
            if cfg.exterior_camera.enabled:
                self.exterior_camera = RemoteCamera(self.client, cfg.exterior_camera, self.sim)

            try:
                self.detector.connect()
            except PerceptionError as exc:
                self.robot.close()
                self.client.close()
                raise PerceptionError(f"detector service: {exc}") from exc

            self.vision = PlanarPerception(
                clock=self.sim,
                detector=self.detector,
                config=cfg.perception,
                hardware=hw,
                homography=cfg.exterior_camera.homography,
                support_height=self.table_top_height,
                workspace_min=cfg.scene.workspace_min,
                workspace_max=cfg.scene.workspace_max,
                camera=cfg.exterior_camera,
                event_logger=self.events,
            )

            self.robot.go_home_immediate()
            self.sim.settle()

            self.planner = JointSpacePlanner(
                robot=self.robot,
                kinematics=self.kinematics,
                config=cfg.motion,
                arm=hw.arm,
                support_height=self.table_top_height,
                event_logger=self.events,
            )
            self.controller = RemoteController(
                sim=self.sim,
                robot=self.robot,
                motion_config=cfg.motion,
                robot_config=cfg.robot,
                hardware_config=hw,
                workspace_min=np.array(cfg.scene.workspace_min),
                workspace_max=np.array(cfg.scene.workspace_max),
                event_logger=self.events,
            )
            self.grasp_scorer = GraspScorer(
                robot=self.robot,
                config=cfg.grasp,
                support_height=self.table_top_height,
                hand_setback=float(np.linalg.norm(cfg.robot.tcp_offset_from_hand)),
            )
            self.grasp_generator = TopDownGraspGenerator(
                config=cfg.grasp,
                hardware=hw,
                support_height=self.table_top_height,
                kinematics=self.kinematics,
            )
            self.memory = WorkingMemory(cfg.memory)
            self.skills = SkillRegistry(
                SkillContext(
                    sim=self.sim,
                    robot=self.robot,
                    vision=self.vision,
                    planner=self.planner,
                    controller=self.controller,
                    grasp_scorer=self.grasp_scorer,
                    memory=self.memory,
                    config=cfg,
                    events=self.events,
                    support_height=self.table_top_height,
                    grasp_generator=self.grasp_generator,
                ),
                skill_classes=HARDWARE_SKILLS,
            )
            self.executor = ClassicalExecutor(self.skills)

        self._built = True
        self.events.emit(
            "runtime.ready",
            {
                "backend": "hardware",
                "robot_server": self.client.endpoint,
                "robot_driver": (self.client.server_info or {}).get("driver"),
                "detector": self.detector.endpoint,
                "detector_backend": self.detector.backend,
                "fake_clock": self.sim.fast,
                "arm_dofs": self.kinematics.dof,
                "tcp_pose": self.robot.tcp_pose().to_log(),
                "skills": list(self.skills.names),
            },
        )
        _log.info(
            "HardwareRuntime ready (robot %s, detector %s/%s, clock %s)",
            self.client.endpoint, self.detector.endpoint, self.detector.backend,
            "virtual" if self.sim.fast else "wall",
        )

    def emergency_stop(self) -> bool:
        """Detach every servo now, on the arm's dedicated stop channel.

        For the Ctrl-C handler in ``scripts/run_assistant.py``: safe to call
        before :meth:`build` finished (returns ``False``, nothing to stop) and
        never raises -- a failed stop is logged, since the caller is about to
        exit anyway and the +6 V switch remains the real mid-motion stop.
        Returns ``True`` when the Jetson acknowledged the estop.
        """
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        try:
            robot.estop()
        except ExecutionError as exc:
            _log.error("emergency stop was not acknowledged: %s", exc)
            return False
        self.events.emit("runtime.emergency_stop", {"source": "operator"})
        return True

    def close(self) -> None:
        """Close the clients (the arm's stop channel too) and the event log. Idempotent."""
        try:
            robot = getattr(self, "robot", None)
            if robot is not None and hasattr(robot, "close"):
                robot.close()
            self.detector.close()
            self.client.close()
            self.sim.close()
        finally:
            self.events.close()

    def __enter__(self) -> "HardwareRuntime":
        self.build()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

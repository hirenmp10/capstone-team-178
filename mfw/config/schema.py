"""Typed, validated configuration.

Pure stdlib + NumPy. This module must never import Isaac Sim.

The "no hardcoding" requirement is enforced structurally, not by convention:
every tunable in the framework is a field here, loaded from YAML. Code that
needs a number reaches for config; if a number appears as a literal in a module
outside this package, that is a bug.

Loading is strict -- unknown keys raise rather than being silently ignored, so a
typo in a YAML file fails at startup instead of leaving a default quietly in
place. Validation runs at construction so an inconsistent config cannot reach
the simulator.
"""

from __future__ import annotations

import collections.abc as _abc
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

__all__ = [
    "SimulationConfig",
    "PhysicsConfig",
    "RobotConfig",
    "CameraConfig",
    "PerceptionConfig",
    "GraspConfig",
    "MotionConfig",
    "MemoryConfig",
    "Gr00tConfig",
    "LoggingConfig",
    "SceneObjectConfig",
    "SceneFurnitureConfig",
    "RandomizationConfig",
    "SceneConfig",
    "HardwareArmConfig",
    "HardwareConfig",
    "FrameworkConfig",
    "load_config",
    "ConfigError",
]


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class SimulationConfig:
    """Simulator bootstrap and stepping."""

    headless: bool = True
    physics_dt: float = 1.0 / 120.0
    """Physics substep. 120 Hz, not the default 60, because contact-rich grasping
    with a parallel gripper needs finer contact resolution to stay stable."""
    rendering_dt: float = 1.0 / 30.0
    stage_units_in_meters: float = 1.0
    renderer: str = "RayTracedLighting"
    settle_steps: int = 60
    """Physics steps to run after spawning before anything is considered at rest."""

    def validate(self) -> None:
        if self.physics_dt <= 0.0:
            raise ConfigError("simulation.physics_dt must be > 0")
        if self.rendering_dt < self.physics_dt:
            raise ConfigError(
                f"simulation.rendering_dt ({self.rendering_dt}) must be >= physics_dt "
                f"({self.physics_dt}); rendering faster than physics is meaningless"
            )
        if self.stage_units_in_meters <= 0.0:
            raise ConfigError("simulation.stage_units_in_meters must be > 0")


@dataclass(frozen=True)
class PhysicsConfig:
    """PhysX solver and material settings.

    These values are what make *pure physics* grasping viable. With PhysX
    defaults a parallel gripper on a rigid object tends to jitter and eject the
    object, which is the failure that tempts people into fake attachment. The
    fix is high friction, generous solver iteration counts, and contact offsets
    matched to the finger scale.
    """

    solver_position_iterations: int = 32
    solver_velocity_iterations: int = 4
    static_friction: float = 1.2
    dynamic_friction: float = 1.1
    restitution: float = 0.0
    """Zero: any bounce on finger contact ejects the object."""
    finger_static_friction: float = 1.6
    finger_dynamic_friction: float = 1.5
    contact_offset: float = 0.002
    rest_offset: float = 0.0005
    stabilization_threshold: float = 0.0
    """Zero disables PhysX stabilization, which otherwise damps the small
    contact velocities that a stable grasp depends on."""
    enable_ccd: bool = True
    sleep_threshold: float = 0.0
    """Zero prevents held objects from being put to sleep mid-carry."""
    gpu_dynamics: bool = False
    max_depenetration_velocity: float = 1.0

    def validate(self) -> None:
        if self.solver_position_iterations < 1:
            raise ConfigError("physics.solver_position_iterations must be >= 1")
        if self.rest_offset > self.contact_offset:
            raise ConfigError(
                f"physics.rest_offset ({self.rest_offset}) must be <= contact_offset "
                f"({self.contact_offset}); PhysX requires this ordering"
            )
        for name in ("static_friction", "dynamic_friction", "finger_static_friction"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"physics.{name} must be >= 0")


@dataclass(frozen=True)
class RobotConfig:
    """Robot asset, joint layout and the TCP definition.

    ``tcp_offset_from_hand`` is the load-bearing field. Isaac Sim's stock Franka
    exposes ``panda_rightfinger`` as its end-effector prim, which sits on one
    finger rather than between them. Every grasp pose in this framework is
    expressed at the true TCP, derived from ``tcp_parent_prim`` plus this offset.
    """

    usd_subpath: str = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    prim_path: str = "/World/Franka"
    name: str = "franka"
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    arm_joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    finger_joint_names: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")

    tcp_parent_prim: str = "panda_hand"
    tcp_offset_from_hand: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    """Translation from ``panda_hand`` to the fingertip midpoint, in the hand
    frame (+Z points out of the palm). 103.4 mm is the Franka hand-to-fingertip
    distance from Franka Robotics' published geometry."""

    left_finger_prim: str = "panda_leftfinger"
    right_finger_prim: str = "panda_rightfinger"

    home_joint_positions: tuple[float, ...] = (0.0, -1.16, 0.0, -2.3, 0.0, 1.6, 0.79)
    gripper_open_width: float = 0.08
    """Franka's maximum span, metres (each finger travels 0.04)."""
    gripper_closed_width: float = 0.0
    gripper_force: float = 40.0
    """Grasp force in newtons. Force control, not position control: commanding a
    closed *position* onto a rigid object fights the solver and ejects it."""
    gripper_speed: float = 0.05
    min_graspable_width: float = 0.008
    """Narrowest object the gripper will claim to be holding, metres.

    Not cosmetic: a close onto empty air settles a few millimetres open through
    finger compliance, so a smaller threshold reports a grasp that does not
    exist. Measured on this robot, an empty close rests at 3.7 mm."""
    joint_velocity_limit: float = 1.0
    joint_position_tolerance: float = 0.01

    lula_robot_description: str = ""
    """Filled from the Isaac install at load time if left empty."""
    lula_urdf: str = ""

    kinematics: str = "lula"
    """Which kinematics model the arm is driven through.

    ``lula`` is the Isaac/Lula solver the Franka sim lane uses and needs a
    6+ DoF arm. ``planar_4dof`` is the closed-form model for the hardware lane's
    hobby arm: base yaw plus three coplanar pitch joints, which is exactly four
    joints and no more. The DoF check below keys off this field because the
    "at least 6 DoF" rule that protects the sim lane would otherwise reject
    every real arm this project owns."""
    gripper_feedback: bool = True
    """Whether the gripper can report its actual opening.

    True for the Franka, whose finger joints are read back. False for PWM
    hobby servos, which have no position feedback at all: a "closed" command
    tells you nothing about whether anything is between the jaws. Skills use
    this to switch grasp verification from gripper-width evidence to
    perception-based evidence (the object is no longer where it was)."""

    KINEMATICS_MODELS = ("lula", "planar_4dof")

    def validate(self) -> None:
        if self.kinematics not in self.KINEMATICS_MODELS:
            raise ConfigError(
                f"robot.kinematics must be one of {list(self.KINEMATICS_MODELS)}, "
                f"got {self.kinematics!r}"
            )
        if self.kinematics == "planar_4dof":
            if len(self.arm_joint_names) != 4:
                raise ConfigError(
                    f"robot.arm_joint_names has {len(self.arm_joint_names)} entries; "
                    "the planar_4dof model is exactly 4 joints (base yaw + 3 pitch)"
                )
        elif len(self.arm_joint_names) < 6:
            raise ConfigError(
                f"robot.arm_joint_names has {len(self.arm_joint_names)} entries; "
                "a manipulator needs at least 6 DoF"
            )
        if self.gripper_open_width <= self.gripper_closed_width:
            raise ConfigError("robot.gripper_open_width must exceed gripper_closed_width")
        if len(self.home_joint_positions) != len(self.arm_joint_names):
            raise ConfigError(
                f"robot.home_joint_positions has {len(self.home_joint_positions)} values "
                f"but there are {len(self.arm_joint_names)} arm joints"
            )
        if self.gripper_force <= 0.0:
            raise ConfigError("robot.gripper_force must be > 0")


@dataclass(frozen=True)
class CameraConfig:
    """One camera. Focal length and aperture are Isaac's native parameterisation;
    pixel intrinsics are derived from them at runtime."""

    name: str = "camera"
    prim_path: str = "/World/Camera"
    parent_prim: str = ""
    """If set, the camera is attached to this prim and moves with it (wrist camera)."""
    resolution: tuple[int, int] = (640, 480)
    focal_length: float = 1.93
    horizontal_aperture: float = 3.896
    vertical_aperture: float = 2.453
    clipping_range: tuple[float, float] = (0.01, 5.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    look_at: tuple[float, float, float] | None = None
    """If set, orientation is derived to aim the camera at this world point and
    ``quat`` is ignored. Preferred for static cameras: a hand-written quaternion
    that is slightly wrong yields an image of empty floor, which is easy to
    misread downstream as "the detector found nothing"."""
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    enable_depth: bool = True
    enable_segmentation: bool = True

    enabled: bool = True
    """False leaves this camera out of the runtime entirely. The hardware lane
    has no wrist camera; the sim lane always has both."""
    source: str = ""
    """Device of a *real* camera (``/dev/video0``, an index, an RTSP URL).
    Ignored in sim, where ``prim_path`` is the source. Informational on the
    laptop -- the frame grabber runs on whichever host owns the USB port."""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    """Measured pixel intrinsics. Zero means "derive from focal_length and the
    apertures", which is right for Isaac cameras (their lens *is* those
    parameters) and wrong for a webcam, whose mm apertures nobody knows. Real
    cameras set ``fx``; ``fy`` falls back to ``fx`` and ``cx``/``cy`` to the
    image centre when left at zero."""
    distortion: tuple[float, ...] = ()
    """OpenCV distortion coefficients ``(k1, k2, p1, p2[, k3[, k4, k5, k6]])``.
    Empty means an ideal pinhole, which is what every sim camera is."""
    homography: tuple[float, ...] = ()
    """Row-major 3x3 pixel -> table-plane (x, y) map for an overhead camera
    with no depth. Empty means "not calibrated"; the hardware runtime refuses
    to build without one rather than guessing, because an uncalibrated
    homography places every object at a plausible-looking wrong spot."""
    pose_measured: bool = False
    """Whether ``position``, ``look_at``, ``up`` and ``fx``/``fy``/``cx``/``cy``
    were *measured* for this camera (checkerboard intrinsics plus a taped or
    PnP-fitted pose), as opposed to being the placeholders a config ships
    with. The depthless estimator only runs its pinhole anchor refinement and
    the lift-prediction evidence when this is true: measured against an exact
    homography, a placeholder pose 10 cm off turned into a 100 mm object
    error in every estimate, worse than doing nothing. False keeps the
    first-order half-footprint step only, which needs just the image-down
    direction on the table to be right (checkable by eye from one frame)."""

    DISTORTION_LENGTHS = (0, 4, 5, 8)

    def resolved_quat(self) -> tuple[float, float, float, float]:
        """Orientation to apply, honouring ``look_at`` when present."""
        if self.look_at is None:
            return self.quat
        from mfw.utils.transforms import look_at_quat  # local: avoid an import cycle

        return tuple(float(v) for v in look_at_quat(self.position, self.look_at, self.up))  # type: ignore[return-value]

    def pixel_intrinsics(self) -> tuple[float, float, float, float]:
        """Pinhole intrinsics ``(fx, fy, cx, cy)`` in pixels.

        Measured values win when ``fx`` is set. Otherwise this is the exact
        derivation :meth:`mfw.vision.camera.Camera.get_intrinsics` uses: fx from
        the horizontal aperture and fy from the *vertical* aperture, separately.
        The default apertures do not match the 4:3 resolution, so fx != fy --
        that is not a bug. Forcing fy = fx was measured to add up to 43 mm of
        vertical error to reconstructions that had been within 5 mm.
        """
        width, height = (int(v) for v in self.resolution)
        if self.fx > 0.0:
            fy = self.fy if self.fy > 0.0 else self.fx
            cx = self.cx if self.cx > 0.0 else width / 2.0
            cy = self.cy if self.cy > 0.0 else height / 2.0
            return (float(self.fx), float(fy), float(cx), float(cy))
        fx = self.focal_length * width / self.horizontal_aperture
        fy = self.focal_length * height / self.vertical_aperture
        return (float(fx), float(fy), width / 2.0, height / 2.0)

    def validate(self) -> None:
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ConfigError(f"camera[{self.name}].resolution must be positive, got {self.resolution}")
        near, far = self.clipping_range
        if near <= 0.0 or far <= near:
            raise ConfigError(
                f"camera[{self.name}].clipping_range must satisfy 0 < near < far, got {self.clipping_range}"
            )
        if self.focal_length <= 0.0 or self.horizontal_aperture <= 0.0:
            raise ConfigError(f"camera[{self.name}] focal_length and horizontal_aperture must be > 0")
        if self.vertical_aperture <= 0.0:
            raise ConfigError(f"camera[{self.name}].vertical_aperture must be > 0")
        for name in ("fx", "fy", "cx", "cy"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"camera[{self.name}].{name} must be >= 0 (0 = derive)")
        if len(self.distortion) not in self.DISTORTION_LENGTHS:
            raise ConfigError(
                f"camera[{self.name}].distortion must have {list(self.DISTORTION_LENGTHS)} "
                f"coefficients (OpenCV layout), got {len(self.distortion)}"
            )
        if len(self.homography) not in (0, 9):
            raise ConfigError(
                f"camera[{self.name}].homography must be empty or a row-major 3x3 "
                f"(9 numbers), got {len(self.homography)}"
            )
        if self.look_at is not None:
            import math

            if math.dist(self.position, self.look_at) < 1e-6:
                raise ConfigError(
                    f"camera[{self.name}].look_at coincides with its position; "
                    "the view direction would be undefined"
                )


@dataclass(frozen=True)
class PerceptionConfig:
    """Perception thresholds. All of these are quality gates on evidence."""

    min_points_per_object: int = 60
    """Below this, a segment is noise rather than an object."""
    min_confidence: float = 0.25
    depth_min: float = 0.05
    depth_max: float = 3.0
    voxel_size: float = 0.004
    """Downsampling leaf size for point clouds, metres."""
    outlier_std_ratio: float = 2.0
    bbox_trim_percentile: float = 2.0
    """Percentile trim when sizing an object's bounding box. Instance masks
    include a halo of edge pixels whose depth interpolates onto the background,
    which with raw min/max sizing inflated a 0.12 m box to 0.22 m on this scene.
    Grasp width comes straight from these extents."""
    max_points_for_outlier_filter: int = 4000
    """Cap on points entering the O(n^2) neighbour filter. Above this the
    distance matrix costs more memory than the filtering is worth."""
    track_match_distance: float = 0.06
    """Max centroid movement between frames still considered the same object."""
    track_max_age_steps: int = 600
    """Steps a track survives unobserved before being dropped (objects may disappear)."""
    ground_plane_z: float = 0.0
    ground_clearance: float = 0.005
    """Points below this height above the support plane are treated as the plane."""
    max_scene_graph_age_s: float = 1.0

    support_constrained_height: bool = False
    """Derive an object's height from the support surface instead of the points.

    A single view sees an object's top face and little of its sides, so it
    under-measures height -- the YCB mustard bottle's 191 mm read as 156 mm.
    Since the object rests on a surface of known height, the bottom is not a
    guess and the height follows from the observed top.

    Off by default because a *more accurate* box is not automatically a *more
    useful* one. Grasp candidates are generated from these extents and filtered
    on them, so changing a height changes which grasps exist and where their
    pre-grasp standoffs sit. Measured on the benchmark scene, enabling this took
    the pick rate from 3/8 to 2/8 and regressed the mug from a clean +118 mm
    lift to "no path to the pregrasp standoff".

    Turn it on when the consumer wants true geometry (measurement, dataset
    export, a learned policy) rather than when it wants grasps."""

    render_frames_before_capture: int = 8
    """Render frames to force before reading the cameras.

    Not optional in headless mode. ``SimulationContext.step`` renders only when
    the app is windowed, so after a physics-only settle the render products hold
    a stale, near-black RGB buffer. Depth and segmentation survive -- they come
    from geometry and prim identity, not from accumulated samples -- so the only
    symptom is that every object's colour reads "black" and colour-qualified
    references stop resolving.

    Eight because that is what was measured, sweeping frame count against the
    colour actually recovered for the YCB mug (a dark red object, and so the
    hardest case in the scene):

        frames:  0     1     2      3       5       8       12
        median: [1,1,1] [1,1,1] [9,2,2] [10,3,2] [13,3,3] [16,4,3] [19,5,4]
        named:  black black black  black   black   red      red

    Note the whole-image mean was ~145/255 at *every* count -- the scene is well
    lit throughout. It is specifically low-albedo surfaces that accumulate
    slowly, so an exposure check on the frame average cannot detect this. Only
    the dark objects are wrong, and they are wrong in a way that reads as "the
    mug is black" rather than as a rendering problem.

    The cost is real: eight render steps per ``observe()``. Lower it only with a
    measurement, not a guess."""
    """A skill refuses to act on a scene graph older than this."""

    def validate(self) -> None:
        if self.depth_max <= self.depth_min:
            raise ConfigError("perception.depth_max must exceed depth_min")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("perception.min_confidence must be in [0, 1]")
        if self.voxel_size <= 0.0:
            raise ConfigError("perception.voxel_size must be > 0")
        if not 0.0 <= self.bbox_trim_percentile < 50.0:
            raise ConfigError(
                f"perception.bbox_trim_percentile must be in [0, 50), got {self.bbox_trim_percentile}"
            )


@dataclass(frozen=True)
class GraspConfig:
    """Grasp synthesis. No pose here is a grasp; these are the rules that generate them."""

    num_orientation_samples: int = 16
    approach_offset: float = 0.10
    """Pregrasp standoff along -approach axis, metres."""
    finger_width_margin: float = 0.012
    """Required clearance between object width and gripper span."""
    max_grasp_width: float = 0.075
    min_grasp_width: float = 0.005
    top_grasp_bias: float = 0.6
    """Weight favouring top-down approaches, which are the most reachable on a table."""
    collision_check_samples: int = 12
    max_candidates: int = 64
    min_score: float = 0.15
    lift_height: float = 0.12
    """How far to lift after closing, metres."""
    place_clearance: float = 0.02
    """Gap above the destination surface at release, metres."""
    recentre_from_standoff: bool = True
    """Re-observe from the pre-grasp standoff and correct the grasp before the
    final descent. Worth it with a wrist camera looking at the object; pointless
    with a single fixed overhead camera, where the standoff view is the same
    view that produced the estimate."""
    verify_min_displacement: float = 0.03
    """Metres a non-rising object must have slid from its pre-grasp table pose
    for a feedback-less verdict to call it 'knocked' rather than 'resting'.
    It is not evidence of a carry: a lifted object's homography position
    shifts only 7-29 mm by parallax (review F1), so carries are judged from
    pixel growth and the lift prediction in mfw.physics.contact. Perception
    noise after homography is around a centimetre; three keeps a stationary
    object from being called knocked."""

    def validate(self) -> None:
        if self.max_grasp_width <= self.min_grasp_width:
            raise ConfigError("grasp.max_grasp_width must exceed min_grasp_width")
        if self.approach_offset <= 0.0:
            raise ConfigError("grasp.approach_offset must be > 0")
        if self.num_orientation_samples < 1:
            raise ConfigError("grasp.num_orientation_samples must be >= 1")
        if self.verify_min_displacement <= 0.0:
            raise ConfigError("grasp.verify_min_displacement must be > 0")


@dataclass(frozen=True)
class MotionConfig:
    """Online motion planning. No prerecorded trajectories anywhere."""

    planner: str = "rrt"
    """``rrt`` for global collision-aware planning, ``rmpflow`` for reactive
    control, ``joint_space`` for the hardware lane's straight-line joint
    interpolation through a transit height (no collision world exists there)."""

    PLANNERS = ("rrt", "rmpflow", "joint_space")
    max_planning_time_s: float = 5.0
    interpolation_dt: float = 0.02
    max_joint_velocity: float = 1.0
    max_joint_acceleration: float = 2.0
    collision_margin: float = 0.015
    obstacle_inflation: float = 0.01
    max_replan_attempts: int = 3
    cartesian_step: float = 0.005
    """Straight-line interpolation resolution for approach and lift, metres."""
    tracking_error_limit: float = 0.15
    """Joint tracking error, radians, above which the arm is considered to be
    falling behind its commanded configuration."""
    tracking_violation_steps: int = 12
    """Consecutive over-limit steps required before aborting. A single sample over
    the limit is transient, not a fault; a deflected arm stays over it."""
    stall_abort_steps: int = 400
    """Control steps with no advance along the path before declaring the arm stuck.
    Deviation alone cannot detect this: a blocked arm sits *on* the path and simply
    stops progressing along it."""
    plan_attempts: int = 4
    """How many RRT queries to run before choosing a path.

    RRT is stochastic and returns the *first* feasible path it finds, not a good
    one. Raw tree paths meander: measured on this scene, a plan to an object
    0.33 m in front of the robot swung the TCP to x = -0.202, behind the arm's
    own base. It was a valid, collision-free path -- just a terrible one -- and
    it aborted the pick on the workspace check.

    The textbook remedy is a collision-checked shortcut pass, but Lula's RRT
    wrapper exposes no state-validity query, so candidate straight-line segments
    cannot be verified. Re-planning with different random seeds and keeping the
    shortest path needs only the public API and attacks the same problem: the
    meandering is a sampling artefact, so another sample is usually far better.

    Costs up to this many planning calls per motion. 1 restores the old
    behaviour."""

    workspace_transit_margin: float = 0.25
    """Slack allowed on the workspace box for *intermediate* trajectory
    waypoints, metres.

    The workspace envelope constrains where the arm should come to rest, not
    every configuration it passes through. RRT plans in joint space, so the path
    between two in-box poses routinely arcs the TCP outside the box -- that is
    ordinary motion, not a safety event.

    Checking every waypoint strictly aborted valid picks and did so
    *nondeterministically*, because RRT is randomised: the same object failed on
    one run and succeeded on the next, and the reported violation appeared at
    y = -0.467 in one run and y = +0.497 in the next for a stationary target.
    The x coordinate always sat at the boundary itself, which is the signature
    of a check firing the instant a path crosses it rather than of a bad goal.

    Goals are still checked with zero margin. This only widens the envelope for
    points the arm is passing through.
    """

    ik_max_iterations: int = 150
    ik_position_tolerance: float = 0.002
    ik_orientation_tolerance: float = 0.02

    def validate(self) -> None:
        if self.planner not in self.PLANNERS:
            raise ConfigError(
                f"motion.planner must be one of {list(self.PLANNERS)}, got {self.planner!r}"
            )
        if self.max_planning_time_s <= 0.0:
            raise ConfigError("motion.max_planning_time_s must be > 0")
        if self.cartesian_step <= 0.0:
            raise ConfigError("motion.cartesian_step must be > 0")


@dataclass(frozen=True)
class MemoryConfig:
    """Working memory and reference resolution."""

    max_scene_history: int = 200
    max_command_history: int = 100
    pronoun_words: tuple[str, ...] = ("it", "that", "this", "them", "the object")
    reference_recency_weight: float = 0.7
    persist_path: str = "logs/memory_state.json"

    def validate(self) -> None:
        if self.max_scene_history < 1:
            raise ConfigError("memory.max_scene_history must be >= 1")


@dataclass(frozen=True)
class Gr00tConfig:
    """GR00T N1.7 policy backend.

    Always out-of-process: Isaac Sim 5.1 ships Python 3.11 and GR00T requires
    3.12, so they cannot share an interpreter regardless of host. ``host``/``port``
    is the whole portability story -- pointing at WSL instead of Windows is a
    config change, not a code change.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5555
    embodiment_tag: str = "oxe_droid_relative_eef_relative_joint"
    """The only manipulation tag that is inference-ready in the base N1.7
    checkpoint. LIBERO_PANDA and ROBOCASA_PANDA_OMRON need finetuned weights."""
    model_path: str = "nvidia/GR00T-N1.7-3B"
    action_horizon: int = 40
    """Actions per returned chunk, fixed by the embodiment config."""
    actions_executed_per_chunk: int = 8
    """Execute a prefix, then re-observe. Open-loop execution of all 40 drifts."""
    observation_history: int = 16
    """Frames buffered to serve the tag's ``delta_indices=[-15, 0]`` video requirement."""
    exterior_camera_key: str = "exterior_image_1_left"
    wrist_camera_key: str = "wrist_image_left"
    image_size: tuple[int, int] = (224, 224)
    request_timeout_s: float = 10.0
    max_relative_translation: float = 0.05
    """Safety clamp on a single predicted EEF delta, metres."""
    max_relative_rotation: float = 0.35
    """Safety clamp on a single predicted EEF delta, radians."""
    actions_are_absolute: bool = True
    """Whether the policy returns world-frame poses rather than deltas.

    True for GR00T N1.7. The embodiment tag reads
    ``oxe_droid_relative_eef_relative_joint``, but "relative" describes the
    *training* representation -- NVIDIA's processor denormalises and converts
    relative to absolute during postprocessing, so ``get_action`` returns absolute
    poses. Measured: with the TCP at [0.45, 0, 0.45], action[0] was
    [0.457, -0.006, 0.443], an 11 mm move. Read as a delta it became 0.61 m and
    was clamped every time."""
    use_mock_server: bool = True
    """Run against a deterministic mock until a real checkpoint is available."""

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"gr00t.port out of range: {self.port}")
        if self.actions_executed_per_chunk > self.action_horizon:
            raise ConfigError(
                f"gr00t.actions_executed_per_chunk ({self.actions_executed_per_chunk}) "
                f"exceeds action_horizon ({self.action_horizon})"
            )
        if self.observation_history < 1:
            raise ConfigError("gr00t.observation_history must be >= 1")
        if self.max_relative_translation <= 0.0:
            raise ConfigError("gr00t.max_relative_translation must be > 0")


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging. Everything is logged as JSONL for replay and analysis."""

    level: str = "INFO"
    log_dir: str = "logs"
    jsonl_filename: str = "events.jsonl"
    console: bool = True
    log_images: bool = False
    image_log_every_n: int = 30
    flush_every: int = 1

    def validate(self) -> None:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid:
            raise ConfigError(f"logging.level must be one of {sorted(valid)}, got {self.level!r}")


@dataclass(frozen=True)
class SceneObjectConfig:
    """A manipulable object to spawn.

    Note what this is *not*: the perception stack never reads these values. They
    exist to build the scene and to let tests assert against ground truth. Any
    runtime code that consults this to locate an object is violating the
    perception-first rule.
    """

    name: str
    kind: str = "cuboid"
    """``cuboid``, ``cylinder``, ``sphere``, ``usd``, or ``asset``.

    ``asset`` is the one to reach for: it names an entry in
    ``configs/assets.yaml`` and inherits that entry's mesh, true mass, real
    dimensions, collider strategy and semantic label. ``usd`` remains for a
    one-off path that does not warrant a catalogue entry; the primitive kinds
    remain for tests, which need a spawn path that does not touch the asset
    server."""
    asset: str = ""
    """Registry key, when ``kind == "asset"``."""
    usd_subpath: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] | None = None
    """Explicit scale, or ``None`` to use the right default for this ``kind``.

    There is no single sensible default, which is why this is ``None`` rather
    than a number. A primitive needs a size, because ``DynamicCuboid`` has no
    intrinsic one -- 50 mm is a reasonable object. A referenced mesh is the
    opposite: YCB assets are authored at true real-world scale, so the only
    correct default is 1.

    Sharing one default across both is not a style question. Applying the
    primitive's 0.05 to a mesh shrinks every asset to 5% of its true size: a
    66 mm soup can becomes 3.3 mm. It still spawns, still has a collider, still
    has the correct mass, and still rests on the table -- so every physics check
    passes -- while occupying two pixels of camera image, far under the
    detector's minimum. The scene is physically valid and completely invisible.
    """
    color: tuple[float, float, float] = (0.8, 0.2, 0.2)
    mass: float = 0.2
    """Ignored for ``kind == "asset"``: the registry's ``mass_kg`` wins. Two
    sources of truth for an object's mass is one too many, and the registry's
    value is the measured YCB figure."""
    semantic_label: str = ""
    collision_approximation: str = ""
    """Per-placement override of the collider strategy. Empty means derive it
    from the asset's category."""

    DEFAULT_PRIMITIVE_SCALE = (0.05, 0.05, 0.05)
    DEFAULT_MESH_SCALE = (1.0, 1.0, 1.0)

    def resolved_scale(self) -> tuple[float, float, float]:
        """The scale to spawn at, defaulted per ``kind``.

        Always use this rather than reading ``scale`` directly -- reading the
        raw field is how a mesh ends up wearing a primitive's default.
        """
        if self.scale is not None:
            return self.scale
        if self.kind in ("usd", "asset"):
            return self.DEFAULT_MESH_SCALE
        return self.DEFAULT_PRIMITIVE_SCALE

    def validate(self) -> None:
        if self.kind not in ("cuboid", "cylinder", "sphere", "usd", "asset"):
            raise ConfigError(f"scene object {self.name!r}: unknown kind {self.kind!r}")
        if self.scale is not None and any(s <= 0.0 for s in self.scale):
            raise ConfigError(f"scene object {self.name!r}: scale must be positive on every axis")
        if self.kind == "usd" and not self.usd_subpath:
            raise ConfigError(f"scene object {self.name!r}: kind 'usd' requires usd_subpath")
        if self.kind == "asset" and not self.asset:
            raise ConfigError(
                f"scene object {self.name!r}: kind 'asset' requires an 'asset' key naming "
                f"an entry in the asset catalogue"
            )
        if self.mass <= 0.0:
            raise ConfigError(f"scene object {self.name!r}: mass must be > 0")


@dataclass(frozen=True)
class SceneFurnitureConfig:
    """A static fixture placed from the asset catalogue.

    Static, so no mass and no rigid body: furniture defines the room, it is not
    a manipulation target. It still gets a collider (so the planner treats it as
    an obstacle) and a semantic label (so perception can name it when asked
    what is in the scene).
    """

    name: str
    asset: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision: bool = True
    """Off for distant decor. A plant across the room costs planner collision
    checks on every query while being unreachable by construction."""

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("scene furniture entry has an empty name")
        if not self.asset:
            raise ConfigError(f"scene furniture {self.name!r}: needs an 'asset' key")


@dataclass(frozen=True)
class RandomizationConfig:
    """Domain randomisation.

    Off by default. Randomisation is for *training* and for measuring
    robustness; a debugging run needs the scene to be identical every time, and
    a framework that silently jitters the world makes every regression
    unreproducible.

    ``seed`` is explicit for the same reason: "randomised" must still mean
    "replayable", or a failure found in a randomised run cannot be investigated.
    """

    enabled: bool = False
    seed: int = 0
    randomize_positions: bool = True
    randomize_yaw: bool = True
    randomize_lighting: bool = False
    randomize_object_set: bool = False
    """Draw a random subset of catalogue objects instead of placing the listed
    ones. Turns a fixed scene into a generator of scenes."""
    num_objects: tuple[int, int] = (3, 6)
    """Inclusive range, used only when ``randomize_object_set`` is on."""
    position_jitter_m: float = 0.08
    yaw_range_rad: tuple[float, float] = (-3.14159, 3.14159)
    lighting_intensity_scale: tuple[float, float] = (0.6, 1.4)
    min_object_separation_m: float = 0.04
    """Gap between object footprints. Zero would let two meshes spawn
    interpenetrating, and PhysX resolves that by launching them apart on the
    first step -- which looks like an explosion, not a scene."""
    max_placement_attempts: int = 200

    def validate(self) -> None:
        lo, hi = self.num_objects
        if lo < 0 or hi < lo:
            raise ConfigError(
                f"randomization.num_objects must be a non-negative (min, max) range, "
                f"got {self.num_objects}"
            )
        if self.position_jitter_m < 0.0:
            raise ConfigError("randomization.position_jitter_m must be >= 0")
        if self.min_object_separation_m < 0.0:
            raise ConfigError("randomization.min_object_separation_m must be >= 0")
        if self.max_placement_attempts < 1:
            raise ConfigError("randomization.max_placement_attempts must be >= 1")
        scale_lo, scale_hi = self.lighting_intensity_scale
        if scale_lo <= 0.0 or scale_hi < scale_lo:
            raise ConfigError(
                f"randomization.lighting_intensity_scale must be a positive increasing "
                f"range, got {self.lighting_intensity_scale}"
            )


@dataclass(frozen=True)
class SceneConfig:
    """Workspace layout."""

    assets_path: str = ""
    """Asset catalogue to load. Empty uses ``configs/assets.yaml``."""
    environment: str = ""
    """Catalogue key of a room USD to load beneath the workspace (``office``,
    ``simple_room``). Empty builds on the bare ground plane, which is what the
    test scenes use -- loading a full room costs seconds per test."""
    environment_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    furniture: tuple[SceneFurnitureConfig, ...] = ()
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    add_ground_plane: bool = True
    table_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    table_scale: tuple[float, float, float] = (0.8, 1.2, 0.4)
    add_table: bool = True

    dome_light_intensity: float = 1000.0
    """Explicit scene lighting. Isaac's implicit default light falls off sharply
    with distance: measured here, a wrist camera 30 cm from the table renders
    correctly while an exterior camera 1.5 m away renders black. Deterministic
    lighting is a perception requirement, not a cosmetic one -- set to 0 to
    disable and inherit whatever the stage provides."""
    distant_light_intensity: float = 1500.0
    """Directional key light, giving the shading gradients that shape-from-
    shading and segmentation boundaries rely on."""
    distant_light_angle: tuple[float, float, float] = (315.0, 0.0, 0.0)

    workspace_light_intensity: float = 0.0
    """A local sphere light above the table. Off by default; **required indoors.**

    A dome light is a sky and a distant light is a sun, and neither reaches
    through a ceiling. Load a room USD and both stop working: measured in the
    office scene, raising the dome from 400 to 9000 -- a factor of 22 -- changed
    mean image luminance by 15%, and switching both lights off entirely made the
    scene *brighter* than leaving them on. They were contributing nothing but
    render cost.

    The failure is quiet and expensive. Physics, semantics and geometry are all
    unaffected, every scene check passes, and detection still works because
    segmentation is not photometric. What breaks is everything that reads
    pixel *values*: colour naming degrades to "black", so "the green box" stops
    resolving, and a VLA fed near-black frames is far outside its training
    distribution.

    Do not assume a room USD lights itself. The Isaac office ships 121 light
    prims with 5 of them enabled."""
    workspace_light_height: float = 1.2
    """Metres above the table top."""
    workspace_light_radius: float = 0.25
    """A broad emitter, not a point. A small source throws hard shadows that
    segmentation edges then have to fight."""
    objects: tuple[SceneObjectConfig, ...] = ()
    workspace_min: tuple[float, float, float] = (0.2, -0.5, 0.0)
    workspace_max: tuple[float, float, float] = (0.8, 0.5, 0.7)

    robot_reach_m: float = 0.80
    """Radius the arm can actually reach, measured from ``robot_base``.

    The workspace min/max pair is a **box**, and an arm's envelope is a
    **sphere**. That difference is not academic: the box
    [0.12,-0.55,0.0]..[0.85,0.55,0.85] has a corner 1.27 m from the base, half a
    metre beyond anything a Franka can touch. Every object in the first
    benchmark layout passed the box test, and five of nine then failed with "no
    path to the pregrasp standoff" because they were simply out of range --
    a placement error that surfaced as a planner error.

    0.80 rather than the Franka's nominal 0.855 because reach at the limit is a
    singular, near-fully-extended posture with no orientation freedom, which is
    useless for grasping. Objects are also checked at their *pre-grasp standoff*
    rather than at their own position, since the arm has to get above them
    first."""
    robot_min_reach_m: float = 0.35
    """Inner reach limit. An arm cannot grasp objects pressed against its own
    base: it has to fold past its joint limits and the elbow fouls the torso.
    Placement without this leaves a dead zone directly in front of the robot
    that looks perfectly reachable on any distance-from-base measure."""
    robot_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Where the arm is mounted, for the reach test. Mirrors
    ``robot.base_position``; kept here so scene-layout code never has to reach
    into the robot config."""

    def validate(self) -> None:
        for obj in self.objects:
            obj.validate()
        for item in self.furniture:
            item.validate()
        self.randomization.validate()

        # Objects and furniture share the /World namespace, so a name collision
        # would have one prim silently overwrite the other.
        names = [o.name for o in self.objects] + [f.name for f in self.furniture]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ConfigError(
                f"scene has duplicate object/furniture names: {sorted(dupes)}"
            )
        if any(a >= b for a, b in zip(self.workspace_min, self.workspace_max)):
            raise ConfigError(
                f"scene.workspace_min {self.workspace_min} must be strictly less than "
                f"workspace_max {self.workspace_max} on every axis"
            )


@dataclass(frozen=True)
class HardwareArmConfig:
    """Geometry of the hardware lane's planar hobby arm.

    Base yaw, then shoulder / elbow / wrist pitch in one vertical plane, then a
    fixed tool. Every length here is *measured on the assembled arm* -- the
    defaults are placeholders sized to a ~28 cm-reach PLA kit so the fake lane
    has something to run against, and nothing in them should be trusted on real
    metal. The kinematics module owns the conventions (which way positive pitch
    goes, where zero is); this dataclass only carries the numbers.
    """

    base_height: float = 0.07
    """Table plane to the shoulder pitch axis, metres, along the yaw axis."""
    shoulder_offset: float = 0.0
    """Horizontal offset of the shoulder axis from the yaw axis, metres. Zero
    for a kit whose shoulder servo sits directly on the turntable."""
    upper_arm: float = 0.105
    """Shoulder axis to elbow axis, metres."""
    forearm: float = 0.10
    """Elbow axis to wrist axis, metres."""
    tool: float = 0.09
    """Wrist axis to the jaw midpoint (the TCP), metres. Mirrors
    ``robot.tcp_offset_from_hand[2]`` on the hardware config."""
    joint_lower: tuple[float, float, float, float] = (-1.5708, -0.5236, -1.5708, -1.5708)
    joint_upper: tuple[float, float, float, float] = (1.5708, 1.5708, 1.5708, 1.5708)
    """Radians, in joint order ``(base_yaw, shoulder, elbow, wrist)``. These
    are *kinematic* limits for the planner; the per-servo pulse end-stops live
    in the Jetson's calibration file and are clamped again there."""
    elbow_up: bool = True
    """Preferred IK branch. The fallback branch is tried when this one is out
    of limits, so this is a preference rather than a constraint."""
    jaw_axis: str = "tangential"
    """Which way the fixed (no wrist roll) jaws close, relative to the ray from
    the base to the object: ``tangential`` closes across it, ``radial`` closes
    along it. Set by how the gripper is bolted to the wrist."""
    transit_height: float = 0.15
    """Floor of the TCP transit height, metres above the table. The joint-space
    planner raises it per plan to clear the tallest perceived obstacle plus
    inflation, held-object hang-down and tilted-finger dip (review GEO-4); 0.10
    sat exactly at the 0.10 m bin's height, so no crossing of it could plan."""

    JAW_AXES = ("tangential", "radial")

    def validate(self) -> None:
        for name in ("base_height", "upper_arm", "forearm", "tool"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"hardware.arm.{name} must be > 0")
        if self.shoulder_offset < 0.0:
            raise ConfigError("hardware.arm.shoulder_offset must be >= 0")
        if len(self.joint_lower) != 4 or len(self.joint_upper) != 4:
            raise ConfigError(
                "hardware.arm.joint_lower/joint_upper must each have 4 entries "
                f"(yaw, shoulder, elbow, wrist), got {len(self.joint_lower)}/{len(self.joint_upper)}"
            )
        for i, (lo, hi) in enumerate(zip(self.joint_lower, self.joint_upper)):
            if lo >= hi:
                raise ConfigError(
                    f"hardware.arm joint {i}: joint_lower ({lo}) must be < joint_upper ({hi})"
                )
        if self.jaw_axis not in self.JAW_AXES:
            raise ConfigError(
                f"hardware.arm.jaw_axis must be one of {list(self.JAW_AXES)}, got {self.jaw_axis!r}"
            )
        if self.transit_height <= 0.0:
            raise ConfigError("hardware.arm.transit_height must be > 0")


@dataclass(frozen=True)
class HardwareConfig:
    """The hardware lane: a Jetson-hosted arm bridge and detector, driven from mfw.

    Only read when ``backend == "hardware"``. Every host/port pair here is the
    whole migration story -- a service moves between the laptop and the Jetson
    by editing an address, never code. Sizes and thresholds exist because the
    overhead camera has no depth: object height and extents cannot be measured,
    so they are looked up per label.
    """

    jetson_host: str = "127.0.0.1"
    jetson_port: int = 5560
    """``jetson/robot_server.py`` (ZMQ REP, msgpack)."""
    detector_host: str = "127.0.0.1"
    detector_port: int = 5558
    """Detector service (TCP newline-JSON). Same protocol whether NanoOWL on the
    Jetson or Florence-2 on the laptop is answering."""
    arm_bridge: str = "uno_serial"
    """How the Jetson reaches the servos: ``uno_serial`` (Arduino USB-CDC,
    default), ``pca9685`` (I2C, for a CH340 Uno clone JetPack 6 cannot see),
    or ``fake`` (in-process driver for tests and the day-0 e2e run)."""
    request_timeout_s: float = 5.0
    """Round-trip budget for a non-motion RPC."""
    trajectory_timeout_margin_s: float = 5.0
    """Added to a chunk's own duration to form its RPC timeout, so a slow
    Uno frame or Wi-Fi jitter does not abort a motion that is still running."""
    trajectory_chunk_s: float = 1.0
    """Seconds of trajectory shipped per ``follow_trajectory`` call. Smaller
    means more chances for ``on_step`` (and an abort) between chunks; larger
    means fewer round trips over a jittery link."""
    settle_steps_after_motion: int = 24
    """Clock steps to wait after a motion before trusting perception. PLA links
    ring for a moment after a stop; a frame grabbed mid-ring smears the bbox."""
    fake_clock: bool = False
    """Advance ``WallClock`` virtually instead of sleeping. Tests only."""
    arm: HardwareArmConfig = field(default_factory=HardwareArmConfig)
    object_sizes: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    """Per-label ``(x, y, z)`` extents, metres, standing in for the depth the
    camera does not have. z sets the grasp height; the larger of x/y (after
    the jaw-axis choice) sets the grasp width."""
    default_object_size: tuple[float, float, float] = (0.04, 0.04, 0.04)
    """Used for a detected label with no ``object_sizes`` entry."""
    labels: tuple[str, ...] = ()
    """Vocabulary sent to the detector on every observe. Empty means "whatever
    the detector defaults to", which is not what you want in a demo."""
    detection_min_score: float = 0.1
    pixel_anchor: str = "bottom_center"
    """Which bbox point the homography is applied to. ``bottom_center`` is
    where the object meets the table for a camera that is not perfectly
    overhead; ``center`` for a truly nadir view."""
    workspace_margin_xy: float = 0.03
    """Detections this far outside ``scene.workspace_min/max`` in XY are
    dropped as background rather than reported as unreachable objects."""
    grasp_pitch_angles_deg: tuple[float, ...] = (90.0, 75.0, 60.0)
    """Approach pitches to try, most vertical first. 90 is straight down; the
    shallower ones buy reach at the workspace edge."""

    ARM_BRIDGES = ("uno_serial", "pca9685", "fake")
    PIXEL_ANCHORS = ("bottom_center", "center")

    def validate(self) -> None:
        for name in ("jetson_port", "detector_port"):
            port = getattr(self, name)
            if not 1 <= port <= 65535:
                raise ConfigError(f"hardware.{name} out of range: {port}")
        if self.arm_bridge not in self.ARM_BRIDGES:
            raise ConfigError(
                f"hardware.arm_bridge must be one of {list(self.ARM_BRIDGES)}, got {self.arm_bridge!r}"
            )
        for name in ("request_timeout_s", "trajectory_timeout_margin_s", "trajectory_chunk_s"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"hardware.{name} must be > 0")
        if self.settle_steps_after_motion < 0:
            raise ConfigError("hardware.settle_steps_after_motion must be >= 0")
        self.arm.validate()
        for label, size in self.object_sizes.items():
            if len(size) != 3 or any(s <= 0.0 for s in size):
                raise ConfigError(
                    f"hardware.object_sizes[{label!r}] must be a positive (x, y, z) triple, got {size}"
                )
        if len(self.default_object_size) != 3 or any(s <= 0.0 for s in self.default_object_size):
            raise ConfigError(
                f"hardware.default_object_size must be a positive (x, y, z) triple, "
                f"got {self.default_object_size}"
            )
        if not 0.0 <= self.detection_min_score <= 1.0:
            raise ConfigError("hardware.detection_min_score must be in [0, 1]")
        if self.pixel_anchor not in self.PIXEL_ANCHORS:
            raise ConfigError(
                f"hardware.pixel_anchor must be one of {list(self.PIXEL_ANCHORS)}, "
                f"got {self.pixel_anchor!r}"
            )
        if self.workspace_margin_xy < 0.0:
            raise ConfigError("hardware.workspace_margin_xy must be >= 0")
        if not self.grasp_pitch_angles_deg:
            raise ConfigError("hardware.grasp_pitch_angles_deg must list at least one pitch")
        for pitch in self.grasp_pitch_angles_deg:
            if not 0.0 < pitch <= 90.0:
                raise ConfigError(
                    f"hardware.grasp_pitch_angles_deg entries must be in (0, 90], got {pitch}"
                )


def _default_wrist_camera() -> CameraConfig:
    """Wrist camera default: attached to the hand so it moves with the gripper.

    The two camera fields need *distinct* defaults, not a shared ``CameraConfig()``:
    their names index observation dicts and are the GR00T modality keys, so
    identical names are a validation error.
    """
    return CameraConfig(
        name="wrist_camera",
        prim_path="/World/Franka/panda_hand/wrist_camera",
        parent_prim="/World/Franka/panda_hand",
        position=(0.05, 0.0, 0.02),
        quat=(0.0, 0.0, 1.0, 0.0),
        clipping_range=(0.01, 3.0),
    )


def _default_exterior_camera() -> CameraConfig:
    """Exterior camera default: static three-quarter view of the workspace."""
    return CameraConfig(
        name="exterior_camera",
        prim_path="/World/exterior_camera",
        parent_prim="",
        focal_length=2.4,
        position=(1.4, -0.8, 1.1),
        look_at=(0.5, 0.0, 0.45),
        clipping_range=(0.05, 6.0),
    )


@dataclass(frozen=True)
class FrameworkConfig:
    """Root configuration object. Everything tunable hangs off this."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    wrist_camera: CameraConfig = field(default_factory=lambda: _default_wrist_camera())
    exterior_camera: CameraConfig = field(default_factory=lambda: _default_exterior_camera())
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    gr00t: Gr00tConfig = field(default_factory=Gr00tConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    default_executor: str = "classical"
    """Backend used when a skill does not name one. ``classical`` or ``gr00t``."""
    executor_overrides: Mapping[str, str] = field(default_factory=dict)
    """Per-skill backend selection, e.g. ``{"pick": "gr00t"}``."""
    backend: str = "sim"
    """What the runtime is built on: ``sim`` (Isaac Sim, the default) or
    ``hardware`` (the Jetson-hosted arm and detector; never imports Isaac)."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    BACKENDS = ("sim", "hardware")

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and hasattr(value, "validate"):
                value.validate()
        if self.backend not in self.BACKENDS:
            raise ConfigError(f"backend must be one of {list(self.BACKENDS)}, got {self.backend!r}")
        if self.backend == "hardware" and self.robot.kinematics != "planar_4dof":
            raise ConfigError(
                f"backend is 'hardware' but robot.kinematics is {self.robot.kinematics!r}; "
                "the hardware lane has no Lula and needs 'planar_4dof'"
            )
        if self.default_executor not in ("classical", "gr00t"):
            raise ConfigError(
                f"default_executor must be 'classical' or 'gr00t', got {self.default_executor!r}"
            )
        for skill, backend in self.executor_overrides.items():
            if backend not in ("classical", "gr00t"):
                raise ConfigError(
                    f"executor_overrides[{skill!r}] must be 'classical' or 'gr00t', got {backend!r}"
                )
        if self.default_executor == "gr00t" and not self.gr00t.enabled:
            raise ConfigError("default_executor is 'gr00t' but gr00t.enabled is false")
        if "gr00t" in self.executor_overrides.values() and not self.gr00t.enabled:
            raise ConfigError("an executor_override selects 'gr00t' but gr00t.enabled is false")
        if self.wrist_camera.name == self.exterior_camera.name:
            raise ConfigError(
                f"wrist_camera and exterior_camera share the name {self.wrist_camera.name!r}; "
                "camera names index observation dicts and must be distinct"
            )

    def executor_for(self, skill_name: str) -> str:
        """Resolve which backend runs a given skill."""
        return self.executor_overrides.get(skill_name, self.default_executor)


T = TypeVar("T")


def _coerce(value: Any, target_type: Any, path: str) -> Any:
    """Recursively convert plain YAML data into the declared dataclass types.

    Tuples matter: the schema uses tuples so config objects stay hashable and
    immutable, but YAML only produces lists, so every sequence field needs an
    explicit conversion.
    """
    origin = get_origin(target_type)

    if is_dataclass(target_type) and isinstance(value, Mapping):
        return _build(target_type, value, path)

    if origin is tuple:
        args = get_args(target_type)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} elements, got {len(value)}")
        return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))

    if origin is list:
        (arg,) = get_args(target_type) or (Any,)
        return [_coerce(v, arg, f"{path}[{i}]") for i, v in enumerate(value)]

    # ``get_origin(typing.Mapping[K, V])`` is ``collections.abc.Mapping``, not
    # ``typing.Mapping``; comparing against the typing alias alone matches
    # nothing and lets a typed mapping fall through with its values uncoerced.
    if origin in (dict, _abc.Mapping) or target_type in (dict, Mapping, _abc.Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        args = get_args(target_type)
        if len(args) != 2:
            return dict(value)
        # Typed mappings coerce their values too, or ``Mapping[str, tuple[...]]``
        # would quietly hold YAML lists and fail the "sequences become tuples"
        # guarantee the rest of the schema relies on.
        key_t, val_t = args
        return {
            _coerce(k, key_t, f"{path}[{k!r}]"): _coerce(v, val_t, f"{path}[{k!r}]")
            for k, v in value.items()
        }

    if target_type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value


def _resolved_hints(cls: type) -> dict[str, Any]:
    """Resolve a dataclass's annotations to real types.

    This module uses ``from __future__ import annotations``, so
    ``dataclasses.Field.type`` is the *string* form of the annotation. Coercing
    against those strings would silently match nothing and leave every tuple
    field as a list, so resolve them properly instead.
    """
    return get_type_hints(cls)


def _build(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'root'}: expected a mapping, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"{path or 'root'}: unknown config key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        sub_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(value, hints[name], sub_path)

    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ConfigError(f"{path or 'root'}: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` recursively, without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a plain dict."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError(
            "PyYAML is required to load config files. Install it into the Isaac "
            "Sim interpreter with: python.bat -m pip install pyyaml"
        ) from exc

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return dict(loaded)


def _load_with_inheritance(path: Path, seen: list[Path]) -> dict[str, Any]:
    """Load a config file, resolving its ``extends:`` chain.

    A specialised config -- a benchmark scene, an ablation, a debugging variant
    -- differs from the default in a handful of keys. Without inheritance the
    only options are to duplicate the whole file or to lose the parts you did
    not restate, and both are bad: the duplicate silently drifts from the
    original, and the omission silently reverts tuned physics to dataclass
    defaults. Either way the config *looks* right while the run is wrong.

    ``extends`` is resolved relative to the extending file, so a config
    directory can be moved or copied without rewriting paths.
    """
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in [*seen, resolved])
        raise ConfigError(f"circular config inheritance: {chain}")
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")

    data = _read_yaml(resolved)
    base_ref = data.pop("extends", None)
    if base_ref is None:
        return data

    if not isinstance(base_ref, str):
        raise ConfigError(f"{resolved}: 'extends' must be a path string, got {type(base_ref).__name__}")

    base_path = (resolved.parent / base_ref).resolve()
    base = _load_with_inheritance(base_path, [*seen, resolved])
    return _deep_merge(base, data)


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> FrameworkConfig:
    """Load and validate configuration from YAML.

    A config may declare ``extends: <path>`` to inherit from another file; the
    extending file's keys are deep-merged over the base's.

    ``overrides`` is deep-merged over the result, which is how tests and CLI
    flags adjust single values without duplicating a whole config file.

    Raises :class:`ConfigError` on anything malformed -- fail at startup, not
    halfway through a grasp.
    """
    data: dict[str, Any] = {}

    if path is not None:
        data = _load_with_inheritance(Path(path), [])

    if overrides:
        data = _deep_merge(data, overrides)

    cfg = _build(FrameworkConfig, data)
    cfg.validate()
    return cfg


def config_to_dict(cfg: FrameworkConfig) -> dict[str, Any]:
    """Serialise a config back to plain data, for logging the exact run parameters."""
    return dataclasses.asdict(cfg)

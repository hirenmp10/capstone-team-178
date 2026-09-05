"""PhysX solver and material configuration for stable contact-based grasping.

Isaac Sim is imported lazily inside functions; see ``mfw.simulation.app``.

Why this module exists
----------------------
A parallel-jaw grasp of a rigid body is one of the harder things to ask of a
physics solver. With PhysX defaults the usual outcome is that the object
buzzes between the fingers and squirts out sideways -- and that failure is
precisely what pushes people into "fake attachment" (parenting the object to the
hand, or writing its pose every frame). Fake attachment is banned here, so the
solver has to be set up to make real contact hold:

* **High friction on the fingers.** The Franka's rubber pads are far grippier
  than PhysX's 0.5 default. Without this the object slides out under gravity no
  matter how hard the fingers squeeze.
* **Many position iterations.** Contact between two nearly-parallel surfaces is
  a stiff constraint problem; the default 4 iterations leaves visible
  penetration and jitter. 32 resolves it.
* **Zero restitution.** Any bounce on finger contact ejects the object.
* **No sleeping, no stabilization.** Both damp the small contact velocities a
  steady grasp depends on, and a slept object stops responding to the hand.
* **CCD.** A fast lift can otherwise tunnel a thin object through a fingertip.
"""

from __future__ import annotations

from typing import Any

from mfw.config.schema import PhysicsConfig
from mfw.utils.logging import get_logger

__all__ = [
    "configure_physx_scene",
    "create_physics_material",
    "bind_physics_material",
    "configure_rigid_body",
    "configure_gripper_contacts",
]

_log = get_logger("physics.materials")


def configure_physx_scene(world: Any, config: PhysicsConfig) -> None:
    """Apply solver settings to the stage's PhysX scene.

    Applied to the ``UsdPhysics.Scene`` prim rather than passed to ``World``
    because several of these knobs (GPU dynamics, solver type) only exist on the
    PhysX schema and are silently ignored elsewhere.
    """
    from pxr import PhysxSchema, UsdPhysics  # noqa: PLC0415

    stage = world.stage
    scene_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_prim = prim
            break

    if scene_prim is None:
        # World.reset() normally creates this; if the caller configures physics
        # before the first reset we create it ourselves rather than silently
        # applying nothing.
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene_prim = scene.GetPrim()
        _log.info("Created /physicsScene (none existed yet)")

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateSolverTypeAttr().Set("TGS")
    """TGS over the default PGS: substantially better at the stiff, nearly
    redundant contact constraints a two-finger grasp creates."""
    physx_scene.CreateEnableCCDAttr().Set(config.enable_ccd)
    physx_scene.CreateEnableStabilizationAttr().Set(config.stabilization_threshold > 0.0)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(config.gpu_dynamics)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU" if config.gpu_dynamics else "MBP")

    _log.info(
        "PhysX scene: solver=TGS ccd=%s gpu_dynamics=%s stabilization=%s",
        config.enable_ccd,
        config.gpu_dynamics,
        config.stabilization_threshold > 0.0,
    )


def create_physics_material(
    stage: Any,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> Any:
    """Create (or fetch) a ``UsdPhysics.MaterialAPI`` at ``path``.

    Idempotent: re-running scene setup must not stack duplicate materials.
    """
    from pxr import UsdPhysics, UsdShade  # noqa: PLC0415

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        UsdShade.Material.Define(stage, path)
        prim = stage.GetPrimAtPath(path)

    material = UsdPhysics.MaterialAPI.Apply(prim)
    material.CreateStaticFrictionAttr().Set(float(static_friction))
    material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material.CreateRestitutionAttr().Set(float(restitution))
    return material


def bind_physics_material(stage: Any, prim_path: str, material_path: str) -> bool:
    """Bind a physics material to a prim and all its descendants.

    Returns ``False`` if the prim does not exist, so callers can log a real
    warning instead of silently proceeding with default friction -- a
    mis-pathed finger prim would otherwise look like a mysterious grasp failure.
    """
    from pxr import UsdShade  # noqa: PLC0415

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        _log.warning("Cannot bind physics material: prim %s does not exist", prim_path)
        return False

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid():
        _log.warning("Cannot bind physics material: material %s does not exist", material_path)
        return False

    material = UsdShade.Material(material_prim)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return True


def configure_rigid_body(stage: Any, prim_path: str, config: PhysicsConfig) -> bool:
    """Apply per-body PhysX settings to a manipulable object.

    The offsets and the sleep/stabilization overrides are what let a grasped
    object stay put. ``contact_offset`` in particular has to be small relative
    to the fingertip: too large and the solver starts resolving contact before
    the fingers touch, producing a visible floating grasp.
    """
    from pxr import PhysxSchema  # noqa: PLC0415

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        _log.warning("Cannot configure rigid body: prim %s does not exist", prim_path)
        return False

    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr().Set(int(config.solver_position_iterations))
    rb.CreateSolverVelocityIterationCountAttr().Set(int(config.solver_velocity_iterations))
    rb.CreateEnableCCDAttr().Set(bool(config.enable_ccd))
    rb.CreateMaxDepenetrationVelocityAttr().Set(float(config.max_depenetration_velocity))
    # Zero keeps the body permanently awake. A slept object ignores the hand.
    rb.CreateSleepThresholdAttr().Set(float(config.sleep_threshold))
    rb.CreateStabilizationThresholdAttr().Set(float(config.stabilization_threshold))

    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr().Set(float(config.contact_offset))
    collision.CreateRestOffsetAttr().Set(float(config.rest_offset))
    return True


def configure_gripper_contacts(stage: Any, config: PhysicsConfig, finger_prim_paths: list[str]) -> None:
    """Give the fingertips their own high-friction material.

    Separate from the general object material because the fingers are the one
    contact pair that must not slip. Using a single global friction value high
    enough for the fingers would also make objects unrealistically sticky
    against the table and each other.
    """
    material_path = "/World/PhysicsMaterials/finger_material"
    create_physics_material(
        stage,
        material_path,
        static_friction=config.finger_static_friction,
        dynamic_friction=config.finger_dynamic_friction,
        restitution=0.0,
    )

    bound = 0
    for path in finger_prim_paths:
        if bind_physics_material(stage, path, material_path):
            bound += 1
        # Fine contact offsets on the fingers themselves, so contact is
        # detected at the pad surface rather than a shell around it.
        from pxr import PhysxSchema  # noqa: PLC0415

        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(float(config.contact_offset))
            collision.CreateRestOffsetAttr().Set(float(config.rest_offset))

    _log.info(
        "Finger material bound to %d/%d prims (static_friction=%.2f)",
        bound,
        len(finger_prim_paths),
        config.finger_static_friction,
    )

"""Scene construction.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

Objects are spawned from :class:`~mfw.config.schema.SceneConfig`, given
semantic labels so the segmentation annotator can see them, and configured with
the PhysX settings that make contact-based grasping stable.

The ground-truth poses used here are *write-only* from the framework's point of
view: they build the world and let tests compare against truth. Nothing in the
perception, planning or skill layers may read them. That rule is what makes the
"never assume object positions" requirement structural instead of aspirational --
runtime code has no accessor that would return them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mfw.config.schema import (
    PhysicsConfig,
    SceneConfig,
    SceneFurnitureConfig,
    SceneObjectConfig,
)
from mfw.core.errors import SimulationError
from mfw.physics.collision import apply_mesh_colliders, collider_approximation_for
from mfw.physics.materials import (
    bind_physics_material,
    configure_physx_scene,
    configure_rigid_body,
    create_physics_material,
)
from mfw.simulation.asset_registry import AssetRegistry
from mfw.utils.logging import get_logger

__all__ = ["SceneBuilder"]

_log = get_logger("simulation.scene")

_OBJECT_MATERIAL_PATH = "/World/PhysicsMaterials/object_material"


def _compose_quat(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product ``outer * inner``, both (w, x, y, z).

    Applies ``inner`` first: the asset's upright correction, then whatever
    orientation the scene asked for.
    """
    w0, x0, y0, z0 = (float(v) for v in outer)
    w1, x1, y1, z1 = (float(v) for v in inner)
    return (
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )


class SceneBuilder:
    """Builds the workspace: room, furniture, table and manipulable objects."""

    def __init__(
        self,
        sim: Any,
        scene_config: SceneConfig,
        physics_config: PhysicsConfig,
        registry: AssetRegistry | None = None,
    ) -> None:
        scene_config.validate()
        physics_config.validate()
        self.config = scene_config
        self.physics_config = physics_config
        self._sim = sim
        self._object_prim_paths: dict[str, str] = {}
        self._furniture_prim_paths: dict[str, str] = {}

        # Loaded lazily and only when something actually references the
        # catalogue, so the primitive-only test scenes never touch the YAML.
        self._registry = registry
        self._needs_registry = bool(
            scene_config.environment
            or scene_config.furniture
            or any(o.kind == "asset" for o in scene_config.objects)
        )
        if self._registry is None and self._needs_registry:
            self._registry = AssetRegistry.load(scene_config.assets_path or None)

    @property
    def registry(self) -> AssetRegistry:
        if self._registry is None:
            self._registry = AssetRegistry.load(self.config.assets_path or None)
        return self._registry

    @property
    def object_prim_paths(self) -> dict[str, str]:
        """Spawned object name -> prim path. For tests and logging only."""
        return dict(self._object_prim_paths)

    @property
    def furniture_prim_paths(self) -> dict[str, str]:
        """Spawned furniture name -> prim path."""
        return dict(self._furniture_prim_paths)

    def build(self) -> None:
        """Construct the full scene. Call before ``World.reset()``."""
        configure_physx_scene(self._sim.world, self.physics_config)

        if self.config.environment:
            self._add_environment()

        if self.config.add_ground_plane:
            self._sim.world.scene.add_default_ground_plane()

        self._add_lighting()

        create_physics_material(
            self._sim.world.stage,
            _OBJECT_MATERIAL_PATH,
            static_friction=self.physics_config.static_friction,
            dynamic_friction=self.physics_config.dynamic_friction,
            restitution=self.physics_config.restitution,
        )

        if self.config.add_table:
            self._add_table()

        for item in self.config.furniture:
            self._add_furniture(item)

        for obj in self.config.objects:
            self._add_object(obj)

        _log.info(
            "Scene built: environment=%s table=%s furniture=%d objects=%d",
            self.config.environment or "none",
            self.config.add_table,
            len(self.config.furniture),
            len(self.config.objects),
        )

    @staticmethod
    def _set_local_transform(
        prim: Any,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Place a referenced prim using a single matrix op.

        Not ``AddTranslateOp`` / ``AddOrientOp`` / ``AddScaleOp``. Those helpers
        default to a fixed precision -- ``AddOrientOp`` creates a ``quatf`` --
        and if the referenced asset already declares ``xformOp:orient`` at
        double precision, redeclaring it at float is a USD type conflict.

        That is not a recoverable exception. Isaac promotes the resulting USD
        coding error to a hard abort: the process dies mid-build with exit code
        0, no traceback and no message. Measured here on
        ``sektion_cabinet_instanceable.usd``, which took down the whole scene
        build between one furniture item and the next.

        ``xformOp:transform`` has exactly one type (``matrix4d``), so there is
        no precision to conflict over.
        """
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        rotation = Gf.Matrix4d().SetRotate(
            Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])))
        )
        scaling = Gf.Matrix4d().SetScale(Gf.Vec3d(*(float(s) for s in scale)))
        translation = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(float(p) for p in position)))

        # Gf uses the row-vector convention (v * M), so scale, then rotate,
        # then translate reads left to right.
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.MakeMatrixXform().Set(scaling * rotation * translation)

    def _assets_root(self) -> str:
        from isaacsim.storage.native import get_assets_root_path  # noqa: PLC0415

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise SimulationError(
                "Isaac assets root is unreachable; cannot load USD assets. Check the "
                "Nucleus/asset-server connection."
            )
        return assets_root

    def _add_environment(self) -> None:
        """Reference a room USD under the workspace.

        Referenced, not merged: the room keeps its own layer, so its prims never
        collide with ``/World/objects`` and the whole thing can be swapped by
        changing one config key.
        """
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: PLC0415

        usd_path = self.registry.environment(self.config.environment)
        prim_path = "/World/environment"
        add_reference_to_stage(usd_path=self._assets_root() + usd_path, prim_path=prim_path)

        prim = self._sim.world.stage.GetPrimAtPath(prim_path)
        self._set_local_transform(prim, self.config.environment_position)

        _log.info("Environment %r loaded from %s", self.config.environment, usd_path)

    def _add_lighting(self) -> None:
        """Add explicit, configured scene lighting.

        A dome light for uniform ambient fill plus a distant key light for
        directional shading. Without this the scene relies on whatever implicit
        light the default ground plane provides, whose intensity falls off fast
        enough that a camera 1.5 m from the table renders essentially black
        while a wrist camera 30 cm away looks perfectly lit -- a discrepancy
        that surfaces downstream as "the detector sees nothing from the
        exterior view".
        """
        from pxr import Sdf, UsdLux  # noqa: PLC0415

        stage = self._sim.world.stage

        if self.config.dome_light_intensity > 0.0:
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome_light"))
            dome.CreateIntensityAttr().Set(float(self.config.dome_light_intensity))
            dome.CreateColorAttr().Set((1.0, 1.0, 1.0))

        if self.config.distant_light_intensity > 0.0:
            from pxr import Gf, UsdGeom  # noqa: PLC0415

            distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/key_light"))
            distant.CreateIntensityAttr().Set(float(self.config.distant_light_intensity))
            distant.CreateAngleAttr().Set(1.0)
            xform = UsdGeom.Xformable(distant.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*self.config.distant_light_angle))

        if self.config.workspace_light_intensity > 0.0:
            self._add_workspace_light()

        _log.info(
            "Lighting: dome=%.0f, distant=%.0f, workspace=%.0f",
            self.config.dome_light_intensity,
            self.config.distant_light_intensity,
            self.config.workspace_light_intensity,
        )

    def _add_workspace_light(self) -> None:
        """A local sphere light directly above the table.

        The only light that works once the scene has a ceiling. Dome and distant
        lights are exterior sources -- a sky and a sun -- and a room USD blocks
        both. See ``SceneConfig.workspace_light_intensity`` for the measurements.

        Placed relative to the table rather than at a fixed world height, so
        moving the workspace moves its illumination with it.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: PLC0415

        stage = self._sim.world.stage
        table_top = self.config.table_position[2] + self.config.table_scale[2] / 2.0

        light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/lights/workspace_light"))
        light.CreateIntensityAttr().Set(float(self.config.workspace_light_intensity))
        light.CreateRadiusAttr().Set(float(self.config.workspace_light_radius))
        light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        # Not a physical object: it must not occlude the cameras looking down
        # through it, nor appear as an instance to the segmentation annotator.
        light.CreateTreatAsPointAttr().Set(False)
        UsdLux.ShapingAPI.Apply(light.GetPrim())

        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(
            Gf.Vec3d(
                float(self.config.table_position[0]),
                float(self.config.table_position[1]),
                float(table_top + self.config.workspace_light_height),
            )
        )

    def _add_table(self) -> None:
        """Add a static table as a fixed cuboid.

        Static rather than a dynamic body with high mass: a static collider
        cannot be nudged by a contact and cannot contribute solver work, which
        keeps the grasp contact problem as small as possible.
        """
        from isaacsim.core.api.objects import FixedCuboid  # noqa: PLC0415

        table = FixedCuboid(
            prim_path="/World/table",
            name="table",
            position=np.array(self.config.table_position, dtype=np.float64),
            scale=np.array(self.config.table_scale, dtype=np.float64),
            color=np.array([0.4, 0.35, 0.3]),
        )
        self._sim.world.scene.add(table)
        bind_physics_material(self._sim.world.stage, "/World/table", _OBJECT_MATERIAL_PATH)
        self._apply_semantics("/World/table", "table")

    def _add_object(self, obj: SceneObjectConfig) -> None:
        """Spawn one manipulable rigid body."""
        from isaacsim.core.api.objects import (  # noqa: PLC0415
            DynamicCuboid,
            DynamicCylinder,
            DynamicSphere,
        )

        prim_path = f"/World/objects/{obj.name}"
        position = np.array(obj.position, dtype=np.float64)
        orientation = np.array(obj.quat, dtype=np.float64)
        color = np.array(obj.color, dtype=np.float64)

        if obj.kind == "cuboid":
            prim = DynamicCuboid(
                prim_path=prim_path,
                name=obj.name,
                position=position,
                orientation=orientation,
                scale=np.array(obj.resolved_scale(), dtype=np.float64),
                color=color,
                mass=float(obj.mass),
            )
        elif obj.kind == "cylinder":
            # scale is (radius, radius, height) for cylinders.
            prim = DynamicCylinder(
                prim_path=prim_path,
                name=obj.name,
                position=position,
                orientation=orientation,
                radius=float(obj.resolved_scale()[0]),
                height=float(obj.resolved_scale()[2]),
                color=color,
                mass=float(obj.mass),
            )
        elif obj.kind == "sphere":
            prim = DynamicSphere(
                prim_path=prim_path,
                name=obj.name,
                position=position,
                orientation=orientation,
                radius=float(obj.resolved_scale()[0]),
                color=color,
                mass=float(obj.mass),
            )
        elif obj.kind in ("usd", "asset"):
            self._add_usd_object(obj, prim_path)
            return  # _add_usd_object owns the rest of the setup
        else:  # pragma: no cover - guarded by SceneObjectConfig.validate
            raise SimulationError(f"Unsupported object kind {obj.kind!r}")

        self._sim.world.scene.add(prim)
        bind_physics_material(self._sim.world.stage, prim_path, _OBJECT_MATERIAL_PATH)
        configure_rigid_body(self._sim.world.stage, prim_path, self.physics_config)
        self._apply_semantics(prim_path, obj.semantic_label or obj.name)
        self._object_prim_paths[obj.name] = prim_path

    def _add_usd_object(self, obj: SceneObjectConfig, prim_path: str) -> None:
        """Reference a mesh asset and make it a properly-collidable rigid body.

        The collider authoring is the part that matters. Seventeen of the
        twenty-one YCB assets are visual-only: they contain meshes and nothing
        else. ``SingleRigidPrim`` will happily wrap one and give you a rigid
        body with a valid pose, a correct mass and no collision geometry
        whatsoever, which falls through the table on the first physics step.

        Colliders are authored before the wrapper is constructed so the stage is
        complete before anything reads it, and the result is *checked* -- see
        the zero-collider branch below -- rather than assumed.
        """
        from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: PLC0415

        spec = self.registry.object(obj.asset) if obj.kind == "asset" else None
        usd_subpath = spec.usd if spec is not None else obj.usd_subpath
        # The registry is the single source of truth for an asset's physical
        # facts; the scene entry only says where to put it.
        mass = spec.mass_kg if spec is not None else float(obj.mass)
        label = obj.semantic_label or (spec.semantic if spec is not None else obj.name)

        # The scene's orientation is applied *on top of* the asset's upright
        # correction, so a config author writes the yaw they want and never has
        # to know that the mesh underneath is Y-up.
        orientation = (
            _compose_quat(obj.quat, spec.upright_quat) if spec is not None else obj.quat
        )

        add_reference_to_stage(usd_path=self._assets_root() + usd_subpath, prim_path=prim_path)

        approximation = collider_approximation_for(
            spec.category if spec is not None else "",
            obj.collision_approximation or (spec.collision_approximation if spec else ""),
        )
        collider_count = apply_mesh_colliders(
            self._sim.world.stage, prim_path, approximation, self.physics_config
        )
        if collider_count == 0:
            # Not recoverable, and not something to warn about and continue: a
            # rigid body with no collider falls through the floor on step one
            # and the object is simply gone.
            raise SimulationError(
                f"asset {obj.name!r} ({usd_subpath}) produced no collision geometry. "
                f"The USD reference resolved but contains no Mesh prims."
            )

        rigid = SingleRigidPrim(
            prim_path=prim_path,
            name=obj.name,
            position=np.array(obj.position, dtype=np.float64),
            orientation=np.array(orientation, dtype=np.float64),
            scale=np.array(obj.resolved_scale(), dtype=np.float64),
            mass=mass,
        )
        self._sim.world.scene.add(rigid)

        bind_physics_material(self._sim.world.stage, prim_path, _OBJECT_MATERIAL_PATH)
        configure_rigid_body(self._sim.world.stage, prim_path, self.physics_config)
        self._apply_semantics(prim_path, label)
        self._object_prim_paths[obj.name] = prim_path

        _log.info(
            "Spawned %s: %s, %.3f kg, %d %s collider(s)",
            obj.name,
            usd_subpath.rsplit("/", 1)[-1],
            mass,
            collider_count,
            approximation,
        )

    def _add_furniture(self, item: SceneFurnitureConfig) -> None:
        """Place a static fixture from the catalogue.

        Static, so no ``RigidBodyAPI``: furniture is scenery and an obstacle,
        never a manipulation target. Colliders are still authored (so the motion
        planner will not route the arm through a chair) using the exact-mesh
        ``none`` approximation, which for a *static* body means "use the
        triangle mesh directly". That is both cheaper and more accurate than a
        convex decomposition, and it is only available because the body never
        moves -- PhysX cannot simulate a concave dynamic collider.
        """
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: PLC0415

        spec = self.registry.furniture_item(item.asset)
        prim_path = f"/World/furniture/{item.name}"
        add_reference_to_stage(usd_path=self._assets_root() + spec.usd, prim_path=prim_path)

        stage = self._sim.world.stage
        prim = stage.GetPrimAtPath(prim_path)
        self._set_local_transform(prim, item.position, item.quat, item.scale)

        if item.collision:
            count = apply_mesh_colliders(stage, prim_path, "none", self.physics_config)
            if count == 0:
                _log.warning(
                    "Furniture %r has no mesh geometry; it will be visible but the "
                    "planner will route through it",
                    item.name,
                )

        self._apply_semantics(prim_path, spec.semantic)
        self._furniture_prim_paths[item.name] = prim_path
        _log.info("Placed furniture %s (%s) at %s", item.name, spec.semantic, item.position)

    def _apply_semantics(self, prim_path: str, label: str) -> None:
        """Tag a prim so the segmentation annotator reports it.

        Without a semantic tag the instance segmentation annotator returns the
        prim as background, and perception simply cannot see the object -- a
        failure that looks like a broken detector rather than missing metadata.
        """
        stage = self._sim.world.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            _log.warning("Cannot apply semantics: prim %s does not exist", prim_path)
            return

        try:
            from isaacsim.core.utils.semantics import add_labels  # noqa: PLC0415

            add_labels(prim, labels=[label], instance_name="class")
        except ImportError:
            try:
                from isaacsim.core.utils.semantics import (  # noqa: PLC0415
                    add_update_semantics,
                )

                add_update_semantics(prim, semantic_label=label, type_label="class")
            except ImportError:  # pragma: no cover - version dependent
                _log.warning("No semantics API available; segmentation labels will be empty")

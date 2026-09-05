"""Collider authoring for referenced USD meshes.

Isaac Sim is imported lazily inside functions; see ``mfw.simulation.app``.

The problem this solves
-----------------------
``PhysxSchema.PhysxRigidBodyAPI`` and ``PhysxCollisionAPI`` are applied to an
object's *root* prim. That is correct for a primitive spawned by
``DynamicCuboid`` -- the root prim carries the geometry. It is a no-op for a
referenced USD asset, where the root is an ``Xform`` and the geometry lives on
``Mesh`` prims somewhere below it.

Applying only the root APIs to a YCB mesh produces a rigid body with **no
collider**. It falls through the table on the first physics step. Nothing warns
you: the rigid body is genuinely configured, the mesh genuinely renders, and the
object genuinely has a pose -- it is simply not solid. Downstream this reads as
"the object vanished", which sends you looking at perception.

Only four of the twenty-one YCB assets shipped with Isaac Sim
(``Axis_Aligned_Physics``) have colliders pre-authored. The other seventeen are
visual-only, so this module is what makes them usable.

Choosing an approximation
-------------------------
PhysX cannot simulate arbitrary concave geometry as a dynamic body, so a mesh
collider must be approximated:

* ``convexHull`` -- one convex shell. Cheapest and stable. Correct for anything
  genuinely convex: cans, boxes, blocks.
* ``convexDecomposition`` -- several convex pieces. Needed whenever the *cavity*
  matters. A mug's hull is a solid lump: objects placed "in" it rest on an
  invisible lid, and the handle is filled in, so a handle grasp closes on
  nothing.
* ``boundingCube`` -- the AABB. A deliberate fallback, not a default.
* ``sdf`` -- signed-distance field, the most accurate for fine features, and
  much more expensive. Reserved for thin geometry like scissors, where a convex
  decomposition of a 14 mm-thick object tends to weld the blades together.
"""

from __future__ import annotations

from typing import Any

from mfw.config.schema import PhysicsConfig
from mfw.utils.logging import get_logger

__all__ = [
    "apply_mesh_colliders",
    "collider_approximation_for",
    "count_colliders",
    "VALID_APPROXIMATIONS",
]

_log = get_logger("physics.collision")

VALID_APPROXIMATIONS = (
    "convexHull",
    "convexDecomposition",
    "boundingCube",
    "boundingSphere",
    "meshSimplification",
    "sdf",
    "none",
)

# Category -> collider strategy. Concave-by-nature categories get a
# decomposition; everything else gets a hull.
_APPROXIMATION_BY_CATEGORY = {
    "can": "convexHull",
    "bottle": "convexHull",
    "box": "convexHull",
    "block": "convexHull",
    "mug": "convexDecomposition",      # handle + cavity
    "bowl": "convexDecomposition",     # cavity is the whole point
    "pitcher": "convexDecomposition",  # handle + spout + cavity
    "food": "convexDecomposition",     # banana's curve is not convex
    "tool": "convexDecomposition",     # handles, jaws, trigger guards
}

_DEFAULT_APPROXIMATION = "convexDecomposition"
"""Safe default: a decomposition of a convex mesh degenerates to its hull, so
guessing wrong here costs cook time, not correctness. Guessing ``convexHull``
wrong loses the cavity silently."""


def collider_approximation_for(category: str, override: str = "") -> str:
    """Pick a collider approximation for an object category."""
    if override:
        if override not in VALID_APPROXIMATIONS:
            raise ValueError(
                f"unknown collision approximation {override!r}; "
                f"valid values are {VALID_APPROXIMATIONS}"
            )
        return override
    return _APPROXIMATION_BY_CATEGORY.get(category, _DEFAULT_APPROXIMATION)


def apply_mesh_colliders(
    stage: Any,
    prim_path: str,
    approximation: str,
    config: PhysicsConfig,
    max_convex_hulls: int = 32,
) -> int:
    """Author colliders on every mesh under ``prim_path``.

    Returns the number of meshes given a collider. **Zero is a failure**, not an
    empty success -- it means the asset had no mesh where one was expected, and
    the resulting body will fall through the world. Callers should treat it as
    an error rather than logging and continuing.
    """
    from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: PLC0415

    if approximation not in VALID_APPROXIMATIONS:
        raise ValueError(
            f"unknown collision approximation {approximation!r}; "
            f"valid values are {VALID_APPROXIMATIONS}"
        )

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        _log.warning("Cannot author colliders: prim %s does not exist", prim_path)
        return 0

    existing = count_colliders(stage, prim_path)
    if existing > 0:
        # The asset ships its own colliders -- the four YCB *_Physics variants,
        # and articulated assets like the Sektion cabinet whose drawer joints
        # come with matched collision geometry. Overwriting those with our own
        # approximation would replace a tuned authored collider with a guess.
        _log.debug(
            "%s already has %d authored collider(s); leaving them alone", prim_path, existing
        )
        return existing

    count = 0
    for prim in _iter_self_and_descendants(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.IsInstanceProxy():
            # Instance proxies are read-only views onto a shared prototype.
            # Applying a schema to one is an authoring error on a prim that
            # does not really exist at that path.
            _log.debug("Skipping instance proxy %s", prim.GetPath())
            continue

        UsdPhysics.CollisionAPI.Apply(prim)

        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(approximation)

        if approximation == "convexDecomposition":
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            # Bounded so a detailed scan cannot explode into hundreds of hulls;
            # every hull is solver work on a body the gripper is squeezing.
            decomposition.CreateMaxConvexHullsAttr().Set(int(max_convex_hulls))
            decomposition.CreateHullVertexLimitAttr().Set(64)
            # Small voxel resolution preserves the features that matter for
            # grasping (handles, rims) instead of smoothing them away.
            decomposition.CreateVoxelResolutionAttr().Set(500_000)
            # Deliberately NOT shrink-wrapped. Shrink-wrap pulls each hull in
            # onto the source mesh, which sounds like more accuracy and is the
            # wrong direction here: a hull inside the visual surface lets the
            # rendered mesh sink into whatever it rests on. Measured on the YCB
            # banana, enabling it took the penetration from 9.5 mm to 12.5 mm.
            # Objects whose curvature defeats a decomposition should use an
            # ``sdf`` collider instead -- see the banana's catalogue entry.
        elif approximation == "sdf":
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf.CreateSdfResolutionAttr().Set(256)

        # Contact offsets must be set on the collider prim itself. On the root
        # Xform they are inherited by nothing and quietly do nothing.
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(float(config.contact_offset))
        collision.CreateRestOffsetAttr().Set(float(config.rest_offset))
        count += 1

    if count == 0:
        _log.warning(
            "No meshes found under %s -- the body will have no collider and will "
            "fall through the ground",
            prim_path,
        )
    else:
        _log.debug("Authored %d %s collider(s) under %s", count, approximation, prim_path)
    return count


def count_colliders(stage: Any, prim_path: str) -> int:
    """How many prims under ``prim_path`` carry a collision API.

    Used by the scene validator to prove an object is solid, rather than
    assuming it because the spawn call returned without raising.
    """
    from pxr import UsdPhysics  # noqa: PLC0415

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return 0

    return sum(
        1
        for prim in _iter_self_and_descendants(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _iter_self_and_descendants(prim: Any) -> Any:
    """Yield ``prim`` and everything beneath it.

    ``GetAllChildren`` rather than ``GetChildren``: YCB assets reference their
    geometry through payloads and class prims, and the filtered traversal used
    by ``GetChildren`` skips exactly those, silently returning zero meshes for
    an asset that visibly has one.
    """
    yield prim
    stack = list(prim.GetAllChildren())
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.GetAllChildren())

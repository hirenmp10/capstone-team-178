"""Procedural object placement and domain randomisation.

Pure NumPy. This module must never import Isaac Sim -- placement is decided
before anything is spawned, so it is fully testable without a GPU.

Placing objects is where a benchmark scene is quietly won or lost. Two failure
modes dominate, and both look like something else:

* **Overlapping spawns.** Two meshes sharing space are resolved by PhysX on the
  first step by driving them apart, hard. The scene appears to explode, and the
  natural conclusion is that the physics tuning is wrong. It is not; the layout
  was invalid before physics ever ran.
* **Unreachable spawns.** An object placed at the far corner of the table is a
  perfectly good object that the arm cannot touch. Every skill attempt fails at
  the IK stage, which reads as a planner bug.

So placement here is rejection sampling against both constraints, and it
*raises* when it cannot satisfy them rather than returning a partial layout.
A scene with four of six requested objects is a silently different experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import RandomizationConfig, SceneConfig, SceneObjectConfig
from mfw.core.errors import SimulationError
from mfw.simulation.asset_registry import AssetRegistry, AssetSpec
from mfw.utils.logging import get_logger

__all__ = ["PlacementRegion", "ObjectPlacer", "randomize_scene"]

_log = get_logger("simulation.layout")


SPAWN_CLEARANCE_M = 0.002
"""Gap left between an object and the surface it is spawned on.

Small on purpose. Objects should be placed *resting*, not dropped: a spawn
30 mm up becomes a bounce, and a bounce becomes a tipped bowl or a can that
rolls somewhere the scene author did not intend. Two millimetres is enough to
guarantee no initial interpenetration and little enough that settling is a
formality.
"""


@dataclass(frozen=True)
class PlacementRegion:
    """A rectangular patch of a support surface that objects may rest on.

    ``surface_z`` is the top of the surface. Turning that into a spawn height
    needs the object's *origin convention*, which differs by asset type -- see
    :meth:`spawn_z`.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    surface_z: float
    reach_center: tuple[float, float, float] | None = None
    reach_radius: float = 0.0
    """Outer arm envelope. Zero disables the test (for scenes with no robot)."""
    min_reach_radius: float = 0.0
    """Inner envelope. An arm has a dead zone around its own base: too close and
    it cannot fold enough to present the gripper, and the elbow collides with
    the torso. Ignoring it puts objects in a hole right in front of the robot
    that looks perfectly reachable on a distance-to-base plot."""
    approach_offset: float = 0.10
    """How far above an object the pre-grasp standoff sits. Reach is tested
    against *that* point, not the object, because the arm must get above the
    object before it can descend onto it -- and the standoff is always the
    harder of the two to reach."""

    def is_reachable(self, x: float, y: float, z: float) -> bool:
        """Whether the arm can actually get to the pre-grasp above this point.

        A sphere, not a box. The workspace box's far corner sits well outside
        anything the arm can touch, so a box test passes placements the planner
        will later reject -- and it rejects them with "no path", which reads as
        a planner bug rather than a layout one.
        """
        if self.reach_center is None or self.reach_radius <= 0.0:
            return True
        cx, cy, cz = self.reach_center
        distance = float(
            np.linalg.norm([x - cx, y - cy, (z + self.approach_offset) - cz])
        )
        if distance > self.reach_radius:
            return False
        return distance >= self.min_reach_radius

    def spawn_z(self, height: float) -> float:
        """Height at which to spawn an object of the given upright height.

        Both the YCB meshes and Isaac's primitives are **centred on their
        origin**, so this is ``surface + height/2`` throughout. Measured: the
        tomato soup can's origin sits at [33.8, 50.9, 33.9] mm from its bounding
        minimum, exactly half its [67.7, 101.9, 67.7] mm extent on every axis.

        ``height`` must be the *upright* height -- the Z extent after the
        asset's ``upright_quat`` is applied. The catalogue records extents that
        way for exactly this reason; using the raw Y-up bounds would place a can
        at half its diameter instead of half its height.

        Getting this wrong is not fatal -- physics resolves it either way --
        but it resolves it by dropping or ejecting the object, which is how a
        carefully composed layout turns into a different one.
        """
        return self.surface_z + height / 2.0 + SPAWN_CLEARANCE_M

    def contains(self, x: float, y: float) -> bool:
        return (
            self.x_range[0] <= x <= self.x_range[1]
            and self.y_range[0] <= y <= self.y_range[1]
        )

    @classmethod
    def from_table(
        cls,
        table_position: tuple[float, float, float],
        table_scale: tuple[float, float, float],
        margin: float = 0.06,
        reach_center: tuple[float, float, float] | None = None,
        reach_radius: float = 0.0,
        min_reach_radius: float = 0.0,
        approach_offset: float = 0.10,
    ) -> PlacementRegion:
        """The usable top of a cuboid table.

        ``margin`` insets the region from the physical edge. An object centred
        on the rim is half-overhanging: it topples during the settle phase, and
        the scene you validated is not the scene you tested against.
        """
        half_x = table_scale[0] / 2.0 - margin
        half_y = table_scale[1] / 2.0 - margin
        if half_x <= 0.0 or half_y <= 0.0:
            raise SimulationError(
                f"table {table_scale} is too small for a {margin} m placement margin"
            )
        return cls(
            x_range=(table_position[0] - half_x, table_position[0] + half_x),
            y_range=(table_position[1] - half_y, table_position[1] + half_y),
            surface_z=table_position[2] + table_scale[2] / 2.0,
            reach_center=reach_center,
            reach_radius=float(reach_radius),
            min_reach_radius=float(min_reach_radius),
            approach_offset=float(approach_offset),
        )

    def clipped_to_workspace(
        self,
        workspace_min: NDArray[np.float64],
        workspace_max: NDArray[np.float64],
    ) -> PlacementRegion:
        """Intersect with the arm's reachable envelope.

        Placing only where the robot can reach is not a nicety. An unreachable
        object makes every command about it fail at IK, and the failure surfaces
        far from its cause.
        """
        x_lo = max(self.x_range[0], float(workspace_min[0]))
        x_hi = min(self.x_range[1], float(workspace_max[0]))
        y_lo = max(self.y_range[0], float(workspace_min[1]))
        y_hi = min(self.y_range[1], float(workspace_max[1]))
        if x_lo >= x_hi or y_lo >= y_hi:
            raise SimulationError(
                f"the table's surface and the workspace envelope do not overlap: "
                f"table x{self.x_range} y{self.y_range} vs workspace "
                f"x({workspace_min[0]:.2f}, {workspace_max[0]:.2f}) "
                f"y({workspace_min[1]:.2f}, {workspace_max[1]:.2f})"
            )
        return PlacementRegion(
            (x_lo, x_hi),
            (y_lo, y_hi),
            self.surface_z,
            reach_center=self.reach_center,
            reach_radius=self.reach_radius,
            min_reach_radius=self.min_reach_radius,
            approach_offset=self.approach_offset,
        )


class ObjectPlacer:
    """Samples non-overlapping poses within a region."""

    def __init__(
        self,
        region: PlacementRegion,
        min_separation: float = 0.04,
        max_attempts: int = 200,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.region = region
        self.min_separation = float(min_separation)
        self.max_attempts = int(max_attempts)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._placed: list[tuple[NDArray[np.float64], float]] = []

    @property
    def placed_count(self) -> int:
        return len(self._placed)

    def reserve(self, position: NDArray[np.float64], radius: float) -> None:
        """Mark a footprint as occupied without sampling it.

        Used for explicitly-positioned objects, so that randomised ones are
        placed around them instead of on top of them.
        """
        self._placed.append((np.asarray(position, dtype=np.float64)[:2], float(radius)))

    def sample(self, footprint_radius: float, height: float) -> NDArray[np.float64]:
        """Find a free spot for an object of the given footprint.

        Raises rather than returning a colliding pose: an overlapping spawn is
        not a degraded scene, it is an invalid one.
        """
        radius = float(footprint_radius)
        x_lo = self.region.x_range[0] + radius
        x_hi = self.region.x_range[1] - radius
        y_lo = self.region.y_range[0] + radius
        y_hi = self.region.y_range[1] - radius
        if x_lo > x_hi or y_lo > y_hi:
            raise SimulationError(
                f"an object with a {radius * 2:.3f} m footprint does not fit in the "
                f"placement region x{self.region.x_range} y{self.region.y_range}"
            )

        for _ in range(self.max_attempts):
            candidate = np.array(
                [self._rng.uniform(x_lo, x_hi), self._rng.uniform(y_lo, y_hi)],
                dtype=np.float64,
            )
            spawn_z = self.region.spawn_z(height)
            if not self.region.is_reachable(candidate[0], candidate[1], spawn_z):
                continue
            if self.is_free(candidate, radius):
                self._placed.append((candidate, radius))
                return np.array(
                    [candidate[0], candidate[1], spawn_z], dtype=np.float64
                )

        raise SimulationError(
            f"could not place an object with a {radius:.3f} m radius after "
            f"{self.max_attempts} attempts ({len(self._placed)} already placed "
            f"within a {self.region.reach_radius:.2f} m reach). The reachable "
            f"region is too crowded -- reduce the object count, shrink "
            f"min_object_separation_m, or move the surface closer to the arm."
        )

    def is_free(self, candidate: NDArray[np.float64], radius: float) -> bool:
        """Circle-overlap test against everything already placed.

        Circles rather than oriented boxes: the footprint circle circumscribes
        the object, so this is conservative. It rejects a few valid tight
        packings, which is the right trade -- a false reject costs one more
        sample, a false accept costs an exploding scene.
        """
        for position, placed_radius in self._placed:
            distance = float(np.linalg.norm(candidate - position))
            if distance < radius + placed_radius + self.min_separation:
                return False
        return True


def footprint_radius(spec: AssetSpec) -> float:
    """Radius of the circle that circumscribes an object's XY footprint."""
    return float(np.hypot(spec.size_m[0], spec.size_m[1]) / 2.0)


def randomize_scene(
    scene: SceneConfig,
    registry: AssetRegistry,
    randomization: RandomizationConfig | None = None,
) -> SceneConfig:
    """Return a new scene with randomised object placement.

    Pure: the input config is never mutated, so the caller keeps the nominal
    scene alongside the randomised one and can report both.

    With randomisation disabled this returns the scene unchanged, which is the
    property that keeps debugging runs reproducible.
    """
    settings = randomization if randomization is not None else scene.randomization
    settings.validate()
    if not settings.enabled:
        return scene

    rng = np.random.default_rng(settings.seed)

    region = PlacementRegion.from_table(
        scene.table_position,
        scene.table_scale,
        reach_center=scene.robot_base,
        reach_radius=scene.robot_reach_m,
        min_reach_radius=scene.robot_min_reach_m,
    )
    region = region.clipped_to_workspace(
        np.asarray(scene.workspace_min, dtype=np.float64),
        np.asarray(scene.workspace_max, dtype=np.float64),
    )
    placer = ObjectPlacer(
        region,
        min_separation=settings.min_object_separation_m,
        max_attempts=settings.max_placement_attempts,
        rng=rng,
    )

    if settings.randomize_object_set:
        objects = _sample_object_set(rng, registry, settings)
    else:
        objects = list(scene.objects)

    # Objects that keep their authored pose reserve their footprint first, so
    # the sampled ones are placed around them rather than through them.
    if not settings.randomize_positions:
        for obj in objects:
            radius = _radius_for(obj, registry)
            placer.reserve(np.asarray(obj.position, dtype=np.float64), radius)

    randomised: list[SceneObjectConfig] = []
    for obj in objects:
        updates: dict[str, object] = {}

        if settings.randomize_positions:
            spec = _spec_for(obj, registry)
            height = spec.size_m[2] if spec is not None else float(obj.resolved_scale()[2])
            radius = _radius_for(obj, registry)
            if settings.position_jitter_m > 0.0 and not settings.randomize_object_set:
                position = _jitter_within(
                    rng,
                    obj,
                    radius,
                    height,
                    placer,
                    settings.position_jitter_m,
                    region,
                )
            else:
                position = placer.sample(radius, height)
            updates["position"] = (float(position[0]), float(position[1]), float(position[2]))

        if settings.randomize_yaw:
            lo, hi = settings.yaw_range_rad
            yaw = float(rng.uniform(lo, hi))
            half = yaw / 2.0
            updates["quat"] = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))

        randomised.append(replace(obj, **updates) if updates else obj)

    updated: dict[str, object] = {"objects": tuple(randomised)}

    if settings.randomize_lighting:
        lo, hi = settings.lighting_intensity_scale
        updated["dome_light_intensity"] = float(scene.dome_light_intensity * rng.uniform(lo, hi))
        updated["distant_light_intensity"] = float(
            scene.distant_light_intensity * rng.uniform(lo, hi)
        )

    _log.info(
        "Randomised scene (seed=%d): %d objects, positions=%s yaw=%s lighting=%s",
        settings.seed,
        len(randomised),
        settings.randomize_positions,
        settings.randomize_yaw,
        settings.randomize_lighting,
    )
    return replace(scene, **updated)


def _jitter_within(
    rng: np.random.Generator,
    obj: SceneObjectConfig,
    radius: float,
    height: float,
    placer: ObjectPlacer,
    jitter: float,
    region: PlacementRegion,
) -> NDArray[np.float64]:
    """Perturb an authored pose, falling back to a free sample.

    Jitter preserves the *composition* a scene author intended -- the can stays
    near where it was meant to be -- while still varying the exact pose. If the
    jittered spot is taken or off the surface, a free sample is better than
    silently leaving the object where it started, which would make that object
    non-random without saying so.
    """
    origin = np.asarray(obj.position, dtype=np.float64)[:2]
    for _ in range(placer.max_attempts):
        candidate = origin + rng.uniform(-jitter, jitter, size=2)
        inside = (
            region.x_range[0] + radius <= candidate[0] <= region.x_range[1] - radius
            and region.y_range[0] + radius <= candidate[1] <= region.y_range[1] - radius
        )
        if inside and placer.is_free(candidate, radius):
            placer.reserve(candidate, radius)
            return np.array(
                [candidate[0], candidate[1], region.spawn_z(height)],
                dtype=np.float64,
            )
    return placer.sample(radius, height)


def _sample_object_set(
    rng: np.random.Generator,
    registry: AssetRegistry,
    settings: RandomizationConfig,
) -> list[SceneObjectConfig]:
    """Draw a random set of graspable catalogue objects.

    Only graspable ones: a scene of objects the robot cannot pick up is a
    scene where every task fails for a reason that has nothing to do with the
    policy being evaluated.
    """
    candidates = registry.graspable()
    if not candidates:
        raise SimulationError("the asset catalogue contains no graspable objects")

    lo, hi = settings.num_objects
    count = int(rng.integers(lo, hi + 1))
    count = min(count, len(candidates))
    chosen = rng.choice(len(candidates), size=count, replace=False)

    return [
        SceneObjectConfig(
            name=candidates[int(i)].name,
            kind="asset",
            asset=candidates[int(i)].name,
            mass=candidates[int(i)].mass_kg,
            semantic_label=candidates[int(i)].semantic,
        )
        for i in chosen
    ]


def _spec_for(obj: SceneObjectConfig, registry: AssetRegistry) -> AssetSpec | None:
    return registry.objects.get(obj.asset) if obj.kind == "asset" else None


def _radius_for(obj: SceneObjectConfig, registry: AssetRegistry) -> float:
    """Footprint radius, from the catalogue for assets and from ``scale`` for
    primitives."""
    spec = _spec_for(obj, registry)
    if spec is not None:
        return footprint_radius(spec)
    scale = obj.resolved_scale()
    return float(np.hypot(scale[0], scale[1]) / 2.0)

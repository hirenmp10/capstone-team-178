"""Scene validation.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

A scene that *loads* is not a scene that *works*. Every check here corresponds
to a failure that is silent at build time and expensive later:

============================  =========================================================
Check                         What its absence looks like downstream
============================  =========================================================
Collision mesh                The object falls through the table on step 1 and
                              perception reports "nothing there".
Physics material              Default friction (0.5); the object slides out of the
                              fingers and the grasp controller looks broken.
Stable mass                   A 0-mass or 1000 kg body either flies off on contact
                              or refuses to move; reads as a controller tuning bug.
Centre of mass                A COM outside the mesh makes the object topple the
                              instant it is released; reads as a placement bug.
Semantic label                The segmentation annotator returns background, so
                              perception genuinely cannot see it.
Reachable                     Every command about the object fails at IK, far from
                              where the real problem is.
No penetration                Interpenetrating spawns get driven apart violently on
                              the first step; reads as unstable physics tuning.
Stable at rest                An object that drifts while untouched invalidates
                              every before/after measurement in the test suite.
============================  =========================================================

The harness settles the scene first and then measures, because several of these
only manifest after physics has run. Checking a freshly-built stage would pass
every one of them and prove nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.simulation.asset_registry import AssetRegistry
from mfw.utils.logging import get_logger

__all__ = ["CheckResult", "ObjectReport", "SceneReport", "SceneValidator"]

_log = get_logger("simulation.validation")


@dataclass(frozen=True)
class CheckResult:
    """One check against one object."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}" if self.detail else f"[{mark}] {self.name}"


@dataclass
class ObjectReport:
    """Every check for one object."""

    name: str
    prim_path: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


@dataclass
class SceneReport:
    """The whole scene's validation result."""

    objects: list[ObjectReport] = field(default_factory=list)
    substitutions: dict[str, str] = field(default_factory=dict)
    scene_checks: list[CheckResult] = field(default_factory=list)
    """Checks about the scene as a whole rather than one object -- currently
    camera exposure, which is a property of the lighting, not of any object."""

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.objects) and all(c.passed for c in self.scene_checks)

    @property
    def failure_count(self) -> int:
        return sum(len(o.failures) for o in self.objects) + sum(
            1 for c in self.scene_checks if not c.passed
        )

    def summary(self) -> str:
        lines: list[str] = []
        checks = self.objects[0].checks if self.objects else []
        header = [c.name for c in checks]

        width = max((len(o.name) for o in self.objects), default=6) + 2
        lines.append("Scene validation")
        lines.append("=" * 72)
        for report in self.objects:
            marks = " ".join("." if c.passed else "X" for c in report.checks)
            status = "OK" if report.passed else "FAILED"
            lines.append(f"  {report.name:<{width}} {marks}   {status}")
        if header:
            lines.append("")
            lines.append("  legend: " + ", ".join(f"{i + 1}={n}" for i, n in enumerate(header)))

        if self.scene_checks:
            lines.append("")
            lines.append("Scene")
            lines.append("-" * 72)
            for check in self.scene_checks:
                lines.append(f"  {check}")

        failed = [o for o in self.objects if not o.passed]
        if failed or any(not c.passed for c in self.scene_checks):
            lines.append("")
            lines.append("Failures")
            lines.append("-" * 72)
            for check in self.scene_checks:
                if not check.passed:
                    lines.append(f"  scene: {check.name} -- {check.detail}")
            for report in failed:
                for check in report.failures:
                    lines.append(f"  {report.name}: {check.name} -- {check.detail}")

        if self.substitutions:
            lines.append("")
            lines.append("Asset substitutions (no branded assets were fabricated)")
            lines.append("-" * 72)
            for name, note in self.substitutions.items():
                lines.append(f"  {name}: {note}")

        lines.append("")
        lines.append(
            f"{len(self.objects) - len(failed)}/{len(self.objects)} objects passed; "
            f"{self.failure_count} failed check(s)"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "objects": [
                {
                    "name": o.name,
                    "prim_path": o.prim_path,
                    "passed": o.passed,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "detail": c.detail} for c in o.checks
                    ],
                }
                for o in self.objects
            ],
            "substitutions": self.substitutions,
        }


class SceneValidator:
    """Runs the object checklist against a built, settled scene."""

    def __init__(
        self,
        sim: Any,
        scene_builder: Any,
        registry: AssetRegistry | None = None,
        workspace_min: NDArray[np.float64] | None = None,
        workspace_max: NDArray[np.float64] | None = None,
        cameras: dict[str, Any] | None = None,
        approach_offset: float = 0.10,
    ) -> None:
        self._sim = sim
        self._builder = scene_builder
        self._registry = registry
        self._cameras = cameras or {}
        self._approach_offset = float(approach_offset)
        config = scene_builder.config
        self.workspace_min = np.asarray(
            workspace_min if workspace_min is not None else config.workspace_min, dtype=np.float64
        )
        self.workspace_max = np.asarray(
            workspace_max if workspace_max is not None else config.workspace_max, dtype=np.float64
        )

    def validate(
        self,
        settle_steps: int = 120,
        observe_steps: int = 120,
        drift_tolerance_m: float = 0.005,
        mass_tolerance: float = 0.05,
    ) -> SceneReport:
        """Settle the scene, then check every object.

        ``drift_tolerance_m`` is deliberately tight. A resting object should not
        move at all; 5 mm over ``observe_steps`` allows for solver noise and
        nothing more. A looser bound would pass an object that is slowly sliding
        off the table.
        """
        from pxr import UsdPhysics  # noqa: PLC0415

        stage = self._sim.world.stage
        paths = self._builder.object_prim_paths

        # Settle first: several of these failures only exist after physics runs.
        self._sim.step(settle_steps)
        settled = {name: self._position(path) for name, path in paths.items()}

        self._sim.step(observe_steps)
        after = {name: self._position(path) for name, path in paths.items()}

        report = SceneReport()
        if self._registry is not None:
            report.substitutions = self._registry.substitutions()

        table_top = self._table_top()

        for name, path in sorted(paths.items()):
            entry = ObjectReport(name=name, prim_path=path)
            spec = self._spec_for(name)

            entry.checks.append(self._check_collider(stage, path))
            entry.checks.append(self._check_material(stage, path, UsdPhysics))
            entry.checks.append(self._check_mass(stage, path, spec, mass_tolerance))
            entry.checks.append(self._check_com(stage, path, spec))
            entry.checks.append(self._check_semantics(stage, path))
            entry.checks.append(self._check_dimensions(stage, path, spec))
            entry.checks.append(self._check_reachable(settled[name]))
            entry.checks.append(self._check_penetration(stage, path, table_top))
            entry.checks.append(
                self._check_stability(settled[name], after[name], drift_tolerance_m)
            )
            report.objects.append(entry)

        for name in sorted(self._cameras):
            report.scene_checks.append(self._check_exposure(name, self._cameras[name]))

        _log.info(
            "Scene validation: %d/%d objects passed",
            sum(1 for o in report.objects if o.passed),
            len(report.objects),
        )
        return report

    # -- individual checks ----------------------------------------------

    def _check_collider(self, stage: Any, path: str) -> CheckResult:
        from mfw.physics.collision import count_colliders  # noqa: PLC0415

        count = count_colliders(stage, path)
        return CheckResult(
            "collision_mesh",
            count > 0,
            f"{count} collider prim(s)"
            if count > 0
            else "no CollisionAPI anywhere under the prim; this body is not solid",
        )

    def _check_material(self, stage: Any, path: str, usd_physics: Any) -> CheckResult:
        from pxr import UsdShade  # noqa: PLC0415

        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return CheckResult("physics_material", False, "prim does not exist")

        binding = UsdShade.MaterialBindingAPI(prim)
        bound = binding.GetDirectBinding("physics").GetMaterial()
        if not bound or not bound.GetPrim().IsValid():
            return CheckResult(
                "physics_material",
                False,
                "no physics material bound; PhysX will use its 0.5 default friction "
                "and the object will slide out of the gripper",
            )

        material = usd_physics.MaterialAPI(bound.GetPrim())
        static = material.GetStaticFrictionAttr().Get()
        return CheckResult("physics_material", True, f"static_friction={static}")

    def _check_mass(
        self, stage: Any, path: str, spec: Any, tolerance: float
    ) -> CheckResult:
        from pxr import UsdPhysics  # noqa: PLC0415

        prim = stage.GetPrimAtPath(path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()

        if mass is None or mass <= 0.0:
            return CheckResult(
                "mass", False, f"mass is {mass}; a zero-mass dynamic body is not simulable"
            )

        if spec is not None:
            expected = spec.mass_kg
            error = abs(mass - expected) / expected
            if error > tolerance:
                return CheckResult(
                    "mass",
                    False,
                    f"{mass:.3f} kg on the stage vs {expected:.3f} kg in the catalogue "
                    f"({error * 100:.1f}% off)",
                )
            return CheckResult("mass", True, f"{mass:.3f} kg (catalogue {expected:.3f})")

        return CheckResult("mass", True, f"{mass:.3f} kg")

    def _check_com(self, stage: Any, path: str, spec: Any) -> CheckResult:
        """The centre of mass must lie inside the object's own extent.

        A COM outside the mesh is physically meaningless and makes the object
        topple the moment it is released -- which looks like a bad placement,
        not a bad inertial property.
        """
        from pxr import UsdPhysics  # noqa: PLC0415

        prim = stage.GetPrimAtPath(path)
        mass_api = UsdPhysics.MassAPI(prim)
        com = mass_api.GetCenterOfMassAttr().Get()

        if com is None:
            # Unauthored means PhysX derives it from the collision geometry,
            # which is the correct default for a scanned mesh.
            return CheckResult("center_of_mass", True, "derived from geometry")

        com_array = np.array([com[0], com[1], com[2]], dtype=np.float64)

        if np.all(np.isneginf(com_array)):
            # (-inf, -inf, -inf) is UsdPhysics's documented sentinel for "compute
            # the centre of mass from the collision geometry", not a corrupt
            # value. Reading it as garbage failed every asset that did not ship
            # pre-authored physics -- which is 17 of the 21 YCB objects, and the
            # ones for which a geometry-derived COM is exactly what we want.
            return CheckResult("center_of_mass", True, "auto (derived from geometry)")

        if not np.all(np.isfinite(com_array)):
            return CheckResult("center_of_mass", False, f"non-finite COM {com_array.tolist()}")

        if spec is not None:
            half = np.asarray(spec.size_m, dtype=np.float64) / 2.0
            if np.any(np.abs(com_array) > half * 1.5):
                return CheckResult(
                    "center_of_mass",
                    False,
                    f"COM {np.round(com_array, 4).tolist()} lies outside the object's "
                    f"own extent {np.round(half * 2, 3).tolist()}",
                )

        return CheckResult("center_of_mass", True, f"{np.round(com_array, 4).tolist()}")

    def _check_semantics(self, stage: Any, path: str) -> CheckResult:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return CheckResult("semantic_label", False, "prim does not exist")

        labels = _semantic_labels(prim)
        if not labels:
            return CheckResult(
                "semantic_label",
                False,
                "no semantic label; the segmentation annotator will report this as "
                "background and perception cannot see it",
            )
        return CheckResult("semantic_label", True, ", ".join(sorted(labels)))

    def _check_exposure(
        self,
        name: str,
        camera: Any,
        min_mean: float = 40.0,
        max_mean: float = 200.0,
        max_saturated_fraction: float = 0.05,
    ) -> CheckResult:
        """The cameras must produce a usable image, not a black one.

        Every other check in this harness is geometric or physical, and none of
        them can see a lighting failure. Physics, mass, semantics and even
        *detection* are unaffected by darkness -- instance segmentation is
        rendered from prim identity, not from photons, so objects segment
        perfectly in an image a human would call black.

        What dies is everything downstream that reads pixel values. Colour
        naming collapses to "black", so "the green box" stops resolving; and a
        VLA handed near-black frames is far outside its training distribution
        while reporting no error at all.

        Measured in this scene before the workspace light was added: mean
        luminance 24/255 on both cameras, with the whole object set rendering as
        silhouettes. Physics validation passed 9/9 at the time.
        """
        try:
            frame = camera.capture()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            return CheckResult(f"exposure[{name}]", False, f"capture failed: {exc}")

        if frame.rgb is None:
            return CheckResult(f"exposure[{name}]", False, "camera produced no RGB")

        luminance = np.asarray(frame.rgb, dtype=np.float32).mean(axis=2)
        mean = float(luminance.mean())
        p95 = float(np.percentile(luminance, 95))
        saturated = float((luminance > 250.0).mean())

        detail = f"mean={mean:.0f}/255 p95={p95:.0f} saturated={saturated * 100:.1f}%"

        if mean < min_mean:
            return CheckResult(
                f"exposure[{name}]",
                False,
                f"{detail} -- too dark. Colour naming will degrade to 'black' and any "
                f"pixel-consuming policy is out of distribution. Indoors, a dome light "
                f"cannot reach through the ceiling: set scene.workspace_light_intensity.",
            )
        if mean > max_mean:
            return CheckResult(
                f"exposure[{name}]", False, f"{detail} -- washed out; geometry is lost to glare"
            )
        if saturated > max_saturated_fraction:
            return CheckResult(
                f"exposure[{name}]",
                False,
                f"{detail} -- {saturated * 100:.1f}% of pixels are clipped",
            )
        return CheckResult(f"exposure[{name}]", True, detail)

    def _check_dimensions(
        self, stage: Any, prim_path: str, spec: Any, tolerance: float = 0.25
    ) -> CheckResult:
        """The object's rendered size must match the catalogue.

        This is the check that every other one misses. An asset spawned at the
        wrong scale still has a collider, still has the correct mass, still
        rests on the table and still holds still -- so collision, material,
        mass, COM, penetration and stability all pass. It is simply the wrong
        size, and nothing that inspects physics can tell.

        Measured here: the scene config's ``scale`` field defaults to 0.05,
        which is a sensible edge length for a primitive cuboid and catastrophic
        for a mesh authored at true scale. Every YCB object spawned at 5% size.
        A 66 mm soup can became 3.3 mm and covered *two pixels* of the exterior
        camera -- far below the detector's 60-point minimum, so perception
        reported an empty table in a scene that passed every physical check.

        The tolerance is loose (25%) because the catalogue records an axis-
        aligned bounding extent while the object may rest rotated. It is not
        trying to catch a few millimetres; it is trying to catch a factor of
        twenty.
        """
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        if spec is None:
            return CheckResult("dimensions", True, "no catalogue entry to compare against")

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return CheckResult("dimensions", False, "prim does not exist")

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bound = cache.ComputeWorldBound(prim)

        # GetRange() is the extent in the object's *own* frame; GetMatrix() is
        # the transform to world. Using ComputeAlignedRange() instead would
        # measure the world-axis-aligned box, which inflates as soon as the
        # object rests at an angle -- a 159 mm bowl tipped 45 degrees reports a
        # 225 mm diagonal and fails a size check it should pass. Scale still
        # applies, so a wrongly-scaled asset is still caught.
        box = bound.GetRange()
        if box.IsEmpty():
            return CheckResult("dimensions", False, "empty bounding box; no geometry")

        scale = np.array(
            [np.linalg.norm([bound.GetMatrix()[r][c] for r in range(3)]) for c in range(3)],
            dtype=np.float64,
        )
        local = np.array([box.GetMax()[i] - box.GetMin()[i] for i in range(3)], dtype=np.float64)
        measured = local * scale
        expected = np.asarray(spec.size_m, dtype=np.float64)

        # Compare sorted extents: a rotated object permutes its axes, and the
        # question here is "is this the right size", not "is it facing the
        # right way".
        measured_sorted = np.sort(measured)
        expected_sorted = np.sort(expected)
        ratio = measured_sorted / expected_sorted
        worst = float(max(abs(ratio.max() - 1.0), abs(ratio.min() - 1.0)))

        detail = (
            f"{np.round(measured_sorted * 1000, 1).tolist()} mm vs catalogue "
            f"{np.round(expected_sorted * 1000, 1).tolist()} mm"
        )
        if worst > tolerance:
            return CheckResult(
                "dimensions",
                False,
                f"{detail} -- off by up to {worst * 100:.0f}%. Check the scene entry's "
                f"'scale': a mesh authored at true scale must spawn at 1.0.",
            )
        return CheckResult("dimensions", True, detail)

    def _check_reachable(self, position: NDArray[np.float64] | None) -> CheckResult:
        """Can the arm actually get above this object?

        Tests a **sphere** around the arm's base, not the workspace box. The box
        is a clamp for policy outputs; it is not a reach envelope, and using it
        as one is how nine objects passed this check while five of them were
        physically out of range. The box
        [0.12,-0.55,0.0]..[0.85,0.55,0.85] has a corner 1.27 m from the base --
        half a metre past anything a Franka can touch.

        Measured against the pre-grasp standoff rather than the object, since
        the arm must reach *above* the object before it can descend, and that
        point is always the harder of the two.
        """
        if position is None:
            return CheckResult("reachable", False, "no pose")

        inside = np.all(position >= self.workspace_min) and np.all(position <= self.workspace_max)
        if not inside:
            return CheckResult(
                "reachable",
                False,
                f"{np.round(position, 3).tolist()} is outside the workspace "
                f"{np.round(self.workspace_min, 2).tolist()}..."
                f"{np.round(self.workspace_max, 2).tolist()}",
            )

        config = self._builder.config
        reach = float(config.robot_reach_m)
        if reach > 0.0:
            base = np.asarray(config.robot_base, dtype=np.float64)
            standoff = position.copy()
            standoff[2] += self._approach_offset
            distance = float(np.linalg.norm(standoff - base))
            if distance > reach:
                return CheckResult(
                    "reachable",
                    False,
                    f"pre-grasp standoff is {distance:.3f} m from the arm base, beyond "
                    f"its {reach:.2f} m reach -- the planner will report 'no path' for "
                    f"what is really a placement error",
                )
            return CheckResult(
                "reachable", True, f"standoff {distance:.3f} m of {reach:.2f} m reach"
            )

        return CheckResult("reachable", True, f"{np.round(position, 3).tolist()}")

    def _check_penetration(
        self,
        stage: Any,
        prim_path: str,
        table_top: float,
        tolerance: float = 0.005,
    ) -> CheckResult:
        """The object must rest *on* the surface, not sunk into it.

        Measured from the mesh's **lowest actual vertex**, not from its bounding
        box. The distinction is not pedantic: a bounding box only touches the
        geometry at a few points, so for anything that is not box-shaped the
        AABB's bottom corner is empty space.

        Measured here on the YCB banana. It spawns level, then rolls about ten
        degrees about its long axis to find a stable resting angle -- correct
        behaviour for a curved object, and it then holds that pose exactly. But
        a curved 197 x 74 x 39 mm shape tilted ten degrees has an AABB whose
        bottom corner hangs 12 mm below where the banana actually touches the
        table. Reading that as penetration failed a perfectly good object, and
        would have kept failing it however the collider was configured -- two
        attempts at fixing the collider made the number worse, because the
        collider was never the problem.

        Vertices are exact and settle the question directly.
        """
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return CheckResult("no_penetration", False, "prim does not exist")

        bottom = _lowest_vertex_z(prim)
        source = "lowest vertex"
        if bottom is None:
            # Analytic gprims (Cube, Cylinder) carry no point data. Their AABB
            # is a faithful bound for the axis-aligned case, so it is a
            # reasonable fallback rather than a silent skip.
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                return CheckResult("no_penetration", False, "no geometry to measure")
            bottom = float(box.GetMin()[2])
            source = "bounding box"

        penetration = table_top - bottom

        if penetration > tolerance:
            return CheckResult(
                "no_penetration",
                False,
                f"lowest point is z={bottom:.4f} ({source}), {penetration * 1000:.1f} mm "
                f"below the surface at {table_top:.3f} -- the object has sunk into or "
                f"fallen through it",
            )
        return CheckResult(
            "no_penetration",
            True,
            f"rests {(bottom - table_top) * 1000:+.1f} mm relative to the surface",
        )

    def _check_stability(
        self,
        before: NDArray[np.float64] | None,
        after: NDArray[np.float64] | None,
        tolerance: float,
    ) -> CheckResult:
        if before is None or after is None:
            return CheckResult("stable_at_rest", False, "no pose")

        drift = float(np.linalg.norm(after - before))
        if drift > tolerance:
            return CheckResult(
                "stable_at_rest",
                False,
                f"drifted {drift * 1000:.1f} mm while untouched (tolerance "
                f"{tolerance * 1000:.0f} mm)",
            )
        return CheckResult("stable_at_rest", True, f"drift {drift * 1000:.2f} mm")

    # -- helpers ---------------------------------------------------------

    def _position(self, prim_path: str) -> NDArray[np.float64] | None:
        """Ground-truth world position.

        Legitimate here and nowhere else: this module's whole job is to compare
        the simulator's truth against what the scene claimed. Perception, grasp
        and skill code must never call anything like it.
        """
        from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

        try:
            position, _ = get_world_pose(prim_path)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            _log.warning("Cannot read pose of %s: %s", prim_path, exc)
            return None
        return np.asarray(position, dtype=np.float64)

    def _table_top(self) -> float:
        config = self._builder.config
        if not config.add_table:
            return 0.0
        return float(config.table_position[2] + config.table_scale[2] / 2.0)

    def _spec_for(self, object_name: str) -> Any:
        """Catalogue entry behind a spawned object, if it came from one."""
        if self._registry is None:
            return None
        for obj in self._builder.config.objects:
            if obj.name == object_name and obj.kind == "asset":
                return self._registry.objects.get(obj.asset)
        return None


def _lowest_vertex_z(root: Any) -> float | None:
    """World-space Z of the lowest mesh vertex under ``root``.

    Returns ``None`` when nothing beneath ``root`` carries point data, which is
    the case for analytic gprims such as ``Cube`` and ``Cylinder``.
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    lowest: float | None = None

    stack = [root]
    while stack:
        prim = stack.pop()
        stack.extend(prim.GetAllChildren())
        if not prim.IsA(UsdGeom.Mesh):
            continue

        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue

        matrix = xform_cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
        # USD is row-vector: world = local * M.
        transform = np.asarray(matrix, dtype=np.float64)
        world_z = vertices @ transform[:3, 2] + transform[3, 2]
        candidate = float(world_z.min())
        lowest = candidate if lowest is None else min(lowest, candidate)

    return lowest


def _semantic_labels(prim: Any) -> set[str]:
    """Read semantic labels back off a prim, across Isaac's two API generations.

    Uses Isaac's own readers rather than reconstructing attribute names.
    Hand-building the name is how this check first went wrong: Isaac 5.x writes
    ``semantics:labels:<instance>`` -- the instance name goes *last* -- and
    guessing ``semantics:<instance>:labels`` reported "no semantic label" for
    every object in a correctly-labelled scene.

    Isaac 5.x replaced ``Semantics.SemanticsAPI`` with ``SemanticsLabelsAPI``,
    so both readers are consulted: ``get_labels`` for the new schema and
    ``get_semantics`` for anything still carrying the old one.
    """
    from isaacsim.core.utils import semantics  # noqa: PLC0415

    labels: set[str] = set()

    try:
        for values in (semantics.get_labels(prim) or {}).values():
            labels.update(str(v) for v in values)
    except Exception as exc:  # noqa: BLE001 - a read failure is a report, not a crash
        _log.debug("get_labels failed on %s: %s", prim.GetPath(), exc)

    try:
        for value in (semantics.get_semantics(prim) or {}).values():
            # The legacy API returns (type_label, semantic_label) pairs.
            if isinstance(value, (tuple, list)) and len(value) == 2:
                labels.add(str(value[1]))
            else:
                labels.add(str(value))
    except Exception as exc:  # noqa: BLE001
        _log.debug("get_semantics failed on %s: %s", prim.GetPath(), exc)

    return labels

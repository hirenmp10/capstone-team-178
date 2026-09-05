"""Asset catalogue, collider selection, placement and config inheritance.

Pure logic -- no Isaac Sim, no GPU. Everything here is decided before anything
is spawned, which is exactly why it can be tested cheaply and should be.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from mfw.config.schema import (
    ConfigError,
    RandomizationConfig,
    SceneConfig,
    SceneObjectConfig,
    load_config,
)
from mfw.core.errors import SimulationError
from mfw.physics.collision import VALID_APPROXIMATIONS, collider_approximation_for
from mfw.simulation.asset_registry import AssetRegistry, AssetSpec
from mfw.simulation.layout import ObjectPlacer, PlacementRegion, randomize_scene


def _spec(name: str, **kwargs) -> AssetSpec:
    defaults = dict(
        usd=f"/Isaac/Props/YCB/Axis_Aligned/{name}.usd",
        category="box",
        semantic=name,
        mass_kg=0.2,
        size_m=(0.05, 0.05, 0.05),
    )
    defaults.update(kwargs)
    return AssetSpec(name=name, **defaults)


class TestAssetRegistry:
    def test_the_shipped_catalogue_loads_and_validates(self):
        registry = AssetRegistry.load()
        assert registry.objects, "the shipped catalogue must not be empty"
        for spec in registry.objects.values():
            spec.validate()
        for item in registry.furniture.values():
            item.validate()

    def test_every_catalogue_usd_path_is_assets_root_relative(self):
        """Paths are concatenated onto the assets root, so a missing leading
        slash silently produces a malformed URL rather than an error."""
        registry = AssetRegistry.load()
        for spec in list(registry.objects.values()) + list(registry.furniture.values()):
            assert spec.usd.startswith("/"), f"{spec.name}: {spec.usd}"

    def test_unknown_asset_raises_rather_than_returning_none(self):
        registry = AssetRegistry.load()
        with pytest.raises(ConfigError, match="unknown object asset"):
            registry.object("no_such_object")

    def test_unknown_keys_are_rejected(self):
        """A misspelled key must fail loudly. Silently ignoring `mass_kgs`
        would leave the object at its default mass while looking configured."""
        with pytest.raises(ConfigError, match="unknown keys"):
            AssetRegistry.from_dict(
                {"objects": [{"name": "x", "usd": "/a.usd", "category": "box",
                              "semantic": "x", "mass_kgs": 1.0, "size_m": [1, 1, 1]}]}
            )

    def test_duplicate_asset_names_are_rejected(self):
        entry = {"name": "dup", "usd": "/a.usd", "category": "box",
                 "semantic": "dup", "mass_kg": 1.0, "size_m": [0.1, 0.1, 0.1]}
        with pytest.raises(ConfigError, match="duplicate object asset"):
            AssetRegistry.from_dict({"objects": [entry, dict(entry)]})

    def test_zero_mass_is_rejected(self):
        with pytest.raises(ConfigError, match="mass_kg must be > 0"):
            _spec("bad", mass_kg=0.0).validate()

    def test_fits_gripper_uses_the_smallest_dimension(self):
        """A 250 mm bottle that is 65 mm across is graspable around its middle.
        Judging by the largest dimension would reject it."""
        bottle = _spec("bottle", size_m=(0.065, 0.098, 0.250))
        assert bottle.fits_gripper(0.08)
        assert not bottle.fits_gripper(0.06)

    def test_wide_objects_are_excluded_from_a_narrow_gripper(self):
        """The YCB master chef can is 102 mm across and genuinely does not fit
        the Franka's 80 mm span, even though it is a manipulation target."""
        registry = AssetRegistry.load()
        graspable = {s.name for s in registry.graspable(0.08)}
        assert "master_chef_can" not in graspable
        assert "master_chef_can" in registry.objects

    def test_containers_are_not_required_to_be_graspable(self):
        """The bowl is a destination, not a target. Requiring destinations to be
        graspable would make 'put it in the bowl' unexpressible."""
        registry = AssetRegistry.load()
        names = {s.name for s in registry.containers()}
        assert "bowl" in names
        assert not registry.object("bowl").graspable

    def test_substitutions_are_recorded_not_implied(self):
        registry = AssetRegistry.load()
        subs = registry.substitutions()
        assert "mustard_bottle" in subs
        assert "water bottle" in subs["mustard_bottle"].lower()


class TestColliderSelection:
    @pytest.mark.parametrize("category", ["can", "bottle", "box", "block"])
    def test_convex_categories_get_a_hull(self, category):
        assert collider_approximation_for(category) == "convexHull"

    @pytest.mark.parametrize("category", ["mug", "bowl", "pitcher", "food", "tool"])
    def test_concave_categories_get_a_decomposition(self, category):
        """A convex hull of a mug is a solid lump: the cavity fills in, so
        nothing can be placed inside and the handle cannot be grasped."""
        assert collider_approximation_for(category) == "convexDecomposition"

    def test_unknown_category_defaults_to_decomposition(self):
        """Guessing a decomposition for a convex mesh costs cook time; guessing
        a hull for a concave one silently loses the cavity."""
        assert collider_approximation_for("something_new") == "convexDecomposition"

    def test_explicit_override_wins(self):
        assert collider_approximation_for("box", "sdf") == "sdf"

    def test_invalid_override_is_rejected(self):
        with pytest.raises(ValueError, match="unknown collision approximation"):
            collider_approximation_for("box", "convexHulls")

    def test_catalogue_overrides_are_valid_approximations(self):
        registry = AssetRegistry.load()
        for spec in registry.objects.values():
            if spec.collision_approximation:
                assert spec.collision_approximation in VALID_APPROXIMATIONS, spec.name


class TestPlacementRegion:
    def test_surface_z_is_the_top_of_the_table(self):
        region = PlacementRegion.from_table((0.5, 0.0, 0.2), (0.8, 1.2, 0.4))
        assert region.surface_z == pytest.approx(0.4)

    def test_margin_insets_the_usable_area(self):
        """An object centred on the rim half-overhangs and topples during the
        settle phase."""
        region = PlacementRegion.from_table((0.5, 0.0, 0.2), (0.8, 1.2, 0.4), margin=0.06)
        assert region.x_range == pytest.approx((0.16, 0.84))

    def test_a_table_smaller_than_the_margin_is_rejected(self):
        with pytest.raises(SimulationError, match="too small"):
            PlacementRegion.from_table((0.5, 0.0, 0.2), (0.05, 0.05, 0.4), margin=0.06)

    def test_clipping_to_workspace_intersects(self):
        region = PlacementRegion.from_table((0.5, 0.0, 0.2), (0.8, 1.2, 0.4))
        clipped = region.clipped_to_workspace(
            np.array([0.3, -0.2, 0.0]), np.array([0.7, 0.2, 0.9])
        )
        assert clipped.x_range == pytest.approx((0.3, 0.7))
        assert clipped.y_range == pytest.approx((-0.2, 0.2))

    def test_disjoint_table_and_workspace_raise(self):
        """An unreachable object makes every command about it fail at IK, far
        from where the real problem is."""
        region = PlacementRegion.from_table((0.5, 0.0, 0.2), (0.8, 1.2, 0.4))
        with pytest.raises(SimulationError, match="do not overlap"):
            region.clipped_to_workspace(np.array([2.0, -0.2, 0.0]), np.array([3.0, 0.2, 0.9]))


class TestObjectPlacer:
    def _region(self) -> PlacementRegion:
        return PlacementRegion((0.2, 0.8), (-0.4, 0.4), 0.4)

    def test_sampled_objects_never_overlap(self):
        placer = ObjectPlacer(self._region(), min_separation=0.03,
                              rng=np.random.default_rng(0))
        placed = [placer.sample(0.04, 0.1) for _ in range(8)]
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                gap = float(np.linalg.norm(a[:2] - b[:2]))
                assert gap >= 0.04 + 0.04 + 0.03 - 1e-9, f"{gap} too close"

    def test_spawn_height_centres_the_object_on_the_surface(self):
        """Both YCB meshes and Isaac primitives are centred on their origin, so
        an object rests at surface + half its height. Spawning at the surface
        buries the lower half and PhysX ejects it on the first step."""
        from mfw.simulation.layout import SPAWN_CLEARANCE_M

        placer = ObjectPlacer(self._region(), rng=np.random.default_rng(0))
        position = placer.sample(0.03, 0.12)
        assert position[2] == pytest.approx(0.4 + 0.06 + SPAWN_CLEARANCE_M)

    def test_an_overcrowded_region_raises_instead_of_overlapping(self):
        """A partial layout is a silently different experiment."""
        placer = ObjectPlacer(PlacementRegion((0.0, 0.2), (0.0, 0.2), 0.0),
                              min_separation=0.05, max_attempts=50,
                              rng=np.random.default_rng(0))
        with pytest.raises(SimulationError, match="could not place"):
            for _ in range(50):
                placer.sample(0.05, 0.05)

    def test_an_object_too_large_for_the_region_raises(self):
        placer = ObjectPlacer(self._region(), rng=np.random.default_rng(0))
        with pytest.raises(SimulationError, match="does not fit"):
            placer.sample(0.5, 0.1)

    def test_reserved_footprints_are_avoided(self):
        placer = ObjectPlacer(self._region(), min_separation=0.02,
                              rng=np.random.default_rng(1))
        placer.reserve(np.array([0.5, 0.0]), 0.2)
        for _ in range(10):
            position = placer.sample(0.03, 0.05)
            gap = float(np.linalg.norm(position[:2] - np.array([0.5, 0.0])))
            assert gap >= 0.2 + 0.03 + 0.02 - 1e-9


class TestRandomization:
    def _scene(self) -> SceneConfig:
        return SceneConfig(
            table_position=(0.5, 0.0, 0.2),
            table_scale=(0.8, 1.2, 0.4),
            workspace_min=(0.12, -0.55, 0.0),
            workspace_max=(0.85, 0.55, 0.85),
            objects=(
                SceneObjectConfig(name="a", kind="asset", asset="mug", position=(0.4, 0.0, 0.45)),
                SceneObjectConfig(name="b", kind="asset", asset="banana", position=(0.6, 0.1, 0.45)),
            ),
        )

    def test_disabled_randomization_returns_the_scene_unchanged(self):
        """A debugging run must be reproducible; a framework that quietly
        jitters the world makes every regression a fresh investigation."""
        scene = self._scene()
        assert randomize_scene(scene, AssetRegistry.load()) is scene

    def test_the_same_seed_produces_the_same_layout(self):
        """'Randomised' must still mean 'replayable', or a failure found in a
        randomised run cannot be investigated."""
        registry = AssetRegistry.load()
        settings = RandomizationConfig(enabled=True, seed=42)
        first = randomize_scene(self._scene(), registry, settings)
        second = randomize_scene(self._scene(), registry, settings)
        assert [o.position for o in first.objects] == [o.position for o in second.objects]
        assert [o.quat for o in first.objects] == [o.quat for o in second.objects]

    def test_different_seeds_produce_different_layouts(self):
        registry = AssetRegistry.load()
        a = randomize_scene(self._scene(), registry, RandomizationConfig(enabled=True, seed=1))
        b = randomize_scene(self._scene(), registry, RandomizationConfig(enabled=True, seed=2))
        assert [o.position for o in a.objects] != [o.position for o in b.objects]

    def test_randomization_does_not_mutate_the_input(self):
        scene = self._scene()
        original = [o.position for o in scene.objects]
        randomize_scene(scene, AssetRegistry.load(),
                        RandomizationConfig(enabled=True, seed=3))
        assert [o.position for o in scene.objects] == original

    def test_randomized_objects_stay_inside_the_workspace(self):
        scene = self._scene()
        result = randomize_scene(
            scene, AssetRegistry.load(),
            RandomizationConfig(enabled=True, seed=5, position_jitter_m=0.15),
        )
        for obj in result.objects:
            assert scene.workspace_min[0] <= obj.position[0] <= scene.workspace_max[0]
            assert scene.workspace_min[1] <= obj.position[1] <= scene.workspace_max[1]

    def test_yaw_randomization_produces_unit_quaternions(self):
        result = randomize_scene(
            self._scene(), AssetRegistry.load(),
            RandomizationConfig(enabled=True, seed=9, randomize_yaw=True),
        )
        for obj in result.objects:
            assert float(np.linalg.norm(obj.quat)) == pytest.approx(1.0)

    def test_sampled_object_sets_are_graspable(self):
        """A scene of objects the robot cannot pick up makes every task fail
        for reasons unrelated to the policy being measured."""
        registry = AssetRegistry.load()
        result = randomize_scene(
            self._scene(), registry,
            RandomizationConfig(enabled=True, seed=11, randomize_object_set=True,
                                num_objects=(3, 5)),
        )
        assert 3 <= len(result.objects) <= 5
        for obj in result.objects:
            assert registry.object(obj.asset).graspable

    def test_invalid_object_count_range_is_rejected(self):
        with pytest.raises(ConfigError, match="num_objects"):
            RandomizationConfig(num_objects=(5, 2)).validate()


class TestConfigInheritance:
    def test_extends_inherits_untouched_keys(self, tmp_path):
        """Without inheritance, a specialised config either duplicates the base
        (and drifts) or omits it (and silently reverts tuned physics)."""
        base = tmp_path / "base.yaml"
        base.write_text(textwrap.dedent("""
            physics:
              solver_position_iterations: 32
              static_friction: 1.2
            scene:
              add_table: true
        """), encoding="utf-8")
        derived = tmp_path / "derived.yaml"
        derived.write_text(textwrap.dedent("""
            extends: base.yaml
            physics:
              static_friction: 0.9
        """), encoding="utf-8")

        cfg = load_config(derived)
        assert cfg.physics.solver_position_iterations == 32   # inherited
        assert cfg.physics.static_friction == pytest.approx(0.9)  # overridden
        assert cfg.scene.add_table is True

    def test_lists_replace_rather_than_append(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("scene:\n  objects:\n    - name: old\n", encoding="utf-8")
        derived = tmp_path / "derived.yaml"
        derived.write_text(
            "extends: base.yaml\nscene:\n  objects:\n    - name: new\n", encoding="utf-8"
        )
        cfg = load_config(derived)
        assert [o.name for o in cfg.scene.objects] == ["new"]

    def test_circular_inheritance_is_detected(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as d:
            a, b = P(d) / "a.yaml", P(d) / "b.yaml"
            a.write_text("extends: b.yaml\n", encoding="utf-8")
            b.write_text("extends: a.yaml\n", encoding="utf-8")
            with pytest.raises(ConfigError, match="circular config inheritance"):
                load_config(a)

    def test_missing_base_is_reported(self, tmp_path):
        derived = tmp_path / "derived.yaml"
        derived.write_text("extends: nope.yaml\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="config file not found"):
            load_config(derived)

    def test_the_benchmark_scene_loads_and_inherits_tuned_physics(self):
        """The benchmark config restates only scene differences. If inheritance
        broke, physics would silently fall back to dataclass defaults."""
        from mfw.simulation.asset_registry import DEFAULT_ASSETS_PATH

        cfg = load_config(DEFAULT_ASSETS_PATH.parent / "benchmark.yaml")
        assert cfg.physics.solver_position_iterations == 32
        assert cfg.robot.tcp_offset_from_hand == pytest.approx((0.0, 0.0, 0.1034))
        assert cfg.simulation.physics_dt == pytest.approx(1.0 / 120.0, rel=1e-3)

    def test_every_benchmark_asset_exists_in_the_catalogue(self):
        """Catches a typo'd asset name at test time rather than as a scene that
        silently comes up one object short."""
        from mfw.simulation.asset_registry import DEFAULT_ASSETS_PATH

        cfg = load_config(DEFAULT_ASSETS_PATH.parent / "benchmark.yaml")
        registry = AssetRegistry.load()
        for obj in cfg.scene.objects:
            if obj.kind == "asset":
                registry.object(obj.asset)
        for item in cfg.scene.furniture:
            registry.furniture_item(item.asset)
        if cfg.scene.environment:
            registry.environment(cfg.scene.environment)


class TestScaleDefaulting:
    """Regression tests for the bug that made the whole benchmark scene invisible.

    ``scale`` used to default to 0.05 for everything. That is right for a
    primitive cuboid and catastrophic for a mesh authored at true scale: every
    YCB object spawned at 5% size. A 66 mm can became 3.3 mm -- it still had a
    collider, still had the correct mass, still rested on the table, so every
    physics check passed -- while covering two pixels of camera image, far under
    the detector's minimum. The scene was physically valid and completely
    invisible to the robot.
    """

    def test_mesh_kinds_default_to_true_scale(self):
        for kind in ("asset", "usd"):
            obj = SceneObjectConfig(name="x", kind=kind, asset="mug", usd_subpath="/a.usd")
            assert obj.resolved_scale() == (1.0, 1.0, 1.0), kind

    def test_primitive_kinds_keep_a_sensible_size(self):
        for kind in ("cuboid", "cylinder", "sphere"):
            assert SceneObjectConfig(name="x", kind=kind).resolved_scale() == (0.05, 0.05, 0.05)

    def test_an_explicit_scale_always_wins(self):
        obj = SceneObjectConfig(name="x", kind="asset", asset="mug", scale=(2.0, 2.0, 2.0))
        assert obj.resolved_scale() == (2.0, 2.0, 2.0)

    def test_non_positive_scale_is_rejected(self):
        with pytest.raises(ConfigError, match="scale must be positive"):
            SceneObjectConfig(name="x", scale=(0.05, 0.0, 0.05)).validate()

    def test_the_benchmark_scene_does_not_rescale_its_assets(self):
        """If a benchmark object ever acquires an explicit scale, it should be
        deliberate -- so this pins the current state to force the question."""
        from mfw.simulation.asset_registry import DEFAULT_ASSETS_PATH

        cfg = load_config(DEFAULT_ASSETS_PATH.parent / "benchmark.yaml")
        for obj in cfg.scene.objects:
            assert obj.resolved_scale() == (1.0, 1.0, 1.0), obj.name


class TestUprightOrientation:
    """The YCB set is authored Y-up; Isaac's world is Z-up.

    Without the correction, cans lie on their sides and roll, bottles are prone,
    and the bowl balances on its rim -- which reads as unstable physics rather
    than a wrong orientation.
    """

    def test_catalogue_assets_carry_the_y_up_correction(self):
        registry = AssetRegistry.load()
        expected = (0.70710678, -0.70710678, 0.0, 0.0)
        for spec in registry.objects.values():
            assert spec.upright_quat == pytest.approx(expected), spec.name

    def test_the_correction_is_a_unit_quaternion(self):
        registry = AssetRegistry.load()
        for spec in registry.objects.values():
            assert float(np.linalg.norm(spec.upright_quat)) == pytest.approx(1.0)

    def test_the_correction_maps_the_asset_up_axis_to_world_up(self):
        """The YCB assets are authored with up along **-Y**, and -90 degrees
        about X sends -Y to +Z.

        The sign is not guessable from bounding boxes, which are magnitudes:
        +90 and -90 give identical extents, and a can looks equally plausible
        either way. It was settled by dropping a brick over the bowl -- at -90
        the brick lands inside it (33.8 mm above the table), at +90 it rests on
        top of an inverted bowl (82.4 mm). Getting this backwards would leave
        every open container upside down while every dimension still checked out.
        """
        from mfw.utils import transforms as tf

        registry = AssetRegistry.load()
        rotation = tf.quat_to_matrix(np.asarray(registry.object("bowl").upright_quat))
        assert (rotation @ np.array([0.0, -1.0, 0.0])) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_composition_applies_the_correction_first(self):
        """A scene author writes the yaw they want; the upright correction is
        applied underneath it, not instead of it."""
        from mfw.simulation.scene import _compose_quat

        identity = (1.0, 0.0, 0.0, 0.0)
        upright = (0.70710678, -0.70710678, 0.0, 0.0)
        assert _compose_quat(identity, upright) == pytest.approx(upright)

        composed = _compose_quat(upright, identity)
        assert composed == pytest.approx(upright)

    def test_composition_is_a_unit_quaternion(self):
        from mfw.simulation.scene import _compose_quat

        yaw = (np.cos(0.4), 0.0, 0.0, np.sin(0.4))
        upright = (0.70710678, -0.70710678, 0.0, 0.0)
        assert float(np.linalg.norm(_compose_quat(yaw, upright))) == pytest.approx(1.0)


class TestCatalogueDimensions:
    """``size_m`` is measured from the meshes, in the upright frame."""

    def test_the_upright_height_is_the_z_extent(self):
        """Sanity anchors from scripts/measure_assets.py. If these drift, either
        the catalogue is stale or the shipped assets changed -- both worth
        knowing, because spawn height and grasp width read these numbers."""
        registry = AssetRegistry.load()
        # A soup can is ~102 mm tall and ~68 mm across.
        can = registry.object("tomato_soup_can")
        assert can.size_m[2] == pytest.approx(0.102, abs=0.003)
        assert can.size_m[0] == pytest.approx(can.size_m[1], abs=0.003)
        # A bowl is wide and shallow: height must be its smallest dimension.
        bowl = registry.object("bowl")
        assert bowl.size_m[2] == min(bowl.size_m)
        assert bowl.size_m[2] == pytest.approx(0.055, abs=0.005)
        # A marker is a thin stick, tall in Z.
        marker = registry.object("large_marker")
        assert marker.size_m[2] == max(marker.size_m)

    def test_no_asset_records_an_implausible_dimension(self):
        registry = AssetRegistry.load()
        for spec in registry.objects.values():
            assert 0.005 < min(spec.size_m) < 0.5, spec.name
            assert 0.01 < max(spec.size_m) < 0.6, spec.name


class TestSceneObjectConfig:
    def test_asset_kind_requires_an_asset_name(self):
        with pytest.raises(ConfigError, match="requires an 'asset' key"):
            SceneObjectConfig(name="x", kind="asset").validate()

    def test_usd_kind_requires_a_path(self):
        with pytest.raises(ConfigError, match="requires usd_subpath"):
            SceneObjectConfig(name="x", kind="usd").validate()

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown kind"):
            SceneObjectConfig(name="x", kind="mesh").validate()

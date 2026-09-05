"""Phase 2 gate: vision, object detection, scene graph.

The central question these answer: does perception, using only sensors, recover
where the objects actually are? Every assertion compares a perceived pose
against ground truth that perception itself cannot see.

Run:
    python.bat scripts/run_isaac_tests.py tests/test_phase2_isaac.py -m isaac
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.core.types import Frame, SceneGraph
from tests.conftest import ground_truth_pose

pytestmark = [pytest.mark.isaac, pytest.mark.phase2]


class TestObservation:
    def test_observe_returns_a_scene_graph(self, restore_scene, vision):
        scene = vision.observe()
        assert isinstance(scene, SceneGraph)
        assert scene.step_index == restore_scene.sim.step_index

    def test_finds_all_spawned_objects(self, restore_scene, vision):
        """Perception must recover every object without being told they exist."""
        vision.observe()
        scene = vision.observe()  # second pass confirms tracks

        expected = len(restore_scene.config.scene.objects)
        assert len(scene.objects) == expected, (
            f"perceived {len(scene.objects)} objects, expected {expected}: "
            f"{[(o.label, np.round(o.pose.position, 3).tolist()) for o in scene.objects.values()]}"
        )

    def test_perceived_positions_match_ground_truth(self, restore_scene, vision):
        """The core claim of the whole perception stack.

        The tolerance scales with object size rather than being a flat number.
        A depth camera measures only the surfaces it can see, so the centroid of
        a single-view cloud is biased toward the camera by roughly half the
        object's unseen depth. That bias is inherent to the sensor, not a defect
        -- but it means a 0.12 m box legitimately reports a larger absolute
        error than a 0.05 m cube.
        """
        vision.observe()
        scene = vision.observe()

        for name in restore_scene.scene_builder.object_prim_paths:
            truth, _ = ground_truth_pose(restore_scene, name)
            errors = {
                obj.label: float(np.linalg.norm(obj.pose.position - truth))
                for obj in scene.objects.values()
            }
            best_label = min(errors, key=errors.get)
            best = next(o for o in scene.objects.values() if o.label == best_label)

            tolerance = 0.35 * float(np.max(best.bbox.extents)) + 0.02
            assert errors[best_label] < tolerance, (
                f"{name}: nearest perceived object is {errors[best_label] * 1000:.0f} mm away "
                f"(tolerance {tolerance * 1000:.0f} mm for a "
                f"{np.max(best.bbox.extents) * 1000:.0f} mm object)"
            )

    def test_graspable_object_extents_are_accurate(self, restore_scene, vision):
        """Grasp width is taken straight from these extents.

        Only checks objects small enough to grasp -- those are the ones whose
        dimensions actually gate a pick.
        """
        vision.observe()
        scene = vision.observe()

        truth_extents = {
            (o.semantic_label or o.name).lower(): np.asarray(o.scale, dtype=float)
            for o in restore_scene.config.scene.objects
        }

        checked = 0
        for obj in scene.objects.values():
            expected = truth_extents.get(obj.label.lower())
            if expected is None or float(np.min(expected)) > 0.08:
                continue
            # Compare the graspable dimension: the smallest side, which is what
            # the gripper has to span.
            measured_min = float(np.min(obj.bbox.extents))
            expected_min = float(np.min(expected))
            assert abs(measured_min - expected_min) < 0.025, (
                f"{obj.label}: narrowest side measured {measured_min * 1000:.0f} mm, "
                f"true {expected_min * 1000:.0f} mm"
            )
            checked += 1
        assert checked >= 1, "no graspable-sized objects were checked"

    def test_perceived_extents_are_plausible(self, restore_scene, vision):
        """Extents drive grasp width; wildly wrong boxes yield ungraspable widths.

        Tolerances are loose because a single viewpoint sees only the visible
        faces, so the fitted box is legitimately smaller than the true object.
        """
        vision.observe()
        scene = vision.observe()
        for obj in scene.objects.values():
            assert np.all(obj.bbox.extents > 0.005), f"{obj.label} box is degenerate"
            assert np.all(obj.bbox.extents < 0.5), f"{obj.label} box is implausibly large"

    def test_objects_sit_above_the_table(self, restore_scene, vision):
        """Catches a mirrored point cloud, which puts objects below the surface."""
        vision.observe()
        scene = vision.observe()
        table_top = (
            restore_scene.config.scene.table_position[2]
            + restore_scene.config.scene.table_scale[2] / 2.0
        )
        for obj in scene.objects.values():
            assert obj.pose.position[2] > table_top - 0.05, (
                f"{obj.label} perceived below the table top"
            )

    def test_poses_are_in_world_frame(self, restore_scene, vision):
        scene = vision.observe()
        for obj in scene.objects.values():
            assert obj.pose.frame is Frame.WORLD

    def test_support_surface_is_not_reported_as_an_object(self, restore_scene, vision):
        """The table is furniture, not something to pick."""
        vision.observe()
        scene = vision.observe()
        labels = {o.label.lower() for o in scene.objects.values()}
        assert "table" not in labels
        assert not any("ground" in label for label in labels)

    def test_semantic_labels_are_classes_not_prim_paths(self, restore_scene, vision):
        """Language grounding matches on classes; a path would never match."""
        vision.observe()
        scene = vision.observe()
        for obj in scene.objects.values():
            assert not obj.label.startswith("/"), f"label is a prim path: {obj.label!r}"
        expected = {
            (o.semantic_label or o.name).lower() for o in restore_scene.config.scene.objects
        }
        assert {o.label.lower() for o in scene.objects.values()} & expected


class TestTracking:
    def test_track_ids_are_stable_across_observations(self, restore_scene, vision):
        """"Place it" three commands later depends on this."""
        vision.observe()
        first = set(vision.observe().objects)
        for _ in range(3):
            restore_scene.sim.step(10, render=True)
            latest = set(vision.observe().objects)
        assert first == latest, f"track ids changed: {first} -> {latest}"

    def test_geometry_comes_from_a_single_view_not_a_blend(self, restore_scene, vision):
        """Guards the fusion decision.

        Averaging or concatenating two partial-surface views produces geometry
        matching neither: measured here it doubled a 0.05 m cube to 0.095 m. Each
        hypothesis must therefore name the one camera its box came from, while
        still recording that a second camera corroborated it.
        """
        vision.observe()
        scene = vision.observe()

        corroborated = [
            o for o in scene.objects.values() if len(o.attributes.get("cameras", [])) > 1
        ]
        assert corroborated, "no object was seen by both cameras; test is inconclusive"

        for obj in corroborated:
            primary = obj.attributes.get("primary_camera")
            assert primary in obj.attributes["cameras"], "primary camera not among the views"
            assert obj.confidence > 0.5, f"{obj.label} corroborated but low confidence"
            assert "view_agreement_m" in obj.attributes

    def test_tracks_a_moved_object_to_its_new_position(self, restore_scene, vision):
        """Objects may move. Perception must report where it is *now*.

        Uses a physics impulse rather than a pose write, so this also confirms
        the pipeline follows genuinely dynamic objects.
        """
        from isaacsim.core.utils.xforms import get_world_pose

        vision.observe()
        scene = vision.observe()

        name = "red_block"
        path = restore_scene.scene_builder.object_prim_paths[name]
        truth_before, _ = ground_truth_pose(restore_scene, name)
        tracked = min(
            scene.objects.values(),
            key=lambda o: float(np.linalg.norm(o.pose.position - truth_before)),
        )
        track_id = tracked.track_id

        from isaacsim.core.prims import SingleRigidPrim

        prim = SingleRigidPrim(prim_path=path, name=f"{name}_probe")
        prim.initialize()
        # Sized against the configured friction. With mu ~1.1 the block
        # decelerates at ~11 m/s^2, so 0.45 m/s slides only ~10 mm -- the
        # physics is right, the nudge was just too small to be a test.
        prim.set_linear_velocity(np.array([0.0, 2.0, 0.0]))
        restore_scene.sim.step(60, render=True)

        truth_after = np.asarray(get_world_pose(path)[0])
        moved = float(np.linalg.norm(truth_after - truth_before))
        assert moved > 0.02, f"object did not actually move ({moved * 1000:.0f} mm); inconclusive"

        scene_after = vision.observe()
        assert track_id in scene_after.objects, "identity lost when the object moved"
        error = float(np.linalg.norm(scene_after.objects[track_id].pose.position - truth_after))
        assert error < 0.06, f"tracked pose is {error * 1000:.0f} mm from truth after motion"


class TestSceneGraphOutput:
    def test_relations_are_computed(self, restore_scene, vision):
        vision.observe()
        scene = vision.observe()
        assert scene.relations, "no spatial relations derived"
        for subject, predicate, obj in scene.relations:
            assert subject in scene.objects and obj in scene.objects
            assert isinstance(predicate, str) and predicate

    def test_by_label_lookup_works(self, restore_scene, vision):
        """How the language layer finds "the can" without any hardcoded id."""
        vision.observe()
        scene = vision.observe()
        labels = [o.label for o in scene.objects.values() if o.label]
        assert labels
        matched = scene.by_label(labels[0])
        assert matched and matched[0].label.lower() == labels[0].lower()

    def test_scene_graph_exposes_no_pixels(self, restore_scene, vision):
        """Structural guarantee: the planner cannot reach raw images through it."""
        scene = vision.observe()
        for obj in scene.objects.values():
            for value in obj.attributes.values():
                assert not isinstance(value, np.ndarray) or value.ndim < 2

    def test_freshness_gate_reobserves_stale_graph(self, restore_scene, vision):
        """Acting on a stale pose sends the gripper where the object *was*."""
        first = vision.observe()
        restore_scene.sim.step(2, render=True)
        assert vision.require_fresh_scene() is first, "re-observed while still fresh"

        steps = int(
            (restore_scene.config.perception.max_scene_graph_age_s + 0.2)
            / restore_scene.config.simulation.physics_dt
        )
        restore_scene.sim.step(steps, render=False)
        restore_scene.sim.render_step(2)
        assert vision.require_fresh_scene() is not first, "stale graph was not refreshed"

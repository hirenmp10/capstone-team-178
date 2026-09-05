"""Phase 2 pure-logic tests: geometry, tracking, scene relations.

No Isaac Sim. These cover the maths that grasping depends on, using synthetic
clouds where the correct answer is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.core.types import BoundingBox3D, Frame, ObjectHypothesis, Pose
from mfw.utils import transforms as tf
from mfw.vision.geometry import (
    cluster_by_euclidean_distance,
    fit_oriented_bbox,
    points_above_plane,
    remove_statistical_outliers,
    voxel_downsample,
)
from mfw.vision.scene_graph import compute_relations
from mfw.vision.tracking import ObjectTracker

pytestmark = pytest.mark.phase2


def _box_cloud(center, extents, yaw=0.0, n=2000, seed=0):
    """Points filling an oriented box, for cases where the truth is known."""
    rng = np.random.default_rng(seed)
    local = (rng.random((n, 3)) - 0.5) * np.asarray(extents)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rotation.T + np.asarray(center)


def _hypothesis(track_id, position, extents=(0.05, 0.05, 0.05), label="obj", step=0):
    pose = Pose(np.asarray(position, dtype=float), np.array([1.0, 0, 0, 0]), Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id,
        label=label,
        pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.asarray(extents, dtype=float)),
        confidence=0.9,
        num_points=500,
        last_seen_sim_time=float(step) * 0.01,
        last_seen_step=step,
    )


class TestVoxelDownsample:
    def test_reduces_density_but_keeps_shape(self):
        points = _box_cloud((0.5, 0.0, 0.5), (0.1, 0.1, 0.1), n=5000)
        out, idx = voxel_downsample(points, 0.01)
        assert out.shape[0] < points.shape[0]
        assert np.allclose(out.mean(axis=0), points.mean(axis=0), atol=0.01)
        assert out.shape[0] == idx.shape[0]

    def test_indices_select_the_returned_points(self):
        """Callers index colour/segmentation arrays with these, so they must align."""
        points = _box_cloud((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), n=800, seed=3)
        out, idx = voxel_downsample(points, 0.02)
        assert np.allclose(points[idx], out)

    def test_empty_input(self):
        out, idx = voxel_downsample(np.empty((0, 3)), 0.01)
        assert out.shape == (0, 3) and idx.shape == (0,)

    def test_rejects_bad_voxel_size(self):
        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((5, 3)), 0.0)


class TestOutlierRemoval:
    def test_removes_a_distant_flying_pixel(self):
        """Depth sensors interpolate across silhouettes; those points inflate the box."""
        points = np.vstack([_box_cloud((0, 0, 0), (0.05, 0.05, 0.05), n=200), [[3.0, 3.0, 3.0]]])
        keep = remove_statistical_outliers(points, std_ratio=2.0)
        assert not keep[-1], "the far outlier was not rejected"
        assert keep[:-1].mean() > 0.9, "too many inliers rejected"

    def test_small_clusters_are_kept_intact(self):
        """Deleting a small-but-real object is worse than keeping noise."""
        points = _box_cloud((0, 0, 0), (0.02, 0.02, 0.02), n=5)
        assert remove_statistical_outliers(points, k=8).all()


class TestOrientedBoundingBox:
    def test_recovers_axis_aligned_box(self):
        center, quat, extents = fit_oriented_bbox(
            _box_cloud((0.5, -0.2, 0.45), (0.06, 0.10, 0.08), n=4000)
        )
        assert np.allclose(center, [0.5, -0.2, 0.45], atol=0.005)
        assert np.allclose(np.sort(extents), np.sort([0.06, 0.10, 0.08]), atol=0.01)

    @pytest.mark.parametrize("yaw", [0.0, 0.3, 0.9, -0.6])
    def test_recovers_yaw(self, yaw):
        _, quat, extents = fit_oriented_bbox(
            _box_cloud((0.4, 0.1, 0.5), (0.12, 0.05, 0.07), yaw=yaw, n=6000)
        )
        rotation = tf.quat_to_matrix(quat)
        # The long side must align with the rotated X axis, up to 180-degree
        # ambiguity, which is inherent for a box.
        expected = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        recovered = rotation[:, int(np.argmax(extents))]
        alignment = abs(float(np.dot(recovered, expected)))
        assert alignment > 0.97, f"yaw not recovered (alignment {alignment:.3f})"

    def test_is_gravity_aligned(self):
        """Local +Z must stay world up.

        A full 3-DoF PCA on single-viewpoint data tilts the box, and every grasp
        derived from it inherits the tilt.
        """
        points = _box_cloud((0.5, 0, 0.5), (0.1, 0.06, 0.04), yaw=0.4, n=3000)
        # Bias the cloud toward one face, as a single camera would.
        points = points[points[:, 2] > 0.49]
        _, quat, _ = fit_oriented_bbox(points, gravity_aligned=True)
        z_axis = tf.quat_to_matrix(quat)[:, 2]
        assert np.allclose(z_axis, [0, 0, 1], atol=1e-9)

    def test_extents_are_full_side_lengths(self):
        _, _, extents = fit_oriented_bbox(_box_cloud((0, 0, 0), (0.2, 0.1, 0.05), n=5000))
        assert np.max(extents) == pytest.approx(0.2, abs=0.01)

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            fit_oriented_bbox(np.zeros((2, 3)))

    def test_handles_vertical_line_without_crashing(self):
        """Yaw is unobservable for a vertical column; it must not raise."""
        points = np.column_stack([np.zeros(50), np.zeros(50), np.linspace(0, 0.2, 50)])
        _, quat, _ = fit_oriented_bbox(points)
        assert np.isfinite(quat).all()


class TestClustering:
    def test_separates_two_distant_blobs(self):
        points = np.vstack(
            [
                _box_cloud((0.0, 0.0, 0.0), (0.04, 0.04, 0.04), n=300, seed=1),
                _box_cloud((0.5, 0.0, 0.0), (0.04, 0.04, 0.04), n=300, seed=2),
            ]
        )
        clusters = cluster_by_euclidean_distance(points, tolerance=0.02, min_cluster_size=20)
        assert len(clusters) == 2
        assert all(len(c) > 200 for c in clusters)

    def test_drops_clusters_below_minimum_size(self):
        points = np.vstack([_box_cloud((0, 0, 0), (0.04, 0.04, 0.04), n=300), [[5.0, 5.0, 5.0]]])
        clusters = cluster_by_euclidean_distance(points, tolerance=0.02, min_cluster_size=20)
        assert len(clusters) == 1

    def test_plane_removal_splits_objects(self):
        """Without stripping the support the table bridges everything into one blob."""
        # A regular grid, so "the table is continuous" is deterministic rather
        # than dependent on where uniform random sampling happened to leave gaps.
        grid = np.arange(-0.3, 0.3001, 0.01)
        gx, gy = np.meshgrid(grid, grid)
        table = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        # Seated *on* the surface (bottom face at z=0), which is what makes the
        # table bridge them into one component.
        objects = np.vstack(
            [
                _box_cloud((-0.2, 0.0, 0.03), (0.04, 0.04, 0.06), n=400, seed=4),
                _box_cloud((0.2, 0.0, 0.03), (0.04, 0.04, 0.06), n=400, seed=5),
            ]
        )
        scene = np.vstack([table, objects])

        merged = cluster_by_euclidean_distance(scene, tolerance=0.03, min_cluster_size=20)
        assert len(merged) == 1, "table should bridge the objects before plane removal"

        above = scene[points_above_plane(scene, plane_z=0.0, clearance=0.01)]
        separated = cluster_by_euclidean_distance(above, tolerance=0.03, min_cluster_size=20)
        assert len(separated) == 2


class TestObjectTracker:
    def test_assigns_stable_ids_across_frames(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=100)
        tracker.update([_hypothesis("", (0.5, 0.0, 0.5))], step_index=0)
        objects = tracker.update([_hypothesis("", (0.51, 0.0, 0.5), step=1)], step_index=1)

        assert len(objects) == 1
        track_id = next(iter(objects))
        objects = tracker.update([_hypothesis("", (0.52, 0.0, 0.5), step=2)], step_index=2)
        assert track_id in objects, "identity changed while the object barely moved"

    def test_requires_corroboration_before_confirming(self):
        """A one-frame blob is usually a segmentation artefact."""
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=100)
        assert tracker.update([_hypothesis("", (0.5, 0, 0.5))], step_index=0) == {}
        assert len(tracker.update([_hypothesis("", (0.5, 0, 0.5), step=1)], step_index=1)) == 1

    def test_new_object_gets_a_new_id(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=100)
        for step in range(2):
            tracker.update([_hypothesis("", (0.5, 0.0, 0.5), step=step)], step_index=step)
        objects = tracker.update(
            [_hypothesis("", (0.5, 0.0, 0.5), step=2), _hypothesis("", (0.2, 0.3, 0.5), step=2)],
            step_index=2,
        )
        tracker.update(
            [_hypothesis("", (0.5, 0.0, 0.5), step=3), _hypothesis("", (0.2, 0.3, 0.5), step=3)],
            step_index=3,
        )
        assert len(tracker.confirmed_objects()) == 2

    def test_track_expires_when_object_disappears(self):
        """Objects may be removed; stale tracks must not linger as phantom targets."""
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=5)
        for step in range(2):
            tracker.update([_hypothesis("", (0.5, 0, 0.5), step=step)], step_index=step)
        assert len(tracker.confirmed_objects()) == 1

        for step in range(2, 12):
            tracker.update([], step_index=step)
        assert tracker.confirmed_objects() == {}

    def test_does_not_swap_identities_of_close_objects(self):
        """The failure global assignment exists to prevent.

        Two objects converging: greedy matching can assign both detections to
        the same nearer track and swap identities. Swapping the id of the held
        object is the worst outcome this layer can produce.
        """
        tracker = ObjectTracker(match_distance=0.10, max_age_steps=100)
        for step in range(2):
            tracker.update(
                [
                    _hypothesis("", (0.50, 0.00, 0.5), step=step),
                    _hypothesis("", (0.50, 0.08, 0.5), step=step),
                ],
                step_index=step,
            )

        objects = tracker.confirmed_objects()
        id_a = next(k for k, v in objects.items() if v.pose.position[1] < 0.04)
        id_b = next(k for k, v in objects.items() if v.pose.position[1] >= 0.04)

        updated = tracker.update(
            [
                _hypothesis("", (0.50, 0.02, 0.5), step=2),
                _hypothesis("", (0.50, 0.07, 0.5), step=2),
            ],
            step_index=2,
        )
        assert updated[id_a].pose.position[1] < updated[id_b].pose.position[1], (
            "tracks swapped identity"
        )

    def test_reidentifies_a_uniquely_labelled_object_that_jumped(self):
        """A pick-and-place moves an object far beyond the proximity gate.

        Losing identity there would break "place it" for the very object just
        manipulated, so a unique class match re-identifies it.
        """
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=100)
        for step in range(2):
            tracker.update(
                [_hypothesis("", (0.5, 0.0, 0.5), label="can", step=step)], step_index=step
            )
        original_id = next(iter(tracker.confirmed_objects()))

        moved = tracker.update(
            [_hypothesis("", (0.5, 0.30, 0.5), label="can", step=2)], step_index=2
        )
        assert original_id in moved, "identity lost after a large displacement"
        assert moved[original_id].pose.position[1] == pytest.approx(0.30)

    def test_does_not_reidentify_when_the_class_is_ambiguous(self):
        """Two cans that both jump must not be guessed at.

        A wrong re-identification is worse than a lost one: it would silently
        rename the object in the gripper.
        """
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=100)
        for step in range(2):
            tracker.update(
                [
                    _hypothesis("", (0.5, 0.00, 0.5), label="can", step=step),
                    _hypothesis("", (0.5, 0.40, 0.5), label="can", step=step),
                ],
                step_index=step,
            )
        original_ids = set(tracker.confirmed_objects())

        for step in (2, 3):
            tracker.update(
                [
                    _hypothesis("", (0.2, -0.30, 0.5), label="can", step=step),
                    _hypothesis("", (0.2, 0.70, 0.5), label="can", step=step),
                ],
                step_index=step,
            )
        assert set(tracker.confirmed_objects()) - original_ids, (
            "ambiguous jumps should produce new tracks, not guesses"
        )

    def test_far_detection_of_a_different_class_is_not_matched(self):
        """The proximity gate in isolation.

        Uses a different label so class re-identification cannot fire: a distant
        detection of an unrelated class must never inherit an existing identity.
        """
        tracker = ObjectTracker(match_distance=0.05, max_age_steps=100)
        for step in range(2):
            tracker.update(
                [_hypothesis("", (0.5, 0, 0.5), label="can", step=step)], step_index=step
            )
        first_id = next(iter(tracker.confirmed_objects()))

        for step in (2, 3):
            tracker.update(
                [_hypothesis("", (1.5, 0, 0.5), label="block", step=step)], step_index=step
            )

        confirmed = tracker.confirmed_objects()
        assert first_id not in confirmed or float(
            np.linalg.norm(confirmed[first_id].pose.position - np.array([0.5, 0.0, 0.5]))
        ) < 0.01, "a distant, differently-classed detection hijacked an existing track"


class TestColorNaming:
    """Colour is what lets an operator say "the green box" instead of memorising
    the class vocabulary, so getting it wrong makes objects unreferenceable."""

    @pytest.mark.parametrize(
        "rgb,expected",
        [
            # Values MEASURED from the rendered scene, not idealised swatches.
            # Lighting lifts every surface toward white: the box is materially
            # (0.15, 0.65, 0.25) yet arrives at [118, 203, 146].
            ((215, 115, 113), "red"),
            ((113, 155, 212), "blue"),
            ((118, 203, 146), "green"),
        ],
    )
    def test_names_washed_out_rendered_colors(self, rgb, expected):
        """The regression that matters.

        An RGB-exemplar classifier scored these as pink/white/white, because
        white lies on the grey diagonal and every lightened colour lies near it.
        Hue is invariant to that lightening.
        """
        pixels = np.tile(np.array(rgb, dtype=np.uint8), (400, 1))
        from mfw.vision.geometry import dominant_color_name

        assert dominant_color_name(pixels) == expected

    @pytest.mark.parametrize(
        "rgb,expected",
        [((200, 30, 30), "red"), ((40, 170, 60), "green"), ((40, 70, 200), "blue")],
    )
    def test_names_saturated_colors(self, rgb, expected):
        from mfw.vision.geometry import dominant_color_name

        assert dominant_color_name(np.tile(np.array(rgb, np.uint8), (400, 1))) == expected

    @pytest.mark.parametrize(
        "rgb,expected",
        [((240, 240, 240), "white"), ((128, 128, 128), "grey"), ((20, 20, 20), "black")],
    )
    def test_names_neutrals_by_lightness(self, rgb, expected):
        """Neutrals have no meaningful hue; naming them by hue would be noise."""
        from mfw.vision.geometry import dominant_color_name

        assert dominant_color_name(np.tile(np.array(rgb, np.uint8), (400, 1))) == expected

    def test_median_resists_specular_highlights(self):
        """A few blown-out pixels must not turn a red object white."""
        from mfw.vision.geometry import dominant_color_name

        body = np.tile(np.array([200, 40, 40], np.uint8), (400, 1))
        glare = np.tile(np.array([255, 255, 255], np.uint8), (60, 1))
        assert dominant_color_name(np.vstack([body, glare])) == "red"

    def test_empty_input_returns_no_claim(self):
        """Empty string, not a guess: callers must distinguish "unknown"."""
        from mfw.vision.geometry import dominant_color_name

        assert dominant_color_name(None) == ""
        assert dominant_color_name(np.empty((0, 3), np.uint8)) == ""


class TestColorAwareLookup:
    def _obj(self, track_id, label, color, position=(0.5, 0.0, 0.45)):
        pose = Pose(np.asarray(position, float), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        return ObjectHypothesis(
            track_id=track_id, label=label, pose=pose,
            bbox=BoundingBox3D(center=pose, extents=np.array([0.1, 0.1, 0.08])),
            confidence=0.9, num_points=500, last_seen_sim_time=0.0, last_seen_step=0,
            attributes={"color": color},
        )

    def _scene(self):
        from mfw.core.types import SceneGraph

        objects = [
            self._obj("o1", "box", "green", (0.62, 0.28, 0.44)),
            self._obj("o2", "can", "blue", (0.50, 0.12, 0.47)),
            self._obj("o3", "block", "red", (0.45, -0.15, 0.43)),
        ]
        return SceneGraph({o.track_id: o for o in objects}, 0.0, 0)

    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("green box", "o1"), ("the green box", "o1"), ("box", "o1"),
            ("blue can", "o2"), ("can", "o2"), ("red block", "o3"),
        ],
    )
    def test_resolves_colour_qualified_references(self, phrase, expected):
        """The exact failure seen in use: "place the can on the green box"
        reported "cannot find 'green box'" while looking straight at one."""
        hits = self._scene().by_label(phrase)
        assert hits and hits[0].track_id == expected

    def test_refuses_a_wrong_colour_class_pair(self):
        """"green can" must find nothing rather than settle for the blue can.
        Acting on a mismatched referent moves the wrong object."""
        assert self._scene().by_label("green can") == []


class TestSceneRelations:
    def test_detects_stacking(self):
        objects = {
            "lower": _hypothesis("lower", (0.5, 0.0, 0.40), extents=(0.12, 0.12, 0.08)),
            "upper": _hypothesis("upper", (0.5, 0.0, 0.47), extents=(0.05, 0.05, 0.05)),
        }
        assert ("upper", "on_top_of", "lower") in compute_relations(objects)

    def test_side_by_side_is_not_stacked(self):
        """Height alone would wrongly report a neighbour as stacked."""
        objects = {
            "a": _hypothesis("a", (0.5, 0.00, 0.40), extents=(0.10, 0.10, 0.08)),
            "b": _hypothesis("b", (0.5, 0.30, 0.47), extents=(0.05, 0.05, 0.05)),
        }
        relations = compute_relations(objects)
        assert not any(predicate == "on_top_of" for _, predicate, _ in relations)

    def test_next_to_is_symmetric(self):
        objects = {
            "a": _hypothesis("a", (0.5, 0.00, 0.45)),
            "b": _hypothesis("b", (0.5, 0.08, 0.45)),
        }
        relations = compute_relations(objects, next_to_distance=0.15)
        assert ("a", "next_to", "b") in relations
        assert ("b", "next_to", "a") in relations

    def test_left_right_use_robot_frame(self):
        """+Y is the robot's left with the base at the origin facing +X."""
        objects = {
            "left": _hypothesis("left", (0.5, 0.2, 0.45)),
            "right": _hypothesis("right", (0.5, -0.2, 0.45)),
        }
        relations = compute_relations(objects)
        assert ("left", "left_of", "right") in relations
        assert ("right", "right_of", "left") in relations

    def test_inside_detected_for_contained_object(self):
        objects = {
            "container": _hypothesis("container", (0.6, 0.0, 0.45), extents=(0.20, 0.20, 0.15)),
            "content": _hypothesis("content", (0.6, 0.0, 0.45), extents=(0.04, 0.04, 0.04)),
        }
        assert ("content", "inside", "container") in compute_relations(objects)

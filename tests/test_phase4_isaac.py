"""Phase 4 gate: grasp generation on real perceived geometry.

The pure-logic suite already covers grasp maths against synthetic boxes. What
only simulation can answer is whether grasps generated from *perceived* extents
are reachable by the actual arm -- perception measures partial surfaces, so a
grasp that is geometrically perfect on a synthetic cube may be unreachable here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.grasp.generator import generate_grasp_candidates

pytestmark = [pytest.mark.isaac, pytest.mark.phase4]


@pytest.fixture
def perceived(restore_scene):
    runtime = restore_scene
    runtime.vision.observe()
    return runtime, runtime.vision.observe()


def _graspable(runtime, scene):
    """Objects narrow enough for the gripper, by perceived extents alone."""
    usable = runtime.config.grasp.max_grasp_width - runtime.config.grasp.finger_width_margin
    return [o for o in scene.objects.values() if float(np.min(o.bbox.extents)) <= usable]


class TestGenerationOnPerceivedObjects:
    def test_generates_grasps_for_at_least_one_perceived_object(self, perceived):
        runtime, scene = perceived
        targets = _graspable(runtime, scene)
        assert targets, (
            "no perceived object is narrow enough to grasp: "
            f"{[(o.label, np.round(o.bbox.extents, 3).tolist()) for o in scene.objects.values()]}"
        )

        total = sum(
            len(generate_grasp_candidates(o, runtime.config.grasp)) for o in targets
        )
        assert total > 0, "no candidates generated from perceived geometry"

    def test_oversized_objects_yield_no_candidates(self, perceived):
        """A box wider than the gripper must be reported ungraspable, not attempted."""
        runtime, scene = perceived
        usable = runtime.config.grasp.max_grasp_width - runtime.config.grasp.finger_width_margin
        for obj in scene.objects.values():
            if float(np.min(obj.bbox.extents)) > usable:
                assert generate_grasp_candidates(obj, runtime.config.grasp) == []

    def test_candidate_widths_come_from_perceived_extents(self, perceived):
        """Widths must trace back to measurement, not to a configured constant."""
        runtime, scene = perceived
        for obj in _graspable(runtime, scene):
            for candidate in generate_grasp_candidates(obj, runtime.config.grasp):
                assert np.any(
                    np.isclose(candidate.width, obj.bbox.extents, atol=1e-9)
                ), "grasp width does not match any perceived box extent"


class TestScoringWithRealKinematics:
    def test_produces_at_least_one_reachable_grasp(self, perceived):
        """The Phase 4 claim: perception plus synthesis yields an executable grasp."""
        runtime, scene = perceived
        state = runtime.robot.get_state()

        best_per_object = {}
        for obj in _graspable(runtime, scene):
            candidates = generate_grasp_candidates(obj, runtime.config.grasp)
            scored = runtime.grasp_scorer.score(candidates, scene, state)
            if scored:
                best_per_object[obj.label] = scored[0]

        assert best_per_object, (
            "no reachable grasp for any perceived object; "
            f"graspable labels were {[o.label for o in _graspable(runtime, scene)]}"
        )
        for label, grasp in best_per_object.items():
            assert 0.0 < grasp.score <= 1.0
            assert grasp.scores_breakdown

    def test_chosen_grasps_have_valid_ik_for_both_poses(self, perceived):
        """Both the standoff and the closure pose must be solvable.

        A reachable grasp with an unreachable pregrasp cannot be approached in a
        straight line, which is the only approach that does not sweep the fingers
        through the object.
        """
        runtime, scene = perceived
        state = runtime.robot.get_state()

        checked = 0
        for obj in _graspable(runtime, scene):
            scored = runtime.grasp_scorer.score(
                generate_grasp_candidates(obj, runtime.config.grasp), scene, state
            )
            for grasp in scored[:3]:
                assert runtime.robot.inverse_kinematics(grasp.pose) is not None
                assert runtime.robot.inverse_kinematics(grasp.pregrasp_pose) is not None
                checked += 1
        assert checked > 0, "no grasps available to verify"

    def test_grasps_stay_above_the_table(self, perceived):
        runtime, scene = perceived
        state = runtime.robot.get_state()
        table = runtime.table_top_height

        for obj in _graspable(runtime, scene):
            for grasp in runtime.grasp_scorer.score(
                generate_grasp_candidates(obj, runtime.config.grasp), scene, state
            ):
                assert grasp.pose.position[2] > table, (
                    f"grasp point at z={grasp.pose.position[2]:.3f} is below the table "
                    f"top at {table:.3f}"
                )

    def test_grasp_target_is_close_to_the_object_it_targets(self, perceived):
        """Guards against a grasp being scored against the wrong track."""
        runtime, scene = perceived
        state = runtime.robot.get_state()

        for obj in _graspable(runtime, scene):
            for grasp in runtime.grasp_scorer.score(
                generate_grasp_candidates(obj, runtime.config.grasp), scene, state
            )[:3]:
                assert grasp.target_track_id == obj.track_id
                distance = float(
                    np.linalg.norm(grasp.pose.position - obj.bbox.center.position)
                )
                assert distance < float(np.max(obj.bbox.extents)) + 0.01

    def test_planner_can_reach_a_chosen_pregrasp(self, perceived):
        """Closes the loop with Phase 3: the grasp must be plannable, not just IK-valid."""
        runtime, scene = perceived
        state = runtime.robot.get_state()

        for obj in _graspable(runtime, scene):
            scored = runtime.grasp_scorer.score(
                generate_grasp_candidates(obj, runtime.config.grasp), scene, state
            )
            for grasp in scored[:4]:
                runtime.planner.exclude_from_collision(grasp.target_track_id)
                try:
                    trajectory = runtime.planner.plan_to_pose(
                        state.joint_state, grasp.pregrasp_pose, scene
                    )
                finally:
                    runtime.planner.include_in_collision(grasp.target_track_id)
                if trajectory is not None:
                    return
        pytest.fail("no chosen grasp had a plannable pregrasp approach")

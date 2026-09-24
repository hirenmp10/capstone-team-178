"""Skills resolve object references through grounding -- no Isaac.

Before this, every skill used ``SceneGraph.by_label`` and the Isaac run of
2026-09-24 (logs/e2e/run_a_default.txt) failed "pick the red object", "pick
the large object" and "pick the object" (three in view) with ObjectNotFound,
the last retried three times instead of asking which one. These tests drive
the real skills (Pick, Place, MoveTo, LookAt, Observe) against the fake world
of ``tests/test_place_logic.py`` and check *which object the skill acted on*,
and that reference errors reach the planner as a question or one refusal.

The last class is the hardware-lane F3 guard, end to end on the fake robot
server and scripted detector: a detector error after the jaw closed used to
let the planner retry Pick, whose first act was to open the jaw at lift
height (re-review, gv/probe2.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfw.core.types import SceneGraph, SkillStatus
from mfw.language.intent_parser import RuleBasedIntentParser
from mfw.planner.task_planner import TaskPlanner
from mfw.skills.primitives import LookAt, MoveTo, Observe, Pick
from mfw.skills.registry import ClassicalExecutor, SkillRegistry
from tests.test_place_logic import Body, FakePlanner, default_bodies, make_rig


class RecordingGenerator:
    """Stands in for grasp synthesis: records which object Pick chose, offers no grasp."""

    def __init__(self) -> None:
        self.targets: list[str] = []

    def generate(self, scene, track_id):
        self.targets.append(track_id)
        return []


def pick_rig(bodies: list[Body] | None = None):
    rig = make_rig(bodies or default_bodies())
    generator = RecordingGenerator()
    rig.context.grasp_generator = generator
    return rig, generator


def _pick(rig, target: str):
    return Pick(rig.context).execute({"target": target})


# default scene: obj_001 red block (right, y=-0.15), obj_002 blue can (y=0.116),
# obj_003 green box (left, y=0.316, the largest)


class TestPickSelectsTheGroundedObject:
    @pytest.mark.parametrize("phrase,expected", [
        ("red object", "obj_001"),          # Isaac: ObjectNotFound
        ("the red one", "obj_001"),
        ("large object", "obj_003"),        # Isaac: ObjectNotFound
        ("the biggest thing", "obj_003"),
        ("the small object", "obj_001"),
        ("the tall one", "obj_002"),
        ("the cube", "obj_001"),            # class synonym
        ("the blue cylinder", "obj_002"),
        ("the object on the left", "obj_003"),
        ("the rightmost object", "obj_001"),
        ("the nearest object", "obj_001"),
        ("the object right of the can", "obj_001"),
        ("obj_002", "obj_002"),             # the re-run after a clarification
        ("blue can", "obj_002"),            # the old path still works
    ])
    def test_qualifiers_choose_the_object(self, phrase, expected):
        rig, generator = pick_rig()
        result = _pick(rig, phrase)
        assert generator.targets == [expected], result.message

    def test_pronoun_goes_through_memory(self):
        rig, generator = pick_rig()
        rig.memory.note_reference("obj_003")
        _pick(rig, "it")
        assert generator.targets == ["obj_003"]


class TestReferenceErrorsPropagate:
    def test_ambiguity_carries_the_candidates_and_ids(self):
        rig, generator = pick_rig()
        result = _pick(rig, "the object")
        assert result.status is SkillStatus.FAILED and generator.targets == []
        assert result.message.startswith("AmbiguousReference:")
        assert result.data["error_type"] == "AmbiguousReference"
        assert result.data["candidates"] == ["green box", "blue can", "red block"]
        assert result.data["track_ids"] == ["obj_003", "obj_002", "obj_001"]

    def test_not_found_lists_what_is_visible_with_colour(self):
        rig, generator = pick_rig()
        result = _pick(rig, "the purple dragon")
        assert result.data["error_type"] == "ObjectNotFound" and generator.targets == []
        assert "currently visible: red block, blue can, green box" in result.message
        assert result.data["visible"] == ["red block", "blue can", "green box"]

    def _planner(self, rig, skills=(Pick,)):
        executor = ClassicalExecutor(SkillRegistry(rig.context, skills))
        calls = []
        original = executor.execute

        def counting(name, params):
            calls.append((name, dict(params)))
            return original(name, params)

        executor.execute = counting  # type: ignore[method-assign]
        planner = TaskPlanner(RuleBasedIntentParser(tuple(s.skill_name for s in skills)),
                              {"classical": executor}, rig.memory, rig.config)
        return planner, calls

    def test_pick_the_object_becomes_one_question(self):
        """Isaac: 'pick the object' -> ObjectNotFound, 3 attempts. Now: a question, 1 attempt."""
        rig, _ = pick_rig()
        planner, calls = self._planner(rig)
        outcome = planner.handle("pick the object")
        assert outcome.needs_clarification and outcome.attempts == 1 and len(calls) == 1
        assert outcome.clarification_options == ("green box", "blue can", "red block")
        assert outcome.clarification_track_ids == ("obj_003", "obj_002", "obj_001")

    def test_a_missing_object_is_refused_once(self):
        rig, _ = pick_rig()
        planner, calls = self._planner(rig)
        outcome = planner.handle("pick the purple dragon")
        assert not outcome.ok and outcome.attempts == 1 and len(calls) == 1
        assert "red block, blue can, green box" in outcome.message

    def test_the_answer_reruns_on_the_chosen_track(self):
        rig, generator = pick_rig()
        planner, _ = self._planner(rig)
        planner.handle("pick the object")
        planner.handle("pick obj_002")
        assert generator.targets == ["obj_002"]


def _two_bowls() -> list[Body]:
    return default_bodies() + [
        Body("obj_004", "bowl", "white", np.array([0.55, 0.05, 0.43]), np.array([0.14, 0.14, 0.06])),
        Body("obj_005", "bowl", "white", np.array([0.55, -0.35, 0.43]), np.array([0.14, 0.14, 0.06])),
    ]


class TestPlaceDestinationGrounding:
    def test_a_qualified_container_is_resolved(self):
        rig = make_rig(_two_bowls(), held="obj_001")
        result = rig.place({"relation": "in", "target": "the bowl on the right"})
        assert result.ok, result.message
        assert result.data["place_target"]["destination_id"] == "obj_005"
        assert np.linalg.norm(rig.world.bodies["obj_001"].centre[:2] - [0.55, -0.35]) < 1e-6

    def test_an_ambiguous_destination_is_a_question_and_nothing_moves(self):
        rig = make_rig(_two_bowls(), held="obj_001")
        result = rig.place({"relation": "in", "target": "the bowl"})
        assert result.data["error_type"] == "AmbiguousReference"
        assert set(result.data["track_ids"]) == {"obj_004", "obj_005"}
        assert rig.controller.followed == [] and rig.world.held == "obj_001"
        assert rig.memory.get_held_object() == "obj_001"

    def test_the_held_object_is_never_its_own_destination(self):
        rig = make_rig(default_bodies(), held="obj_002")
        result = rig.place({"relation": "next_to", "target": "it"})
        assert result.data["error_type"] == "ObjectNotFound" and rig.controller.followed == []
        result = rig.place({"relation": "next_to", "target": "the can"})
        assert result.data["error_type"] == "ObjectNotFound"

    def test_place_to_uses_the_grounded_destination_class(self):
        rig = make_rig(_two_bowls(), held="obj_001")
        result = rig.place({"relation": "to", "target": "the green box"})
        assert result.ok, result.message
        assert result.data["place_target"]["relation"] == "on"


class _PosePlanner(FakePlanner):
    def plan_to_pose(self, start, goal, scene):
        return self.plan_with_retries(start, goal, scene)


class TestOtherSkillsGround:
    def test_move_to_the_red_one(self):
        rig = make_rig(default_bodies())
        rig.context.planner = rig.planner = _PosePlanner()
        result = MoveTo(rig.context).execute({"target": "the red one"})
        assert result.ok, result.message
        assert rig.planner.plans[0].goal.position[:2] == pytest.approx([0.4598, -0.1504])

    def test_look_at_the_large_object(self):
        rig = make_rig(default_bodies())
        rig.context.planner = rig.planner = _PosePlanner()
        result = LookAt(rig.context).execute({"target": "the large object"})
        assert result.ok and result.data["track_id"] == "obj_003"

    def test_observe_reports_colour(self):
        rig = make_rig(default_bodies())
        result = Observe(rig.context).execute({})
        assert result.message == "observed 3 object(s): red block, blue can, green box"
        assert [o["color"] for o in result.data["objects"]] == ["red", "blue", "green"]


# ----------------------------------------------------------------------
# a second look before refusing
# ----------------------------------------------------------------------


def _first_look_sees_only(rig, *track_ids: str) -> list[int]:
    """The observation that re-detects aged-out tracks holds them unconfirmed.

    Isaac run_c, events 20260924_104645_0cc762 seq 46: 3 raw instances, one
    object in the scene (the red block); the next observation had all three.
    Returns the per-call object counts, for asserting how often Pick looked.
    """
    original = rig.vision.observe
    counts: list[int] = []

    def observe():
        scene = original()
        if not counts:
            scene = SceneGraph({k: v for k, v in scene.objects.items() if k in track_ids},
                               scene.sim_time, scene.step_index, scene.relations)
            rig.vision.last = scene
        counts.append(len(scene.objects))
        return scene

    rig.vision.observe = observe  # type: ignore[method-assign]
    return counts


class TestALookAgainBeforeRefusing:
    def test_pick_the_blue_cube_lists_everything_and_says_why(self):
        """Isaac: "cannot find 'blue cube'; currently visible: red block" with three in view."""
        rig, generator = pick_rig()
        counts = _first_look_sees_only(rig, "obj_001")
        result = _pick(rig, "the blue cube")
        assert counts == [1, 3] and generator.targets == []
        assert result.data["error_type"] == "ObjectNotFound"
        assert result.message == (
            "ObjectNotFound: cannot find 'the blue cube': the only block is the red block; "
            "currently visible: red block, blue can, green box"
        )
        assert result.data["visible"] == ["red block", "blue can", "green box"]

    def test_an_object_missing_from_the_first_look_is_picked(self):
        rig, generator = pick_rig()
        counts = _first_look_sees_only(rig, "obj_001")
        _pick(rig, "the blue can")
        assert counts == [1, 3] and generator.targets == ["obj_002"]

    def test_a_found_object_costs_no_second_look(self):
        rig, generator = pick_rig()
        counts = _first_look_sees_only(rig, "obj_001", "obj_002", "obj_003")
        _pick(rig, "the blue can")
        assert counts == [3] and generator.targets == ["obj_002"]

    def test_an_ambiguous_reference_is_asked_not_looked_at_again(self):
        rig, _ = pick_rig()
        counts = _first_look_sees_only(rig, "obj_001", "obj_002", "obj_003")
        result = _pick(rig, "the object")
        assert counts == [3] and result.data["error_type"] == "AmbiguousReference"

    def test_place_destination_missing_from_the_first_look(self):
        rig = make_rig(default_bodies(), held="obj_001")
        counts = _first_look_sees_only(rig, "obj_001", "obj_002")
        result = rig.place({"relation": "on", "target": "the green box"})
        assert result.ok, result.message
        assert counts[:2] == [2, 3]
        assert result.data["place_target"]["destination_id"] == "obj_003"


# ----------------------------------------------------------------------
# F3 on the hardware fake lane
# ----------------------------------------------------------------------


@pytest.fixture
def failing_lane(tmp_path):
    """The e2e fake lane whose detector raises once, on the first observe after the lift."""
    # The simulation-only checkout omits the hardware lane; skip rather than error.
    pytest.importorskip("jetson.robot_server", reason="hardware lane (jetson/) is not in this checkout")
    pytest.importorskip("mfw.hardware.kinematics", reason="hardware lane (mfw/hardware) is not in this checkout")
    from jetson.detector_service import DetectorServer, ScriptedDetector, SyntheticPinhole
    from jetson.robot_server import FakeDriver, RobotServer, ServoCalibration
    from mfw.config.schema import load_config
    from mfw.hardware.kinematics import PlanarKinematics
    from tests.test_hardware_e2e import (
        BOWL_XY,
        HARDWARE_FAKE_YAML,
        MARKER_XY,
        FakeWorld,
        _load_fake_config,
    )

    base = load_config(HARDWARE_FAKE_YAML)
    kinematics = PlanarKinematics(base.hardware.arm, base.robot.arm_joint_names)
    footprints = {k: tuple(v) for k, v in base.hardware.object_sizes.items()}
    world = FakeWorld(kinematics, {"marker": MARKER_XY, "bowl": BOWL_XY}, footprints)
    driver = FakeDriver(ServoCalibration.default(), on_command=world.on_command)
    robot_server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                               follower_sleep=lambda _s: None)
    _, robot_port = robot_server.serve_in_thread(port=0)
    detector = ScriptedDetector(world.scene, SyntheticPinhole(), arm=lambda: world.tcp)
    state: dict[str, Any] = {"failures": 0}
    original = detector.detect

    def detect(rgb, labels, min_score=0.1):
        lifted = world.attached is not None and world.tcp is not None and world.tcp[2] > 0.05
        if lifted and state["failures"] == 0:
            state["failures"] += 1
            raise RuntimeError("injected: detector hiccup after the lift")
        return original(rgb, labels, min_score)

    detector.detect = detect  # type: ignore[method-assign]
    detector_server = DetectorServer("127.0.0.1", 0, detector)
    _, detector_port = detector_server.serve_in_thread(port=0)
    try:
        yield _load_fake_config(robot_port, detector_port, tmp_path), world, state
    finally:
        detector_server.stop()
        robot_server.stop()


class TestUnverifiedPickGuard:
    def test_a_detector_error_after_the_lift_keeps_the_jaw_closed(self, failing_lane):
        """Re-review F3: attempts=2, 'none are reachable', and a retry that opens the jaw."""
        cfg, world, state = failing_lane
        from mfw.assistant import Assistant

        assistant = Assistant(config=cfg)
        try:
            assert assistant.command("what do you see").ok
            picked = assistant.command("pick up the marker")
            held = assistant.runtime.memory.get_held_object()
            attached, width = world.attached, world.last_width
            events = [json.loads(line) for line in
                      Path(assistant.runtime.events.path).read_text(encoding="utf-8").splitlines()
                      if line.strip()]
            released = assistant.command("open the gripper")
        finally:
            assistant.close()
        assert state["failures"] == 1
        assert not picked.ok and picked.attempts == 1
        assert "could not be verified" in picked.message and "injected" in picked.message
        assert "jaw is left closed" in picked.message and "reachable" not in picked.message
        assert picked.result.status is SkillStatus.INFEASIBLE
        # The marker really is held, and was never dropped at lift height.
        assert attached == "marker" and width is not None and width < 0.019
        assert held is None  # unverified: memory does not claim a hold
        assert [e for e in events if e.get("event") == "pick.verification_error"]
        assert released.ok and world.attached is None

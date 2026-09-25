"""Grounding, move-X-to-Y, clarification and the voice gate -- pure logic.

No Isaac Sim, no runtime. Scenes are synthetic ``SceneGraph``s built the way
tests/test_vision_logic.py builds hypotheses, laid out like configs/default.yaml
(red block right, blue can centre, green box left and furthest) so the cases
the 2026-09-24 end-to-end run failed are replayed here exactly:

* "pick the red object"   -> ObjectNotFound  (now: the red block)
* "pick the large object" -> ObjectNotFound  (now: the green box)
* "pick the object" with three in view -> ObjectNotFound (now: a question)
* "move the red object to the left" -> move_relative, object dropped
  (now: pick the red object, then place it 10 cm to the left)
* an ambiguous "can" retried three times and never asked about (now: CLARIFY,
  one attempt, options that name real objects)

New parser phrases live here rather than in tests/test_language_logic.py on
purpose: every literal handed to ``.parse``/``.handle`` there is frozen by
tests/test_hardware_language.py, and these are new mappings, not frozen ones.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mfw.assistant import (
    NON_MOTION_SKILLS,
    Assistant,
    clarified_command,
    classify_confirmation,
    expand_clauses,
    requires_confirmation,
    run_clauses,
    run_with_clarification,
    split_conjoined,
)
from mfw.config.schema import MemoryConfig, load_config
from mfw.core.errors import (
    AmbiguousReference,
    ObjectNotFound,
    PlanningError,
    SafetyViolation,
)
from mfw.core.types import (
    BoundingBox3D,
    Frame,
    ObjectHypothesis,
    Pose,
    SceneGraph,
    SkillResult,
    SkillStatus,
)
from mfw.language.grounding import (
    clarification_question,
    describe,
    describe_candidates,
    interpret_clarification,
    parse_referent,
    resolve_reference,
    substitute_referent,
)
from mfw.language.intent_parser import (
    DEFAULT_PLACE_OFFSET_M,
    PLACE_DIRECTIONS,
    PLACE_RELATIONS,
    RuleBasedIntentParser,
    UnparsedCommand,
    split_transfer,
)
from mfw.memory.working_memory import WorkingMemory
from mfw.planner.state_machine import TRANSITIONS, State
from mfw.planner.task_planner import CommandOutcome, TaskPlanner

REPO_ROOT = Path(__file__).resolve().parents[1]

SKILLS = (
    "observe", "scan_scene", "look_at", "move_to", "move_relative", "pick", "place",
    "open_gripper", "close_gripper", "rotate_wrist", "go_home", "wait", "stop",
    "emergency_stop",
)


# ----------------------------------------------------------------------
# scene builders
# ----------------------------------------------------------------------


def _obj(track_id, label, color, position, extents=(0.05, 0.05, 0.05)):
    pose = Pose(np.asarray(position, dtype=float), np.array([1.0, 0, 0, 0]), Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id,
        label=label,
        pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.asarray(extents, dtype=float)),
        confidence=0.9,
        num_points=500,
        last_seen_sim_time=0.0,
        last_seen_step=0,
        attributes={"color": color},
    )


def _scene(*objects, relations=None):
    return SceneGraph(
        objects={o.track_id: o for o in objects}, sim_time=0.0, step_index=0,
        relations=list(relations or []),
    )


def default_scene():
    """configs/default.yaml's three objects, at their authored poses and sizes."""
    return _scene(
        _obj("obj_001", "block", "red", (0.45, -0.15, 0.45), (0.05, 0.05, 0.05)),
        _obj("obj_002", "can", "blue", (0.50, 0.12, 0.46), (0.066, 0.066, 0.12)),
        _obj("obj_003", "box", "green", (0.62, 0.28, 0.44), (0.12, 0.12, 0.08)),
    )


def benchmark_scene():
    """configs/benchmark.yaml's objects with their semantic labels (assets.yaml)."""
    return _scene(
        _obj("obj_001", "can", "red", (0.262, -0.347, 0.453), (0.068, 0.068, 0.10)),
        _obj("obj_002", "brick", "red", (0.522, -0.106, 0.428), (0.05, 0.075, 0.05)),
        _obj("obj_003", "box", "brown", (0.279, 0.184, 0.447), (0.09, 0.038, 0.09)),
        _obj("obj_004", "banana", "yellow", (0.467, 0.320, 0.422), (0.19, 0.036, 0.031)),
        _obj("obj_005", "marker", "black", (0.380, -0.027, 0.463), (0.12, 0.019, 0.019)),
        _obj("obj_006", "bowl", "red", (0.467, -0.315, 0.430), (0.16, 0.16, 0.055)),
    )


def _ground(phrase, scene, **kwargs):
    return resolve_reference(phrase, scene, **kwargs).track_id


# ----------------------------------------------------------------------
# grounding: the end-to-end failures
# ----------------------------------------------------------------------


class TestEndToEndFailuresNowGround:
    def test_red_object_is_the_red_block(self):
        assert _ground("red object", default_scene()) == "obj_001"

    def test_large_object_is_the_green_box(self):
        assert _ground("large object", default_scene()) == "obj_003"

    def test_bare_object_with_three_in_view_is_a_question_not_a_refusal(self):
        with pytest.raises(AmbiguousReference) as info:
            resolve_reference("object", default_scene())
        exc = info.value
        # Left to right as the robot sees them (+Y is left): box, can, block.
        assert exc.candidates == ("green box", "blue can", "red block")
        assert exc.track_ids == ("obj_003", "obj_002", "obj_001")
        assert "the green box, the blue can or the red block" in str(exc)

    def test_purple_dragon_is_refused_listing_what_is_visible(self):
        with pytest.raises(ObjectNotFound) as info:
            resolve_reference("purple dragon", default_scene())
        message = str(info.value)
        for seen in ("red block", "blue can", "green box"):
            assert seen in message
        assert info.value.visible == ("red block", "blue can", "green box")

    def test_a_named_class_that_is_absent_is_refused(self):
        with pytest.raises(ObjectNotFound, match="currently visible"):
            resolve_reference("teapot", default_scene())


class TestGenericNouns:
    @pytest.mark.parametrize("phrase", ["red object", "red thing", "red one", "red item", "red piece", "red"])
    def test_generic_noun_is_a_class_wildcard(self, phrase):
        assert _ground(phrase, default_scene()) == "obj_001"

    @pytest.mark.parametrize("phrase", ["object", "thing", "one", "item", "stuff"])
    def test_single_object_in_view_answers_a_generic_noun(self, phrase):
        scene = _scene(_obj("obj_007", "can", "blue", (0.5, 0.0, 0.46)))
        assert _ground(phrase, scene) == "obj_007"

    def test_generic_noun_with_no_colour_match_is_not_found(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("yellow object", default_scene())


class TestColour:
    def test_a_different_known_colour_never_matches(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("blue block", default_scene())

    def test_adjacent_hue_is_accepted_when_nothing_matches_exactly(self):
        """A lit red block measured as "orange" is still "the red block"."""
        scene = _scene(_obj("obj_001", "block", "orange", (0.5, 0.0, 0.45)),
                       _obj("obj_002", "can", "blue", (0.5, 0.2, 0.45)))
        assert _ground("red block", scene) == "obj_001"

    def test_exact_colour_beats_adjacent_hue(self):
        scene = _scene(_obj("obj_001", "block", "orange", (0.5, 0.0, 0.45)),
                       _obj("obj_002", "block", "red", (0.5, 0.2, 0.45)))
        assert _ground("red block", scene) == "obj_002"

    def test_unknown_colour_is_a_fallback_not_a_mismatch(self):
        """The hardware lane reports colour "" -- colour words must not kill it."""
        scene = _scene(_obj("obj_001", "cube", "", (0.3, 0.0, 0.02)),
                       _obj("obj_002", "banana", "", (0.3, 0.2, 0.02)))
        assert _ground("red cube", scene) == "obj_001"

    def test_colour_synonyms(self):
        scene = _scene(_obj("obj_001", "block", "grey", (0.5, 0.0, 0.45)),
                       _obj("obj_002", "block", "purple", (0.5, 0.2, 0.45)))
        assert _ground("gray block", scene) == "obj_001"
        assert _ground("violet block", scene) == "obj_002"


class TestClassSynonyms:
    @pytest.mark.parametrize(
        "phrase,expected",
        [("cube", "obj_001"), ("blue cylinder", "obj_002"), ("cylinder", "obj_002"),
         ("tin", "obj_002"), ("carton", "obj_003"), ("red cube", "obj_001"), ("brick", "obj_001"),
         ("blocks", "obj_001")],
    )
    def test_synonym_reaches_the_class(self, phrase, expected):
        assert _ground(phrase, default_scene()) == expected

    def test_blue_cube_is_still_refused_the_cube_is_red(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("blue cube", default_scene())

    def test_exact_class_is_preferred_and_distinct_classes_are_not_merged(self):
        scene = _scene(_obj("obj_001", "block", "red", (0.5, 0.2, 0.45)),
                       _obj("obj_002", "brick", "red", (0.5, -0.2, 0.45)))
        assert _ground("block", scene) == "obj_001"
        assert _ground("brick", scene) == "obj_002"
        with pytest.raises(AmbiguousReference):
            resolve_reference("cube", scene)  # a synonym of both, exact for neither

    def test_hardware_cube_label_answers_block(self):
        scene = _scene(_obj("obj_001", "cube", "", (0.3, 0.0, 0.02)),
                       _obj("obj_002", "bowl", "", (0.3, 0.2, 0.02), (0.15, 0.15, 0.06)))
        assert _ground("block", scene) == "obj_001"

    @pytest.mark.parametrize(
        "phrase,expected",
        [("soup can", "obj_001"), ("foam brick", "obj_002"), ("block", "obj_002"),
         ("pudding box", "obj_003"), ("pen", "obj_005"), ("large marker", "obj_005"),
         ("container", "obj_006"), ("bowl", "obj_006")],
    )
    def test_benchmark_asset_names_ground(self, phrase, expected):
        """Audit: by_label('soup can'|'foam brick'|'pudding box'|'large marker') -> []."""
        assert _ground(phrase, benchmark_scene()) == expected


class TestSize:
    def test_superlatives_rank_by_volume(self):
        scene = default_scene()
        assert _ground("biggest object", scene) == "obj_003"
        assert _ground("smallest object", scene) == "obj_001"
        assert _ground("the little one", scene) == "obj_001"
        assert _ground("tall one", scene) == "obj_002"  # by height, not volume

    def test_size_adjective_on_a_unique_class_is_that_object(self):
        assert _ground("small box", default_scene()) == "obj_003"

    def test_less_than_twenty_percent_apart_is_a_question(self):
        scene = _scene(_obj("obj_a", "block", "red", (0.5, 0.2, 0.45), (0.10, 0.10, 0.01)),
                       _obj("obj_b", "block", "red", (0.5, -0.2, 0.45), (0.11, 0.10, 0.01)))
        with pytest.raises(AmbiguousReference):
            resolve_reference("large block", scene)

    def test_a_clear_margin_decides(self):
        scene = _scene(_obj("obj_a", "block", "red", (0.5, 0.2, 0.45), (0.10, 0.10, 0.01)),
                       _obj("obj_b", "block", "red", (0.5, -0.2, 0.45), (0.13, 0.10, 0.01)))
        assert _ground("large block", scene) == "obj_b"
        assert _ground("small block", scene) == "obj_a"


class TestPosition:
    @pytest.mark.parametrize(
        "phrase,expected",
        [("object on the left", "obj_003"), ("the one on the right", "obj_001"),
         ("leftmost object", "obj_003"), ("rightmost", "obj_001"), ("left-most thing", "obj_003"),
         ("thing in the middle", "obj_002"), ("nearest object", "obj_001"),
         ("the object closest to you", "obj_001"), ("the one nearest to me", "obj_001"),
         ("farthest object", "obj_003"), ("object on your left", "obj_003"),
         ("object on the left hand side", "obj_003")],
    )
    def test_spatial_selectors(self, phrase, expected):
        assert _ground(phrase, default_scene()) == expected

    def test_distance_is_measured_from_the_robot_base(self):
        """robot_xy moves the origin: the box becomes the nearest."""
        assert _ground("nearest object", default_scene(), robot_xy=(0.7, 0.35)) == "obj_003"

    def test_level_objects_are_not_separated_by_left(self):
        scene = _scene(_obj("obj_a", "block", "red", (0.5, 0.10, 0.45)),
                       _obj("obj_b", "block", "red", (0.6, 0.12, 0.45)))
        with pytest.raises(AmbiguousReference):
            resolve_reference("block on the left", scene)


def _relation_scene():
    return _scene(
        _obj("obj_a", "block", "red", (0.50, 0.00, 0.45)),
        _obj("obj_c", "can", "blue", (0.50, 0.08, 0.47), (0.066, 0.066, 0.12)),
        _obj("obj_b", "block", "red", (0.50, -0.30, 0.45)),
    )


class TestRelations:
    def test_next_to(self):
        assert _ground("block next to the can", _relation_scene()) == "obj_a"
        assert _ground("the block near the can", _relation_scene()) == "obj_a"
        assert _ground("block beside the blue can", _relation_scene()) == "obj_a"

    def test_left_and_right_of(self):
        # Both blocks are right of the can; the clearly closer one is meant.
        assert _ground("block right of the can", _relation_scene()) == "obj_a"
        assert _ground("block to the right of the can", _relation_scene()) == "obj_a"
        with pytest.raises(ObjectNotFound, match="left of"):
            resolve_reference("block to the left of the can", _relation_scene())

    def test_on_in_under_from_geometry(self):
        scene = _scene(
            _obj("box", "box", "green", (0.6, 0.2, 0.44), (0.12, 0.12, 0.08)),
            _obj("can_top", "can", "blue", (0.6, 0.2, 0.54), (0.066, 0.066, 0.12)),
            _obj("can_table", "can", "blue", (0.4, -0.2, 0.46), (0.066, 0.066, 0.12)),
            _obj("bowl", "bowl", "white", (0.4, 0.3, 0.43), (0.16, 0.16, 0.06)),
            _obj("banana", "banana", "yellow", (0.4, 0.3, 0.43), (0.10, 0.03, 0.03)),
        )
        assert _ground("can on the box", scene) == "can_top"
        assert _ground("can on top of the green box", scene) == "can_top"
        assert _ground("box under the can", scene) == "box"
        assert _ground("the can on the table", scene) == "can_table"
        assert _ground("the thing in the bowl", scene) == "banana"

    def test_scene_relations_are_used_when_present(self):
        """The scene graph's own triples are authoritative, not re-derived."""
        scene = _relation_scene()
        scene.relations = [("obj_b", "next_to", "obj_c"), ("obj_c", "next_to", "obj_b")]
        assert _ground("block next to the can", scene) == "obj_b"

    def test_missing_reference_object_is_not_found(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("block next to the teapot", _relation_scene())

    def test_closest_to_another_object(self):
        assert _ground("block closest to the can", _relation_scene()) == "obj_a"
        assert _ground("the closest block to the can", _relation_scene()) == "obj_a"
        assert _ground("block farthest from the can", _relation_scene()) == "obj_b"


class TestCombinedQualifiers:
    def _blocks(self):
        return _scene(
            _obj("small_left", "block", "red", (0.5, 0.20, 0.45), (0.04, 0.04, 0.04)),
            _obj("small_right", "block", "red", (0.5, -0.20, 0.45), (0.04, 0.04, 0.04)),
            _obj("big_left", "block", "red", (0.5, 0.25, 0.45), (0.08, 0.08, 0.08)),
            _obj("blue_left", "block", "blue", (0.5, 0.35, 0.45), (0.04, 0.04, 0.04)),
        )

    def test_small_red_block_on_the_left(self):
        assert _ground("the small red block on the left", self._blocks()) == "small_left"

    def test_each_qualifier_matters(self):
        scene = self._blocks()
        assert _ground("red block on the left", scene) == "big_left"
        assert _ground("big red block", scene) == "big_left"
        assert _ground("small red block on the right", scene) == "small_right"
        assert _ground("block on the left", scene) == "blue_left"

    def test_parse_referent_keeps_every_qualifier(self):
        query = parse_referent("the small red block on the left")
        assert query.nouns == ("block",)
        assert query.colours == ("red",)
        assert set(query.selectors) == {"smallest", "left"}


class TestAmbiguityOptions:
    def test_duplicates_get_distinguishing_descriptions_that_round_trip(self):
        scene = _scene(_obj("obj_l", "block", "red", (0.5, 0.2, 0.45)),
                       _obj("obj_r", "block", "red", (0.5, -0.2, 0.45)))
        with pytest.raises(AmbiguousReference) as info:
            resolve_reference("red block", scene)
        exc = info.value
        assert exc.candidates == ("red block on the left", "red block on the right")
        assert exc.track_ids == ("obj_l", "obj_r")
        for description, track_id in zip(exc.candidates, exc.track_ids):
            assert _ground(description, scene) == track_id

    def test_three_duplicates_include_the_middle(self):
        objs = [_obj(f"obj_{i}", "can", "blue", (0.5, y, 0.45)) for i, y in enumerate((0.2, 0.0, -0.2))]
        assert describe_candidates(objs) == [
            "blue can on the left", "blue can in the middle", "blue can on the right"]

    def test_level_duplicates_fall_back_to_the_track_id(self):
        objs = [_obj("obj_1", "can", "blue", (0.5, 0.0, 0.45)), _obj("obj_2", "can", "blue", (0.6, 0.01, 0.45))]
        assert describe_candidates(objs) == ["blue can obj_1", "blue can obj_2"]
        assert _ground("blue can obj_2", _scene(*objs)) == "obj_2"

    def test_describe(self):
        assert describe(default_scene().objects["obj_001"]) == "red block"
        assert describe(_obj("x", "cube", "", (0, 0, 0))) == "cube"


class TestPronounsAndIds:
    def test_pronoun_goes_through_memory_first(self):
        memory = WorkingMemory(MemoryConfig())
        scene = default_scene()
        memory.update_scene(scene)
        memory.set_held_object("obj_002")
        assert _ground("it", scene, memory=memory) == "obj_002"
        assert _ground("that one", scene, memory=memory) == "obj_002"

    def test_unresolved_pronoun_is_a_question_when_several_are_visible(self):
        memory = WorkingMemory(MemoryConfig())
        memory.update_scene(default_scene())
        with pytest.raises(AmbiguousReference):
            resolve_reference("it", default_scene(), memory=memory)

    def test_unresolved_pronoun_with_one_object_is_that_object(self):
        scene = _scene(_obj("obj_9", "can", "blue", (0.5, 0.0, 0.45)))
        assert _ground("it", scene, memory=WorkingMemory(MemoryConfig())) == "obj_9"

    def test_referent_that_vanished_is_not_replaced_by_another_object(self):
        memory = WorkingMemory(MemoryConfig())
        memory.update_scene(_scene(_obj("obj_9", "can", "blue", (0.5, 0.0, 0.45))))
        memory.set_held_object("obj_9")
        only_other = _scene(_obj("obj_1", "block", "red", (0.5, 0.2, 0.45)))
        with pytest.raises(ObjectNotFound, match="no longer visible"):
            resolve_reference("it", only_other, memory=memory)

    def test_exact_track_id(self):
        assert _ground("obj_002", default_scene()) == "obj_002"

    def test_excluded_objects_cannot_be_meant(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("block", default_scene(), exclude_ids={"obj_001"})
        assert _ground("object on the left", default_scene(), exclude_ids={"obj_003"}) == "obj_002"

    def test_empty_phrase_is_not_found(self):
        with pytest.raises(ObjectNotFound):
            resolve_reference("  ", default_scene())


# ----------------------------------------------------------------------
# parser: referents kept whole, move-X-to-Y, place params
# ----------------------------------------------------------------------


@pytest.fixture
def parser():
    return RuleBasedIntentParser(SKILLS)


def _outcome(parser, utterance):
    try:
        intent = parser.parse(utterance)
    except UnparsedCommand:
        return "UNPARSED"
    return intent.skill, dict(intent.params)


def _approx_params(actual, expected):
    assert set(actual) == set(expected), (actual, expected)
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value), key
        else:
            assert actual[key] == value, key


class TestReferentPhraseIsKeptWhole:
    @pytest.mark.parametrize(
        "utterance,target",
        [("pick the small red block on the left", "small red block on the left"),
         ("grab the can next to the box", "can next to the box"),
         ("pick up the biggest object", "biggest object"),
         ("pick the object", "object"),
         ("pick the red one", "red one"),
         ("go pick the block on the left", "block on the left")],
    )
    def test_pick_target_keeps_qualifiers(self, parser, utterance, target):
        assert _outcome(parser, utterance) == ("pick", {"target": target})


TRANSFERS = [
    ("move the red object to the left", "red object",
     {"relation": "direction", "direction": "left", "distance": 0.10}),
    ("move the red block to the bowl", "red block", {"relation": "to", "target": "bowl"}),
    ("put the red block in the bowl", "red block", {"relation": "in", "target": "bowl"}),
    ("place the can on the green box", "can", {"relation": "on", "target": "green box"}),
    ("bring the marker next to the bowl", "marker", {"relation": "next_to", "target": "bowl"}),
    ("transfer the banana into the bin", "banana", {"relation": "in", "target": "bin"}),
    ("carry the marker over to the bin", "marker", {"relation": "to", "target": "bin"}),
    ("move the can to the left of the box", "can", {"relation": "left_of", "target": "box"}),
    ("move the can to the right of the box", "can", {"relation": "right_of", "target": "box"}),
    ("put the block in front of the box", "block", {"relation": "in_front_of", "target": "box"}),
    ("put the block behind the box", "block", {"relation": "behind", "target": "box"}),
    ("move the red block 5 cm to the left", "red block",
     {"relation": "direction", "direction": "left", "distance": 0.05}),
    ("move the red block to the left by 5 cm", "red block",
     {"relation": "direction", "direction": "left", "distance": 0.05}),
    ("move the can forward 20 mm", "can", {"relation": "direction", "direction": "forward", "distance": 0.02}),
    ("move the can back 5 cm", "can", {"relation": "direction", "direction": "back", "distance": 0.05}),
    ("shift the can right", "can", {"relation": "direction", "direction": "right", "distance": 0.10}),
    ("move the block next to the can to the bowl", "block next to the can", {"relation": "to", "target": "bowl"}),
    ("put the block on the left in the bowl", "block on the left", {"relation": "in", "target": "bowl"}),
    ("put the can in the bowl on the left", "can", {"relation": "in", "target": "bowl on the left"}),
    ("move the block on the left to the right", "block on the left",
     {"relation": "direction", "direction": "right", "distance": 0.10}),
    ("could you put the marker in the bowl please", "marker", {"relation": "in", "target": "bowl"}),
    ("put the can in the open box", "can", {"relation": "in", "target": "open box"}),
    ("put the can on the left", "can", {"relation": "direction", "direction": "left", "distance": 0.10}),
    ("take the can to the bowl", "can", {"relation": "to", "target": "bowl"}),
]


class TestMoveObjectToDestination:
    @pytest.mark.parametrize("utterance,obj,place_params", TRANSFERS, ids=[t[0] for t in TRANSFERS])
    def test_parser_returns_the_first_action_only(self, parser, utterance, obj, place_params):
        """One utterance, one intent: the pick. Never a silently dropped object."""
        assert _outcome(parser, utterance) == ("pick", {"target": obj})

    @pytest.mark.parametrize("utterance,obj,place_params", TRANSFERS, ids=[t[0] for t in TRANSFERS])
    def test_split_clauses_parse_to_pick_then_the_contract_params(self, parser, utterance, obj, place_params):
        split = split_transfer(utterance)
        assert split is not None
        assert split.object_phrase == obj
        _approx_params(split.place_params, place_params)
        pick = parser.parse(split.pick_clause)
        assert (pick.skill, pick.params) == ("pick", {"target": obj})
        place = parser.parse(split.place_clause)
        assert place.skill == "place"
        _approx_params(place.params, place_params)

    def test_audit_case_no_longer_moves_the_empty_arm(self, parser):
        """intent_parser.py:344 used to read this as move_relative{left}."""
        assert parser.parse("move the red object to the left").skill != "move_relative"
        assert expand_clauses("move the red object to the left") == [
            "pick the red object", "place it left 10 cm"]

    @pytest.mark.parametrize(
        "utterance,expected",
        [("move the gripper to the left", ("move_relative", {"direction": "left"})),
         ("move it to the left", ("move_relative", {"direction": "left"})),
         ("move your hand up", ("move_relative", {"direction": "up"})),
         ("move a bit to the left", ("move_relative", {"direction": "left"})),
         ("move 5 cm to the left", ("move_relative", {"direction": "left", "distance": 0.05})),
         ("move to the can", ("move_to", {"target": "can"})),
         ("put the can back", ("place", {})),
         ("put the can down", ("place", {})),
         ("put the can there", ("place", {})),
         ("put it in the box", ("place", {"relation": "in", "target": "box"})),
         ("move it to the bowl", ("place", {"relation": "to", "target": "bowl"})),
         ("bring it next to the can", ("place", {"relation": "next_to", "target": "can"})),
         ("drop the can next to the bowl", ("place", {"relation": "next_to", "target": "bowl"}))],
    )
    def test_not_a_transfer(self, parser, utterance, expected):
        """The robot itself, a pronoun, or no destination: not expanded."""
        skill, params = _outcome(parser, utterance)
        assert skill == expected[0]
        _approx_params(params, expected[1])
        split = split_transfer(utterance)
        assert split is None or split.pick_clause is None

    @pytest.mark.parametrize("utterance", ["nudge the can left", "move the can up", "step the block back"])
    def test_a_named_object_is_never_dropped_from_a_relative_move(self, parser, utterance):
        """Refusing beats moving the empty arm while the operator meant the object."""
        with pytest.raises(UnparsedCommand):
            parser.parse(utterance)

    VERBS = ("move", "put", "place", "bring", "transfer", "carry", "set")
    OBJECTS = ("the red block", "the can on the left", "the biggest object", "the soup can")
    DESTINATIONS = ("in the bowl", "on the green box", "next to the can", "to the bin",
                    "behind the box", "to the left of the bowl")

    @pytest.mark.parametrize("verb", VERBS)
    @pytest.mark.parametrize("obj", OBJECTS)
    @pytest.mark.parametrize("dest", DESTINATIONS)
    def test_transfer_grammar_property(self, parser, verb, obj, dest):
        """Every verb x object x destination: pick the whole object, place per contract P."""
        clauses = expand_clauses(f"{verb} {obj} {dest}")
        assert len(clauses) == 2
        pick, place = parser.parse(clauses[0]), parser.parse(clauses[1])
        assert (pick.skill, pick.params) == ("pick", {"target": obj.removeprefix("the ")})
        assert place.skill == "place"
        assert place.params["relation"] in PLACE_RELATIONS
        assert place.params["target"] == dest.rsplit(" the ", 1)[1]


class TestPlaceParams:
    @pytest.mark.parametrize(
        "utterance,params",
        [("put it on the left", {"relation": "direction", "direction": "left", "distance": 0.10}),
         ("place it to the right by 5 cm", {"relation": "direction", "direction": "right", "distance": 0.05}),
         ("place it forward", {"relation": "direction", "direction": "forward", "distance": 0.10}),
         ("place it back 10 cm", {"relation": "direction", "direction": "back", "distance": 0.10}),
         ("put it back", {}),
         ("put it right there", {}),
         ("place it to the bowl", {"relation": "to", "target": "bowl"}),
         ("place it close to the can", {"relation": "next_to", "target": "can"}),
         ("place it to the left of the can", {"relation": "left_of", "target": "can"}),
         ("place it in front of the box", {"relation": "in_front_of", "target": "box"}),
         ("place it behind the box", {"relation": "behind", "target": "box"}),
         ("place it in the bowl on the left", {"relation": "in", "target": "bowl on the left"}),
         ("place it next to the small red block", {"relation": "next_to", "target": "small red block"})],
    )
    def test_place_params_follow_the_contract(self, parser, utterance, params):
        skill, actual = _outcome(parser, utterance)
        assert skill == "place"
        _approx_params(actual, params)
        if actual.get("relation") == "direction":
            assert actual["direction"] in PLACE_DIRECTIONS
        elif actual:
            assert actual["relation"] in PLACE_RELATIONS

    def test_default_offset_is_ten_centimetres(self):
        assert DEFAULT_PLACE_OFFSET_M == pytest.approx(0.10)


# ----------------------------------------------------------------------
# assistant: clauses
# ----------------------------------------------------------------------


class TestSplitConjoined:
    @pytest.mark.parametrize(
        "utterance,clauses",
        [("pick up the block and place it in the box", ["pick up the block", "place it in the box"]),
         ("pick the can then go home", ["pick the can", "go home"]),
         ("pick the can and then put it in the bowl", ["pick the can", "put it in the bowl"]),
         ("observe, after that pick the can", ["observe,", "pick the can"]),
         ("place it and pick the marker", ["place it", "pick the marker"]),
         ("pick the block between the can and the box", ["pick the block between the can and the box"]),
         ("what do you see", ["what do you see"])],
    )
    def test_split(self, utterance, clauses):
        assert split_conjoined(utterance) == clauses

    def test_expand_combines_conjunctions_and_transfers(self):
        assert expand_clauses("move the can to the bowl and then go home") == [
            "pick the can", "place it to the bowl", "go home"]


def _outcome_obj(clause, ok=True, **fields):
    status = SkillStatus.SUCCESS if ok else SkillStatus.FAILED
    outcome = CommandOutcome(utterance=clause, result=SkillResult(skill_name="x", status=status))
    outcome.duration_s = 1.0
    for key, value in fields.items():
        setattr(outcome, key, value)
    return outcome


class TestRunClauses:
    def test_runs_in_order_and_sums_durations(self):
        seen = []
        out = run_clauses(["a", "b", "c"], lambda c: (seen.append(c), _outcome_obj(c))[1])
        assert seen == ["a", "b", "c"] and out.utterance == "c" and out.duration_s == 3.0

    def test_stops_at_the_first_failed_clause_and_reports_what_did_not_run(self):
        seen = []
        out = run_clauses(["a", "b", "c"], lambda c: (seen.append(c), _outcome_obj(c, ok=c != "b"))[1])
        assert seen == ["a", "b"]
        assert out.utterance == "b" and not out.ok
        assert out.remaining_clauses == ("c",)
        assert out.duration_s == 2.0


class FakeExecutor:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    backend_name = "fake"

    def supports(self, skill_name):
        return True

    def execute(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        status = SkillStatus.INFEASIBLE if skill_name in self.fail_on else SkillStatus.SUCCESS
        return SkillResult(skill_name=skill_name, status=status, message="scripted")


def _assistant(executor, memory=None, config=None):
    """An Assistant wired to a real parser and planner, without a runtime."""
    memory = memory or WorkingMemory(MemoryConfig())
    config = config or load_config(overrides={})
    assistant = Assistant.__new__(Assistant)
    assistant.config = config
    assistant.runtime = SimpleNamespace(memory=memory, robot=SimpleNamespace(get_state=lambda: None))
    assistant.parser = RuleBasedIntentParser(SKILLS)
    assistant.executors = {"classical": executor}
    assistant.planner = TaskPlanner(assistant.parser, assistant.executors, memory, config)
    return assistant


class TestAssistantCommand:
    def test_move_x_to_y_runs_pick_then_place(self):
        executor = FakeExecutor()
        outcome = _assistant(executor).command("move the red block to the bowl")
        assert outcome.ok
        assert executor.calls == [("pick", {"target": "red block"}),
                                  ("place", {"relation": "to", "target": "bowl"})]

    def test_move_object_left_places_it_relative_to_its_pick_origin(self):
        executor = FakeExecutor()
        _assistant(executor).command("move the red object to the left")
        assert executor.calls[0] == ("pick", {"target": "red object"})
        assert executor.calls[1][0] == "place"
        _approx_params(executor.calls[1][1], {"relation": "direction", "direction": "left", "distance": 0.10})

    def test_a_failed_pick_never_places(self):
        executor = FakeExecutor(fail_on={"pick"})
        outcome = _assistant(executor).command("put the red block in the bowl")
        assert not outcome.ok
        assert [c[0] for c in executor.calls] == ["pick"]
        assert outcome.remaining_clauses == ("place it in the bowl",)

    def test_single_clause_is_unchanged(self):
        executor = FakeExecutor()
        _assistant(executor).command("pick the can")
        assert executor.calls == [("pick", {"target": "can"})]


# ----------------------------------------------------------------------
# planner recovery policy
# ----------------------------------------------------------------------


class RaisingExecutor:
    backend_name = "raising"

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def supports(self, skill_name):
        return True

    def execute(self, skill_name, params):
        self.calls += 1
        raise self.error


class ResultExecutor:
    backend_name = "result"

    def __init__(self, message, status=SkillStatus.FAILED, data=None):
        self.message, self.status, self.data = message, status, data or {}
        self.calls = 0

    def supports(self, skill_name):
        return True

    def execute(self, skill_name, params):
        self.calls += 1
        return SkillResult(skill_name=skill_name, status=self.status, message=self.message, data=dict(self.data))


def _planner(executor, memory=None):
    memory = memory or WorkingMemory(MemoryConfig())
    return TaskPlanner(RuleBasedIntentParser(SKILLS), {"classical": executor}, memory, load_config(overrides={}))


def _visited(outcome):
    return [entry["to"] for entry in outcome.state_trace]


class TestRecoveryPolicy:
    def test_config_retries_more_than_once(self):
        """Otherwise "attempts == 1" below would prove nothing."""
        assert load_config(overrides={}).motion.max_replan_attempts > 1

    def test_raised_ambiguity_clarifies_once_with_the_exception_options(self):
        error = AmbiguousReference("two cans", phrase="can", candidates=("blue can on the left", "blue can on the right"),
                                   track_ids=("obj_1", "obj_2"))
        executor = RaisingExecutor(error)
        planner = _planner(executor)
        outcome = planner.handle("pick the can")
        assert executor.calls == 1 and outcome.attempts == 1
        assert outcome.needs_clarification
        assert outcome.clarification_options == ("blue can on the left", "blue can on the right")
        assert outcome.clarification_track_ids == ("obj_1", "obj_2")
        assert outcome.clarification_phrase == "can"
        assert "clarify" in _visited(outcome)
        assert planner.machine.state is State.WAIT_FOR_COMMAND

    def test_ambiguity_reported_as_a_failed_result_is_not_retried(self):
        """Skill.execute converts the exception; the audit saw 3 attempts, no question."""
        memory = WorkingMemory(MemoryConfig())
        memory.update_scene(_scene(_obj("obj_1", "can", "blue", (0.5, 0.2, 0.45)),
                                   _obj("obj_2", "can", "blue", (0.5, -0.2, 0.45))))
        executor = ResultExecutor("AmbiguousReference: 'can' matches 2 objects; say which one")
        outcome = _planner(executor, memory).handle("pick the can")
        assert executor.calls == 1 and outcome.attempts == 1
        assert outcome.needs_clarification
        assert outcome.clarification_options == ("blue can on the left", "blue can on the right")
        assert outcome.clarification_track_ids == ("obj_1", "obj_2")
        assert "clarify" in _visited(outcome)

    def test_ambiguity_declared_in_result_data(self):
        executor = ResultExecutor("could not choose", data={
            "error_type": "AmbiguousReference", "candidates": ["red block", "blue can"],
            "track_ids": ["a", "b"]})
        outcome = _planner(executor).handle("pick the object")
        assert executor.calls == 1
        assert outcome.clarification_options == ("red block", "blue can")
        assert outcome.clarification_track_ids == ("a", "b")

    @pytest.mark.parametrize("executor_factory", [
        lambda: RaisingExecutor(ObjectNotFound("cannot find 'teapot'")),
        lambda: ResultExecutor("ObjectNotFound: cannot find 'teapot'; currently visible: red block"),
    ])
    def test_object_not_found_is_not_retried(self, executor_factory):
        executor = executor_factory()
        outcome = _planner(executor).handle("pick up the teapot")
        assert executor.calls == 1 and outcome.attempts == 1
        assert not outcome.ok and not outcome.needs_clarification
        assert "failed" in _visited(outcome) and "replan" not in _visited(outcome)

    def test_safety_violation_ends_aborted_and_is_never_retried(self):
        executor = RaisingExecutor(SafetyViolation("outside workspace"))
        planner = _planner(executor)
        outcome = planner.handle("pick the can")
        assert executor.calls == 1
        assert "aborted" in _visited(outcome)
        assert "safety violation" in outcome.message
        assert planner.machine.state is State.WAIT_FOR_COMMAND

    def test_planning_errors_are_still_retried(self):
        executor = RaisingExecutor(PlanningError("no path"))
        outcome = _planner(executor).handle("pick the can")
        assert executor.calls == load_config(overrides={}).motion.max_replan_attempts
        assert "replan" in _visited(outcome)

    def test_plain_failures_are_still_retried(self):
        executor = ResultExecutor("grasp slipped")
        _planner(executor).handle("pick the can")
        assert executor.calls == load_config(overrides={}).motion.max_replan_attempts

    def test_execute_may_go_to_clarify(self):
        assert State.CLARIFY in TRANSITIONS[State.EXECUTE]
        assert TRANSITIONS[State.CLARIFY] == (State.WAIT_FOR_COMMAND,)

    def test_a_real_skill_that_cannot_choose_leads_to_a_question(self):
        """Behavioural: a real Skill through the real ClassicalExecutor and planner."""
        from mfw.skills.base import Skill, SkillContext
        from mfw.skills.registry import ClassicalExecutor, SkillRegistry

        class GroundingPick(Skill):
            skill_name = "pick"

            def _run(self, params):
                target = resolve_reference(params["target"], self.ctx.memory.current_scene, memory=self.ctx.memory)
                return self._ok("picked", track_id=target.track_id)

        memory = WorkingMemory(MemoryConfig())
        memory.update_scene(default_scene())
        config = load_config(overrides={})
        context = SkillContext(sim=None, robot=None, vision=None, planner=None, controller=None,
                               grasp_scorer=None, memory=memory, config=config)
        executor = ClassicalExecutor(SkillRegistry(context, (GroundingPick,)))
        planner = TaskPlanner(RuleBasedIntentParser(("pick",)), {"classical": executor}, memory, config)

        asked = planner.handle("pick the object")
        assert asked.needs_clarification and asked.attempts == 1
        assert asked.clarification_options == ("green box", "blue can", "red block")

        picked = planner.handle("pick the red object")
        assert picked.ok and picked.result.data["track_id"] == "obj_001"


# ----------------------------------------------------------------------
# clarification dialogue
# ----------------------------------------------------------------------

OPTIONS = ("green box", "blue can", "red block")
IDS = ("obj_003", "obj_002", "obj_001")


class TestInterpretClarification:
    @pytest.mark.parametrize(
        "answer,index",
        [("the red one", 2), ("red", 2), ("the can", 1), ("cube", 2), ("the green box", 0),
         ("the first one", 0), ("second", 1), ("3", 2), ("the last one", 2), ("number two", 1),
         ("Blue can.", 1)],
    )
    def test_answers_without_a_scene(self, answer, index):
        assert interpret_clarification(answer, OPTIONS).index == index

    @pytest.mark.parametrize("answer", ["cancel", "never mind", "Never mind.", "forget it", "none of them", "no"])
    def test_cancel(self, answer):
        assert interpret_clarification(answer, OPTIONS).kind == "cancel"

    @pytest.mark.parametrize("answer", ["banana", "the fifth one", "one", "", "hmm"])
    def test_unknown(self, answer):
        assert interpret_clarification(answer, OPTIONS).kind == "unknown"

    @pytest.mark.parametrize(
        "answer,index", [("the one on the left", 0), ("the rightmost one", 2), ("the biggest", 0),
                         ("the nearest one", 2), ("the tall one", 1)],
    )
    def test_answers_grounded_against_the_scene(self, answer, index):
        choice = interpret_clarification(answer, OPTIONS, track_ids=IDS, scene=default_scene())
        assert (choice.kind, choice.index) == ("choice", index)

    def test_question_text(self):
        assert clarification_question(("red block", "blue can", "green box")) == (
            "Which one do you mean: the red block, the blue can or the green box?")


class TestClarifiedCommand:
    def test_substitutes_the_phrase_with_the_track_id(self):
        outcome = _outcome_obj("pick the object", ok=False, clarification_phrase="object",
                               params={"target": "object"}, skill="pick")
        assert clarified_command(outcome, "obj_001") == "pick obj_001"

    def test_remaining_clauses_are_appended(self):
        outcome = _outcome_obj("pick the red object", ok=False, clarification_phrase="red object",
                               remaining_clauses=("place it in the bowl",), skill="pick")
        assert clarified_command(outcome, "obj_001") == "pick obj_001 then place it in the bowl"

    def test_place_destination_is_substituted_last_occurrence(self):
        assert substitute_referent("place the can next to the can", "can", "obj_9") == "place the can next to obj_9"

    def test_paraphrased_phrase_falls_back_to_the_intent(self):
        outcome = _outcome_obj("put it beside that fizzy thing", ok=False, clarification_phrase="soda",
                               skill="place", params={"relation": "next_to", "target": "soda"})
        assert clarified_command(outcome, "obj_4") == "place it next to obj_4"

    def test_rerun_parses_to_the_chosen_object(self, parser):
        assert parser.parse("pick obj_001").params == {"target": "obj_001"}
        assert parser.parse("place it next to obj_004").params == {"relation": "next_to", "target": "obj_004"}


def _ambiguous_outcome(clause="pick the object"):
    return _outcome_obj(clause, ok=False, needs_clarification=True, clarification_options=OPTIONS,
                        clarification_track_ids=IDS, clarification_phrase="object",
                        params={"target": "object"}, skill="pick")


class TestRunWithClarification:
    def _command(self, script):
        calls = []

        def command(text):
            calls.append(text)
            return script[len(calls) - 1] if len(calls) <= len(script) else _outcome_obj(text)

        return command, calls

    def test_answer_reruns_the_command_with_the_chosen_object(self):
        command, calls = self._command([_ambiguous_outcome()])
        questions, notes = [], []
        result = run_with_clarification(command, "pick the object",
                                         lambda q: (questions.append(q), "the red one")[1], notify=notes.append)
        assert calls == ["pick the object", "pick obj_001"]
        assert result.ok
        assert questions == ["Which one do you mean: the green box, the blue can or the red block?"]
        assert notes == ["OK, the red block."]

    @pytest.mark.parametrize("answer", ["cancel", "never mind", None])
    def test_cancel_runs_nothing_more(self, answer):
        command, calls = self._command([_ambiguous_outcome()])
        result = run_with_clarification(command, "pick the object", lambda q: answer)
        assert calls == ["pick the object"] and result.needs_clarification

    def test_unrecognised_answer_is_asked_again_never_guessed(self):
        command, calls = self._command([_ambiguous_outcome()])
        answers = iter(["banana", "the second one"])
        questions = []
        run_with_clarification(command, "pick the object", lambda q: (questions.append(q), next(answers))[1])
        assert calls == ["pick the object", "pick obj_002"]
        assert questions[1].startswith("Sorry, I did not catch that.")

    def test_gives_up_after_the_question_budget(self):
        command, calls = self._command([_ambiguous_outcome()])
        asked = []
        run_with_clarification(command, "pick the object", lambda q: (asked.append(q), "banana")[1],
                               max_questions=2)
        assert calls == ["pick the object"] and len(asked) == 2

    def test_scene_grounded_answer(self):
        command, calls = self._command([_ambiguous_outcome()])
        run_with_clarification(command, "pick the object", lambda q: "the one on the left",
                               scene_provider=default_scene)
        assert calls[-1] == "pick obj_003"

    def test_move_command_resumes_with_the_place(self):
        """"move the object to the bowl": the pick asked; the place still runs after."""
        executor = FakeExecutor()
        assistant = _assistant(executor)
        first = _ambiguous_outcome()
        first.remaining_clauses = ("place it to the bowl",)
        calls = []

        def command(text):
            calls.append(text)
            return first if len(calls) == 1 else assistant.command(text)

        result = run_with_clarification(command, "move the object to the bowl", lambda q: "the can")
        assert calls[1] == "pick obj_002 then place it to the bowl"
        assert result.ok
        assert executor.calls == [("pick", {"target": "obj_002"}), ("place", {"relation": "to", "target": "bowl"})]


# ----------------------------------------------------------------------
# voice confirmation gate
# ----------------------------------------------------------------------


class TestConfirmationGate:
    @pytest.mark.parametrize("skill", sorted(set(SKILLS) - NON_MOTION_SKILLS))
    def test_every_motion_skill_is_confirmed(self, skill):
        assert requires_confirmation(skill)

    @pytest.mark.parametrize("skill", sorted(NON_MOTION_SKILLS) + [None])
    def test_nothing_that_moves_nothing_is_confirmed(self, skill):
        assert not requires_confirmation(skill)

    def test_an_unknown_future_skill_is_confirmed(self):
        assert requires_confirmation("push")

    @pytest.mark.parametrize(
        "utterance",
        # Every phrasing the audit found slipping past the old keyword list.
        ["get the can", "fetch the can", "set it on the box", "position it on the box", "shift left",
         "nudge left 5 cm", "step back", "turn the wrist 45 degrees", "twist the gripper",
         "survey the table", "look at the can", "return home", "reset arm", "home position",
         "squeeze the gripper", "clamp", "let go", "move the red object to the left",
         "observe and then pick the can"],
    )
    def test_motion_phrasings_the_keyword_list_missed(self, utterance):
        assert _assistant(FakeExecutor()).needs_confirmation(utterance), utterance

    @pytest.mark.parametrize("utterance", ["what do you see", "observe", "wait", "hold on", "stop",
                                           "emergency stop", "recite poetry"])
    def test_no_confirmation_when_nothing_would_move(self, utterance):
        assert not _assistant(FakeExecutor()).needs_confirmation(utterance)

    def test_gate_executes_nothing(self):
        executor = FakeExecutor()
        _assistant(executor).needs_confirmation("pick the can and put it in the bowl")
        assert executor.calls == []

    @pytest.mark.parametrize(
        "reply,verdict",
        [("yes", True), ("Yeah.", True), ("okay", True), ("ok go ahead", True), ("that's right", True),
         ("sure", True), ("no", False), ("nope", False), ("cancel", False), ("don't", False),
         ("stop", False), ("never mind", False),
         # substring traps of the old matcher
         ("I know", None), ("nothing", None), ("yesterday", None), ("notice", None), ("snowy", None),
         ("yes no", None), ("", None), ("hmm", None)],
    )
    def test_reply_matching_is_word_based(self, reply, verdict):
        assert classify_confirmation(reply) is verdict


# ----------------------------------------------------------------------
# the CLI wiring (scripts/run_assistant.py) -- helpers only, no runtime
# ----------------------------------------------------------------------


def _load_run_assistant():
    spec = importlib.util.spec_from_file_location("run_assistant_under_test", REPO_ROOT / "scripts" / "run_assistant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunAssistantWiring:
    def test_keyword_gate_is_gone(self):
        source = (REPO_ROOT / "scripts" / "run_assistant.py").read_text(encoding="utf-8")
        assert "MOTION_WORDS" not in source
        assert "assistant.needs_confirmation(text)" in source
        assert "classify_confirmation(heard)" in source
        assert source.count("command_with_clarification(") >= 2  # voice and interactive

    def test_listen_for_reply_skips_silence(self):
        module = _load_run_assistant()
        replies = iter([None, SimpleNamespace(text="  "), SimpleNamespace(text="the red one")])
        recognizer = SimpleNamespace(listen=lambda: next(replies))
        assert module._listen_for_reply(recognizer) == "the red one"

    def test_typed_ask_returns_none_at_end_of_input(self, monkeypatch):
        module = _load_run_assistant()
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert module._typed_ask("Which one?") is None
        monkeypatch.setattr("sys.stdin", io.StringIO("the can\n"))
        assert module._typed_ask("Which one?") == "the can"


# ----------------------------------------------------------------------
# ObjectNotFound says why, and lists everything in view
# ----------------------------------------------------------------------


def _not_found(phrase, scene, **kwargs):
    with pytest.raises(ObjectNotFound) as info:
        resolve_reference(phrase, scene, **kwargs)
    return str(info.value), info.value


class TestNotFoundExplainsTheMismatch:
    """Isaac run_c: "pick the blue cube" -> "cannot find 'blue cube'; currently visible: red block".

    The reason is now stated (what the class word matched, and why it was
    rejected) and the visible list is always the whole scene, never what a
    filter left.
    """

    ALL = "currently visible: red block, blue can, green box"

    def test_a_colour_that_eliminates_the_only_match_names_it(self):
        message, exc = _not_found("blue cube", default_scene())
        assert message == f"cannot find 'blue cube': the only block is the red block; {self.ALL}"
        assert exc.visible == ("red block", "blue can", "green box")

    def test_several_of_the_class_are_listed_none_of_that_colour(self):
        scene = _scene(*default_scene().objects.values(),
                       _obj("obj_004", "block", "yellow", (0.40, 0.30, 0.45)))
        message, exc = _not_found("the blue block", scene)
        assert "the blocks are the red block and the yellow block, none of them blue" in message
        assert exc.visible == ("red block", "blue can", "green box", "yellow block")

    def test_a_class_absent_from_view(self):
        message, _ = _not_found("the bowl", default_scene())
        assert message == f"cannot find 'the bowl': there is no bowl in view; {self.ALL}"

    def test_an_unknown_word(self):
        message, _ = _not_found("purple dragon", default_scene())
        assert message == f"cannot find 'purple dragon': nothing in view is a dragon; {self.ALL}"

    def test_a_colour_with_no_class_word(self):
        message, _ = _not_found("the yellow one", default_scene())
        assert message == f"cannot find 'the yellow one': nothing in view is yellow; {self.ALL}"

    def test_a_relation_names_what_matched_and_where_it_is_not(self):
        message, _ = _not_found("block to the left of the can", _relation_scene())
        assert "the red block on the left and the red block on the right are not to the left of " \
               "the blue can" in message
        assert message.endswith("currently visible: 2 x red block, blue can")

    def test_the_excluded_held_object_is_named_as_the_one_being_moved(self):
        message, exc = _not_found("the can", default_scene(), exclude_ids=("obj_002",))
        assert "the only can in view is the blue can, the object being moved" in message
        # Still the whole scene, the held can included.
        assert exc.visible == ("red block", "blue can", "green box")

    def test_the_visible_list_is_never_a_filters_survivors(self):
        """Offered options restrict the choice, never what is reported as visible."""
        message, exc = _not_found("the blue one", default_scene(), allowed_ids=("obj_001", "obj_003"))
        assert message.endswith(self.ALL)
        assert exc.visible == ("red block", "blue can", "green box")

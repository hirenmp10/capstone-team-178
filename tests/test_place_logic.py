"""Place: destination synthesis, aim correction and honest success -- no Isaac.

What these tests pin, each from a measured failure:

* "place it next to the green box" released at y=0.461, the far edge of
  reach, because ``next_to`` was always +Y and unclamped; the can was pressed
  into the table (release z 0.45 for a 125 mm can gripped at its middle) and
  the skill still said *success*, "settled 350 mm from the target" (Isaac,
  events 20260924_083548_0ab54f). :class:`TestNextToSideChoice`,
  :class:`TestReleaseHeight`, :class:`TestHonestyThroughPlace`.
* The aim correction was measured in the carry orientation and applied after
  the transit re-oriented the wrist: a 25 mm offset gave 19/35/50 mm landing
  error at 45/90/180 deg of yaw (audit probe P4). :class:`TestAimWithRotatedWrist`.
* On the hardware lane Place fell back to an unchecked straight line when the
  planner refused a transit: 12 of 108 bin crossings drove a held cube 37-41
  mm into the bin (re-review, probe_fallback.py). :class:`TestHardwareTransitFallback`
  replays that sweep with the real ``JointSpacePlanner``.

The fake world below is the only physics here: an object held by the fake
gripper is rigidly attached at a fixed offset *in the gripper frame* and
follows the TCP through every orientation; on release it drops straight down
onto whatever is under it. That is exactly the property the aim correction
has to respect, and nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

import importlib.util as _importlib_util

#: The hardware lane (mfw/hardware, jetson/, tests/test_hardware_*.py) is not part of
#: every checkout -- the simulation-only branch omits it. Tests of the hardware lane
#: skip there instead of breaking collection of the simulation tests in this file.
HARDWARE_LANE_PRESENT = all(
    _importlib_util.find_spec(m) is not None
    for m in (
        "mfw.hardware.kinematics",
        "jetson.robot_server",
        "jetson.detector_service",
        "tests.test_hardware_e2e",
    )
)
needs_hardware_lane = pytest.mark.skipif(
    not HARDWARE_LANE_PRESENT,
    reason="hardware lane (mfw.hardware.kinematics, jetson.detector_service, tests.test_hardware_e2e) is not in this checkout",
)

from mfw.config.schema import MemoryConfig, load_config
from mfw.core.types import (
    BoundingBox3D,
    Frame,
    GripperState,
    JointState,
    ObjectHypothesis,
    Pose,
    RobotState,
    SceneGraph,
    SkillStatus,
)
from mfw.language.intent_parser import PLACE_DIRECTIONS as PARSER_DIRECTIONS
from mfw.language.intent_parser import PLACE_RELATIONS as PARSER_RELATIONS
from mfw.language.intent_parser import RuleBasedIntentParser
from mfw.memory.working_memory import WorkingMemory
from mfw.planner.task_planner import TaskPlanner
from mfw.skills.base import SkillContext
from mfw.skills.primitives import (
    DEFAULT_PLACE_TOLERANCE_M,
    NEXT_TO_GAP_M,
    PLACE_DIRECTIONS,
    PLACE_RELATIONS,
    HeldGrasp,
    Place,
    PlaceTarget,
    aim_shift_xy,
    assess_placement,
    footprint_gap,
    half_extent_along,
    offset_in_gripper,
    refine_place_target,
    relation_for_to,
    release_tcp_position,
    synthesise_place_target,
)
from mfw.skills.registry import ClassicalExecutor, SkillRegistry
from mfw.utils import transforms as tf
from mfw.vision.scene_graph import compute_relations

TABLE_TOP = 0.40
WS_MIN = (0.12, -0.55, 0.0)
WS_MAX = (0.85, 0.55, 0.85)
CLEARANCE = 0.02
TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# scene helpers
# ----------------------------------------------------------------------


def _yaw_quat(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return tf.matrix_to_quat(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


def obj(track_id: str, label: str, colour: str, centre, extents, yaw: float = 0.0,
        step: int = 0) -> ObjectHypothesis:
    pose = Pose(np.asarray(centre, dtype=float), _yaw_quat(yaw), Frame.WORLD)
    return ObjectHypothesis(
        track_id=track_id, label=label, pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.asarray(extents, dtype=float)),
        confidence=0.9, num_points=500, last_seen_sim_time=0.0, last_seen_step=step,
        attributes={"color": colour},
    )


def scene_of(*objects: ObjectHypothesis, step: int = 0) -> SceneGraph:
    return SceneGraph(objects={o.track_id: o for o in objects}, sim_time=0.0, step_index=step,
                      relations=compute_relations({o.track_id: o for o in objects}))


def isaac_default_scene() -> SceneGraph:
    """The perceived default scene of the 2026-09-24 Isaac run (run_a_default.txt)."""
    return scene_of(
        obj("obj_001", "block", "red", (0.4598, -0.1504, 0.4334), (0.0577, 0.0499, 0.0491)),
        obj("obj_002", "can", "blue", (0.5157, 0.1160, 0.4722), (0.0699, 0.0570, 0.1250)),
        obj("obj_003", "box", "green", (0.6575, 0.3164, 0.4445), (0.1459, 0.1233, 0.0741)),
    )


def synth(relation: str, scene: SceneGraph, held: str = "obj_002", dest: str | None = None,
          **kwargs: Any):
    held_obj = scene.get(held)
    return synthesise_place_target(
        relation, scene=scene, held_id=held, support_height=TABLE_TOP,
        workspace_min=WS_MIN, workspace_max=WS_MAX, clearance=CLEARANCE,
        destination=scene.get(dest) if dest else None,
        held_extents=None if held_obj is None else held_obj.bbox.extents,
        **kwargs,
    )


# ----------------------------------------------------------------------
# the fake world (shared with tests/test_skills_grounding.py)
# ----------------------------------------------------------------------


@dataclass
class Body:
    track_id: str
    label: str
    colour: str
    centre: np.ndarray
    extents: np.ndarray
    yaw: float = 0.0
    visible: bool = True


class FakeWorld:
    """Objects on a table, one of which may be rigidly held at a gripper-frame offset."""

    def __init__(self, bodies: list[Body], support: float = TABLE_TOP) -> None:
        self.bodies = {b.track_id: b for b in bodies}
        self.support = support
        self.held: str | None = None
        self.offset_g = np.zeros(3)
        self.tcp = Pose(np.array([0.4, 0.0, 0.7]), TOP_DOWN.copy(), Frame.WORLD)
        self.slip_on_first_move = False
        self.hide_after_release: set[str] = set()
        self.releases: list[tuple[str, np.ndarray]] = []
        self.moves = 0

    def attach(self, track_id: str, offset_g, tcp_quat) -> None:
        """Hold ``track_id`` at ``offset_g`` (gripper frame) with the hand at ``tcp_quat``."""
        body = self.bodies[track_id]
        self.held = track_id
        self.offset_g = np.asarray(offset_g, dtype=float)
        rotation = tf.quat_to_matrix(tcp_quat)
        self.tcp = Pose(body.centre - rotation @ self.offset_g, np.asarray(tcp_quat, float), Frame.WORLD)

    def move_tcp(self, pose: Pose) -> None:
        self.moves += 1
        self.tcp = pose
        if self.held is not None and self.slip_on_first_move:
            # The object slips out as the transit starts: it falls where it is.
            self.slip_on_first_move = False
            self._drop(self.held)
            return
        if self.held is not None:
            body = self.bodies[self.held]
            body.centre = np.asarray(pose.position, float) + pose.rotation_matrix() @ self.offset_g

    def open(self) -> None:
        if self.held is not None:
            self._drop(self.held)

    def _drop(self, track_id: str) -> None:
        body = self.bodies[track_id]
        surface = self.support
        for other in self.bodies.values():
            if other.track_id == track_id:
                continue
            probe = obj(other.track_id, other.label, other.colour, other.centre, other.extents, other.yaw)
            if footprint_gap(probe, body.centre[:2]) <= 0.0:
                surface = max(surface, float(other.centre[2] + other.extents[2] / 2.0))
        body.centre = np.array([body.centre[0], body.centre[1], surface + body.extents[2] / 2.0])
        self.releases.append((track_id, body.centre.copy()))
        if self.held == track_id:
            self.held = None
        if track_id in self.hide_after_release:
            body.visible = False

    def scene(self, step: int) -> SceneGraph:
        return scene_of(*[obj(b.track_id, b.label, b.colour, b.centre, b.extents, b.yaw, step=step)
                          for b in self.bodies.values() if b.visible], step=step)


class FakeSim:
    def __init__(self) -> None:
        self.step_index = 0

    def render_step(self, n: int = 1) -> None:
        self.step_index += int(n)

    def step(self, n: int = 1) -> None:
        self.step_index += int(n)


class FakeVision:
    def __init__(self, world: FakeWorld, sim: FakeSim) -> None:
        self.world, self.sim = world, sim
        self.last: SceneGraph | None = None
        self.fail_next: Callable[[], bool] | None = None

    def observe(self) -> SceneGraph:
        if self.fail_next is not None and self.fail_next():
            from mfw.core.errors import PerceptionError
            raise PerceptionError("injected: detector hiccup")
        self.sim.step_index += 1
        self.last = self.world.scene(self.sim.step_index)
        return self.last

    def require_fresh_scene(self) -> SceneGraph:
        return self.last if self.last is not None else self.observe()

    def last_scene_graph(self) -> SceneGraph | None:
        return self.last


@dataclass
class FakeTrajectory:
    goal: Pose
    kind: str


class FakeRobot:
    def __init__(self, world: FakeWorld, reachable: Callable[[Pose], bool] = lambda p: True) -> None:
        self.world = world
        self.reachable = reachable

    def tcp_pose(self) -> Pose:
        return self.world.tcp

    def get_state(self) -> RobotState:
        return RobotState(
            joint_state=JointState(positions=np.zeros(7), names=tuple(f"j{i}" for i in range(7))),
            tcp_pose=self.world.tcp,
            gripper=GripperState(0.04, 0.04, False, self.world.held is not None),
            sim_time=0.0, step_index=0,
        )

    def inverse_kinematics(self, pose: Pose, seed=None):
        return np.zeros(7) if self.reachable(pose) else None


class FakePlanner:
    """Every goal is reachable along the requested kind of path (overridable)."""

    def __init__(self) -> None:
        self.excluded: set[str] = set()
        self.plans: list[FakeTrajectory] = []
        self.transit_ok = True

    def exclude_from_collision(self, track_id: str) -> None:
        self.excluded.add(track_id)

    def include_in_collision(self, track_id: str) -> None:
        self.excluded.discard(track_id)

    def plan_with_retries(self, start, goal, scene):
        if not self.transit_ok and goal.position[2] > 0.0 and len(self.plans) == 0:
            return None
        plan = FakeTrajectory(goal, "planned")
        self.plans.append(plan)
        return plan

    def plan_cartesian_line(self, start, goal, scene):
        plan = FakeTrajectory(goal, "cartesian")
        self.plans.append(plan)
        return plan

    def plan_to_joint(self, start, goal, scene):
        return None


class FakeController:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        self.followed: list[FakeTrajectory] = []
        self.opened = 0
        self.closed = 0

    def follow_trajectory(self, trajectory, on_step=None) -> bool:
        self.followed.append(trajectory)
        self.world.move_tcp(trajectory.goal)
        return True

    def maintain_grasp(self) -> None:
        pass

    def open_gripper_blocking(self) -> float:
        self.opened += 1
        self.world.open()
        return 0.08

    def close_gripper_blocking(self) -> float:
        self.closed += 1
        return 0.02


@dataclass
class Rig:
    world: FakeWorld
    sim: FakeSim
    vision: FakeVision
    robot: FakeRobot
    planner: FakePlanner
    controller: FakeController
    memory: WorkingMemory
    context: SkillContext
    config: Any

    def place(self, params: dict[str, Any] | None = None):
        return Place(self.context).execute(params or {})


def make_rig(bodies: list[Body], *, held: str | None = None, offset_g=(0.0, 0.0, 0.0),
             carry_quat=TOP_DOWN, gripper_feedback: bool = True, record_grasp: bool = False,
             **config_overrides: Any) -> Rig:
    overrides: dict[str, Any] = {"robot": {"gripper_feedback": gripper_feedback}}
    for key, value in config_overrides.items():
        overrides.setdefault(key, {}).update(value)
    config = load_config(overrides=overrides)
    world = FakeWorld(bodies)
    sim = FakeSim()
    vision = FakeVision(world, sim)
    robot = FakeRobot(world)
    planner = FakePlanner()
    controller = FakeController(world)
    memory = WorkingMemory(MemoryConfig())
    memory.update_scene(vision.observe())
    context = SkillContext(sim=sim, robot=robot, vision=vision, planner=planner,
                           controller=controller, grasp_scorer=None, memory=memory,
                           config=config, support_height=TABLE_TOP)
    if held is not None:
        memory.set_held_object(held)  # snapshots the resting scene: the pick origin
        world.attach(held, offset_g, carry_quat)
        if record_grasp:
            body = world.bodies[held]
            context.held_grasp = HeldGrasp(held, tuple(float(v) for v in offset_g),
                                           tuple(float(v) for v in carry_quat),
                                           tuple(float(v) for v in body.extents))
    return Rig(world, sim, vision, robot, planner, controller, memory, context, config)


def default_bodies() -> list[Body]:
    """The Isaac default scene as perceived on 2026-09-24."""
    return [
        Body("obj_001", "block", "red", np.array([0.4598, -0.1504, 0.4334]), np.array([0.0577, 0.0499, 0.0491])),
        Body("obj_002", "can", "blue", np.array([0.5157, 0.1160, 0.4625]), np.array([0.0699, 0.0570, 0.1250])),
        Body("obj_003", "box", "green", np.array([0.6575, 0.3164, 0.4371]), np.array([0.1459, 0.1233, 0.0741])),
    ]


def with_bowl(bodies: list[Body]) -> list[Body]:
    return bodies + [Body("obj_004", "bowl", "white", np.array([0.55, -0.30, 0.43]),
                          np.array([0.16, 0.16, 0.06]))]


def rz(deg: float) -> np.ndarray:
    return tf.quat_to_matrix(_yaw_quat(math.radians(deg)))


def yawed_top_down(deg: float) -> np.ndarray:
    """The top-down hand turned ``deg`` about world Z (a carry after a turned grasp)."""
    return tf.matrix_to_quat(rz(deg) @ tf.quat_to_matrix(TOP_DOWN))


# ----------------------------------------------------------------------
# contract
# ----------------------------------------------------------------------


class TestContract:
    def test_relations_and_directions_match_the_parser(self):
        assert PLACE_RELATIONS == PARSER_RELATIONS
        assert set(PLACE_DIRECTIONS) == set(PARSER_DIRECTIONS)

    @pytest.mark.parametrize("params,error", [
        ({"relation": "direction", "direction": "up"}, "unknown direction"),
        ({"relation": "direction", "direction": "left", "distance": 0.9}, "out of range"),
        ({"relation": "direction", "direction": "left", "distance": "far"}, "not a number"),
        ({"relation": "under", "target": "box"}, "unknown place relation"),
        ({"relation": "next_to"}, "needs a destination"),
    ])
    def test_bad_parameters_are_refused_before_any_motion(self, params, error):
        rig = make_rig(default_bodies(), held="obj_002")
        result = rig.place(params)
        assert result.status is SkillStatus.INFEASIBLE and error in result.message
        assert rig.controller.followed == [] and rig.world.held == "obj_002"

    def test_every_parser_relation_is_accepted(self):
        scene = isaac_default_scene()
        for relation in PARSER_RELATIONS:
            target, reason = synth(relation, scene, dest="obj_003")
            assert target is not None, (relation, reason)


# ----------------------------------------------------------------------
# destination synthesis
# ----------------------------------------------------------------------


class TestNextToSideChoice:
    def test_isaac_case_chooses_the_side_facing_the_robot(self):
        """The 350 mm run: the old rule released at y=0.458 (+Y, far edge)."""
        scene = isaac_default_scene()
        target, _ = synth("next_to", scene, dest="obj_003")
        box = scene.get("obj_003")
        assert target.relation == "next_to" and target.description == "next to the green box"
        xy = np.array(target.object_xy)
        # The -X side: between the box and the robot base, same Y as the box.
        assert xy[0] < box.pose.position[0] and xy[1] == pytest.approx(box.pose.position[1], abs=1e-9)
        assert np.linalg.norm(xy) < np.linalg.norm(box.pose.position[:2])
        old_y = box.pose.position[1] + box.bbox.extents[1] / 2 + 0.08
        assert old_y == pytest.approx(0.458, abs=0.002) and xy[1] < old_y - 0.1
        # Clear of the box by the gap, and close enough to *read* as next to it.
        can_radius = 0.5 * max(scene.get("obj_002").bbox.extents[:2])
        assert footprint_gap(box, xy, can_radius) == pytest.approx(NEXT_TO_GAP_M, abs=1e-9)
        placed = obj("obj_002", "can", "blue", (xy[0], xy[1], 0.4625), (0.0699, 0.057, 0.125))
        relations = compute_relations({"obj_002": placed, "obj_003": box})
        assert ("obj_002", "next_to", "obj_003") in relations

    def test_an_occupied_side_is_skipped(self):
        scene = isaac_default_scene()
        first, _ = synth("next_to", scene, dest="obj_003")
        blocker = obj("obj_009", "cube", "yellow", (first.object_xy[0], first.object_xy[1], 0.425), (0.05, 0.05, 0.05))
        crowded = scene_of(*scene.objects.values(), blocker)
        second, _ = synth("next_to", crowded, dest="obj_003")
        assert second is not None and second.object_xy != first.object_xy
        for other in crowded.objects.values():
            if other.track_id != "obj_002":
                assert footprint_gap(other, second.object_xy, 0.035) >= 0.01 - 1e-9

    def test_near_the_workspace_edge_the_spot_stays_inside(self):
        """Audit probe P5: a bowl at y=0.50 gave a release at y=0.666 > 0.55."""
        bowl = obj("obj_004", "bowl", "white", (0.55, 0.50, 0.43), (0.16, 0.16, 0.06))
        can = obj("obj_002", "can", "blue", (0.5, 0.0, 0.46), (0.07, 0.057, 0.125))
        target, _ = synth("next_to", scene_of(bowl, can), dest="obj_004")
        xy = np.array(target.object_xy)
        assert np.all(xy >= np.array(WS_MIN[:2])) and np.all(xy <= np.array(WS_MAX[:2]))
        assert footprint_gap(bowl, xy, 0.035) >= 0.01

    def test_a_rotated_destination_is_measured_along_its_own_sides(self):
        box = obj("obj_003", "box", "green", (0.6, 0.0, 0.44), (0.20, 0.06, 0.08), yaw=math.radians(90))
        can = obj("obj_002", "can", "blue", (0.4, -0.3, 0.46), (0.07, 0.057, 0.125))
        target, _ = synth("next_to", scene_of(box, can), dest="obj_003")
        # Rotated 90 deg, the box is 60 mm deep along X: the near side is 30 mm from its centre.
        assert half_extent_along(box, (1.0, 0.0)) == pytest.approx(0.03)
        assert target.object_xy[0] == pytest.approx(0.6 - 0.03 - 0.035 - NEXT_TO_GAP_M)

    def test_an_unreachable_side_is_skipped(self):
        scene = isaac_default_scene()
        target, _ = synth("next_to", scene, dest="obj_003", reachable=lambda xy: xy[0] > 0.6)
        assert target.object_xy[0] > 0.6
        assert footprint_gap(scene.get("obj_003"), target.object_xy, 0.035) == pytest.approx(NEXT_TO_GAP_M, abs=1e-4)

    def test_no_reachable_side_is_refused_before_moving(self):
        target, reason = synth("next_to", isaac_default_scene(), dest="obj_003", reachable=lambda xy: False)
        assert target is None and "out of the arm's reach" in reason
        target, reason = synth("left_of", isaac_default_scene(), dest="obj_003", reachable=lambda xy: False)
        assert target is None and "out of the arm's reach" in reason
        target, reason = synth("direction", isaac_default_scene(), origin_xy=(0.5, 0.0),
                               direction="forward", reachable=lambda xy: False)
        assert target is None and "out of the arm's reach" in reason

    def test_no_free_side_is_refused_with_a_reason(self):
        box = obj("obj_003", "box", "green", (0.5, 0.0, 0.44), (0.1, 0.1, 0.08))
        can = obj("obj_002", "can", "blue", (0.3, -0.4, 0.46), (0.07, 0.057, 0.125))
        walls = [obj(f"w{i}", "brick", "red", (0.5 + dx, dy, 0.43), (0.08, 0.08, 0.06))
                 for i, (dx, dy) in enumerate([(0.13, 0), (-0.13, 0), (0, 0.13), (0, -0.13)])]
        target, reason = synth("next_to", scene_of(box, can, *walls), dest="obj_003")
        assert target is None and "no free space next to the green box" in reason


class TestFixedSides:
    @pytest.mark.parametrize("relation,check", [
        ("left_of", lambda xy, c: xy[1] > c[1] and xy[0] == pytest.approx(c[0])),
        ("right_of", lambda xy, c: xy[1] < c[1] and xy[0] == pytest.approx(c[0])),
        ("in_front_of", lambda xy, c: np.linalg.norm(xy) < np.linalg.norm(c) - 0.05),
        ("behind", lambda xy, c: np.linalg.norm(xy) > np.linalg.norm(c) + 0.05),
    ])
    def test_side_relations_use_the_robot_frame(self, relation, check):
        scene = scene_of(obj("obj_003", "box", "green", (0.55, 0.0, 0.44), (0.1, 0.1, 0.08)),
                         obj("obj_002", "can", "blue", (0.4, -0.3, 0.46), (0.07, 0.057, 0.125)))
        target, reason = synth(relation, scene, dest="obj_003")
        assert target is not None, reason
        assert check(np.array(target.object_xy), np.array([0.55, 0.0]))
        assert footprint_gap(scene.get("obj_003"), target.object_xy, 0.035) == pytest.approx(NEXT_TO_GAP_M, abs=1e-6)

    def test_no_room_to_the_left_at_the_workspace_edge(self):
        scene = scene_of(obj("obj_003", "box", "green", (0.55, 0.50, 0.44), (0.1, 0.1, 0.08)),
                         obj("obj_002", "can", "blue", (0.4, -0.3, 0.46), (0.07, 0.057, 0.125)))
        target, reason = synth("left_of", scene, dest="obj_003")
        assert target is None and "no room to the left of the green box" in reason

    def test_a_taken_spot_names_what_is_in_the_way(self):
        scene = isaac_default_scene()
        target, reason = synth("right_of", scene_of(
            *scene.objects.values(),
            obj("obj_009", "cube", "yellow", (0.6575, 0.19, 0.425), (0.05, 0.05, 0.05))), dest="obj_003")
        assert target is None and "taken by the yellow cube" in reason


class TestToAndContainers:
    @pytest.mark.parametrize("label,extents,expected", [
        ("bowl", (0.16, 0.16, 0.06), "in"),
        ("sorting_bin", (0.3, 0.2, 0.1), "in"),
        ("basket", (0.2, 0.2, 0.1), "in"),
        ("box", (0.146, 0.123, 0.074), "on"),
        ("foam_brick", (0.05, 0.075, 0.05), "on"),
        ("pudding_box", (0.09, 0.038, 0.11), "next_to"),  # standing up: not flat
        ("can", (0.068, 0.068, 0.10), "next_to"),
        ("marker", (0.12, 0.019, 0.019), "next_to"),
    ])
    def test_to_lets_place_choose(self, label, extents, expected):
        assert relation_for_to(obj("d", label, "", (0.5, 0, 0.45), extents)) == expected

    def test_to_a_bowl_is_in_with_the_verified_rim_drop(self):
        scene = scene_of(*[obj(b.track_id, b.label, b.colour, b.centre, b.extents)
                           for b in with_bowl(default_bodies())])
        target, _ = synth("to", scene, dest="obj_004")
        assert target.relation == "in" and target.check == "inside"
        assert target.object_xy == pytest.approx((0.55, -0.30))
        assert target.tcp_z == pytest.approx(0.43 + 0.03 + CLEARANCE)

    def test_on_is_centred_on_top(self):
        target, _ = synth("on", isaac_default_scene(), dest="obj_003")
        assert target.check == "on_top" and target.object_xy == pytest.approx((0.6575, 0.3164))
        assert target.surface_z == pytest.approx(0.4445 + 0.0741 / 2)


class TestDirection:
    @pytest.mark.parametrize("direction,delta", [
        ("left", (0.0, 0.10)), ("right", (0.0, -0.10)), ("forward", (0.10, 0.0)), ("back", (-0.10, 0.0)),
        ("backward", (-0.10, 0.0)),
    ])
    def test_relative_to_the_pick_origin(self, direction, delta):
        target, reason = synth("direction", isaac_default_scene(), origin_xy=(0.5157, 0.116),
                               direction=direction)
        assert target is not None, reason
        assert np.array(target.object_xy) == pytest.approx(np.array([0.5157, 0.116]) + np.array(delta))

    def test_distance_is_honoured(self):
        target, _ = synth("direction", isaac_default_scene(), origin_xy=(0.5, 0.0),
                          direction="forward", distance=0.05)
        assert target.object_xy == pytest.approx((0.55, 0.0))
        assert "5 cm forward" in target.description

    def test_a_clamped_direction_is_refused_not_shortened(self):
        target, reason = synth("direction", isaac_default_scene(), origin_xy=(0.5, 0.48),
                               direction="left")
        assert target is None and "outside the workspace" in reason

    def test_a_taken_spot_is_refused(self):
        target, reason = synth("direction", isaac_default_scene(), origin_xy=(0.4598, -0.2504),
                               direction="left", distance=0.10)
        assert target is None and "taken by the red block" in reason


# ----------------------------------------------------------------------
# aim correction
# ----------------------------------------------------------------------


class TestAimWithRotatedWrist:
    OFFSET_G = np.array([0.025, 0.0, 0.0])  # 25 mm along the gripper X, as the audit probe

    @pytest.mark.parametrize("yaw_deg,legacy_error_mm", [(0, 0.0), (45, 19.1), (90, 35.4), (180, 50.0)])
    def test_rotating_the_gripper_frame_offset_lands_the_object_on_target(self, yaw_deg, legacy_error_mm):
        carry = yawed_top_down(yaw_deg)
        tcp = Pose(np.array([0.5, 0.1, 0.6]), carry, Frame.WORLD)
        centre = tcp.position + tcp.rotation_matrix() @ self.OFFSET_G
        g = offset_in_gripper(centre, tcp)
        assert g == pytest.approx(self.OFFSET_G)
        target = PlaceTarget("next_to", (0.52, 0.2), TABLE_TOP, "next to x")
        release = release_tcp_position(target, TOP_DOWN, clearance=CLEARANCE, held_half_height=0.03,
                                       offset_gripper=g)
        landed = release[:2] + aim_shift_xy(g, TOP_DOWN)
        assert np.linalg.norm(landed - np.array(target.object_xy)) < 1e-9
        # The old correction subtracted the carry-frame world offset.
        legacy_release = np.array(target.object_xy) - (centre - tcp.position)[:2]
        legacy_landed = legacy_release + aim_shift_xy(g, TOP_DOWN)
        assert np.linalg.norm(legacy_landed - np.array(target.object_xy)) * 1000 == pytest.approx(
            legacy_error_mm, abs=0.1)

    @pytest.mark.parametrize("yaw_deg", [45, 90, 180])
    def test_place_lands_the_object_when_the_transit_turns_the_wrist(self, yaw_deg):
        """Through the skill: carried turned, transited top-down (the first candidate)."""
        rig = make_rig(with_bowl(default_bodies()), held="obj_002", offset_g=self.OFFSET_G,
                       carry_quat=yawed_top_down(yaw_deg))
        result = rig.place({"relation": "next_to", "target": "green box"})
        assert result.ok, result.message
        target = np.array(result.data["object_target"])
        landed = rig.world.bodies["obj_002"].centre[:2]
        assert np.linalg.norm(landed - target) < 1e-6
        assert result.data["settled_offset"] < 1e-6

    def test_the_pick_record_is_used_when_the_held_object_is_hidden(self):
        rig = make_rig(default_bodies(), held="obj_002", offset_g=self.OFFSET_G,
                       carry_quat=yawed_top_down(90), record_grasp=True)

        def observe(original=rig.vision.observe):
            # The gripper occludes the held can: it is only seen once released.
            scene = original()
            if rig.world.held == "obj_002":
                return SceneGraph({k: v for k, v in scene.objects.items() if k != "obj_002"},
                                  scene.sim_time, scene.step_index, scene.relations)
            return scene

        rig.vision.observe = observe  # type: ignore[method-assign]
        result = rig.place({"relation": "left_of", "target": "the green box"})
        assert result.ok, result.message
        landed = rig.world.bodies["obj_002"].centre[:2]
        assert np.linalg.norm(landed - np.array(result.data["object_target"])) < 1e-6

    def test_no_aim_correction_on_the_depthless_lane(self):
        rig = make_rig(default_bodies(), held="obj_002", offset_g=(0.028, 0.0, 0.0),
                       gripper_feedback=False)
        result = rig.place({"relation": "next_to", "target": "green box"})
        release = np.array(result.data["release_position"])
        assert release[:2] == pytest.approx(result.data["object_target"])

    def test_place_it_back_keeps_the_tcp_over_the_pick_origin(self):
        """"place it" (8-26 mm on the sim lane) is unchanged: no aim shift at the origin."""
        rig = make_rig(default_bodies(), held="obj_002", offset_g=(0.0, 0.0, 0.0))
        result = rig.place({})
        assert result.ok, result.message
        assert result.data["release_position"][:2] == pytest.approx([0.5157, 0.1160])
        assert "where it was picked up" in result.message


class TestReleaseHeight:
    def test_next_to_sets_a_tall_object_down_instead_of_pressing_it_in(self):
        """Isaac: release z 0.45 for a 125 mm can gripped at its middle (32 mm into the table)."""
        rig = make_rig(default_bodies(), held="obj_002")
        result = rig.place({"relation": "next_to", "target": "green box"})
        release_z = result.data["release_position"][2]
        assert release_z == pytest.approx(TABLE_TOP + CLEARANCE + 0.125 / 2)
        assert release_z - 0.125 / 2 >= TABLE_TOP  # the can's bottom stays above the table

    def test_on_uses_the_destination_top_plus_half_the_held_height(self):
        rig = make_rig(default_bodies(), held="obj_001")
        result = rig.place({"relation": "on", "target": "green box"})
        top = 0.4371 + 0.0741 / 2
        assert result.data["release_position"][2] == pytest.approx(top + CLEARANCE + 0.0491 / 2)
        assert result.ok, result.message


# ----------------------------------------------------------------------
# honesty
# ----------------------------------------------------------------------


class TestAssessPlacement:
    TARGET = PlaceTarget("next_to", (0.5, 0.2), TABLE_TOP, "next to the green box")

    def test_within_tolerance(self):
        placed = obj("h", "can", "blue", (0.52, 0.2, 0.46), (0.07, 0.057, 0.125))
        check = assess_placement(self.TARGET, placed, None)
        assert check.ok and check.offset_m == pytest.approx(0.02)

    def test_the_350_mm_success_is_now_a_miss(self):
        placed = obj("h", "can", "blue", (0.5, -0.15, 0.46), (0.07, 0.057, 0.125))
        check = assess_placement(self.TARGET, placed, None, name="blue can")
        assert not check.ok and "350 mm" in check.reason and "50 mm allowed" in check.reason

    def test_tolerance_is_configurable(self):
        placed = obj("h", "can", "blue", (0.56, 0.2, 0.46), (0.07, 0.057, 0.125))
        assert not assess_placement(self.TARGET, placed, None).ok
        assert assess_placement(self.TARGET, placed, None, tolerance_m=0.07).ok
        assert DEFAULT_PLACE_TOLERANCE_M == 0.05

    def test_not_seen_is_not_success(self):
        check = assess_placement(self.TARGET, None, None, name="blue can")
        assert not check.ok and not check.seen and "not seen after release" in check.reason

    def test_in_means_inside_the_container_footprint(self):
        bowl = obj("b", "bowl", "red", (0.5, -0.3, 0.43), (0.16, 0.16, 0.06))
        target = PlaceTarget("in", (0.5, -0.3), 0.46, "in the red bowl", "b", check="inside", tcp_z=0.48)
        inside = obj("h", "marker", "black", (0.56, -0.25, 0.42), (0.12, 0.02, 0.02))
        outside = obj("h", "marker", "black", (0.5, -0.395, 0.41), (0.12, 0.02, 0.02))
        assert assess_placement(target, inside, bowl).ok
        miss = assess_placement(target, outside, bowl, name="black marker")
        assert not miss.ok and "95 mm from the centre of the red bowl, outside it" in miss.reason

    def test_on_means_on_top_when_heights_are_measured(self):
        box = obj("b", "box", "green", (0.6, 0.3, 0.44), (0.14, 0.12, 0.08))
        target = PlaceTarget("on", (0.6, 0.3), 0.48, "on the green box", "b", check="on_top")
        on_top = obj("h", "block", "red", (0.61, 0.3, 0.505), (0.05, 0.05, 0.05))
        beside_low = obj("h", "block", "red", (0.61, 0.3, 0.425), (0.05, 0.05, 0.05))
        assert assess_placement(target, on_top, box).ok
        assert not assess_placement(target, beside_low, box).ok
        # Depthless lane: z is pinned to the table, only the footprint counts.
        assert assess_placement(target, beside_low, box, vertical_known=False).ok


class TestHonestyThroughPlace:
    def test_a_slipped_object_is_a_failed_place_with_the_numbers(self):
        rig = make_rig(default_bodies(), held="obj_002")
        rig.world.slip_on_first_move = True
        result = rig.place({"relation": "next_to", "target": "green box"})
        assert result.status is SkillStatus.FAILED
        assert result.data["retryable"] is False and result.data["released"] is True
        assert "place missed next to the green box" in result.message
        assert f"{result.data['settled_offset'] * 1000:.0f} mm" in result.message
        assert rig.memory.get_held_object() is None

    def test_not_seen_after_release_is_a_failed_place(self):
        rig = make_rig(default_bodies(), held="obj_002")
        rig.world.hide_after_release = {"obj_002"}
        result = rig.place({})
        assert result.status is SkillStatus.FAILED and "not seen after release" in result.message
        assert result.data["placement"]["seen"] is False

    def test_a_good_place_reports_the_measured_offset(self):
        rig = make_rig(default_bodies(), held="obj_002")
        result = rig.place({"relation": "next_to", "target": "the green box"})
        assert result.ok
        assert result.message.startswith("placed next to the green box (settled 0 mm")
        assert result.data["placement"]["ok"] is True

    def test_the_planner_does_not_retry_a_released_miss(self):
        """Through the real TaskPlanner: one attempt, and the miss is what the operator hears."""
        rig = make_rig(default_bodies(), held="obj_002")
        rig.world.slip_on_first_move = True
        executor = ClassicalExecutor(SkillRegistry(rig.context, (Place,)))
        planner = TaskPlanner(RuleBasedIntentParser(("place",)), {"classical": executor},
                              rig.memory, rig.config)
        outcome = planner.handle("place it next to the green box")
        assert not outcome.ok and outcome.attempts == 1
        assert outcome.message.startswith("place missed next to the green box")
        assert "not holding" not in outcome.message

    def test_the_tolerance_is_set_from_the_config_file(self):
        """grasp.place_tolerance_m must be loadable (strict schema) and must move the verdict."""
        strict = make_rig(default_bodies(), held="obj_002")
        strict.world.slip_on_first_move = True
        missed = strict.place({"relation": "next_to", "target": "green box"})
        assert missed.status is SkillStatus.FAILED
        offset = float(missed.data["settled_offset"])
        loose = make_rig(default_bodies(), held="obj_002",
                         grasp={"place_tolerance_m": offset + 0.01})
        loose.world.slip_on_first_move = True
        assert loose.config.grasp.place_tolerance_m == pytest.approx(offset + 0.01)
        assert loose.place({"relation": "next_to", "target": "green box"}).ok
        with pytest.raises(Exception, match="place_tolerance_m"):
            load_config(overrides={"grasp": {"place_tolerance_m": 0.0}})

    def test_in_the_bowl_succeeds_inside_the_footprint(self):
        rig = make_rig(with_bowl(default_bodies()), held="obj_001")
        result = rig.place({"relation": "in", "target": "bowl"})
        assert result.ok, result.message
        assert result.message.startswith("placed in the white bowl")


# ----------------------------------------------------------------------
# hardware lane, end to end on the fake robot server and scripted camera
# ----------------------------------------------------------------------

if HARDWARE_LANE_PRESENT:
    from tests.test_hardware_e2e import lane_factory  # noqa: E402,F401  (pytest fixture)


@needs_hardware_lane
class TestHardwareFakeLane:
    def test_direction_places_and_an_honest_refusal(self, lane_factory):  # noqa: F811
        """The 0.06-0.30 m hardware table: "5 cm forward" was refused as outside the
        workspace while the marker's half-length was subtracted from it; a next_to side
        the arm could transit above but not lower to left the marker held at 11 cm."""
        from mfw.assistant import Assistant

        def pick_then(command: str):
            cfg, world, *_ = lane_factory()
            assistant = Assistant(config=cfg)
            try:
                assert assistant.command("pick up the marker").ok
                outcome = assistant.command(command)
                return (outcome, world.objects["marker"].copy(), world.attached,
                        assistant.runtime.memory.get_held_object())
            finally:
                assistant.close()

        forward, at_forward, _, _ = pick_then("put it 5 cm forward")
        left, at_left, _, _ = pick_then("put it to the left")
        beside, _, attached, held = pick_then("place it next to the bowl")
        assert forward.ok, forward.message
        assert at_forward[:2] == pytest.approx([0.23, 0.05], abs=0.005)
        assert left.ok, left.message
        assert at_left[:2] == pytest.approx([0.18, 0.15], abs=0.005)
        assert "horizontally" in left.message
        # Refused before any motion toward it, with the reason; still held, jaw shut.
        assert not beside.ok and beside.attempts == 1
        assert "out of the arm's reach" in beside.message
        assert attached == "marker" and held is not None


# ----------------------------------------------------------------------
# hardware lane: the transit fallback against the real planner
# ----------------------------------------------------------------------


def _held_box_clearance(kin, trajectory, obstacle, held_size) -> float:
    """Worst signed distance of the held box's corners to ``obstacle`` along the path (negative: inside)."""
    import itertools

    centre = np.asarray(obstacle.bbox.center.position)
    half = np.asarray(obstacle.bbox.extents) / 2.0
    worst = float("inf")
    h = np.asarray(held_size) / 2.0
    for waypoint in trajectory.waypoints:
        pose = kin.fk(waypoint.positions)
        rotation, position = pose.rotation_matrix(), pose.position
        for signs in itertools.product((-1, 1), (-1, 1), (-1, 1)):
            corner = position + rotation @ (np.array(signs) * h)
            d = np.abs(corner - centre) - half
            value = float(np.linalg.norm(np.maximum(d, 0.0))) if np.any(d > 0) else float(d.max())
            worst = min(worst, value)
    return worst


@needs_hardware_lane
class TestHardwareTransitFallback:
    def test_place_never_carries_a_cube_through_the_bin(self):
        """The re-review sweep (probe_fallback.py): bin at (0.17, 0), held cube, start and goal
        on opposite sides. Before: 12 of 108 crossings ran an unchecked Cartesian line 37-41 mm
        into the bin. Now every executed transit clears it, or Place refuses."""
        from tests import test_hardware_grasp_planner as T
        from mfw.hardware.planner import JointSpacePlanner

        cfg = load_config(T.REPO_ROOT / "configs" / "hardware.yaml")
        kin = T.PlanarKinematics(cfg.hardware.arm, T.JOINTS)
        sizes = {k: tuple(v) for k, v in cfg.hardware.object_sizes.items()}
        cube = sizes["cube"]

        def reach(xy, z):
            q = kin.ik(T._top_down(xy, z))
            if q is not None:
                return q
            bearing = math.atan2(xy[1], xy[0])
            for pitch in (75.0, 60.0, 45.0):
                tilt = math.radians(90 - pitch)
                radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
                tangential = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
                approach = math.sin(tilt) * radial - math.cos(tilt) * np.array([0.0, 0.0, 1.0])
                q = kin.ik(T.build_grasp_pose(np.array([xy[0], xy[1], z]), tangential, approach))
                if q is not None:
                    return q
            return None

        class Controller:
            def __init__(self, robot):
                self.robot, self.followed = robot, []

            def follow_trajectory(self, trajectory, on_step=None):
                self.followed.append(trajectory)
                self.robot.q = np.asarray(trajectory.waypoints[-1].positions, dtype=float)
                return True

            def maintain_grasp(self):
                pass

            def open_gripper_blocking(self):
                return 0.03

        class Vision:
            def __init__(self, scene):
                self.scene = scene

            def observe(self):
                return self.scene

            def require_fresh_scene(self):
                return self.scene

        executed = refused = 0
        penetrations = []
        for rs in np.arange(0.13, 0.235, 0.02):
            for rg in np.arange(0.13, 0.235, 0.02):
                for bs, bg in [(0.9, -0.9), (0.7, -0.7), (1.1, -0.6)]:
                    start_xy = (rs * math.cos(bs), rs * math.sin(bs))
                    goal_xy = (rg * math.cos(bg), rg * math.sin(bg))
                    q0 = reach(start_xy, cube[2] / 2 + 0.06)
                    if q0 is None:
                        continue
                    bin_ = T._object("bin", "bin", (0.17, 0.0), sizes["bin"])
                    held = T._object("held", "cube", start_xy, cube)
                    scene = T._scene(bin_, held)
                    robot = T.PlanarRobot(kin, q=q0)
                    planner = JointSpacePlanner(robot, kin, cfg.motion, cfg.hardware.arm)
                    planner.update_collision_world(scene)
                    controller = Controller(robot)
                    memory = WorkingMemory(MemoryConfig())
                    memory.update_scene(scene)
                    memory.set_held_object("held")
                    context = SkillContext(sim=FakeSim(), robot=robot, vision=Vision(scene), planner=planner,
                                           controller=controller, grasp_scorer=None, memory=memory,
                                           config=cfg, support_height=0.0)
                    zg = min(cube[2] / 2 + 0.005, 0.12)
                    result = Place(context).execute({"position": [goal_xy[0], goal_xy[1], zg]})
                    if not controller.followed:
                        refused += 1
                        assert result.status is SkillStatus.INFEASIBLE, result.message
                        continue
                    executed += 1
                    transit = controller.followed[0]
                    clearance = _held_box_clearance(kin, transit, bin_, cube)
                    if clearance < 0.0:
                        penetrations.append((round(float(rs), 2), round(float(rg), 2), bs, bg,
                                             transit.planner_name, round(clearance * 1000, 1)))
        # Measured: HEAD's Place executes 29 of these, 12 of them on the
        # unchecked line, 37-41 mm into the bin. Now 17 run (all clear) and the
        # rest are refused with "no collision-free path".
        assert executed >= 10, (executed, refused)
        assert penetrations == [], penetrations


# ----------------------------------------------------------------------
# "on": where the destination's centre comes from
# ----------------------------------------------------------------------
#
# Isaac, events 20260924_104645_0cc762 (logs/e2e/run_c_default_phase8.txt),
# "move the red block onto the green box": failed, "the red block settled 72
# mm from the centre of the green box, off it". Every number below is copied
# from that event log (seq 140-162) or configs/default.yaml.

# Every observation of the run but two (the exterior camera alone):
RUN_C_BOX_EXTERIOR = {"centre": (0.657460137394592, 0.31640833106241756, 0.44445699981237186),
                      "extents": (0.14593978504187977, 0.12326180425409583, 0.07406548609766239),
                      "yaw_deg": 85.0}
# seq 59/61, the arm over the can with the wrist camera near the box:
RUN_C_BOX_WRIST = {"centre": (0.6141, 0.2954, 0.4457), "extents": (0.146, 0.113, 0.069),
                   "yaw_deg": 91.0}
# Where configs/default.yaml spawns it:
RUN_C_BOX_TRUE = {"centre": (0.62, 0.28, 0.44), "extents": (0.12, 0.12, 0.08)}
RUN_C_OFFSET_G = (0.0069, -0.0029, 0.0106)          # held_grasp.offset_in_gripper, seq 151
RUN_C_GRASP_QUAT = (0.0052, 0.6059, -0.7955, -0.0062)
RUN_C_BLOCK_REST = (0.0604, 0.0484, 0.044)          # held_grasp.rest_extents
RUN_C_HELD_HEIGHT = 0.044326972664332664            # the held block's z extent, seq 153
RUN_C_RELEASE = (0.6567713478743151, 0.319319517198502, 0.5236532291933694)  # seq 162


def _box(spec: dict, track_id: str = "obj_003", step: int = 0) -> ObjectHypothesis:
    return obj(track_id, "box", "green", spec["centre"], spec["extents"],
               math.radians(spec.get("yaw_deg", 0.0)), step=step)


def _edge_distance(support: ObjectHypothesis, xy) -> float:
    """How far inside ``support``'s footprint ``xy`` is (negative: outside)."""
    return -footprint_gap(support, xy)


def _sixty_degree_quat(object_xy) -> np.ndarray:
    """Place's second release candidate (60 deg radial), the one the Isaac run used."""
    rig = make_rig(default_bodies())
    return Place(rig.context)._candidate_quats(object_xy)[1]


class TestRunCPlaceOnReconstructed:
    """What was wrong: not the aim, the frame or the height -- the box's centre."""

    def test_the_release_was_exactly_where_place_aimed_it(self):
        target = PlaceTarget("on", RUN_C_BOX_EXTERIOR["centre"][:2], 0.4815, "on the green box",
                             "obj_005", check="on_top")
        quat = _sixty_degree_quat(target.object_xy)
        release = release_tcp_position(target, quat, clearance=CLEARANCE,
                                       held_half_height=RUN_C_HELD_HEIGHT / 2,
                                       offset_gripper=RUN_C_OFFSET_G)
        assert release == pytest.approx(RUN_C_RELEASE, abs=5e-4)
        # The gripper-frame correction was 3 mm and put the block's centre on the target.
        landed = release[:2] + aim_shift_xy(RUN_C_OFFSET_G, quat)
        assert np.linalg.norm(landed - target.object_xy) < 1e-9

    def test_the_perceived_centre_was_inside_the_real_box_far_corner(self):
        real = _box(RUN_C_BOX_TRUE)
        aim = np.array(RUN_C_BOX_EXTERIOR["centre"][:2])
        # 37 mm and 36 mm off the real centre: 21-23 mm inside both far edges,
        # less than the block's own 24-30 mm half size, so it overhung a corner.
        assert aim - np.array(RUN_C_BOX_TRUE["centre"][:2]) == pytest.approx([0.0375, 0.0364], abs=5e-4)
        assert 0.020 < _edge_distance(real, aim) < min(RUN_C_BLOCK_REST[:2]) / 2
        # The one close (wrist) view of the run was 48 mm from the exterior one
        # and would have left 45 mm to the nearest real edge.
        wrist = np.array(RUN_C_BOX_WRIST["centre"][:2])
        assert np.linalg.norm(aim - wrist) == pytest.approx(0.048, abs=1e-3)
        assert _edge_distance(real, wrist) > max(RUN_C_BLOCK_REST[:2]) / 2 + 0.01

    def test_the_same_run_put_a_can_next_to_the_box_onto_its_near_edge(self):
        """seq 80-81: next_to aimed at (0.533, 0.327); the can was then seen at z 0.540, on the box."""
        perceived = _box(RUN_C_BOX_EXTERIOR)
        real = _box(RUN_C_BOX_TRUE)
        can_radius = 0.070 / 2
        spot = (0.533, 0.3273)
        assert footprint_gap(perceived, spot, can_radius) == pytest.approx(NEXT_TO_GAP_M, abs=2e-3)
        assert footprint_gap(real, spot, can_radius) < 0.0  # overlaps the real box
        assert 0.4 + 0.08 + 0.12 / 2 == pytest.approx(0.540, abs=1e-3)  # a 120 mm can on its top


class TestRefinePlaceTarget:
    def _target(self, relation="on"):
        return PlaceTarget(relation, RUN_C_BOX_EXTERIOR["centre"][:2], 0.4815, "on the green box",
                           "obj_003", check={"on": "on_top", "in": "inside"}[relation],
                           tcp_z=0.50 if relation == "in" else None)

    def test_the_close_view_re_centres_the_target(self):
        target, shift, reason = refine_place_target(self._target(), _box(RUN_C_BOX_EXTERIOR),
                                                    _box(RUN_C_BOX_WRIST))
        assert reason == "" and shift == pytest.approx(0.048, abs=1e-3)
        assert target.object_xy == pytest.approx(RUN_C_BOX_WRIST["centre"][:2])
        assert target.surface_z == 0.4815 and target.check == "on_top"

    def test_in_keeps_its_verified_rim_height(self):
        target, _, reason = refine_place_target(self._target("in"), _box(RUN_C_BOX_EXTERIOR),
                                                _box(RUN_C_BOX_WRIST))
        assert reason == "" and target.tcp_z == 0.50
        assert target.object_xy == pytest.approx(RUN_C_BOX_WRIST["centre"][:2])

    def test_not_seen_from_above_keeps_the_target(self):
        target = self._target()
        kept, shift, reason = refine_place_target(target, _box(RUN_C_BOX_EXTERIOR), None)
        assert kept is target and shift == 0.0 and "not seen" in reason

    def test_a_jump_beyond_the_limit_is_not_a_better_estimate(self):
        far = {**RUN_C_BOX_WRIST, "centre": (0.567460137394592, 0.31640833106241756, 0.44)}  # 90 mm
        kept, shift, reason = refine_place_target(self._target(), _box(RUN_C_BOX_EXTERIOR), _box(far))
        assert kept.object_xy == self._target().object_xy
        assert shift == pytest.approx(0.090, abs=1e-6) and "more than the 80 mm" in reason

    def test_a_view_truncated_by_the_held_object_is_not_used(self):
        cut = {**RUN_C_BOX_WRIST, "extents": (0.146, 0.070, 0.069)}  # 70 of 123 mm seen
        kept, _, reason = refine_place_target(self._target(), _box(RUN_C_BOX_EXTERIOR), _box(cut))
        assert kept.object_xy == self._target().object_xy and "only part" in reason

    def test_side_relations_are_not_touched(self):
        target = PlaceTarget("next_to", (0.533, 0.327), TABLE_TOP, "next to the green box", "obj_003")
        kept, _, reason = refine_place_target(target, _box(RUN_C_BOX_EXTERIOR), _box(RUN_C_BOX_WRIST))
        assert kept is target and reason


class TippingWorld(FakeWorld):
    """The fake world plus the one physical fact the Isaac run showed.

    An object released where its footprint can reach past the edge of what it
    lands on (centre nearer the edge than half its longest side) tips off it
    onto the table, outwards. Otherwise it stays, as in :class:`FakeWorld`.
    """

    def _drop(self, track_id: str) -> None:
        body = self.bodies[track_id]
        longest = float(np.max(body.extents[:2]))
        for other in self.bodies.values():
            if other.track_id == track_id:
                continue
            support = obj(other.track_id, other.label, other.colour, other.centre, other.extents, other.yaw)
            inside = _edge_distance(support, body.centre[:2])
            if 0.0 <= inside < longest / 2.0:
                outward = body.centre[:2] - other.centre[:2]
                outward = outward / max(float(np.linalg.norm(outward)), 1e-9)
                xy = body.centre[:2] + outward * (inside + longest)
                body.centre = np.array([xy[0], xy[1], self.support + body.extents[2] / 2.0])
                self.releases.append((track_id, body.centre.copy()))
                self.held = None
                return
        super()._drop(track_id)


class ExteriorBiasedVision(FakeVision):
    """The green box as the Isaac run perceived it.

    From afar (the exterior camera) it is the run's exterior estimate; with the
    hand within 10 cm of it and low (the wrist camera close) it is the run's
    wrist estimate -- unless ``close_view`` is off, the control case.
    """

    def __init__(self, world: FakeWorld, sim: FakeSim, close_view: bool = True) -> None:
        super().__init__(world, sim)
        self.close_view = close_view
        self.close_looks = 0

    def observe(self) -> SceneGraph:
        scene = super().observe()
        tcp = np.asarray(self.world.tcp.position, dtype=float)
        near = bool(np.linalg.norm(tcp[:2] - np.array(RUN_C_BOX_TRUE["centre"][:2])) < 0.10
                    and tcp[2] < 0.75)
        spec = RUN_C_BOX_WRIST if (near and self.close_view) else RUN_C_BOX_EXTERIOR
        self.close_looks += int(near and self.close_view)
        objects = dict(scene.objects)
        objects["obj_003"] = _box(spec, step=scene.step_index)
        self.last = scene_of(*objects.values(), step=scene.step_index)
        return self.last


def run_c_rig(*, close_view: bool = True, gripper_feedback: bool = True) -> Rig:
    bodies = [
        Body("obj_001", "block", "red", np.array([0.4678, -0.1799, TABLE_TOP + RUN_C_BLOCK_REST[2] / 2]),
             np.array(RUN_C_BLOCK_REST)),
        Body("obj_002", "can", "blue", np.array([0.4921, -0.0166, 0.4625]),
             np.array([0.0699, 0.0570, 0.1250])),
        Body("obj_003", "box", "green", np.array(RUN_C_BOX_TRUE["centre"]),
             np.array(RUN_C_BOX_TRUE["extents"])),
    ]
    base = make_rig(bodies, held="obj_001", offset_g=RUN_C_OFFSET_G,
                    carry_quat=np.array(RUN_C_GRASP_QUAT), gripper_feedback=gripper_feedback,
                    record_grasp=True)
    world = TippingWorld(list(base.world.bodies.values()))
    world.held, world.offset_g, world.tcp = base.world.held, base.world.offset_g, base.world.tcp
    vision = ExteriorBiasedVision(world, base.sim, close_view=close_view)
    # Top-down has no IK over the box, as in the run: the 60 deg candidate is used.
    robot = FakeRobot(world, reachable=lambda pose: not np.allclose(np.abs(pose.quat), TOP_DOWN))
    controller = FakeController(world)
    base.context.vision, base.context.robot, base.context.controller = vision, robot, controller
    return Rig(world, base.sim, vision, robot, base.planner, controller, base.memory,
               base.context, base.config)


class TestPlaceOnLooksAgainFromAbove:
    def test_without_a_better_view_the_isaac_miss_is_reproduced(self):
        rig = run_c_rig(close_view=False)
        result = rig.place({"relation": "on", "target": "the green box"})
        # The release Isaac logged ...
        assert result.data["release_position"] == pytest.approx(RUN_C_RELEASE, abs=1e-3)
        # ... and its outcome: the block tipped off the real box's corner.
        assert result.status is SkillStatus.FAILED and not result.ok
        assert "place missed on the green box" in result.message
        landed = rig.world.bodies["obj_001"].centre
        assert landed[2] == pytest.approx(TABLE_TOP + RUN_C_BLOCK_REST[2] / 2)
        assert result.data["destination_refinement"]["applied"] is False
        assert result.data["destination_refinement"]["reason"] == "the close view agrees"

    def test_the_close_view_puts_the_block_on_the_box(self):
        rig = run_c_rig(close_view=True)
        result = rig.place({"relation": "on", "target": "the green box"})
        assert result.ok, result.message
        refinement = result.data["destination_refinement"]
        assert refinement["applied"] is True
        assert refinement["shift_m"] == pytest.approx(0.048, abs=1e-3)
        assert result.data["object_target"] == pytest.approx(RUN_C_BOX_WRIST["centre"][:2])
        landed = rig.world.bodies["obj_001"].centre
        assert landed[2] == pytest.approx(0.48 + RUN_C_BLOCK_REST[2] / 2)  # on the real top
        assert _edge_distance(_box(RUN_C_BOX_TRUE), landed[:2]) > max(RUN_C_BLOCK_REST[:2]) / 2
        # Across at the height above, then lowered straight down.
        transit, across, lower = (t.goal.position for t in rig.controller.followed[:3])
        assert across[:2] == pytest.approx(lower[:2])
        assert across[2] == pytest.approx(lower[2] + 0.08) == pytest.approx(transit[2])

    def test_no_straight_line_across_keeps_the_original_release(self):
        rig = run_c_rig(close_view=True)
        original = rig.planner.plan_cartesian_line

        def line(start, goal, scene):
            if goal.position[2] > 0.58:  # the move across at the height above
                return None
            return original(start, goal, scene)

        rig.planner.plan_cartesian_line = line  # type: ignore[method-assign]
        result = rig.place({"relation": "on", "target": "the green box"})
        assert result.data["destination_refinement"]["applied"] is False
        assert "no straight line" in result.data["destination_refinement"]["reason"]
        assert result.data["release_position"] == pytest.approx(RUN_C_RELEASE, abs=1e-3)

    def test_the_depthless_lane_does_not_look_again(self):
        """One fixed overhead camera: over the destination it would see only the arm."""
        rig = run_c_rig(close_view=True, gripper_feedback=False)
        result = rig.place({"relation": "on", "target": "the green box"})
        assert result.data["destination_refinement"] is None

    def test_next_to_is_unchanged(self):
        rig = run_c_rig(close_view=True)
        result = rig.place({"relation": "next_to", "target": "the green box"})
        assert result.data["destination_refinement"] is None

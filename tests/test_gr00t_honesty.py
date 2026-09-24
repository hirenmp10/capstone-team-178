"""GR00T executor honesty and observability.

The scenario these tests pin down was measured in Isaac on 2026-09-24
(``logs/e2e/run_f_groot.txt``): with the real N1.7 server, "pick up the blue
can" came back SUCCESS -- "policy completed after 1 iteration(s), 8 step(s)" --
while the executor itself logged that no perceived object was in the hand. The
first observation had coloured every object black, so the target did not
resolve, ``target_track_id`` was ``None``, and the goal test accepted *any*
gripper reporting a grasp. Closing on air reports one.

No sockets, no Isaac, no model: the policy client is an in-process fake that
counts ``predict`` calls, so "the policy was never queried" is an assertion,
not an inference. The world attaches the object only when the fingers close
within 2 cm of it, so "held" is decided by geometry; ``air_grasp=True``
reproduces the simulator's false ``is_grasping`` on an empty close.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mfw.config.schema import MemoryConfig, load_config
from mfw.core.errors import SafetyViolation
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
from mfw.gr00t_bridge.executor import MIN_LIFT_M, Gr00tExecutor, chunk_stats
from mfw.memory.working_memory import WorkingMemory
from mfw.planner.task_planner import _reference_error

pytestmark = pytest.mark.phase7

TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
HORIZON = 40
CAN_AT = (0.50, 0.00, 0.45)

ITERATION_FIELDS = (
    "tcp_start",
    "tcp_end",
    "tcp_displacement_m",
    "max_step_translation_m",
    "max_commanded_step_m",
    "gripper_closure_commanded",
    "gripper_commanded_m",
    "gripper_measured_m",
    "gripper_commands",
    "clamped",
    "clamped_actions",
    "rejected",
    "rejected_actions",
    "instruction",
    "action_chunk",
    "goal_reached",
    "goal_reason",
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class World:
    """Robot + controller + cameras + vision over one shared state."""

    def __init__(self, config, objects: dict[str, tuple[str, str, tuple]], air_grasp: bool = False):
        self.config = config
        self.pose = Pose(np.array([0.30, 0.0, 0.60]), TOP_DOWN_QUAT, Frame.WORLD)
        self.width = config.robot.gripper_open_width
        # track id -> [label, colour, position]
        self.objects = {
            tid: [label, colour, np.asarray(pos, dtype=np.float64)]
            for tid, (label, colour, pos) in objects.items()
        }
        self.attached: str | None = None
        self.air_grasp = air_grasp
        self.servo_targets: list[np.ndarray] = []
        self.refuse_servo = False

    # robot ------------------------------------------------------------
    def tcp_pose(self) -> Pose:
        return self.pose

    def get_gripper_state(self) -> GripperState:
        closed = self.width < self.config.robot.gripper_open_width * 0.5
        grasping = self.attached is not None or (self.air_grasp and closed)
        return GripperState(width=self.width, target_width=self.width,
                            is_moving=False, is_grasping=grasping)

    def get_state(self) -> RobotState:
        return RobotState(
            joint_state=JointState(positions=np.zeros(7), names=tuple(f"j{i}" for i in range(7))),
            tcp_pose=self.pose,
            gripper=self.get_gripper_state(),
            sim_time=0.0,
            step_index=0,
        )

    def open_gripper(self) -> None:
        self.width = self.config.robot.gripper_open_width
        self.attached = None

    def close_gripper(self) -> None:
        if self.attached is not None:
            return
        for tid, (_, _, position) in self.objects.items():
            if np.linalg.norm(position - self.pose.position) < 0.02:
                self.attached = tid
                self.width = 0.06
                return
        self.width = self.config.robot.gripper_closed_width

    # controller -------------------------------------------------------
    def servo_to_pose(self, pose: Pose) -> bool:
        if self.refuse_servo:
            return False
        self.servo_targets.append(np.asarray(pose.position, dtype=np.float64).copy())
        self.pose = pose
        if self.attached is not None:
            self.objects[self.attached][2] = np.asarray(pose.position, dtype=np.float64).copy()
        return True

    # vision -----------------------------------------------------------
    def _scene(self) -> SceneGraph:
        objects = {}
        for tid, (label, colour, position) in self.objects.items():
            pose = Pose(position.copy(), np.array([1.0, 0, 0, 0]), Frame.WORLD)
            objects[tid] = ObjectHypothesis(
                track_id=tid, label=label, pose=pose,
                bbox=BoundingBox3D(center=pose, extents=np.array([0.066, 0.066, 0.1])),
                confidence=0.9, num_points=500, last_seen_sim_time=0.0, last_seen_step=0,
                attributes={"color": colour} if colour else {},
            )
        return SceneGraph(objects=objects, sim_time=0.0, step_index=0)

    def require_fresh_scene(self) -> SceneGraph:
        return self._scene()

    def observe(self) -> SceneGraph:
        return self._scene()

    def cameras(self) -> dict[str, Any]:
        frame = types.SimpleNamespace(rgb=np.zeros((120, 160, 3), np.uint8))
        camera = types.SimpleNamespace(capture=lambda: frame)
        return {self.config.exterior_camera.name: camera, self.config.wrist_camera.name: camera}


class FakeClient:
    """``IPolicyClient`` stand-in that counts every ``predict`` (get_action) call."""

    def __init__(self, policy: Callable[[dict[str, Any]], dict[str, Any]]):
        self.policy = policy
        self.calls = 0
        self.instructions: list[str] = []

    def is_ready(self) -> bool:
        return True

    def connect(self, *args: Any, **kwargs: Any) -> None:
        return None

    def predict(self, observation: dict[str, Any]) -> Any:
        self.calls += 1
        self.instructions.append(
            observation["language"]["annotation.language.language_instruction"][0][0]
        )
        return self.policy(observation), {}


def _chunk(poses: np.ndarray, gripper: float) -> dict[str, Any]:
    return {
        "eef_9d": poses[None].astype(np.float32),
        "gripper_position": np.full((1, HORIZON, 1), gripper, dtype=np.float32),
        "joint_position": np.zeros((1, HORIZON, 7), dtype=np.float32),
    }


def _state_of(observation):
    """Position, rot6d, DROID gripper closure (0 open .. 1 closed), instruction."""
    eef = np.asarray(observation["state"]["eef_9d"], dtype=np.float64).reshape(9)
    closure = float(np.asarray(observation["state"]["gripper_position"]).reshape(-1)[0])
    instruction = observation["language"]["annotation.language.language_instruction"][0][0]
    return eef[:3], eef[3:], closure, instruction


OPEN, CLOSE = 0.0, 1.0  # DROID closure, as N1.7 emits it


def reach_policy(goal, lift_m: float = 0.10, close_at_m: float = 0.01):
    """Absolute poses 5 mm/step toward ``goal``; close there; lift once closed.

    'put down' opens instead of closing.
    """
    goal = np.asarray(goal, dtype=np.float64)

    def policy(observation):
        position, rot6d, closure, instruction = _state_of(observation)
        releasing = instruction.startswith("put down")
        holding = not releasing and closure > 0.1
        aim = goal + np.array([0.0, 0.0, lift_m]) if holding else goal
        to_aim = aim - position
        distance = float(np.linalg.norm(to_aim))
        direction = to_aim / distance if distance > 1e-9 else np.zeros(3)
        travel = np.minimum(np.arange(1, HORIZON + 1) * 0.005, distance)
        poses = np.hstack([position + travel[:, None] * direction, np.tile(rot6d, (HORIZON, 1))])
        close = holding or (distance < close_at_m and not releasing)
        return _chunk(poses, CLOSE if close else OPEN)

    return policy


def close_in_place_policy(observation):
    """Hold the current pose and close: the real run's 1-iteration 'success'."""
    position, rot6d, _, _ = _state_of(observation)
    poses = np.tile(np.concatenate([position, rot6d]), (HORIZON, 1))
    return _chunk(poses, CLOSE)


class Events:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]


def _config(gripper_feedback: bool = True):
    return load_config(
        overrides={
            "gr00t": {"enabled": True, "use_mock_server": False, "port": 5555},
            "robot": {"gripper_feedback": gripper_feedback},
        }
    )


def _build(world_objects, policy, *, air_grasp=False, gripper_feedback=True, memory=True):
    config = _config(gripper_feedback)
    world = World(config, world_objects, air_grasp=air_grasp)
    client = FakeClient(policy)
    events = Events()
    mem = WorkingMemory(MemoryConfig()) if memory else None
    context = types.SimpleNamespace(held_grasp="stale classical record")
    executor = Gr00tExecutor(
        client=client, robot=world, vision=world, controller=world,
        cameras=world.cameras(), config=config, memory=mem, events=events,
        skill_context=context,
    )
    return executor, world, client, mem, events, context


BLUE_CAN = {"obj_7": ("can", "blue", CAN_AT)}


# ---------------------------------------------------------------------------
# 1. the measured false success
# ---------------------------------------------------------------------------


class TestNoFalsePickSuccess:
    def test_close_on_air_with_an_unresolved_target_never_queries_the_policy(self):
        """The Isaac run, reproduced: every object perceived black, 'blue can' asked for.

        Old behaviour: target None, policy run, gripper closed on air reported
        grasping, SUCCESS. Now: ObjectNotFound before a single get_action.
        """
        black = {"obj_1": ("block", "black", (0.45, -0.2, 0.43)),
                 "obj_7": ("can", "black", CAN_AT),
                 "obj_9": ("box", "black", (0.65, 0.3, 0.44))}
        executor, world, client, memory, _, _ = _build(black, close_in_place_policy, air_grasp=True)

        result = executor.execute("pick", {"target": "blue can"})

        assert client.calls == 0, "the policy must not run blind for a named object"
        assert result.status is SkillStatus.FAILED
        assert result.message.startswith("ObjectNotFound:")
        assert _reference_error(result) == "ObjectNotFound"
        assert "black can" in result.data["visible"]
        assert world.servo_targets == []
        assert memory.get_held_object() is None

    def test_grasping_on_air_near_nothing_is_a_failure_not_a_success(self):
        """Target resolved, but the policy closes where it stands and the sim
        (air_grasp) reports a grasp. The object never came near the hand."""
        executor, world, client, memory, events, _ = _build(
            BLUE_CAN, close_in_place_policy, air_grasp=True
        )

        result = executor.execute("pick", {"target": "blue can", "max_iterations": 2})

        assert world.get_gripper_state().is_grasping, "precondition: the flag lies"
        assert client.calls == 2
        assert result.status is SkillStatus.FAILED
        assert result.message.startswith("policy ran 2 iteration(s) / 16 step(s)")
        assert "no object grasped" in result.message
        assert memory.get_held_object() is None
        assert "track_id" not in result.data
        assert [p["goal_reached"] for p in events.named("gr00t.iteration")] == [False, False]

    def test_goal_test_without_a_resolved_target_is_never_reached(self):
        executor, world, *_ = _build(BLUE_CAN, close_in_place_policy, air_grasp=True)
        world.close_gripper()
        assert world.get_gripper_state().is_grasping
        assert executor._goal_reached("pick") is False

    def test_closing_beside_the_object_without_lifting_is_not_a_pick(self):
        """At the fingertips is not enough on a depth lane: it must follow the hand up."""
        executor, world, _, memory, _, _ = _build(
            BLUE_CAN, reach_policy(CAN_AT, lift_m=0.0)
        )
        result = executor.execute("pick", {"target": "blue can", "max_iterations": 8})

        assert world.attached == "obj_7", "precondition: the fingers really closed on it"
        assert result.status is SkillStatus.FAILED
        assert "not been lifted" in result.message
        assert memory.get_held_object() is None

    def test_a_real_grasp_and_lift_succeeds_and_is_recorded(self):
        executor, world, _, memory, _, context = _build(BLUE_CAN, reach_policy(CAN_AT))
        result = executor.execute("pick", {"target": "blue can", "max_iterations": 10})

        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"
        assert result.data["track_id"] == "obj_7"
        assert world.objects["obj_7"][2][2] - CAN_AT[2] >= MIN_LIFT_M
        assert context.held_grasp is None, "a stale classical grasp record must not survive"

    def test_depthless_lane_does_not_demand_a_measured_lift(self):
        """Without depth (gripper_feedback false) a lift cannot be measured, so it is not required."""
        executor, _, _, memory, _, _ = _build(
            BLUE_CAN, reach_policy(CAN_AT, lift_m=0.0), gripper_feedback=False
        )
        result = executor.execute("pick", {"target": "blue can", "max_iterations": 8})
        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"

    def test_pick_without_memory_is_refused_before_the_policy(self):
        executor, _, client, _, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT), memory=False)
        result = executor.execute("pick", {"target": "blue can"})
        assert result.status is SkillStatus.INFEASIBLE
        assert client.calls == 0

    def test_pick_while_holding_is_refused_before_the_policy(self):
        executor, _, client, memory, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        memory.set_held_object("obj_3")
        result = executor.execute("pick", {"target": "blue can"})
        assert result.status is SkillStatus.INFEASIBLE
        assert "already holding" in result.message
        assert client.calls == 0


# ---------------------------------------------------------------------------
# 2. grounding before the policy
# ---------------------------------------------------------------------------


class TestGroundingFirst:
    def test_ambiguous_target_asks_instead_of_running(self):
        two_cans = {"obj_7": ("can", "blue", CAN_AT), "obj_8": ("can", "blue", (0.5, 0.2, 0.45))}
        executor, _, client, _, _, _ = _build(two_cans, reach_policy(CAN_AT))

        result = executor.execute("pick", {"target": "can"})

        assert client.calls == 0
        assert result.status is SkillStatus.FAILED
        assert _reference_error(result) == "AmbiguousReference"
        assert sorted(result.data["track_ids"]) == ["obj_7", "obj_8"]
        assert len(result.data["candidates"]) == 2

    def test_colour_and_class_select_one_of_several(self):
        scene = {"obj_7": ("can", "blue", CAN_AT), "obj_8": ("can", "red", (0.5, 0.2, 0.45))}
        executor, _, client, memory, _, _ = _build(scene, reach_policy(CAN_AT))
        result = executor.execute("pick", {"target": "the blue can", "max_iterations": 10})
        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"
        assert client.instructions[0] == "pick up the blue can"

    def test_pronoun_resolves_through_memory_and_the_policy_hears_the_name(self):
        executor, _, client, memory, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        memory.note_reference("obj_7")
        result = executor.execute("pick", {"target": "it", "max_iterations": 10})
        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"
        assert client.instructions[0] == "pick up the blue can", "never 'pick up the it'"

    def test_unresolvable_place_destination_fails_before_the_policy(self):
        executor, world, client, memory, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        world.pose = Pose(np.asarray(CAN_AT), TOP_DOWN_QUAT, Frame.WORLD)
        world.close_gripper()
        memory.set_held_object("obj_7")

        result = executor.execute("place", {"target": "green box"})

        assert client.calls == 0
        assert _reference_error(result) == "ObjectNotFound"
        assert memory.get_held_object() == "obj_7", "nothing moved, so nothing changed"


    def test_a_target_missing_from_the_cached_scene_gets_one_more_look(self):
        """Same rule as the classical skills: an aged-out track is re-confirmed by a second look.

        Measured (events 20260924_104645_0cc762 seq 46): after a pick and place
        the can's track had expired and the next scene held only the red block.
        """
        executor, world, client, memory, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        world.require_fresh_scene = lambda: SceneGraph(objects={}, sim_time=0.0, step_index=0)
        looks = []  # policy calls made before each observation
        real_observe = world.observe
        world.observe = lambda: looks.append(client.calls) or real_observe()

        result = executor.execute("pick", {"target": "blue can", "max_iterations": 10})

        assert looks.count(0) == 1, "exactly one extra observation before the policy runs"
        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"

    def test_a_second_miss_still_refuses_without_running_the_policy(self):
        executor, world, client, _, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        looks = []
        real_observe = world.observe
        world.observe = lambda: looks.append(1) or real_observe()

        result = executor.execute("pick", {"target": "green box"})

        assert looks == [1]
        assert client.calls == 0
        assert _reference_error(result) == "ObjectNotFound"

    def test_ambiguity_is_not_retried(self):
        two_cans = {"obj_7": ("can", "blue", CAN_AT), "obj_8": ("can", "blue", (0.5, 0.2, 0.45))}
        executor, world, client, _, _, _ = _build(two_cans, reach_policy(CAN_AT))
        looks = []
        world.observe = lambda: looks.append(1)

        result = executor.execute("pick", {"target": "can"})

        assert looks == []
        assert _reference_error(result) == "AmbiguousReference"


# ---------------------------------------------------------------------------
# 3. place needs a release
# ---------------------------------------------------------------------------


def _holding(executor, world, memory):
    world.pose = Pose(np.asarray(CAN_AT), TOP_DOWN_QUAT, Frame.WORLD)
    world.close_gripper()
    assert world.attached == "obj_7"
    memory.set_held_object("obj_7")


class TestPlaceNeedsARelease:
    def test_already_open_gripper_is_refused_before_the_policy(self):
        """Audit P3: memory says held, the hand is open -> the old goal test passed at once."""
        executor, _, client, memory, _, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        memory.set_held_object("obj_7")  # but the gripper is fully open

        result = executor.execute("place", {"max_iterations": 3})

        assert result.status is SkillStatus.INFEASIBLE
        assert "already open" in result.message
        assert client.calls == 0
        assert memory.get_held_object() == "obj_7"

    def test_policy_that_never_opens_does_not_place(self):
        executor, world, client, memory, _, context = _build(BLUE_CAN, close_in_place_policy)
        _holding(executor, world, memory)

        result = executor.execute("place", {"max_iterations": 2})

        assert client.calls == 2
        assert result.status is SkillStatus.FAILED
        assert "was not released" in result.message
        assert memory.get_held_object() == "obj_7"
        assert context.held_grasp == "stale classical record", "untouched on failure"

    def test_release_clears_memory_and_the_classical_grasp_record(self):
        executor, world, _, memory, _, context = _build(BLUE_CAN, reach_policy(CAN_AT))
        _holding(executor, world, memory)

        result = executor.execute("place", {"max_iterations": 3})

        assert result.status is SkillStatus.SUCCESS, result.message
        assert result.data["track_id"] == "obj_7"
        assert "released" in result.message
        assert memory.get_held_object() is None
        assert context.held_grasp is None
        assert world.attached is None


# ---------------------------------------------------------------------------
# 4. observability
# ---------------------------------------------------------------------------


class TestIterationEvents:
    def test_every_iteration_carries_the_motion_evidence(self):
        executor, _, _, _, events, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        executor.execute("pick", {"target": "blue can", "max_iterations": 10})

        iterations = events.named("gr00t.iteration")
        assert iterations
        per_chunk = executor.config.gr00t.actions_executed_per_chunk
        for payload in iterations:
            missing = [f for f in ITERATION_FIELDS if f not in payload]
            assert not missing, missing
            assert payload["instruction"] == "pick up the blue can"
            start, end = np.array(payload["tcp_start"]), np.array(payload["tcp_end"])
            assert payload["tcp_displacement_m"] == pytest.approx(
                float(np.linalg.norm(end - start)), abs=1e-4
            )
            assert len(payload["gripper_commanded_m"]) == payload["steps"] == per_chunk
            assert len(payload["gripper_measured_m"]) == per_chunk
            eef = payload["action_chunk"]["eef_9d"]
            assert eef["shape"] == [1, HORIZON, 9]
            assert len(eef["min"]) == len(eef["max"]) == 9
            assert payload["action_chunk"]["gripper_position"]["shape"] == [1, HORIZON, 1]

        first = iterations[0]
        # 8 steps of 5 mm straight toward the can.
        assert first["tcp_displacement_m"] == pytest.approx(0.04, abs=1e-3)
        assert first["max_step_translation_m"] == pytest.approx(0.005, abs=1e-4)
        assert iterations[-1]["goal_reached"] is True
        assert "close" in [c for p in iterations for c in p["gripper_commands"]]

    def test_a_policy_that_does_not_move_shows_zero_displacement(self):
        """The field that disproves motion: the real run's 8 'steps' could be this."""
        executor, _, _, _, events, _ = _build(BLUE_CAN, close_in_place_policy, air_grasp=True)
        executor.execute("pick", {"target": "blue can", "max_iterations": 1})
        (payload,) = events.named("gr00t.iteration")
        assert payload["tcp_displacement_m"] == 0.0
        assert payload["max_step_translation_m"] == 0.0
        assert payload["gripper_closure_commanded"] == [1.0] * payload["steps"]
        assert payload["gripper_commanded_m"] == [0.0] * payload["steps"]
        assert payload["goal_reached"] is False
        assert "mm from the TCP" in payload["goal_reason"]

    def test_servo_refusals_are_counted_as_rejected(self):
        executor, world, _, _, events, _ = _build(BLUE_CAN, reach_policy(CAN_AT))
        world.refuse_servo = True
        result = executor.execute("pick", {"target": "blue can", "max_iterations": 2})
        assert result.status is SkillStatus.FAILED
        assert result.data["rejected_actions"] == 2
        assert [p["rejected"] for p in events.named("gr00t.iteration")] == [1, 1]
        assert all(p["tcp_displacement_m"] == 0.0 for p in events.named("gr00t.iteration"))

    def test_a_safety_violation_is_logged_then_raised(self):
        def nan_policy(observation):
            poses = np.full((HORIZON, 9), np.nan)
            return _chunk(poses, OPEN)

        executor, _, _, _, events, _ = _build(BLUE_CAN, nan_policy)
        with pytest.raises(SafetyViolation):
            executor.execute("pick", {"target": "blue can", "max_iterations": 2})
        (payload,) = events.named("gr00t.iteration")
        assert payload["rejected"] == 1
        assert "non-finite" in payload["safety_violation"]
        assert payload["goal_reached"] is False

    def test_the_real_event_log_round_trips_the_evidence_as_json(self, tmp_path):
        """What the orchestrator reads is events.jsonl, so check it there."""
        import json

        from mfw.utils.logging import EventLogger

        config = _config()
        world = World(config, BLUE_CAN)
        client = FakeClient(reach_policy(CAN_AT))
        log = EventLogger(log_dir=tmp_path, run_id="groot", console=False)
        try:
            executor = Gr00tExecutor(
                client=client, robot=world, vision=world, controller=world,
                cameras=world.cameras(), config=config,
                memory=WorkingMemory(MemoryConfig()), events=log,
            )
            result = executor.execute("pick", {"target": "blue can", "max_iterations": 10})
        finally:
            log.close()

        records = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
        iterations = [r for r in records if r["event"] == "gr00t.iteration"]
        assert result.ok and len(iterations) == result.data["iterations"]
        for record in iterations:
            assert all(f in record for f in ITERATION_FIELDS)
            assert record["action_chunk"]["eef_9d"]["shape"] == [1, HORIZON, 9]
        (outcome,) = [r for r in records if r["event"] == "gr00t.result"]
        assert outcome["status"] == "success"
        assert sum(r["tcp_displacement_m"] for r in iterations) > 0.1

    def test_chunk_stats_reports_raw_shape_and_per_dimension_extremes(self):
        eef = np.zeros((1, 3, 9))
        eef[0, :, 2] = [0.1, 0.5, 0.3]
        stats = chunk_stats({"eef_9d": eef, "gripper_position": np.array([[[0.0], [1.0], [0.5]]]),
                             "note": "not numeric"})
        assert stats["eef_9d"]["shape"] == [1, 3, 9]
        assert stats["eef_9d"]["min"][2] == 0.1 and stats["eef_9d"]["max"][2] == 0.5
        assert stats["gripper_position"]["max"] == [1.0]
        assert "note" not in stats


# ---------------------------------------------------------------------------
# 5. the gripper speaks DROID closure, not metres
# ---------------------------------------------------------------------------


class TestDroidGripperConvention:
    """N1.7's ``gripper_position`` is closure: 0 open .. 1 closed.

    The checkpoint's statistics.json gives min 0.0 / max 1.0 for both state and
    action, and NVIDIA's DROID client binarises the action at 0.5 with 1 =
    close. The executor used to read it as a width clipped to [0, 0.08 m] and
    split at 0.04 m, so "close" (1.0) opened the hand and "open" (0.0) closed
    it -- consistent with the real run closing on air on its first chunk.
    """

    def test_the_observed_state_is_closure(self):
        from mfw.gr00t_bridge.observation import ObservationBuilder

        config = _config()
        builder = ObservationBuilder(config.gr00t, 0.08, 0.0)
        frame = np.zeros((120, 160, 3), np.uint8)
        builder.prime(frame, frame)
        world = World(config, BLUE_CAN)

        def closure_at(width):
            world.width = width
            state = builder.build(world.get_state(), "x")["state"]["gripper_position"]
            return float(state.reshape(-1)[0])

        assert closure_at(0.08) == 0.0, "fully open is closure 0, not 0.08"
        assert closure_at(0.0) == 1.0
        assert closure_at(0.06) == pytest.approx(0.25), "closed on a 60 mm can"

    @pytest.mark.parametrize("closure, expected", [(CLOSE, "close"), (0.9, "close"),
                                                   (OPEN, "open"), (0.2, "open")])
    def test_policy_closure_drives_the_gripper_the_right_way(self, closure, expected):
        def policy(observation):
            position, rot6d, _, _ = _state_of(observation)
            poses = np.tile(np.concatenate([position, rot6d]), (HORIZON, 1))
            return _chunk(poses, closure)

        executor, world, _, _, events, _ = _build(BLUE_CAN, policy)
        if expected == "open":
            world.width = 0.0  # start closed so an open is visible
        executor.execute("move_to", {"target": "can", "max_iterations": 1})

        (payload,) = events.named("gr00t.iteration")
        assert set(payload["gripper_commands"]) == {expected}
        open_width = executor.config.robot.gripper_open_width
        assert world.width == (open_width if expected == "open" else 0.0)
        assert payload["gripper_closure_commanded"][0] == pytest.approx(closure, abs=1e-6)

    def test_a_non_finite_gripper_command_is_a_safety_violation(self):
        def policy(observation):
            position, rot6d, _, _ = _state_of(observation)
            poses = np.tile(np.concatenate([position, rot6d]), (HORIZON, 1))
            return _chunk(poses, float("nan"))

        executor, world, *_ = _build(BLUE_CAN, policy)
        with pytest.raises(SafetyViolation, match="gripper"):
            executor.execute("move_to", {"target": "can", "max_iterations": 1})
        assert world.servo_targets == []

    def test_conversion_round_trips_and_clips(self):
        from mfw.gr00t_bridge.observation import (
            gripper_closure,
            gripper_width_from_closure,
        )

        for width in (0.0, 0.02, 0.04, 0.08):
            assert gripper_width_from_closure(gripper_closure(width, 0.08, 0.0), 0.08, 0.0) == \
                pytest.approx(width)
        assert gripper_width_from_closure(0.5, 0.08, 0.0) == pytest.approx(0.04)
        assert gripper_width_from_closure(1.7, 0.08, 0.0) == 0.0
        assert gripper_width_from_closure(-0.3, 0.08, 0.0) == 0.08


class TestAssistantWiresTheSkillContext:
    """The executor can only clear a stale classical ``held_grasp`` if it is given the context."""

    def test_attach_gr00t_passes_the_classical_skill_context(self, monkeypatch):
        from mfw.assistant import Assistant
        from mfw.gr00t_bridge.client import Gr00tTcpClient

        monkeypatch.setattr(Gr00tTcpClient, "connect", lambda self: None)
        config = load_config(
            overrides={"gr00t": {"enabled": True, "use_mock_server": True, "port": 5555}}
        )
        context = types.SimpleNamespace(held_grasp="stale classical record")
        skills = types.SimpleNamespace(_context=context)
        runtime = types.SimpleNamespace(
            skills=skills, robot=None, vision=None, controller=None, cameras={},
            memory=None, events=None,
        )
        assistant = Assistant.__new__(Assistant)
        assistant.config = config
        assistant.runtime = runtime
        assistant.executors = {}
        assistant._gr00t_client = None

        assistant._attach_gr00t()

        assert assistant.executors["gr00t"].skill_context is context

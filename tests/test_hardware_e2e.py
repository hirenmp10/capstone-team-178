"""Phase 9 gate: the hardware lane end to end, against in-process fakes.

No Isaac Sim, no servos, no webcam. ``jetson/robot_server.py`` runs in a
thread on a ``FakeDriver`` and ``jetson/detector_service.py`` runs in a thread
on a ``ScriptedDetector``; between them sits :class:`FakeWorld`, which is the
only piece of physics here: it watches every pulse the fake driver writes,
computes the TCP through the same kinematics the planner uses, and attaches
the nearest object when the jaw closes on it. Attached objects follow the
TCP.

The scripted camera runs in its *honest* mode: a lifted marker stays in view,
its raised box projected like any other (so it grows ~9 % and its table-plane
estimate shifts 10-30 mm by parallax), and objects are hidden only under the
projected jaw footprint. That is what the real overhead webcam produces, and
the feedback-less verifier has to reach "carried" from it. The review found
the previous version of this file green only because the detector deleted
every object lifted above 4 cm (``HIDE_ABOVE_Z``) and the verifier read the
absence as a carry; :class:`TestVerdictPaths` keeps that legacy rule only to
show the verdict now treats an unexplained absence as *unknown*, and pins the
other outcomes a student will meet: a missed grasp, a knocked object.

Why this is the demo-of-record for Stage 0: every laptop-side module (config,
kinematics, transport, perception, grasp synthesis, planner, controller,
skills, task planner, memory, language) runs in its production wiring;
only the two network peers are fake, and they are fake at the *protocol*
boundary, so swapping in the Jetson is a host/port change.

Wall time: the fake clock makes settling free and the robot server is built
with a no-op follower sleep, so a 2 s trajectory costs one round trip. The
whole pick-and-place stays well under the 20 s budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from jetson.detector_service import DetectorServer, ScriptedDetector, SyntheticPinhole
from jetson.robot_server import FakeDriver, RobotServer, ServoCalibration
from mfw.config.schema import load_config
from mfw.core.errors import ExecutionError, SafetyViolation
from mfw.core.types import Frame, Pose, Trajectory, Waypoint
from mfw.hardware.clock import WallClock
from mfw.hardware.controller import RemoteController
from mfw.hardware.jetson_client import JetsonClient
from mfw.hardware.kinematics import PlanarKinematics
from mfw.hardware.remote_arm import RemoteArm
from mfw.planner.state_machine import State

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_FAKE_YAML = REPO_ROOT / "configs" / "hardware_fake.yaml"

MARKER_XY = (0.18, 0.05)
BOWL_XY = (0.15, -0.12)

#: The legacy scripted rule (lifted objects vanish); used only by
#: TestVerdictPaths.test_legacy_hide_rule_is_not_evidence_of_a_carry.
from jetson.detector_service import HIDE_ABOVE_Z  # noqa: E402


# ----------------------------------------------------------------------
# the fake world
# ----------------------------------------------------------------------


class FakeWorld:
    """Objects on the table, attached to the jaw when it closes on them.

    Sizes are the scripted detector's footprints (``x, y, z`` metres). The
    jaw-relevant width is the narrow side. Rules, all in metres:

    * attach when the commanded width drops below ``width + 0.005`` while the
      TCP is within 2 cm (XY) of the object and below its top + 1 cm;
    * an attached object's centre follows the TCP in XY and z;
    * release when the width opens beyond ``width + 0.005``; the object drops
      to the table under the TCP.

    ``grip`` scripts the physical outcome of a close: ``"grip"`` (the rules
    above), ``"slip"`` (the jaw closes on air: nothing ever attaches),
    ``"knock"`` (the closing jaw shoves the object ``KNOCK_M`` sideways, +Y,
    and does not hold it) or ``"spin"`` (the jaw catches the object
    off-centre and spins it ``spin_rad`` about its own centre, in place, and
    does not hold it -- the re-review's critical false "carried"). ``yaw``
    gives an object's heading on the table.
    """

    KNOCK_M = 0.05

    ATTACH_XY_M = 0.02
    ATTACH_Z_ABOVE_TOP_M = 0.01
    WIDTH_SLACK_M = 0.005

    def __init__(
        self,
        kinematics: PlanarKinematics,
        objects: dict[str, tuple[float, float]],
        footprints: dict[str, tuple[float, float, float]],
        grip: str = "grip",
        yaw: dict[str, float] | None = None,
        spin_rad: float = 0.35,
    ) -> None:
        self._kin = kinematics
        self.spin_rad = float(spin_rad)
        self.footprints = footprints
        self.objects: dict[str, np.ndarray] = {
            label: np.array([x, y, 0.0], dtype=np.float64) for label, (x, y) in objects.items()
        }
        self.grip = grip
        self.yaw = dict(yaw or {})
        self.attached: str | None = None
        self.tcp: np.ndarray | None = None
        self.last_width: float | None = None
        self.commands = 0
        self.attach_events: list[tuple[str, np.ndarray]] = []
        self.knocked: list[str] = []

    def scene(self) -> dict[str, tuple[float, float, float, float]]:
        """Live positions for the scripted detector (base z, yaw)."""
        return {
            label: (float(p[0]), float(p[1]), float(p[2]), float(self.yaw.get(label, 0.0)))
            for label, p in self.objects.items()
        }

    def jaw_width(self, label: str) -> float:
        sx, sy, _ = self.footprints[label]
        return float(min(sx, sy))

    def on_command(self, q: np.ndarray, width: float) -> None:
        """``FakeDriver.on_command`` hook: one call per pulse write."""
        self.commands += 1
        tcp = np.asarray(self._kin.fk(q).position, dtype=np.float64)
        self.tcp = tcp
        self.last_width = float(width)
        if self.attached is not None:
            label = self.attached
            if width > self.jaw_width(label) + self.WIDTH_SLACK_M:
                self.objects[label] = np.array([tcp[0], tcp[1], 0.0])
                self.attached = None
            else:
                self.objects[label] = tcp.copy()
            return
        best: tuple[float, str] | None = None
        for label, pos in self.objects.items():
            if width >= self.jaw_width(label) + self.WIDTH_SLACK_M:
                continue
            xy = float(np.linalg.norm(tcp[:2] - pos[:2]))
            top = float(pos[2] + self.footprints[label][2])
            if xy <= self.ATTACH_XY_M and tcp[2] <= top + self.ATTACH_Z_ABOVE_TOP_M:
                if best is None or xy < best[0]:
                    best = (xy, label)
        if best is not None:
            label = best[1]
            if self.grip == "slip":
                return
            if self.grip == "knock":
                if label not in self.knocked:
                    self.knocked.append(label)
                    self.objects[label] = self.objects[label] + np.array([0.0, self.KNOCK_M, 0.0])
                return
            if self.grip == "spin":
                if label not in self.knocked:
                    self.knocked.append(label)
                    self.yaw[label] = self.yaw.get(label, 0.0) + self.spin_rad
                return
            self.attached = label
            self.attach_events.append((label, tcp.copy()))
            self.objects[label] = tcp.copy()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def _load_fake_config(robot_port: int, detector_port: int, tmp_path: Path, **extra: Any):
    overrides: dict[str, Any] = {
        "hardware": {"jetson_port": robot_port, "detector_port": detector_port},
        "logging": {"console": False, "log_dir": str(tmp_path / "logs")},
        "memory": {"persist_path": str(tmp_path / "memory_state.json")},
    }
    for key, value in extra.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(HARDWARE_FAKE_YAML, overrides=overrides)


@pytest.fixture
def lane_factory(tmp_path):
    """Build robot server + detector server + world lanes; all stopped at teardown.

    ``hide_above_z=None`` is the honest scripted camera (the jaw footprint,
    from the world's TCP, is the only occluder); a number is the legacy rule.
    """
    stoppers: list[Callable[[], None]] = []

    def build(grip: str = "grip", hide_above_z: float | None = None,
              objects: dict[str, tuple[float, float]] | None = None, yaw: dict[str, float] | None = None,
              spin_rad: float = 0.35, detector_kw: dict[str, Any] | None = None):
        base = load_config(HARDWARE_FAKE_YAML)
        kinematics = PlanarKinematics(base.hardware.arm, base.robot.arm_joint_names)
        footprints = {k: tuple(v) for k, v in base.hardware.object_sizes.items()}
        world = FakeWorld(kinematics, objects or {"marker": MARKER_XY, "bowl": BOWL_XY}, footprints,
                          grip=grip, yaw=yaw, spin_rad=spin_rad)

        driver = FakeDriver(ServoCalibration.default(), on_command=world.on_command)
        robot_server = RobotServer(
            "127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
            follower_sleep=lambda _s: None,
        )
        _, robot_port = robot_server.serve_in_thread(port=0)
        stoppers.append(robot_server.stop)

        detector = ScriptedDetector(world.scene, SyntheticPinhole(), hide_above_z=hide_above_z,
                                    arm=lambda: world.tcp, **(detector_kw or {}))
        detector_server = DetectorServer("127.0.0.1", 0, detector)
        _, detector_port = detector_server.serve_in_thread(port=0)
        stoppers.append(detector_server.stop)

        cfg = _load_fake_config(robot_port, detector_port, tmp_path)
        assert cfg.exterior_camera.pose_measured, "the scripted camera is exact; hardware_fake.yaml must say so"
        return cfg, world, robot_server, detector_server, driver

    try:
        yield build
    finally:
        for stop in reversed(stoppers):
            stop()


@pytest.fixture
def fake_lane(lane_factory):
    """Robot server + detector server + world, wired to a fresh fake config (honest camera)."""
    return lane_factory()


@pytest.fixture
def remote_arm(fake_lane):
    """A connected RemoteArm + fast clock + RemoteController on the fake server."""
    cfg, world, robot_server, _detector_server, driver = fake_lane
    clock = WallClock(cfg.simulation.physics_dt, cfg.simulation.settle_steps, fast=True)
    client = JetsonClient(cfg.hardware.jetson_host, cfg.hardware.jetson_port,
                          cfg.hardware.request_timeout_s, cfg.hardware.trajectory_timeout_margin_s)
    client.connect()
    robot = None
    try:
        # Everything that can raise lives inside the try: a RemoteArm that refuses
        # to build (e.g. limits outside the pulse map) must not leak the client,
        # or the traceback keeps a live socket alive and pytest hangs at exit.
        kinematics = PlanarKinematics(cfg.hardware.arm, cfg.robot.arm_joint_names)
        robot = RemoteArm(client, kinematics, cfg.robot, clock)
        robot.go_home_immediate()
        controller = RemoteController(
            sim=clock, robot=robot, motion_config=cfg.motion, robot_config=cfg.robot,
            hardware_config=cfg.hardware,
            workspace_min=np.array(cfg.scene.workspace_min), workspace_max=np.array(cfg.scene.workspace_max),
        )
        yield cfg, robot, controller, kinematics, driver, robot_server
    finally:
        if robot is not None and hasattr(robot, "close"):
            robot.close()
        client.close()


#: A configuration whose TCP sits comfortably inside the workspace box (about
#: x=0.21, y=0.04, z=0.03 on the placeholder geometry). Chosen in joint space
#: so the test does not depend on which IK branch the wrist limit allows.
GOAL_Q = np.array([0.2, 0.9, 0.9, 0.9])


def _inside_goal(kin: PlanarKinematics, cfg) -> np.ndarray:
    tcp = kin.fk(GOAL_Q).position
    assert np.all(tcp >= np.array(cfg.scene.workspace_min)) and np.all(tcp <= np.array(cfg.scene.workspace_max)), tcp
    assert kin.within_limits(GOAL_Q)
    return GOAL_Q.copy()


def _two_point_trajectory(q0: np.ndarray, q1: np.ndarray, duration: float) -> Trajectory:
    return Trajectory(
        waypoints=(Waypoint(q0, 0.0), Waypoint(q1, duration)),
        joint_names=("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"),
        planner_name="test",
        planning_time_s=0.0,
    )


# ----------------------------------------------------------------------
# controller against the in-thread robot server
# ----------------------------------------------------------------------


class TestRemoteController:
    def test_chunks_and_on_step(self, remote_arm):
        cfg, robot, controller, kin, driver, _server = remote_arm
        home = robot.get_arm_joint_positions()
        goal = _inside_goal(kin, cfg)
        trajectory = _two_point_trajectory(home, goal, 3.0)

        seen: list[tuple[int, np.ndarray]] = []

        def on_step(index: int, q: np.ndarray) -> bool:
            seen.append((index, np.asarray(q)))
            return True

        writes_before = driver.write_count
        assert controller.follow_trajectory(trajectory, on_step=on_step)
        # 3 s at the controller grid, chunked into ~1 s pieces: several chunks,
        # one on_step per chunk, the last reporting the final waypoint.
        assert controller.chunks_shipped >= 3
        assert len(seen) == controller.chunks_shipped
        assert seen[-1][0] == int(np.ceil(3.0 / controller._effective_dt()))
        np.testing.assert_allclose(seen[-1][1], goal, atol=1e-6)
        np.testing.assert_allclose(robot.get_arm_joint_positions(), goal, atol=1e-6)
        assert driver.write_count > writes_before

    def test_on_step_false_aborts_between_chunks(self, remote_arm):
        cfg, robot, controller, kin, driver, _server = remote_arm
        home = robot.get_arm_joint_positions()
        goal = _inside_goal(kin, cfg)
        trajectory = _two_point_trajectory(home, goal, 3.0)
        calls = {"n": 0}

        def stop_after_first(_index: int, _q: np.ndarray) -> bool:
            calls["n"] += 1
            return False

        assert controller.follow_trajectory(trajectory, on_step=stop_after_first) is False
        assert calls["n"] == 1
        assert controller.chunks_shipped == 1

    def test_safety_violation_before_any_motion(self, remote_arm):
        cfg, robot, controller, kin, driver, _server = remote_arm
        home = robot.get_arm_joint_positions()
        # Base swung hard left with the arm stretched: the TCP leaves the box in y.
        outside = np.array([1.5, 1.5, 0.0, 0.0])
        assert not np.allclose(outside, home)
        tcp = kin.fk(outside).position
        assert tcp[1] > cfg.scene.workspace_max[1]

        writes_before = driver.write_count
        with pytest.raises(SafetyViolation):
            controller.follow_trajectory(_two_point_trajectory(home, outside, 2.0))
        assert driver.write_count == writes_before
        assert controller.chunks_shipped == 0
        np.testing.assert_allclose(robot.get_arm_joint_positions(), home, atol=1e-9)

    def test_home_goal_is_exempt_from_the_workspace_box(self, remote_arm):
        cfg, robot, controller, kin, driver, _server = remote_arm
        home = robot.get_arm_joint_positions()
        tcp_home = kin.fk(home).position
        low, high = np.array(cfg.scene.workspace_min), np.array(cfg.scene.workspace_max)
        assert np.any(tcp_home < low) or np.any(tcp_home > high), "home is outside the box by design"
        goal = _inside_goal(kin, cfg)
        assert controller.follow_trajectory(_two_point_trajectory(home, goal, 1.0))
        assert controller.follow_trajectory(_two_point_trajectory(goal, home, 1.0))

    def test_emergency_stop_never_touches_articulation(self, remote_arm):
        cfg, robot, controller, kin, driver, server = remote_arm
        with pytest.raises(ExecutionError):
            _ = robot.articulation
        controller.emergency_stop()
        assert server.state()["estopped"] is True
        assert driver.commands[-1] == ("D", ())
        robot.refresh()
        assert robot.estopped is True
        # Motion is refused until cleared, and refused cleanly (no exception
        # escapes the controller).
        home = robot.get_arm_joint_positions()
        goal = _inside_goal(kin, cfg)
        assert controller.follow_trajectory(_two_point_trajectory(home, goal, 0.5)) is False
        robot.clear_estop()
        assert controller.follow_trajectory(_two_point_trajectory(home, goal, 0.5)) is True

    def test_planned_grasp_width_reaches_the_jaw_servo(self, remote_arm, fake_lane):
        """What the servo is driven to, read off the fake world's pulse hook --
        not the controller's own echo of what it commanded (review F5)."""
        cfg, robot, controller, kin, driver, _server = remote_arm
        _cfg, world, *_ = fake_lane
        # No grasp planned: an empty close drives the jaw fully shut.
        controller.close_gripper_blocking()
        assert world.last_width == pytest.approx(cfg.robot.gripper_closed_width, abs=1e-3)
        controller.open_gripper_blocking()
        assert world.last_width == pytest.approx(cfg.robot.gripper_open_width, abs=1e-3)
        # The marker's chord: the jaw squeezes to just under it, not to zero.
        controller.set_grasp_width(0.019)
        closed = controller.close_gripper_blocking()
        assert world.last_width == pytest.approx(closed, abs=1e-3)
        assert cfg.robot.gripper_closed_width < world.last_width < 0.019
        # Opening forgets the planned width: the next empty close is a full close.
        controller.open_gripper_blocking()
        controller.close_gripper_blocking()
        assert world.last_width == pytest.approx(cfg.robot.gripper_closed_width, abs=1e-3)
        assert robot.get_gripper_state().is_grasping is False


# ----------------------------------------------------------------------
# the whole assistant
# ----------------------------------------------------------------------


def _labels(outcome) -> set[str]:
    objects = (outcome.result.data if outcome.result is not None else {}).get("objects", [])
    return {o["label"] for o in objects}


def _states(outcome) -> list[str]:
    return [str(entry.get("to", entry.get("state", ""))) for entry in outcome.state_trace]


def _events(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class TestEndToEnd:
    def test_see_pick_place(self, fake_lane):
        cfg, world, _robot_server, _detector_server, _driver = fake_lane
        from mfw.assistant import Assistant

        started = time.perf_counter()
        assistant = Assistant(config=cfg)
        try:
            skills = set(assistant.describe()["skills"])
            assert skills.isdisjoint({"look_at", "rotate_wrist"})
            # scan_scene is the fixed-camera look-sweep-report, not the sim's wrist sweep.
            from mfw.skills.primitives import FixedCameraScan, HardwareObserve

            assert "scan_scene" in skills
            assert type(assistant.runtime.skills.get("scan_scene")) is FixedCameraScan
            assert type(assistant.runtime.skills.get("observe")) is HardwareObserve

            seen = assistant.command("what do you see")
            assert seen.ok, seen.message
            assert {"marker", "bowl"} <= _labels(seen), seen.result.data
            marker = next(o for o in seen.result.data["objects"] if o["label"] == "marker")
            assert np.linalg.norm(np.array(marker["position"][:2]) - np.array(MARKER_XY)) < 0.015

            picked = assistant.command("pick up the marker")
            assert picked.ok, picked.message
            assert State.COMPLETE.value in _states(picked)
            assert world.attached == "marker"
            assert world.objects["marker"][2] > 0.05
            assert assistant.runtime.memory.get_held_object() == picked.result.data["track_id"]
            # The jaw closed to the planned chord, not to zero (the MG90S stall).
            assert 0.0 < world.last_width < cfg.hardware.object_sizes["marker"][1]

            placed = assistant.command("place it in the bowl")
            assert placed.ok, placed.message
            assert world.attached is None
            assert np.linalg.norm(world.objects["marker"][:2] - np.array(BOWL_XY)) < 0.03
            # The skill's own settled offset is horizontal on this lane: the
            # estimator pins z to the table, so a 3-D offset read ~70 mm
            # "below" a correct release into the bowl.
            assert placed.result.data["settled_offset_axes"] == "xy"
            assert placed.result.data["settled_offset"] < 0.015
            assert "horizontally" in placed.message

            looked = assistant.command("look at the marker")
            assert not looked.ok
            assert looked.skill in (None, "look_at")

            events = _events(assistant.runtime.events.path)
        finally:
            assistant.close()
        elapsed = time.perf_counter() - started
        assert elapsed < 20.0, f"end-to-end took {elapsed:.1f}s"

        verification = [e for e in events if e.get("event") == "pick.verification"]
        assert len(verification) == 1, "one grasp, one verdict"
        v = verification[0]
        assert v["gripper_feedback"] is False
        # The verdict path: the marker stayed in view, its box grew with the
        # lift, the tool projects inside it and it sits where a carried marker
        # would -- while its table-plane shift is under the old 30 mm gate.
        assert v["verdict"] == "carried"
        assert v["object_visible"] is True and v["visible_before"] is True
        # The reference box came from the standoff, before the descent -- not
        # from the grasp pose with the jaw parked over the marker.
        assert v["before_view"] == "standoff"
        assert v["pixel_grew"] is True and v["pixel_scale"] > 1.04
        assert v["tcp_in_object_box"] is True and v["on_prediction"] is True
        assert v["object_displaced"] is False
        assert v["lift_prediction"]["model_available"] is True
        follows = [e for e in events if e.get("event") == "controller.follow_trajectory"]
        assert any(e.get("outcome") == "completed" for e in follows)

    def test_pick_of_an_unseen_object_is_refused(self, fake_lane):
        cfg, world, _rs, _ds, _driver = fake_lane
        from mfw.assistant import Assistant

        assistant = Assistant(config=cfg)
        try:
            outcome = assistant.command("pick up the banana")
            assert not outcome.ok
            assert world.attached is None
            outcome = assistant.command("pick up the bowl")
            assert not outcome.ok  # 150 mm across a 35 mm jaw
            assert world.attached is None
        finally:
            assistant.close()


def _verdicts(assistant) -> list[dict[str, Any]]:
    return [e for e in _events(assistant.runtime.events.path) if e.get("event") == "pick.verification"]


class TestVerdictPaths:
    """What a student meets when the grasp does not go to plan (review F1/F3)."""

    def _pick(self, lane):
        cfg, world, *_ = lane
        from mfw.assistant import Assistant

        assistant = Assistant(config=cfg)
        try:
            assert assistant.command("what do you see").ok
            picked = assistant.command("pick up the marker")
            verdicts = _verdicts(assistant)
            held = assistant.runtime.memory.get_held_object()
            self.attached_after_pick = world.attached
            self.width_after_pick = world.last_width
            after = assistant.command("place it in the bowl")
            released = assistant.command("open the gripper")
        finally:
            assistant.close()
        return picked, verdicts, held, after, released, world

    def _assert_stopped_with_jaw_closed(self, picked, verdicts, held, after, world):
        assert not picked.ok
        # One verdict, no retry: a retry would open the jaw at lift height and
        # its own unrelated failure would replace this message.
        assert len(verdicts) == 1 and picked.attempts == 1
        assert picked.result.data["verification_failed"] is True
        assert picked.result.data["evidence"]["verdict"] == verdicts[0]["verdict"]
        assert "reachable" not in picked.message
        assert "jaw is left closed" in picked.message
        # Commanded to just under the marker's 19 mm chord and left there.
        assert self.width_after_pick is not None and self.width_after_pick < 0.019
        assert held is None
        assert not after.ok and "not holding" in after.message

    def test_missed_grasp_reports_the_verdict_and_keeps_the_jaw_closed(self, lane_factory):
        picked, verdicts, held, after, released, world = self._pick(lane_factory(grip="slip"))
        assert verdicts[0]["verdict"] == "resting" and verdicts[0]["object_visible"] is True
        assert verdicts[0]["pixel_grew"] is False
        assert "still resting" in picked.message
        self._assert_stopped_with_jaw_closed(picked, verdicts, held, after, world)
        assert world.attached is None
        assert released.ok

    def test_knocked_object_is_reported_as_knocked(self, lane_factory):
        picked, verdicts, held, after, released, world = self._pick(lane_factory(grip="knock"))
        assert world.knocked == ["marker"]
        assert verdicts[0]["verdict"] == "knocked" and verdicts[0]["object_displaced"] is True
        assert "knocked" in picked.message
        self._assert_stopped_with_jaw_closed(picked, verdicts, held, after, world)

    @pytest.mark.parametrize("spin_deg", [20.0, 30.0])
    def test_a_marker_the_jaw_spins_is_not_carried(self, lane_factory, spin_deg):
        """Re-review critical (gv/probe2.py): the jaw spun the marker 20-30 deg
        and never held it; pick said 'bbox grew 8 %', memory held the marker,
        and Place reported success 171 mm off with an empty jaw."""
        picked, verdicts, held, after, released, world = self._pick(
            lane_factory(grip="spin", spin_rad=np.radians(spin_deg)))
        assert world.knocked == ["marker"] and world.attached is None
        assert verdicts[0]["verdict"] in {"reshaped", "resting", "unknown"}, verdicts[0]
        assert verdicts[0]["holding"] is False
        assert "rose with the jaw" not in picked.message
        self._assert_stopped_with_jaw_closed(picked, verdicts, held, after, world)

    def test_a_detector_that_boxes_the_fingers_with_a_missed_marker_is_not_carried(self, lane_factory):
        picked, verdicts, held, after, released, world = self._pick(
            lane_factory(grip="slip", detector_kw={"merge_jaw": True}))
        assert world.attached is None
        assert verdicts[0]["holding"] is False and verdicts[0]["verdict"] != "carried", verdicts[0]
        self._assert_stopped_with_jaw_closed(picked, verdicts, held, after, world)

    def test_legacy_hide_rule_is_not_evidence_of_a_carry(self, lane_factory):
        """The old fake camera deleted every lifted object. The marker IS held
        here, but a 140 mm marker cannot hide under a 45 mm jaw, so its
        disappearance explains nothing: unknown, jaw stays shut, and memory and
        the operator are told the truth instead of "holding"."""
        picked, verdicts, held, after, released, world = self._pick(lane_factory(hide_above_z=HIDE_ABOVE_Z))
        assert verdicts[0]["verdict"] == "unknown" and verdicts[0]["object_visible"] is False
        self._assert_stopped_with_jaw_closed(picked, verdicts, held, after, world)
        # Physically still in the jaw until the human says so...
        assert self.attached_after_pick == "marker"
        # ...and "open the gripper" is what lets it go.
        assert released.ok and world.attached is None
        assert world.objects["marker"][2] == pytest.approx(0.0)

    def _see_and_pick(self, lane):
        cfg, world, *_ = lane
        from mfw.assistant import Assistant

        assistant = Assistant(config=cfg)
        try:
            seen = assistant.command("what do you see")
            picked = assistant.command("pick up the marker")
            events = _events(assistant.runtime.events.path)
        finally:
            assistant.close()
        marker = next(o for o in seen.result.data["objects"] if o["label"] == "marker")
        return marker, picked, events, world

    def test_marker_lying_across_the_ray_is_placed_and_measured_as_it_lies(self, lane_factory):
        """GEO-3, the review's case: marker along Y at (0.16, 0) seen by the
        fake camera looking along +X. It used to be placed 60 mm off and
        accepted with a 19 mm chord; now its yaw is read off the pixel box, it
        is placed where it lies, and the fixed tangential jaw -- which would
        have to span its 140 mm length -- honestly refuses it."""
        marker, picked, events, world = self._see_and_pick(
            lane_factory(objects={"marker": (0.16, 0.0)}, yaw={"marker": float(np.pi / 2.0)})
        )
        assert np.linalg.norm(np.array(marker["position"][:2]) - np.array([0.16, 0.0])) < 0.015
        assert marker["size"][:2] == pytest.approx([0.019, 0.14])
        assert not picked.ok and "not lying the way this jaw closes" in picked.message
        assert not [e for e in events if e.get("event") == "pick.grasp_selected"]
        assert world.attached is None and world.last_width > 0.04  # the jaw never closed

    def test_marker_along_the_jaw_reach_is_picked_at_its_true_centre(self, lane_factory):
        """Same yaw, but at a bearing where the tangential jaw closes across its
        19 mm side: the grasp lands on the marker and the chord is ~20 mm."""
        truth = np.array([0.08, 0.18])
        marker, picked, events, world = self._see_and_pick(
            lane_factory(objects={"marker": tuple(truth), "bowl": BOWL_XY}, yaw={"marker": float(np.pi / 2.0)})
        )
        assert np.linalg.norm(np.array(marker["position"][:2]) - truth) < 0.015
        chosen = [e for e in events if e.get("event") == "pick.grasp_selected"][0]["chosen"]
        assert np.linalg.norm(np.array(chosen["pose"]["position"][:2]) - truth) < 0.015
        assert 0.019 <= chosen["width"] < 0.025
        assert picked.ok, picked.message
        assert world.attached == "marker"

    def test_diagonal_marker_is_refused_with_a_reason(self, lane_factory):
        marker, picked, events, world = self._see_and_pick(lane_factory(yaw={"marker": float(np.pi / 4.0)}))
        assert not picked.ok and "which way" in picked.message
        assert world.attached is None


class TestJetsonFakeWorld:
    """The robot server's own fake world must agree with the laptop's kinematics."""

    def test_planar_tcp_matches_planar_kinematics(self):
        from jetson.robot_server import ArmGeometry, planar_tcp

        cfg = load_config(HARDWARE_FAKE_YAML)
        arm = cfg.hardware.arm
        kin = PlanarKinematics(arm, cfg.robot.arm_joint_names)
        geometry = ArmGeometry(arm.base_height, arm.shoulder_offset, arm.upper_arm, arm.forearm, arm.tool)
        rng = np.random.default_rng(9)
        for _ in range(200):
            q = rng.uniform(kin.lower, kin.upper)
            np.testing.assert_allclose(planar_tcp(q, geometry), kin.fk(q).position, atol=1e-12)

    def test_default_geometry_equals_hardware_yaml(self):
        from jetson.robot_server import ArmGeometry

        arm = load_config(HARDWARE_FAKE_YAML).hardware.arm
        g = ArmGeometry()
        assert (g.base_height, g.shoulder_offset, g.upper_arm, g.forearm, g.tool) == pytest.approx(
            (arm.base_height, arm.shoulder_offset, arm.upper_arm, arm.forearm, arm.tool)
        )

    def test_detector_follows_the_robot_servers_world(self):
        """Two standalone fakes, coupled over the wire: close the jaw on the
        marker through the robot server and lift it. The honest detector keeps
        seeing it -- bigger, and around the TCP it reads from the same world --
        while the legacy rule would have deleted it."""
        from jetson.detector_service import RobotWorldScene, parse_scene
        from jetson.robot_server import FakeWorld

        cfg = load_config(HARDWARE_FAKE_YAML)
        kin = PlanarKinematics(cfg.hardware.arm, cfg.robot.arm_joint_names)
        scene_text = "marker:0.18,0.05 bowl:0.15,-0.12"
        footprints = {k: tuple(v) for k, v in cfg.hardware.object_sizes.items()}
        world = FakeWorld(parse_scene(scene_text), footprints)
        driver = FakeDriver(ServoCalibration.default(), on_command=world.on_command)
        server = RobotServer("127.0.0.1", 0, driver, None, driver.calibration,
                             follower_sleep=lambda _s: None, fake_world=world)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port)
        try:
            source = RobotWorldScene("127.0.0.1", port, fallback=parse_scene(scene_text))
            detector = ScriptedDetector(source, SyntheticPinhole())
            legacy = ScriptedDetector(source, SyntheticPinhole(), hide_above_z=HIDE_ABOVE_Z)
            assert detector.honest and not legacy.honest
            rest = {d["label"]: d["bbox_px"] for d in detector.detect(None, ["marker", "bowl"])}
            assert set(rest) == {"marker", "bowl"}

            client.connect()
            client.home()  # the bring-up: a fresh bridge accepts nothing before home
            # Park the jaw on the marker with the 75-degree pitch the generator
            # picks for it: a pure vertical tool cannot be lifted 6 cm out here.
            from mfw.grasp.generator import build_grasp_pose
            from mfw.hardware.grasp import TopDownGraspGenerator

            approach, closing = TopDownGraspGenerator._frame(np.arctan2(0.05, 0.18), np.radians(15.0), "tangential")
            grasp = build_grasp_pose(np.array([0.18, 0.05, 0.0095]), closing, approach)
            q = kin.ik(grasp)
            assert q is not None
            client.follow_trajectory(q.reshape(1, -1), 0.05)
            client.set_gripper(0.0)
            assert world.attached == "marker"
            # Lift 6 cm: the marker leaves the detector's view.
            up = kin.ik(Pose(grasp.position + np.array([0.0, 0.0, 0.06]), grasp.quat, Frame.WORLD), seed=q)
            assert up is not None
            client.follow_trajectory(np.vstack([q, up]), 0.5)
            lifted = {d["label"]: d["bbox_px"] for d in detector.detect(None, ["marker", "bowl"])}
            assert set(lifted) == {"marker", "bowl"}
            # The occluder is the world's own TCP, read over the same wire.
            np.testing.assert_allclose(source.tcp(), world.snapshot()["tcp"])

            def diag(b):
                return float(np.hypot(b[2] - b[0], b[3] - b[1]))

            assert diag(lifted["marker"]) > 1.05 * diag(rest["marker"])
            u, v = SyntheticPinhole().project(np.asarray(world.snapshot()["tcp"]))
            box = lifted["marker"]
            assert box[0] <= u <= box[2] and box[1] <= v <= box[3]
            assert {d["label"] for d in legacy.detect(None, ["marker", "bowl"])} == {"bowl"}
            client.set_gripper(cfg.robot.gripper_open_width)
            assert world.attached is None
        finally:
            client.close()
            server.stop()


class TestRuntimeGuards:
    def test_empty_homography_refused_before_connecting(self, tmp_path):
        from mfw.core.errors import ConfigurationError
        from mfw.hardware.runtime import HardwareRuntime

        cfg = load_config(
            HARDWARE_FAKE_YAML,
            overrides={"exterior_camera": {"homography": []}, "logging": {"console": False, "log_dir": str(tmp_path)}},
        )
        with pytest.raises(ConfigurationError, match="calibrate_table"):
            HardwareRuntime(cfg)

    def test_missing_robot_server_is_reported_plainly(self, tmp_path):
        from mfw.hardware.runtime import HardwareRuntime

        cfg = load_config(
            HARDWARE_FAKE_YAML,
            overrides={
                "hardware": {"jetson_port": 1, "request_timeout_s": 0.2},
                "logging": {"console": False, "log_dir": str(tmp_path)},
            },
        )
        runtime = HardwareRuntime(cfg)
        try:
            with pytest.raises(ExecutionError, match="robot_server.py"):
                runtime.build()
        finally:
            runtime.close()


# ----------------------------------------------------------------------
# first home after a bridge reset: never sent silently to a real arm
# ----------------------------------------------------------------------


def _never_prompt(reason: str) -> Callable[..., str]:
    def prompt(*_a: Any) -> str:
        pytest.fail(reason)

    return prompt


class TestFirstHomeGate:
    """The bring-up ``home`` after every robot_server start attaches every
    servo AT home at full speed (the bridge knows no position). Against a
    real driver that must wait for the operator.

    The "real" driver here is the in-process FakeDriver with the server's
    ``fake`` flag cleared: the ping then says ``fake: false`` exactly as the
    Uno driver does, and the FakeDriver models the physical hazard the gate
    exists for -- straight after start ``position_known`` is False, and the
    first ``home`` attaches AT home in one P frame (pinned in
    ``test_hardware_bridge.py::TestSlowReattach``). "Nothing sent" is
    checked on the driver itself: no P frame, position still unknown.
    """

    @pytest.fixture
    def real_lane(self, lane_factory):
        cfg, _world, robot_server, _det, driver = lane_factory()
        robot_server.fake = False  # what ping reports for the uno/pca9685 drivers
        assert not driver.position_known and driver.write_count == 0
        return cfg, robot_server, driver

    @staticmethod
    def _runtime(cfg, gate):
        from mfw.hardware.runtime import HardwareRuntime

        return HardwareRuntime(cfg, first_home_gate=gate)

    def test_unknown_position_waits_for_the_typed_word_before_the_first_home(self, real_lane):
        """was: a bare Enter confirmed. now: the operator types ``home`` (fixer, 2026-09-28)."""
        import io

        from mfw.hardware.runtime import FirstHomeGate

        cfg, _server, driver = real_lane
        at_prompt: list[tuple[int, bool]] = []

        def enter(_text: str) -> str:
            at_prompt.append((driver.write_count, driver.position_known))
            return "home"

        out = io.StringIO()
        runtime = self._runtime(cfg, FirstHomeGate(prompt=enter, out=out, interactive=True))
        try:
            runtime.build()
            # Asked exactly once, and at that moment nothing had been pulsed.
            assert at_prompt == [(0, False)]
            assert runtime.first_home == "operator"
            # Only after Enter: the home was sent (P frames, position now known).
            assert driver.write_count >= 1 and driver.position_known
            text = out.getvalue()
            assert "Hand-pose the arm at home" in text
            assert "E-stop" in text and "clear of the arm" in text and "FULL SPEED" in text
        finally:
            runtime.close()

    def test_a_queued_enter_does_not_confirm(self, real_lane):
        """An Enter pressed during the multi-second bring-up is already in the
        console buffer when the prompt appears. The fake console models that
        buffer: ``flush`` discards it, and a bare line that still arrives is
        asked again, never a confirmation."""
        import io

        from mfw.hardware.runtime import FirstHomeGate

        cfg, _server, driver = real_lane
        typeahead = ["", ""]  # two Enters typed while the runtime was connecting
        answers = ["", "  HOME "]  # at the prompt: one more stray Enter, then the word
        seen: list[tuple[str, int]] = []

        def flush() -> int:
            n = len(typeahead)
            typeahead.clear()
            return n

        def console(_text: str) -> str:
            line = typeahead.pop(0) if typeahead else answers.pop(0)
            seen.append((line, driver.write_count))
            return line

        out = io.StringIO()
        gate = FirstHomeGate(prompt=console, out=out, interactive=True, flush=flush)
        runtime = self._runtime(cfg, gate)
        try:
            runtime.build()
            assert seen == [("", 0), ("  HOME ", 0)], "the typeahead must never reach the prompt"
            assert gate.asked == 2 and runtime.first_home == "operator"
            assert driver.position_known
            assert "discarded 2 keystroke" in out.getvalue() and "type the word home" in out.getvalue()
        finally:
            runtime.close()

    def test_a_console_without_flush_would_have_confirmed_on_the_queued_enter(self, real_lane):
        """The fake must be able to fail: with no flush the queued Enter is read
        first -- and still does not confirm, because a bare line is not the word."""
        import io

        from mfw.hardware.runtime import FirstHomeGate

        cfg, _server, driver = real_lane
        lines = ["", "home"]
        gate = FirstHomeGate(prompt=lambda _t: lines.pop(0), out=io.StringIO(), interactive=True,
                             flush=lambda: 0)
        runtime = self._runtime(cfg, gate)
        try:
            runtime.build()
            assert gate.asked == 2 and driver.position_known
        finally:
            runtime.close()

    def test_only_stray_lines_refuse_after_three_prompts(self, real_lane):
        import io

        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused

        cfg, _server, driver = real_lane
        answers = iter(["", "y", "ok"])
        gate = FirstHomeGate(prompt=lambda _t: next(answers), out=io.StringIO(), interactive=True)
        runtime = self._runtime(cfg, gate)
        try:
            with pytest.raises(FirstHomeRefused, match="nothing was sent"):
                runtime.build()
            assert gate.asked == 3
            assert driver.write_count == 0 and not driver.position_known
        finally:
            runtime.close()

    def test_the_default_prompt_flushes_the_console_first(self, monkeypatch):
        """With the real ``input``, the console is drained before every prompt."""
        import builtins
        import io

        from mfw.hardware import runtime as runtime_mod

        order: list[str] = []
        monkeypatch.setattr(runtime_mod, "_flush_console_input", lambda: order.append("flush") or 0)
        monkeypatch.setattr(builtins, "input", lambda _t="": order.append("input") or "home")
        gate = runtime_mod.FirstHomeGate(out=io.StringIO(), interactive=True)
        assert gate({"endpoint": "tcp://x:5560", "driver": "uno", "home_q": [0, 0, 0, 0]}) == "operator"
        assert order == ["flush", "input"]

    def test_eof_at_the_prompt_aborts_and_sends_nothing(self, real_lane):
        import io

        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused

        cfg, _server, driver = real_lane

        def eof(_text: str) -> str:
            raise EOFError

        runtime = self._runtime(cfg, FirstHomeGate(prompt=eof, out=io.StringIO(), interactive=True))
        try:
            with pytest.raises(FirstHomeRefused, match="nothing was sent"):
                runtime.build()
            assert driver.write_count == 0 and not driver.position_known
            assert runtime.first_home is None and runtime.skills is None
        finally:
            runtime.close()

    def test_ctrl_c_at_the_prompt_propagates_and_sends_nothing(self, real_lane):
        import io

        from mfw.hardware.runtime import FirstHomeGate

        cfg, _server, driver = real_lane

        def ctrl_c(_text: str) -> str:
            raise KeyboardInterrupt

        runtime = self._runtime(cfg, FirstHomeGate(prompt=ctrl_c, out=io.StringIO(), interactive=True))
        try:
            with pytest.raises(KeyboardInterrupt):
                runtime.build()
            assert driver.write_count == 0 and not driver.position_known
        finally:
            runtime.close()

    def test_no_terminal_refuses_and_names_the_flag(self, real_lane):
        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused

        cfg, _server, driver = real_lane
        gate = FirstHomeGate(prompt=_never_prompt("prompted with no terminal"), interactive=False)
        runtime = self._runtime(cfg, gate)
        try:
            with pytest.raises(FirstHomeRefused, match="--home-confirmed"):
                runtime.build()
            assert driver.write_count == 0 and not driver.position_known
        finally:
            runtime.close()

    def test_home_confirmed_flag_proceeds_without_a_prompt(self, real_lane):
        import io

        from mfw.hardware.runtime import FirstHomeGate

        cfg, _server, driver = real_lane
        out = io.StringIO()
        gate = FirstHomeGate(confirmed=True, prompt=_never_prompt("prompted despite --home-confirmed"),
                             out=out, interactive=False)
        runtime = self._runtime(cfg, gate)
        try:
            runtime.build()
            assert runtime.first_home == "flag"
            assert driver.write_count >= 1 and driver.position_known
            assert "--home-confirmed given" in out.getvalue() and "Hand-pose the arm at home" in out.getvalue()
        finally:
            runtime.close()

    def test_estopped_server_is_refused_before_prompting(self, real_lane):
        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused

        cfg, _server, driver = real_lane
        client = JetsonClient(cfg.hardware.jetson_host, cfg.hardware.jetson_port)
        try:
            client.connect()
            client.estop()  # e.g. the Ctrl-C estop sent while an earlier session was at the prompt
        finally:
            client.close()
        gate = FirstHomeGate(prompt=_never_prompt("asked a person to pose an arm whose server refuses motion"),
                             interactive=True)
        runtime = self._runtime(cfg, gate)
        try:
            with pytest.raises(FirstHomeRefused, match="estopped.*clear"):
                runtime.build()
            assert driver.write_count == 0 and not driver.position_known
        finally:
            runtime.close()

    def test_a_known_position_homes_without_asking(self, real_lane):
        """Second run against the same server: the bridge knows its pose, the
        home is an ordinary move at the velocity ceiling, nothing to confirm."""
        cfg, _server, driver = real_lane
        client = JetsonClient(cfg.hardware.jetson_host, cfg.hardware.jetson_port)
        try:
            client.connect()
            client.home()
        finally:
            client.close()
        assert driver.position_known
        runtime = self._runtime(cfg, _never_prompt("asked although the bridge knows its position"))
        try:
            runtime.build()
            assert runtime.first_home == "not needed"
        finally:
            runtime.close()

    @staticmethod
    def _leave_the_arm_limp_away_from_home(cfg, robot_server, driver):
        """A previous session homed and moved the arm, then ended; the server's
        host timeout detached the servos (what ``check_host_liveness`` does)."""
        client = JetsonClient(cfg.hardware.jetson_host, cfg.hardware.jetson_port)
        try:
            client.connect()
            client.home()
            q = [float(v) for v in client.get_state()["q"]]
            away = [q[0] + 0.2, q[1] + 0.1, q[2] - 0.1, q[3]]
            client.follow_trajectory([q, away], 0.5)
        finally:
            client.close()
        last_pulses = tuple(int(round(v)) for v in driver.pulses())
        with driver.lock:
            driver.detach("host timeout: no client request for 5.0 s")
        robot_server.host_lost = True
        assert driver.position_known and not driver.attached
        return last_pulses

    def test_limp_servos_at_a_known_position_ask_first(self, real_lane):
        """Second run on the same robot_server: the position is known, but the
        servos are detached, and ``home`` re-attaches AT the last pulsed pose at
        full speed before its slow move -- the FakeDriver records it as the
        first ``P`` frame after the prompt. A sagged arm snaps back there.

        was (before the fixer, 2026-09-28): no prompt, home sent at once.
        now: the gate runs with ``reason == "limp"`` and the limp wording.
        """
        import io

        from mfw.hardware.runtime import FIRST_HOME_LIMP, FirstHomeGate

        cfg, robot_server, driver = real_lane
        last_pulses = self._leave_the_arm_limp_away_from_home(cfg, robot_server, driver)
        detached_at = len(driver.commands)
        at_prompt: list[int] = []
        contexts: list[dict] = []

        def confirm(_text: str) -> str:
            at_prompt.append(len(driver.commands))
            return "home"

        out = io.StringIO()
        gate = FirstHomeGate(prompt=confirm, out=out, interactive=True)

        def recording_gate(context):
            contexts.append(dict(context))
            return gate(context)

        runtime = self._runtime(cfg, recording_gate)
        try:
            runtime.build()
            assert contexts[0]["reason"] == FIRST_HOME_LIMP and gate.asked == 1
            assert runtime.first_home == "operator"
            text = out.getvalue()
            assert "DETACHED (limp)" in text and "LAST POSE" in text and "not at home" in text
            assert "host timeout" in text and "E-stop" in text
            # Nothing was pulsed between the detach and the typed "home" ...
            assert not [c for c in driver.commands[detached_at:at_prompt[0]] if c[0] == "P"]
            # ... and the first frame after it re-energised AT the last pose,
            # not at home: the snap a sagged arm makes.
            first_after = next(c for c in driver.commands[at_prompt[0]:] if c[0] == "P")
            assert first_after[1] == last_pulses
            assert driver.attached
        finally:
            runtime.close()

    def test_limp_servos_with_no_terminal_refuse_with_the_limp_reason(self, real_lane):
        from mfw.hardware.runtime import FirstHomeGate, FirstHomeRefused

        cfg, robot_server, driver = real_lane
        self._leave_the_arm_limp_away_from_home(cfg, robot_server, driver)
        writes = len(driver.commands)
        gate = FirstHomeGate(prompt=_never_prompt("prompted with no terminal"), interactive=False)
        runtime = self._runtime(cfg, gate)
        try:
            with pytest.raises(FirstHomeRefused, match="detached.*last pose.*--home-confirmed"):
                runtime.build()
            assert not [c for c in driver.commands[writes:] if c[0] == "P"]
            assert not driver.attached
        finally:
            runtime.close()

    def test_fake_driver_is_unchanged_no_prompt(self, fake_lane):
        """The fake lane (ping says fake: true) homes from an unknown position
        exactly as before: no prompt."""
        cfg, _world, _server, _det, driver = fake_lane
        assert not driver.position_known
        runtime = self._runtime(cfg, _never_prompt("the fake lane must never prompt"))
        try:
            runtime.build()
            assert runtime.first_home == "not needed"
            assert driver.position_known
        finally:
            runtime.close()

    def test_default_gate_is_the_installed_module_gate(self, real_lane):
        """Assistant builds HardwareRuntime(config) with no gate argument;
        run_assistant.py installs its gate as the module default."""
        from mfw.hardware import runtime as runtime_mod

        cfg, _server, driver = real_lane
        seen: list[dict] = []

        def record(context) -> str:
            seen.append(dict(context))
            assert driver.write_count == 0
            return "flag"

        previous = runtime_mod.set_first_home_gate(record)
        try:
            runtime = runtime_mod.HardwareRuntime(cfg)
            try:
                runtime.build()
            finally:
                runtime.close()
        finally:
            assert runtime_mod.set_first_home_gate(previous) is record
        assert len(seen) == 1 and seen[0]["endpoint"].endswith(f":{cfg.hardware.jetson_port}")
        assert seen[0]["home_q"] == pytest.approx(list(cfg.robot.home_joint_positions))
        assert seen[0]["state"]["bridge_position_known"] is False

    def test_initial_module_default_refuses_without_a_terminal(self, monkeypatch):
        """Any caller that never installs a gate (another script, a notebook)
        gets the safe default: prompt at a TTY, refuse otherwise."""
        import io
        import sys

        from mfw.hardware import runtime as runtime_mod

        assert isinstance(runtime_mod._default_first_home_gate, runtime_mod.FirstHomeGate)
        assert runtime_mod._default_first_home_gate.confirmed is False
        gate = runtime_mod.FirstHomeGate()
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a TTY
        with pytest.raises(runtime_mod.FirstHomeRefused, match="--home-confirmed"):
            gate({"endpoint": "tcp://x:5560", "driver": "uno", "home_q": [0, 0, 0, 0]})
        monkeypatch.setattr(sys, "stdin", None)  # pythonw / detached
        assert gate.is_interactive() is False

    def test_stdin_from_the_null_device_is_not_a_terminal(self, monkeypatch):
        """``run_assistant.py < NUL`` on Windows: NUL is a character device, so
        ``isatty()`` says True (measured); the gate must still see no terminal
        and refuse with the flag's name rather than prompt into the void."""
        import os
        import sys

        from mfw.hardware import runtime as runtime_mod

        with open(os.devnull, encoding="utf-8") as null:
            monkeypatch.setattr(sys, "stdin", null)
            gate = runtime_mod.FirstHomeGate(prompt=_never_prompt("prompted on the null device"))
            assert gate.is_interactive() is False
            with pytest.raises(runtime_mod.FirstHomeRefused, match="--home-confirmed"):
                gate({"endpoint": "tcp://x:5560", "driver": "uno", "home_q": [0, 0, 0, 0]})

    @pytest.mark.parametrize(
        "info, state, needed",
        [
            ({"fake": False}, {"bridge_position_known": False}, True),
            ({"fake": False}, {}, True),  # an older server that cannot say: cautious
            ({"fake": False}, {"bridge_position_known": True}, False),
            ({"fake": False}, {"bridge_position_known": True, "attached": True}, False),
            # was False (no prompt) until the fixer, 2026-09-28: limp servos at a known pose
            ({"fake": False}, {"bridge_position_known": True, "attached": False}, True),
            ({"fake": False}, {"bridge_position_known": True, "attached": False, "host_lost": True}, True),
            ({"fake": True}, {"bridge_position_known": True, "attached": False}, False),
            ({"fake": True}, {"bridge_position_known": False}, False),
            (None, None, True),
        ],
    )
    def test_first_home_needs_operator_table(self, info, state, needed):
        from mfw.hardware.runtime import first_home_needs_operator

        assert first_home_needs_operator(info, state) is needed

    def test_first_home_reason_names_the_case(self):
        from mfw.hardware.runtime import FIRST_HOME_LIMP, FIRST_HOME_UNKNOWN, first_home_reason

        assert first_home_reason({"fake": False}, {"bridge_position_known": False}) == FIRST_HOME_UNKNOWN
        assert first_home_reason({"fake": False}, {"bridge_position_known": True, "attached": False}) == (
            FIRST_HOME_LIMP)
        assert first_home_reason({"fake": False}, {"bridge_position_known": True, "attached": True}) is None

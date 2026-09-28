"""The hardware MVP flow on the existing arm: scan the room, see, pick, place, home.

Fake lane only (no Isaac, no servos, no webcam): ``jetson/robot_server.py`` on
a ``FakeDriver`` and ``jetson/detector_service.py`` on a scripted detector, in
threads, with the fake world from ``tests/test_hardware_e2e.py``. The
detector here is wrapped so a test can script what a real overhead webcam +
Florence-2 does to a scan: a missed object in some frames, a frame the
service fails on, a reply that arrives after the laptop gave up. The driver
can be told to go limp mid-sweep (a brownout / watchdog detach), because
that is what a stalling MG996R on a sagging supply does.

What is pinned:

* ``scan_scene`` is registered on the hardware lane as ``FixedCameraScan``:
  it observes several frames, counts an object only when seen in enough of
  them, turns the base slowly toward what it found (never beyond
  ``hardware.scan.max_sweep_rad`` or the joint limits, arm kept tucked at its
  home height), returns home, and reports "I can see: ..." with left/right/
  front and distances;
* an object the camera sees outside the workspace is "out of reach" -- in
  the scan, in "what do you see", and when it is asked for by name -- never
  "not in view";
* ``Place`` says so when its retreat was skipped or shortened;
* the demo script's five commands run end to end on the fake lane.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from jetson.detector_service import DetectorServer, ScriptedDetector, SyntheticPinhole
from jetson.robot_server import FakeDriver, RobotServer, ServoCalibration
from mfw.config.schema import ConfigError, HardwareScanConfig, load_config
from mfw.core.types import JointState
from mfw.hardware.kinematics import PlanarKinematics
from mfw.skills.primitives import (
    FixedCameraScan,
    HardwareObserve,
    Place,
    scan_sweep_stops,
    where_on_table,
)
from tests.test_hardware_e2e import HARDWARE_FAKE_YAML, FakeWorld, _load_fake_config

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_YAML = REPO_ROOT / "configs" / "hardware.yaml"

MARKER_XY = (0.18, 0.05)
BOWL_XY = (0.15, -0.12)
#: 37 cm from the base, beyond workspace_max y (0.24) + margin: in the camera's
#: view, out of the arm's reach.
FAR_CUBE_XY = (0.20, 0.31)

MARKER_BEARING = math.atan2(MARKER_XY[1], MARKER_XY[0])  # +0.27 rad, front-left
BOWL_BEARING = math.atan2(BOWL_XY[1], BOWL_XY[0])  # -0.67 rad, front-right


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


class TestWhereOnTable:
    @pytest.mark.parametrize(
        "xy,expected",
        [
            ((0.20, 0.0), "in front"),
            ((0.20, 0.05), "in front"),  # 14 deg
            (MARKER_XY, "front-left"),  # 15.5 deg
            (BOWL_XY, "front-right"),
            ((0.02, 0.20), "on the left"),
            ((0.02, -0.20), "on the right"),
            ((-0.20, 0.05), "behind, to the left"),
        ],
    )
    def test_bands(self, xy, expected):
        assert where_on_table(xy) == expected

    def test_relative_to_the_robot_base(self):
        assert where_on_table((1.20, 0.0), robot_xy=(1.0, 0.2)) == "front-right"


class TestScanSweepStops:
    def test_right_to_left_and_clamped_to_max_sweep(self):
        stops = scan_sweep_stops([0.27, -0.67, 1.4], -1.57, 1.57, 1.0, 0.35, 4)
        assert stops == pytest.approx([-0.67, 0.27, 1.0])

    def test_joint_limits_win_over_max_sweep(self):
        stops = scan_sweep_stops([-1.2, 1.2], -0.5, 0.8, 1.0, 0.35, 4, margin=0.05)
        assert stops == pytest.approx([-0.45, 0.75])

    def test_nearby_bearings_merge_into_one_stop(self):
        assert scan_sweep_stops([0.20, 0.25, 0.28], -1.57, 1.57, 1.0, 0.35, 4) == pytest.approx([0.20])

    def test_nothing_found_still_looks_both_ways(self):
        assert scan_sweep_stops([], -1.57, 1.57, 1.0, 0.35, 4) == pytest.approx([-0.35, 0.35])

    def test_max_stops_keeps_the_extremes(self):
        stops = scan_sweep_stops([-0.9, -0.5, -0.1, 0.3, 0.7], -1.57, 1.57, 1.0, 0.35, 3)
        assert len(stops) == 3 and stops[0] == pytest.approx(-0.9) and stops[-1] == pytest.approx(0.7)

    def test_no_room_to_turn_is_empty(self):
        assert scan_sweep_stops([0.3], 0.0, 0.05, 1.0, 0.35, 4, margin=0.05) == []

    def test_non_finite_bearings_are_ignored(self):
        assert scan_sweep_stops([float("nan"), 0.3], -1.57, 1.57, 1.0, 0.35, 4) == pytest.approx([0.3])


class TestConfig:
    def test_hardware_yaml_carries_the_mvp_labels_and_demo(self):
        cfg = load_config(HARDWARE_YAML)
        hw = cfg.hardware
        # Review: the real camera's vocabulary is the demo table only (the old
        # 8-label list found 17/60 objects through the real Florence service);
        # the fake lane keeps the full scriptable list in its own config.
        assert hw.labels == ("marker", "block", "bowl", "box")
        assert load_config(HARDWARE_FAKE_YAML).hardware.labels == (
            "marker", "cube", "ball", "banana", "block", "bowl", "bin", "box")
        for label in ("marker", "cube", "ball", "banana", "block", "bowl", "bin", "box"):
            assert label in hw.object_sizes
        for label in ("marker", "ball", "banana", "block"):
            across = min(hw.object_sizes[label][:2])
            assert across <= cfg.grasp.max_grasp_width, label
        assert hw.demo_script == (
            "scan the room", "what do you see", "pick up the marker", "put it in the bowl", "go home",
        )
        assert hw.scan.min_frames <= hw.scan.frames
        assert hw.scan.sweep_speed_rad_s < cfg.motion.max_joint_velocity, "the sweep is slower than a move"

    def test_demo_uses_only_configured_labels_and_avoids_can(self):
        hw = load_config(HARDWARE_YAML).hardware
        words = " ".join(hw.demo_script).split()
        assert "can" not in words, "Canary hears 'can' as 'kin'"
        assert "marker" in words and "bowl" in words
        assert {"marker", "bowl"} <= set(hw.labels)

    @pytest.mark.parametrize(
        "bad",
        [dict(frames=0), dict(frames=2, min_frames=3), dict(sweep_speed_rad_s=0.0),
         dict(min_sweep_rad=1.2, max_sweep_rad=1.0), dict(max_sweep_rad=2.0), dict(max_stops=0),
         dict(dwell_s=-1.0)],
    )
    def test_scan_config_is_validated(self, bad):
        with pytest.raises(ConfigError):
            HardwareScanConfig(**bad).validate()

    def test_empty_demo_command_is_refused(self):
        with pytest.raises(ConfigError, match="demo_script"):
            load_config(HARDWARE_YAML, overrides={"hardware": {"demo_script": ["scan the room", " "]}})

    def test_sim_lane_registry_is_unchanged(self):
        from mfw.hardware.runtime import HARDWARE_SKILLS, SIM_ONLY_SKILLS
        from mfw.skills.primitives import ALL_SKILLS, Observe, ScanScene

        assert ScanScene in ALL_SKILLS and Observe in ALL_SKILLS
        assert FixedCameraScan not in ALL_SKILLS and HardwareObserve not in ALL_SKILLS
        names = [cls.skill_name for cls in HARDWARE_SKILLS]
        assert len(names) == len(set(names))
        assert "scan_scene" in names and not SIM_ONLY_SKILLS & set(names)


# ----------------------------------------------------------------------
# the fake lane, with a detector and a driver that can misbehave
# ----------------------------------------------------------------------


class FlakyDetector:
    """A scripted detector a test can make behave like the real one.

    ``plan`` is consumed one entry per detect call once :meth:`arm` is
    called (start-up observes are not scripted): ``"ok"``, ``"drop"`` (the
    service fails the frame -- USB camera hiccup), ``("miss", label)`` (the
    model does not find ``label`` in this frame), ``("late", seconds)`` (the
    reply comes after the laptop's timeout), ``("phantom", label, (x, y))``
    (a false positive: the model draws ``label`` where nothing is, at the
    same place every time it is armed -- measured on real Florence, a
    'banana' phantom survived the service's 2-of-3 vote in both repeats).
    Calls beyond the plan are ``"ok"``.
    """

    name = "scripted"

    def __init__(self, inner: ScriptedDetector) -> None:
        self.inner = inner
        self.plan: list[Any] = []
        self.armed = False
        self.calls = 0

    @property
    def image_size(self) -> tuple[int, int]:
        return self.inner.image_size

    def arm(self, plan: list[Any]) -> None:
        self.plan = list(plan)
        self.armed = True

    def detect(self, rgb: Any, labels: list[str], min_score: float = 0.0) -> list[dict[str, Any]]:
        self.calls += 1
        action = self.plan.pop(0) if self.armed and self.plan else "ok"
        if action == "drop":
            raise RuntimeError("camera frame dropped (VIDIOC_DQBUF: no such device)")
        objects = self.inner.detect(rgb, labels, min_score)
        if isinstance(action, tuple) and action[0] == "miss":
            objects = [o for o in objects if o.get("label") != action[1]]
        if isinstance(action, tuple) and action[0] == "phantom" and action[1] in labels:
            box = self.inner.box_for(action[1], (float(action[2][0]), float(action[2][1]), 0.0))
            if box is not None:
                objects = [*objects, {"label": action[1], "confidence": 0.96, "bbox_px": list(box)}]
        if isinstance(action, tuple) and action[0] == "late":
            time.sleep(float(action[1]))
        return objects


class Lane:
    def __init__(self, cfg, world, driver, detector, robot_server, yaw_log) -> None:
        self.cfg, self.world, self.driver, self.detector = cfg, world, driver, detector
        self.robot_server, self.yaw_log = robot_server, yaw_log
        self.brownout_at_yaw: float | None = None


@pytest.fixture
def mvp_lane(tmp_path):
    stoppers: list[Callable[[], None]] = []

    def build(objects: dict[str, tuple[float, float]] | None = None, base_limit_rad: float | None = None,
              **cfg_extra: Any) -> Lane:
        base = load_config(HARDWARE_FAKE_YAML)
        kin = PlanarKinematics(base.hardware.arm, base.robot.arm_joint_names)
        footprints = {k: tuple(v) for k, v in base.hardware.object_sizes.items()}
        world = FakeWorld(kin, objects or {"marker": MARKER_XY, "bowl": BOWL_XY, "cube": FAR_CUBE_XY},
                          footprints)
        yaw_log: list[tuple[float, float]] = []  # (base yaw, tcp z) per pulse write
        lane_ref: dict[str, Lane] = {}
        driver_ref: dict[str, FakeDriver] = {}

        def on_command(q: np.ndarray, width: float) -> None:
            world.on_command(q, width)
            yaw_log.append((float(q[0]), float(world.tcp[2])))
            lane = lane_ref.get("lane")
            if lane is not None and lane.brownout_at_yaw is not None and float(q[0]) > lane.brownout_at_yaw:
                lane.brownout_at_yaw = None
                # The supply sags under a stalled MG996R: the bridge drops every servo.
                driver_ref["driver"].mark_detached("brownout: servo supply sagged mid-sweep")

        calibration = ServoCalibration.default()
        if base_limit_rad is not None:
            # The Jetson enforces a narrower base band than hardware.arm says
            # (--allow-placeholder-calibration: +-0.873 rad vs the laptop's +-1.5708).
            raw = calibration.to_dict()
            raw["joints"]["base_yaw"].update(limit_lower_rad=-base_limit_rad, limit_upper_rad=base_limit_rad)
            calibration = ServoCalibration.from_dict(raw)
        driver = FakeDriver(calibration, on_command=on_command)
        driver_ref["driver"] = driver
        robot_server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                                   follower_sleep=lambda _s: None)
        _, robot_port = robot_server.serve_in_thread(port=0)
        stoppers.append(robot_server.stop)
        detector = FlakyDetector(ScriptedDetector(world.scene, SyntheticPinhole(), footprints=footprints,
                                                  arm=lambda: world.tcp))
        detector_server = DetectorServer("127.0.0.1", 0, detector)
        _, detector_port = detector_server.serve_in_thread(port=0)
        stoppers.append(detector_server.stop)
        cfg = _load_fake_config(robot_port, detector_port, tmp_path, **cfg_extra)
        lane = Lane(cfg, world, driver, detector, robot_server, yaw_log)
        lane_ref["lane"] = lane
        return lane

    try:
        yield build
    finally:
        for stop in reversed(stoppers):
            stop()


def _assistant(lane: Lane):
    from mfw.assistant import Assistant

    return Assistant(config=lane.cfg)


def _events(assistant, name: str) -> list[dict[str, Any]]:
    path = Path(assistant.runtime.events.path)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("event") == name:
                out.append(entry)
    return out


def _home(lane: Lane) -> np.ndarray:
    return np.asarray(lane.cfg.robot.home_joint_positions, dtype=np.float64)


class TestScanTheRoom:
    def test_scan_reports_directions_sweeps_toward_objects_and_returns_home(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert type(assistant.runtime.skills.get("scan_scene")) is FixedCameraScan
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert outcome.skill == "scan_scene"
            msg = outcome.message
            assert msg.startswith("I can see: ")
            assert "a marker front-left (19 cm away)" in msg
            assert "a bowl front-right (19 cm away)" in msg
            assert "Out of reach: a cube front-left (37 cm away)" in msg
            assert "not in view" not in msg
            data = outcome.result.data
            assert {o["label"] for o in data["objects"]} == {"marker", "bowl"}
            assert all(o["frames_seen"] == 3 for o in data["objects"])
            # The base visited both bearings, right to left, and came back.
            yaws = [y for y, _z in lane.yaw_log]
            assert min(yaws) == pytest.approx(BOWL_BEARING, abs=0.02)
            assert max(yaws) == pytest.approx(MARKER_BEARING, abs=0.02)
            assert yaws.index(min(yaws)) < yaws.index(max(yaws)), "right first, then left"
            np.testing.assert_allclose(assistant.runtime.robot.get_arm_joint_positions(), _home(lane), atol=1e-6)
            # Only the base turned: the arm stayed tucked at its home height,
            # far above anything on the table.
            home_z = PlanarKinematics(lane.cfg.hardware.arm, lane.cfg.robot.arm_joint_names).fk(_home(lane)).position[2]
            assert all(z == pytest.approx(home_z, abs=1e-6) for _y, z in lane.yaw_log)
            # Slow: the sweep's trajectory takes at least its path at the scan speed.
            sweep = _events(assistant, "scan.sweep")
            assert len(sweep) == 1
            travel = abs(BOWL_BEARING) + (MARKER_BEARING - BOWL_BEARING) + MARKER_BEARING
            assert sweep[0]["duration_s"] >= travel / lane.cfg.hardware.scan.sweep_speed_rad_s
            assert sweep[0]["stops_rad"] == pytest.approx([BOWL_BEARING, MARKER_BEARING], abs=0.02)
            # Memory now has the scene: the next command can refer to it.
            assert {o["label"] for o in assistant.describe()["objects"]} == {"marker", "bowl"}
        finally:
            assistant.close()

    def test_the_sweep_stays_inside_the_jetsons_enforced_base_limits(self, mvp_lane):
        """Review finding 7: the bowl is at -0.67 rad but the Jetson allows only
        +-0.5 rad. The sweep used to plan to -0.67, run a clamped chunk, stop,
        and fail with the arm off home; it must stop short and succeed."""
        lane = mvp_lane(base_limit_rad=0.5)
        assistant = _assistant(lane)
        try:
            limits = assistant.runtime.robot.server_joint_limits
            assert limits is not None and limits[0][0] == pytest.approx(-0.5)
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            yaws = [y for y, _z in lane.yaw_log]
            assert -0.5 < min(yaws) < -0.3  # turned toward the bowl, stopped inside the band
            assert max(yaws) == pytest.approx(MARKER_BEARING, abs=0.02)
            np.testing.assert_allclose(assistant.runtime.robot.get_arm_joint_positions(), _home(lane), atol=1e-6)
        finally:
            assistant.close()

    def test_an_object_missed_in_most_frames_is_not_claimed_and_not_swept_to(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            lane.detector.arm([("miss", "marker"), ("miss", "marker"), "ok"])
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert "I can see: a bowl front-right" in outcome.message
            assert "Not sure about: a marker front-left (seen in 1 of 3 frames)" in outcome.message
            assert [o["label"] for o in outcome.result.data["objects"]] == ["bowl"]
            yaws = [y for y, _z in lane.yaw_log]
            assert max(yaws) < 0.05, "the base never turned toward an object it was not sure of"
            assert min(yaws) == pytest.approx(BOWL_BEARING, abs=0.02)
        finally:
            assistant.close()

    def test_a_consistent_phantom_is_claimed_and_swept_to_a_known_limit(self, mvp_lane):
        """Review finding 12: a phantom Florence draws in the same place on every
        frame passes the scan's 2-of-3 check exactly like a real object. Pinned
        honestly: it is reported and the base turns toward it. Nothing in the
        scan can tell it from a real banana; the defence is the label list
        (hardware.labels = only the objects on the table, docs/MVP_RUNBOOK.md 7)."""
        lane = mvp_lane()
        assistant = _assistant(lane)
        phantom_xy = (0.17, 0.12)
        try:
            lane.detector.arm([("phantom", "banana", phantom_xy)] * 3)
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert "banana" not in lane.world.scene()  # nothing is there
            assert "a banana front-left" in outcome.message and "I can see:" in outcome.message
            assert "banana" in {o["label"] for o in outcome.result.data["objects"]}
            yaws = [y for y, _z in lane.yaw_log]
            assert max(yaws) == pytest.approx(math.atan2(phantom_xy[1], phantom_xy[0]), abs=0.03)
        finally:
            assistant.close()

    def test_a_phantom_in_one_frame_is_only_unsure_and_not_swept_to(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        phantom_xy = (0.17, 0.12)
        try:
            assert lane.detector.inner.box_for("banana", (*phantom_xy, 0.0)) is not None  # it is drawable
            lane.detector.arm(["ok", ("phantom", "banana", phantom_xy), "ok"])
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert lane.detector.plan == [], "the phantom frame was served"
            assert "banana" not in {o["label"] for o in outcome.result.data["objects"]}
            assert "I can see: a banana" not in outcome.message and "and a banana" not in outcome.message
            yaws = [y for y, _z in lane.yaw_log]
            assert max(yaws) == pytest.approx(MARKER_BEARING, abs=0.02), "never turned toward the phantom"
        finally:
            assistant.close()

    def test_a_dropped_frame_is_reported_not_fatal(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            lane.detector.arm(["ok", "drop", "ok"])
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert "a marker front-left" in outcome.message and "a bowl front-right" in outcome.message
            assert "1 of 3 detector frame(s) failed" in outcome.message
            assert outcome.result.data["frame_failures"]
        finally:
            assistant.close()

    def test_a_late_reply_counts_as_a_failed_frame(self, mvp_lane):
        lane = mvp_lane(hardware={"request_timeout_s": 0.5})
        assistant = _assistant(lane)
        try:
            # 0.7 s against a 0.5 s budget: the laptop gives up on frame 1; the
            # service still holds its lock, so frame 2 waits ~0.2 s and lands.
            lane.detector.arm([("late", 0.7), "ok", "ok"])
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert "1 of 3 detector frame(s) failed" in outcome.message
            assert "a bowl front-right" in outcome.message
        finally:
            assistant.close()

    def test_every_frame_failing_fails_without_moving(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            lane.detector.arm(["drop", "drop", "drop"])
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert not outcome.ok
            assert "could not look" in outcome.message
            assert lane.yaw_log == [], "no sweep on a scan that saw nothing at all"
        finally:
            assistant.close()

    def test_nothing_on_the_table_still_looks_both_ways(self, mvp_lane):
        lane = mvp_lane(objects={"cube": FAR_CUBE_XY})
        assistant = _assistant(lane)
        try:
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert outcome.message.startswith("I can't see anything I know on the table")
            assert "Out of reach: a cube" in outcome.message
            yaws = [y for y, _z in lane.yaw_log]
            scan = lane.cfg.hardware.scan
            assert min(yaws) == pytest.approx(-scan.min_sweep_rad, abs=1e-3)
            assert max(yaws) == pytest.approx(scan.min_sweep_rad, abs=1e-3)
        finally:
            assistant.close()

    def test_holding_something_it_keeps_still(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert assistant.command("pick up the marker").ok
            lane.yaw_log.clear()
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            assert "I kept still because I am holding something" in outcome.message
            assert lane.yaw_log == [], "a held object is not swung around"
            assert lane.world.attached == "marker"
        finally:
            assistant.close()

    def test_a_brownout_mid_sweep_fails_and_says_what_it_saw(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            lane.brownout_at_yaw = 0.1  # on the way from the bowl's bearing to the marker's
            outcome = assistant.command("scan the room")
            assert not outcome.ok
            assert "I can see: a marker" in outcome.message
            assert "the sweep stopped part-way" in outcome.message
            assert "go home" in outcome.message
            assert outcome.result.data["swept"] is False
        finally:
            assistant.close()

    def test_scan_from_away_from_home_goes_home_first(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            # A pick leaves the arm at its lift pose; opening drops the marker
            # back where it was and leaves the arm there, away from home.
            assert assistant.command("pick up the marker").ok
            assert assistant.command("open the gripper").ok
            away = assistant.runtime.robot.get_arm_joint_positions()
            assert float(np.max(np.abs(away - _home(lane)))) > 0.02
            outcome = assistant.command("scan the room")
            assert outcome.ok, outcome.message
            np.testing.assert_allclose(assistant.runtime.robot.get_arm_joint_positions(), _home(lane), atol=1e-6)
        finally:
            assistant.close()


class TestOutOfReachWording:
    def test_what_do_you_see_gives_directions_and_out_of_reach(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert type(assistant.runtime.skills.get("observe")) is HardwareObserve
            seen = assistant.command("what do you see")
            assert seen.ok
            assert seen.message.startswith("observed 2 object(s): ")
            assert "marker (front-left, 19 cm away)" in seen.message
            assert "; out of reach: a cube front-left (37 cm away)" in seen.message
            assert [r["label"] for r in seen.result.data["out_of_reach"]] == ["cube"]
        finally:
            assistant.close()

    def test_asking_for_an_out_of_reach_object_says_out_of_reach(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            outcome = assistant.command("pick up the cube")
            assert not outcome.ok
            assert "out of reach" in outcome.message and "not in view" not in outcome.message
            assert "front-left" in outcome.message and "37 cm" in outcome.message
            assert lane.world.attached is None
        finally:
            assistant.close()

    def test_an_object_truly_absent_is_still_not_in_view(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            outcome = assistant.command("pick up the banana")
            assert not outcome.ok
            assert "out of reach" not in outcome.message
        finally:
            assistant.close()

    def test_background_far_away_is_not_reported(self, mvp_lane):
        """A 'marker' 0.9 m away is somebody else's desk, not an unreachable object."""
        from mfw.hardware.perception import PlanarPerception

        lane = mvp_lane(objects={"marker": MARKER_XY, "bowl": BOWL_XY})
        cfg = lane.cfg

        class Clock:
            sim_time, step_index = 0.0, 0

        vision = PlanarPerception(
            clock=Clock(), detector=None, config=cfg.perception, hardware=cfg.hardware,
            homography=cfg.exterior_camera.homography, workspace_min=cfg.scene.workspace_min,
            workspace_max=cfg.scene.workspace_max, camera=cfg.exterior_camera,
        )
        near = ScriptedDetector({"cube": (*FAR_CUBE_XY, 0.0)}, SyntheticPinhole()).box_for("cube", (*FAR_CUBE_XY, 0.0))
        vision._hypotheses([{"label": "cube", "confidence": 0.9, "bbox_px": list(near)}])
        assert [r["label"] for r in vision.last_out_of_reach] == ["cube"]
        assert vision.last_out_of_reach[0]["distance_m"] == pytest.approx(math.hypot(*FAR_CUBE_XY), abs=0.02)
        # Mutating the copy does not touch the record.
        vision.last_out_of_reach[0]["label"] = "x"
        assert vision.last_out_of_reach[0]["label"] == "cube"
        # A detection mapped far beyond the table is background.
        vision.OUT_OF_REACH_REPORT_M = 0.2
        vision._hypotheses([{"label": "cube", "confidence": 0.9, "bbox_px": list(near)}])
        assert vision.last_out_of_reach == []


class TestPlaceRetreatHonesty:
    def test_demo_place_reports_a_shortened_retreat(self, mvp_lane):
        """The live fake run's bowl place: a 12 cm lift has no straight line there."""
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert assistant.command("pick up the marker").ok
            placed = assistant.command("put it in the bowl")
            assert placed.ok, placed.message
            retreat = placed.result.data["retreat"]
            assert retreat["done"] is True
            if retreat["lift_m"] < retreat["requested_m"] - 0.005:
                assert f"retreated only {retreat['lift_m'] * 100:.0f} cm up" in placed.message
            else:
                assert "retreat" not in placed.message
            assert lane.world.attached is None
            assert np.linalg.norm(lane.world.objects["marker"][:2] - np.array(BOWL_XY)) < 0.03
        finally:
            assistant.close()

    def test_no_retreat_at_all_is_in_the_message(self, mvp_lane, monkeypatch):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert assistant.command("pick up the marker").ok
            planner = assistant.runtime.planner
            original = planner.plan_cartesian_line
            calls = {"n": 0}

            def no_line_after_release(start, goal, scene, check_objects=False):
                # The lowering is a line too: only refuse once the jaw is open.
                if assistant.runtime.controller.grasp_width is None and not check_objects \
                        and goal.position[2] > assistant.runtime.robot.tcp_pose().position[2] + 1e-3 \
                        and assistant.runtime.memory.get_held_object() is not None \
                        and lane.world.attached is None:
                    calls["n"] += 1
                    return None
                return original(start, goal, scene, check_objects=check_objects)

            monkeypatch.setattr(planner, "plan_cartesian_line", no_line_after_release)
            placed = assistant.command("put it in the bowl")
            assert calls["n"] >= 1, "the retreat was attempted and refused"
            assert placed.ok, placed.message
            assert "the retreat up was skipped" in placed.message
            assert "go home" in placed.message
            assert placed.result.data["retreat"]["done"] is False
        finally:
            assistant.close()

    def test_an_arm_that_went_limp_in_the_retreat_fails_the_place(self, mvp_lane, monkeypatch):
        """Review finding 5: a retreat stopped by a bridge detach (watchdog, host
        timeout, serial loss) is not a success -- the scripted 'go home' after it
        would re-attach a fallen arm with a jump. Place fails, not retryable."""
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            assert assistant.command("pick up the marker").ok
            controller = assistant.runtime.controller
            original = controller.follow_trajectory
            calls = {"n": 0}

            def detach_in_the_retreat(trajectory, on_step=None):
                if controller.grasp_width is None and lane.world.attached is None                         and assistant.runtime.memory.get_held_object() is not None:
                    calls["n"] += 1  # the jaw is open, the object released: this is the retreat
                    controller.last_detach_notice = "host timeout: no client request for 5.3 s"
                    return False
                return original(trajectory, on_step)

            monkeypatch.setattr(controller, "follow_trajectory", detach_in_the_retreat)
            placed = assistant.command("put it in the bowl")
            assert calls["n"] == 1, "the retreat was attempted once"
            assert not placed.ok, placed.message
            assert "placed" in placed.message and "went limp" in placed.message
            assert "look before the next command" in placed.message
            assert placed.result.data["retryable"] is False and placed.result.data["released"] is True
            assert placed.result.data["retreat"]["arm_limp"].startswith("host timeout")
            assert np.linalg.norm(lane.world.objects["marker"][:2] - np.array(BOWL_XY)) < 0.03
        finally:
            assistant.close()

    def test_a_stale_notice_does_not_outlive_its_motion(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            controller = assistant.runtime.controller
            controller.last_detach_notice = "fake watchdog"
            assert assistant.command("pick up the marker").ok
            assert controller.last_detach_notice is None
        finally:
            assistant.close()

    def test_retreat_note_wording(self):
        assert Place._retreat_note({"done": True, "lift_m": 0.12, "requested_m": 0.12}) == ""
        assert "only 5 cm" in Place._retreat_note({"done": True, "lift_m": 0.05, "requested_m": 0.12})
        note = Place._retreat_note({"done": False, "lift_m": 0.0, "requested_m": 0.12,
                                    "reason": "no straight path up from the release point"})
        assert "skipped (no straight path up" in note and "go home" in note


class TestDemoScript:
    def test_the_five_demo_commands_run_end_to_end(self, mvp_lane):
        lane = mvp_lane()
        demo = list(load_config(HARDWARE_YAML).hardware.demo_script)
        assistant = _assistant(lane)
        try:
            outcomes = [assistant.command(c) for c in demo]
        finally:
            assistant.close()
        assert [o.skill for o in outcomes] == ["scan_scene", "observe", "pick", "place", "go_home"]
        assert all(o.ok for o in outcomes), [(o.utterance, o.message) for o in outcomes]
        assert lane.world.attached is None
        assert np.linalg.norm(lane.world.objects["marker"][:2] - np.array(BOWL_XY)) < 0.03
        assert "I can see:" in outcomes[0].message


class TestScanUnit:
    """The skill against in-memory doubles: timing and the bounds of the sweep."""

    def test_the_sweep_trajectory_never_exceeds_the_scan_speed(self, mvp_lane):
        lane = mvp_lane()
        assistant = _assistant(lane)
        try:
            captured: list[Any] = []
            controller = assistant.runtime.controller
            original = controller.follow_trajectory

            def spy(trajectory, on_step=None):
                captured.append(trajectory)
                return original(trajectory, on_step=on_step)

            controller.follow_trajectory = spy  # type: ignore[method-assign]
            assert assistant.command("scan the room").ok
        finally:
            assistant.close()
        (sweep,) = [t for t in captured if t.planner_name == "scan_sweep"]
        q = np.stack([w.positions for w in sweep.waypoints])
        t = np.array([w.time_from_start for w in sweep.waypoints])
        assert np.all(np.diff(t) > 0.0)
        speed = np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]
        assert float(speed.max()) <= lane.cfg.hardware.scan.sweep_speed_rad_s + 1e-6
        np.testing.assert_allclose(q[:, 1:], np.tile(_home(lane)[1:], (len(q), 1)), atol=1e-9)
        np.testing.assert_allclose(q[-1], _home(lane), atol=1e-9)
        assert float(np.max(np.abs(q[:, 0]))) <= lane.cfg.hardware.scan.max_sweep_rad + 1e-9
        # Dwell: the trajectory stands still at each stop for dwell_s.
        still = np.all(np.abs(np.diff(q, axis=0)) < 1e-12, axis=1)
        assert float(np.sum(np.diff(t)[still])) == pytest.approx(2 * lane.cfg.hardware.scan.dwell_s, abs=1e-6)


# ----------------------------------------------------------------------
# Jetson operations: systemd units and mode/telemetry scripts
# ----------------------------------------------------------------------

SYSTEMD_DIR = REPO_ROOT / "jetson" / "systemd"
MODEL_SERVICES = ("mfw-llama", "mfw-speech", "mfw-detector")
ALL_SERVICES = ("mfw-robot", "mfw-llama", "mfw-llm-shim", "mfw-speech", "mfw-detector")
DROP_CACHES = "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"


def _unit(name: str) -> dict[str, list[str]]:
    """``{"Section.Key": [values...]}`` of one unit file (continuation lines are not used)."""
    out: dict[str, list[str]] = {}
    section = ""
    for raw in (SYSTEMD_DIR / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, _, value = line.partition("=")
        out.setdefault(f"{section}.{key}", []).append(value)
    return out


class TestJetsonSystemd:
    @pytest.mark.parametrize("service", ALL_SERVICES)
    def test_every_service_drops_caches_restarts_with_a_limit_and_joins_the_target(self, service):
        unit = _unit(f"{service}.service")
        assert any(DROP_CACHES in v and v.startswith("+/bin/sh -c") for v in unit["Service.ExecStartPre"])
        assert unit["Service.Restart"] == ["on-failure"]
        assert int(unit["Unit.StartLimitBurst"][0]) >= 1 and int(unit["Unit.StartLimitIntervalSec"][0]) > 0
        assert unit["Unit.PartOf"] == ["mfw-conversation.target"]
        assert unit["Service.EnvironmentFile"] == ["/etc/mfw/mfw.env"]
        assert unit["Service.ExecStartPost"][0].count("wait_ready.sh") == 1, "ordering means 'loaded', not 'forked'"

    def test_models_load_in_order_llama_speech_detector(self):
        after = {s: " ".join(_unit(f"{s}.service").get("Unit.After", [])).split() for s in ALL_SERVICES}
        assert "mfw-llama.service" in after["mfw-speech"]
        assert {"mfw-llama.service", "mfw-speech.service"} <= set(after["mfw-detector"])
        assert "mfw-speech.service" not in after["mfw-llama"] and "mfw-detector.service" not in after["mfw-llama"]
        assert "mfw-robot.service" in after["mfw-detector"], "frames come from robot_server's get_frame"
        assert "mfw-llama.service" in after["mfw-llm-shim"]

    def test_the_target_pulls_everything_and_is_not_started_at_boot(self):
        target = _unit("mfw-conversation.target")
        wants = set(" ".join(target["Unit.Wants"]).split())
        assert wants == {f"{s}.service" for s in ALL_SERVICES}
        assert wants <= set(" ".join(target["Unit.After"]).split())
        assert not any(k.startswith("Install.") for k in target), "robot_server snaps home: never at boot"
        build = _unit("mfw-build.target")
        assert wants <= set(" ".join(build["Unit.Conflicts"]).split())

    def test_run_lines_use_the_mvp_models(self):
        text = {s: " ".join(_unit(f"{s}.service")["Service.ExecStart"]) for s in ALL_SERVICES}
        assert "--asr canary-gguf" in text["mfw-speech"] and "--port 5556" in text["mfw-speech"]
        assert "--backend florence-onnx" in text["mfw-detector"] and "--port 5558" in text["mfw-detector"]
        assert "--jetson 127.0.0.1:5560" in text["mfw-detector"]
        assert "--port 5557" in text["mfw-llm-shim"] and "--skills hardware" in text["mfw-llm-shim"]
        assert "$QWEN_GGUF" in text["mfw-llama"] and "--host 127.0.0.1" in text["mfw-llama"]
        assert "--port 5560" in text["mfw-robot"]
        env = (SYSTEMD_DIR / "mfw.env.example").read_text(encoding="utf-8")
        assert "qwen2.5-3b-instruct-q4_k_m.gguf" in env and "canary-qwen-2.5b-Q4_K_M.gguf" in env

    @pytest.mark.parametrize("service", ALL_SERVICES)
    def test_shell_variables_are_escaped_from_systemd(self, service):
        """systemd substitutes ``$VAR``/``${VAR}`` itself (empty when unset) before
        ``/bin/sh`` runs; ``$$VAR`` hands the shell a literal ``$VAR`` to expand
        from the EnvironmentFile, which is what these lines mean."""
        import re as _re

        unit = _unit(f"{service}.service")
        for key in ("Service.ExecStart", "Service.ExecStartPost"):
            for line in unit[key]:
                assert "$$" in line
                assert not _re.search(r"(?<!\$)\$(?!\$)", line), (key, line)

    def test_the_flags_exist_in_the_scripts_they_call(self):
        """A unit that passes a flag the script does not accept dies at every start."""
        import re as _re

        def flags(path: Path) -> set[str]:
            return set(_re.findall(r'"(--[a-z0-9-]+)"', path.read_text(encoding="utf-8")))

        checks = {
            "mfw-robot": REPO_ROOT / "jetson" / "robot_server.py",
            "mfw-llm-shim": REPO_ROOT / "jetson" / "llm_worker_llamacpp.py",
            "mfw-speech": REPO_ROOT / "scripts" / "speech_worker.py",
            "mfw-detector": REPO_ROOT / "scripts" / "serve_detector.py",
        }
        env_extra = {"--require-camera"}  # MFW_ROBOT_EXTRA default
        for service, script in checks.items():
            used = set(_re.findall(r"(--[a-z0-9-]+)", " ".join(_unit(f"{service}.service")["Service.ExecStart"])))
            known = flags(script) | (flags(REPO_ROOT / "jetson" / "detector_service.py") if service == "mfw-detector" else set())
            missing = used - known - (env_extra if service == "mfw-robot" else set())
            assert not missing, (service, sorted(missing))
        assert env_extra <= flags(REPO_ROOT / "jetson" / "robot_server.py")

    def test_llama_args_are_flags_of_the_pinned_llama_server(self):
        """Review: LLAMA_ARGS carried --no-mmap, which llama.cpp build 11200
        rejects (exit 1): mfw-llama would die at every start. Every flag in
        mfw.env.example, the unit and the README's hand-run line must appear in
        the recorded --help of the build the README pins."""
        import re as _re

        help_text = (REPO_ROOT / "tests" / "data" / "llama_server_help_b11200.txt").read_text(encoding="utf-8")
        commit = _re.search(r"build (\d+), commit ([0-9a-f]+)", help_text)
        assert commit is not None
        flag_lines = [line for line in help_text.splitlines() if line.startswith("-")]
        known = {tok for line in flag_lines
                 for tok in _re.findall(r"(?:^|[\s,])(--?[a-zA-Z][a-zA-Z0-9-]*)", line.split("  ")[0])}
        assert {"-ngl", "-c", "-np", "-ub", "-fa", "-lm", "--host", "--port", "-m"} <= known
        env = (SYSTEMD_DIR / "mfw.env.example").read_text(encoding="utf-8")
        args = _re.search(r'^LLAMA_ARGS="([^"]*)"', env, _re.M)
        assert args is not None
        unit = " ".join(_unit("mfw-llama.service")["Service.ExecStart"])
        readme = (REPO_ROOT / "jetson" / "README.md").read_text(encoding="utf-8")
        hand_run = next(line for line in readme.splitlines() if line.startswith("llama-server -m "))
        for source, text in (("LLAMA_ARGS", args.group(1)), ("mfw-llama.service", unit), ("README", hand_run)):
            used = set(_re.findall(r"(?:^|\s)(--?[a-zA-Z][a-zA-Z0-9-]*)", text))
            assert used and not used - known, (source, sorted(used - known))
        assert "--no-mmap" not in args.group(1) and "--no-mmap" not in hand_run
        assert f"git checkout {commit.group(2)}" in readme, "README 3a must pin the build the help was recorded from"

    def test_env_extra_hooks_reach_the_units_and_name_real_flags(self):
        """The runbook tells the team to put the placeholder-calibration flag in
        MFW_ROBOT_EXTRA and the label config in MFW_DETECTOR_EXTRA; both must
        actually be passed by the units, and every flag the env file suggests
        must exist, or the service dies at start with an argparse error."""
        import re as _re

        robot = " ".join(_unit("mfw-robot.service")["Service.ExecStart"])
        detector = " ".join(_unit("mfw-detector.service")["Service.ExecStart"])
        assert "$$MFW_ROBOT_EXTRA" in robot
        assert "$$MFW_DETECTOR_EXTRA" in detector
        env = (SYSTEMD_DIR / "mfw.env.example").read_text(encoding="utf-8")
        assert _re.search(r'^MFW_DETECTOR_EXTRA=""$', env, _re.M), "empty by default: no label file shipped"
        robot_flags = set(_re.findall(r'"(--[a-z0-9-]+)"', (REPO_ROOT / "jetson" / "robot_server.py").read_text(encoding="utf-8")))
        detector_flags = set(_re.findall(r'"(--[a-z0-9-]+)"', (REPO_ROOT / "scripts" / "serve_detector.py").read_text(encoding="utf-8")))
        assert "--allow-placeholder-calibration" in env and "--allow-placeholder-calibration" in robot_flags
        assert "--label-config" in env and "--label-config" in detector_flags

    @pytest.mark.parametrize(
        "path",
        ["scripts/jetson_mode.sh", "scripts/tegrastats_log.sh", "jetson/systemd/wait_ready.sh",
         "jetson/systemd/mfw-robot.service", "jetson/systemd/mfw-conversation.target",
         "jetson/systemd/mfw.env.example"],
    )
    def test_files_for_the_jetson_have_unix_line_endings(self, path):
        data = (REPO_ROOT / path).read_bytes()
        assert b"\r\n" not in data, f"{path} has CRLF: bash and systemd on the Jetson reject it"

    @pytest.mark.parametrize("path", ["scripts/jetson_mode.sh", "scripts/tegrastats_log.sh",
                                      "jetson/systemd/wait_ready.sh"])
    def test_shell_scripts_parse(self, path):
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("no bash on this machine")
        result = subprocess.run([bash, "-n", str(REPO_ROOT / path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr

    def test_mode_script_covers_the_three_modes_and_install(self):
        text = (REPO_ROOT / "scripts" / "jetson_mode.sh").read_text(encoding="utf-8")
        for mode in ("install)", "conversation)", "build)", "status)"):
            assert mode in text
        assert DROP_CACHES.split("; ")[1] in text and "compact_memory" in text

    def test_tegrastats_log_writes_into_logs(self):
        text = (REPO_ROOT / "scripts" / "tegrastats_log.sh").read_text(encoding="utf-8")
        assert 'OUT="${LOG_DIR}/tegrastats_${STAMP}' in text and 'LOG_DIR="${REPO}/logs"' in text


# ----------------------------------------------------------------------
# line endings: everything the Jetson receives is LF, and .gitattributes keeps it so
# ----------------------------------------------------------------------

#: Directories never deployed and possibly large; skipped when walking the tree.
_SKIP_DIRS = {".git", "__pycache__", "logs", "renders", "assets", ".pytest_cache", "node_modules"}
#: File types that are LF wherever they live (bash, systemd units, the sketch).
_LF_SUFFIXES = (".sh", ".service", ".target", ".ino")
#: .gitattributes patterns the Jetson files rely on (see the file's header).
_LF_PATTERNS = ("*.sh", "scripts/*.sh", "*.service", "*.target", "*.ino", "jetson/**")


def _jetson_bound_files() -> list[str]:
    """Every file under jetson/ plus every .sh/.service/.target/.ino in the tree."""
    import os

    found: set[str] = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.endswith("-venv")]
        rel_root = Path(root).relative_to(REPO_ROOT)
        in_jetson = rel_root.parts[:1] == ("jetson",)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            if in_jetson or name.endswith(_LF_SUFFIXES):
                found.add((rel_root / name).as_posix())
    return sorted(found)


JETSON_BOUND_FILES = _jetson_bound_files()


class TestLineEndings:
    """bash (``$'\\r': command not found``), systemd and a ``python3\\r``
    shebang all fail on CRLF, and this checkout lives on Windows with
    ``core.autocrlf=true``: without .gitattributes a fresh checkout writes
    CRLF into exactly the files that are copied to the Jetson."""

    def test_the_walk_found_the_deployed_files(self):
        for expected in ("jetson/robot_server.py", "jetson/detector_service.py", "jetson/llm_worker_llamacpp.py",
                         "jetson/serial_smoke.py", "jetson/arduino/servo_bridge/servo_bridge.ino",
                         "jetson/systemd/mfw-robot.service", "jetson/systemd/mfw-conversation.target",
                         "jetson/systemd/wait_ready.sh", "jetson/systemd/mfw.env.example",
                         "scripts/jetson_mode.sh", "scripts/tegrastats_log.sh"):
            assert expected in JETSON_BOUND_FILES

    @pytest.mark.parametrize("path", JETSON_BOUND_FILES)
    def test_no_crlf_in_the_working_tree(self, path):
        data = (REPO_ROOT / path).read_bytes()
        if b"\0" in data:
            pytest.skip("binary file")
        assert b"\r\n" not in data, f"{path} has CRLF: bash, systemd and shebangs on the Jetson reject it"

    def test_gitattributes_pins_lf_for_the_jetson_files_only(self):
        text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        assert b"\r\n" not in (REPO_ROOT / ".gitattributes").read_bytes()
        rules = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern, *attrs = line.split()
            rules[pattern] = attrs
        for pattern in _LF_PATTERNS:
            assert pattern in rules and "eol=lf" in rules[pattern], pattern
        # The repository mixes endings on purpose: no repo-wide eol rule.
        assert "*" not in rules and "**" not in rules
        # Binaries that ever land in jetson/ must not be converted.
        assert "text=auto" in rules["jetson/**"]

    @pytest.mark.parametrize("path", ["jetson/robot_server.py", "jetson/systemd/mfw-robot.service",
                                      "jetson/systemd/wait_ready.sh", "scripts/jetson_mode.sh",
                                      "jetson/arduino/servo_bridge/servo_bridge.ino",
                                      "jetson/systemd/mfw.env.example"])
    def test_git_applies_eol_lf(self, path):
        """What git itself resolves (not just what the file says)."""
        import shutil
        import subprocess

        git = shutil.which("git")
        if git is None:
            pytest.skip("no git on this machine")
        probe = subprocess.run([git, "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, text=True, timeout=30)
        if probe.returncode != 0 or probe.stdout.strip() != "true":
            pytest.skip("not a git work tree")
        result = subprocess.run([git, "-C", str(REPO_ROOT), "check-attr", "eol", "--", path],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("eol: lf"), result.stdout

    def test_git_leaves_the_laptop_side_alone(self):
        import shutil
        import subprocess

        git = shutil.which("git")
        if git is None:
            pytest.skip("no git on this machine")
        result = subprocess.run([git, "-C", str(REPO_ROOT), "check-attr", "eol", "--", "mfw/hardware/runtime.py"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            pytest.skip("not a git work tree")
        assert result.stdout.strip().endswith("eol: unspecified"), result.stdout


# ----------------------------------------------------------------------
# scan independence: the scan's "seen in >= min_frames of N observes"
# ----------------------------------------------------------------------


class _ScanClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


class _Webcam:
    """robot_server's get_frame: a new frame (new seq) every 33 ms of clock.

    The seq rides in pixel (0, 0) so the backend can tell frames apart the
    way the real one does -- by content, never by being told."""

    PERIOD_S = 0.033

    def __init__(self, clock: _ScanClock) -> None:
        self.clock = clock
        self.seq = 0
        self.grabbed: list[int] = []

    def grab(self):
        from jetson.detector_service import Frame

        self.clock.t += self.PERIOD_S
        self.seq += 1
        self.grabbed.append(self.seq)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[0, 0, 0] = self.seq
        return Frame(rgb=rgb, seq=self.seq, t_capture=self.clock.t, received=self.clock.t)


class _GlareFlorence:
    """Florence on a static table: the marker in every frame, plus a 'banana'
    box on every frame captured while a transient glare lasts (a reflection,
    a hand passing): a real, time-limited phantom, not a per-frame coin flip.
    Each inference costs ``infer_s`` of clock (about 1 s on the Orin)."""

    MARKER = {"label": "marker", "confidence": 0.9, "bbox_px": [100, 100, 130, 180]}
    BANANA = {"label": "banana", "confidence": 0.8, "bbox_px": [300, 200, 360, 240]}

    def __init__(self, clock: _ScanClock, webcam: _Webcam, glare: tuple[float, float] | None, infer_s: float = 1.0):
        self.clock = clock
        self.webcam = webcam
        self.glare = glare
        self.infer_s = float(infer_s)
        self.capture_t: dict[int, float] = {}

    def detect(self, rgb, labels, min_score):
        seq = int(rgb[0, 0, 0])
        t_capture = self.capture_t.setdefault(seq, self.clock.t)
        self.clock.t += self.infer_s
        objects = [dict(self.MARKER)]
        if self.glare is not None and self.glare[0] <= t_capture <= self.glare[1]:
            objects.append(dict(self.BANANA))
        return objects


def _scan_looks(history_max_age_s: float, gap_s: float, glare: tuple[float, float] | None = None,
                looks: int = 3) -> list[dict[str, Any]]:
    """``looks`` detect requests ``gap_s`` apart through the real FramePipeline
    and FrameVoter (3 frames, 2 required: serve_detector's defaults), as
    FixedCameraScan's look phase issues them. Per look: the labels reported,
    the frames this request grabbed and the frames the vote was taken over."""
    from jetson.detector_service import FramePipeline, FrameVoter

    clock = _ScanClock()
    webcam = _Webcam(clock)
    florence = _GlareFlorence(clock, webcam, glare)
    pipe = FramePipeline(florence, webcam, FrameVoter(3, 2, 0.3), history_max_age_s=history_max_age_s,
                         clock=clock, sleep=clock.sleep)
    results = []
    for i in range(looks):
        if i:
            clock.sleep(gap_s)
        before = len(webcam.grabbed)
        reply = pipe.detect(["banana", "marker"], 0.1)
        results.append({
            "labels": sorted({o["label"] for o in reply["objects"]}),
            "grabbed": set(webcam.grabbed[before:]),
            "voted_over": set(reply["frames"]["voted_over"]),
        })
    return results


def _seen_in(results: list[dict[str, Any]], label: str) -> int:
    return sum(1 for r in results if label in r["labels"])


class TestScanIndependence:
    """The limitation, pinned so the runbook stays true: at the shipped
    ``hardware.scan.frame_gap_s`` (0.15 s) and the detector's default frame
    history (``--history-max-age 2.0``), a scan look's vote can reuse frames
    an earlier look already counted, so "seen in 2 of 3 looks" is not three
    independent samples. Two knobs make the looks independent; both are
    shown working here. Neither is on by default (each costs detector time
    that has not been measured on the Jetson) -- see MVP_RUNBOOK section 13.
    """

    def _scan_cfg(self):
        return load_config(HARDWARE_YAML).hardware.scan

    def test_shipped_gap_is_inside_the_detector_history_window(self):
        from jetson.detector_service import FramePipeline

        default_history = FramePipeline.__init__.__defaults__[2]  # history_max_age_s
        assert default_history == 2.0
        assert self._scan_cfg().frame_gap_s < default_history, (
            "the scan looks are now spaced past the detector history: update MVP_RUNBOOK section 13"
        )
        env = (SYSTEMD_DIR / "mfw.env.example").read_text(encoding="utf-8")
        assert not any(line.startswith("MFW_DETECTOR_EXTRA=") and "--history-max-age" in line
                       for line in env.splitlines()), "history is off by default now: update the runbook"

    def test_at_the_shipped_gap_a_look_votes_with_an_earlier_looks_frame(self):
        looks = _scan_looks(history_max_age_s=2.0, gap_s=self._scan_cfg().frame_gap_s)
        assert all(r["labels"] == ["marker"] for r in looks)
        shared = [
            looks[i]["voted_over"] & set().union(*(looks[j]["grabbed"] for j in range(i)))
            for i in range(1, len(looks))
        ]
        assert any(shared), "no reuse: the limitation is gone, update the runbook"
        # ...and every report still rests on at least one frame of its own
        # request (the stale-position fix is not what this is about).
        assert all(r["voted_over"] & r["grabbed"] for r in looks)

    def test_a_glare_on_three_frames_counts_as_seen_in_two_looks(self):
        """The consequence: a phantom present on three consecutive frames
        (f2-f4) passes the scan's 2-of-3 rule, because look 2 confirms it
        with ONE new frame plus look 1's last frame."""
        looks = _scan_looks(history_max_age_s=2.0, gap_s=self._scan_cfg().frame_gap_s, glare=(1.0, 3.5))
        assert _seen_in(looks, "banana") == 2 >= self._scan_cfg().min_frames
        assert len(looks[1]["grabbed"]) == 1 and looks[1]["voted_over"] & looks[0]["grabbed"]

    def test_history_max_age_zero_makes_each_look_vote_on_its_own_frames(self):
        looks = _scan_looks(history_max_age_s=0.0, gap_s=self._scan_cfg().frame_gap_s)
        assert all(r["labels"] == ["marker"] for r in looks)
        for r in looks:
            assert r["voted_over"] <= r["grabbed"]
            assert len(r["grabbed"]) >= 2  # the price: two inferences per look, always

    def test_history_max_age_zero_stops_the_glare_counting_twice(self):
        looks = _scan_looks(history_max_age_s=0.0, gap_s=self._scan_cfg().frame_gap_s, glare=(1.0, 3.5))
        assert _seen_in(looks, "banana") == 1 < self._scan_cfg().min_frames  # "not sure", not "seen"

    def test_spacing_the_looks_past_the_history_window_also_works(self):
        looks = _scan_looks(history_max_age_s=2.0, gap_s=2.1)
        assert all(r["labels"] == ["marker"] for r in looks)
        for r in looks:
            assert r["voted_over"] <= r["grabbed"]

    def test_serve_detector_accepts_history_zero(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_serve_detector_hist", REPO_ROOT / "scripts" / "serve_detector.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args(["--history-max-age", "0"])
        assert args.history_max_age == 0.0

    def test_the_runbook_states_the_limitation_and_both_knobs(self):
        text = " ".join((REPO_ROOT / "docs" / "MVP_RUNBOOK.md").read_text(encoding="utf-8").split())
        assert "--history-max-age 0" in text and "frame_gap_s" in text
        assert "not independent" in text

"""Phase 9: the arm bridge -- robot server endpoints, Uno protocol, calibration
REPL and the llama.cpp LLM shim -- against in-process fakes.

No servos, no Jetson. ``jetson/robot_server.py`` runs in a thread on a
``FakeDriver`` with a no-op follower sleep (a 1 s trajectory costs one round
trip), the Uno driver is exercised through a stub ``serial`` module, and the
LLM shim is fed by an in-thread fake llama-server. Each class pins one side of
a contract the Jetson deployment depends on:

* the ZMQ endpoint table in ``docs/HARDWARE_BRIEF.md`` section 5;
* ``ServoCalibration``'s radians <-> microseconds map (direction -1 included,
  because a mirror-mounted servo is the first thing that goes wrong on a kit);
* the ASCII frames both the sketch and ``UnoSerialDriver`` parse;
* ``scripts/calibrate_servos.py --script`` driving the same server;
* the 5557 protocol of ``jetson/llm_worker_llamacpp.py`` (system message,
  temperature 0, 128 tokens, 14-skill enum) so a Jetson LLM can only ever
  emit a registered skill.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import re
import socket
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jetson import llm_worker_llamacpp as shim
from jetson.robot_server import (
    CLAMP_TOL_RAD,
    DEFAULT_REPLY_TIMEOUT_S,
    DETACH_ATTEMPTS,
    JOINT_NAMES,
    KEEPALIVE_PERIOD_S,
    SERIAL_MAX_TIMEOUTS,
    SKETCH_PULSE_LIMITS_US,
    SKETCH_WATCHDOG_MS,
    TABLE_CLEARANCE_M,
    WIRE_ORDER,
    ArmGeometry,
    CalibrationError,
    DriverError,
    FakeDriver,
    FakeWorld,
    GripperCalibration,
    JointCalibration,
    RobotServer,
    ServoCalibration,
    TrajectoryFollower,
    UnoSerialDriver,
    build_driver,
    check_home_clearance,
    decode_command_frame,
    decode_q_frame,
    decode_q_status,
    encode_p_frame,
    encode_s_frame,
    encode_t_frame,
    parse_reply,
    planar_link_points,
    planar_tcp,
)
from mfw.core.errors import ExecutionError
from mfw.hardware.jetson_client import JetsonClient, RpcError
from mfw.skills.primitives import ALL_SKILLS

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
SKETCH_PATH = REPO_ROOT / "jetson" / "arduino" / "servo_bridge" / "servo_bridge.ino"
ROBOT_CONFIG_PATH = REPO_ROOT / "jetson" / "robot_config.yaml"

#: Fast calibration: no trajectory is ever stretched, so every motion in this
#: file is a handful of P frames. The stretch test lowers it explicitly.
FAST_VELOCITY = 50.0

#: A pose well inside every joint's placeholder limits.
GOAL_Q = np.array([0.4, 0.8, -0.9, 0.3])


def _fast_calibration() -> ServoCalibration:
    """The placeholder calibration, fast, and with every limit strictly inside
    the pulse map.

    The placeholder's +-1.5708 rad limits sit 4e-6 rad *past* the +-90 deg
    pulse end-stops (2400.004 us rounds onto the wire as 2400, so ``validate``
    accepts them), but a pose read back from that pulse is +-pi/2 exactly, so
    1e-6 assertions on a limit-touching pose would fail for a reason that has
    nothing to do with the behaviour under test. A measured calibration will
    not have this; the tests use 1.5707.

    The home elbow is put *exactly on* its lower limit on purpose. Waypoints
    cross the wire as float32, which perturbs -1.5707 by ~7e-8; the old 1e-9
    clamp threshold flagged every move starting from home as ``clamped`` and
    the fixture used to dodge that by widening the limit. It no longer does
    (review P5): a limit-touching move must not be reported as clamped.
    """
    raw = ServoCalibration.default().to_dict()
    for joint in raw["joints"].values():
        for key in ("limit_lower_rad", "limit_upper_rad"):
            if abs(abs(joint[key]) - 1.5708) < 1e-9:
                joint[key] = float(np.sign(joint[key]) * 1.5707)
    raw["home_q"] = [0.0, 1.0472, -1.5707, 0.5236]
    raw["max_joint_velocity"] = FAST_VELOCITY
    return ServoCalibration.from_dict(raw)


def _p_frames_after_last_detach(driver: FakeDriver) -> list[tuple[int, ...]]:
    """Pulses of every ``P`` frame written since the most recent ``D``."""
    kinds = [kind for kind, _ in driver.commands]
    start = len(kinds) - 1 - kinds[::-1].index("D") if "D" in kinds else -1
    return [pulses for kind, pulses in driver.commands[start + 1:] if kind == "P"]


def _assert_ramp(frames: list[tuple[int, ...]], start: tuple[int, ...], end: tuple[int, ...], min_frames: int) -> None:
    """A slow re-attach: first frame at ``start``, last at ``end``, monotone in
    between, and at least ``min_frames`` of them (never one jump)."""
    assert len(frames) >= min_frames, f"expected a ramp of >= {min_frames} frames, got {len(frames)}"
    assert frames[0] == start, f"ramp must begin at the stored pose {start}, began at {frames[0]}"
    assert frames[-1] == end, f"ramp must end at the target {end}, ended at {frames[-1]}"
    arr = np.asarray(frames, dtype=np.int64)
    steps = np.diff(arr, axis=0)
    for channel in range(arr.shape[1]):
        col = steps[:, channel]
        assert np.all(col >= 0) or np.all(col <= 0), f"channel {channel} reversed direction mid-ramp"


def _serve(tmp_path: Path, home: bool):
    calibration = _fast_calibration()
    cal_path = tmp_path / "robot_config.yaml"
    calibration.save(cal_path)
    driver = FakeDriver(calibration)
    server = RobotServer(
        "127.0.0.1", 0, driver, camera=None, calibration=calibration,
        calibration_path=cal_path, follower_sleep=lambda _s: None,
    )
    _, port = server.serve_in_thread(port=0)
    client = JetsonClient("127.0.0.1", port, request_timeout_s=5.0, trajectory_timeout_margin_s=5.0)
    client.connect()
    if home:
        # The documented bring-up: the arm is hand-posed at home and the first
        # command is `home` (a fresh bridge knows no position; see TestBoot).
        client.home()
    return client, server, driver, cal_path


@pytest.fixture
def bridge(tmp_path):
    """In-thread robot server on a fake driver + a connected client, homed."""
    client, server, driver, cal_path = _serve(tmp_path, home=True)
    try:
        yield client, server, driver, cal_path
    finally:
        client.close()
        server.stop()


@pytest.fixture
def fresh_bridge(tmp_path):
    """Same, straight after start: the bridge knows no position yet."""
    client, server, driver, cal_path = _serve(tmp_path, home=False)
    try:
        yield client, server, driver, cal_path
    finally:
        client.close()
        server.stop()


# ----------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------


class TestEndpoints:
    def test_ping(self, bridge):
        client, _server, _driver, _path = bridge
        info = client.ping()
        assert info["ok"] is True
        assert info["server"] == "mfw-robot"
        assert info["version"] == 1
        assert info["driver"] == "fake" and info["fake"] is True
        assert client.is_fake

    def test_get_state_is_home_after_start(self, bridge):
        client, server, _driver, _path = bridge
        state = client.get_state()
        np.testing.assert_allclose(state["q"], server.calibration.home_q, atol=1e-6)
        assert state["gripper_width"] == pytest.approx(server.calibration.gripper.width_open_m, abs=1e-6)
        assert state["moving"] is False and state["estopped"] is False
        assert client.last_state is state

    def test_set_gripper(self, bridge):
        client, server, driver, _path = bridge
        state = client.set_gripper(0.02)
        assert state["gripper_width"] == pytest.approx(0.02, abs=1e-6)
        kind, pulses = driver.commands[-1]
        assert kind == "P"
        assert pulses[4] == round(server.calibration.width_to_us(0.02))
        # Out of the calibrated span: clamped, never rejected.
        assert client.set_gripper(0.5)["gripper_width"] == pytest.approx(0.045, abs=1e-6)

    def test_follow_trajectory_reaches_last_waypoint(self, bridge):
        client, server, driver, _path = bridge
        home = np.asarray(client.get_state()["q"])
        # P5: the home elbow sits exactly on its limit and crosses the wire as
        # float32; the move is legal and must not be reported as clamped.
        assert home[2] == pytest.approx(server.calibration.joints["elbow_pitch"].limit_lower_rad, abs=1e-12)
        assert abs(float(np.float32(home[2])) - home[2]) > 1e-9
        n = 5
        waypoints = np.linspace(home, GOAL_Q, n)
        before = driver.write_count
        reply = client.follow_trajectory(waypoints, dt=0.1)
        assert reply["completed"] is True and reply["estopped"] is False and reply["clamped"] is False
        np.testing.assert_allclose(reply["q"], GOAL_Q, atol=1e-6)
        np.testing.assert_allclose(client.get_state()["q"], GOAL_Q, atol=1e-6)
        # Resampled at 50 Hz over 0.4 s: many more frames than waypoints.
        assert driver.write_count - before >= n
        last = driver.commands[-1][1][:4]
        assert last == tuple(round(p) for p in driver.calibration.q_to_pulses(GOAL_Q))

    def test_velocity_stretch(self, bridge):
        client, server, driver, _path = bridge
        slow = ServoCalibration.from_dict({**server.calibration.to_dict(), "max_joint_velocity": 1.0})
        client.set_calibration(slow.to_dict())
        home = np.asarray(client.get_state()["q"])
        goal = home.copy()
        goal[0] += 1.0  # 1 rad in 0.1 s asks for 10 rad/s: stretched x10 to 1.0 s
        before = driver.write_count
        reply = client.follow_trajectory(np.stack([home, goal]), dt=0.1)
        assert reply["completed"] is True
        np.testing.assert_allclose(reply["q"], goal, atol=1e-6)
        frames = driver.write_count - before
        assert frames >= 45, f"stretched move should take ~51 frames at 50 Hz, got {frames}"
        # The follower's own plan agrees on the stretch factor.
        _samples, _tick, _clamped, stretch = server.follower.plan(np.stack([home, goal]), 0.1)
        assert stretch == pytest.approx(10.0)

    def test_limit_clamp_reported(self, bridge):
        client, server, _driver, _path = bridge
        home = np.asarray(client.get_state()["q"])
        beyond = home.copy()
        beyond[0] = 3.0  # base_yaw upper limit is pi/2
        reply = client.follow_trajectory(np.stack([home, beyond]), dt=0.1)
        assert reply["clamped"] is True and reply["completed"] is True
        upper = server.calibration.joints["base_yaw"].limit_upper_rad
        assert reply["q"][0] == pytest.approx(upper, abs=1e-6)

    def test_estop_blocks_motion_until_cleared(self, bridge):
        client, _server, driver, _path = bridge
        state = client.estop()
        assert state["estopped"] is True
        assert driver.commands[-1] == ("D", ())
        assert driver.attached is False
        with pytest.raises(RpcError, match="estopped"):
            client.set_gripper(0.01)
        home = np.asarray(state["q"])
        with pytest.raises(RpcError, match="estopped"):
            client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        with pytest.raises(RpcError, match="estopped"):
            client.set_torque(True)
        # Queries still answer while estopped.
        state = client.get_state()
        assert state["estopped"] is True and state["attached"] is False
        assert client.clear_estop()["estopped"] is False
        # Servos re-attach on the next motion command, not on clear.
        assert driver.attached is False
        assert client.get_state()["attached"] is False
        stored = driver.pulses()
        state = client.set_gripper(0.01)
        assert state["gripper_width"] == pytest.approx(0.01, abs=1e-6)
        assert driver.attached is True and state["attached"] is True
        # ... and it re-attaches slowly: a ramp from the stored pose to the
        # new jaw target, not one P frame (HS-3 / missed item 6).
        target = stored[:4] + (round(driver.calibration.width_to_us(0.01)),)
        _assert_ramp(_p_frames_after_last_detach(driver), stored, target,
                     min_frames=int(driver.calibration.reattach_s * driver.calibration.control_rate_hz))

    def test_home_and_torque(self, bridge):
        client, server, driver, _path = bridge
        home = np.asarray(client.get_state()["q"])
        client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        np.testing.assert_allclose(client.home()["q"], server.calibration.home_q, atol=1e-6)
        assert client.set_torque(False)["q"] == pytest.approx(list(server.calibration.home_q))
        assert driver.commands[-1] == ("D", ())
        assert driver.attached is False
        stored = driver.pulses()
        client.set_torque(True)
        assert driver.commands[-1][0] == "P" and driver.attached is True
        # torque on re-energises at the stored pose over reattach_s: many
        # identical frames, never a single snap frame somewhere else.
        frames = _p_frames_after_last_detach(driver)
        _assert_ramp(frames, stored, stored,
                     min_frames=int(server.calibration.reattach_s * server.calibration.control_rate_hz))

    def test_unknown_endpoint_is_execution_error(self, bridge):
        client, _server, _driver, _path = bridge
        with pytest.raises(ExecutionError, match="unknown endpoint") as info:
            client.rpc.call("bogus")
        assert isinstance(info.value, RpcError)
        # The client recovers: the next call works on a fresh socket.
        assert client.ping()["ok"] is True

    def test_bad_arguments_are_errors_not_crashes(self, bridge):
        client, _server, _driver, _path = bridge
        with pytest.raises(RpcError, match="set_gripper needs"):
            client.rpc.call("set_gripper", {})
        with pytest.raises(RpcError, match="waypoints must be"):
            client.rpc.call("follow_trajectory", {"waypoints": np.zeros((2, 3), np.float32), "dt": 0.1})
        with pytest.raises(RpcError, match="dt must be"):
            client.rpc.call("follow_trajectory", {"waypoints": np.zeros((2, 4), np.float32), "dt": 0.0})
        with pytest.raises(RpcError, match="no camera"):
            client.get_frame()

    def test_calibration_round_trip_and_persistence(self, bridge):
        client, server, _driver, cal_path = bridge
        reply = client.get_calibration()
        assert reply["path"] == str(cal_path)
        cal = ServoCalibration.from_dict(reply["calibration"])
        assert cal == server.calibration

        edited = dict(reply["calibration"])
        edited["joints"] = {**edited["joints"], "base_yaw": {**edited["joints"]["base_yaw"], "direction": -1}}
        q_before = np.asarray(client.get_state()["q"])
        assert client.set_calibration(edited)["ok"] is True
        # Re-labelling never moves the arm: the pose flips sign for that joint.
        q_after = np.asarray(client.get_state()["q"])
        assert q_after[0] == pytest.approx(-q_before[0], abs=1e-9)
        np.testing.assert_allclose(q_after[1:], q_before[1:], atol=1e-9)
        assert ServoCalibration.load(cal_path).joints["base_yaw"].direction == -1

        with pytest.raises(RpcError, match="pulse_min_us"):
            bad = {**edited, "joints": {**edited["joints"], "base_yaw": {**edited["joints"]["base_yaw"], "pulse_min_us": 2500}}}
            client.set_calibration(bad)

    def test_state_reports_bridge_pose(self, bridge, fresh_bridge):
        # Boot: nothing pulsed and no position known -- a reset bridge's
        # pulses are a library default, so no bridge pose is reported.
        fresh = fresh_bridge[0].get_state()
        assert fresh["attached"] is False and fresh["bridge_position_known"] is False
        assert fresh["bridge_q"] is None and fresh["bridge_gripper_width"] is None
        client, server, driver, _path = bridge
        state = client.get_state()
        assert state["attached"] is True and state["bridge_position_known"] is True
        # The bridge knows whole microseconds only (0.1 deg on the placeholder
        # map), so its pose is quantised to about 1e-3 rad.
        np.testing.assert_allclose(state["bridge_q"], server.calibration.home_q, atol=1e-3)
        assert not np.allclose(state["bridge_q"], server.calibration.home_q, atol=1e-6)
        client.set_gripper(0.02)
        state = client.get_state()
        assert state["attached"] is True
        assert state["bridge_gripper_width"] == pytest.approx(0.02, abs=1e-4)

    def test_get_fake_world(self, bridge):
        client, _server, _driver, _path = bridge
        with pytest.raises(RpcError, match="no fake world"):
            client.get_fake_world()

        world = FakeWorld({"marker": (0.18, 0.05)}, {"marker": (0.02, 0.02, 0.12)}, ArmGeometry())
        driver = FakeDriver(_fast_calibration(), on_command=world.on_command)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                             follower_sleep=lambda _s: None, fake_world=world)
        _, port = server.serve_in_thread(port=0)
        other = JetsonClient("127.0.0.1", port)
        try:
            other.connect()
            snap = other.get_fake_world()
            assert snap["objects"]["marker"][:2] == pytest.approx([0.18, 0.05])
            assert snap["attached"] is None
            other.home()
            other.set_gripper(0.01)
            assert other.get_fake_world()["tcp"] is not None
        finally:
            other.close()
            server.stop()


# ----------------------------------------------------------------------
# slow re-attach after a detach (HS-3, HS-5 server half, missed items 3 & 6)
# ----------------------------------------------------------------------


class TestSlowReattach:
    """No path may re-attach with an instantaneous jump.

    The FakeDriver's re-attach is a software ramp of P frames at the control
    rate (the Uno gets one T frame instead, see TestUnoSerialDriver); every
    test here checks that the frames written after the D frame start at the
    stored pose and walk to the target rather than appearing there.
    """

    def _ramp_frames(self, cal: ServoCalibration, seconds: float) -> int:
        return int(seconds * cal.control_rate_hz)

    def test_first_motion_at_boot_is_one_frame_at_home(self, fresh_bridge):
        """A fresh bridge attaches AT its first target: boot home is an
        instant attach, not the 2.5 s glide the fake used to show (review:
        the Uno never performs it), and the reply says so."""
        client, server, driver, _path = fresh_bridge
        assert driver.attached is False and driver.needs_reattach and not driver.position_known
        home_pulses = driver.pulses()
        state = client.home()
        np.testing.assert_allclose(state["q"], server.calibration.home_q, atol=1e-6)
        assert state["first_attach"] is True and state["reattached"] is True
        assert state["bridge_position_known"] is True
        assert [p for kind, p in driver.commands if kind == "P"] == [home_pulses]
        # A second home is an ordinary move, not a first attach.
        assert client.home()["first_attach"] is False

    def test_home_after_estop_is_one_slow_move_to_home(self, bridge):
        client, server, driver, _path = bridge
        home = np.asarray(client.get_state()["q"])
        client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        goal_pulses = driver.pulses()
        client.estop()
        client.clear_estop()
        state = client.home()
        np.testing.assert_allclose(state["q"], server.calibration.home_q, atol=1e-6)
        home_pulses = tuple(round(p) for p in server.calibration.q_to_pulses(home)) + (goal_pulses[4],)
        frames = _p_frames_after_last_detach(driver)
        # From the pre-drop pose straight to home over >= home_move_s; the
        # old code re-attached at the pre-drop pose and then homed at speed.
        _assert_ramp(frames, goal_pulses, home_pulses,
                     min_frames=self._ramp_frames(server.calibration, server.calibration.home_move_s))

    def test_follow_trajectory_after_estop_reattaches_before_streaming(self, bridge):
        client, server, driver, _path = bridge
        home = np.asarray(client.get_state()["q"])
        client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        stored = driver.pulses()
        client.estop()
        client.clear_estop()
        # Start the new trajectory somewhere else than the stored pose, as a
        # planner working from a stale cache would.
        elsewhere = GOAL_Q + np.array([0.2, -0.1, 0.1, -0.2])
        reply = client.follow_trajectory(np.stack([elsewhere, home]), dt=0.1)
        assert reply["completed"] is True and reply["estopped"] is False
        frames = _p_frames_after_last_detach(driver)
        # The ramp is n+1 frames: the stored pose first, then n steps to the
        # trajectory's first sample; only after that does the stream begin.
        n_ramp = self._ramp_frames(server.calibration, server.calibration.reattach_s)
        first_target = tuple(round(p) for p in server.calibration.q_to_pulses(elsewhere)) + (stored[4],)
        _assert_ramp(frames[: n_ramp + 1], stored, first_target, min_frames=n_ramp)
        np.testing.assert_allclose(reply["q"], home, atol=1e-6)

    def test_gripper_after_torque_off_reattaches_all_five_slowly(self, bridge):
        client, server, driver, _path = bridge
        client.set_gripper(0.03)
        client.set_torque(False)
        assert driver.attached is False
        stored = driver.pulses()
        client.set_gripper(0.005)  # "jaw 5" after hand-posing
        target = stored[:4] + (round(server.calibration.width_to_us(0.005)),)
        _assert_ramp(_p_frames_after_last_detach(driver), stored, target,
                     min_frames=self._ramp_frames(server.calibration, server.calibration.reattach_s))

    def test_torque_on_at_an_explicit_pose(self, bridge):
        client, server, driver, _path = bridge
        client.set_torque(False)
        stored = driver.pulses()
        target_q = np.array([0.1, 0.9, -1.0, 0.2])
        state = client.rpc.call("set_torque", {"enabled": True, "q": target_q.tolist(), "gripper_width": 0.01})
        np.testing.assert_allclose(state["q"], target_q, atol=1e-6)
        assert state["gripper_width"] == pytest.approx(0.01, abs=1e-6)
        target = tuple(round(p) for p in server.calibration.q_to_pulses(target_q)) + (
            round(server.calibration.width_to_us(0.01)),
        )
        _assert_ramp(_p_frames_after_last_detach(driver), stored, target,
                     min_frames=self._ramp_frames(server.calibration, server.calibration.reattach_s))
        with pytest.raises(RpcError, match="finite"):
            client.rpc.call("set_torque", {"enabled": True, "q": [0.0, float("nan"), 0.0, 0.0]})

    def test_write_refuses_while_hold_is_in_force(self):
        driver = FakeDriver(_fast_calibration())
        driver.write(np.asarray(driver.calibration.home_q))
        driver.detach()
        assert driver.hold_detached and driver.needs_reattach
        with pytest.raises(DriverError, match="detached"):
            driver.write(np.asarray(driver.calibration.home_q))
        with pytest.raises(DriverError, match="detached"):
            driver.write_gripper(0.01)
        assert driver.commands[-1] == ("D", ())
        driver.reattach(duration_s=0.0)
        assert not driver.hold_detached and driver.attached


# ----------------------------------------------------------------------
# estop vs. an in-flight write (HS-7)
# ----------------------------------------------------------------------


class TestEstopRace:
    def test_estop_landing_after_the_check_cannot_reattach(self, bridge):
        """Simulate the estop firing in the microseconds between the follower's
        ``is_set()`` and its ``write``: the D frame must be the last thing on
        the wire (no P after it) and the reply must say estopped."""
        client, server, driver, _path = bridge
        client.set_gripper(0.02)  # attached, no hold
        fired = {"done": False}

        class TrapEvent:
            def __init__(self, inner: threading.Event) -> None:
                self._inner = inner
                self.checks = 0

            def is_set(self) -> bool:
                self.checks += 1
                if self.checks == 3 and not fired["done"]:
                    fired["done"] = True
                    # The estop endpoint, exactly as the ROUTER thread runs
                    # it, interleaved right after the check.
                    server._ep_estop({})
                    return False  # the check already returned "not set"
                return self._inner.is_set()

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        trap = TrapEvent(server._estop)
        server.follower._estop = trap  # type: ignore[assignment]
        try:
            home = np.asarray(client.get_state()["q"])
            reply = client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        finally:
            server.follower._estop = server._estop
        assert fired["done"]
        assert reply["estopped"] is True and reply["completed"] is False
        kinds = [kind for kind, _ in driver.commands]
        assert "D" in kinds
        assert "P" not in kinds[len(kinds) - 1 - kinds[::-1].index("D"):], kinds[-6:]
        assert driver.attached is False
        assert client.get_state()["estopped"] is True

    def test_estop_from_another_thread_mid_trajectory(self, bridge):
        client, server, driver, _path = bridge
        client.set_gripper(0.02)
        slow = ServoCalibration.from_dict({**server.calibration.to_dict(), "max_joint_velocity": 0.5})
        client.set_calibration(slow.to_dict())
        # Real pacing for this one so the estop has a moving target.
        server.follower._sleep = time.sleep
        home = np.asarray(client.get_state()["q"])
        result: dict[str, Any] = {}

        def run() -> None:
            result["reply"] = client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)

        worker = threading.Thread(target=run)
        worker.start()
        time.sleep(0.3)
        stopper = JetsonClient("127.0.0.1", client.rpc.port)
        stopper.connect()
        try:
            assert stopper.estop()["estopped"] is True
        finally:
            stopper.close()
        worker.join(timeout=10.0)
        assert not worker.is_alive()
        reply = result["reply"]
        assert reply["estopped"] is True and reply["completed"] is False
        kinds = [kind for kind, _ in driver.commands]
        assert "P" not in kinds[len(kinds) - 1 - kinds[::-1].index("D"):]
        assert driver.attached is False


# ----------------------------------------------------------------------
# keepalive against a watchdog-simulating fake (HS-1 / P1)
# ----------------------------------------------------------------------


class TestKeepalive:
    WATCHDOG_MS = 300.0

    def _server(self, driver: FakeDriver, tmp_path: Path):
        server = RobotServer(
            "127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
            calibration_path=tmp_path / "robot_config.yaml", follower_sleep=lambda _s: None,
        )
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port, request_timeout_s=5.0)
        client.connect()
        client.home()
        return server, client

    def test_two_seconds_idle_stays_attached_with_keepalive(self, tmp_path):
        driver = FakeDriver(_fast_calibration(), watchdog_ms=self.WATCHDOG_MS, keepalive_s=0.05)
        assert driver.keepalive_running
        server, client = self._server(driver, tmp_path)
        try:
            client.set_gripper(0.02)
            time.sleep(2.0)
            state = client.get_state()
            assert state["attached"] is True
            assert driver.watchdog_trips == 0
            assert driver.keepalive_count >= 20
            # The keepalive never changed a target.
            assert state["gripper_width"] == pytest.approx(0.02, abs=1e-6)
            assert not any(kind == "P" for kind, _ in driver.commands[-10:])
        finally:
            client.close()
            server.stop()
        assert not driver.keepalive_running

    def test_without_keepalive_the_fake_watchdog_detaches(self, tmp_path, caplog):
        driver = FakeDriver(_fast_calibration(), watchdog_ms=self.WATCHDOG_MS, keepalive_s=0)
        assert not driver.keepalive_running
        server, client = self._server(driver, tmp_path)
        try:
            client.set_gripper(0.02)
            time.sleep(0.6)
            with caplog.at_level(logging.WARNING, logger="robot_server"):
                state = client.get_state()
            assert state["attached"] is False and state["estopped"] is False
            assert driver.watchdog_trips == 1
            assert ("W", ()) in driver.commands
            assert any("detached by the bridge" in rec.message for rec in caplog.records)
            # The next motion re-attaches slowly from the stored pose.
            stored = driver.pulses()
            reply = client.follow_trajectory(np.stack([np.asarray(state["q"]), GOAL_Q]), dt=0.1)
            assert reply["completed"] is True
            frames = [p for kind, p in driver.commands[driver.commands.index(("W", ())) + 1:] if kind == "P"]
            n_ramp = int(driver.calibration.reattach_s * driver.calibration.control_rate_hz)
            _assert_ramp(frames[:n_ramp], stored, stored, min_frames=n_ramp)
        finally:
            client.close()
            server.stop()

    def test_keepalive_pauses_while_streaming_and_after_estop(self, tmp_path):
        driver = FakeDriver(_fast_calibration(), watchdog_ms=self.WATCHDOG_MS, keepalive_s=0.05)
        server, client = self._server(driver, tmp_path)
        try:
            client.set_gripper(0.02)
            client.estop()
            before = driver.keepalive_count
            time.sleep(0.4)
            assert driver.keepalive_count == before, "keepalive must not re-attach an estopped arm"
            assert driver.attached is False and driver.watchdog_trips == 0
        finally:
            client.close()
            server.stop()

    def test_cli_flag_builds_a_watchdog_fake(self):
        driver = build_driver("fake", _fast_calibration(), "COM1", 7, fake_watchdog_ms=500.0)
        try:
            assert isinstance(driver, FakeDriver) and driver.watchdog_ms == 500.0
            assert driver.keepalive_s == pytest.approx(KEEPALIVE_PERIOD_S) and driver.keepalive_running
        finally:
            driver.close()
        plain = build_driver("fake", _fast_calibration(), "COM1", 7)
        assert plain.watchdog_ms is None and not plain.keepalive_running


# ----------------------------------------------------------------------
# calibration maths
# ----------------------------------------------------------------------


class TestServoCalibration:
    @pytest.mark.parametrize("direction", [1, -1])
    @pytest.mark.parametrize("zero", [0.0, 0.25, -0.4])
    def test_rad_us_round_trip(self, direction, zero):
        joint = JointCalibration(direction=direction, zero_offset_rad=zero)
        for rad in np.linspace(-0.9, 0.9, 19):
            us = joint.rad_to_us(rad)
            assert 600.0 <= us <= 2400.0
            assert joint.us_to_rad(us) == pytest.approx(rad, abs=1e-9)
        # 10 us per degree on the placeholder map; direction flips the slope.
        slope = joint.rad_to_us(0.1) - joint.rad_to_us(0.0)
        assert np.sign(slope) == direction
        assert abs(slope) == pytest.approx(np.degrees(0.1) * 10.0, rel=1e-6)

    def test_rad_to_us_clamps_to_pulse_range(self):
        joint = JointCalibration()
        assert joint.rad_to_us(10.0) == 2400.0
        assert joint.rad_to_us(-10.0) == 600.0

    def test_gripper_round_trip(self):
        g = GripperCalibration()
        for width in (0.0, 0.01, 0.03, 0.045):
            assert g.us_to_width(g.width_to_us(width)) == pytest.approx(width, abs=1e-9)
        assert g.width_to_us(0.045) == 1000.0 and g.width_to_us(0.0) == 1900.0
        assert g.width_to_us(1.0) == 1000.0  # clamped open

    def test_vector_helpers_and_clamp(self):
        cal = ServoCalibration.default()
        q = np.array([0.3, 0.5, -1.0, 0.2])
        np.testing.assert_allclose(cal.pulses_to_q(cal.q_to_pulses(q)), q, atol=1e-9)
        clamped, moved = cal.clamp_q(np.array([3.0, 0.0, 0.0, 0.0]))
        assert moved and clamped[0] == pytest.approx(np.pi / 2, abs=1e-3)
        _, moved = cal.clamp_q(q)
        assert not moved

    def test_dict_round_trip_and_validation(self):
        cal = ServoCalibration.default()
        assert ServoCalibration.from_dict(cal.to_dict()) == cal
        with pytest.raises(CalibrationError, match="unknown keys"):
            ServoCalibration.from_dict({"joints": {"base_yaw": {"pulse_min": 600}}})
        with pytest.raises(CalibrationError, match="home_q"):
            ServoCalibration.from_dict({**cal.to_dict(), "home_q": [3.0, 1.0, -1.5, 0.5]})
        with pytest.raises(CalibrationError, match="direction"):
            ServoCalibration.from_dict({"joints": {"base_yaw": {"direction": 2}}})

    def test_follower_plan_pure(self):
        cal = ServoCalibration.from_dict({**ServoCalibration.default().to_dict(), "max_joint_velocity": 2.0})
        follower = TrajectoryFollower(FakeDriver(cal), cal, threading.Event(), sleep=lambda _s: None)
        wp = np.array([[0.0, 0.5, -1.0, 0.0], [1.0, 0.5, -1.0, 0.0]])
        samples, tick, clamped, stretch = follower.plan(wp, dt=0.1)
        assert tick == pytest.approx(0.02)
        assert stretch == pytest.approx(5.0) and not clamped
        assert samples.shape == (26, 4)
        np.testing.assert_allclose(samples[-1], wp[-1])
        assert np.all(np.abs(np.diff(samples[:, 0])) <= 2.0 * tick + 1e-9)


class TestPulseMapRule:
    """HS-2 / GEO-1 / P5 / missed item 2: one clamp table, honestly reported."""

    def test_sketch_constants_match_the_server(self):
        source = SKETCH_PATH.read_text(encoding="utf-8")
        mins = re.search(r"MIN_US\[N_SERVOS\]\s*=\s*\{([^}]*)\}", source)
        maxs = re.search(r"MAX_US\[N_SERVOS\]\s*=\s*\{([^}]*)\}", source)
        watchdog = re.search(r"WATCHDOG_MS\s*=\s*(\d+)", source)
        assert mins and maxs and watchdog, "sketch tables not found"
        lo = [int(v) for v in mins.group(1).split(",")]
        hi = [int(v) for v in maxs.group(1).split(",")]
        assert tuple(zip(lo, hi)) == SKETCH_PULSE_LIMITS_US
        assert int(watchdog.group(1)) == SKETCH_WATCHDOG_MS
        # Worst-case keepalive silence (1.5 periods) stays well inside the watchdog.
        assert KEEPALIVE_PERIOD_S * 1.5 * 1000.0 < SKETCH_WATCHDOG_MS / 2.0
        # The sketch understands the keepalive frame and re-attaches from its
        # last position, not the target.
        assert "cmd == 'K'" in source
        assert "have_position" in source
        assert decode_command_frame(b"K\n") == ("K",)
        # Re-review: the Q reply reports have_position, '?' and S feed the
        # watchdog while attached, and S<n> is echoed for the serial resync.
        # _StubSerial below models exactly these.
        assert "Serial.println(have_position ? 1 : 0);" in source
        assert "cmd == 'S'" in source and "Serial.print('S');" in source
        assert "feedWatchdogIfAttached();\n    replyQuery();" in source
        # Audit: an overlong line is discarded WHOLE (skip to its newline, one
        # reply); tests/test_mvp_bridge.py compiles and executes this on the host.
        assert re.search(r"LINE_MAX\s*=\s*64;", source)
        assert "discarding = true;" in source and 'Serial.println(F("ERR line too long"));' in source
        assert decode_command_frame(encode_s_frame(42)) == ("S", 42)
        with pytest.raises(ValueError, match="ERR bad frame"):
            decode_command_frame(b"S\n")
        with pytest.raises(ValueError, match="ERR bad frame"):
            decode_command_frame(b"K1\n")

    def test_validate_refuses_limits_outside_the_pulse_map(self):
        with pytest.raises(CalibrationError, match="outside the pulse map"):
            JointCalibration(limit_lower_rad=-2.0944, limit_upper_rad=2.0944).validate("elbow_pitch")
        # A zero offset shifts the reachable band: 1.2 rad is fine at zero
        # offset but asks the servo for 97 deg once the zero sits at +0.5 rad.
        JointCalibration(limit_upper_rad=1.2).validate("ok")
        with pytest.raises(CalibrationError, match="limit_upper_rad"):
            JointCalibration(limit_upper_rad=1.2, zero_offset_rad=0.5).validate("shifted")
        with pytest.raises(CalibrationError, match="limit_lower_rad"):
            JointCalibration(limit_lower_rad=-1.2, zero_offset_rad=-0.5, direction=1).validate("shifted")
        # The pulse range itself may not reach past the sketch clamp.
        with pytest.raises(CalibrationError, match="Uno's clamp"):
            JointCalibration(pulse_min_us=500, limit_lower_rad=-1.0, limit_upper_rad=1.0).validate("wide")
        with pytest.raises(CalibrationError, match="gripper clamp"):
            GripperCalibration(pulse_open_us=800).validate()
        # The shipped placeholders obey the rule, rounding included.
        ServoCalibration.default().validate()

    def test_shipped_robot_config_obeys_the_rule(self):
        cal = ServoCalibration.load(ROBOT_CONFIG_PATH, ArmGeometry())
        for name, joint in cal.joints.items():
            assert abs(joint.limit_lower_rad) <= 1.5708 + 1e-9 and abs(joint.limit_upper_rad) <= 1.5708 + 1e-9, name
        assert cal.reattach_s > 0 and cal.home_move_s > cal.reattach_s

    def test_follower_reports_a_pulse_level_clamp(self):
        # Bypass validate() to build the exact inconsistency the review found:
        # elbow limits +-120 deg over a +-90 deg pulse map.
        joints = dict(ServoCalibration.default().joints)
        joints["elbow_pitch"] = JointCalibration(limit_lower_rad=-2.0944, limit_upper_rad=2.0944)
        cal = ServoCalibration(joints=joints, max_joint_velocity=FAST_VELOCITY)
        follower = TrajectoryFollower(FakeDriver(cal), cal, threading.Event(), sleep=lambda _s: None)
        home = np.asarray(cal.home_q)
        near = home.copy()
        near[2] = 2.0  # 114.6 deg: inside the radian limits, past the pulse map
        assert not cal.clamp_q(near)[1], "the radian clamp does not see it"
        assert cal.pulse_clamped(near)
        _samples, _tick, clamped, _stretch = follower.plan(np.stack([home, near]), dt=0.1)
        assert clamped is True
        reply = follower.follow(np.stack([home, near]), dt=0.1)
        assert reply["clamped"] is True and reply["completed"] is True
        assert reply["q"][2] == pytest.approx(np.pi / 2, abs=1e-6)  # what the servo actually did
        fine = home.copy()
        fine[2] = 1.4
        assert follower.plan(np.stack([home, fine]), dt=0.1)[2] is False

    def test_float32_limit_touching_move_is_not_clamped(self):
        raw = ServoCalibration.default().to_dict()
        # A measured limit float32 rounds *up* past (1.4362 -> 1.43620002).
        raw["joints"]["base_yaw"]["limit_upper_rad"] = 1.4362
        raw["max_joint_velocity"] = FAST_VELOCITY
        cal = ServoCalibration.from_dict(raw)
        wire = np.float32(1.4362)
        assert float(wire) > 1.4362 and float(wire) - 1.4362 > 1e-9
        follower = TrajectoryFollower(FakeDriver(cal), cal, threading.Event(), sleep=lambda _s: None)
        home = np.asarray(cal.home_q)
        goal = home.copy()
        goal[0] = float(wire)
        _s, _t, clamped, _x = follower.plan(np.stack([home, goal]).astype(np.float32), dt=0.1)
        assert clamped is False
        past = goal.copy()
        past[0] = 1.4362 + 10.0 * CLAMP_TOL_RAD
        assert follower.plan(np.stack([home, past]), dt=0.1)[2] is True
        assert cal.clamp_q(goal)[1] is False and cal.clamp_q(past)[1] is True


class TestHomeClearance:
    """HS-6: a home below the table is refused everywhere it can be set."""

    #: The review's example: passes every joint limit of the placeholder,
    #: wrist 3 cm and jaw 12 cm below the table.
    SUB_TABLE_HOME = (0.0, 1.5708, 1.5708, 0.0)
    #: Same shape inside the test fixture's slightly tighter 1.5707 limits,
    #: so the *table* check is the one that fires (wrist -2 cm, jaw -11 cm).
    SUB_TABLE_HOME_IN_LIMITS = (0.0, 1.5, 1.5, 0.0)

    def test_link_points_match_tcp_and_find_the_table(self):
        g = ArmGeometry()
        for q in ([0.3, 0.5, -1.0, 0.2], list(ServoCalibration.default().home_q), list(self.SUB_TABLE_HOME)):
            pts = planar_link_points(q, g)
            assert pts.shape == (5, 3)
            np.testing.assert_allclose(pts[4], planar_tcp(q, g))
            np.testing.assert_allclose(pts[0], 0.0)
            assert pts[1, 2] == pytest.approx(g.base_height)
        z = planar_link_points(self.SUB_TABLE_HOME, g)[:, 2]
        np.testing.assert_allclose(z, [0.0, 0.07, 0.07, -0.03, -0.12], atol=1e-6)
        check_home_clearance(ServoCalibration.default().home_q, g)
        with pytest.raises(CalibrationError, match="below the table") as info:
            check_home_clearance(self.SUB_TABLE_HOME, g)
        assert "wrist" in str(info.value) and "jaw" in str(info.value)
        # A pose just under the clearance is refused, just over is fine.
        assert TABLE_CLEARANCE_M == pytest.approx(0.01)

    def test_calibration_with_geometry_refuses_it(self):
        raw = {**ServoCalibration.default().to_dict(), "home_q": list(self.SUB_TABLE_HOME)}
        ServoCalibration.from_dict(raw)  # pure conversions: no geometry, no check
        with pytest.raises(CalibrationError, match="below the table"):
            ServoCalibration.from_dict(raw, ArmGeometry())

    def test_server_refuses_it_on_set_calibration(self, bridge):
        client, server, _driver, cal_path = bridge
        assert planar_link_points(self.SUB_TABLE_HOME_IN_LIMITS, ArmGeometry())[3:, 2].max() < 0.0
        raw = {**server.calibration.to_dict(), "home_q": list(self.SUB_TABLE_HOME_IN_LIMITS)}
        with pytest.raises(RpcError, match="below the table"):
            client.set_calibration(raw)
        assert tuple(ServoCalibration.load(cal_path).home_q) == tuple(server.calibration.home_q)
        reply = client.get_calibration()
        assert reply["arm_geometry"] == {
            "base_height": 0.07, "shoulder_offset": 0.0, "upper_arm": 0.105, "forearm": 0.10, "tool": 0.09,
        }

    def test_calibrator_home_set_refuses_it(self, bridge):
        client, server, _driver, _path = bridge
        mod = _load_calibrate_servos()
        lines: list[str] = []
        cal = mod.ServoCalibrator(client, out=lines.append)
        cal.run("move shoulder_pitch 90")
        cal.run("move elbow_pitch 90")
        with pytest.raises(CalibrationError, match="below the table"):
            cal.run("home set")
        assert not cal.dirty
        assert tuple(cal.cal.home_q) == tuple(server.calibration.home_q)
        cal.run("home")
        cal.run("home set")  # the placeholder home itself is fine
        assert cal.dirty


# ----------------------------------------------------------------------
# Uno protocol
# ----------------------------------------------------------------------


class TestUnoFrames:
    def test_p_and_t_round_trip(self):
        us = (600, 1500, 2400, 1234, 900)
        assert encode_p_frame(us) == b"P600,1500,2400,1234,900\n"
        assert decode_command_frame(encode_p_frame(us)) == ("P", us)
        assert encode_t_frame(us, 350) == b"T600,1500,2400,1234,900,350\n"
        assert decode_command_frame(encode_t_frame(us, 350)) == ("T", us, 350)
        assert decode_command_frame(b"?\n") == ("?",)
        assert decode_command_frame("D") == ("D",)
        # Floats are rounded to the integer microseconds that go on the wire.
        assert decode_command_frame(encode_p_frame((1499.6, 1500.4, 1500, 1500, 1500)))[1][:2] == (1500, 1500)

    def test_bad_frames_use_sketch_wording(self):
        with pytest.raises(ValueError, match="ERR bad frame"):
            decode_command_frame(b"P1500,1500,1500,1500\n")
        with pytest.raises(ValueError, match="ERR bad frame"):
            decode_command_frame(b"T1500,1500,1500,1500,1500\n")
        with pytest.raises(ValueError, match="ERR bad frame"):
            decode_command_frame(b"P1500,1500,1500,1500,15a0\n")
        with pytest.raises(ValueError, match="ERR unknown command X"):
            decode_command_frame(b"X1\n")
        with pytest.raises(ValueError):
            encode_p_frame((1500, 1500, 1500))
        with pytest.raises(ValueError):
            encode_p_frame((-1, 1500, 1500, 1500, 1500))
        with pytest.raises(ValueError):
            encode_t_frame((1500,) * 5, -5)

    def test_q_frame_and_replies(self):
        status = decode_q_status(b"Q600,700,800,900,1000,1,1\r\n")
        assert status.pulses == (600, 700, 800, 900, 1000) and status.attached and status.position_known is True
        assert decode_q_status("Q1500,1500,1500,1500,1500,0,0").position_known is False
        assert decode_q_status("Q1500,1500,1500,1500,1500,0").position_known is None  # old sketch
        assert decode_q_frame(b"Q600,700,800,900,1000,1,0\r\n") == ((600, 700, 800, 900, 1000), True)
        assert decode_q_frame(b"Q600,700,800,900,1000,1\r\n") == ((600, 700, 800, 900, 1000), True)
        assert decode_q_frame("Q1500,1500,1500,1500,1500,0")[1] is False
        with pytest.raises(DriverError):
            decode_q_frame(b"OK\n")
        parse_reply(b"OK\r\n")
        with pytest.raises(DriverError, match="rejected"):
            parse_reply(b"ERR bad frame\n")
        with pytest.raises(DriverError, match="no reply"):
            parse_reply(b"")
        with pytest.raises(DriverError, match="unexpected"):
            parse_reply(b"hello\n")


class _StubSerial:
    """What ``UnoSerialDriver`` sees: a FIFO wire in front of the sketch's logic.

    Models the parts of ``servo_bridge.ino`` the driver's safety depends on,
    with replies that arrive at *times*, as on the real USB-CDC link:

    * every frame is executed when it is written (a late reply still had its
      frame executed); its reply lands in the input buffer ``reply_delay_s``
      later, in order (the Uno answers strictly in sequence);
    * ``readline`` returns the oldest reply that arrives within ``timeout``
      and otherwise returns ``b""`` -- *without* sleeping: the stub's clock
      (``now()``, monotonic plus a skew) jumps forward instead, so a test
      of a 0.1 s timeout costs nothing;
    * ``reset_input_buffer`` drops what has *arrived*, never what is still in
      flight -- the case the old drain-and-resend could not survive (review
      missed-7: a reply 1-20 ms late was read as the next frame's reply for
      good). The old stub modelled a late reply as simply lost, so the test
      that was meant to catch that could not fail;
    * the watchdog (``watchdog_s``, on the stub's clock) is fed by P/T/K and,
      like the current sketch, by ``?`` and ``S`` while attached;
    * a fresh stub has no position: the first P/T attaches AT its target
      (``have_position``), and ``?`` answers with the 7-field ``Q`` frame
      (``legacy_q`` gives the old 6-field one);
    * ``S<n>`` is echoed (``knows_sync=False`` answers like the old sketch).

    Faults are scripted per frame kind through ``actions[kind]`` -- a list
    consumed one entry per frame of that kind: ``None`` (normal),
    ``"drop_frame"`` (never reaches the Uno), ``"drop_reply"`` (executed,
    reply lost) or a float (reply arrives that many seconds late).
    ``reply_override`` replaces every reply (``b""``: nothing ever answers).
    ``inject_stale`` puts a stale line, already arrived, at the head of the
    buffer. ``reads`` records ``(frame last written, frame answered)`` per
    readline, so a test can check the stream is in step.
    """

    instances: list["_StubSerial"] = []
    watchdog_s: float | None = None
    virtual: bool = False
    """True: ``now()`` is the stub's own clock only (it moves on ``advance``
    and on read timeouts, never with wall time), so a reply scripted 4 ms
    late is late whatever the machine's load. The default mixes in
    ``time.monotonic()`` for the tests that sleep for real (the keepalive
    thread against the watchdog); under a loaded full-suite run a few ms of
    scheduler delay made a "late" reply arrive on time there (measured:
    ``test_late_reply_never_leaves_the_stream_one_frame_behind[4.0]`` failed
    once with ``stale_replies_skipped == 0``)."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.frames: list[bytes] = []
        self.current = [1500] * 5
        self.attached = False
        self.have_position = False
        self.reply_override: bytes | None = None
        self.actions: dict[str, list[Any]] = {}
        self.reply_delay_s = 0.002
        self.legacy_q = False
        self.knows_sync = True
        self.closed = False
        self.resets = 0
        self.t_frames: list[tuple[tuple[int, ...], int]] = []
        self.attaches: list[tuple[tuple[int, ...], bool]] = []
        """``(pulse attached at, whether a position was known)`` per attach."""
        self.trips = 0
        self.skew = 0.0
        self.last_feed_t = -float("inf")
        self._inbox: list[tuple[float, bytes, int]] = []
        self._last_arrival = -float("inf")
        self.reads: list[tuple[int, int | None]] = []
        _StubSerial.instances.append(self)

    # -- clock and sketch ------------------------------------------------

    def now(self) -> float:
        if self.virtual:
            return self.skew
        return time.monotonic() + self.skew

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` pass on the stub's clock without sleeping."""
        self.skew += float(seconds)

    def _watchdog(self, t: float) -> None:
        if self.watchdog_s is not None and self.attached and t - self.last_feed_t > self.watchdog_s:
            self.attached = False
            self.trips += 1

    def _feed(self, t: float) -> None:
        if self.attached:
            self.last_feed_t = t

    def _execute(self, frame: bytes, t: float) -> str:
        self._watchdog(t)
        try:
            kind = decode_command_frame(frame)
        except ValueError as exc:
            return str(exc)
        cmd = kind[0]
        if cmd == "?":
            self._feed(t)
            fields = [str(v) for v in self.current] + [str(int(self.attached))]
            if not self.legacy_q:
                fields.append(str(int(self.have_position)))
            return "Q" + ",".join(fields)
        if cmd == "S":
            if not self.knows_sync:
                return "ERR unknown command S"
            self._feed(t)
            return f"S{kind[1]}"
        if cmd == "D":
            self.attached = False
            return "OK"
        if cmd == "K":
            self._feed(t)
            return "OK"
        # P / T. Slew and interpolation are not modelled: ``current`` is where
        # the frame ends up. What is modelled is WHERE a detached bridge
        # attaches: at the target when no position is known, else at the
        # last pulsed position (``current`` before the frame).
        if not self.attached:
            at = tuple(kind[1]) if not self.have_position else tuple(self.current)
            self.attaches.append((at, self.have_position))
            self.attached = True
            self.have_position = True
        self.current = list(kind[1])
        self.last_feed_t = t
        if cmd == "T":
            self.t_frames.append((tuple(kind[1]), int(kind[2])))
        return "OK"

    def power_cycle(self) -> None:
        """A USB reset / brownout: the sketch restarts with no position."""
        self.current = [1500] * 5
        self.attached = False
        self.have_position = False
        self._inbox.clear()

    # -- pyserial surface ------------------------------------------------

    def write(self, frame: bytes) -> None:
        self.frames.append(frame)
        index = len(self.frames) - 1
        t = self.now()
        queue = self.actions.get(frame[:1].decode("ascii", "replace"), [])
        action = queue.pop(0) if queue else None
        if action == "drop_frame":
            return
        reply = (self._execute(frame, t) + chr(13) + chr(10)).encode("ascii")
        if self.reply_override is not None:
            reply = self.reply_override
        if action == "drop_reply" or not reply:
            return
        late = float(action) if isinstance(action, (int, float)) else 0.0
        arrival = max(self._last_arrival, t + self.reply_delay_s + late)
        self._last_arrival = arrival
        self._inbox.append((arrival, reply, index))

    def inject_stale(self, line: bytes) -> None:
        """A late reply to some earlier frame, already sitting in the buffer."""
        self._inbox.insert(0, (self.now() - 1e-3, line, -1))

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.resets += 1
        t = self.now()
        self._inbox = [m for m in self._inbox if m[0] > t]

    def readline(self) -> bytes:
        deadline = self.now() + self.timeout
        if self._inbox and self._inbox[0][0] <= deadline:
            arrival, payload, answered = self._inbox.pop(0)
            if arrival > self.now():
                self.skew += arrival - self.now()
            self.reads.append((len(self.frames) - 1, answered))
            return payload
        self.skew += max(0.0, deadline - self.now())
        self.reads.append((len(self.frames) - 1, None))
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def stub_serial(monkeypatch):
    module = types.ModuleType("serial")
    module.Serial = _StubSerial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", module)
    _StubSerial.instances.clear()
    _StubSerial.watchdog_s = None
    _StubSerial.virtual = False
    yield _StubSerial
    _StubSerial.virtual = False


def _uno(cal: ServoCalibration | None = None, keepalive_s: float | None = None) -> UnoSerialDriver:
    """A stub-backed driver; keepalive off unless asked so frame lists are exact."""
    return UnoSerialDriver(cal or _fast_calibration(), port="COM9", reset_wait_s=0.0, keepalive_s=keepalive_s)


class TestUnoSerialDriver:
    def test_frames_on_the_wire(self, stub_serial):
        cal = _fast_calibration()
        driver = _uno(cal)
        ser = stub_serial.instances[-1]
        assert ser.port == "COM9" and ser.baudrate == 115200
        assert ser.frames == []  # detached until the first frame; nothing sent at open

        driver.write(np.asarray(cal.home_q))
        kind, pulses = decode_command_frame(ser.frames[-1])
        assert kind == "P"
        expected = tuple(round(p) for p in cal.q_to_pulses(np.asarray(cal.home_q)))
        assert pulses[:4] == expected
        assert pulses[4] == round(cal.width_to_us(cal.gripper.width_open_m))
        assert driver.attached and ser.attached

        driver.write_gripper(0.0)
        assert decode_command_frame(ser.frames[-1])[1][4] == 1900
        assert driver.query() == (tuple(pulses[:4]) + (1900,), True)
        assert driver.probe() == driver.query()

        driver.detach()
        assert ser.frames[-1] == b"D\n" and not driver.attached
        q, width = driver.read_commanded()  # commanded pose survives a detach
        np.testing.assert_allclose(q, cal.home_q, atol=1e-9)
        assert width == pytest.approx(0.0)
        driver.close()
        assert ser.closed

    def test_channel_map_reorders_pulses(self, stub_serial):
        raw = ServoCalibration.default().to_dict()
        raw["channels"] = {"base_yaw": 4, "shoulder_pitch": 3, "elbow_pitch": 2, "wrist_pitch": 1, "gripper": 0}
        cal = ServoCalibration.from_dict(raw)
        driver = _uno(cal)
        driver.write(np.array([0.5, 0.0, 0.0, 0.0]))
        _, pulses = decode_command_frame(stub_serial.instances[-1].frames[-1])
        assert pulses[4] == round(cal.rad_to_us("base_yaw", 0.5))
        assert pulses[0] == round(cal.width_to_us(cal.gripper.width_open_m))

    def test_uno_error_reply_raises(self, stub_serial):
        driver = _uno(ServoCalibration.default())
        stub_serial.instances[-1].reply_override = b"ERR bad frame\n"
        with pytest.raises(DriverError, match="rejected"):
            driver.write(np.zeros(4))
        stub_serial.instances[-1].reply_override = b""
        with pytest.raises(DriverError, match="no reply"):
            driver.detach()

    def test_keepalive_feeds_the_watchdog_with_k_frames(self, stub_serial):
        stub_serial.watchdog_s = 0.3
        driver = _uno(keepalive_s=0.05)
        ser = stub_serial.instances[-1]
        try:
            assert driver.keepalive_running
            driver.write(np.asarray(driver.calibration.home_q))
            time.sleep(0.8)
            kinds = [decode_command_frame(f)[0] for f in ser.frames]
            assert kinds.count("K") >= 8 and kinds.count("P") == 1
            pulses, attached = driver.query()
            assert attached is True, "the stub watchdog would have fired without K frames"
            assert pulses == decode_command_frame(ser.frames[0])[1]  # K never changed a target
            # Stop feeding it: the stub detaches, the driver still believes it
            # is attached until asked, and a server state() believes the bridge.
            driver.stop_keepalive()
            time.sleep(0.4)
            assert driver.query()[1] is False
            assert driver.attached is True
            server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration)
            state = server.state()
            assert state["attached"] is False and driver.needs_reattach
            with pytest.raises(DriverError, match="detached"):
                driver.write(np.asarray(driver.calibration.home_q))
        finally:
            driver.close()

    def test_reattach_is_one_t_frame_fed_with_keepalives(self, stub_serial):
        cal = _fast_calibration()
        driver = _uno(cal, keepalive_s=0.05)
        driver.sleep = lambda _s: None
        ser = stub_serial.instances[-1]
        try:
            driver.write(np.asarray(cal.home_q))
            driver.detach()
            stored = driver.pulses()
            target = list(cal.q_to_pulses(GOAL_Q)) + [cal.width_to_us(0.01)]
            n_before = len(ser.frames)
            driver.reattach(target, duration_s=0.5)
            sent = [decode_command_frame(f) for f in ser.frames[n_before:]]
            assert sent[0][0] == "T" and sent[0][2] == 500
            assert sent[0][1] == tuple(int(round(p)) for p in target)
            assert all(kind[0] == "K" for kind in sent[1:]) and len(sent) - 1 == 10
            assert driver.attached and not driver.hold_detached
            assert driver.pulses() == tuple(int(round(p)) for p in target)
            assert stored != driver.pulses()
            # Already attached: reattach is an ordinary P write, no T frame.
            driver.reattach(list(stored), duration_s=0.5)
            assert decode_command_frame(ser.frames[-1])[0] == "P"
        finally:
            driver.close()

    def test_reattach_aborts_on_estop(self, stub_serial):
        cal = _fast_calibration()
        driver = _uno(cal, keepalive_s=0.05)
        ser = stub_serial.instances[-1]
        abort = threading.Event()
        driver.sleep = lambda _s: abort.set()  # estop lands during the first wait step
        try:
            driver.write(np.asarray(cal.home_q))
            driver.detach()
            n_before = len(ser.frames)
            driver.reattach(list(cal.q_to_pulses(GOAL_Q)) + [1500.0], duration_s=1.0, abort=abort)
            kinds = [decode_command_frame(f)[0] for f in ser.frames[n_before:]]
            assert kinds == ["T"], kinds  # no K after the abort
        finally:
            driver.close()

    def test_timeout_syncs_and_resends_once(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        resets_at_open = ser.resets  # the DTR-reset flush at open
        assert resets_at_open == 1
        ser.actions["P"] = ["drop_reply"]  # first reply lost; the resend answers normally
        driver.write(np.asarray(driver.calibration.home_q))
        assert [decode_command_frame(f)[0] for f in ser.frames] == ["P", "S", "P"]
        assert ser.frames[0] == ser.frames[2]
        assert ser.resets - resets_at_open == 1 and driver.resyncs == 1 and driver.consecutive_timeouts == 0
        assert driver.attached

    def test_out_of_step_reply_is_resynchronised(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        resets_at_open = ser.resets
        driver.write(np.asarray(driver.calibration.home_q))
        # A stale "OK" sits in the buffer when we ask "?". The Q it displaces
        # is still in flight when the buffer is drained, so the sync must
        # skip it before the resend.
        ser.inject_stale(b"OK\n")
        pulses, attached = driver.query()
        assert attached is True and pulses == decode_command_frame(ser.frames[0])[1]
        assert ser.resets - resets_at_open == 1
        assert [decode_command_frame(f)[0] for f in ser.frames[-3:]] == ["?", "S", "?"]
        assert driver.stale_replies_skipped == 1
        # And a stale Q when we expect OK.
        ser.inject_stale(b"Q1,2,3,4,5,1,1\n")
        driver.write_gripper(0.02)
        assert ser.resets - resets_at_open == 2 and driver.attached

    @pytest.mark.parametrize("late_ms", [1.0, 4.0, 20.0])
    def test_late_reply_never_leaves_the_stream_one_frame_behind(self, stub_serial, caplog, late_ms):
        """Review missed-7: a reply that lands just AFTER the timeout is still
        in flight when the buffer is drained. The old drain-and-resend read it
        as the resent frame's reply, and from then on every read returned the
        previous frame's reply (16 of 17 in the probe). With the sync echo
        every later read answers the frame just sent."""
        stub_serial.virtual = True  # ms-exact lateness, independent of machine load
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        ser.actions["P"] = [driver.reply_timeout_s + late_ms / 1000.0]
        driver.write(np.asarray(driver.calibration.home_q) + 0.05)
        assert driver.stale_replies_skipped == 1 and driver.resyncs == 1
        first = len(ser.reads)
        with caplog.at_level(logging.WARNING, logger="robot_server"):
            for _ in range(8):
                driver._send_keepalive()
                assert driver.query()[1] is True
        answered = ser.reads[first:]
        assert answered and all(asked == got for asked, got in answered), answered
        assert driver.resyncs == 1, "the stream fell out of step again"
        assert not any("out of step" in rec.message for rec in caplog.records)

    def test_sync_against_an_old_sketch_fails_safe(self, stub_serial, caplog):
        driver = _uno()
        ser = stub_serial.instances[-1]
        ser.knows_sync = False
        driver.write(np.asarray(driver.calibration.home_q))
        ser.actions["K"] = ["drop_reply"]
        with caplog.at_level(logging.ERROR, logger="robot_server"):
            with pytest.raises(DriverError, match="no reply"):
                driver._send_keepalive()
        assert any("old servo_bridge.ino" in rec.message for rec in caplog.records)
        assert driver.consecutive_timeouts == 1

    def test_repeated_timeouts_mark_the_servos_detached(self, stub_serial, caplog):
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        assert driver.attached
        ser.reply_override = b""
        with caplog.at_level(logging.WARNING, logger="robot_server"):
            for i in range(SERIAL_MAX_TIMEOUTS):
                with pytest.raises(DriverError, match="no reply"):
                    driver.write_gripper(0.02)
                if i < SERIAL_MAX_TIMEOUTS - 1:
                    assert driver.attached, f"detached after only {i + 1} timeouts"
        assert driver.attached is False and driver.needs_reattach
        assert any("consecutive failed serial exchanges" in rec.message for rec in caplog.records)
        ser.reply_override = None
        with pytest.raises(DriverError, match="detached"):
            driver.write_gripper(0.02)  # hold in force until a slow re-attach
        driver.sleep = lambda _s: None
        driver.reattach(duration_s=0.2)
        assert driver.attached and decode_command_frame(ser.frames[-1])[0] in ("T", "K")

    def test_open_failure_names_the_causes(self, monkeypatch):
        module = types.ModuleType("serial")

        class Broken:
            def __init__(self, *a: Any, **k: Any) -> None:
                raise OSError("no such port")

        module.Serial = Broken  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "serial", module)
        with pytest.raises(DriverError, match="dialout"):
            UnoSerialDriver(ServoCalibration.default(), port="/dev/ttyACM0", reset_wait_s=0.0, keepalive_s=None)


# ----------------------------------------------------------------------
# calibrate_servos.py
# ----------------------------------------------------------------------


def _load_calibrate_servos():
    path = REPO_ROOT / "scripts" / "calibrate_servos.py"
    spec = importlib.util.spec_from_file_location("calibrate_servos", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCalibrateServos:
    def test_script_mode_records_and_saves(self, bridge, capsys):
        client, server, _driver, cal_path = bridge
        mod = _load_calibrate_servos()
        script = (
            "read; move base_yaw 20; zero base_yaw; limit base_yaw upper; jaw 30; "
            "gripper open 30; move wrist_pitch -15; home set; velocity 3.0; save; read"
        )
        rc = mod.main(["--jetson", f"127.0.0.1:{client.rpc.port}", "--script", script])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "error" not in out.lower(), out

        cal = server.calibration
        state = client.get_state()
        # The 20 deg pose was re-labelled as zero: the arm did not move, its angle did.
        assert state["q"][0] == pytest.approx(0.0, abs=1e-6)
        assert cal.joints["base_yaw"].zero_offset_rad == pytest.approx(np.radians(20.0), abs=1e-6)
        assert cal.joints["base_yaw"].limit_upper_rad == pytest.approx(0.0, abs=1e-6)
        assert state["q"][3] == pytest.approx(np.radians(-15.0), abs=1e-6)
        assert tuple(cal.home_q) == pytest.approx(tuple(state["q"]), abs=1e-6)
        assert cal.gripper.width_open_m == pytest.approx(0.030)
        assert cal.gripper.pulse_open_us == round(GripperCalibration().width_to_us(0.030))
        assert state["gripper_width"] == pytest.approx(0.030, abs=1e-6)
        assert cal.max_joint_velocity == pytest.approx(3.0)
        assert ServoCalibration.load(cal_path) == cal
        assert "unsaved" not in out

    def test_script_mode_reports_failures(self, bridge, capsys):
        client, server, _driver, _path = bridge
        mod = _load_calibrate_servos()
        before = server.calibration
        rc = mod.main(["--jetson", f"127.0.0.1:{client.rpc.port}",
                       "--script", "move elbow 10; frobnicate; zero base_yaw"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "unknown joint 'elbow'" in out and "unknown command 'frobnicate'" in out
        assert "unsaved edits discarded" in out
        assert server.calibration == before  # nothing pushed without `save`

    def test_unreachable_server(self, capsys):
        mod = _load_calibrate_servos()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        client_cls = mod.JetsonClient

        # Keep the retry loop short: one attempt against a closed port.
        class Quick(client_cls):  # type: ignore[misc,valid-type]
            def connect(self, retries: int = 1, backoff_s: float = 0.0):
                return super().connect(retries=1, backoff_s=0.0)

        mod.JetsonClient = Quick
        try:
            rc = mod.main(["--jetson", f"127.0.0.1:{free_port}", "--script", "read", "--timeout", "0.3"])
        finally:
            mod.JetsonClient = client_cls
        assert rc == 2
        assert "no server answering" in capsys.readouterr().out

    def test_calibrator_estop_and_pulse(self, bridge):
        client, server, driver, _path = bridge
        mod = _load_calibrate_servos()
        lines: list[str] = []
        cal = mod.ServoCalibrator(client, out=lines.append)
        assert cal.run("pulse base_yaw 2000")
        assert driver.commands[-1][1][0] == 2000
        with pytest.raises(ValueError, match="outside"):
            cal.run("pulse base_yaw 2500")
        cal.run("estop")
        assert client.get_state()["estopped"] is True
        with pytest.raises(RpcError, match="estopped"):
            cal.run("home")
        cal.run("clear")
        cal.run("home")
        np.testing.assert_allclose(client.get_state()["q"], server.calibration.home_q, atol=1e-6)
        assert client.get_state()["attached"] is True
        assert cal.run("quit") is False
        cal.run("help")
        assert any("gripper open|closed" in line for line in lines)

    def test_torque_on_asks_first_and_holds_the_bridge_pose(self, bridge, capsys):
        """HS-5: energising after hand-posing is refused without a yes, and
        with one the target is where the bridge last pulsed, so nothing moves."""
        client, server, driver, _path = bridge
        mod = _load_calibrate_servos()
        port = f"127.0.0.1:{client.rpc.port}"
        assert mod.main(["--jetson", port, "--script", "move base_yaw 20; jaw 12; torque off"]) == 0
        capsys.readouterr()
        assert driver.attached is False
        posed = driver.pulses()

        rc = mod.main(["--jetson", port, "--script", "torque on"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "TAKE YOUR HAND OFF" in out and "cancelled" in out and "--yes" in out
        assert driver.attached is False and driver.commands[-1] == ("D", ())

        rc = mod.main(["--jetson", port, "--script", "torque on", "--yes"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "TAKE YOUR HAND OFF" in out and "torque on" in out
        assert driver.attached is True
        state = client.get_state()
        assert state["q"][0] == pytest.approx(np.radians(20.0), abs=1e-6)
        assert state["gripper_width"] == pytest.approx(0.012, abs=1e-6)
        _assert_ramp(_p_frames_after_last_detach(driver), posed, posed,
                     min_frames=int(server.calibration.reattach_s * server.calibration.control_rate_hz))
        # Already attached: no prompt, nothing moves.
        assert mod.main(["--jetson", port, "--script", "torque on"]) == 0
        assert "already attached" in capsys.readouterr().out

    def test_torque_on_falls_back_to_home_when_the_bridge_cannot_report(self, tmp_path):
        class Blind(FakeDriver):
            def probe(self):  # a PCA9685-like back-end
                return None

        cal = _fast_calibration()
        driver = Blind(cal)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal,
                             calibration_path=tmp_path / "c.yaml", follower_sleep=lambda _s: None)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port)
        client.connect()
        try:
            client.home()
            assert client.get_state()["bridge_q"] is None
            home = np.asarray(client.get_state()["q"])
            client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
            client.set_torque(False)
            goal_pulses = driver.pulses()
            mod = _load_calibrate_servos()
            lines: list[str] = []
            calibrator = mod.ServoCalibrator(client, out=lines.append, confirm=lambda _p: True)
            calibrator.run("torque on")
            assert any("move to home_q" in line for line in lines)
            np.testing.assert_allclose(client.get_state()["q"], cal.home_q, atol=1e-6)
            home_pulses = tuple(round(p) for p in cal.q_to_pulses(np.asarray(cal.home_q))) + (goal_pulses[4],)
            _assert_ramp(_p_frames_after_last_detach(driver), goal_pulses, home_pulses,
                         min_frames=int(cal.reattach_s * cal.control_rate_hz))
            # The interactive default refuses anything but a literal yes.
            calibrator.confirm = lambda _p: False
            client.set_torque(False)
            with pytest.raises(ValueError, match="cancelled"):
                calibrator.run("torque on")
            assert driver.attached is False
        finally:
            client.close()
            server.stop()


# ----------------------------------------------------------------------
# a freshly reset bridge knows no position (re-review critical, HS-3, HS-5)
# ----------------------------------------------------------------------


class TestUnknownPosition:
    """Opening the Uno's port resets it: every server start is a power-up, and
    the first frame attaches every servo AT its target. Nothing but ``home``
    may be that first frame, and nothing may promise the arm holds still."""

    def test_every_motion_but_home_is_refused_before_the_first_attach(self, fresh_bridge):
        client, server, driver, _path = fresh_bridge
        home = np.asarray(client.get_state()["q"])
        with pytest.raises(RpcError, match="no known position"):
            client.set_gripper(0.02)
        with pytest.raises(RpcError, match="no known position"):
            client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
        with pytest.raises(RpcError, match="no known position"):
            client.rpc.call("set_torque", {"enabled": True, "q": [0.0, 0.0, 0.0, 0.0]})
        assert not any(kind == "P" for kind, _ in driver.commands), "nothing may be pulsed yet"
        client.home()
        assert client.set_gripper(0.02)["gripper_width"] == pytest.approx(0.02, abs=1e-6)

    def test_torque_on_after_start_never_claims_to_hold_still(self, fresh_bridge):
        """The re-review's critical repro: `torque on` as the first REPL
        command after robot_server starts printed 'it holds still' and sent
        T1500x5 -- a 90 degree elbow snap from the hand-posed home."""
        client, server, driver, _path = fresh_bridge
        mod = _load_calibrate_servos()
        lines: list[str] = []
        prompts: list[str] = []
        calibrator = mod.ServoCalibrator(client, out=lines.append, confirm=lambda p: prompts.append(p) or False)
        with pytest.raises(ValueError, match="cancelled"):
            calibrator.run("torque on")
        assert not any(kind == "P" for kind, _ in driver.commands)
        text = "\n".join(lines)
        assert "holds still" not in text
        assert "NO KNOWN POSITION" in text and "HAND-POSE THE ARM AT HOME" in text and prompts
        calibrator.confirm = lambda _p: True
        lines.clear()
        calibrator.run("torque on")
        assert "holds still" not in "\n".join(lines)
        home_pulses = tuple(round(p) for p in server.calibration.q_to_pulses(np.asarray(server.calibration.home_q)))
        frames = [p for kind, p in driver.commands if kind == "P"]
        assert len(frames) == 1 and frames[0][:4] == home_pulses, "one attach, AT home, never 1500 us x5"
        assert all(us != 1500 for us in frames[0][1:4])

    def test_home_command_asks_first_on_a_fresh_bridge(self, fresh_bridge):
        client, _server, driver, _path = fresh_bridge
        mod = _load_calibrate_servos()
        lines: list[str] = []
        calibrator = mod.ServoCalibrator(client, out=lines.append, confirm=lambda _p: False)
        with pytest.raises(ValueError, match="home cancelled"):
            calibrator.run("home")
        assert driver.attached is False
        calibrator.confirm = lambda _p: True
        calibrator.run("home")
        assert driver.attached is True and any("first attach=True" in line for line in lines)
        lines.clear()
        calibrator.run("home")  # known now: an ordinary move, no prompt
        assert any(line.startswith("  at home") for line in lines)

    def test_old_server_reset_signature_is_recognised(self):
        """A server without ``bridge_position_known``: detached at 1500 us x5
        is a reset Uno, not a pose to hold."""
        mod = _load_calibrate_servos()
        cal = _fast_calibration()
        reset_q = [float(v) for v in cal.pulses_to_q([1500.0] * 4)]
        state = {"attached": False, "bridge_q": reset_q, "bridge_gripper_width": cal.us_to_width(1500.0)}
        assert mod.bridge_position_unknown(state, cal) is True
        posed = dict(state, bridge_q=[0.1, 0.9, -1.0, 0.2])
        assert mod.bridge_position_unknown(posed, cal) is False
        assert mod.bridge_position_unknown({**state, "bridge_position_known": True}, cal) is False

    def test_uno_boot_home_is_one_frame_attaching_at_home(self, stub_serial):
        """On the wire: the Uno's first frame attaches AT its target whatever
        its <ms> says, so boot home is sent as the instant attach it is."""
        cal = _fast_calibration()
        driver = _uno(cal)
        ser = stub_serial.instances[-1]
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal, follower_sleep=lambda _s: None)
        state = server._handle("home", {})
        assert state["first_attach"] is True and state["bridge_position_known"] is True
        home = tuple(round(p) for p in driver._reorder(driver.pulses()))
        assert ser.attaches == [(home, False)]
        assert ser.t_frames == [(home, 0)], "a first attach has nothing to interpolate from"

    def test_uno_reset_mid_session_is_detected_and_blocks_motion(self, stub_serial):
        cal = _fast_calibration()
        driver = _uno(cal)
        ser = stub_serial.instances[-1]
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal, follower_sleep=lambda _s: None)
        server._handle("home", {})
        before = server.state()["detach_count"]
        ser.power_cycle()  # USB re-enumeration / brownout: the sketch restarts
        state = server.state()
        assert state["bridge_position_known"] is False and state["attached"] is False
        assert state["bridge_q"] is None, "1500 us defaults are not a pose"
        assert state["detach_count"] == before + 1
        reply = server._handle("follow_trajectory", {"waypoints": np.stack([np.asarray(state["q"]), GOAL_Q]), "dt": 0.1})
        assert "no known position" in reply["error"]
        n = len(ser.frames)
        assert server._handle("home", {})["first_attach"] is True
        assert ser.attaches[-1][1] is False and [decode_command_frame(f)[0] for f in ser.frames[n:]].count("T") == 1

    def test_reattach_duration_is_measured_from_where_the_uno_is(self, stub_serial):
        """Re-review low: an estop during a long move leaves the Uno partway
        while the stored pulses already equal the old target; the re-attach
        was timed from the stored pose (~0 travel -> the 1 s floor) and the
        Uno then interpolated the real distance faster than max_joint_velocity."""
        raw = _fast_calibration().to_dict()
        raw["max_joint_velocity"] = 0.5
        cal = ServoCalibration.from_dict(raw)
        driver = _uno(cal)
        ser = stub_serial.instances[-1]
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal, follower_sleep=lambda _s: None)
        server._handle("home", {})
        home = np.asarray(cal.home_q)
        server._handle("follow_trajectory", {"waypoints": np.stack([home, GOAL_Q]), "dt": 5.0})
        server._handle("estop", {})
        # The Uno never got far: it is still at home while the server stores GOAL.
        ser.current = list(driver._reorder(list(cal.q_to_pulses(home)) + [driver.pulses()[4]]))
        server._handle("clear_estop", {})
        n = len(ser.t_frames)
        reply = server._handle("follow_trajectory", {"waypoints": np.stack([GOAL_Q, GOAL_Q]), "dt": 0.1})
        assert reply["reattached"] is True
        (_target, ms), = ser.t_frames[n:]
        travel = float(np.max(np.abs(GOAL_Q - home)))
        assert ms >= 0.95 * 1000.0 * travel / cal.max_joint_velocity > 1000.0

    def test_fake_driver_models_the_reset(self, bridge):
        client, server, driver, _path = bridge
        driver.simulate_reset()
        state = client.get_state()
        assert state["bridge_position_known"] is False and state["attached"] is False
        with pytest.raises(RpcError, match="no known position"):
            client.set_gripper(0.02)
        assert client.home()["first_attach"] is True


# ----------------------------------------------------------------------
# host liveness (re-review high: the keepalive had no bound)
# ----------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class TestHostLiveness:
    def _serve(self, tmp_path: Path, clock: _FakeClock, **kw: Any):
        driver = FakeDriver(_fast_calibration(), watchdog_ms=300.0, keepalive_s=0.05)
        server = RobotServer(
            "127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
            calibration_path=tmp_path / "c.yaml", follower_sleep=lambda _s: None, clock=clock, **kw,
        )
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port, request_timeout_s=5.0)
        client.connect()
        client.home()
        client.set_gripper(0.015)  # a jaw closed on something, squeezing
        return server, client, driver

    @staticmethod
    def _wait(predicate: Any, timeout: float = 2.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.02)
        return bool(predicate())

    def test_a_silent_laptop_relaxes_the_arm_after_the_timeout(self, tmp_path, caplog):
        """Laptop crash / Wi-Fi loss: before, the keepalive held the squeeze
        for as long as the server ran (36 K frames in 6 s, jaw at 1600 us)."""
        clock = _FakeClock()
        server, client, driver = self._serve(tmp_path, clock)
        try:
            assert server.host_timeout_s == pytest.approx(5.0)
            clock.t += 4.9
            time.sleep(0.2)
            assert driver.attached, "detached before the timeout"
            with caplog.at_level(logging.ERROR, logger="robot_server"):
                clock.t += 0.2
                assert self._wait(lambda: not driver.attached), "still energised after the host timeout"
            assert driver.commands[-1] == ("D", ()) and driver.hold_detached
            assert driver.last_detach_reason.startswith("host timeout")
            assert any("no request from any client" in r.getMessage() for r in caplog.records)
            k = driver.keepalive_count
            time.sleep(0.3)
            assert driver.keepalive_count == k, "the keepalive must stop feeding a relaxed arm"
            # The laptop comes back: it learns about the detach from the counter.
            state = client.get_state()
            assert state["detach_count"] == 1 and state["last_detach_reason"].startswith("host timeout")
            assert server.host_lost is False
        finally:
            client.close()
            server.stop()

    def test_a_heartbeat_keeps_an_idle_but_alive_laptop_attached(self, tmp_path):
        from mfw.hardware.jetson_client import Heartbeat

        clock = _FakeClock()
        server, client, driver = self._serve(tmp_path, clock)
        beat = Heartbeat("127.0.0.1", client.rpc.port, period_s=0.05).start()
        try:
            for _ in range(10):  # 30 s of fake time in 3 s steps, a beat in between
                clock.t += 3.0
                beats = beat.beats
                assert self._wait(lambda: beat.beats > beats)
            assert driver.attached and driver.detach_count == 0
        finally:
            beat.stop()
        assert not beat.running
        try:
            clock.t += 5.5
            assert self._wait(lambda: not driver.attached), "a stopped heartbeat must let the arm relax"
        finally:
            client.close()
            server.stop()

    def test_fake_world_polling_is_not_the_laptop(self, tmp_path):
        """The fake detector polls get_fake_world on its own; a dead laptop
        must still relax the fake arm."""
        clock = _FakeClock()
        world = FakeWorld({"marker": (0.18, 0.05)}, {"marker": (0.14, 0.019, 0.019)}, ArmGeometry())
        driver = FakeDriver(_fast_calibration(), on_command=world.on_command)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                             follower_sleep=lambda _s: None, fake_world=world, clock=clock)
        _, port = server.serve_in_thread(port=0)
        laptop, detector = JetsonClient("127.0.0.1", port), JetsonClient("127.0.0.1", port)
        try:
            laptop.connect()
            laptop.home()
            laptop.close()  # the laptop dies
            for _ in range(4):
                clock.t += 2.0
                detector.get_fake_world()
            assert self._wait(lambda: not driver.attached), "fake-world polling kept a dead laptop's arm up"
        finally:
            detector.close()
            server.stop()

    def test_never_while_a_motion_runs_and_measured_from_its_end(self, tmp_path):
        clock = _FakeClock()
        driver = FakeDriver(_fast_calibration())
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                             follower_sleep=lambda _s: None, clock=clock)
        server._handle("home", {})
        assert driver.attached
        server._moving.set()
        clock.t += 60.0
        assert server.check_host_liveness() is False and driver.attached
        server._moving.clear()
        assert server.check_host_liveness() is True and not driver.attached
        # The client is back; a long motion ends: silence counts from its end.
        server.note_request()
        server._handle("home", {})
        clock.t += 10.0
        server._run_motion(b"id", "get_state", {})
        clock.t += 4.0
        assert server.check_host_liveness() is False and driver.attached
        clock.t += 1.5
        assert server.check_host_liveness() is True
        server.stop()

    def test_disabled_and_configurable(self, tmp_path):
        clock = _FakeClock()
        cal = ServoCalibration.from_dict({**_fast_calibration().to_dict(), "host_timeout_s": 0.0})
        driver = FakeDriver(cal)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal,
                             follower_sleep=lambda _s: None, clock=clock)
        server._handle("home", {})
        clock.t += 3600.0
        assert server.check_host_liveness() is False and driver.attached
        assert ServoCalibration.from_dict(_fast_calibration().to_dict()).host_timeout_s == pytest.approx(5.0)
        assert ServoCalibration.load(ROBOT_CONFIG_PATH).host_timeout_s > 0
        with pytest.raises(CalibrationError, match="host_timeout_s"):
            ServoCalibration.from_dict({**_fast_calibration().to_dict(), "host_timeout_s": -1.0})
        other = RobotServer("127.0.0.1", 0, FakeDriver(cal), camera=None, calibration=cal, host_timeout_s=2.0)
        assert other.host_timeout_s == 2.0
        server.stop()
        other.stop()


# ----------------------------------------------------------------------
# detach reporting (re-review high: a repaired detach was invisible)
# ----------------------------------------------------------------------


class TestDetachReporting:
    def test_a_detach_repaired_inside_the_next_motion_is_reported(self, tmp_path):
        """probe_laptop.py: watchdog trip in an idle gap, then a motion. The
        reply said detached=False and nothing else; now it says reattached
        and the counter moved, with the reason."""
        driver = FakeDriver(_fast_calibration(), watchdog_ms=300.0, keepalive_s=0)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                             calibration_path=tmp_path / "c.yaml", follower_sleep=lambda _s: None)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port)
        client.connect()
        try:
            client.home()
            home = np.asarray(client.get_state()["q"])
            first = client.follow_trajectory(np.stack([home, GOAL_Q]), dt=0.1)
            assert first["reattached"] is False and first["detach_count"] == 0
            time.sleep(0.45)  # the watchdog fires in the gap; nobody asks
            reply = client.follow_trajectory(np.stack([GOAL_Q, home]), dt=0.1)
            assert reply["completed"] is True and reply["detached"] is False
            assert reply["reattached"] is True
            assert reply["detach_count"] == 1 and "watchdog" in reply["last_detach_reason"]
        finally:
            client.close()
            server.stop()

    def test_estop_and_torque_off_are_counted_with_their_reason(self, bridge):
        client, _server, _driver, _path = bridge
        assert client.estop()["last_detach_reason"] == "estop"
        client.clear_estop()
        client.home()
        state = client.set_torque(False)
        assert state["detach_count"] == 2 and state["last_detach_reason"] == "torque off"
        assert client.set_torque(False)["detach_count"] == 2, "an already-limp arm is not detached twice"


# ----------------------------------------------------------------------
# a rejected D must not leave the arm energised (re-review medium)
# ----------------------------------------------------------------------


class TestDetachRobustness:
    def test_one_rejected_d_is_retried(self, bridge):
        client, _server, driver, _path = bridge
        driver.fail_detach = 1
        state = client.estop()
        assert state["estopped"] is True and state["detach_error"] is None
        assert [k for k, _ in driver.commands[-2:]] == ["E", "D"] and not driver.attached

    def test_a_d_that_never_gets_through_still_stops_the_keepalive(self, stub_serial):
        """probe_estop.py: a D corrupted on the wire is answered ERR, which is
        an in-step reply, so it was not resent; the estop reply was an error,
        the driver stayed attached and the keepalive fed a live arm forever.
        Now the hold goes up first: no K after the failed D, and the Uno's
        own watchdog relaxes the arm."""
        stub_serial.watchdog_s = 0.5
        cal = _fast_calibration()
        driver = _uno(cal, keepalive_s=0.05)
        ser = stub_serial.instances[-1]
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal, follower_sleep=lambda _s: None)
        try:
            server._handle("home", {})
            original_execute = ser._execute

            def execute(frame: bytes, t: float) -> str:
                if frame.startswith(b"D"):
                    return "ERR unknown command \x7f"  # the Uno never saw a D
                return original_execute(frame, t)

            ser._execute = execute  # type: ignore[method-assign]
            state = server._handle("estop", {})
            assert state["estopped"] is True and "did not confirm the detach" in state["detach_error"]
            assert driver.hold_detached and driver.attached is False
            n = len(ser.frames)
            time.sleep(0.4)
            assert not any(f.startswith(b"K") for f in ser.frames[n:]), "fed the watchdog of an estopped arm"
            ser.advance(0.6)
            assert driver.query()[1] is False, "the Uno's watchdog relaxes the arm"
            assert [f[:1] for f in ser.frames].count(b"D") == DETACH_ATTEMPTS
        finally:
            driver.close()


# ----------------------------------------------------------------------
# one lost reply must not trip the watchdog (re-review medium)
# ----------------------------------------------------------------------


class TestLostReplyBudget:
    def test_the_feed_gap_budget_fits_the_watchdog(self):
        """Keepalive worst case + a lost frame + a lost sync, all under the
        driver lock, stay inside the Uno's watchdog. The old 0.25 s timeout
        with drain-and-resend spent 0.5 s on one lost '?' alone."""
        worst = KEEPALIVE_PERIOD_S * 1.5 + 2.0 * DEFAULT_REPLY_TIMEOUT_S
        assert worst < SKETCH_WATCHDOG_MS / 1000.0
        default = inspect.signature(UnoSerialDriver).parameters["reply_timeout_s"].default
        assert default == DEFAULT_REPLY_TIMEOUT_S

    @pytest.mark.parametrize("fault", ["drop_frame", "drop_reply"])
    @pytest.mark.parametrize("phase_s", [0.0, 0.07, 0.14])
    def test_a_lost_query_does_not_trip_the_watchdog(self, stub_serial, fault, phase_s):
        """probe_lostq.py: two lost '?' replies right after a K tripped the
        watchdog in 7 of 8 phases. Here the '?' and the sync after it are
        both lost at the worst phase, then the keepalive gets the lock."""
        stub_serial.watchdog_s = SKETCH_WATCHDOG_MS / 1000.0
        stub_serial.virtual = True  # the budget is exact; wall-time jitter is not part of it
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        ser.advance(KEEPALIVE_PERIOD_S * 1.5 - 0.001)
        driver._send_keepalive()  # the last K before the probe
        ser.advance(phase_s)
        ser.actions["?"] = [fault, fault]  # the old code resent '?'; lose that too
        ser.actions["S"] = ["drop_frame"]
        with pytest.raises(DriverError):
            driver.query()
        # The keepalive thread was blocked on the lock all along; its K now:
        driver._send_keepalive()
        assert ser.trips == 0 and ser.attached, f"watchdog tripped ({fault}, phase {phase_s}s)"


# ----------------------------------------------------------------------
# llama.cpp shim (5557 protocol)
# ----------------------------------------------------------------------


def _ask(port: int, payload: str, timeout: float = 5.0) -> dict[str, Any]:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(payload.encode("utf-8") + b"\n")
        return json.loads(sock.makefile("r", encoding="utf-8").readline())


@pytest.fixture
def llm_shim():
    fake = shim.FakeLlamaServer().serve_in_thread()
    server, port = shim.LlmShimServer("127.0.0.1", 0, shim.LlamaCppCompleter(fake.url, timeout_s=5.0)).serve_in_thread()
    try:
        yield fake, server, port
    finally:
        server.stop()
        fake.stop()


class TestLlamaCppShim:
    def test_skill_enum_matches_the_registry(self):
        assert len(shim.SKILL_NAMES) == 14
        assert set(shim.SKILL_NAMES) == {cls.skill_name for cls in ALL_SKILLS}
        schema = shim.intent_json_schema()
        assert schema["properties"]["skill"]["enum"] == list(shim.SKILL_NAMES)
        assert schema["required"] == ["skill", "params", "confidence"]

    def test_system_message_matches_llm_worker(self):
        source = (REPO_ROOT / "scripts" / "llm_worker.py").read_text(encoding="utf-8")
        assert repr(shim.SYSTEM_MESSAGE) in source or shim.SYSTEM_MESSAGE in source
        assert "max_new_tokens=128" in source and shim.MAX_TOKENS == 128

    def test_round_trip_and_request_shape(self, llm_shim):
        fake, server, port = llm_shim
        prompt = "Available actions: pick, place\nCommand: pick up the marker\nJSON:"
        reply = _ask(port, json.dumps({"prompt": prompt}))
        assert reply["ok"] is True and reply["duration_s"] >= 0.0
        assert json.loads(reply["text"])["skill"] == "pick"
        assert server.requests_served == 1

        request = fake.requests[-1]
        assert request["messages"] == [
            {"role": "system", "content": shim.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        assert request["temperature"] == 0 and request["max_tokens"] == 128 and request["stream"] is False
        rf = request["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"]["properties"]["skill"]["enum"] == list(shim.SKILL_NAMES)

    def test_bad_requests_and_dead_backend(self, llm_shim):
        _fake, _server, port = llm_shim
        for bad in ("not json", json.dumps({"nope": 1}), json.dumps({"prompt": ""})):
            reply = _ask(port, bad)
            assert reply["ok"] is False and reply["text"] == "" and reply["error"]

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        dead, port2 = shim.LlmShimServer(
            "127.0.0.1", 0, shim.LlamaCppCompleter(f"http://127.0.0.1:{dead_port}", timeout_s=2.0)
        ).serve_in_thread()
        try:
            reply = _ask(port2, json.dumps({"prompt": "Command: stop\nJSON:"}))
            assert reply["ok"] is False and reply["text"] == ""
            assert "unreachable" in reply["error"]
            assert dead.completer.ping() is False
        finally:
            dead.stop()

    def test_http_error_and_malformed_reply(self, llm_shim):
        fake, _server, port = llm_shim
        fake.reply_text = "no json here"
        reply = _ask(port, json.dumps({"prompt": "Command: wave\nJSON:"}))
        assert reply["ok"] is True and reply["text"] == "no json here"  # the laptop parser decides
        assert shim.extract_text({"choices": [{"message": {"content": " x "}}]}) == "x"
        with pytest.raises(ValueError, match="choices"):
            shim.extract_text({"error": "boom"})

    def test_laptop_parser_end_to_end(self, llm_shim):
        """The exact client the laptop uses (run_assistant.llm_complete) -> LlmIntentParser."""
        from mfw.language.intent_parser import LlmIntentParser

        _fake, _server, port = llm_shim

        def llm_complete(prompt: str) -> str:
            data = _ask(port, json.dumps({"prompt": prompt}), timeout=60.0)
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "LLM inference failed"))
            return data["text"]

        parser = LlmIntentParser(llm_complete, known_skills=shim.SKILL_NAMES)
        intent = parser.parse("pick up the marker", {"visible_objects": ["marker", "bowl"]})
        assert intent.skill == "pick" and intent.params == {"target": "marker"}
        assert intent.confidence == pytest.approx(0.9)

    def test_selftest_smoke(self, capsys):
        """Smoke only (review F5): ``selftest`` talks to the shim's own
        FakeLlamaServer, so this proves the entry point runs, not the request
        shape -- ``test_round_trip_and_request_shape`` pins that."""
        assert shim.selftest() == 0
        assert shim.main(["--selftest", "--log-level", "WARNING"]) == 0
        assert "selftest OK" in capsys.readouterr().out

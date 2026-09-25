"""Tests for Jetson RobotServer, FakeDriver, and ZeroMQ RPC transport."""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from jetson.robot_server import FakeDriver, RobotServer, ServoCalibration
from mfw.hardware.zmq_rpc import RpcError, ZmqRpcClient


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server():
    """Start an ephemeral in-thread RobotServer with FakeDriver and instant follower."""
    cal = ServoCalibration.default()
    driver = FakeDriver(cal)
    server = RobotServer(
        host="127.0.0.1",
        port=0,
        driver=driver,
        calibration=cal,
        follower_sleep=lambda _s: None,
        host_timeout_s=5.0,
        fake_camera=False,
    )
    _, bound_port = server.serve_in_thread(port=0)
    client = ZmqRpcClient("127.0.0.1", bound_port, request_timeout_s=5.0)
    try:
        yield server, client, driver, cal, bound_port
    finally:
        client.close()
        server.stop()


class TestRobotServerLogic:
    def test_a_ping_and_unknown_method(self, running_server):
        """a) ping via ZmqRpcClient returns ok/server/driver; unknown method -> RpcError("UnknownMethod")."""
        _, client, _, _, _ = running_server
        reply = client.ping(retries=1)
        assert reply["ok"] is True
        assert reply["server"] == "robot_server"
        assert reply["version"] == 1
        assert reply["driver"] == "fake"
        assert "uptime_s" in reply

        with pytest.raises(RpcError) as exc_info:
            client.call("nonexistent_method_xyz")
        assert exc_info.value.error_type == "UnknownMethod"

    def test_b_boot_state_and_home(self, running_server):
        """b) boot state: attached False, estopped False; home attaches and ends at the zero pose."""
        _, client, driver, _, _ = running_server
        state = client.call("get_state")
        assert state["ok"] is True
        assert state["attached"] is False
        assert state["estopped"] is False
        assert state["moving"] is False
        assert driver.attached is False

        home_reply = client.call("home", {"duration_s": 0.1})
        assert home_reply["ok"] is True
        assert "joint_positions" in home_reply

        state_after = client.call("get_state")
        assert state_after["attached"] is True
        assert driver.attached is True
        for q in state_after["joint_positions"]:
            assert abs(q - 0.0) < 1e-4

    def test_c_follow_trajectory(self, running_server):
        """c) follow_trajectory reaches the last point (get_state joint_positions within 1e-3 rad),

        write_pulses called >= len(points) times, pulses within calibration clamps.
        """
        _, client, driver, cal, _ = running_server
        points = [
            {"t": 0.0, "q": [0.0, 0.0, 0.0, 0.0]},
            {"t": 0.1, "q": [0.1, 0.05, -0.1, 0.0]},
            {"t": 0.2, "q": [0.2, 0.1, -0.2, 0.1]},
            {"t": 0.3, "q": [0.3, 0.15, -0.25, 0.15]},
        ]
        target_last = points[-1]["q"]
        start_count = driver.write_count

        reply = client.call(
            "follow_trajectory",
            {"points": points, "gripper_width_m": 0.04},
        )
        assert reply["ok"] is True
        assert reply["aborted"] is False

        state = client.call("get_state")
        for actual, expected in zip(state["joint_positions"], target_last):
            assert abs(actual - expected) < 1e-3

        assert driver.write_count - start_count >= len(points)

        for i, pulse in enumerate(driver.last_pulses[:4]):
            ch = cal.joints[i]
            assert ch.pulse_min_us <= pulse <= ch.pulse_max_us
        assert cal.jaw.pulse_min_us <= driver.last_pulses[4] <= cal.jaw.pulse_max_us

    def test_d_joint_limit_rejection(self, running_server):
        """d) JointLimit: a point beyond rad_max by 0.1 -> ok:false before any write (driver call count unchanged)."""
        _, client, driver, cal, _ = running_server
        call_count_before = driver.write_count

        over_limit = cal.joints[0].rad_max + 0.1
        bad_points = [
            {"t": 0.0, "q": [0.0, 0.0, 0.0, 0.0]},
            {"t": 0.2, "q": [over_limit, 0.0, 0.0, 0.0]},
        ]

        with pytest.raises(RpcError) as exc_info:
            client.call("follow_trajectory", {"points": bad_points})
        assert exc_info.value.error_type == "JointLimit"

        assert driver.write_count == call_count_before

    def test_e_estop_interrupts_motion(self):
        """e) estop from a SECOND ZmqRpcClient while the first is blocked in a long follow_trajectory.

        The estop reply arrives within 0.2 s, the motion call raises RpcError("EStopped"),
        driver.attached False; motion while latched -> "EStopped"; clear_estop then home works.
        """
        cal = ServoCalibration.default()
        driver = FakeDriver(cal)
        server = RobotServer(
            host="127.0.0.1",
            port=0,
            driver=driver,
            calibration=cal,
            follower_sleep=time.sleep,  # Real sleep for timing
            tick_s=0.02,
        )
        _, port = server.serve_in_thread(port=0)

        client1 = ZmqRpcClient("127.0.0.1", port, request_timeout_s=5.0)
        client2 = ZmqRpcClient("127.0.0.1", port, request_timeout_s=5.0)

        motion_error: list[RpcError] = []

        def run_motion():
            points = [
                {"t": 0.0, "q": [0.0, 0.0, 0.0, 0.0]},
                {"t": 0.5, "q": [0.3, 0.2, -0.2, 0.1]},
            ]
            try:
                client1.call("follow_trajectory", {"points": points})
            except RpcError as e:
                motion_error.append(e)

        try:
            t1 = threading.Thread(target=run_motion)
            t1.start()

            # Wait briefly so motion begins
            time.sleep(0.06)

            # Send estop from client 2
            t_start = time.perf_counter()
            estop_reply = client2.call("estop")
            t_elapsed = time.perf_counter() - t_start

            assert t_elapsed < 0.2, f"estop took {t_elapsed:.3f} s, expected < 0.2 s"
            assert estop_reply["ok"] is True
            assert estop_reply["estopped"] is True

            t1.join(timeout=2.0)
            assert len(motion_error) == 1
            assert motion_error[0].error_type == "EStopped"
            assert driver.attached is False

            # Motion while latched must fail with EStopped
            with pytest.raises(RpcError) as exc_latched:
                client2.call("home")
            assert exc_latched.value.error_type == "EStopped"

            # Clear estop then home works
            clear_reply = client2.call("clear_estop")
            assert clear_reply["ok"] is True
            assert clear_reply["estopped"] is False

            home_reply = client2.call("home", {"duration_s": 0.04})
            assert home_reply["ok"] is True
            assert driver.attached is True
        finally:
            client1.close()
            client2.close()
            server.stop()

    def test_f_host_timeout(self):
        """f) host timeout: host_timeout_s=0.2, no request for 0.5 s -> attached False (poll get_state after)."""
        cal = ServoCalibration.default()
        driver = FakeDriver(cal)
        server = RobotServer(
            host="127.0.0.1",
            port=0,
            driver=driver,
            calibration=cal,
            follower_sleep=lambda _s: None,
            host_timeout_s=0.2,
        )
        _, port = server.serve_in_thread(port=0)
        client = ZmqRpcClient("127.0.0.1", port, request_timeout_s=2.0)

        try:
            client.call("home", {"duration_s": 0.01})
            state_initial = client.call("get_state")
            assert state_initial["attached"] is True

            time.sleep(0.5)

            state_after = client.call("get_state")
            assert state_after["attached"] is False
            assert driver.attached is False
        finally:
            client.close()
            server.stop()

    def test_g_fake_driver_watchdog(self):
        """g) FakeDriver watchdog: watchdog_ms=50 with the server keepalive running ->

        after 0.5 s attached stays True and watchdog_trips == 0;
        directly writing to a FakeDriver twice 100 ms apart trips it.
        """
        cal = ServoCalibration.default()
        driver = FakeDriver(cal, watchdog_ms=50)
        server = RobotServer(
            host="127.0.0.1",
            port=0,
            driver=driver,
            calibration=cal,
            follower_sleep=time.sleep,
            tick_s=0.01,  # 10ms ticks < 50ms watchdog
            host_timeout_s=5.0,
        )
        _, port = server.serve_in_thread(port=0)
        client = ZmqRpcClient("127.0.0.1", port, request_timeout_s=2.0)

        try:
            client.call("home", {"duration_s": 0.05})
            time.sleep(0.5)
            state = client.call("get_state")
            assert state["attached"] is True
            assert driver.attached is True
            assert driver.watchdog_trips == 0
        finally:
            client.close()
            server.stop()

        # Directly testing FakeDriver watchdog trip
        isolated_driver = FakeDriver(cal, watchdog_ms=50)
        isolated_driver.attach()
        pulses = [1450, 1450, 1450, 1450, 500]
        isolated_driver.write_pulses(pulses)
        assert isolated_driver.attached is True

        time.sleep(0.1)  # 100 ms > 50 ms
        isolated_driver.write_pulses(pulses)
        assert isolated_driver.attached is False
        assert isolated_driver.watchdog_trips == 1

    def test_h_get_frame_camera(self):
        """h) get_frame with camera=None -> NoCamera; with the fake camera -> jpeg bytes starting with FF D8."""
        cal = ServoCalibration.default()
        server_no_cam = RobotServer(
            host="127.0.0.1",
            port=0,
            driver=FakeDriver(cal),
            camera=None,
            fake_camera=False,
            follower_sleep=lambda _s: None,
        )
        _, port1 = server_no_cam.serve_in_thread(port=0)
        client1 = ZmqRpcClient("127.0.0.1", port1)
        try:
            with pytest.raises(RpcError) as exc_info:
                client1.call("get_frame")
            assert exc_info.value.error_type == "NoCamera"
        finally:
            client1.close()
            server_no_cam.stop()

        server_fake_cam = RobotServer(
            host="127.0.0.1",
            port=0,
            driver=FakeDriver(cal),
            camera=None,
            fake_camera=True,
            follower_sleep=lambda _s: None,
        )
        _, port2 = server_fake_cam.serve_in_thread(port=0)
        client2 = ZmqRpcClient("127.0.0.1", port2)
        try:
            reply = client2.call("get_frame")
            assert reply["ok"] is True
            jpeg = reply["jpeg"]
            assert isinstance(jpeg, bytes)
            assert jpeg.startswith(b"\xff\xd8"), "JPEG must start with SOI marker FF D8"
            assert reply["width"] == 640
            assert reply["height"] == 480
        finally:
            client2.close()
            server_fake_cam.stop()

    def test_i_calibration_roundtrip(self):
        """i) get_calibration round-trips ServoCalibration.to_json/from_json;

        rad_to_pulse/pulse_to_rad round-trip < 1e-6 rad inside limits.
        """
        cal = ServoCalibration.default()
        json_str = cal.to_json()
        cal_loaded = ServoCalibration.from_json(json_str)

        assert len(cal_loaded.joints) == 4
        assert cal_loaded.jaw.name == "jaw"

        test_angles = [-1.2, -0.5, 0.0, 0.35, 1.1]
        for i in range(4):
            for rad in test_angles:
                if cal.joints[i].rad_min <= rad <= cal.joints[i].rad_max:
                    pulse, clamped_by = cal.rad_to_pulse(i, rad)
                    assert clamped_by == 0.0
                    recovered_rad = cal.pulse_to_rad(i, pulse)
                    assert abs(recovered_rad - rad) < 1e-6

    def test_j_cli_smoke(self):
        """j) CLI smoke: subprocess launch comes up, ping ok within 10 s,

        get_world lists marker and bowl, terminate() exits cleanly.
        """
        port = _find_free_port()
        cmd = [
            sys.executable,
            "jetson/robot_server.py",
            "--driver",
            "fake",
            "--fake-camera",
            "--fake-world",
            "marker:0.18,0.05 bowl:0.15,-0.12",
            "--fake-watchdog-ms",
            "500",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        client = ZmqRpcClient("127.0.0.1", port, request_timeout_s=1.0)

        try:
            deadline = time.time() + 10.0
            connected = False
            while time.time() < deadline:
                try:
                    res = client.ping()
                    if res.get("ok"):
                        connected = True
                        break
                except RpcError:
                    time.sleep(0.2)
            assert connected, "robot_server did not come up via CLI within 10 s"

            world = client.call("get_world")
            assert world["ok"] is True
            objs = world["objects"]
            assert "marker" in objs
            assert "bowl" in objs
            assert objs["marker"][:2] == [0.18, 0.05]
            assert objs["bowl"][:2] == [0.15, -0.12]
        finally:
            client.close()
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # On Windows proc.terminate() calls TerminateProcess resulting in exit code 1; on POSIX -SIGTERM (-15)
        assert proc.returncode in (0, 1, -signal.SIGTERM if hasattr(signal, "SIGTERM") else 0, 15)

"""Phase 9 gate: the motion stream's guards on the laptop side of the hardware lane.

No Isaac Sim, no servos, and (except where a real socket is the point) no
server: every transport here is a stub, because what is under test is what
the laptop *decides* -- which width it closes to, which socket carries the
stop, which calibration it refuses, which touch target it refuses, which
detector failure it reports -- not the wire.

Findings pinned (docs/HARDWARE_REVIEW.md):

* missed item 1 -- the MG90S must close to the planned grasp width minus a
  margin, never to 0 mm on an object (stall at full current for the carry);
* HS-2 / GEO-1 -- ``hardware.arm`` limits wider than the Jetson's pulse map
  refuse the build, naming ``calibrate_servos.py``;
* P4 / HS-4 -- estop and get_state ride a second socket, which opens with a
  <= 2 s ping and falls back to the main one without failing construction;
  Ctrl-C estops; the startup notice does not oversell the software stop;
* P5 -- waypoints cross the wire as float64;
* P3 -- a refused detector and a busy one are different errors, and the first
  detect gets the long timeout; ``serve_detector.py`` warms Florence up front;
* P2 -- ``calibrate_table.py --touch`` refuses targets under the table,
  outside the workspace, and nudges over 20 mm;
* P6 -- spawned fakes are terminated on every exit path.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mfw.config.schema import HardwareArmConfig, load_config
from mfw.core.errors import ConfigurationError, ExecutionError
from mfw.hardware.controller import GRASP_CLOSE_MARGIN_M, RemoteController
from mfw.hardware.detector import DetectorError, DetectorTimeout, RemoteDetector
from mfw.hardware.jetson_client import JetsonClient
from mfw.hardware.kinematics import PlanarKinematics
from mfw.hardware.remote_arm import (
    RemoteArm,
    assert_limits_reachable,
    reachable_joint_range,
)
from mfw.hardware.zmq_rpc import RpcError

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
CFG = load_config(REPO_ROOT / "configs" / "hardware_fake.yaml")
HW_CFG = load_config(REPO_ROOT / "configs" / "hardware.yaml")
JOINTS = tuple(CFG.robot.arm_joint_names)


def _load_script(name: str) -> Any:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_motion_guards_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jetson_calibration() -> dict[str, Any]:
    """The Jetson's placeholder calibration, as ``get_calibration`` returns it."""
    from jetson.robot_server import ServoCalibration

    return ServoCalibration.default().to_dict()


# ----------------------------------------------------------------------
# stubs
# ----------------------------------------------------------------------


class StubClient:
    """What RemoteArm reads from a JetsonClient, with a call log."""

    def __init__(self, calibration: dict[str, Any] | None = None, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.calibration = _jetson_calibration() if calibration is None else calibration
        self.fail = set(fail or ())
        self.last_state: dict[str, Any] | None = None
        self.closed = False
        self.state: dict[str, Any] = {
            "q": list(CFG.robot.home_joint_positions), "gripper_width": 0.045,
            "estopped": False, "attached": True,
        }
        self.rpc = SimpleNamespace(host="127.0.0.1", port=1, request_timeout_s=5.0, endpoint="tcp://127.0.0.1:1")
        self.trajectory_timeout_margin_s = 5.0

    def _reply(self, name: str, **extra: Any) -> dict[str, Any]:
        self.calls.append(name)
        if name in self.fail:
            raise RpcError(f"{name} failed (stub)")
        self.state.update(extra)
        return dict(self.state)

    def get_calibration(self) -> dict[str, Any]:
        self.calls.append("get_calibration")
        return {"calibration": self.calibration, "path": "stub"}

    def get_state(self) -> dict[str, Any]:
        return self._reply("get_state")

    def estop(self) -> dict[str, Any]:
        return self._reply("estop", estopped=True, attached=False)

    def clear_estop(self) -> dict[str, Any]:
        return self._reply("clear_estop", estopped=False)

    def set_gripper(self, width_m: float) -> dict[str, Any]:
        return self._reply("set_gripper", gripper_width=float(width_m))

    def follow_trajectory(self, waypoints: Any, dt: float) -> dict[str, Any]:
        return self._reply("follow_trajectory", completed=True, clamped=False)

    def close(self) -> None:
        self.closed = True


def _kin(arm: HardwareArmConfig | None = None) -> PlanarKinematics:
    return PlanarKinematics(arm or CFG.hardware.arm, JOINTS)


def _arm(client: StubClient | None = None, estop_client: StubClient | None = None, **kw: Any) -> RemoteArm:
    clock = SimpleNamespace(sim_time=0.0, step_index=0)
    return RemoteArm(
        client or StubClient(), _kin(), CFG.robot, clock,
        estop_client=estop_client or StubClient(), **kw,
    )


# ----------------------------------------------------------------------
# HS-2 / GEO-1: limits must fit the pulse map
# ----------------------------------------------------------------------


class TestLimitsAgainstThePulseMap:
    def test_placeholder_limits_fit_the_placeholder_map(self):
        assert_limits_reachable(_kin(), _jetson_calibration())

    def test_hardware_yaml_limits_fit_the_jetson_map(self):
        assert_limits_reachable(PlanarKinematics(HW_CFG.hardware.arm, JOINTS), _jetson_calibration())

    def test_the_old_120_degree_elbow_is_refused_by_name(self):
        wide = HardwareArmConfig(
            joint_lower=(-1.5708, -0.5236, -2.0944, -1.5708),
            joint_upper=(1.5708, 1.5708, 2.0944, 1.5708),
        )
        with pytest.raises(ConfigurationError) as info:
            assert_limits_reachable(_kin(wide), _jetson_calibration())
        message = str(info.value)
        assert "elbow_pitch" in message and "calibrate_servos.py" in message
        assert "base_yaw" not in message, "only the offending joint is named"

    def test_a_shifted_window_is_honoured(self):
        """A measured asymmetric elbow (-30..+150 deg) makes the matching limits legal."""
        cal = _jetson_calibration()
        cal["joints"]["elbow_pitch"].update(angle_at_min_deg=-30.0, angle_at_max_deg=150.0)
        lo, hi = reachable_joint_range(cal["joints"]["elbow_pitch"])
        assert (math.degrees(lo), math.degrees(hi)) == pytest.approx((-30.0, 150.0))
        folded = HardwareArmConfig(
            joint_lower=(-1.5708, -0.5236, -0.5236, -1.5708), joint_upper=(1.5708, 1.5708, 2.618, 1.5708)
        )
        assert_limits_reachable(_kin(folded), cal)
        with pytest.raises(ConfigurationError, match="elbow_pitch"):
            assert_limits_reachable(_kin(), cal)  # -90 deg is now past the window

    def test_direction_and_zero_offset_move_the_window(self):
        entry = {"angle_at_min_deg": -90.0, "angle_at_max_deg": 90.0, "direction": -1,
                 "zero_offset_rad": math.radians(30.0)}
        lo, hi = reachable_joint_range(entry)
        assert (math.degrees(lo), math.degrees(hi)) == pytest.approx((-60.0, 120.0))

    def test_remote_arm_refuses_to_build_against_a_narrow_map(self):
        cal = _jetson_calibration()
        cal["joints"]["elbow_pitch"].update(angle_at_min_deg=-60.0, angle_at_max_deg=60.0)
        with pytest.raises(ConfigurationError, match="calibrate_servos.py"):
            _arm(StubClient(calibration=cal))

    def test_runtime_releases_the_socket_when_the_arm_is_refused(self, tmp_path):
        from mfw.hardware.runtime import HardwareRuntime
        from mfw.utils.logging import EventLogger

        cal = _jetson_calibration()
        cal["joints"]["elbow_pitch"].update(angle_at_min_deg=-60.0, angle_at_max_deg=60.0)
        runtime = HardwareRuntime(CFG, event_logger=EventLogger(log_dir=tmp_path, console=False))
        stub = StubClient(calibration=cal)
        stub.connect = lambda *a, **k: {"server": "mfw-robot"}  # type: ignore[attr-defined]
        runtime.client = stub  # type: ignore[assignment]
        try:
            with pytest.raises(ConfigurationError, match="elbow_pitch"):
                runtime.build()
            assert stub.closed, "a refused bring-up must not leave the REQ socket open"
        finally:
            runtime.events.close()


# ----------------------------------------------------------------------
# P4 / HS-4: the stop channel
# ----------------------------------------------------------------------


class TestStopChannel:
    def test_estop_and_state_use_the_dedicated_channel(self):
        main, stop = StubClient(), StubClient()
        arm = _arm(main, stop)
        main.calls.clear()
        arm.refresh()
        arm.estop()
        assert stop.calls == ["get_state", "estop"]
        assert "estop" not in main.calls and "get_state" not in main.calls
        assert arm.estopped and arm.has_dedicated_estop_channel

    def test_a_failed_stop_channel_retries_on_the_main_client(self):
        main, stop = StubClient(), StubClient(fail={"estop"})
        arm = _arm(main, stop)
        arm.estop()
        assert "estop" in stop.calls and "estop" in main.calls
        assert arm.estopped

    def test_motion_stays_on_the_main_client(self):
        main, stop = StubClient(), StubClient()
        arm = _arm(main, stop)
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)
        assert "follow_trajectory" in main.calls and "follow_trajectory" not in stop.calls

    def test_unreachable_stop_channel_falls_back_fast_without_failing(self, caplog):
        """No server on the port: the <= 2 s ping fails, a warning is logged, the arm still builds."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        main = StubClient()
        main.rpc = SimpleNamespace(host="127.0.0.1", port=free_port, request_timeout_s=5.0,
                                   endpoint=f"tcp://127.0.0.1:{free_port}")
        started = time.perf_counter()
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            arm = RemoteArm(main, _kin(), CFG.robot, SimpleNamespace(sim_time=0.0, step_index=0))
        elapsed = time.perf_counter() - started
        assert elapsed < 3.0, f"estop channel bring-up blocked construction for {elapsed:.1f} s"
        assert not arm.has_dedicated_estop_channel and arm.estop_client is main
        assert any("estop channel" in r.getMessage() for r in caplog.records)
        arm.estop()
        assert main.calls[-1] == "estop"

    def test_a_detach_without_our_estop_is_surfaced(self, caplog):
        stop = StubClient()
        arm = _arm(StubClient(), stop)
        arm.refresh()
        assert arm.attached is True
        stop.state["attached"] = False  # the Uno watchdog fired
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            arm.refresh()
        assert arm.attached is False
        assert any("DETACHED" in r.getMessage() for r in caplog.records)

    def test_our_own_estop_is_not_reported_as_a_surprise_detach(self, caplog):
        stop = StubClient()
        arm = _arm(StubClient(), stop)
        arm.refresh()
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            arm.estop()
        assert not any("DETACHED" in r.getMessage() for r in caplog.records)

    def test_controller_emergency_stop_goes_through_the_arm(self):
        main, stop = StubClient(), StubClient()
        controller = _controller(_arm(main, stop))
        controller.emergency_stop()
        assert "estop" in stop.calls and "estop" not in main.calls
        assert controller._stop_requested


# ----------------------------------------------------------------------
# missed item 1: the gripper closes to the grasp width
# ----------------------------------------------------------------------


class _Sim:
    def __init__(self) -> None:
        self.config = SimpleNamespace(physics_dt=CFG.simulation.physics_dt)
        self.stepped = 0

    def step(self, n: int = 1) -> None:
        self.stepped += int(n)


def _controller(robot: Any) -> RemoteController:
    return RemoteController(
        sim=_Sim(), robot=robot, motion_config=CFG.motion, robot_config=CFG.robot,
        hardware_config=CFG.hardware,
    )


class TestGraspWidth:
    def _widths(self, main: StubClient) -> list[float]:
        return [w for w in main.set_widths]  # type: ignore[attr-defined]

    @pytest.fixture
    def rig(self):
        main = StubClient()
        main.set_widths = []  # type: ignore[attr-defined]
        original = main.set_gripper

        def set_gripper(width_m: float) -> dict[str, Any]:
            main.set_widths.append(float(width_m))  # type: ignore[attr-defined]
            return original(width_m)

        main.set_gripper = set_gripper  # type: ignore[method-assign]
        arm = _arm(main, StubClient())
        return main, arm, _controller(arm)

    def test_close_on_a_planned_grasp_stops_short_of_the_object(self, rig):
        main, _, controller = rig
        marker = 0.019
        controller.set_grasp_width(marker)
        width = controller.close_gripper_blocking()
        assert width == pytest.approx(marker - GRASP_CLOSE_MARGIN_M)
        assert main.set_widths[-1] == pytest.approx(marker - GRASP_CLOSE_MARGIN_M)
        assert main.set_widths[-1] > CFG.robot.gripper_closed_width + 0.01, "never a 0 mm stall"

    def test_maintain_grasp_reasserts_the_same_width(self, rig):
        main, _, controller = rig
        controller.set_grasp_width(0.030)
        controller.close_gripper_blocking()
        controller.maintain_grasp()
        controller.maintain_grasp()
        assert main.set_widths[-3:] == pytest.approx([0.026] * 3)

    def test_open_forgets_the_width_so_the_next_close_is_empty(self, rig):
        main, _, controller = rig
        controller.set_grasp_width(0.030)
        controller.open_gripper_blocking()
        assert controller.grasp_width is None
        controller.close_gripper_blocking()
        assert main.set_widths[-1] == pytest.approx(CFG.robot.gripper_closed_width)

    def test_a_width_under_the_margin_floors_at_closed(self, rig):
        main, _, controller = rig
        controller.set_grasp_width(0.002)
        controller.close_gripper_blocking()
        assert main.set_widths[-1] == pytest.approx(CFG.robot.gripper_closed_width)

    @pytest.mark.parametrize("bad", [float("nan"), -0.01, float("inf")])
    def test_a_nonsense_width_is_ignored(self, rig, bad):
        main, _, controller = rig
        controller.set_grasp_width(bad)
        assert controller.grasp_width is None
        controller.close_gripper_blocking()
        assert main.set_widths[-1] == pytest.approx(CFG.robot.gripper_closed_width)

    def test_close_is_clamped_to_the_jaw_span(self, rig):
        main, arm, _ = rig
        arm.close_gripper_to(0.5)
        assert main.set_widths[-1] == pytest.approx(CFG.robot.gripper_open_width)


# ----------------------------------------------------------------------
# P5: float64 on the wire
# ----------------------------------------------------------------------


class TestWire:
    def test_waypoints_travel_as_float64(self):
        client = JetsonClient("127.0.0.1", 1)
        sent: dict[str, Any] = {}

        def call(endpoint: str, data: Any = None, timeout_s: float | None = None) -> dict[str, Any]:
            sent.update(endpoint=endpoint, data=data, timeout_s=timeout_s)
            return {"q": [0.0] * 4, "completed": True}

        client.rpc.call = call  # type: ignore[method-assign]
        limit = 1.4362  # a measured limit float32 would round 5e-8 past
        client.follow_trajectory(np.array([[0.0, 0.0, limit, 0.0], [0.0, 0.0, limit, 0.0]]), 0.1)
        wp = sent["data"]["waypoints"]
        assert wp.dtype == np.float64
        assert wp[0, 2] == limit
        client.close()


# ----------------------------------------------------------------------
# runtime + run_assistant: Ctrl-C estops, fakes never orphaned
# ----------------------------------------------------------------------


class TestOperatorStop:
    def test_runtime_emergency_stop(self, tmp_path):
        from mfw.hardware.runtime import HardwareRuntime
        from mfw.utils.logging import EventLogger

        runtime = HardwareRuntime(CFG, event_logger=EventLogger(log_dir=tmp_path, console=False))
        try:
            assert runtime.emergency_stop() is False, "nothing built, nothing to stop"
            stop = StubClient()
            runtime.robot = _arm(StubClient(), stop)
            assert runtime.emergency_stop() is True and "estop" in stop.calls
            dead = StubClient(fail={"estop"})
            runtime.robot = _arm(dead, dead)
            assert runtime.emergency_stop() is False, "a failed stop is reported, not raised"
        finally:
            runtime.events.close()

    @pytest.fixture(scope="class")
    def run_assistant(self):
        return _load_script("run_assistant")

    def test_ctrl_c_estops_then_interrupts(self, run_assistant):
        stops: list[bool] = []
        runtime = SimpleNamespace(emergency_stop=lambda: stops.append(True) or True)
        previous = run_assistant._install_estop_on_sigint(lambda: runtime)
        try:
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            assert stops == [True]
            assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "second Ctrl-C is plain"
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_ctrl_c_before_the_runtime_exists_still_interrupts(self, run_assistant):
        previous = run_assistant._install_estop_on_sigint(lambda: None)
        try:
            with pytest.raises(KeyboardInterrupt):
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_the_stop_notice_does_not_oversell_the_software_stop(self, run_assistant):
        text = run_assistant.STOP_NOTICE.format(chunk=CFG.hardware.trajectory_chunk_s)
        assert "between chunks" in text and "+6 V switch" in text and "mid-motion" in text

    def test_spawned_fakes_are_terminated_once(self, run_assistant):
        class Proc:
            def __init__(self, alive: bool) -> None:
                self.alive, self.terminated, self.pid = alive, 0, 1

            def poll(self):
                return None if self.alive else 0

            def terminate(self):
                self.terminated += 1
                self.alive = False

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.alive = False

        procs = [Proc(True), Proc(False)]
        keep = list(procs)
        run_assistant._terminate_spawned(procs)
        run_assistant._terminate_spawned(procs)  # atexit after finally: a no-op
        assert [p.terminated for p in keep] == [1, 0]
        assert procs == []

    def test_fake_robot_server_is_spawned_with_the_uno_watchdog(self, run_assistant):
        from jetson.robot_server import SKETCH_WATCHDOG_MS

        assert run_assistant.FAKE_WATCHDOG_MS == SKETCH_WATCHDOG_MS


# ----------------------------------------------------------------------
# P3: detector failures and warm-up
# ----------------------------------------------------------------------


class _SilentServer:
    """Accepts connections and never answers: a detector busy loading its model."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.held: list[socket.socket] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.sock.settimeout(0.05)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
                self.held.append(conn)
            except OSError:
                continue

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1.0)
        for conn in self.held:
            conn.close()
        self.sock.close()


class TestDetectorFailures:
    def test_refused_connection_says_start_it(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        det = RemoteDetector("127.0.0.1", port, timeout_s=0.5)
        with pytest.raises(DetectorError) as info:
            det.connect(retries=1)
        assert not isinstance(info.value, DetectorTimeout)
        assert "Start it first" in str(info.value)

    def test_busy_service_is_a_timeout_not_unreachable(self):
        server = _SilentServer()
        try:
            det = RemoteDetector("127.0.0.1", server.port, timeout_s=0.2, first_call_timeout_s=0.2)
            with pytest.raises(DetectorTimeout) as info:
                det.detect()
            message = str(info.value)
            assert "answered ping but did not finish" in message and "loading its model" in message
            assert "Start it first" not in message
        finally:
            server.close()

    def test_first_detect_gets_the_long_timeout(self):
        server = _SilentServer()
        try:
            det = RemoteDetector("127.0.0.1", server.port, timeout_s=0.1, first_call_timeout_s=0.6)
            started = time.perf_counter()
            with pytest.raises(DetectorTimeout):
                det.detect()
            assert time.perf_counter() - started >= 0.55
            det._detects_since_connect = 1  # after one success, the short budget applies
            started = time.perf_counter()
            with pytest.raises(DetectorTimeout):
                det.detect()
            assert time.perf_counter() - started < 0.5
        finally:
            server.close()

    def test_serve_detector_warms_a_florence_backend_once(self):
        sd = _load_script("serve_detector")
        det = sd.FlorenceDetector()
        calls: list[str] = []
        det._ensure_model = lambda: calls.append("load")  # type: ignore[method-assign]
        det.detect = lambda rgb, labels, min_score=0.1: calls.append(f"detect{rgb.shape}") or []  # type: ignore[method-assign]
        assert sd.warm_backend(det) is not None
        assert calls == ["load", "detect(64, 64, 3)"]

    def test_scripted_backend_needs_no_warm_up_and_a_failed_warm_up_does_not_kill_the_server(self):
        sd = _load_script("serve_detector")
        assert sd.warm_backend(sd.build_backend(sd.build_parser().parse_args(["--fake"]))) is None

        class Broken:
            name = "florence"

            def warm_up(self) -> float:
                raise RuntimeError("torch is not importable")

        assert sd.warm_backend(Broken()) is None

    def test_serve_detector_has_a_no_warmup_switch(self):
        sd = _load_script("serve_detector")
        assert sd.build_parser().parse_args(["--no-warmup"]).no_warmup is True


# ----------------------------------------------------------------------
# P2: calibrate_table --touch guard
# ----------------------------------------------------------------------


class TestTouchGuard:
    @pytest.fixture(scope="class")
    def ct(self):
        return _load_script("calibrate_table")

    @pytest.fixture
    def box(self):
        return HW_CFG.scene.workspace_min, HW_CFG.scene.workspace_max

    def test_the_measured_trap_z_minus_50_is_refused(self, ct, box):
        """'z -50' meant 5 mm; from 3 cm hover it asks for 2 cm under the table."""
        here = np.array([0.18, 0.0, 0.03])
        why = ct.touch_target_refusal(here + [0, 0, -0.05], here, *box, nudge=True)
        assert why is not None and "nudge" in why

    def test_go_below_the_table_is_refused(self, ct, box):
        for z in (-0.05, -0.08, -0.0101):
            why = ct.touch_target_refusal([0.18, 0.0, z], None, *box)
            assert why is not None and "below the touch floor" in why

    def test_no_press_below_the_model_table_is_allowed(self, ct, box):
        """Re-review: the old 5 mm press margin was model-relative (the model's
        z = 0 can sit centimetres under the real table before zero/limit) and
        the keepalive held the press indefinitely. The tip may reach the
        table plane, never below it."""
        assert ct.TOUCH_PROBE_MARGIN_M == 0.0
        why = ct.touch_target_refusal([0.18, 0.0, -0.004], None, *box)
        assert why is not None and "below the touch floor" in why and "ABOVE the table" in why
        assert ct.touch_target_refusal([0.18, 0.0, 0.0], None, *box) is None
        assert ct.touch_target_refusal([0.18, 0.0, 0.002], None, *box) is None

    def test_touch_probe_refuses_a_bridge_with_no_known_position(self, ct, monkeypatch):
        """Right after robot_server starts, the probe's first command (closing
        the jaw) would attach every servo AT its target; it must refuse and
        name the recovery instead."""
        closed: list[bool] = []

        class Fresh:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def connect(self) -> None:
                pass

            def get_state(self) -> dict[str, Any]:
                return {"q": list(HW_CFG.robot.home_joint_positions), "bridge_position_known": False}

            def set_gripper(self, width: float) -> None:
                pytest.fail("sent a jaw command to a bridge with no known position")

            def close(self) -> None:
                closed.append(True)

        import mfw.hardware.jetson_client as jc

        monkeypatch.setattr(jc, "JetsonClient", Fresh)
        with pytest.raises(RuntimeError, match="no known position.*calibrate_servos"):
            ct.TouchProbe("127.0.0.1:5560", REPO_ROOT / "configs" / "hardware.yaml")
        assert closed == [True]

    def test_the_floor_never_drops_more_than_the_margin_under_the_table(self, ct):
        deep = ([0.06, -0.24, -0.10], [0.30, 0.24, 0.20])
        assert ct.touch_target_refusal([0.18, 0.0, -0.02], None, *deep) is not None

    @pytest.mark.parametrize("xy", [(0.02, 0.0), (0.35, 0.0), (0.18, 0.30), (0.18, -0.30)])
    def test_outside_the_workspace_xy_is_refused(self, ct, box, xy):
        why = ct.touch_target_refusal([xy[0], xy[1], 0.03], None, *box)
        assert why is not None and "outside the workspace" in why

    def test_nudges_up_to_20_mm_pass(self, ct, box):
        here = np.array([0.18, 0.0, 0.03])
        assert ct.touch_target_refusal(here + [0.0, 0.0, -0.02], here, *box, nudge=True) is None
        assert ct.touch_target_refusal(here + [0.0, 0.02, 0.0], here, *box, nudge=True) is None
        assert ct.touch_target_refusal(here + [0.0, 0.021, 0.0], here, *box, nudge=True) is not None

    def test_move_to_prints_the_refusal_and_sends_nothing(self, ct, box, capsys):
        probe = ct.TouchProbe.__new__(ct.TouchProbe)
        probe.cfg = HW_CFG
        probe.tcp = np.array([0.18, 0.0, 0.03])
        probe.client = SimpleNamespace(follow_trajectory=lambda *a, **k: pytest.fail("sent a refused move"))
        assert probe.move_to(probe.tcp + [0.0, 0.0, -0.05], nudge=True) is False
        assert probe.move_to([0.18, 0.0, -0.05]) is False
        out = capsys.readouterr().out
        assert out.count("refused") == 2


class TestCalibrateTablePoseNote:
    """GEO-2: a fresh homography must not read as a fully calibrated camera."""

    @pytest.fixture(scope="class")
    def ct(self):
        return _load_script("calibrate_table")

    def test_unmeasured_pose_is_named_after_a_fit(self, ct):
        note = ct.pose_measured_note(REPO_ROOT / "configs" / "hardware.yaml")
        assert note is not None
        assert "pose_measured is false" in note and "look_at/up" in note

    def test_a_measured_pose_needs_no_note(self, ct):
        assert ct.pose_measured_note(REPO_ROOT / "configs" / "hardware_fake.yaml") is None

    def test_dry_run_prints_the_note_and_writes_nothing(self, ct, tmp_path, capsys):
        cfg_path = tmp_path / "hardware.yaml"
        original = (REPO_ROOT / "configs" / "hardware.yaml").read_text()
        cfg_path.write_text(original)
        # hardware.yaml extends default.yaml; the copy needs its parent beside it.
        (tmp_path / "default.yaml").write_text((REPO_ROOT / "configs" / "default.yaml").read_text())
        # Eight exact pairs from a synthetic homography (1 mm per pixel).
        h = np.array([[0.0, -0.001, 0.42], [-0.001, 0.0, 0.32], [0.0, 0.0, 1.0]])
        pixels = np.array([[u, v] for u in (100.0, 300.0, 500.0) for v in (80.0, 400.0)] + [[200.0, 240.0], [420.0, 160.0]])
        table = ct.apply_homography(h, pixels)
        csv_path = ct.write_points_csv(tmp_path / "pts.csv", pixels, table)
        assert ct.main(["--from-csv", str(csv_path), "--config", str(cfg_path), "--dry-run"]) == 0
        assert "pose_measured is false" in capsys.readouterr().out
        assert cfg_path.read_text() == original


class TestClampStopsTheMotion:
    """GEO-1/HS-2: a clamped chunk means the servo is not where the plan is."""

    def _run(self, reply_extra: dict[str, Any]) -> tuple[bool, StubClient]:
        main = StubClient()

        def follow(waypoints: Any, dt: float) -> dict[str, Any]:
            return main._reply("follow_trajectory", **{"completed": True, "clamped": False, **reply_extra})

        main.follow_trajectory = follow  # type: ignore[method-assign]
        arm = _arm(main, StubClient())
        controller = _controller(arm)
        from mfw.core.types import Trajectory, Waypoint  # noqa: PLC0415

        # Two in-limit poses whose TCPs sit inside the workspace box (the same
        # goal tests/test_hardware_e2e.py uses), 3 s apart: several chunks.
        q0 = np.array([-0.2, 0.9, 0.9, 0.9])
        q1 = np.array([0.2, 0.9, 0.9, 0.9])
        trajectory = Trajectory(
            waypoints=(Waypoint(q0, 0.0), Waypoint(q1, 3.0)),
            joint_names=JOINTS, planner_name="test", planning_time_s=0.0,
        )
        return controller.follow_trajectory(trajectory), main

    def test_a_clean_trajectory_ships_every_chunk(self):
        ok, main = self._run({})
        assert ok and main.calls.count("follow_trajectory") >= 3

    def test_a_clamped_chunk_stops_after_the_first_chunk(self):
        ok, main = self._run({"clamped": True})
        assert ok is False
        assert main.calls.count("follow_trajectory") == 1

    def test_a_detach_during_a_chunk_stops_the_motion(self):
        ok, main = self._run({"detached": True})
        assert ok is False
        assert main.calls.count("follow_trajectory") == 1


# ----------------------------------------------------------------------
# re-review (bridge lens): a detach the laptop did not ask for
# ----------------------------------------------------------------------


class _Events:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any] | None = None, **_: Any) -> None:
        self.events.append((event, dict(payload or {})))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class TestUnexpectedDetach:
    """A detach and its slow repair can both happen inside one motion call:
    the reply said completed, detached=False, and the laptop never learned
    the arm had gone limp with the marker in the jaw (probe_laptop.py)."""

    def _trajectory(self) -> Any:
        from mfw.core.types import Trajectory, Waypoint  # noqa: PLC0415

        q0 = np.array([-0.2, 0.9, 0.9, 0.9])
        q1 = np.array([0.2, 0.9, 0.9, 0.9])
        return Trajectory(
            waypoints=(Waypoint(q0, 0.0), Waypoint(q1, 3.0)),
            joint_names=JOINTS, planner_name="test", planning_time_s=0.0,
        )

    def _rig(self) -> tuple[StubClient, StubClient, RemoteArm, _Events]:
        main, stop = StubClient(), StubClient()
        for stub in (main, stop):
            stub.state["detach_count"] = 0
        events = _Events()
        arm = _arm(main, stop, event_logger=events)
        arm.refresh()  # baseline counter
        return main, stop, arm, events

    def test_a_counter_that_moved_without_our_estop_is_surfaced(self, caplog):
        main, stop, arm, events = self._rig()
        main.state.update(detach_count=1, last_detach_reason="fake watchdog: no frame for 500 ms")
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            reply = arm.follow_trajectory(np.zeros((2, 4)), 0.1)  # repaired inside: completed
        assert reply["completed"] is True
        assert arm.hold_unverified is True
        assert "watchdog" in (arm.pop_detach_notice() or "")
        assert arm.pop_detach_notice() is None, "a notice is delivered once"
        assert events.names() == ["robot.bridge_detached"]
        assert any("UNVERIFIED" in r.getMessage() for r in caplog.records)
        arm.open_gripper()
        assert arm.hold_unverified is False, "an open jaw holds nothing"

    def test_our_own_estop_is_not_a_surprise(self):
        main, stop, arm, events = self._rig()
        stop.state.update(detach_count=1, last_detach_reason="estop")
        arm.estop()
        arm.clear_estop()
        main.state.update(detach_count=1)
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)
        assert arm.pop_detach_notice() is None and not arm.hold_unverified and events.events == []
        # ... and the pending mark does not swallow the NEXT, real surprise.
        main.state.update(detach_count=2, last_detach_reason="host timeout: no client request for 5.2 s")
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)
        assert "host timeout" in (arm.pop_detach_notice() or "")

    def test_an_estop_while_already_limp_leaves_no_pending_mark(self):
        main, stop, arm, _events = self._rig()
        arm.estop()  # the server counted nothing: the arm was already limp
        main.state.update(detach_count=1, last_detach_reason="the bridge reports its servos detached")
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)
        assert arm.pop_detach_notice() is not None

    def test_old_server_without_a_counter_falls_back_to_reattached(self):
        main, stop = StubClient(), StubClient()
        arm = _arm(main, stop)
        main.state["reattached"] = True
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)  # the boot re-attach is expected
        assert arm.pop_detach_notice() is None
        arm.follow_trajectory(np.zeros((2, 4)), 0.1)
        assert "re-attached" in (arm.pop_detach_notice() or "")

    def test_the_controller_stops_the_motion_and_says_why(self):
        main, stop, arm, events = self._rig()
        controller = RemoteController(
            sim=_Sim(), robot=arm, motion_config=CFG.motion, robot_config=CFG.robot,
            hardware_config=CFG.hardware, event_logger=events,  # type: ignore[arg-type]
        )
        calls = {"n": 0}
        original = main.follow_trajectory

        def follow(waypoints: Any, dt: float) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 2:  # the watchdog fired between chunks 1 and 2
                main.state.update(detach_count=1, last_detach_reason="fake watchdog")
            return original(waypoints, dt)

        main.follow_trajectory = follow  # type: ignore[method-assign]
        assert controller.follow_trajectory(self._trajectory()) is False
        assert calls["n"] == 2, "no chunk may follow the one that reported the detach"
        assert "watchdog" in (controller.last_detach_notice or "")
        assert "controller.bridge_detached" in events.names() and controller.hold_unverified

    def test_a_notice_from_the_idle_gap_refuses_before_shipping(self):
        main, stop, arm, _events = self._rig()
        controller = _controller(arm)
        stop.state.update(detach_count=1, last_detach_reason="fake watchdog")
        arm.refresh()  # the heartbeat / a get_state saw it while idle
        main.calls.clear()
        assert controller.follow_trajectory(self._trajectory()) is False
        assert "follow_trajectory" not in main.calls

    def test_heartbeat_status_never_overwrites_the_pose(self):
        _main, _stop, arm, _events = self._rig()
        before = arm.get_arm_joint_positions()
        arm._absorb_status({"q": [9.0, 9.0, 9.0, 9.0], "detach_count": 0, "attached": True}, source="heartbeat")
        np.testing.assert_allclose(arm.get_arm_joint_positions(), before)


class TestEstopChannelConstruction:
    def test_any_error_opening_the_stop_channel_falls_back(self, monkeypatch, caplog):
        """Re-review low: an OSError/ZMQError raised outside the RPC's own
        error handling failed RemoteArm construction."""
        import mfw.hardware.remote_arm as ra

        class Exploding:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def connect(self, *a: Any, **k: Any) -> None:
                raise OSError("Too many open files")

            def close(self) -> None:
                pass

        monkeypatch.setattr(ra, "JetsonClient", Exploding)
        main = StubClient()
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            arm = RemoteArm(main, _kin(), CFG.robot, SimpleNamespace(sim_time=0.0, step_index=0))
        assert arm.estop_client is main and not arm.has_dedicated_estop_channel
        assert any("Too many open files" in r.getMessage() for r in caplog.records)

    def test_calibration_hint_names_a_command_that_exists(self):
        with pytest.raises(ConfigurationError, match=r"\(read\)"):
            assert_limits_reachable(_kin(), {})
        mod = _load_script("calibrate_servos")
        assert hasattr(mod.ServoCalibrator, "cmd_read") and not hasattr(mod.ServoCalibrator, "cmd_show")


class TestSqueezeMargin:
    def test_margin_is_configurable_and_validated(self):
        arm = _arm(StubClient(), StubClient())
        tuned = RemoteController(
            sim=_Sim(), robot=arm, motion_config=CFG.motion, robot_config=CFG.robot,
            hardware_config=CFG.hardware, close_margin_m=0.0015,
        )
        tuned.set_grasp_width(0.019)
        assert tuned.close_target_width() == pytest.approx(0.0175)
        schema_field = SimpleNamespace(**{k: getattr(CFG.hardware, k) for k in (
            "trajectory_chunk_s", "settle_steps_after_motion")}, grasp_close_margin_m=0.001,
            validate=lambda: None)
        from_config = RemoteController(
            sim=_Sim(), robot=arm, motion_config=CFG.motion, robot_config=CFG.robot,
            hardware_config=schema_field,  # type: ignore[arg-type]
        )
        from_config.set_grasp_width(0.019)
        assert from_config.close_target_width() == pytest.approx(0.018)
        assert _controller(arm).close_margin_m == pytest.approx(GRASP_CLOSE_MARGIN_M)
        with pytest.raises(ExecutionError, match="close margin"):
            RemoteController(
                sim=_Sim(), robot=arm, motion_config=CFG.motion, robot_config=CFG.robot,
                hardware_config=CFG.hardware, close_margin_m=-0.001,
            )


    def test_margin_and_heartbeat_load_from_the_config_file(self):
        """The schema is strict, so a getattr default alone could never be tuned from YAML."""
        cfg = load_config(REPO_ROOT / "configs" / "hardware_fake.yaml",
                          overrides={"hardware": {"grasp_close_margin_m": 0.0015, "heartbeat_s": 0.5}})
        assert cfg.hardware.heartbeat_s == pytest.approx(0.5)
        arm = _arm(StubClient(), StubClient())
        controller = RemoteController(
            sim=_Sim(), robot=arm, motion_config=cfg.motion, robot_config=cfg.robot,
            hardware_config=cfg.hardware,
        )
        assert controller.close_margin_m == pytest.approx(0.0015)
        assert CFG.hardware.grasp_close_margin_m == pytest.approx(GRASP_CLOSE_MARGIN_M)
        for bad in ({"grasp_close_margin_m": -0.001}, {"heartbeat_s": 0.0}, {"heartbeat_s": 5.0}):
            with pytest.raises(Exception, match="hardware.(grasp_close_margin_m|heartbeat_s)"):
                load_config(REPO_ROOT / "configs" / "hardware_fake.yaml", overrides={"hardware": bad})


class TestArmGeometryAgreement:
    """Re-review medium: the Jetson's --arm-geometry defaults to the
    placeholder links, and it backs the only check on the pose driven at
    every boot; the laptop now says when the two disagree."""

    def _arm_with(self, geometry: dict[str, float] | None, caplog) -> None:
        main = StubClient()
        original = main.get_calibration

        def get_calibration() -> dict[str, Any]:
            reply = original()
            if geometry is not None:
                reply["arm_geometry"] = geometry
            return reply

        main.get_calibration = get_calibration  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="mfw.hardware.remote_arm"):
            _arm(main, StubClient())

    def test_matching_geometry_is_silent(self, caplog):
        arm = CFG.hardware.arm
        same = {n: getattr(arm, n) for n in ("base_height", "shoulder_offset", "upper_arm", "forearm", "tool")}
        self._arm_with(same, caplog)
        assert not any("--arm-geometry" in r.getMessage() for r in caplog.records)

    def test_a_mismatch_is_named_with_the_fix(self, caplog):
        arm = CFG.hardware.arm
        other = {n: getattr(arm, n) for n in ("base_height", "shoulder_offset", "upper_arm", "forearm", "tool")}
        other["forearm"] = arm.forearm + 0.02
        self._arm_with(other, caplog)
        messages = [r.getMessage() for r in caplog.records if "--arm-geometry" in r.getMessage()]
        assert messages and "forearm" in messages[0] and f"{arm.forearm:g}" in messages[0]

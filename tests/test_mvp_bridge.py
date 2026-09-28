"""Hardware MVP, bridge stream: the servo bridge and the robot server's safety
guards, against fakes that model the physical failure each guard exists for.

* **The sketch itself runs.** ``servo_bridge.ino`` is compiled for the host
  with g++ against a mock ``Serial``/``Servo``/``millis`` and fed bytes, so
  the overlong-line fix, the watchdog and the parser are *executed*, not
  regex-matched. Skipped only when no C++ compiler is on PATH.
* **Late serial replies, twice.** A stub wire whose replies arrive late (and
  whose late replies still land in the buffer later, in order) proves that a
  second late reply no longer leaves the stream one frame behind -- and a
  control run of the old behaviour proves the stub can show the failure.
* **Placeholder calibration.** ``measured: false`` stops a real driver unless
  ``--allow-placeholder-calibration``, which keeps every pulse in a middle
  band and caps the speed; the band never reaches the file.
* **Home is not at a pulse extreme** and matches the laptop's mirror.
* **calibrate_table --touch heartbeats** at its prompt, measured against the
  server's 5 s host timeout on a fake clock (and without it, the arm relaxes).
* **Camera ownership / staleness**: ``get_frame`` carries the capture time and
  age of the LATEST frame and refuses a stale one from a camera that stalled.
* **PCA9685 backup driver** against a fake SMBus that models the chip's
  registers (PRESCALE is only writable asleep) and I2C NACKs.
"""

from __future__ import annotations

import builtins
import importlib.util
import logging
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jetson import robot_server as rs
from jetson.robot_server import (
    DETACH_ATTEMPTS,
    JOINT_NAMES,
    PLACEHOLDER_HOME_Q,
    PLACEHOLDER_MAX_JOINT_VELOCITY,
    PLACEHOLDER_PULSE_BAND_US,
    SKETCH_ARM_PULSE_US,
    SKETCH_PULSE_LIMITS_US,
    SKETCH_WATCHDOG_MS,
    WIRE_ORDER,
    ArmGeometry,
    CalibrationError,
    CameraGrabber,
    CameraSettings,
    DriverError,
    FakeDriver,
    Pca9685Driver,
    RobotServer,
    ServoCalibration,
    UnoSerialDriver,
    check_home_clearance,
    decode_command_frame,
    placeholder_safe_calibration,
    startup_calibration,
)
from mfw.hardware.jetson_client import JetsonClient, RpcError
from tests.test_hardware_bridge import _StubSerial

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
SKETCH_DIR = REPO_ROOT / "jetson" / "arduino" / "servo_bridge"
ROBOT_CONFIG_PATH = REPO_ROOT / "jetson" / "robot_config.yaml"
HARDWARE_YAML = REPO_ROOT / "configs" / "hardware.yaml"


class _FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _wait(predicate: Any, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_mvp_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _measured(cal: ServoCalibration) -> ServoCalibration:
    return ServoCalibration.from_dict({**cal.to_dict(), "measured": True})


# ======================================================================
# 1. servo_bridge.ino, compiled and executed on the host
# ======================================================================

_ARDUINO_H = r"""
#pragma once
#include <cstdint>
#include <deque>
#include <string>
#define F(x) (x)
extern unsigned long g_millis;
inline unsigned long millis() { return g_millis; }
struct MockSerial {
  std::deque<char> in;
  std::string out;
  void begin(unsigned long) {}
  int available() { return (int)in.size(); }
  int read() { if (in.empty()) return -1; char c = in.front(); in.pop_front(); return (unsigned char)c; }
  void print(const char *s) { out += s; }
  void print(char c) { out += c; }
  void print(int v) { out += std::to_string(v); }
  void println(const char *s) { out += s; out += "\r\n"; }
  void println(char c) { out += c; out += "\r\n"; }
  void println(int v) { out += std::to_string(v); out += "\r\n"; }
};
extern MockSerial Serial;
"""

_SERVO_H = r"""
#pragma once
struct Servo {
  int pin = -1; bool att = false; int us = 1500; int writes_before_attach = 0;
  void attach(int p, int, int) { pin = p; att = true; }
  void detach() { att = false; }
  void writeMicroseconds(int v) { us = v; if (!att) ++writes_before_attach; }
};
"""

_HARNESS_CPP = r"""
#include "Arduino.h"
unsigned long g_millis = 0;
MockSerial Serial;
#include "servo_bridge.ino"
#include <iostream>
static void flush_out() {
  size_t pos;
  while ((pos = Serial.out.find("\r\n")) != std::string::npos) {
    std::cout << "OUT " << Serial.out.substr(0, pos) << "\n";
    Serial.out.erase(0, pos + 2);
  }
}
static int hexval(char c) { return (c >= '0' && c <= '9') ? c - '0' : (c - 'a' + 10); }
int main() {
  setup();
  std::string cmd;
  while (std::cin >> cmd) {
    if (cmd == "W") {
      std::string hex; std::cin >> hex;
      for (size_t i = 0; i + 1 < hex.size(); i += 2) Serial.in.push_back((char)(hexval(hex[i]) * 16 + hexval(hex[i + 1])));
      loop();
    } else if (cmd == "A") {
      long ms; std::cin >> ms;
      for (long i = 0; i < ms; ++i) { ++g_millis; loop(); }
    } else if (cmd == "S") {
      std::cout << "STATE " << (attached ? 1 : 0) << " " << (have_position ? 1 : 0);
      for (int i = 0; i < N_SERVOS; ++i) std::cout << " " << current_us[i];
      for (int i = 0; i < N_SERVOS; ++i) std::cout << " " << (servos[i].att ? 1 : 0);
      for (int i = 0; i < N_SERVOS; ++i) std::cout << " " << servos[i].us;
      std::cout << "\n";
    }
    flush_out();
    std::cout << "END" << std::endl;
  }
  return 0;
}
"""


class _Sketch:
    """Runs one scripted session of the compiled sketch."""

    def __init__(self, exe: Path) -> None:
        self.exe = exe

    def run(self, *steps: tuple[str, Any]) -> list[list[str]]:
        """Steps: ``("w", bytes)`` write, ``("a", ms)`` let time pass, ``("s", None)``
        state. Returns the output lines of each step."""
        lines = []
        for kind, arg in steps:
            if kind == "w":
                lines.append(f"W {bytes(arg).hex()}")
            elif kind == "a":
                lines.append(f"A {int(arg)}")
            else:
                lines.append("S")
        proc = subprocess.run([str(self.exe)], input="\n".join(lines) + "\n", capture_output=True,
                              text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        blocks: list[list[str]] = [[]]
        for line in proc.stdout.splitlines():
            if line == "END":
                blocks.append([])
            else:
                blocks[-1].append(line)
        return blocks[:-1]

    @staticmethod
    def replies(block: list[str]) -> list[str]:
        return [line[4:] for line in block if line.startswith("OUT ")]

    @staticmethod
    def state(block: list[str]) -> dict[str, Any]:
        (line,) = [line for line in block if line.startswith("STATE ")]
        v = [int(x) for x in line.split()[1:]]
        return {"attached": bool(v[0]), "have_position": bool(v[1]), "current": tuple(v[2:7]),
                "servo_attached": tuple(bool(x) for x in v[7:12]), "servo_us": tuple(v[12:17])}


@pytest.fixture(scope="module")
def sketch(tmp_path_factory) -> _Sketch:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler on PATH: the sketch cannot be executed on the host")
    build = tmp_path_factory.mktemp("sketch")
    (build / "Arduino.h").write_text(_ARDUINO_H, encoding="utf-8")
    (build / "Servo.h").write_text(_SERVO_H, encoding="utf-8")
    (build / "harness.cpp").write_text(_HARNESS_CPP, encoding="utf-8")
    exe = build / ("sketch.exe" if sys.platform == "win32" else "sketch")
    proc = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", f"-I{build}", f"-I{SKETCH_DIR}", "-o", str(exe),
         str(build / "harness.cpp")],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"servo_bridge.ino does not compile:\n{proc.stderr}"
    return _Sketch(exe)


P_HOME = b"P1500,1500,1150,1600,1000\n"


class TestSketchOverlongLine:
    """Audit: ``servo_bridge.ino:316-320`` reset the buffer on an overlong line
    and then ran its TAIL as a frame -- one line, two replies, the host's
    reply stream shifted by one without noticing."""

    def test_an_overlong_line_gets_exactly_one_reply_and_runs_nothing(self, sketch):
        junk = b"X" * 64  # the 64th byte overflows LINE_MAX - 1 = 63
        tail = b"P1200,1200,1200,1200,1200"  # a VALID frame hiding behind the overflow
        out = sketch.run(("w", junk + tail + b"\n"), ("s", None))
        assert sketch.replies(out[0]) == ["ERR line too long"], out[0]
        st = sketch.state(out[1])
        assert not st["attached"] and not any(st["servo_attached"]), "the tail of an overlong line attached servos"
        assert st["current"] == (1500,) * 5 and not st["have_position"]

    def test_the_stream_stays_one_reply_per_line_around_it(self, sketch):
        long_p = b"P" + b",".join([b"1500"] * 20) + b"\r\n"  # CRLF: still one reply
        out = sketch.run(("w", b"?\n"), ("w", long_p), ("w", P_HOME), ("w", b"?\n"), ("s", None))
        assert [len(sketch.replies(b)) for b in out[:4]] == [1, 1, 1, 1], out
        assert sketch.replies(out[1]) == ["ERR line too long"]
        assert sketch.replies(out[2]) == ["OK"]
        assert sketch.replies(out[3]) == ["Q1500,1500,1150,1600,1000,1,1"]

    def test_a_line_that_just_fits_is_run(self, sketch):
        body = b"1500,1500,1500,1500,1500"
        frame = b"P" + b"0" * (63 - 1 - len(body)) + body  # leading zeros: still 1500
        assert len(frame) == 63  # LINE_MAX - 1: the longest line the sketch keeps
        out = sketch.run(("w", frame + b"\n"), ("w", frame + b"0\n"))
        assert sketch.replies(out[0]) == ["OK"]
        assert sketch.replies(out[1]) == ["ERR line too long"]

    def test_bytes_split_across_reads_are_one_line(self, sketch):
        out = sketch.run(("w", b"P1500,1500,"), ("w", b"1150,1600,1000"), ("w", b"\n"), ("s", None))
        assert sketch.replies(out[0]) == [] and sketch.replies(out[1]) == [] and sketch.replies(out[2]) == ["OK"]
        assert sketch.state(out[3])["current"] == (1500, 1500, 1150, 1600, 1000)


class TestSketchProtocolParity:
    """The host's parser (``decode_command_frame``) and the sketch agree on
    every frame, executed on both sides."""

    FRAMES = [
        b"P1500,1500,1500,1500,1500", b"P1500,1500,1500,1500", b"P1500,1500,1500,1500,1500,",
        b"P1500,1500,1500,1500,15a0", b"T1500,1500,1500,1500,1500,300", b"T1500,1500,1500,1500,1500",
        b"K", b"K1", b"S", b"S42", b"S4x", b"X1", b"?", b"D",
    ]

    def test_accept_and_reject_agree(self, sketch):
        steps = [("w", f + b"\n") for f in self.FRAMES]
        out = sketch.run(*steps)
        for frame, block in zip(self.FRAMES, out):
            (reply,) = sketch.replies(block)
            try:
                decoded = decode_command_frame(frame)
            except ValueError as exc:
                assert reply == str(exc), (frame, reply, str(exc))
                continue
            assert not reply.startswith("ERR"), (frame, reply)
            if decoded[0] == "S":
                assert reply == f"S{decoded[1]}"

    def test_constants_are_the_servers(self, sketch):
        text = (SKETCH_DIR / "servo_bridge.ino").read_text(encoding="utf-8")
        assert "LINE_MAX         = 64" in text and "discarding" in text
        assert SKETCH_PULSE_LIMITS_US[0] == SKETCH_ARM_PULSE_US
        # Clamp, executed: a pulse past the table is clamped silently and answered OK.
        out = sketch.run(("w", b"P100,3000,1500,1500,3000\n"), ("s", None))
        assert sketch.replies(out[0]) == ["OK"]
        assert sketch.state(out[1])["current"] == (600, 2400, 1500, 1500, 2100)


class TestSketchSafety:
    def test_first_frame_attaches_at_target_and_never_writes_1500_first(self, sketch):
        out = sketch.run(("w", P_HOME), ("s", None))
        st = sketch.state(out[1])
        assert st["attached"] and st["have_position"] and st["current"] == (1500, 1500, 1150, 1600, 1000)
        assert st["servo_us"] == st["current"]

    def test_watchdog_detaches_and_reattach_starts_from_the_last_pulse(self, sketch):
        out = sketch.run(
            ("w", P_HOME), ("a", SKETCH_WATCHDOG_MS - 50), ("s", None),   # still fed by the P
            ("a", 100), ("s", None),                                     # 550 ms of silence
            ("w", b"P1500,1500,1400,1600,1000\n"), ("s", None),          # re-attach
            ("a", 40), ("s", None),                                      # two ticks of slew
        )
        assert sketch.state(out[2])["attached"] is True
        assert sketch.state(out[4])["attached"] is False, "the 500 ms watchdog did not fire"
        st = sketch.state(out[6])
        assert st["attached"] and st["current"][2] == 1150, "a re-attach jumped to the new target"
        assert sketch.state(out[8])["current"][2] == 1150 + 2 * 20, "slew is 20 us per 20 ms tick"

    def test_k_keeps_it_attached_and_never_attaches(self, sketch):
        out = sketch.run(("w", b"K\n"), ("s", None), ("w", P_HOME),
                         *[step for _ in range(6) for step in (("a", 150), ("w", b"K\n"))], ("s", None))
        assert sketch.replies(out[0]) == ["OK"] and sketch.state(out[1])["attached"] is False
        assert sketch.state(out[-1])["attached"] is True

    def test_t_frame_interpolates(self, sketch):
        out = sketch.run(("w", P_HOME), ("w", b"T1500,1500,1350,1600,1000,200\n"), ("a", 100), ("s", None),
                         ("a", 120), ("s", None))
        mid = sketch.state(out[3])["current"][2]
        assert 1230 <= mid <= 1270, mid
        assert sketch.state(out[5])["current"][2] == 1350


class TestSerialSmokeAgainstTheSketch:
    """Review: jetson/serial_smoke.py (the S3 bench test) was checked only
    against a hand-written FakeUno, so protocol drift between it and the real
    sketch would pass. Its five exchanges now run against the compiled
    servo_bridge.ino (the bootloader window stays FakeUno's job)."""

    def _replies_from_sketch(self, sketch: _Sketch, pulses: tuple[int, ...]) -> list[str]:
        from jetson import serial_smoke as ss

        sent: list[bytes] = []

        class Recorder:
            def reset_input_buffer(self) -> None:
                pass

            def flush(self) -> None:
                pass

            def write(self, data: bytes) -> int:
                sent.append(bytes(data))
                return len(data)

            def readline(self) -> bytes:
                return b"\n"

        ss.run_smoke(Recorder(), pulses=pulses, sleep=lambda _s: None)  # the frames do not depend on replies
        assert len(sent) == 5
        steps: list[tuple[str, Any]] = []
        for frame in sent:
            steps += [("w", frame), ("a", 5)]
        blocks = sketch.run(*steps)
        replies = [sketch.replies(block) for block in blocks[0::2]]
        assert all(len(r) == 1 for r in replies), replies  # one line per request, as run_smoke reads
        return [r[0] for r in replies]

    @pytest.mark.parametrize("pulses", [None, (1400, 1600, 1200, 1500, 1100)], ids=["default", "custom"])
    def test_every_step_passes_on_the_real_sketch(self, sketch, pulses):
        from jetson import serial_smoke as ss

        pulses = tuple(ss.DEFAULT_PULSES) if pulses is None else pulses
        replies = iter(self._replies_from_sketch(sketch, pulses))

        class Replay:
            def reset_input_buffer(self) -> None:
                pass

            def flush(self) -> None:
                pass

            def write(self, data: bytes) -> int:
                return len(data)

            def readline(self) -> bytes:
                return (next(replies) + "\r\n").encode("ascii")

        steps = ss.run_smoke(Replay(), pulses=pulses, sleep=lambda _s: None)
        assert [ok for _sent, _reply, ok, _expect in steps] == [True] * 5, ss.format_steps(steps)
        assert steps[2][1].startswith("Q" + ",".join(map(str, pulses)))


# ======================================================================
# 2. UnoSerialDriver: late replies, twice
# ======================================================================


@pytest.fixture
def stub_serial(monkeypatch):
    module = types.ModuleType("serial")
    module.Serial = _StubSerial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", module)
    _StubSerial.instances.clear()
    _StubSerial.watchdog_s = None
    _StubSerial.virtual = True  # lateness in ms, exact whatever the machine's load
    yield _StubSerial
    _StubSerial.virtual = False


def _uno(cal: ServoCalibration | None = None) -> UnoSerialDriver:
    return UnoSerialDriver(cal or _measured(ServoCalibration.default()), port="COM9", reset_wait_s=0.0,
                           keepalive_s=None)


def _in_step(ser: _StubSerial, since: int) -> list[tuple[int, int | None]]:
    """Reads since ``since`` that returned another frame's reply as the reply
    to the frame just written. Reads made while waiting for a sync echo are
    excluded: skipping stale lines there is the sync's job."""
    return [
        (asked, got) for asked, got in ser.reads[since:]
        if got is not None and asked != got and not ser.frames[asked].startswith(b"S")
    ]


class TestUnoLateTwice:
    """Audit Missed-7: the reply to the RESENT frame was late too; the old
    exchange returned the empty reply, the late OK then answered the next
    frame, and every later read was one frame behind for good."""

    LATE = rs.DEFAULT_REPLY_TIMEOUT_S + 0.004

    def test_second_late_reply_is_skipped_by_a_sync_before_the_next_frame(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        home = np.asarray(driver.calibration.home_q)
        driver.write(home)
        ser.actions["P"] = [self.LATE, self.LATE]
        with pytest.raises(DriverError, match="no reply"):
            driver.write(home + 0.02)
        first = len(ser.reads)
        for _ in range(6):
            driver._send_keepalive()
            assert driver.query()[1] is True
        assert _in_step(ser, first) == [], "a late reply was read as the next frame's reply"
        assert driver.stale_replies_skipped == 2  # the first late OK, then the resend's
        kinds = [decode_command_frame(f)[0] for f in ser.frames]
        # P, S (sync after the first timeout), P (resend), then S BEFORE the next K.
        assert kinds[:6] == ["P", "P", "S", "P", "S", "K"], kinds
        assert driver.consecutive_timeouts == 0

    def test_control_the_old_behaviour_is_one_frame_behind(self, stub_serial):
        """Same wire, with the new pre-sync switched off: the stub must show
        the failure, or the test above proves nothing."""
        driver = _uno()
        ser = stub_serial.instances[-1]
        home = np.asarray(driver.calibration.home_q)
        driver.write(home)
        ser.actions["P"] = [self.LATE, self.LATE]
        with pytest.raises(DriverError):
            driver.write(home + 0.02)
        driver._out_of_step = False  # what the old _exchange left behind
        first = len(ser.reads)
        driver._send_keepalive()  # reads the resend's late OK as its own
        assert _in_step(ser, first), "the stub did not reproduce the one-frame-behind stream"

    def test_a_late_sync_echo_is_skipped_too(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        ser.actions["K"] = [self.LATE]
        ser.actions["S"] = [0.15]  # the sync echo itself misses its read window
        with pytest.raises(DriverError):
            driver._send_keepalive()
        first = len(ser.reads)
        # Until the late echo has arrived the stream is not provably in step:
        # exchanges fail (nothing misattributed), then recover by themselves.
        failures = 0
        for _ in range(3):
            try:
                assert driver.query()[1] is True
                break
            except DriverError:
                failures += 1
        assert failures < 3, "never recovered after the late echo arrived"
        for _ in range(4):
            driver._send_keepalive()
            assert driver.query()[1] is True
        assert _in_step(ser, first) == [], "a stale line was taken as a reply"
        assert driver.attached and not driver.hold_detached

    def test_noise_is_not_an_in_step_reply(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        ser.inject_stale(b"\x7f\x00garbage\n")  # line noise already in the buffer
        driver._send_keepalive()
        assert driver.resyncs == 1 and driver.stale_replies_skipped >= 1

    def test_a_dead_link_still_sends_every_detach(self, stub_serial):
        driver = _uno()
        ser = stub_serial.instances[-1]
        driver.write(np.asarray(driver.calibration.home_q))
        ser.reply_override = b""  # nothing ever answers
        with pytest.raises(DriverError, match="did not confirm the detach"):
            driver.detach("estop")
        assert [f[:1] for f in ser.frames].count(b"D") == DETACH_ATTEMPTS
        assert not ser.attached and driver.hold_detached


# ======================================================================
# 3. placeholder calibration guard
# ======================================================================


class TestPlaceholderGuard:
    def test_shipped_file_is_marked_unmeasured(self):
        assert ServoCalibration.load(ROBOT_CONFIG_PATH).measured is False
        assert ServoCalibration.default().measured is False
        assert ServoCalibration.from_dict({}).measured is False  # missing means false
        with pytest.raises(CalibrationError, match="true or false"):
            ServoCalibration.from_dict({"measured": "no"})
        assert ServoCalibration.from_dict(_measured(ServoCalibration.default()).to_dict()).measured

    @pytest.mark.parametrize("driver", ["uno", "pca9685"])
    def test_a_real_driver_refuses_an_unmeasured_file(self, driver):
        with pytest.raises(CalibrationError, match="measured: false.*--allow-placeholder-calibration"):
            startup_calibration(driver, ROBOT_CONFIG_PATH, ArmGeometry())

    def test_the_fake_driver_does_not(self):
        file_cal, active, band = startup_calibration("fake", ROBOT_CONFIG_PATH, ArmGeometry())
        assert band is None and active is file_cal

    def test_missing_file_is_refused_for_real_servos(self, tmp_path):
        with pytest.raises(CalibrationError, match="not found"):
            startup_calibration("uno", tmp_path / "nope.yaml", ArmGeometry(), allow_placeholder=True)

    def test_the_flag_bands_pulses_and_caps_speed(self):
        file_cal, active, band = startup_calibration("uno", ROBOT_CONFIG_PATH, ArmGeometry(), allow_placeholder=True)
        assert band == PLACEHOLDER_PULSE_BAND_US and file_cal.measured is False
        lo, hi = band
        for name in JOINT_NAMES:
            j = active.joints[name]
            for limit in (j.limit_lower_rad, j.limit_upper_rad):
                assert lo - 0.5 <= active.rad_to_us(name, limit) <= hi + 0.5, name
            assert j.pulse_min_us == file_cal.joints[name].pulse_min_us  # the MAP is unchanged
        assert active.max_joint_velocity <= PLACEHOLDER_MAX_JOINT_VELOCITY
        assert active.reattach_s >= rs.PLACEHOLDER_MIN_REATTACH_S
        assert active.home_move_s >= rs.PLACEHOLDER_MIN_HOME_MOVE_S
        g = active.gripper
        assert lo <= g.pulse_open_us <= hi and lo <= g.pulse_closed_us <= hi

    def test_a_measured_file_ignores_the_flag(self, tmp_path):
        path = tmp_path / "robot_config.yaml"
        _measured(ServoCalibration.load(ROBOT_CONFIG_PATH)).save(path)
        file_cal, active, band = startup_calibration("uno", path, ArmGeometry(), allow_placeholder=True)
        assert band is None and active == file_cal and active.measured

    def test_band_refusals(self):
        cal = ServoCalibration.load(ROBOT_CONFIG_PATH)
        with pytest.raises(CalibrationError, match="Uno's clamp"):
            placeholder_safe_calibration(cal, (500, 2500))
        with pytest.raises(CalibrationError, match="empty"):
            placeholder_safe_calibration(cal, (1500, 1500))
        with pytest.raises(CalibrationError, match="home_q.*placeholder pulse band"):
            placeholder_safe_calibration(cal, (1300, 1700))  # elbow home is 1150 us
        wide = placeholder_safe_calibration(cal, SKETCH_ARM_PULSE_US)  # S5 end-stop search
        assert wide.joints["elbow_pitch"].limit_lower_rad == pytest.approx(
            cal.joints["elbow_pitch"].limit_lower_rad, abs=1e-4)

    def test_gripper_points_move_along_their_own_map(self):
        raw = ServoCalibration.load(ROBOT_CONFIG_PATH).to_dict()
        raw["gripper"] = {"pulse_open_us": 900, "pulse_closed_us": 2100, "width_open_m": 0.048, "width_closed_m": 0.0}
        cal = ServoCalibration.from_dict(raw)
        safe = placeholder_safe_calibration(cal)
        assert (safe.gripper.pulse_open_us, safe.gripper.pulse_closed_us) == (1000, 2000)
        for us in (1000.0, 1500.0, 2000.0):  # the same pulse still means the same opening
            assert safe.us_to_width(us) == pytest.approx(cal.us_to_width(us), abs=1e-9)

    def test_main_refuses_before_touching_the_serial_port(self, caplog, monkeypatch):
        opened: list[Any] = []
        module = types.ModuleType("serial")
        module.Serial = lambda *a, **k: opened.append(a) or pytest.fail("opened the port")  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "serial", module)
        with caplog.at_level(logging.ERROR, logger="robot_server"):
            rc = rs.main(["--driver", "uno", "--config", str(ROBOT_CONFIG_PATH), "--no-camera", "--port", "5571"])
        assert rc == 2 and not opened
        assert any("measured: false" in r.getMessage() for r in caplog.records)
        assert rs.main(["--driver", "uno", "--placeholder-band-us", "2000,1000", "--no-camera"]) == 2

    def test_server_keeps_every_pulse_in_the_band(self, tmp_path):
        file_cal = ServoCalibration.load(ROBOT_CONFIG_PATH)
        active = placeholder_safe_calibration(file_cal, geometry=ArmGeometry())
        driver = FakeDriver(active)
        path = tmp_path / "robot_config.yaml"
        file_cal.save(path)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=active, calibration_path=path,
                             follower_sleep=lambda _s: None, file_calibration=file_cal,
                             placeholder_band_us=PLACEHOLDER_PULSE_BAND_US)
        server._handle("home", {})
        home = np.asarray(active.home_q)
        far = np.array([1.4, 1.4, -1.4, 1.4])  # reachable on the file's map, far outside the band
        reply = server._handle("follow_trajectory", {"waypoints": np.stack([home, far]), "dt": 0.05})
        assert reply["clamped"] is True and reply["completed"] is True
        lo, hi = PLACEHOLDER_PULSE_BAND_US
        pulses = [p for kind, p in driver.commands if kind == "P"]
        assert pulses and all(lo <= us <= hi for frame in pulses for us in frame[:4])
        assert reply["elapsed_s"] >= 0.0
        _s, _t, _c, stretch = server.follower.plan(np.stack([home, far]), 0.05)
        assert stretch > 1.0, "the speed cap must stretch a fast trajectory"
        state = server.state()
        assert state["calibration_measured"] is False and state["placeholder_band_us"] == [lo, hi]
        # get_calibration: the enforced copy AND the file copy.
        cal_reply = server._handle("get_calibration", {})
        assert cal_reply["file_calibration"] == file_cal.to_dict()
        assert cal_reply["calibration"]["max_joint_velocity"] <= PLACEHOLDER_MAX_JOINT_VELOCITY
        # A pushed calibration keeps the band for this run and never writes it.
        pushed = {**file_cal.to_dict(), "measured": True}
        assert server._handle("set_calibration", {"calibration": pushed})["placeholder_band_us"] == [lo, hi]
        on_disk = ServoCalibration.load(path)
        assert on_disk.measured is True and on_disk.max_joint_velocity == file_cal.max_joint_velocity
        assert on_disk.joints == file_cal.joints
        assert server.calibration.max_joint_velocity <= PLACEHOLDER_MAX_JOINT_VELOCITY
        server.stop()

    def test_repl_edits_the_file_copy_and_refuses_pulses_outside_the_band(self, tmp_path):
        file_cal = ServoCalibration.load(ROBOT_CONFIG_PATH)
        active = placeholder_safe_calibration(file_cal, geometry=ArmGeometry())
        path = tmp_path / "robot_config.yaml"
        file_cal.save(path)
        server = RobotServer("127.0.0.1", 0, FakeDriver(active), camera=None, calibration=active,
                             calibration_path=path, follower_sleep=lambda _s: None,
                             file_calibration=file_cal, placeholder_band_us=PLACEHOLDER_PULSE_BAND_US)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port, request_timeout_s=5.0)
        client.connect()
        try:
            client.home()
            mod = _load_script("calibrate_servos")
            lines: list[str] = []
            cal = mod.ServoCalibrator(client, out=lines.append, confirm=lambda _p: True)
            assert any("measured: false" in line for line in lines)
            assert any("PLACEHOLDER" in line and "1000..2000" in line for line in lines)
            with pytest.raises(ValueError, match="placeholder band.*600,2400"):
                cal.run("pulse base_yaw 2300")
            cal.run("pulse base_yaw 1900")
            cal.run("measured yes")
            cal.run("save")
            on_disk = ServoCalibration.load(path)
            assert on_disk.measured is True
            assert on_disk.joints["base_yaw"].limit_upper_rad == file_cal.joints["base_yaw"].limit_upper_rad, \
                "the bench band leaked into the file"
            assert cal.server_cal.max_joint_velocity <= PLACEHOLDER_MAX_JOINT_VELOCITY
            with pytest.raises(ValueError, match="usage"):
                cal.run("measured maybe")
        finally:
            client.close()
            server.stop()


# ======================================================================
# 4. home_q is not at a pulse extreme
# ======================================================================


class TestHomePose:
    MARGIN_US = 300

    def test_shipped_home_is_well_inside_the_pulse_map_and_the_band(self):
        cal = ServoCalibration.load(ROBOT_CONFIG_PATH, ArmGeometry())
        assert tuple(cal.home_q) == PLACEHOLDER_HOME_Q == tuple(ServoCalibration.default().home_q)
        lo_band, hi_band = PLACEHOLDER_PULSE_BAND_US
        for name, q in zip(JOINT_NAMES, cal.home_q):
            j = cal.joints[name]
            us = cal.rad_to_us(name, q)
            assert j.pulse_min_us + self.MARGIN_US <= us <= j.pulse_max_us - self.MARGIN_US, (name, us)
            assert lo_band <= us <= hi_band, (name, us)
        check_home_clearance(cal.home_q, ArmGeometry())
        # The audit's case: the elbow used to sit exactly on the 600 us extreme.
        assert round(cal.rad_to_us("elbow_pitch", cal.home_q[2])) != SKETCH_ARM_PULSE_US[0]

    def test_laptop_mirror_is_equal(self):
        from mfw.config.schema import load_config

        cfg = load_config(HARDWARE_YAML)
        np.testing.assert_allclose(cfg.robot.home_joint_positions, PLACEHOLDER_HOME_Q, atol=1e-9)
        for q, lo, hi in zip(PLACEHOLDER_HOME_Q, cfg.hardware.arm.joint_lower, cfg.hardware.arm.joint_upper):
            assert lo <= q <= hi


# ======================================================================
# 5. calibrate_table --touch heartbeats at its prompt
# ======================================================================


class TestTouchProbeHeartbeat:
    def _serve(self, clock: _FakeClock):
        cal = _measured(ServoCalibration.default())
        driver = FakeDriver(cal)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal,
                             follower_sleep=lambda _s: None, clock=clock)
        _, port = server.serve_in_thread(port=0)
        homer = JetsonClient("127.0.0.1", port)
        homer.connect()
        homer.home()
        homer.close()
        return server, driver, port

    def _prompt(self, clock: _FakeClock, probe: Any, seconds: float, steps: int):
        """A person thinking at ``touch>``: fake time passes in steps; with a
        heartbeat each step waits for a beat, as real time would give it."""

        def fake_input(_prompt: str = "") -> str:
            for _ in range(steps):
                beats = probe.heartbeat.beats if probe.heartbeat is not None else None
                clock.t += seconds / steps
                if beats is not None:
                    assert _wait(lambda: probe.heartbeat.beats > beats), "no heartbeat while at the prompt"
                else:
                    time.sleep(0.15)  # let the server's liveness loop look at the silence
            return "ok"

        return fake_input

    def test_the_arm_stays_held_while_the_student_thinks(self, monkeypatch):
        ct = _load_script("calibrate_table")
        clock = _FakeClock()
        server, driver, port = self._serve(clock)
        probe = ct.TouchProbe(f"127.0.0.1:{port}", HARDWARE_YAML, heartbeat_s=0.05)
        try:
            assert probe.heartbeat is not None and probe.heartbeat.running
            monkeypatch.setattr(builtins, "input", self._prompt(clock, probe, 12.0, 6))
            xy = probe.measure(1, (320.0, 240.0))
            assert xy is not None
            assert driver.attached and driver.detach_count == 0, driver.last_detach_reason
            assert server.host_lost is False
        finally:
            probe.close()
        assert probe.heartbeat is None
        clock.t += server.host_timeout_s + 0.5
        assert _wait(lambda: not driver.attached), "a closed probe must let the server relax the arm"
        server.stop()

    def test_control_without_a_heartbeat_the_host_timeout_detaches(self, monkeypatch):
        ct = _load_script("calibrate_table")
        clock = _FakeClock()
        server, driver, port = self._serve(clock)
        probe = ct.TouchProbe(f"127.0.0.1:{port}", HARDWARE_YAML, heartbeat_s=0)
        try:
            assert probe.heartbeat is None
            monkeypatch.setattr(builtins, "input", self._prompt(clock, probe, 12.0, 6))
            probe.measure(1, (320.0, 240.0))
            assert _wait(lambda: not driver.attached), "the fake server's host timeout never fired"
            assert driver.last_detach_reason.startswith("host timeout")
        finally:
            probe.close()
            server.stop()


    def _scripted(self, clock: _FakeClock, lines: list[tuple[float, str]]):
        """``input()`` that lets ``seconds`` of fake time pass (in real-time
        steps the server's liveness loop can see) before answering ``line``."""
        script = list(lines)

        def fake_input(_prompt: str = "") -> str:
            seconds, line = script.pop(0)
            for _ in range(6):
                clock.t += seconds / 6
                time.sleep(0.12)
            return line

        return fake_input

    def test_a_jaw_at_the_table_is_not_held_and_the_host_timeout_relaxes_it(self, monkeypatch):
        """Review finding 2: the model's z = 0 may be under the real table; a
        held jaw there stalls the shoulder/elbow. Below 5 mm the probe stops
        beating, the server's host timeout detaches, and the next move up
        re-attaches (slowly, server side) and holds again."""
        ct = _load_script("calibrate_table")
        clock = _FakeClock()
        widths: list[float] = []
        cal = _measured(ServoCalibration.default())
        driver = FakeDriver(cal, on_command=lambda _q, w: widths.append(float(w)))
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal,
                             follower_sleep=lambda _s: None, clock=clock)
        _, port = server.serve_in_thread(port=0)
        homer = JetsonClient("127.0.0.1", port)
        homer.connect()
        homer.home()
        homer.close()
        probe = ct.TouchProbe(f"127.0.0.1:{port}", HARDWARE_YAML, heartbeat_s=0.05, clock=clock)
        try:
            assert probe.heartbeat is not None and probe.heartbeat.running  # home is well clear
            assert widths and widths[-1] == pytest.approx(ct.TOUCH_JAW_WIDTH_M)  # never the closed end point
            assert ct.TOUCH_JAW_WIDTH_M > 0.0
            assert probe.move_to((0.18, 0.0, 0.03)) is True and probe.heartbeat.running
            assert probe.move_to((0.18, 0.0, 0.001)) is True
            assert probe.heartbeat is None and "of the model table" in probe.hold_stopped
            monkeypatch.setattr(builtins, "input", self._scripted(clock, [(12.0, "ok")]))
            assert probe.measure(1, (320.0, 240.0)) == pytest.approx((0.18, 0.0), abs=2e-3)
            assert _wait(lambda: not driver.attached), "a jaw at the table was still held after the host timeout"
            assert driver.last_detach_reason.startswith("host timeout")
            assert probe.move_to(probe.tcp + [0.0, 0.0, 0.01], nudge=True) is True  # back up: re-attach, hold
            assert driver.attached and probe.heartbeat is not None and probe.heartbeat.running
        finally:
            probe.close()
            server.stop()

    def test_an_idle_prompt_stops_holding_after_the_bound(self, monkeypatch):
        """A hung input() or a student who walked away: the hold ends after max_hold_s."""
        ct = _load_script("calibrate_table")
        clock = _FakeClock()
        server, driver, port = self._serve(clock)
        probe = ct.TouchProbe(f"127.0.0.1:{port}", HARDWARE_YAML, heartbeat_s=0.05, max_hold_s=3.0, clock=clock)
        try:
            assert probe.heartbeat is not None and probe.heartbeat.running
            monkeypatch.setattr(builtins, "input", self._scripted(clock, [(12.0, "ok")]))
            probe.measure(1, (320.0, 240.0))
            assert not probe.heartbeat.running and "idle" in probe.hold_stopped
            assert _wait(lambda: not driver.attached), "the idle bound never let the server relax the arm"
            assert driver.last_detach_reason.startswith("host timeout")
        finally:
            probe.close()
            server.stop()


# ======================================================================
# 6. camera: one owner, latest frame, capture time, stale refusal
# ======================================================================


class _ScriptedCapture:
    """A webcam: each read() returns the next scripted frame, ``None`` = a
    failed read (USB hiccup), and once the script is exhausted it behaves
    as unplugged (every read fails)."""

    def __init__(self, script: list[int | None]) -> None:
        self.script = list(script)
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.reads += 1
        if not self.script:
            return False, None
        value = self.script.pop(0)
        if value is None:
            return False, None
        return True, np.full((48, 64, 3), value, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _grabber(script: list[int | None], clock: _FakeClock) -> tuple[CameraGrabber, _ScriptedCapture]:
    cap = _ScriptedCapture(script)
    grabber = CameraGrabber(CameraSettings(device="0", width=64, height=48), capture_factory=lambda _d: cap,
                            clock=clock)
    return grabber, cap


def _pixel(reply: dict[str, Any]) -> int:
    import cv2

    img = cv2.imdecode(np.frombuffer(reply["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
    return int(np.median(img))


class TestCameraFrames:
    def test_latest_frame_with_capture_time_age_and_seq(self):
        clock = _FakeClock()
        grabber, _cap = _grabber([10, None, 60, None, None, 200], clock)
        grabber._cap = _cap
        for _ in range(6):
            grabber.grab_once()
            clock.t += 0.033
        reply = grabber.get_frame()
        assert reply["seq"] == 3, "only captured frames count; failed reads do not"
        assert abs(_pixel(reply) - 200) <= 3, "not the latest frame"
        assert reply["age_s"] == pytest.approx(0.033, abs=1e-9)
        assert reply["t_server"] >= reply["t_capture"]

    def test_a_stalled_camera_is_refused_not_served_forever(self):
        clock = _FakeClock()
        grabber, cap = _grabber([50], clock)
        grabber._cap = cap
        assert grabber.grab_once()
        for _ in range(40):  # unplugged: every read fails from now on
            assert not grabber.grab_once()
        clock.t += rs.DEFAULT_MAX_FRAME_AGE_S + 0.2
        with pytest.raises(RuntimeError, match=r"stale camera frame.*40 failed reads"):
            grabber.get_frame()
        assert grabber.get_frame(max_age_s=5.0)["seq"] == 1  # a caller may accept older
        assert grabber.get_frame(max_age_s=0)["seq"] == 1  # 0 disables the check

    def test_no_frame_yet_says_so(self):
        grabber, cap = _grabber([None, None], _FakeClock())
        grabber._cap = cap
        grabber.grab_once()
        with pytest.raises(RuntimeError, match="no camera frame captured yet"):
            grabber.get_frame()

    def test_thread_keeps_only_the_latest_and_the_endpoint_carries_it(self):
        clock = _FakeClock()
        grabber, cap = _grabber(list(range(1, 200)), clock)
        grabber.start()
        cal = ServoCalibration.default()
        server = RobotServer("127.0.0.1", 0, FakeDriver(cal), camera=grabber, calibration=cal,
                             follower_sleep=lambda _s: None, clock=clock)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port)
        client.connect()
        try:
            assert _wait(lambda: not cap.script)  # the thread drained the whole queue
            reply = client.get_frame()
            assert reply["seq"] == 199 and abs(_pixel(reply) - 199) <= 3
            assert "age_s" in reply and "t_capture" in reply
            assert client.ping()["camera"] is True
            clock.t += 2.0  # unplugged for two seconds
            with pytest.raises(RpcError, match="stale camera frame"):
                client.get_frame()
            assert client.rpc.call("get_frame", {"max_age_s": 5.0})["seq"] == 199
        finally:
            client.close()
            server.stop()
        assert cap.released

    def test_camera_is_on_by_default_and_require_camera_is_enforced(self, monkeypatch, tmp_path):
        assert rs.main(["--driver", "fake", "--no-camera", "--require-camera"]) == 2
        # A real (pca9685) server whose camera cannot open exits with --require-camera.
        _install_fake_smbus(monkeypatch)
        path = tmp_path / "robot_config.yaml"
        raw = _measured(ServoCalibration.default()).to_dict()
        raw["camera"]["device"] = "no_such_camera_device_mvp"
        ServoCalibration.from_dict(raw).save(path)
        assert rs.main(["--driver", "pca9685", "--config", str(path), "--require-camera", "--port", "5572"]) == 2
        bus = _FakeSMBus.instances[-1]
        assert bus.closed and all(bus.full_off[ch] for ch in range(5)), "exit must leave every servo full-off"


# ======================================================================
# 7. torque-on: the docstring and the REPL say it re-energises at the last pulse
# ======================================================================


class TestTorqueOnWording:
    def test_docstring_no_longer_promises_nothing_moves(self):
        doc = RobotServer._ep_set_torque.__doc__ or ""
        assert "moves nothing" not in doc and "LAST PULSED" in doc and "JUMPS" in doc

    def test_repl_warns_about_the_jump_before_energising(self, tmp_path):
        cal = _measured(ServoCalibration.default())
        driver = FakeDriver(cal)
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=cal,
                             follower_sleep=lambda _s: None)
        _, port = server.serve_in_thread(port=0)
        client = JetsonClient("127.0.0.1", port)
        client.connect()
        try:
            client.home()
            client.set_torque(False)
            mod = _load_script("calibrate_servos")
            lines: list[str] = []
            asked: list[str] = []

            def confirm(prompt: str) -> bool:
                asked.append("\n".join(lines))  # what had been printed when the question came
                return False

            calibrator = mod.ServoCalibrator(client, out=lines.append, confirm=confirm)
            with pytest.raises(ValueError, match="cancelled"):
                calibrator.run("torque on")
            assert asked and "JUMPS back" in asked[0] and "TAKE YOUR HAND OFF" in asked[0]
            assert "holds still" not in asked[0]
            assert driver.attached is False, "energised before the answer"
        finally:
            client.close()
            server.stop()


# ======================================================================
# 8. PCA9685 backup driver against a fake SMBus
# ======================================================================


class _FakeSMBus:
    """The PCA9685 as its registers see it.

    * PRESCALE is only writable while MODE1.SLEEP is set (datasheet 7.3.5);
      a write while awake is silently ignored, as on the chip.
    * LEDn_ON/OFF are four registers from ``0x06 + 4n``; bit 4 of OFF_H is
      FULL-OFF. The pulse a channel outputs follows from OFF and PRESCALE.
    * ``nack`` makes the next that-many writes fail with ``OSError(121)``,
      what Linux i2c-dev raises when the chip does not acknowledge.
    """

    instances: list["_FakeSMBus"] = []

    def __init__(self, bus: int) -> None:
        self.bus = bus
        self.mode1 = 0x11
        self.prescale = 30  # power-on default (200 Hz)
        self.ignored_prescale_writes = 0
        self.off = {ch: 0 for ch in range(16)}
        self.full_off = {ch: True for ch in range(16)}
        self.log: list[tuple] = []
        self.nack = 0
        self.closed = False
        _FakeSMBus.instances.append(self)

    def _maybe_nack(self) -> None:
        if self.closed:
            raise OSError(9, "Bad file descriptor")
        if self.nack > 0:
            self.nack -= 1
            raise OSError(121, "Remote I/O error")

    def write_byte_data(self, addr: int, reg: int, value: int) -> None:
        self._maybe_nack()
        assert addr == 0x40
        self.log.append(("byte", reg, value))
        if reg == 0x00:
            self.mode1 = value
        elif reg == 0xFE:
            if self.mode1 & 0x10:
                self.prescale = value
            else:
                self.ignored_prescale_writes += 1

    def write_i2c_block_data(self, addr: int, reg: int, data: list[int]) -> None:
        self._maybe_nack()
        assert addr == 0x40 and (reg - 0x06) % 4 == 0 and len(data) == 4
        ch = (reg - 0x06) // 4
        self.log.append(("block", ch, tuple(data)))
        self.full_off[ch] = bool(data[3] & 0x10)
        self.off[ch] = data[2] | ((data[3] & 0x0F) << 8)

    def close(self) -> None:
        self.closed = True

    # -- what the servos see ---------------------------------------------
    def frequency_hz(self) -> float:
        return 25e6 / (4096.0 * (self.prescale + 1))

    def pulse_us(self, ch: int) -> float | None:
        if self.full_off[ch] or self.mode1 & 0x10:
            return None
        return self.off[ch] / 4096.0 * 1e6 / self.frequency_hz()


def _install_fake_smbus(monkeypatch) -> None:
    module = types.ModuleType("smbus2")
    module.SMBus = _FakeSMBus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smbus2", module)
    _FakeSMBus.instances.clear()


@pytest.fixture
def pca(monkeypatch):
    _install_fake_smbus(monkeypatch)
    cal = _measured(ServoCalibration.default())
    driver = Pca9685Driver(cal, bus=7)
    yield driver, _FakeSMBus.instances[-1]
    driver.close()


class TestPca9685:
    def test_init_sets_50_hz_through_sleep(self, pca):
        driver, bus = pca
        assert bus.log[:4] == [("byte", 0x00, 0x10), ("byte", 0xFE, 121), ("byte", 0x00, 0x00), ("byte", 0x00, 0xA0)]
        assert bus.ignored_prescale_writes == 0 and bus.prescale == 121
        assert bus.frequency_hz() == pytest.approx(50.0, abs=0.1)
        assert all(bus.pulse_us(ch) is None for ch in range(5)), "nothing pulses before the first frame"
        assert driver.has_watchdog is False and not driver.keepalive_running

    def test_pulses_land_on_their_channels(self, pca):
        driver, bus = pca
        driver.reattach(duration_s=0.0)  # first attach: one frame at the stored (home) pose
        home_us = driver.pulses()
        for ch, us in enumerate(home_us):
            assert bus.pulse_us(ch) == pytest.approx(us, abs=5.0), ch  # 4.9 us per count
        driver.write(np.array([0.3, 0.2, -0.4, 0.1]))
        want = driver.pulses()
        for ch in range(4):
            assert bus.off[ch] == round(want[ch] * 4096 / 20000)

    def test_channel_map(self, monkeypatch):
        _install_fake_smbus(monkeypatch)
        raw = _measured(ServoCalibration.default()).to_dict()
        raw["channels"] = {"base_yaw": 8, "shoulder_pitch": 9, "elbow_pitch": 10, "wrist_pitch": 11, "gripper": 15}
        driver = Pca9685Driver(ServoCalibration.from_dict(raw))
        bus = _FakeSMBus.instances[-1]
        driver.reattach(duration_s=0.0)
        assert [bus.pulse_us(ch) is not None for ch in (8, 9, 10, 11, 15)] == [True] * 5
        assert all(bus.pulse_us(ch) is None for ch in range(8))
        driver.close()
        assert all(bus.full_off[ch] for ch in (8, 9, 10, 11, 15))

    def test_detach_is_full_off_on_every_servo_channel(self, pca):
        driver, bus = pca
        driver.reattach(duration_s=0.0)
        driver.detach("estop")
        assert all(bus.full_off[ch] and bus.pulse_us(ch) is None for ch in range(5))

    def test_a_nacked_full_off_is_retried(self, pca):
        driver, bus = pca
        driver.reattach(duration_s=0.0)
        bus.nack = 2  # the first two channel writes of the detach are not acknowledged
        driver.detach("estop")
        assert all(bus.full_off[ch] for ch in range(5))

    def test_a_detach_that_never_lands_says_cut_the_power(self, pca):
        driver, bus = pca
        driver.reattach(duration_s=0.0)
        bus.nack = 10_000
        with pytest.raises(DriverError, match="NO watchdog.*CUT THE"):
            driver.detach("estop")
        assert driver.hold_detached

    def test_the_host_timeout_is_the_only_watchdog(self, pca):
        """No hardware watchdog: after a laptop dies, only the server's host
        timeout stops the pulses -- by writing full-off."""
        driver, bus = pca
        clock = _FakeClock()
        server = RobotServer("127.0.0.1", 0, driver, camera=None, calibration=driver.calibration,
                             follower_sleep=lambda _s: None, clock=clock)
        server._handle("home", {})
        assert all(bus.pulse_us(ch) is not None for ch in range(5))
        clock.t += server.host_timeout_s - 0.1
        assert server.check_host_liveness() is False
        assert all(bus.pulse_us(ch) is not None for ch in range(5)), "it keeps pulsing on its own"
        clock.t += 0.2
        assert server.check_host_liveness() is True
        assert all(bus.full_off[ch] for ch in range(5))
        server.stop()

    def test_close_writes_full_off_and_releases_the_bus(self, pca):
        driver, bus = pca
        driver.reattach(duration_s=0.0)
        driver.close()
        assert bus.closed and all(bus.full_off[ch] for ch in range(5))
        driver.close()  # idempotent

    def test_main_refuses_a_disabled_host_timeout(self, monkeypatch, tmp_path, caplog):
        _install_fake_smbus(monkeypatch)
        path = tmp_path / "robot_config.yaml"
        _measured(ServoCalibration.default()).save(path)
        with caplog.at_level(logging.ERROR, logger="robot_server"):
            rc = rs.main(["--driver", "pca9685", "--config", str(path), "--no-camera",
                          "--host-timeout-s", "0", "--port", "5573"])
        assert rc == 2 and any("no watchdog" in r.getMessage() for r in caplog.records)

    def test_missing_smbus2_names_the_fix(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "smbus2", None)
        with pytest.raises(DriverError, match="pip3 install smbus2"):
            Pca9685Driver(_measured(ServoCalibration.default()))

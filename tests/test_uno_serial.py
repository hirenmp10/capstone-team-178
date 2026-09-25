"""tests/test_uno_serial.py — Unit tests for UnoSerialDriver.

Protocol conformance of FakeUnoSerial against servo_bridge.ino
---------------------------------------------------------------
FakeUnoSerial is a Python re-implementation of the sketch's line protocol.
It is deliberately isomorphic to the firmware logic so that every test that
passes here is strong evidence the real driver/sketch interface is correct.

Real conformance (electrical behaviour, watchdog timing, servo movement) is
proved only by smoke test S1–S6 in docs/hardware/BENCH.md; record results in
docs/hardware/BENCH_LOG.md.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Iterator

import pytest

from jetson.robot_server import (
    DriverError,
    RobotServer,
    ServoCalibration,
    UnoSerialDriver,
)


# ===========================================================================
# FakeUnoSerial — Python re-implementation of servo_bridge.ino
# ===========================================================================

N_CH = 5
PULSE_MIN_US = 500
PULSE_MAX_US = 2500
DEFAULT_WATCHDOG_MS = 500


class FakeUnoSerial:
    """Python implementation of the Uno serial protocol for testing UnoSerialDriver.

    Parameters
    ----------
    n_ch:
        Number of servo channels to advertise (default 5).
    watchdog_ms:
        Initial watchdog timeout in ms; 0 disables it.
    clock:
        Callable returning current time in seconds (default ``time.monotonic``).
        Inject a controlled clock in tests to simulate time passing.
    auto_read:
        If True (default) the object returns the reply string automatically
        via ``readline()``.  Set False to simulate a timeout (readline returns b'').
    """

    def __init__(
        self,
        n_ch: int = N_CH,
        watchdog_ms: int = DEFAULT_WATCHDOG_MS,
        clock=None,
        auto_read: bool = True,
    ) -> None:
        self.n_ch = n_ch
        self.wd_ms = watchdog_ms
        self._clock = clock or time.monotonic
        self.auto_read = auto_read

        # Internal state mirroring the sketch
        self._attached = False
        self._moving = False
        self._pulses: list[int] = [1500] * n_ch
        self._last_frame_t: float = self._clock()

        # Queued reply
        self._reply_buf: bytes = b""

        # Thread safety
        self._lock = threading.Lock()

        # Command log (for test assertions)
        self.commands_received: list[str] = []

    # ------------------------------------------------------------------
    # pyserial-compatible interface required by UnoSerialDriver
    # ------------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Receive a command from the driver (host writes to us)."""
        with self._lock:
            line = data.decode(errors="replace").rstrip("\r\n")
            self.commands_received.append(line)
            reply = self._handle(line)
            self._reply_buf += (reply + "\n").encode()

    def readline(self) -> bytes:
        """Return the queued reply (or empty bytes if auto_read is False)."""
        if not self.auto_read:
            return b""
        with self._lock:
            if b"\n" in self._reply_buf:
                line, rest = self._reply_buf.split(b"\n", 1)
                self._reply_buf = rest
                return line + b"\n"
            r = self._reply_buf
            self._reply_buf = b""
            return r

    def reset_input_buffer(self) -> None:
        """Clear any pending reply bytes (simulating clearing serial RX buffer)."""
        with self._lock:
            self._reply_buf = b""

    @property
    def output_buffer(self) -> bytes:
        """Pending bytes to be sent to host."""
        with self._lock:
            return self._reply_buf

    @output_buffer.setter
    def output_buffer(self, data: bytes) -> None:
        with self._lock:
            self._reply_buf = data

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Watchdog check (called by each frame handler)
    # ------------------------------------------------------------------

    def _check_watchdog(self) -> None:
        if self._attached and self.wd_ms > 0:
            elapsed_ms = (self._clock() - self._last_frame_t) * 1000.0
            if elapsed_ms > self.wd_ms:
                self._attached = False
                self._moving = False

    # ------------------------------------------------------------------
    # Protocol handler (mirrors sketch handle_line())
    # ------------------------------------------------------------------

    def _handle(self, line: str) -> str:
        tokens = line.split()
        if not tokens:
            return "OK"  # blank line — sketch ignores it; we echo OK

        cmd = tokens[0]

        # V — version
        if cmd == "V":
            if len(tokens) != 1:
                return "ERR argc"
            return f"V servo_bridge 1 {self.n_ch}"

        # S — status (keepalive)
        if cmd == "S":
            if len(tokens) != 1:
                return "ERR argc"
            self._check_watchdog()
            self._last_frame_t = self._clock()
            att = 1 if self._attached else 0
            mov = 1 if self._moving else 0
            pulses_str = " ".join(str(p) for p in self._pulses)
            return f"S {att} {mov} {self.wd_ms} {pulses_str}"

        # D — detach
        if cmd == "D":
            if len(tokens) != 1:
                return "ERR argc"
            self._attached = False
            self._moving = False
            return "OK"

        # W — set watchdog
        if cmd == "W":
            if len(tokens) != 2:
                return "ERR argc"
            ms = int(tokens[1])
            if ms != 0 and not (50 <= ms <= 10000):
                return "ERR range"
            self.wd_ms = ms
            return "OK"

        # P — set immediately
        if cmd == "P":
            if len(tokens) != 1 + self.n_ch:
                return "ERR argc"
            try:
                new_pulses = [int(t) for t in tokens[1:]]
            except ValueError:
                return "ERR range"
            if any(p < PULSE_MIN_US or p > PULSE_MAX_US for p in new_pulses):
                return "ERR range"
            self._check_watchdog()
            self._last_frame_t = self._clock()
            self._pulses = new_pulses
            self._attached = True
            self._moving = False
            return "OK"

        # T — interpolate
        if cmd == "T":
            if len(tokens) != 2 + self.n_ch:
                return "ERR argc"
            try:
                ms = int(tokens[1])
                new_pulses = [int(t) for t in tokens[2:]]
            except ValueError:
                return "ERR range"
            if not (0 <= ms <= 30000):
                return "ERR range"
            if any(p < PULSE_MIN_US or p > PULSE_MAX_US for p in new_pulses):
                return "ERR range"
            self._check_watchdog()
            self._last_frame_t = self._clock()
            self._pulses = new_pulses
            self._attached = True
            self._moving = ms > 0
            return "OK"

        return "ERR cmd"

    # ------------------------------------------------------------------
    # Convenience accessors for tests
    # ------------------------------------------------------------------

    @property
    def attached(self) -> bool:
        return self._attached

    def simulate_watchdog_expiry(self) -> None:
        """Push last_frame_t back by watchdog + 1 ms to trigger the WD on next frame."""
        self._last_frame_t -= (self.wd_ms + 1) / 1000.0


# ===========================================================================
# Factory helper used in all tests
# ===========================================================================

def _make_driver(
    fake: FakeUnoSerial,
    watchdog_ms: int = DEFAULT_WATCHDOG_MS,
) -> UnoSerialDriver:
    """Create a UnoSerialDriver that injects fake as the serial port."""
    def factory(port, baud, timeout):
        return fake

    driver = UnoSerialDriver(
        port="/dev/fake",
        baud=115200,
        watchdog_ms=watchdog_ms,
        timeout_s=0.1,
        serial_factory=factory,
    )
    return driver


# ===========================================================================
# Tests
# ===========================================================================


class TestOpenHandshake:
    """(a) open() handshake: V mismatch raises; success sends W then S."""

    def test_channel_count_mismatch_raises(self) -> None:
        """If the firmware reports a different channel count, RuntimeError."""
        fake = FakeUnoSerial(n_ch=3)  # firmware says 3
        driver = _make_driver(fake)
        cal = ServoCalibration.default()  # 4 joints + 1 jaw = 5 channels
        with pytest.raises(RuntimeError, match="channel count 3 != calibration channel count 5"):
            driver.open(calibration=cal)

    def test_successful_open_sends_W_then_S(self) -> None:
        """Successful open: sequence is V -> W <ms> -> S; driver is not attached."""
        fake = FakeUnoSerial(n_ch=5)
        driver = _make_driver(fake, watchdog_ms=600)
        cal = ServoCalibration.default()
        driver.open(calibration=cal)

        cmds = fake.commands_received
        assert cmds[0] == "V"
        assert cmds[1] == f"W {600}"
        assert cmds[2] == "S"
        # After S at boot, attached = False
        assert driver.attached is False

    def test_open_without_calibration_uses_n5(self) -> None:
        """open(calibration=None) accepts firmware n_ch=5."""
        fake = FakeUnoSerial(n_ch=5)
        driver = _make_driver(fake)
        driver.open(calibration=None)  # should not raise

    def test_open_flushes_pending_serial_input(self) -> None:
        """Pre-load the fake's output buffer with b"OK\\n" before open() and assert V handshake succeeds."""
        fake = FakeUnoSerial(n_ch=5)
        fake.output_buffer = b"OK\n"
        driver = _make_driver(fake)
        driver.open(calibration=ServoCalibration.default())
        assert fake.commands_received[0] == "V"
        assert driver.attached is False


class TestWritePulses:
    """(b) write_pulses: sends correct bytes; out-of-range raises DriverError."""

    def _opened_driver(self, fake: FakeUnoSerial | None = None):
        if fake is None:
            fake = FakeUnoSerial()
        driver = _make_driver(fake)
        driver.open(calibration=ServoCalibration.default())
        fake.commands_received.clear()
        return driver, fake

    def test_write_pulses_sends_P_frame(self) -> None:
        driver, fake = self._opened_driver()
        driver.write_pulses([1500, 1500, 1500, 1500, 1500])
        assert fake.commands_received[-1] == "P 1500 1500 1500 1500 1500"

    def test_write_pulses_with_duration_sends_T_frame(self) -> None:
        driver, fake = self._opened_driver()
        driver.write_pulses([1000, 1500, 1500, 1500, 1500], duration_ms=1000)
        assert fake.commands_received[-1] == "T 1000 1000 1500 1500 1500 1500"

    def test_out_of_range_pulse_raises_DriverError(self) -> None:
        driver, fake = self._opened_driver()
        snapshot_before = list(fake._pulses)
        with pytest.raises(DriverError):
            driver.write_pulses([300, 1500, 1500, 1500, 1500])  # 300 < PULSE_MIN_US
        # Fake state must be unchanged (ERR range is applied atomically)
        assert fake._pulses == snapshot_before

    def test_write_pulses_marks_driver_attached(self) -> None:
        driver, fake = self._opened_driver()
        assert driver.attached is False
        driver.write_pulses([1500, 1500, 1500, 1500, 1500])
        assert driver.attached is True


class TestKeepalive:
    """(c) keepalive() updates attached; after watchdog expiry fake shows attached=0."""

    def _opened_driver(self):
        fake = FakeUnoSerial(watchdog_ms=200)
        driver = _make_driver(fake, watchdog_ms=200)
        driver.open(calibration=ServoCalibration.default())
        # Attach
        driver.write_pulses([1500, 1500, 1500, 1500, 1500])
        assert driver.attached is True
        return driver, fake

    def test_keepalive_refreshes_attached_state(self) -> None:
        driver, fake = self._opened_driver()
        driver.keepalive()
        assert driver.attached is True

    def test_keepalive_after_watchdog_shows_detached(self) -> None:
        driver, fake = self._opened_driver()
        # Simulate the watchdog firing on the fake side
        fake.simulate_watchdog_expiry()
        # Next S command will trigger the watchdog check inside the fake
        driver.keepalive()
        assert driver.attached is False


class TestDetach:
    """(d) detach() sends D; timeout raises DriverError."""

    def test_detach_sends_D(self) -> None:
        fake = FakeUnoSerial()
        driver = _make_driver(fake)
        driver.open(calibration=ServoCalibration.default())
        driver.write_pulses([1500] * 5)
        fake.commands_received.clear()
        driver.detach()
        assert fake.commands_received[-1] == "D"
        assert driver.attached is False

    def test_timeout_from_silent_fake_does_not_propagate(self) -> None:
        """UnoSerialDriver.detach() swallows DriverError from a silent port."""
        fake = FakeUnoSerial(auto_read=False)  # readline returns b''
        driver = _make_driver(fake)
        # We bypass open() since open() would also fail; inject state manually
        driver._ser = fake
        driver.attached = True
        # detach() should not raise even with a silent Uno
        driver.detach()

    def test_explicit_cmd_timeout_raises_DriverError(self) -> None:
        """A command sent when the fake is silent raises DriverError."""
        fake = FakeUnoSerial(auto_read=False)
        driver = _make_driver(fake)
        driver._ser = fake
        with pytest.raises(DriverError, match="no reply"):
            driver._cmd("P 1500 1500 1500 1500 1500")


class TestRobotServerKeepaliveIntegration:
    """(e) RobotServer with UnoSerialDriver: idle -> keepalive path; estop -> D."""

    def _make_server_with_fake(self, fake: FakeUnoSerial) -> tuple[RobotServer, UnoSerialDriver]:
        cal = ServoCalibration.default()
        driver = _make_driver(fake, watchdog_ms=DEFAULT_WATCHDOG_MS)
        driver.open(calibration=cal)
        server = RobotServer(
            driver=driver,
            calibration=cal,
            host_timeout_s=99.0,  # do not fire host timeout during the test
            tick_s=0.02,
        )
        return server, driver

    def test_idle_server_keeps_arm_attached_via_keepalive(self) -> None:
        """1 second idle: the server sends S frames (not P), fake stays attached."""
        fake = FakeUnoSerial(watchdog_ms=DEFAULT_WATCHDOG_MS)
        server, driver = self._make_server_with_fake(fake)

        # Attach by writing pulses once
        driver.write_pulses([1500] * 5)
        fake.commands_received.clear()

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(1.0)  # server idles for 1 second (~50 ticks at 20 ms each)

        # Snapshot attached state BEFORE stop() calls detach()
        was_attached = fake.attached
        s_cmds = [c for c in fake.commands_received if c == "S"]
        p_cmds = [c for c in fake.commands_received if c.startswith("P")]

        server.stop()
        t.join(timeout=2.0)

        # The follower must have sent S frames (keepalive); fake stayed attached
        assert len(s_cmds) >= 10, f"Expected >= 10 S frames, got {len(s_cmds)}"
        # Ensure NO P frame was sent during idle (keepalive hook used instead)
        assert len(p_cmds) == 0, f"Expected 0 P frames during idle, got {len(p_cmds)}"
        # Fake was attached before stop() detached
        assert was_attached is True

    def test_estop_sends_D(self) -> None:
        """estop() calls driver.detach() which sends D."""
        from mfw.hardware.zmq_rpc import ZmqRpcClient

        fake = FakeUnoSerial(watchdog_ms=DEFAULT_WATCHDOG_MS)
        server, driver = self._make_server_with_fake(fake)

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            client = ZmqRpcClient("127.0.0.1", server.port)
            client.call("estop")
            client.close()
        finally:
            server.stop()
            t.join(timeout=2.0)

        d_cmds = [c for c in fake.commands_received if c == "D"]
        assert len(d_cmds) >= 1, "Expected at least one D frame after estop"

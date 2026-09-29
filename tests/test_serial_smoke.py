"""jetson/serial_smoke.py -- the S3 smoke test that replaced ``printf > /dev/ttyACM0``.

The fake port models the two physical facts that broke the printf version:

* opening the port RESETS the Uno (DTR), and for ``boot_s`` its bootloader
  swallows every byte sent to it;
* the sketch prints nothing useful until it has booted, and some boards print
  a boot banner the host must drain.

It answers like servo_bridge.ino (``Q<us>x5,<attached>,<known>``, first P
attaches AT its targets, D keeps the last position, 500 ms watchdog), and a
reply the host does not wait for is simply a readline() timeout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from jetson import serial_smoke

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeUno:
    def __init__(self, boot_s: float = 1.6, banner: bool = True, old_sketch: bool = False,
                 exchange_s: float = 0.004, watchdog_s: float = 0.5) -> None:
        self.now = 0.0  # the port was opened (and the Uno reset) at t = 0
        self.boot_s = boot_s
        self.exchange_s = exchange_s
        self.watchdog_s = watchdog_s
        self.old_sketch = old_sketch
        self.pending: list[bytes] = []
        self.booted_banner = banner
        self.us = [1500] * 5
        self.attached = False
        self.known = False
        self.last_frame = 0.0
        self.lost_in_bootloader = 0

    # the clock the smoke test sleeps on
    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def _booted(self) -> bool:
        return self.now >= self.boot_s

    def reset_input_buffer(self) -> None:
        if self._booted() and self.booted_banner:
            self.booted_banner = False  # the banner is drained here, never read as a reply
        self.pending.clear()

    def flush(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        self.now += self.exchange_s
        if self.attached and self.now - self.last_frame > self.watchdog_s:
            self.attached = False
        if not self._booted():
            self.lost_in_bootloader += 1
            return len(data)
        if self.booted_banner:
            self.pending.append(b"servo_bridge ready\r\n")
            self.booted_banner = False
        line = data.decode("ascii").strip()
        self.last_frame = self.now
        self.pending.append((self._handle(line) + "\r\n").encode("ascii"))
        return len(data)

    def _handle(self, line: str) -> str:
        if line == "?":
            head = "Q" + ",".join(str(u) for u in self.us)
            if self.old_sketch:
                return head
            return f"{head},{int(self.attached)},{int(self.known)}"
        if line == "D":
            self.attached = False
            return "OK"
        if line.startswith("P"):
            values = [int(v) for v in line[1:].split(",")]
            if len(values) != 5:
                return "ERR bad frame"
            values = [min(max(v, 900 if i == 4 else 600), 2100 if i == 4 else 2400) for i, v in enumerate(values)]
            if not self.attached:
                if not self.known:
                    self.us = values  # first frame since reset: attach AT the target
                self.attached = True
                self.known = True
            self.us = values
            return "OK"
        return f"ERR unknown command {line[:1]}"

    def readline(self) -> bytes:
        if self.pending:
            return self.pending.pop(0)
        self.now += serial_smoke.REPLY_TIMEOUT_S  # nothing came: a timeout
        return b""


class TestSmoke:
    def test_waiting_for_the_reset_passes_all_five_steps(self):
        uno = FakeUno()
        steps = serial_smoke.run_smoke(uno, (1500, 1500, 1150, 1600, 1000), sleep=uno.sleep)
        assert [ok for _s, _r, ok, _e in steps] == [True] * 5, serial_smoke.format_steps(steps)
        assert steps[2][1] == "Q1500,1500,1150,1600,1000,1,1"
        assert uno.lost_in_bootloader == 0
        assert not uno.attached, "the smoke test leaves the arm limp"

    def test_printf_timing_loses_the_frame_in_the_bootloader(self):
        """What `printf ... > /dev/ttyACM0` did: write the moment the port opens."""
        uno = FakeUno()
        steps = serial_smoke.run_smoke(uno, reset_wait_s=0.0, sleep=uno.sleep)
        assert uno.lost_in_bootloader >= 1
        assert not all(ok for _s, _r, ok, _e in steps)
        assert steps[0][1] == ""
        assert "no reply at all" in serial_smoke.hint(steps)

    def test_a_slow_bootloader_needs_a_longer_wait(self):
        uno = FakeUno(boot_s=2.5)
        assert not all(ok for *_x, ok, _e in serial_smoke.run_smoke(uno, sleep=uno.sleep))
        uno = FakeUno(boot_s=2.5)
        assert all(ok for *_x, ok, _e in serial_smoke.run_smoke(uno, reset_wait_s=3.0, sleep=uno.sleep))

    def test_boot_banner_is_drained_not_taken_for_a_reply(self):
        uno = FakeUno(banner=True)
        steps = serial_smoke.run_smoke(uno, sleep=uno.sleep)
        assert steps[0][1].startswith("Q") and steps[0][2]

    def test_old_sketch_is_named(self):
        uno = FakeUno(old_sketch=True)
        steps = serial_smoke.run_smoke(uno, sleep=uno.sleep)
        assert not steps[0][2]
        assert "OLD sketch" in serial_smoke.hint(steps)

    def test_a_watchdog_detach_between_frames_fails_step_three(self):
        """A host that dawdles past 500 ms between P and ? sees the Uno go limp."""
        uno = FakeUno(exchange_s=0.6)
        steps = serial_smoke.run_smoke(uno, sleep=uno.sleep)
        assert steps[1][2] and not steps[2][2]
        assert steps[2][1].endswith(",0,1")

    def test_a_clamped_pulse_is_caught(self):
        """The sketch clamps silently and still answers OK; the echo shows it."""
        uno = FakeUno()
        steps = serial_smoke.run_smoke(uno, (1500, 1500, 1500, 1500, 2300), sleep=uno.sleep)
        assert steps[1][2] and not steps[2][2]
        assert steps[2][1].startswith("Q1500,1500,1500,1500,2100")


class TestCli:
    def test_pulses_are_validated(self):
        assert serial_smoke.parse_pulses("1500,1500,1150,1600,1000") == (1500, 1500, 1150, 1600, 1000)
        with pytest.raises(ValueError):
            serial_smoke.parse_pulses("1500,1500")
        with pytest.raises(ValueError):
            serial_smoke.parse_pulses("1500,1500,1500,1500,3000")

    def test_unopenable_port_exits_2(self, capsys, monkeypatch):
        """Port absent or held by robot_server: pyserial raises on open."""
        import sys
        import types

        fake = types.ModuleType("serial")

        def _serial(*_a, **_k):
            raise OSError(16, "Device or resource busy")

        fake.Serial = _serial
        monkeypatch.setitem(sys.modules, "serial", fake)
        assert serial_smoke.main(["--serial", "/dev/ttyACM0", "--reset-wait-s", "0"]) == 2
        assert "cannot open" in capsys.readouterr().err

    def test_jetson_side_constraints(self):
        source = (REPO_ROOT / "jetson" / "serial_smoke.py").read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 10))
        assert "mfw" not in {n.split(".")[0] for n in _imports(source)}
        assert "\r" not in source


def _imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names

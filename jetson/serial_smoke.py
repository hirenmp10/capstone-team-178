#!/usr/bin/env python3
"""Smoke test S3: talk to servo_bridge.ino directly, without robot_server.

    python3 jetson/serial_smoke.py --serial /dev/ttyACM0
    python3 jetson/serial_smoke.py --serial /dev/ttyACM0 --pulses 1500,1500,1150,1600,1000

Why not ``printf 'P1500,...\\n' > /dev/ttyACM0``: opening the port pulses DTR,
which RESETS the Uno, and its bootloader ignores the serial line for about
1-2 s. printf opens, writes at once and closes, so the frame lands in the
bootloader and is lost (and the tty may not even be at 115200). This script
opens the port once at 115200, waits ``--reset-wait-s`` (2 s) for the sketch
to boot, drains whatever the boot printed, then runs five exchanges:

1. ``?``  -> ``Q<us>x5,0,0``   freshly reset: detached, no position known
2. ``P``  -> ``OK``            attaches every servo AT the given pulses
3. ``?``  -> ``Q<us>x5,1,1``   attached, position known, pulses echoed
4. ``D``  -> ``OK``            detach: the arm goes limp
5. ``?``  -> ``Q<us>x5,0,1``   detached, last position kept

Run it with the LINKAGES OFF THE HORNS (step 2 moves every servo at full
speed) and with robot_server stopped (only one process may own the port).
Exit status 0 when every step passed, 1 otherwise, 2 when the port cannot be
opened. Runs on the Jetson's Python 3.10; needs only pyserial; no mfw imports.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

BAUD = 115200
N_SERVOS = 5
DEFAULT_PULSES = (1500, 1500, 1500, 1500, 1500)
DEFAULT_RESET_WAIT_S = 2.0
REPLY_TIMEOUT_S = 0.5

Step = Tuple[str, str, bool, str]
"""(sent, reply, passed, expectation)"""


def parse_pulses(text: str) -> Tuple[int, ...]:
    values = tuple(int(v) for v in text.split(","))
    if len(values) != N_SERVOS:
        raise ValueError(f"need {N_SERVOS} pulses (base, shoulder, elbow, wrist, gripper), got {len(values)}")
    for v in values:
        if not 500 <= v <= 2500:
            raise ValueError(f"pulse {v} us is outside 500..2500")
    return values


def _query_ok(reply: str, attached: int, known: int, pulses: Optional[Sequence[int]]) -> bool:
    if not reply.startswith("Q"):
        return False
    fields = reply[1:].split(",")
    if len(fields) != N_SERVOS + 2:
        return False
    try:
        numbers = [int(f) for f in fields]
    except ValueError:
        return False
    if numbers[N_SERVOS:] != [attached, known]:
        return False
    return pulses is None or list(numbers[:N_SERVOS]) == list(pulses)


def _exchange(ser: Any, line: str) -> str:
    ser.write((line + "\n").encode("ascii"))
    ser.flush()
    return ser.readline().decode("ascii", errors="replace").strip()


def run_smoke(
    ser: Any,
    pulses: Sequence[int] = DEFAULT_PULSES,
    reset_wait_s: float = DEFAULT_RESET_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> List[Step]:
    """The five S3 exchanges on an already-open port; every step is reported."""
    sleep(reset_wait_s)
    ser.reset_input_buffer()  # anything the sketch printed while booting
    frame = "P" + ",".join(str(int(p)) for p in pulses)
    plan = [
        ("?", lambda r: _query_ok(r, 0, 0, None), "Q<us>x5,0,0 (fresh reset: detached, no position)"),
        (frame, lambda r: r == "OK", "OK (servos attach AT these pulses)"),
        ("?", lambda r: _query_ok(r, 1, 1, pulses), "Q" + ",".join(map(str, pulses)) + ",1,1"),
        ("D", lambda r: r == "OK", "OK (detached, arm limp)"),
        ("?", lambda r: _query_ok(r, 0, 1, None), "Q<us>x5,0,1 (detached, last position kept)"),
    ]
    steps: List[Step] = []
    for sent, check, expect in plan:
        reply = _exchange(ser, sent)
        steps.append((sent, reply, bool(check(reply)), expect))
    return steps


def format_steps(steps: Sequence[Step]) -> str:
    lines = []
    for sent, reply, ok, expect in steps:
        lines.append(f"{'PASS' if ok else 'FAIL'}  sent {sent!r:<28} got {reply or '(nothing)'!r}  expected {expect}")
    return "\n".join(lines)


def hint(steps: Sequence[Step]) -> str:
    if steps and not steps[0][1]:
        return ("no reply at all: wrong port, sketch not flashed, a different baud rate, "
                "or --reset-wait-s too short for this board's bootloader")
    if steps and steps[0][1] and not steps[0][2] and steps[0][1].startswith("Q") and steps[0][1].count(",") == 4:
        return "5-field Q reply: an OLD sketch -- flash the current jetson/arduino/servo_bridge/servo_bridge.ino"
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S3: servo_bridge.ino serial smoke test (linkages OFF)")
    parser.add_argument("--serial", default="/dev/ttyACM0")
    parser.add_argument("--pulses", default=",".join(map(str, DEFAULT_PULSES)),
                        help="base,shoulder,elbow,wrist,gripper in us for the P frame")
    parser.add_argument("--reset-wait-s", type=float, default=DEFAULT_RESET_WAIT_S,
                        help="wait after opening the port (the Uno resets on open)")
    args = parser.parse_args(argv)
    try:
        pulses = parse_pulses(args.pulses)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import serial  # pyserial
    except ImportError:
        print("pyserial is missing: pip install pyserial", file=sys.stderr)
        return 2
    try:
        ser = serial.Serial(args.serial, BAUD, timeout=REPLY_TIMEOUT_S)
    except Exception as exc:  # SerialException / OSError: port absent or busy
        print(f"cannot open {args.serial}: {exc} (is robot_server running? are you in 'dialout'?)",
              file=sys.stderr)
        return 2
    try:
        steps = run_smoke(ser, pulses, args.reset_wait_s)
    finally:
        ser.close()
    print(format_steps(steps))
    extra = hint(steps)
    if extra:
        print(extra)
    passed = all(ok for _s, _r, ok, _e in steps)
    print("S3 PASS" if passed else "S3 FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

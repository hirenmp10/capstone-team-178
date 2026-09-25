"""calibrate_servos.py — interactive servo calibration tool.

Talks directly to the Arduino Uno over serial (not via robot_server), so it
can be used BEFORE the robot server is running or even on the bench without
the Jetson.

Usage (on the Jetson or any machine with pyserial):

    python scripts/calibrate_servos.py \\
        --port /dev/servo_bridge \\
        --baud 115200 \\
        --calibration jetson/servo_calibration.json

Interactive commands
--------------------
  ch <i>          Select channel i (0=base_yaw, 1=shoulder, 2=elbow, 3=wrist, 4=jaw).
  p <us>          Set the selected channel to <us> microseconds (others hold).
  +               Step selected channel up 10 µs.
  -               Step selected channel down 10 µs.
  min             Record current pulse as pulse_min_us for this channel.
  max             Record current pulse as pulse_max_us for this channel.
  zero            Record current pulse as pulse_zero_us (the angle-zero pose).
  angle <deg>     Record that the current pulse corresponds to <deg> degrees.
                  Two angle records with different pulses compute us_per_rad.
  open            (jaw only) Record current pulse as the open-jaw pulse;
                  you will be prompted for the jaw width in mm.
  closed          (jaw only) Record current pulse as the closed-jaw pulse;
                  you will be prompted for the jaw width in mm.
  relax           Send D — detach all servos.
  save            Write calibration to the JSON file; prints the diff.
  quit / q        Send D, close the port, and exit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Add repo root to path so jetson.robot_server is importable from here
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jetson.robot_server import ServoCalibration  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny serial helper — avoids importing the whole driver stack
# ---------------------------------------------------------------------------

class _SerialBridge:
    """Thin wrapper that sends ASCII commands and reads one reply line."""

    def __init__(self, port: str, baud: int, timeout_s: float = 2.0) -> None:
        try:
            import serial  # pyserial
        except ImportError:
            print("ERROR: pyserial is not installed.  Run: pip install pyserial")
            sys.exit(1)
        print(f"Opening {port} at {baud} baud …", end=" ", flush=True)
        self._ser = serial.Serial(port, baud, timeout=timeout_s)
        time.sleep(2.0)  # wait for Uno auto-reset
        print("OK")

    def cmd(self, line: str) -> str:
        self._ser.write((line + "\n").encode())
        reply = self._ser.readline().decode(errors="replace").strip()
        return reply

    def close(self) -> None:
        self._ser.close()


# ---------------------------------------------------------------------------
# Calibration session
# ---------------------------------------------------------------------------

PULSE_MIN_HARD = 500
PULSE_MAX_HARD = 2500

JOINT_NAMES = ["base_yaw", "shoulder", "elbow", "wrist", "jaw"]


def _apply(bridge: _SerialBridge, pulses: list[int]) -> None:
    """Send P with all N channels."""
    line = "P " + " ".join(str(p) for p in pulses)
    reply = bridge.cmd(line)
    if reply.startswith("ERR"):
        print(f"  Uno error: {reply}")


def _clamp(us: int) -> int:
    return max(PULSE_MIN_HARD, min(PULSE_MAX_HARD, us))


def _diff(a: dict, b: dict) -> list[str]:
    """Recursively collect changed leaf values between two nested dicts."""
    lines = []

    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, full_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        out.update(_flatten(item, f"{full_key}[{i}]"))
                    else:
                        out[f"{full_key}[{i}]"] = item
            else:
                out[full_key] = v
        return out

    fa, fb = _flatten(a), _flatten(b)
    for k in sorted(set(fa) | set(fb)):
        av, bv = fa.get(k), fb.get(k)
        if av != bv:
            lines.append(f"  {k}: {av} -> {bv}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive servo calibration for hardware lane (Uno serial)"
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument(
        "--calibration",
        default="jetson/servo_calibration.json",
        help="Calibration JSON path; loaded if exists, else defaults used (default: jetson/servo_calibration.json)",
    )
    args = parser.parse_args()

    cal_path = Path(args.calibration)
    if cal_path.is_file():
        cal = ServoCalibration.from_json(cal_path)
        print(f"Loaded calibration from {cal_path}")
    else:
        cal = ServoCalibration.default()
        print(f"No calibration found at {cal_path}; using defaults.")

    original_dict = cal.to_dict()

    bridge = _SerialBridge(args.port, args.baud)

    # --- sync initial pulses from firmware state ---
    s_reply = bridge.cmd("S")
    parts = s_reply.split()
    n_ch = len(cal.joints) + 1  # joints + jaw
    if len(parts) >= 3 + n_ch and parts[0] == "S":
        pulses = [int(p) for p in parts[3: 3 + n_ch]]
    else:
        pulses = [ch.pulse_zero_us for ch in cal.joints] + [cal.jaw.pulse_min_us]

    sel = 0  # currently selected channel

    # Per-channel angle measurement accumulator for us_per_rad computation
    # { ch_idx: [(pulse, rad), ...] }
    angle_records: dict[int, list[tuple[int, float]]] = {}

    print("\nChannels: 0=base_yaw  1=shoulder  2=elbow  3=wrist  4=jaw")
    print("Type 'help' for available commands, 'quit' to exit.\n")

    def _current_name() -> str:
        return JOINT_NAMES[sel] if sel < len(JOINT_NAMES) else f"ch{sel}"

    while True:
        try:
            raw = input(f"[ch{sel}/{_current_name()}  p={pulses[sel]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "quit"

        if not raw:
            continue

        tokens = raw.split()
        cmd = tokens[0].lower()

        if cmd == "help":
            print(__doc__)

        elif cmd == "ch":
            if len(tokens) < 2 or not tokens[1].isdigit():
                print("  Usage: ch <index>")
            else:
                i = int(tokens[1])
                if 0 <= i < n_ch:
                    sel = i
                    print(f"  Selected channel {sel} ({_current_name()})")
                else:
                    print(f"  Channel must be 0..{n_ch - 1}")

        elif cmd == "p":
            if len(tokens) < 2:
                print("  Usage: p <us>")
            else:
                us = _clamp(int(tokens[1]))
                pulses[sel] = us
                _apply(bridge, pulses)

        elif cmd == "+":
            pulses[sel] = _clamp(pulses[sel] + 10)
            _apply(bridge, pulses)

        elif cmd == "-":
            pulses[sel] = _clamp(pulses[sel] - 10)
            _apply(bridge, pulses)

        elif cmd == "min":
            ch = cal.joints[sel] if sel < 4 else cal.jaw
            old = ch.pulse_min_us
            ch.pulse_min_us = pulses[sel]
            print(f"  pulse_min_us: {old} -> {pulses[sel]}")

        elif cmd == "max":
            ch = cal.joints[sel] if sel < 4 else cal.jaw
            old = ch.pulse_max_us
            ch.pulse_max_us = pulses[sel]
            print(f"  pulse_max_us: {old} -> {pulses[sel]}")

        elif cmd == "zero":
            if sel >= 4:
                print("  'zero' is for joints only; use 'open'/'closed' for the jaw.")
            else:
                ch = cal.joints[sel]
                old = ch.pulse_zero_us
                ch.pulse_zero_us = pulses[sel]
                print(f"  pulse_zero_us: {old} -> {pulses[sel]}")

        elif cmd == "angle":
            if sel >= 4:
                print("  'angle' is for joints only.")
            elif len(tokens) < 2:
                print("  Usage: angle <deg>")
            else:
                deg = float(tokens[1])
                rad = math.radians(deg)
                recs = angle_records.setdefault(sel, [])
                recs.append((pulses[sel], rad))
                print(f"  Recorded angle {deg:.2f}° ({rad:.4f} rad) at pulse {pulses[sel]} µs")
                if len(recs) >= 2:
                    (p1, r1), (p2, r2) = recs[-2], recs[-1]
                    if abs(r2 - r1) > 0.01:
                        us_per_rad = (p2 - p1) / (r2 - r1)
                        ch = cal.joints[sel]
                        old = ch.us_per_rad
                        setattr(ch, "us_per_rad", round(us_per_rad, 4))
                        print(f"  us_per_rad computed: {old:.4f} -> {us_per_rad:.4f}")

        elif cmd == "open":
            if sel != 4:
                print("  'open' is for the jaw channel only (select ch 4 first).")
            else:
                width_mm = float(input("  Jaw width when open (mm): ").strip())
                width_m = width_mm / 1000.0
                cal.jaw.pulse_min_us = pulses[sel]  # open = low pulse
                cal.jaw.width_open_m = width_m
                print(f"  jaw open: pulse_min_us={pulses[sel]}, width_open_m={width_m:.4f} m")

        elif cmd == "closed":
            if sel != 4:
                print("  'closed' is for the jaw channel only (select ch 4 first).")
            else:
                width_mm = float(input("  Jaw width when closed (mm): ").strip())
                width_m = width_mm / 1000.0
                cal.jaw.pulse_max_us = pulses[sel]  # closed = high pulse
                cal.jaw.width_closed_m = width_m
                print(f"  jaw closed: pulse_max_us={pulses[sel]}, width_closed_m={width_m:.4f} m")

        elif cmd == "relax":
            bridge.cmd("D")
            print("  Servos detached.")

        elif cmd == "save":
            new_dict = cal.to_dict()
            diff_lines = _diff(
                {f"{k}.{kk}": vv for k, v in original_dict.items() if isinstance(v, dict)
                 for kk, vv in v.items()},
                {f"{k}.{kk}": vv for k, v in new_dict.items() if isinstance(v, dict)
                 for kk, vv in v.items()},
            )
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal.to_json(cal_path)
            if diff_lines:
                print(f"  Saved to {cal_path}.  Changes:")
                for l in diff_lines:
                    print(l)
            else:
                print(f"  Saved to {cal_path} (no changes vs original).")

        elif cmd in ("quit", "q"):
            bridge.cmd("D")
            bridge.close()
            print("  Servos detached. Goodbye.")
            sys.exit(0)

        else:
            print(f"  Unknown command: {cmd!r}.  Type 'help'.")


if __name__ == "__main__":
    main()

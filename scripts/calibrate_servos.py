"""Interactive servo calibration over ``JetsonClient`` (smoke test S5 and after).

    py -3.12 scripts/calibrate_servos.py --jetson <ip>:5560
    py -3.12 scripts/calibrate_servos.py --jetson 127.0.0.1:5560 --script "home;read;move base_yaw 10;save" --yes

Why a REPL and not a YAML you edit by hand: every number in
``jetson/robot_config.yaml`` is *measured* on the assembled arm -- the pulse at
which a joint reads zero, the pulse at each mechanical end-stop, the jaw
opening at two pulses -- and the only instrument is the arm itself. The loop
is: command a pose, look at the arm, record what it means. The server keeps
the pose in **pulses**, so recording a new zero or limit re-labels the pose
without moving anything (``set_calibration`` swaps the map atomically).

All edits are made to a *working copy* here and pushed with ``save``; ``move``
converts degrees through the working copy so what you see is what you will
get after saving. The one measured trap: ``zero`` after ``limit`` on the same
joint shifts the limit you just recorded (limits are joint angles, and the
zero defines joint angles). Record limits last, or re-record them after
``zero``; ``read`` shows both so the mistake is visible.

Three safety rules this REPL enforces (review HS-5, HS-6 and the re-review's
critical bridge finding):

* **A freshly reset bridge has no position.** Opening the Uno's serial port
  resets it (DTR), so right after *every* ``robot_server.py`` start -- and
  after any USB reset or brownout -- the Uno knows no position and its first
  frame attaches every servo AT that frame's target at full speed. The old
  ``torque on`` read the reset Uno's 1500 us defaults as "where the Uno last
  pulsed", promised "it holds still" and snapped the elbow ~90 degrees. Now
  the server reports ``bridge_position_known`` and refuses everything but
  ``home`` in that state, and here ``torque on`` and ``home`` both say
  plainly that the servos will attach AT home, tell you to hand-pose the arm
  there, ask for ``yes``, and send ``home``. Nothing claims to hold still
  unless the bridge actually reports a position to hold.
* ``torque on`` -- and *any* motion command after ``torque off``, because
  every frame carries all five channels -- re-energises servos with no
  position feedback, AT THE LAST PULSED POSE: the Uno keeps its last pulse
  across a detach and its first pulse after re-attach is that one. The
  target for ``torque on`` is that same position (the server's ``bridge_q``;
  ``home``, slowly, when the bridge cannot tell), so an arm that was NOT
  moved while limp stays where it is -- but an arm that was moved by hand
  JUMPS back to the last pulsed pose at full speed (three MG996R snap a
  hand-posed linkage back in well under 200 ms). So it always prints that
  warning and asks for ``yes`` first; ``--script`` runs must pass ``--yes``.
  Hand-posing is no use for ``zero``/``limit`` anyway: they read the
  *commanded* pulses.
* **Placeholder calibration.** ``robot_config.yaml`` ships ``measured:
  false``; a real driver starts on it only with
  ``--allow-placeholder-calibration``, and then keeps every pulse inside a
  middle band (``pulse`` refuses outside it here too) and caps the speed.
  After S5 (end-stops, zero, limits, gripper, home), ``measured yes`` then
  ``save`` marks the file measured; restart the server without the flag.
* ``home set`` and ``save`` refuse a home with any link below the table
  (planar FK with the server's ``arm_geometry``); the runtime drives to home
  at every boot with no planner in the loop.

The REPL runs a heartbeat (``mfw.hardware.jetson_client.Heartbeat``) while it
is connected: the server relaxes an arm nobody talks to for 5 s, and a person
looking at a joint between commands is not a crashed laptop.

Commands (``help`` prints them)::

    read                       commanded pose: joint deg / pulse us / limits
    move <joint> <deg>         go to a joint angle (degrees, working calibration)
    pulse <joint> <us>         go to a raw pulse width (finding end-stops)
    jaw <mm>                   command the jaw opening
    torque on|off              on re-energises (asks first; see above); off lets the arm go limp
                               (no known position: sends `home` after a hand-pose warning)
    zero <joint>               current pose of <joint> becomes 0 rad
    limit <joint> lower|upper  current pose of <joint> becomes that limit
    gripper open|closed <mm>   current gripper pulse == jaw opening of <mm>
    home set                   current pose becomes home_q (refused below the table)
    home                       move to home_q (server calibration; asks first when the
                               bridge has no known position -- it attaches AT home)
    velocity <rad_s>           max_joint_velocity
    measured yes|no            mark the calibration measured (after S5) or not; `save` writes it
    estop | clear              software estop / clear it
    save                       validate and push the working copy (set_calibration)
    quit
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetson.robot_server import (  # noqa: E402
    GRIPPER_NAME,
    JOINT_NAMES,
    ArmGeometry,
    CalibrationError,
    JointCalibration,
    ServoCalibration,
    check_home_clearance,
)
from mfw.hardware.jetson_client import Heartbeat, JetsonClient, RpcError  # noqa: E402

__all__ = ["ServoCalibrator", "bridge_position_unknown", "main"]

_RESET_DEFAULT_US = 1500
"""The pulse a reset Uno reports on every channel: a library default."""


def bridge_position_unknown(state: dict[str, Any], calibration: ServoCalibration) -> bool:
    """Whether the bridge has no position to hold (a freshly reset Uno).

    Believes the server's ``bridge_position_known`` when it sends one. An
    older server does not, so the reset signature is recognised instead:
    detached and every channel at the 1500 us library default.
    """
    known = state.get("bridge_position_known")
    if known is not None:
        return not bool(known)
    if state.get("attached"):
        return False
    bridge_q = state.get("bridge_q")
    if bridge_q is None:
        return False
    pulses = list(calibration.q_to_pulses(np.asarray(bridge_q, dtype=np.float64)))
    width = state.get("bridge_gripper_width")
    if width is not None:
        pulses.append(calibration.width_to_us(float(width)))
    return all(abs(us - _RESET_DEFAULT_US) <= 1.0 for us in pulses)


def _ask_yes(prompt: str) -> bool:
    """Interactive confirmation: only a literal ``yes`` counts."""
    try:
        return input(prompt).strip().lower() == "yes"
    except (EOFError, KeyboardInterrupt):
        return False


class ServoCalibrator:
    """Command interpreter over a connected :class:`JetsonClient`.

    ``server_cal`` is what the Jetson currently applies (position truth is
    derived from pulses through it); ``cal`` is the working copy edited by the
    commands and pushed by ``save``. ``confirm(prompt) -> bool`` gates
    ``torque on``; the default asks on stdin, ``--script`` passes a constant.
    """

    def __init__(
        self,
        client: JetsonClient,
        out: Callable[[str], None] = print,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.client = client
        self.out = out
        self.confirm = confirm if confirm is not None else _ask_yes
        self.dirty = False
        self._load_server_calibration()
        if not self.cal.measured:
            self.out("  NOTE: this calibration says measured: false -- every value is a kit placeholder.")
            self.out("  Record end-stops, zero, limits, gripper and home (smoke test S5), then")
            self.out("  `measured yes` and `save`.")
        if self.band_us is not None:
            self.out(
                f"  NOTE: the server runs the PLACEHOLDER calibration: every pulse is kept inside "
                f"{self.band_us[0]}..{self.band_us[1]} us and the speed is capped (bench bring-up)."
            )

    def _load_server_calibration(self) -> None:
        """``server_cal``: what the server enforces now (band-limited in
        placeholder mode; position truth is derived through it). ``cal``: the
        working copy, started from the FILE calibration -- never from the
        band-limited one, or ``save`` would write the bench band into the
        file."""
        reply = self.client.get_calibration()
        self.server_cal = ServoCalibration.from_dict(reply["calibration"])
        file_raw = reply.get("file_calibration")
        self.cal = ServoCalibration.from_dict(file_raw) if isinstance(file_raw, dict) else self.server_cal
        band = reply.get("placeholder_band_us")
        self.band_us: tuple[int, int] | None = (int(band[0]), int(band[1])) if band else None
        """Pulse band the server keeps every channel inside (placeholder mode), else ``None``."""
        geometry_raw = reply.get("arm_geometry")
        self.geometry = ArmGeometry(**geometry_raw) if isinstance(geometry_raw, dict) else ArmGeometry()
        """Link lengths the server checks ``home_q`` against; defaults when an
        older server does not report them."""

    # -- position in pulse space --------------------------------------------

    def _pulses(self) -> tuple[list[float], float]:
        """``(arm pulses us[4], gripper pulse us)`` of the commanded pose."""
        state = self.client.get_state()
        q = np.asarray(state["q"], dtype=np.float64)
        return self.server_cal.q_to_pulses(q), self.server_cal.width_to_us(float(state["gripper_width"]))

    def _joint(self, name: str) -> str:
        if name not in JOINT_NAMES:
            raise ValueError(f"unknown joint {name!r}; joints: {', '.join(JOINT_NAMES)}")
        return name

    def _replace_joint(self, name: str, **changes: Any) -> None:
        joint = dataclasses.replace(self.cal.joints[name], **changes)
        joints = dict(self.cal.joints)
        joints[name] = joint
        self.cal = dataclasses.replace(self.cal, joints=joints)
        self.dirty = True

    def _goto_pulses(self, pulses: list[float]) -> dict[str, Any]:
        """Move the arm to ``pulses`` through the server's own calibration."""
        q_now = np.asarray(self.client.get_state()["q"], dtype=np.float64)
        q_goal = self.server_cal.pulses_to_q(pulses)
        clamped, moved = self.server_cal.clamp_q(q_goal)
        if moved:
            self.out("  note: target clamped to the server's joint limits (widen with `limit`, then `save`)")
        dt = float(np.max(np.abs(clamped - q_now))) / self.server_cal.max_joint_velocity
        dt = max(dt, 1.0 / self.server_cal.control_rate_hz)
        return self.client.follow_trajectory(np.stack([q_now, clamped]), dt)

    # -- commands ------------------------------------------------------------

    def cmd_read(self, _args: list[str]) -> None:
        arm, grip = self._pulses()
        flag = " (unsaved edits)" if self.dirty else ""
        self.out(f"  joint            deg      us   limits deg{flag}")
        for name, us in zip(JOINT_NAMES, arm):
            j = self.cal.joints[name]
            self.out(
                f"  {name:<15} {math.degrees(j.us_to_rad(us)):7.1f} {us:7.0f}   "
                f"[{math.degrees(j.limit_lower_rad):6.1f}, {math.degrees(j.limit_upper_rad):6.1f}]"
            )
        self.out(f"  {GRIPPER_NAME:<15} {self.cal.us_to_width(grip) * 1000.0:7.1f}mm {grip:6.0f}")
        state = self.client.last_state or {}
        self.out(f"  estopped={state.get('estopped')} moving={state.get('moving')}")

    def cmd_move(self, args: list[str]) -> None:
        if len(args) != 2:
            raise ValueError("usage: move <joint> <deg>")
        name = self._joint(args[0])
        arm, _ = self._pulses()
        arm[JOINT_NAMES.index(name)] = self.cal.rad_to_us(name, math.radians(float(args[1])))
        reply = self._goto_pulses(arm)
        self.out(f"  {name} -> {float(args[1]):.1f} deg (completed={reply.get('completed')})")

    def cmd_pulse(self, args: list[str]) -> None:
        if len(args) != 2:
            raise ValueError("usage: pulse <joint> <us>")
        name = self._joint(args[0])
        us = float(args[1])
        j = self.cal.joints[name]
        if not (j.pulse_min_us <= us <= j.pulse_max_us):
            raise ValueError(f"{us:.0f} us is outside {name}'s pulse range [{j.pulse_min_us}, {j.pulse_max_us}]")
        if self.band_us is not None and not (self.band_us[0] <= us <= self.band_us[1]):
            raise ValueError(
                f"{us:.0f} us is outside the placeholder band {self.band_us[0]}..{self.band_us[1]} us the server "
                "enforces; for the S5 end-stop search (linkages OFF) restart robot_server.py with "
                "--allow-placeholder-calibration --placeholder-band-us 600,2400"
            )
        arm, _ = self._pulses()
        arm[JOINT_NAMES.index(name)] = us
        reply = self._goto_pulses(arm)
        self.out(f"  {name} -> {us:.0f} us (completed={reply.get('completed')})")

    def cmd_jaw(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("usage: jaw <mm>")
        us = self.cal.width_to_us(float(args[0]) / 1000.0)
        self.client.set_gripper(self.server_cal.us_to_width(us))
        self.out(f"  jaw -> {float(args[0]):.1f} mm ({us:.0f} us)")

    def cmd_torque(self, args: list[str]) -> None:
        if args != ["on"] and args != ["off"]:
            raise ValueError("usage: torque on|off")
        if args == ["off"]:
            self.client.set_torque(False)
            self.out("  torque off: arm is limp. NOTE: the next move/jaw/home/`torque on` re-energises")
            self.out("  ALL FIVE servos (every frame carries every channel), slowly, from the last pulsed pose.")
            return
        state = self.client.get_state()
        if state.get("attached"):
            self.client.set_torque(True)
            self.out("  torque on (servos were already attached; nothing moved)")
            return
        if bridge_position_unknown(state, self.server_cal):
            self._first_attach_home("torque on")
            return
        bridge_q = state.get("bridge_q")
        payload: dict[str, Any] = {"enabled": True}
        if bridge_q is not None:
            payload["q"] = [float(v) for v in bridge_q]
            if state.get("bridge_gripper_width") is not None:
                payload["gripper_width"] = float(state["bridge_gripper_width"])
            where = "at the position the Uno last pulsed"
        else:
            payload["q"] = [float(v) for v in self.server_cal.home_q]
            where = (f"at the last pulsed position, then move to home_q over {self.server_cal.reattach_s:.1f} s+ "
                     "(the bridge cannot report a pose)")
        self.out("  WARNING: `torque on` energises all five servos. TAKE YOUR HAND OFF THE ARM.")
        self.out(f"  They re-attach {where}. Feedback-less servos cannot be read: if the arm")
        self.out("  was moved by hand while limp, every moved servo JUMPS back to its last pulsed")
        self.out("  position at full speed. Only an arm left where it went limp stays still.")
        if not self.confirm("  type yes to energise: "):
            raise ValueError("torque on cancelled: nothing energised (scripts must pass --yes)")
        self.client.rpc.call("set_torque", payload)
        self.out("  torque on")

    def _first_attach_home(self, command: str) -> None:
        """Warn, confirm and send ``home`` to a bridge with no known position."""
        home_deg = [round(math.degrees(v), 1) for v in self.server_cal.home_q]
        self.out("  WARNING: the servo bridge has NO KNOWN POSITION. The Uno resets whenever the")
        self.out("  server opens its port (every robot_server start) and after a USB reset or")
        self.out("  brownout; its first frame attaches EVERY servo AT that frame's target at full")
        self.out("  speed -- there is no position to hold still at or to slew from.")
        self.out(f"  `{command}` therefore sends `home`: all five servos attach AT home_q {home_deg} deg")
        self.out("  instantly. HAND-POSE THE ARM AT HOME FIRST, then take your hand off it.")
        if not self.confirm("  type yes when the arm is at home: "):
            raise ValueError(f"{command} cancelled: nothing energised (scripts must pass --yes)")
        reply = self.client.home()
        self.out(f"  servos attached at home (first attach={reply.get('first_attach')})")

    def cmd_zero(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("usage: zero <joint>")
        name = self._joint(args[0])
        arm, _ = self._pulses()
        j = self.cal.joints[name]
        servo_rad = j.direction * j.us_to_rad(arm[JOINT_NAMES.index(name)]) + j.zero_offset_rad
        self._replace_joint(name, zero_offset_rad=float(servo_rad))
        self.out(f"  {name}: zero_offset_rad = {servo_rad:.4f} (this pose now reads 0 deg)")

    def cmd_limit(self, args: list[str]) -> None:
        if len(args) != 2 or args[1] not in ("lower", "upper"):
            raise ValueError("usage: limit <joint> lower|upper")
        name = self._joint(args[0])
        arm, _ = self._pulses()
        angle = self.cal.joints[name].us_to_rad(arm[JOINT_NAMES.index(name)])
        self._replace_joint(name, **{f"limit_{args[1]}_rad": float(angle)})
        self.out(f"  {name}: limit_{args[1]}_rad = {angle:.4f} ({math.degrees(angle):.1f} deg)")

    def cmd_gripper(self, args: list[str]) -> None:
        if len(args) != 2 or args[0] not in ("open", "closed"):
            raise ValueError("usage: gripper open|closed <mm>")
        _, grip = self._pulses()
        changes = {f"pulse_{args[0]}_us": int(round(grip)), f"width_{args[0]}_m": float(args[1]) / 1000.0}
        self.cal = dataclasses.replace(self.cal, gripper=dataclasses.replace(self.cal.gripper, **changes))
        self.dirty = True
        self.out(f"  gripper {args[0]}: {grip:.0f} us == {float(args[1]):.1f} mm")

    def cmd_home(self, args: list[str]) -> None:
        if args == ["set"]:
            arm, _ = self._pulses()
            home_q = tuple(float(v) for v in self.cal.pulses_to_q(arm))
            # Home is driven to at every boot with no planner: refuse a pose
            # with any link below the table before it can reach the YAML.
            check_home_clearance(home_q, self.geometry)
            self.cal = dataclasses.replace(self.cal, home_q=home_q)  # type: ignore[arg-type]
            self.dirty = True
            self.out(f"  home_q = {[round(v, 4) for v in home_q]}")
        elif not args:
            state = self.client.get_state()
            if not state.get("estopped") and bridge_position_unknown(state, self.server_cal):
                self._first_attach_home("home")
                return
            reply = self.client.home()
            self.out(f"  at home (estopped={reply.get('estopped')})")
        else:
            raise ValueError("usage: home | home set")

    def cmd_measured(self, args: list[str]) -> None:
        if args not in (["yes"], ["no"]):
            raise ValueError("usage: measured yes|no")
        self.cal = dataclasses.replace(self.cal, measured=args == ["yes"])
        self.dirty = True
        if self.cal.measured:
            self.out("  measured: true (after `save`, restart robot_server.py WITHOUT --allow-placeholder-calibration)")
        else:
            self.out("  measured: false (a real driver will refuse it without --allow-placeholder-calibration)")

    def cmd_velocity(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("usage: velocity <rad_s>")
        self.cal = dataclasses.replace(self.cal, max_joint_velocity=float(args[0]))
        self.dirty = True
        self.out(f"  max_joint_velocity = {float(args[0]):.3f} rad/s")

    def cmd_estop(self, _args: list[str]) -> None:
        self.client.estop()
        self.out("  ESTOP: servos detached; `clear` then `home` to recover")

    def cmd_clear(self, _args: list[str]) -> None:
        self.client.clear_estop()
        self.out("  estop cleared (servos re-attach on the next move)")

    def cmd_save(self, _args: list[str]) -> None:
        self.cal.validate(self.geometry)
        reply = self.client.set_calibration(self.cal.to_dict())
        # Re-read: in placeholder mode the server enforces a band-limited copy.
        self._load_server_calibration()
        self.dirty = False
        self.out(f"  saved to {reply.get('path') or '<server memory only>'} (measured: {self.cal.measured})")

    def cmd_help(self, _args: list[str]) -> None:
        self.out(__doc__.split("Commands (``help`` prints them)::", 1)[1].rstrip())

    # -- dispatch ------------------------------------------------------------

    def run(self, line: str) -> bool:
        """Execute one command line; returns False when the session should end."""
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            return True
        verb, args = parts[0].lower(), parts[1:]
        if verb in ("quit", "exit", "q"):
            return False
        handler = getattr(self, f"cmd_{verb}", None)
        if handler is None:
            raise ValueError(f"unknown command {verb!r}; type `help`")
        handler(args)
        return True

    def run_script(self, script: str) -> int:
        """Run ``;``-separated commands; returns the number that failed."""
        failures = 0
        for command in (c.strip() for c in script.split(";") if c.strip()):
            self.out(f"> {command}")
            try:
                if not self.run(command):
                    break
            except (ValueError, CalibrationError, RpcError) as exc:
                failures += 1
                self.out(f"  error: {exc}")
        return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Servo calibration REPL over the Jetson robot server")
    parser.add_argument("--jetson", default="127.0.0.1:5560", metavar="HOST[:PORT]")
    parser.add_argument("--script", default=None, help='non-interactive: "cmd;cmd;..."')
    parser.add_argument("--timeout", type=float, default=5.0, help="request timeout, s")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="answer the `torque on` confirmation with yes (scripts only; a hand on the arm "
        "cannot be checked from here)",
    )
    args = parser.parse_args(argv)

    host, _, port = args.jetson.partition(":")
    client = JetsonClient(host or "127.0.0.1", int(port or 5560), request_timeout_s=args.timeout)
    try:
        client.connect()
    except RpcError as exc:
        print(f"error: {exc}")
        return 2
    heartbeat = Heartbeat(host or "127.0.0.1", int(port or 5560), request_timeout_s=min(args.timeout, 2.0)).start()
    try:
        confirm: Callable[[str], bool] | None = None
        if args.script is not None:
            confirm = (lambda _prompt: True) if args.yes else (lambda _prompt: False)
        calibrator = ServoCalibrator(client, confirm=confirm)
        if args.script is not None:
            failures = calibrator.run_script(args.script)
            if calibrator.dirty:
                print("warning: unsaved edits discarded (add `;save` to the script)")
            return 1 if failures else 0

        print(f"connected to {client.endpoint} (driver={client.server_info.get('driver')}); `help` lists commands")
        calibrator.cmd_read([])
        while True:
            try:
                line = input("cal> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            try:
                if not calibrator.run(line):
                    break
            except (ValueError, CalibrationError, RpcError) as exc:
                print(f"  error: {exc}")
        if calibrator.dirty:
            print("warning: unsaved edits discarded")
        return 0
    finally:
        heartbeat.stop()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

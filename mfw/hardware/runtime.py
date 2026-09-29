"""Runtime assembly for the hardware lane: Jetson arm + detector, no simulator.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers; the network clients import pyzmq/msgpack/cv2 lazily.

Mirrors :class:`mfw.simulation.runtime.Runtime` attribute for attribute, because
``Assistant``, ``run_assistant.py`` and ``TaskPlanner`` read the runtime blind:
``config, events, sim, robot, vision, planner, controller, grasp_scorer,
grasp_generator, memory, skills, executor, cameras, table_top_height, build(),
close()`` and context-manager support. Anything the sim runtime exposes that
has no hardware meaning (``scene_builder``, ``wrist_camera``) is deliberately
absent, so a caller that needs it fails at the attribute rather than getting
a plausible fake.

Bring-up order (each step needs the previous one):

1. ``WallClock`` -- the stepping seam every skill and controller uses.
2. ``JetsonClient.connect`` -- proves the robot server answers *before* any
   model, camera or planner is built; a missing server is the most common
   startup failure and should be reported first and plainly.
3. ``PlanarKinematics`` -> ``RemoteArm`` -> ``RemoteCamera``.
4. ``RemoteDetector.connect`` -> ``PlanarPerception``.
5. First-home gate, then ``robot.go_home_immediate()`` + settle -- the arm
   starts from a known posture, clear of the camera's view of the table.
   Against a real driver (the server's ping says ``fake: false``) whose
   bridge knows no position (``get_state`` -> ``bridge_position_known`` not
   true: every robot_server start on the Uno), that home attaches every servo
   AT home at full speed, so it is never sent silently: a
   :class:`FirstHomeGate` asks the operator to confirm the hand-pose (type
   ``home`` at a TTY, after anything typed earlier is discarded; Ctrl-C or
   EOF aborts before anything moves) or, with no TTY, refuses unless
   ``run_assistant.py --home-confirmed`` pre-confirmed it. The same gate runs
   when the position is known but the servos are detached (``attached``
   false: the server's host timeout after the previous session, a cleared
   estop): that home re-attaches AT the last pulsed pose at full speed first,
   so a limp arm that sagged snaps back. A fake driver, or a real one whose
   servos are attached at a known position, homes as before with no prompt.
6. ``JointSpacePlanner`` -> ``RemoteController`` -> ``GraspScorer`` (reused
   from the sim lane, with ``hand_setback = |tcp_offset_from_hand|``) ->
   ``TopDownGraspGenerator`` -> ``WorkingMemory`` -> ``SkillRegistry`` with
   :data:`HARDWARE_SKILLS` -> ``ClassicalExecutor``.

Refuses to build on an empty ``exterior_camera.homography``: the schema allows
it (``configs/hardware.yaml`` ships it empty on purpose) but an uncalibrated
homography puts every object at a plausible wrong place, and the arm would
then close on air with a confident log.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable, Mapping, TextIO

import numpy as np

from mfw.config.schema import FrameworkConfig
from mfw.core.errors import ConfigurationError, ExecutionError, PerceptionError
from mfw.grasp.scorer import GraspScorer
from mfw.hardware.clock import WallClock
from mfw.hardware.controller import RemoteController
from mfw.hardware.detector import RemoteDetector
from mfw.hardware.grasp import TopDownGraspGenerator
from mfw.hardware.jetson_client import HEARTBEAT_PERIOD_S, JetsonClient
from mfw.hardware.kinematics import PlanarKinematics
from mfw.hardware.perception import PlanarPerception
from mfw.hardware.planner import JointSpacePlanner
from mfw.hardware.remote_arm import RemoteArm
from mfw.hardware.remote_camera import RemoteCamera
from mfw.memory.working_memory import WorkingMemory
from mfw.skills.base import Skill, SkillContext
from mfw.skills.primitives import ALL_SKILLS, FixedCameraScan, HardwareObserve
from mfw.skills.registry import ClassicalExecutor, SkillRegistry
from mfw.utils.logging import EventLogger, get_logger

__all__ = [
    "HardwareRuntime",
    "HARDWARE_SKILLS",
    "HARDWARE_SKILL_OVERRIDES",
    "SIM_ONLY_SKILLS",
    "FirstHomeGate",
    "FirstHomeRefused",
    "HOME_CONFIRMED_FLAG",
    "FIRST_HOME_UNKNOWN",
    "FIRST_HOME_LIMP",
    "FIRST_HOME_CONFIRM_WORDS",
    "first_home_reason",
    "first_home_needs_operator",
    "set_first_home_gate",
]

_log = get_logger("hardware.runtime")

SIM_ONLY_SKILLS: frozenset[str] = frozenset({"look_at", "rotate_wrist"})
"""Skills that need a wrist camera or a wrist roll joint; this arm has neither."""

HARDWARE_SKILL_OVERRIDES: dict[str, type[Skill]] = {
    # One fixed overhead camera already sees the table: "scan the room" looks
    # (several corroborated frames), sweeps the base visibly toward what it
    # found and reports it with directions. The sim ScanScene's wrist-camera
    # viewpoints have no meaning here.
    "scan_scene": FixedCameraScan,
    # Same scene, reported with where each object is and what is out of reach.
    "observe": HardwareObserve,
}
"""Hardware implementations that replace a sim skill of the same name."""

HARDWARE_SKILLS: tuple[type[Skill], ...] = tuple(
    HARDWARE_SKILL_OVERRIDES.get(cls.skill_name, cls)
    for cls in ALL_SKILLS
    if cls.skill_name not in SIM_ONLY_SKILLS
)
"""The registry's skill classes minus :data:`SIM_ONLY_SKILLS`, with
:data:`HARDWARE_SKILL_OVERRIDES` swapped in by name."""


# ----------------------------------------------------------------------
# first home after a bridge reset: the operator confirms the hand-pose
# ----------------------------------------------------------------------


class FirstHomeRefused(ExecutionError):
    """The first home was not confirmed (no TTY and no ``--home-confirmed``,
    EOF at the prompt, or the server is estopped). Nothing was sent to the arm."""


HOME_CONFIRMED_FLAG = "--home-confirmed"
"""The ``run_assistant.py`` flag that pre-confirms the hand-pose for a run with
no terminal (scripted runs, a service unit). Named in every refusal."""


FIRST_HOME_UNKNOWN = "unknown"
""":func:`first_home_reason`: the bridge knows no position (every robot_server
start on the Uno); the first ``home`` attaches every servo AT home at full speed."""

FIRST_HOME_LIMP = "limp"
""":func:`first_home_reason`: the bridge knows its last pulsed pose but the
servos are detached (the server's host timeout after the previous laptop
session ended or crashed, an estop that was cleared, ``torque off``). The
first ``home`` re-attaches every servo AT that last pose at full speed and
only then moves slowly to home, so an arm that sagged while limp snaps back
to where it was (``ServoDriver.reattach``, ``jetson/robot_server.py``)."""


def first_home_reason(server_info: Mapping[str, Any] | None, state: Mapping[str, Any] | None) -> str | None:
    """Why the next ``home`` would jump a real arm at full speed, or ``None``.

    ``server_info`` is the robot server's ``ping`` reply, ``state`` its
    ``get_state`` reply. For a real driver (``fake`` not true):

    * :data:`FIRST_HOME_UNKNOWN` when ``bridge_position_known`` is false *or
      absent* (an older server that cannot say is treated as unknown -- the
      cautious reading);
    * :data:`FIRST_HOME_LIMP` when the position is known but ``attached`` is
      false: the common second-run case, since the server detaches the servos
      ``host_timeout_s`` (5 s) after the previous session's last request
      (fixer, 2026-09-28; before this, a second run homed without a prompt and
      the limp arm snapped back to its last pose).

    A fake driver never needs the operator: nothing can jump.
    """
    if bool((server_info or {}).get("fake")):
        return None
    state = state or {}
    if state.get("bridge_position_known") is not True:
        return FIRST_HOME_UNKNOWN
    if state.get("attached") is False:
        return FIRST_HOME_LIMP
    return None


def first_home_needs_operator(server_info: Mapping[str, Any] | None, state: Mapping[str, Any] | None) -> bool:
    """Whether the next ``home`` would jump a real arm at full speed (see :func:`first_home_reason`)."""
    return first_home_reason(server_info, state) is not None


def _degrees(q: Any) -> str | None:
    if q is None:
        return None
    try:
        return "[" + ", ".join(f"{math.degrees(float(v)):.0f}" for v in q) + "] deg"
    except (TypeError, ValueError):
        return None


def first_home_instructions(context: Mapping[str, Any]) -> str:
    """The hand-pose instructions printed before the first home (and in refusals)."""
    home_text = _degrees(context.get("home_q")) or "robot.home_joint_positions"
    where = f"({context.get('driver') or 'real'} driver at {context.get('endpoint') or 'the robot server'})"
    bar = "!" * 72
    if context.get("reason") == FIRST_HOME_LIMP:
        state = context.get("state") or {}
        last = _degrees(state.get("bridge_q")) or _degrees(state.get("q")) or "its last pulsed pose"
        return (
            "\n" + bar + "\n"
            f"  FIRST HOME -- the servos are DETACHED (limp) {where}:\n"
            "  the previous session ended (host timeout), or an estop / torque off\n"
            f"  left them off (last detach: {state.get('last_detach_reason') or 'unknown'}).\n"
            f"  The first `home` re-attaches EVERY servo AT ITS LAST POSE {last}\n"
            "  at FULL SPEED, then moves slowly to home. An arm that sagged or was moved\n"
            "  while limp snaps back to that pose first. Before you confirm:\n"
            f"    1. Hand-pose the arm at that last pose {last}\n"
            "       (not at home: the first pulse goes to the last pose), and support it\n"
            "       there with the upper arm, not the jaws.\n"
            "    2. E-stop (+6 V switch) within reach of a second person.\n"
            "    3. Other hands, cables and objects clear of the arm's reach.\n"
            + bar
        )
    return (
        "\n" + bar + "\n"
        f"  FIRST HOME -- the servo bridge knows no position {where}.\n"
        "  The first `home` attaches EVERY servo AT home at FULL SPEED: an arm that is\n"
        "  not already there jumps there. Before you confirm:\n"
        f"    1. Hand-pose the arm at home {home_text}: base straight ahead,\n"
        "       upper arm vertical, forearm folded back, jaws up.\n"
        "    2. E-stop (+6 V switch) within reach of a second person.\n"
        "    3. Hands, cables and objects clear of the arm's reach.\n"
        + bar
    )


def _stdin_is_terminal() -> bool:
    """Whether stdin is a terminal a person can type Enter into.

    ``isatty()`` alone is not enough on Windows: ``NUL`` is a character
    device, so ``run_assistant.py < NUL`` (and Git Bash's ``< /dev/null``)
    reports a TTY -- measured on this laptop. There only a console handle has
    a console mode, so ``GetConsoleMode`` decides. If that check itself
    cannot run, the answer stays "terminal": the prompt's EOF refusal still
    guards a non-interactive stdin.
    """
    stdin = sys.stdin
    if stdin is None:
        return False
    try:
        if not stdin.isatty():
            return False
    except (AttributeError, ValueError, OSError):
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            import msvcrt

            handle = msvcrt.get_osfhandle(stdin.fileno())
            mode = ctypes.c_uint32()
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        except Exception:  # noqa: BLE001 - see the docstring
            return True
    return True


def _flush_console_input() -> int:
    """Discard keystrokes typed before the prompt appeared; best effort.

    An Enter pressed during the multi-second bring-up (connecting to the robot
    server, detector and LLM) sits in the console buffer, and :func:`input`
    would read it at once. Windows: drain the console with ``msvcrt.kbhit`` /
    ``getwch`` (returns how many characters were discarded). Elsewhere:
    ``termios.tcflush(stdin, TCIFLUSH)`` (returns 0; the count is unknown).
    Any failure (no console, not a TTY) discards nothing.
    """
    discarded = 0
    try:
        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit() and discarded < 65536:
                msvcrt.getwch()
                discarded += 1
        else:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - see the docstring
        pass
    return discarded


FIRST_HOME_CONFIRM_WORDS = ("home", "yes")
"""What the operator must type at the first-home prompt. A bare Enter (or any
other line) is asked again: an Enter is too easy to have pressed already."""


class FirstHomeGate:
    """Decides whether the first home that would jump a real arm may be sent.

    Called by :meth:`HardwareRuntime.build` only when
    :func:`first_home_needs_operator` is true (no known position, or limp
    servos), right before the home. It returns to allow the home and raises
    :class:`FirstHomeRefused` to refuse it; nothing has been sent to the arm
    either way.

    * ``confirmed=True`` (``run_assistant.py --home-confirmed``): prints the
      instructions as a record and allows it -- for runs with no terminal,
      where a person hand-posed the arm before starting the run.
    * Interactive (stdin is a TTY): prints the instructions, discards anything
      typed before the prompt (``flush``), and waits for the word ``home`` (or
      ``yes``); anything else, a bare Enter included, is asked again, up to
      ``max_asks`` times, then refused. EOF refuses. Ctrl-C propagates as
      ``KeyboardInterrupt`` (under ``run_assistant.py`` its SIGINT handler has
      already sent an estop; the arm was not moved).
    * Otherwise: refuses, naming :data:`HOME_CONFIRMED_FLAG`.

    ``prompt`` (default :func:`input`), ``out`` (default ``sys.stdout``),
    ``interactive`` (default: stdin is a console) and ``flush`` (default:
    :func:`_flush_console_input` when ``prompt`` is the real :func:`input`,
    nothing for an injected prompt) are seams for tests.
    """

    def __init__(
        self,
        confirmed: bool = False,
        prompt: Callable[[str], str] | None = None,
        out: TextIO | None = None,
        interactive: bool | Callable[[], bool] | None = None,
        flush: Callable[[], int | None] | None = None,
        max_asks: int = 3,
    ) -> None:
        self.confirmed = bool(confirmed)
        self._prompt = prompt
        self._out = out
        self._interactive = interactive
        self._flush = flush
        self.max_asks = max(1, int(max_asks))
        self.asked = 0
        """How many times the operator was prompted (tests read it)."""

    def is_interactive(self) -> bool:
        if callable(self._interactive):
            return bool(self._interactive())
        if self._interactive is not None:
            return bool(self._interactive)
        return _stdin_is_terminal()

    def _print(self, text: str) -> None:
        out = self._out if self._out is not None else sys.stdout
        if out is None:
            return
        print(text, file=out, flush=True)

    def _discard_typeahead(self) -> None:
        flush = self._flush
        if flush is None and self._prompt is None:
            flush = _flush_console_input
        if flush is None:
            return
        discarded = flush() or 0
        if discarded:
            self._print(f"  (discarded {discarded} keystroke(s) typed before this prompt)")

    def __call__(self, context: Mapping[str, Any]) -> str:
        """Allow (returns how: ``"flag"`` or ``"operator"``) or raise :class:`FirstHomeRefused`."""
        instructions = first_home_instructions(context)
        if self.confirmed:
            self._print(instructions)
            self._print(f"  {HOME_CONFIRMED_FLAG} given: sending the first home now.")
            return "flag"
        limp = context.get("reason") == FIRST_HOME_LIMP
        if not self.is_interactive():
            why = (
                "the servos are detached, so it would re-attach every servo AT its last pose at full "
                "speed (an arm that sagged while limp snaps back), and there is no terminal to "
                "confirm the arm was hand-posed there. Hand-pose the arm at its last pose"
                if limp else
                "the servo bridge knows no position, so it would attach every servo AT home at full "
                "speed, and there is no terminal to confirm the arm was hand-posed there. Hand-pose "
                "the arm at home"
            )
            raise FirstHomeRefused(
                f"refusing to send the first home: {why} (E-stop in reach, hands clear), then run "
                f"again from a terminal, or pass {HOME_CONFIRMED_FLAG} to run_assistant.py for a "
                "scripted run. Nothing was sent to the arm."
            )
        self._print(instructions)
        prompt = self._prompt if self._prompt is not None else input
        words = " or ".join(FIRST_HOME_CONFIRM_WORDS)
        for _ in range(self.max_asks):
            self._discard_typeahead()
            self.asked += 1
            try:
                answer = prompt(
                    f"  Type {FIRST_HOME_CONFIRM_WORDS[0]} and press Enter to send the first home "
                    "(Ctrl-C aborts without moving): "
                )
            except EOFError:
                raise FirstHomeRefused(
                    "first home not confirmed (end of input at the prompt); nothing was sent to the arm"
                ) from None
            if str(answer or "").strip().lower() in FIRST_HOME_CONFIRM_WORDS:
                return "operator"
            self._print(f"  Not sent: type the word {words} to confirm (Ctrl-C aborts).")
        raise FirstHomeRefused(
            f"first home not confirmed ('{FIRST_HOME_CONFIRM_WORDS[0]}' was not typed after "
            f"{self.max_asks} prompts); nothing was sent to the arm"
        )


_default_first_home_gate: Callable[[Mapping[str, Any]], Any] = FirstHomeGate()


def set_first_home_gate(gate: Callable[[Mapping[str, Any]], Any]) -> Callable[[Mapping[str, Any]], Any]:
    """Install the gate every :class:`HardwareRuntime` built without its own
    ``first_home_gate`` uses; returns the previous one (to restore).

    ``run_assistant.py`` installs ``FirstHomeGate(confirmed=args.home_confirmed)``
    because ``Assistant`` builds the runtime itself. The initial default is
    ``FirstHomeGate()``: prompt at a TTY, refuse otherwise.
    """
    global _default_first_home_gate
    previous = _default_first_home_gate
    _default_first_home_gate = gate
    return previous


class HardwareRuntime:
    """Owns the network clients and every framework object built on them."""

    def __init__(
        self,
        config: FrameworkConfig,
        event_logger: EventLogger | None = None,
        first_home_gate: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """``first_home_gate``: see :class:`FirstHomeGate`; ``None`` uses the
        module default (:func:`set_first_home_gate`), read at :meth:`build`."""
        config.validate()
        self._first_home_gate = first_home_gate
        self.first_home: str | None = None
        """How the bring-up home was allowed: ``"flag"`` / ``"operator"`` (gate),
        ``"not needed"`` (fake driver or a known position); ``None`` before build."""
        if config.backend != "hardware":
            raise ConfigurationError(
                f"HardwareRuntime needs backend: hardware, got {config.backend!r}"
            )
        if len(config.exterior_camera.homography) == 0:
            raise ConfigurationError(
                "exterior_camera.homography is empty: the table has not been calibrated. "
                "Run `py -3.12 scripts/calibrate_table.py --jetson <ip>:5560 --config "
                "configs/hardware.yaml --touch` (or `--image frame.png` and type the table "
                "XY of >= 4 clicked points); it writes exterior_camera.homography into the "
                "YAML itself, keeps a .bak, and `--dry-run` only prints the fit. "
                "configs/hardware_fake.yaml carries the scripted detector's exact matrix "
                "(`scripts/serve_detector.py --fake --print-homography`)."
            )
        self.config = config
        self.events = event_logger or EventLogger(
            log_dir=config.logging.log_dir,
            filename=config.logging.jsonl_filename,
            flush_every=config.logging.flush_every,
            console_level=config.logging.level,
            console=config.logging.console,
        )
        hw = config.hardware
        self.sim = WallClock(
            physics_dt=config.simulation.physics_dt,
            settle_steps=config.simulation.settle_steps,
            fast=hw.fake_clock,
        )
        self.client = JetsonClient(
            hw.jetson_host,
            hw.jetson_port,
            request_timeout_s=hw.request_timeout_s,
            trajectory_timeout_margin_s=hw.trajectory_timeout_margin_s,
        )
        self.detector = RemoteDetector(
            hw.detector_host,
            hw.detector_port,
            labels=hw.labels,
            min_score=hw.detection_min_score,
            timeout_s=hw.request_timeout_s,
        )
        self.kinematics = PlanarKinematics(hw.arm, config.robot.arm_joint_names)

        self.robot: RemoteArm | None = None
        self.exterior_camera: RemoteCamera | None = None
        self.vision: PlanarPerception | None = None
        self.planner: JointSpacePlanner | None = None
        self.controller: RemoteController | None = None
        self.grasp_scorer: GraspScorer | None = None
        self.grasp_generator: TopDownGraspGenerator | None = None
        self.memory: WorkingMemory | None = None
        self.skills: SkillRegistry | None = None
        self.executor: ClassicalExecutor | None = None
        self._built = False

    # ------------------------------------------------------------------

    @property
    def table_top_height(self) -> float:
        """Support plane, by the same rule as the sim runtime.

        ``scene.add_table`` is false on this lane, so this is
        ``perception.ground_plane_z``: the robot base frame's z = 0 *is* the
        table the arm is bolted to.
        """
        if not self.config.scene.add_table:
            return float(self.config.perception.ground_plane_z)
        return float(
            self.config.scene.table_position[2] + self.config.scene.table_scale[2] / 2.0
        )

    @property
    def cameras(self) -> dict[str, RemoteCamera]:
        """All cameras by name (one, fixed, overhead)."""
        if self.exterior_camera is None:
            return {}
        return {self.exterior_camera.name: self.exterior_camera}

    def build(self) -> None:
        """Run the bring-up sequence. Safe to call once."""
        if self._built:
            return
        cfg = self.config
        hw = cfg.hardware

        with self.events.timed("runtime.build"):
            try:
                self.client.connect()
            except ExecutionError as exc:
                raise ExecutionError(f"robot server: {exc}") from exc

            try:
                # Refuses (ConfigurationError) when hardware.arm limits exceed the
                # Jetson's pulse map (review HS-2 / GEO-1); opens the dedicated
                # estop channel with a <= 2 s ping and falls back if it cannot;
                # starts the heartbeat that keeps the Jetson's host-liveness
                # bound from relaxing an idle-but-alive arm, and reports every
                # detach this laptop did not ask for as an event.
                self.robot = RemoteArm(
                    self.client,
                    self.kinematics,
                    cfg.robot,
                    self.sim,
                    event_logger=self.events,
                    heartbeat_s=float(getattr(hw, "heartbeat_s", HEARTBEAT_PERIOD_S)),
                )
            except Exception:
                # Not swallowed: re-raised after the socket is released, so a
                # refused bring-up does not leave a REQ socket (and the process)
                # hanging on exit.
                self.client.close()
                raise
            if cfg.exterior_camera.enabled:
                self.exterior_camera = RemoteCamera(self.client, cfg.exterior_camera, self.sim)

            try:
                self.detector.connect()
            except PerceptionError as exc:
                self.robot.close()
                self.client.close()
                raise PerceptionError(f"detector service: {exc}") from exc

            self.vision = PlanarPerception(
                clock=self.sim,
                detector=self.detector,
                config=cfg.perception,
                hardware=hw,
                homography=cfg.exterior_camera.homography,
                support_height=self.table_top_height,
                workspace_min=cfg.scene.workspace_min,
                workspace_max=cfg.scene.workspace_max,
                camera=cfg.exterior_camera,
                event_logger=self.events,
            )

            try:
                self.first_home = self._confirm_first_home()
            except BaseException:
                # Refused, EOF or Ctrl-C at the prompt: nothing was sent. Release
                # the sockets and the heartbeat before re-raising (close() is
                # idempotent, so a later runtime.close() is harmless).
                self.robot.close()
                self.client.close()
                raise
            self.robot.go_home_immediate()
            self.sim.settle()

            self.planner = JointSpacePlanner(
                robot=self.robot,
                kinematics=self.kinematics,
                config=cfg.motion,
                arm=hw.arm,
                support_height=self.table_top_height,
                event_logger=self.events,
            )
            self.controller = RemoteController(
                sim=self.sim,
                robot=self.robot,
                motion_config=cfg.motion,
                robot_config=cfg.robot,
                hardware_config=hw,
                workspace_min=np.array(cfg.scene.workspace_min),
                workspace_max=np.array(cfg.scene.workspace_max),
                event_logger=self.events,
            )
            self.grasp_scorer = GraspScorer(
                robot=self.robot,
                config=cfg.grasp,
                support_height=self.table_top_height,
                hand_setback=float(np.linalg.norm(cfg.robot.tcp_offset_from_hand)),
            )
            self.grasp_generator = TopDownGraspGenerator(
                config=cfg.grasp,
                hardware=hw,
                support_height=self.table_top_height,
                kinematics=self.kinematics,
            )
            self.memory = WorkingMemory(cfg.memory)
            self.skills = SkillRegistry(
                SkillContext(
                    sim=self.sim,
                    robot=self.robot,
                    vision=self.vision,
                    planner=self.planner,
                    controller=self.controller,
                    grasp_scorer=self.grasp_scorer,
                    memory=self.memory,
                    config=cfg,
                    events=self.events,
                    support_height=self.table_top_height,
                    grasp_generator=self.grasp_generator,
                ),
                skill_classes=HARDWARE_SKILLS,
            )
            self.executor = ClassicalExecutor(self.skills)

        self._built = True
        self.events.emit(
            "runtime.ready",
            {
                "backend": "hardware",
                "robot_server": self.client.endpoint,
                "robot_driver": (self.client.server_info or {}).get("driver"),
                "detector": self.detector.endpoint,
                "detector_backend": self.detector.backend,
                "fake_clock": self.sim.fast,
                "first_home": self.first_home,
                "arm_dofs": self.kinematics.dof,
                "tcp_pose": self.robot.tcp_pose().to_log(),
                "skills": list(self.skills.names),
            },
        )
        _log.info(
            "HardwareRuntime ready (robot %s, detector %s/%s, clock %s)",
            self.client.endpoint, self.detector.endpoint, self.detector.backend,
            "virtual" if self.sim.fast else "wall",
        )

    def _confirm_first_home(self) -> str:
        """Run the first-home gate when the bring-up home would jump a real arm.

        Returns ``"not needed"`` (fake driver, or the bridge knows its position
        and the servos are attached: an ordinary home) or what the gate
        returned. The gate runs for an unknown position AND for limp servos
        (:func:`first_home_reason`); ``context["reason"]`` says which. Raises
        :class:`FirstHomeRefused` -- before anything is sent -- when the gate
        refuses, or when the server is estopped (its ``home`` would be refused
        anyway; the prompt would ask a person to pose the arm for nothing).
        """
        info = self.client.server_info or {}
        if bool(info.get("fake")):
            return "not needed"
        state = self.client.get_state()
        reason = first_home_reason(info, state)
        if reason is None:
            return "not needed"
        if state.get("estopped"):
            raise FirstHomeRefused(
                "the robot server is estopped (an E-stop or a Ctrl-C from an earlier session), so its "
                "servos are detached and its home would be refused. Clear it first: `py -3.12 scripts/calibrate_servos.py "
                f"--jetson {self.client.endpoint.replace('tcp://', '')}` -> `clear` -> `quit` (or "
                "restart mfw-robot), then start again and follow the hand-pose prompt. Nothing was "
                "sent to the arm."
            )
        gate = self._first_home_gate if self._first_home_gate is not None else _default_first_home_gate
        context = {
            "endpoint": self.client.endpoint,
            "driver": info.get("driver"),
            "home_q": [float(v) for v in self.config.robot.home_joint_positions],
            "state": dict(state),
            "reason": reason,
        }
        how = gate(context)
        how = str(how) if how else "operator"
        self.events.emit(
            "runtime.first_home_confirmed",
            {"by": how, "reason": reason, "robot_server": self.client.endpoint,
             "robot_driver": info.get("driver")},
        )
        _log.warning("first home (%s servos) confirmed (%s); sending it",
                     "limp" if reason == FIRST_HOME_LIMP else "unknown-position", how)
        return how

    def emergency_stop(self) -> bool:
        """Detach every servo now, on the arm's dedicated stop channel.

        For the Ctrl-C handler in ``scripts/run_assistant.py``: safe to call
        before :meth:`build` finished (returns ``False``, nothing to stop) and
        never raises -- a failed stop is logged, since the caller is about to
        exit anyway and the +6 V switch remains the real mid-motion stop.
        Returns ``True`` when the Jetson acknowledged the estop.
        """
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        try:
            robot.estop()
        except ExecutionError as exc:
            _log.error("emergency stop was not acknowledged: %s", exc)
            return False
        self.events.emit("runtime.emergency_stop", {"source": "operator"})
        return True

    def close(self) -> None:
        """Close the clients (the arm's stop channel too) and the event log. Idempotent."""
        try:
            robot = getattr(self, "robot", None)
            if robot is not None and hasattr(robot, "close"):
                robot.close()
            self.detector.close()
            self.client.close()
            self.sim.close()
        finally:
            self.events.close()

    def __enter__(self) -> "HardwareRuntime":
        self.build()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

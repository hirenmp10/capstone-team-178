"""scripts/run_assistant.py: compound-command output and the HS-4 bring-up stop.

No Isaac Sim, no robot server, no sockets: the clause runner, the Assistant and
the Jetson client are fakes. What is asserted is what the operator sees and
which stop was sent.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mfw.assistant import expand_clauses
from mfw.core.types import SkillResult, SkillStatus
from mfw.planner.task_planner import CommandOutcome

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_assistant.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("_run_assistant_cli_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restore_sigint():
    previous = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, previous)


def _outcome(clause: str, skill: str, ok: bool, message: str, **params: Any) -> CommandOutcome:
    status = SkillStatus.SUCCESS if ok else SkillStatus.FAILED
    return CommandOutcome(
        utterance=clause,
        skill=skill,
        params=dict(params),
        result=SkillResult(skill_name=skill, status=status, message=message),
        message=message,
        duration_s=1.5,
    )


# ----------------------------------------------------------------------
# item 1: every clause of a compound command is printed
# ----------------------------------------------------------------------


PICK_OK = "picked the red block"
PLACE_MISSED = (
    "place missed on the green box: the red block settled 72 mm from the centre "
    "of the green box, off it"
)


class TestCompoundOutput:
    def _runner(self, script: dict[str, CommandOutcome], ran: list[str]):
        def run_clause(clause: str) -> CommandOutcome:
            ran.append(clause)
            return script[clause]

        return run_clause

    def test_move_x_onto_y_prints_the_pick_and_the_place(self, cli):
        """The run_c evidence: only the place half used to be visible."""
        ran: list[str] = []
        script = {
            "pick the red block": _outcome("pick the red block", "pick", True, PICK_OK,
                                           target="red block"),
            "place it on the green box": _outcome("place it on the green box", "place", False,
                                                  PLACE_MISSED, target="green box", relation="on"),
        }
        report = cli.run_command(
            "move the red block onto the green box", expand_clauses, self._runner(script, ran)
        )
        text = cli.format_report(report)

        assert ran == ["pick the red block", "place it on the green box"]
        assert '"move the red block onto the green box"' in text
        assert "[1/2] OK" in text and '"pick the red block"' in text
        assert "skill    : pick" in text and "'target': 'red block'" in text
        assert PICK_OK in text
        assert "[2/2] --" in text and "skill    : place" in text
        assert "'relation': 'on'" in text and PLACE_MISSED in text
        assert "1 ok, 1 failed, 0 not run" in text
        assert report.final.skill == "place" and not report.ok

    def test_clauses_after_a_failure_are_listed_as_not_run(self, cli):
        ran: list[str] = []
        script = {
            "pick up the blue cube": _outcome(
                "pick up the blue cube", "pick", False,
                "ObjectNotFound: cannot find 'blue cube'", target="blue cube"),
        }
        utterance = "pick up the blue cube and place it on the green box then go home"
        report = cli.run_command(utterance, expand_clauses, self._runner(script, ran))
        text = cli.format_report(report)

        assert ran == ["pick up the blue cube"], "nothing runs after a failed clause"
        assert report.not_run == ("place it on the green box", "go home")
        assert '[2/3] NOT RUN "place it on the green box"' in text
        assert '[3/3] NOT RUN "go home"' in text
        assert "not run  : place it on the green box / go home" in text
        assert "stopped at clause 1 of 3" in text
        assert "0 ok, 1 failed, 2 not run" in text

    def test_all_clauses_ok(self, cli):
        ran: list[str] = []
        script = {
            "open the gripper": _outcome("open the gripper", "open_gripper", True, "opened"),
            "go home": _outcome("go home", "go_home", True, "home"),
        }
        report = cli.run_command("open the gripper then go home", expand_clauses,
                                 self._runner(script, ran))
        text = cli.format_report(report)
        assert report.ok and ran == ["open the gripper", "go home"]
        assert text.startswith('OK  "open the gripper then go home"')
        assert "2 ok, 0 failed, 0 not run" in text and "NOT RUN" not in text
        assert "total    : 3.00s" in text

    def test_single_clause_output_is_unchanged(self, cli):
        """One clause is handed to the runner whole and printed as before."""
        ran: list[str] = []
        outcome = _outcome("what do you see", "observe", True, "observed 3 object(s)")
        report = cli.run_command("what do you see", expand_clauses,
                                 self._runner({"what do you see": outcome}, ran))
        assert ran == ["what do you see"]
        assert cli.format_report(report) == "\n".join(cli.format_outcome("what do you see", outcome))
        assert "clauses" not in cli.format_report(report)

    def test_a_clarified_clause_shows_what_it_ran_as(self, cli):
        clarified = _outcome("pick obj_2", "pick", True, "picked", target="obj_2")
        lines = cli.format_outcome("pick the block", clarified)
        assert "    ran as   : pick obj_2" in lines

    def test_every_cli_path_prints_through_the_report(self, cli):
        """-c/--demo, interactive and voice all go through run_command."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert source.count("run_command(") >= 4  # definition + three call sites
        assert "_print_outcome(utterance, assistant.command(utterance))" not in source


# ----------------------------------------------------------------------
# item 2 (HS-4): Ctrl-C during bring-up sends an estop
# ----------------------------------------------------------------------


class FakeJetsonClient:
    """Stands in for JetsonClient: records construction and calls, dials nothing."""

    instances: list["FakeJetsonClient"] = []

    def __init__(self, host: str, port: int, request_timeout_s: float = 5.0,
                 reply: Any = None, fail: bool = False) -> None:
        self.host, self.port, self.timeout = host, port, request_timeout_s
        self.calls: list[str] = []
        self.reply = {"estopped": True} if reply is None else reply
        self.fail = fail
        FakeJetsonClient.instances.append(self)

    def estop(self) -> dict:
        self.calls.append("estop")
        if self.fail:
            raise RuntimeError("no answer from tcp://%s:%d" % (self.host, self.port))
        return self.reply

    def close(self) -> None:
        self.calls.append("close")


@pytest.fixture
def fake_clients():
    FakeJetsonClient.instances = []
    yield FakeJetsonClient.instances


def _press_ctrl_c() -> None:
    """Invoke the installed SIGINT handler the way the interpreter would."""
    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)


class FakeRuntime:
    def __init__(self, robot: Any = None, acknowledged: bool = True) -> None:
        self.robot = robot
        self.acknowledged = acknowledged
        self.stops = 0
        self.closed = 0

    def emergency_stop(self) -> bool:
        self.stops += 1
        return self.acknowledged

    def close(self) -> None:
        self.closed += 1


class TestBringUpEstop:
    def test_ctrl_c_before_the_arm_exists_sends_a_one_shot_estop(
        self, cli, fake_clients, restore_sigint, capsys
    ):
        """Mid-construction (runtime built, arm not yet): the old handler found
        no assistant and sent nothing."""
        guard = cli.HardwareStopGuard("10.0.0.7", 5560, client_factory=FakeJetsonClient)
        guard.install()
        seen: dict = {}

        class InterruptedAssistant:
            def __init__(self, config=None, llm_complete=None) -> None:
                self.runtime = FakeRuntime(robot=None)  # connected, arm not built
                seen["runtime"] = self.runtime
                _press_ctrl_c()  # the operator hits Ctrl-C here

        with pytest.raises(KeyboardInterrupt):
            guard.construct(InterruptedAssistant, config=None, llm_complete=None)

        assert guard.bringing_up
        assert len(fake_clients) == 1
        client = fake_clients[0]
        assert (client.host, client.port) == ("10.0.0.7", 5560)
        assert client.calls == ["estop", "close"]
        assert client.timeout == cli.ONE_SHOT_ESTOP_TIMEOUT_S
        assert seen["runtime"].stops == 0, "the half-built runtime has no arm to stop"
        out = capsys.readouterr().out
        assert "during bring-up" in out and "tcp://10.0.0.7:5560" in out
        assert "estop acknowledged." in out
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

        guard.release_partial()
        assert seen["runtime"].closed == 1, "the half-built runtime's sockets are released"

    def test_ctrl_c_during_boot_homing_uses_the_arms_dedicated_channel(
        self, cli, fake_clients, restore_sigint, capsys
    ):
        guard = cli.HardwareStopGuard("10.0.0.7", 5560, client_factory=FakeJetsonClient)
        guard.install()
        runtime = FakeRuntime(robot=object())

        class HomingAssistant:
            def __init__(self, **_: Any) -> None:
                self.runtime = runtime
                _press_ctrl_c()  # arm built, homing in progress

        with pytest.raises(KeyboardInterrupt):
            guard.construct(HomingAssistant)

        assert runtime.stops == 1
        assert fake_clients == [], "acknowledged on the dedicated channel: no second stop"
        assert "dedicated stop channel" in capsys.readouterr().out

    def test_an_unacknowledged_arm_stop_is_retried_on_a_fresh_connection(
        self, cli, fake_clients, restore_sigint, capsys
    ):
        guard = cli.HardwareStopGuard("h", 1, client_factory=FakeJetsonClient)
        guard.install()
        runtime = FakeRuntime(robot=object(), acknowledged=False)

        class A:
            def __init__(self) -> None:
                self.runtime = runtime
                _press_ctrl_c()

        with pytest.raises(KeyboardInterrupt):
            guard.construct(A)
        assert runtime.stops == 1 and [c.calls for c in fake_clients] == [["estop", "close"]]
        assert "estop acknowledged." in capsys.readouterr().out

    def test_after_bring_up_ctrl_c_stops_the_finished_assistants_arm(
        self, cli, fake_clients, restore_sigint
    ):
        guard = cli.HardwareStopGuard("h", 1, client_factory=FakeJetsonClient)
        guard.install()
        runtime = FakeRuntime(robot=object())

        class Ready:
            def __init__(self) -> None:
                self.runtime = runtime

        assistant = guard.construct(Ready)
        assert guard.assistant is assistant and not guard.bringing_up
        with pytest.raises(KeyboardInterrupt):
            _press_ctrl_c()
        assert runtime.stops == 1 and fake_clients == []
        guard.release_partial()
        assert runtime.closed == 0, "a finished assistant is closed by main, not the guard"

    def test_a_failed_one_shot_stop_says_to_cut_the_servo_supply(
        self, cli, fake_clients, capsys
    ):
        def failing(host, port, request_timeout_s):
            return FakeJetsonClient(host, port, request_timeout_s, fail=True)

        assert cli._one_shot_estop("h", 9, client_factory=failing) is False
        assert fake_clients[0].calls == ["estop", "close"]
        result = cli._send_interrupt_estop(None, lambda: False, "tcp://h:9")
        out = capsys.readouterr().out
        assert result is False and "+6 V" in out and "NOT acknowledged" in out

    def test_an_estop_reply_that_says_not_estopped_is_not_acknowledged(self, cli, fake_clients):
        def denying(host, port, request_timeout_s):
            return FakeJetsonClient(host, port, request_timeout_s, reply={"estopped": False})

        assert cli._one_shot_estop("h", 9, client_factory=denying) is False

    def test_restore_puts_back_the_previous_handler(self, cli, restore_sigint):
        before = signal.getsignal(signal.SIGINT)
        guard = cli.HardwareStopGuard("h", 1, client_factory=FakeJetsonClient)
        guard.install()
        assert signal.getsignal(signal.SIGINT) is not before
        guard.restore()
        assert signal.getsignal(signal.SIGINT) is before

    def test_main_constructs_the_hardware_assistant_under_the_guard(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "stop_guard.construct(Assistant" in source
        assert "stop_guard.install()" in source and "stop_guard.release_partial()" in source

    def test_the_startup_notice_is_truthful(self, cli):
        text = cli.STOP_NOTICE.format(chunk=0.5)
        assert "between chunks" in text and "0.5 s" in text
        assert "+6 V switch" in text and "mid-motion" in text
        assert "Ctrl-C" in text and "homing" in text


def test_help_runs_without_isaac(cli, capsys):
    """--help parses and exits before anything is imported from Isaac."""
    argv = sys.argv
    sys.argv = ["run_assistant.py", "--help"]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        sys.argv = argv
    assert exc.value.code == 0
    assert "--fake-hardware" in capsys.readouterr().out

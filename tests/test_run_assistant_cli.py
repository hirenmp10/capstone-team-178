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


# ----------------------------------------------------------------------
# MVP: hardware demo, "look around" routing, exit status on failure
# ----------------------------------------------------------------------


class TestHardwarePhraseRouting:
    @pytest.mark.parametrize(
        "utterance",
        ["look around", "Look around!", "scan the table", "scan the room", "please look around the room",
         "can you look around", "have a look around", "scan the table please", "look at the table"],
    )
    def test_scan_phrases_route_to_scan_the_room(self, cli, utterance):
        assert cli.route_hardware_phrase(utterance) == "scan the room"

    @pytest.mark.parametrize(
        "utterance",
        ["pick up the marker", "look at the marker", "what do you see", "look around and pick up the marker",
         "scan the marker", "go home", ""],
    )
    def test_everything_else_is_untouched(self, cli, utterance):
        assert cli.route_hardware_phrase(utterance) == utterance

    def test_the_rule_parser_now_reaches_scan_scene(self, cli):
        from mfw.language.intent_parser import RuleBasedIntentParser

        parser = RuleBasedIntentParser()
        for phrase in ("look around", "scan the table", "scan the room"):
            assert parser.parse(cli.route_hardware_phrase(phrase))["skill"] == "scan_scene", phrase

    def test_routed_applies_per_clause(self, cli):
        ran: list[str] = []
        clauses_of, run_clause = cli.route_clauses(expand_clauses, lambda c: (ran.append(c), _outcome(c, "x", True, "ok"))[1])
        report = cli.run_command("look around and then pick up the marker", clauses_of, run_clause)
        assert ran == ["scan the room", "pick up the marker"]
        assert report.ok


def _report(cli, utterance: str, ok: bool, message: str = ""):
    return cli.CommandReport(utterance, [(utterance, _outcome(utterance, "x", ok, message or utterance))])


class TestScriptedRunExitStatus:
    def test_stops_at_the_first_failure_by_default(self, cli):
        results = {"a": True, "b": False, "c": True}
        ran: list[str] = []

        def run_one(u):
            ran.append(u)
            return _report(cli, u, results[u], "ObjectNotFound: no marker" if not results[u] else "")

        reports, not_run = cli.run_script(["a", "b", "c"], run_one, emit=lambda r: None)
        assert ran == ["a", "b"] and not_run == ("c",)
        assert cli.script_exit_code(reports, not_run) == 1
        text = cli.script_summary(reports, not_run)
        assert "3 -> 1 ok, 1 failed, 1 not run" in text
        assert 'first failure: "b" -> ObjectNotFound: no marker' in text
        assert "--keep-going" in text

    def test_keep_going_runs_everything_and_still_fails(self, cli):
        ran: list[str] = []

        def run_one(u):
            ran.append(u)
            return _report(cli, u, u != "b")

        reports, not_run = cli.run_script(["a", "b", "c"], run_one, keep_going=True, emit=lambda r: None)
        assert ran == ["a", "b", "c"] and not_run == ()
        assert cli.script_exit_code(reports, not_run) == 1

    def test_all_ok_is_zero(self, cli):
        reports, not_run = cli.run_script(["a", "b"], lambda u: _report(cli, u, True), emit=lambda r: None)
        assert cli.script_exit_code(reports, not_run) == 0
        assert "2 -> 2 ok, 0 failed, 0 not run" in cli.script_summary(reports, not_run)

    def test_only_the_hardware_lanes_exit_non_zero(self, cli):
        """Review: the sim lane exited 0 at HEAD whatever its commands did; keep it."""
        reports, not_run = cli.run_script(["a", "b"], lambda u: _report(cli, u, u != "b"),
                                          keep_going=True, emit=lambda r: None)
        assert cli.lane_exit_code(True, reports, not_run) == 1
        assert cli.lane_exit_code(False, reports, not_run) == 0
        source = (Path(cli.__file__)).read_text(encoding="utf-8")
        assert "exit_code = lane_exit_code(hardware_mode, reports, not_run)" in source


class _FakeSkills:
    names = ("go_home", "observe", "pick", "place", "scan_scene")

    def has(self, name: str) -> bool:
        return name in self.names


class _FakeAssistant:
    """Stands in for mfw.assistant.Assistant in main(): no runtime, no sockets."""

    fail: set[str] = set()
    ran: list[str] = []

    def __init__(self, config=None, llm_complete=None) -> None:
        self.runtime = SimpleNamespace(skills=_FakeSkills(), events=SimpleNamespace(path="events.jsonl"))
        self.closed = False

    def describe(self) -> dict:
        return {"skills": list(_FakeSkills.names), "backends": ["classical"], "objects": []}

    def clauses(self, utterance: str) -> list:
        return expand_clauses(utterance)

    def command(self, utterance: str) -> CommandOutcome:
        _FakeAssistant.ran.append(utterance)
        return _outcome(utterance, "x", utterance not in _FakeAssistant.fail, "done")

    def close(self) -> None:
        self.closed = True


def _main(cli, monkeypatch, argv: list[str], fail: set[str] = frozenset()) -> tuple[int, list[str]]:
    import mfw.assistant

    _FakeAssistant.fail = set(fail)
    _FakeAssistant.ran = []
    monkeypatch.setattr(mfw.assistant, "Assistant", _FakeAssistant)
    monkeypatch.setattr(sys, "argv", ["run_assistant.py", *argv])
    code = cli.main()
    return code, list(_FakeAssistant.ran)


class TestMainExitStatus:
    def test_a_failed_command_exits_1_and_stops(self, cli, monkeypatch, restore_sigint, capsys):
        code, ran = _main(cli, monkeypatch,
                          ["--hardware", "-c", "pick up the marker", "-c", "put it in the bowl"],
                          fail={"pick up the marker"})
        assert code == 1
        assert ran == ["pick up the marker"]
        assert "not run  : put it in the bowl" in capsys.readouterr().out

    def test_sim_lane_still_runs_every_command_and_exits_0_as_at_head(self, cli, monkeypatch, restore_sigint,
                                                                      capsys):
        # The sim demo always ran all of its commands and exited 0; only the
        # hardware lane stops at the first failure (a real arm should not keep
        # moving) and exits 1 (review: the sim lane's behaviour is unchanged).
        code, ran = _main(cli, monkeypatch, ["-c", "pick up the marker", "-c", "go home"],
                          fail={"pick up the marker"})
        assert code == 0
        assert ran == ["pick up the marker", "go home"]
        assert "2 -> 1 ok, 1 failed, 0 not run" in capsys.readouterr().out

    def test_keep_going_runs_the_rest_and_still_exits_1(self, cli, monkeypatch, restore_sigint):
        code, ran = _main(cli, monkeypatch, ["--hardware", "--keep-going", "-c", "pick up the marker", "-c", "go home"],
                          fail={"pick up the marker"})
        assert code == 1 and ran == ["pick up the marker", "go home"]

    def test_all_ok_exits_0(self, cli, monkeypatch, restore_sigint):
        code, ran = _main(cli, monkeypatch, ["-c", "go home"])
        assert code == 0 and ran == ["go home"]

    def test_hardware_demo_is_the_mvp_script_with_routing(self, cli, monkeypatch, restore_sigint, capsys):
        code, ran = _main(cli, monkeypatch, ["--hardware", "--demo"])
        assert code == 0
        assert ran == ["scan the room", "what do you see", "pick up the marker", "put it in the bowl", "go home"]
        out = capsys.readouterr().out
        assert "demo     : scan the room -> what do you see" in out
        assert "5 -> 5 ok, 0 failed, 0 not run" in out

    def test_look_around_reaches_scan_on_the_hardware_lane(self, cli, monkeypatch, restore_sigint):
        code, ran = _main(cli, monkeypatch, ["--hardware", "-c", "look around"])
        assert code == 0 and ran == ["scan the room"]

    def test_sim_lane_is_not_routed(self, cli, monkeypatch, restore_sigint):
        _code, ran = _main(cli, monkeypatch, ["-c", "scan the table"])
        assert ran == ["scan the table"]

    def test_builtin_hardware_demo_matches_the_yaml(self, cli):
        from mfw.config.schema import load_config

        hw = load_config(REPO_ROOT / "configs" / "hardware.yaml").hardware
        assert tuple(cli.HARDWARE_DEMO_SCRIPT) == hw.demo_script
        assert "can" not in " ".join(cli.HARDWARE_DEMO_SCRIPT).split()

    def test_fake_scene_has_an_out_of_reach_object(self, cli):
        from jetson.detector_service import parse_scene

        scene = parse_scene(cli.FAKE_SCENE)
        assert {"marker", "bowl", "cube"} <= set(scene)
        x, y = scene["cube"][:2]
        assert y > 0.24 + 0.03, "beyond workspace_max y + margin: seen but out of reach"


class TestRemoteSpeechServerIsNeverAutostarted:
    """``--voice --voice-server <jetson>:5556`` with the Jetson's speech service
    still loading (Canary's warm-up takes a while on a cold board): the laptop
    must wait for it, not probe for a local NeMo install and not launch a local
    worker bound to the Jetson's address (which can only fail)."""

    class _Stop(Exception):
        pass

    def _run(self, cli, monkeypatch, server: str) -> list[str]:
        import mfw.assistant

        calls: list[str] = []

        def _probe(*_a, **_k):
            calls.append("find_voice_python")
            return "python"

        def _autostart(*_a, **_k):
            calls.append("autostart")
            return False

        stop = self._Stop

        class _Boom:
            def __init__(self, *a, **k):
                raise stop()

        monkeypatch.setattr(cli, "_server_is_up", lambda *a, **k: False)
        monkeypatch.setattr(cli, "_find_voice_python", _probe)
        monkeypatch.setattr(cli, "_autostart_speech_server", _autostart)
        monkeypatch.setattr(mfw.assistant, "Assistant", _Boom)
        monkeypatch.setattr(sys, "argv", ["run_assistant.py", "--hardware", "--voice",
                                          "--voice-server", server, "-c", "go home"])
        with pytest.raises(self._Stop):
            cli.main()
        return calls

    def test_remote_server_is_waited_for_not_started(self, cli, monkeypatch, restore_sigint, capsys):
        assert self._run(cli, monkeypatch, "10.9.8.7:5556") == []
        assert "remote: not autostarted" in capsys.readouterr().out

    def test_loopback_server_is_still_autostarted(self, cli, monkeypatch, restore_sigint):
        assert self._run(cli, monkeypatch, "127.0.0.1:5556") == ["find_voice_python", "autostart"]


# ----------------------------------------------------------------------
# first home: --home-confirmed and the no-terminal refusal
# ----------------------------------------------------------------------


class _RealArmAssistant(_FakeAssistant):
    """Builds like ``Assistant`` against a real driver whose bridge knows no
    position: ``HardwareRuntime.build`` calls the installed module gate before
    the home. ``homes`` counts the homes that would have been sent."""

    homes = 0
    contexts: list = []

    def __init__(self, config=None, llm_complete=None) -> None:
        from mfw.hardware import runtime as runtime_mod

        context = {"endpoint": "tcp://10.0.0.5:5560", "driver": "uno",
                   "home_q": [0.0, 0.0, -0.6109, 0.1745], "state": {"bridge_position_known": False}}
        _RealArmAssistant.contexts.append(context)
        runtime_mod._default_first_home_gate(context)  # raises FirstHomeRefused to refuse
        _RealArmAssistant.homes += 1
        super().__init__(config, llm_complete)


class TestFirstHomeFlag:
    def _run(self, cli, monkeypatch, argv: list[str]) -> int:
        import io

        import mfw.assistant

        _RealArmAssistant.homes = 0
        _RealArmAssistant.contexts = []
        _FakeAssistant.ran = []
        _FakeAssistant.fail = set()
        monkeypatch.setattr(mfw.assistant, "Assistant", _RealArmAssistant)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # a scripted run: no terminal
        monkeypatch.setattr(sys, "argv", ["run_assistant.py", *argv])
        return cli.main()

    def test_no_terminal_without_the_flag_refuses_before_moving(self, cli, monkeypatch, restore_sigint, capsys):
        from mfw.hardware import runtime as runtime_mod

        before = runtime_mod._default_first_home_gate
        code = self._run(cli, monkeypatch, ["--hardware", "-c", "go home"])
        out = capsys.readouterr().out
        assert code == 1
        assert _RealArmAssistant.homes == 0 and _FakeAssistant.ran == []
        assert "FIRST HOME NOT SENT" in out and "--home-confirmed" in out and "Nothing was sent" in out
        assert "Traceback" not in out
        assert runtime_mod._default_first_home_gate is before, "main() must restore the module gate"

    def test_home_confirmed_proceeds_and_runs_the_commands(self, cli, monkeypatch, restore_sigint, capsys):
        from mfw.hardware import runtime as runtime_mod

        before = runtime_mod._default_first_home_gate
        code = self._run(cli, monkeypatch, ["--hardware", "--home-confirmed", "-c", "go home"])
        out = capsys.readouterr().out
        assert code == 0
        assert _RealArmAssistant.homes == 1 and _FakeAssistant.ran == ["go home"]
        assert "--home-confirmed given" in out and "Hand-pose the arm at home" in out
        assert "1st home : pre-confirmed (--home-confirmed)" in out
        assert runtime_mod._default_first_home_gate is before

    def test_the_banner_says_the_first_home_will_be_confirmed(self, cli, monkeypatch, restore_sigint, capsys):
        self._run(cli, monkeypatch, ["--hardware", "-c", "go home"])
        assert "1st home : a real arm with no known position or limp servos asks you to type 'home'" in capsys.readouterr().out

    def test_sim_lane_never_installs_a_home_gate(self, cli, monkeypatch, restore_sigint):
        from mfw.hardware import runtime as runtime_mod

        installed: list = []
        monkeypatch.setattr(runtime_mod, "set_first_home_gate", lambda gate: installed.append(gate))
        code, ran = _main(cli, monkeypatch, ["-c", "go home"])
        assert code == 0 and ran == ["go home"] and installed == []

    def test_help_documents_the_flag(self, cli, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run_assistant.py", "--help"])
        with pytest.raises(SystemExit):
            cli.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--home-confirmed" in text and "not a terminal" in text

"""The robotic assistant: the top-level entry point.

Isaac Sim is reached only through :class:`~mfw.simulation.runtime.Runtime`;
with ``backend: hardware`` the same seams are filled by
:class:`~mfw.hardware.runtime.HardwareRuntime` and nothing from Isaac is
imported at all (see :func:`_build_runtime`).

Wires the whole stack together::

    text / voice -> intent parser -> task planner -> executor (classical | GR00T)
                                  -> skills -> motion -> PhysX

and exposes one method that matters: :meth:`command`. Each clause the operator
actually said is one action, then the robot waits. Nothing here invents a step
-- the planner drives a single skill per clause and returns to
``WAIT_FOR_COMMAND``; the clauses come only from the operator's own words.

The GR00T backend is attached only if it is enabled *and* its server answers.
A configured-but-unreachable policy degrades to the classical backend with a
warning rather than making the robot unusable.

**Clauses.** One utterance can carry several atomic actions, and each is run as
its own planner command, in order, stopping at the first that does not succeed
(the failed outcome is returned with ``remaining_clauses`` naming what was not
run). Two sources:

* conjunctions -- "pick up the block and place it in the box" (``split_conjoined``);
* moving a named object -- "move the red block to the bowl" becomes "pick the
  red block" then "place it to the bowl" (:func:`~mfw.language.intent_parser.split_transfer`).
  The audit measured the old reading of "move the red object to the left":
  ``move_relative{left}``, the object dropped and the empty arm moved.

**Dialogue helpers** (module level, pure, unit-tested without a runtime):
:func:`run_with_clarification` answers ``needs_clarification`` by asking "Which
one do you mean: ...?", matching the reply and re-running the command with the
chosen object; :func:`requires_confirmation` decides from the PARSED skill
whether a spoken command must be confirmed (the old gate was a substring list
that missed "get", "fetch", "set", "shift", "turn", "survey", "look at",
"return home", "squeeze", "let go" ...); :func:`classify_confirmation` reads a
yes/no reply by whole words ("I know" is not "no").
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Sequence

from mfw.config.schema import FrameworkConfig, load_config
from mfw.core.interfaces import ISkillExecutor
from mfw.language.grounding import (
    clarification_question,
    interpret_clarification,
    substitute_referent,
)
from mfw.language.intent_parser import (
    IIntentParser,
    LlmIntentParser,
    RuleBasedIntentParser,
    split_transfer,
)
from mfw.language.speech import ISpeechRecognizer, TextRecognizer, VoiceCommandLoop
from mfw.planner.task_planner import CommandOutcome, TaskPlanner
from mfw.utils.logging import get_logger

__all__ = [
    "Assistant",
    "DEFAULT_CONFIG_PATH",
    "split_conjoined",
    "expand_clauses",
    "run_clauses",
    "requires_confirmation",
    "classify_confirmation",
    "clarified_command",
    "run_with_clarification",
    "NON_MOTION_SKILLS",
]

_log = get_logger("assistant")

#: The shipped configuration, used when no config is supplied.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# ----------------------------------------------------------------------
# clauses
# ----------------------------------------------------------------------

_CONJUNCTION = re.compile(
    r"\b(?:and\s+then|then|after\s+that|and\s+(?=(?:place|put|set|move|go|drop|open|close|rotate"
    r"|wait|look|pick|grab|take|lift|bring|transfer|carry|return|observe|scan)\b))\b",
    re.IGNORECASE,
)


def split_conjoined(utterance: str) -> list[str]:
    """Split "pick up the block and place it in the box" into its clauses.

    "and" splits only before a command verb, so "pick the block between the
    can and the box" stays one clause.
    """
    parts = [p.strip() for p in _CONJUNCTION.split(utterance) if p and p.strip()]
    return parts if len(parts) > 1 else [utterance]


def expand_clauses(utterance: str) -> list[str]:
    """Every atomic clause of ``utterance``, in order (conjunctions, then transfers)."""
    clauses: list[str] = []
    for clause in split_conjoined(utterance):
        split = split_transfer(clause)
        if split is not None and split.pick_clause is not None:
            clauses.extend(split.clauses)
        else:
            clauses.append(clause)
    return clauses


def run_clauses(
    clauses: Sequence[str],
    handle: Callable[[str], CommandOutcome],
    before_each: Callable[[], None] | None = None,
) -> CommandOutcome:
    """Run clauses in order through ``handle``; stop at the first that is not ok.

    The returned outcome is the last one run, its ``duration_s`` the total, and
    ``remaining_clauses`` the clauses that were not attempted -- so a caller
    that resolves a clarification can resume exactly where the command stopped.
    """
    if not clauses:
        raise ValueError("no clauses to run")
    total = 0.0
    outcome: CommandOutcome | None = None
    for index, clause in enumerate(clauses):
        if before_each is not None:
            before_each()
        outcome = handle(clause)
        total += outcome.duration_s
        if not outcome.ok:
            outcome.remaining_clauses = tuple(clauses[index + 1:])
            if len(clauses) > 1 and index + 1 < len(clauses):
                _log.info(
                    "Clause %d/%d (%r) did not succeed; not running %s",
                    index + 1, len(clauses), clause, list(clauses[index + 1:]),
                )
            break
    assert outcome is not None
    outcome.duration_s = total
    return outcome


# ----------------------------------------------------------------------
# voice confirmation gate
# ----------------------------------------------------------------------

#: Skills that change nothing in the world. Everything else -- including skills
#: added later -- moves the arm or the gripper and is confirmed when spoken.
NON_MOTION_SKILLS = frozenset({"observe", "wait", "stop", "emergency_stop"})


def requires_confirmation(skill: str | None) -> bool:
    """Whether a spoken command that parsed to ``skill`` must be confirmed first.

    Decided by the parsed skill, not by keywords in the transcript: the parser
    is what will actually act, so it is the only honest judge of whether the
    arm is about to move. ``None`` (did not parse) moves nothing.
    """
    return skill is not None and skill not in NON_MOTION_SKILLS


_YES_WORDS = frozenset(
    {"yes", "yeah", "yep", "yup", "yea", "sure", "correct", "ok", "okay", "affirmative",
     "confirm", "confirmed", "right", "proceed", "absolutely", "definitely"}
)
_YES_PHRASES = ("go ahead", "do it", "thats right", "that is right", "sounds good")
_NO_WORDS = frozenset(
    {"no", "nope", "nah", "cancel", "stop", "wrong", "negative", "abort", "not", "dont",
     "incorrect", "never"}
)
_NO_PHRASES = ("never mind", "do not")


def classify_confirmation(reply: str) -> bool | None:
    """``True`` for yes, ``False`` for no, ``None`` for neither or both.

    Whole words only: the old substring test read "I know" and "nothing" as
    "no" and "yesterday" as "yes". Both at once ("no... yes") is ``None`` so the
    operator is asked again -- a mixed answer is not consent.
    """
    text = str(reply or "").lower().replace("'", "").replace("’", "")
    text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    words = set(text.split())
    padded = f" {text} "
    said_yes = bool(words & _YES_WORDS) or any(f" {p} " in padded for p in _YES_PHRASES)
    said_no = bool(words & _NO_WORDS) or any(f" {p} " in padded for p in _NO_PHRASES)
    if said_yes and not said_no:
        return True
    if said_no and not said_yes:
        return False
    return None


# ----------------------------------------------------------------------
# clarification dialogue
# ----------------------------------------------------------------------

_VERB_FOR_SKILL = {"pick": "pick", "look_at": "look at", "move_to": "move to"}


def clarified_command(outcome: CommandOutcome, track_id: str) -> str:
    """The command to re-run once the operator chose ``track_id``.

    The ambiguous phrase is replaced by the track id in the clause that
    stopped (a track id is exact: it cannot be re-grounded to something else),
    and the clauses that never ran are appended with "then".
    """
    phrase = outcome.clarification_phrase or str(outcome.params.get("target") or "")
    clause = substitute_referent(outcome.utterance, phrase, track_id)
    if clause is None:
        # The parser paraphrased the phrase (an LLM can); rebuild from the intent.
        skill = outcome.skill or "pick"
        if skill == "place":
            relation = str(outcome.params.get("relation") or "next_to")
            prep = {"next_to": "next to", "left_of": "to the left of",
                    "right_of": "to the right of", "in_front_of": "in front of"}.get(relation, relation)
            clause = f"place it {prep} {track_id}"
        else:
            clause = f"{_VERB_FOR_SKILL.get(skill, skill.replace('_', ' '))} {track_id}"
    return " then ".join([clause, *outcome.remaining_clauses])


def run_with_clarification(
    command: Callable[[str], CommandOutcome],
    utterance: str,
    ask: Callable[[str], str | None],
    *,
    scene_provider: Callable[[], Any] | None = None,
    robot_xy: tuple[float, float] = (0.0, 0.0),
    notify: Callable[[str], None] | None = None,
    max_questions: int = 3,
) -> CommandOutcome:
    """Run ``utterance``; when it needs clarification, ask and re-run.

    ``ask(question)`` returns the operator's reply (``None`` = no reply, which
    cancels). Replies are matched by :func:`interpret_clarification`: "the red
    one", "the can", "the one on the left", "the second one"; "cancel" or
    "never mind" abandons the command. An answer that matches nothing is asked
    again, up to ``max_questions`` times in total, then abandoned -- never
    guessed. A re-run that is itself ambiguous asks again (same budget).
    """
    outcome = command(utterance)
    questions = 0
    while outcome.needs_clarification and outcome.clarification_options:
        options = outcome.clarification_options
        track_ids = outcome.clarification_track_ids
        if len(track_ids) != len(options):
            # Legacy options without ids cannot be re-run exactly; say so.
            if notify is not None:
                notify(clarification_question(options) + " Please repeat the command naming it.")
            return outcome
        question = clarification_question(options)
        chosen: int | None = None
        while chosen is None:
            if questions >= max_questions:
                if notify is not None:
                    notify("I still do not know which one, so I will not do anything.")
                return outcome
            questions += 1
            reply = ask(question)
            if reply is None:
                return outcome
            scene = scene_provider() if scene_provider is not None else None
            choice = interpret_clarification(
                reply, options, track_ids=track_ids, scene=scene, robot_xy=robot_xy
            )
            if choice.kind == "cancel":
                if notify is not None:
                    notify("Cancelled.")
                return outcome
            if choice.kind == "choice" and choice.index is not None:
                chosen = choice.index
            else:
                question = f"Sorry, I did not catch that. {clarification_question(options)}"
        if notify is not None:
            notify(f"OK, the {options[chosen]}.")
        outcome = command(clarified_command(outcome, track_ids[chosen]))
    return outcome


def _build_runtime(config: FrameworkConfig) -> Any:
    """The runtime for ``config.backend``, imported only when chosen.

    Lazy on purpose: the hardware lane runs on a laptop (or the Jetson) with
    no Isaac Sim, and importing the simulation runtime there would drag in
    modules that assume one exists. Both runtimes expose the same attributes,
    so nothing below this line knows which one it got.
    """
    if config.backend == "hardware":
        from mfw.hardware.runtime import HardwareRuntime

        return HardwareRuntime(config)
    from mfw.simulation.runtime import Runtime

    return Runtime(config)


class Assistant:
    """A natural-language robotic manipulation assistant."""

    def __init__(
        self,
        config: FrameworkConfig | None = None,
        config_path: str | Path | None = None,
        llm_complete: Callable[[str], str] | None = None,
    ) -> None:
        if config is None:
            # Fall back to the packaged config, not to bare dataclass defaults.
            # Those defaults describe an *empty* scene (SceneConfig.objects is
            # ``()``), so Assistant() with no arguments would bring up a robot
            # with nothing to manipulate and every command would correctly but
            # confusingly report that it cannot see anything.
            config = load_config(config_path if config_path else DEFAULT_CONFIG_PATH)
        config.validate()
        self.config = config

        self.runtime = _build_runtime(config)
        self.runtime.build()

        skill_names = self.runtime.skills.names
        self.parser: IIntentParser = (
            LlmIntentParser(llm_complete, skill_names)
            if llm_complete is not None
            else RuleBasedIntentParser(skill_names)
        )

        self.executors: dict[str, ISkillExecutor] = {"classical": self.runtime.executor}
        self._gr00t_client: Any = None
        if config.gr00t.enabled:
            self._attach_gr00t()

        self.planner = TaskPlanner(
            parser=self.parser,
            executors=self.executors,
            memory=self.runtime.memory,
            config=config,
            vision=self.runtime.vision,
            events=self.runtime.events,
        )

        # Seed memory so the very first command can resolve a reference without
        # the operator having to say "observe" first.
        #
        # Twice, deliberately: the tracker requires corroboration across two
        # frames before confirming a track (a single-frame blob is usually a
        # segmentation artefact), so one observation always reports an empty
        # scene and the robot would claim to see nothing at startup.
        self.runtime.skills.execute("observe")
        self.runtime.skills.execute("observe")
        self._prime_colors()
        _log.info(
            "Assistant ready (backends: %s, default: %s)",
            ", ".join(sorted(self.executors)),
            config.default_executor,
        )

    #: Bound on the observes spent waiting for object colours to settle at start-up.
    _MAX_COLOR_PRIMING_OBSERVES = 3

    def _prime_colors(self) -> None:
        """Observe again (bounded) until every object's colour has been confirmed.

        Two observes confirm the *tracks*; they do not confirm the *colours*.
        Colour is the one attribute the renderer can get wrong early without
        anything else noticing -- depth and segmentation were correct in the
        runs where the first "what do you see" came back "black block, black
        can, black box". Asks the perception layer whether each confirmed
        object's latest frame agreed with the colour it reports, and observes
        again while it did not. A perception backend that does not measure
        colour (the hardware lane) has no ``colors_settled`` and is skipped, so
        its start-up is unchanged.
        """
        settled = getattr(self.runtime.vision, "colors_settled", None)
        if not callable(settled):
            return
        for _ in range(self._MAX_COLOR_PRIMING_OBSERVES):
            if settled():
                return
            self.runtime.skills.execute("observe")
        if not settled():
            _log.warning(
                "Object colours had not settled after %d extra start-up observes",
                self._MAX_COLOR_PRIMING_OBSERVES,
            )

    # ------------------------------------------------------------------

    def _attach_gr00t(self) -> None:
        """Connect the policy backend, or carry on without it."""
        from mfw.core.errors import PolicyError
        from mfw.gr00t_bridge.executor import Gr00tExecutor

        # NVIDIA's own ZeroMQ/msgpack protocol when available, falling back to
        # the pickle transport that serves the mock server in tests. Both satisfy
        # IPolicyClient, so nothing downstream changes.
        if self.config.gr00t.use_mock_server:
            from mfw.gr00t_bridge.client import Gr00tTcpClient

            client = Gr00tTcpClient(self.config.gr00t)
        else:
            from mfw.gr00t_bridge.zmq_client import Gr00tZmqClient

            client = Gr00tZmqClient(self.config.gr00t)
        try:
            client.connect()
        except PolicyError as exc:
            # Degrade rather than fail: a missing policy server should not take the
            # gripper and the motion planner down with it.
            _log.warning(
                "GR00T is enabled but the server is unreachable (%s); "
                "continuing with the classical backend only",
                exc,
            )
            return

        self._gr00t_client = client
        self.executors["gr00t"] = Gr00tExecutor(
            client=client,
            robot=self.runtime.robot,
            vision=self.runtime.vision,
            controller=self.runtime.controller,
            cameras=self.runtime.cameras,
            config=self.config,
            memory=self.runtime.memory,
            events=self.runtime.events,
            # The classical backend's SkillContext, so a GR00T pick or place can
            # clear the classical held_grasp record it would otherwise leave stale.
            skill_context=getattr(self.runtime.skills, "_context", None),
        )
        _log.info("GR00T backend attached")

    # ------------------------------------------------------------------

    @staticmethod
    def _split_conjoined(utterance: str) -> list[str]:
        """Kept for callers of the old private name; see :func:`split_conjoined`."""
        return split_conjoined(utterance)

    def clauses(self, utterance: str) -> list[str]:
        """The atomic clauses :meth:`command` will run for ``utterance``, in order."""
        return expand_clauses(utterance)

    def command(self, utterance: str) -> CommandOutcome:
        """Execute natural-language command(s): each clause in order, stopping at
        the first that does not succeed."""
        return run_clauses(
            self.clauses(utterance),
            self.planner.handle,
            before_each=lambda: self.runtime.memory.update_robot(self.runtime.robot.get_state()),
        )

    def planned_skills(self, utterance: str) -> list[str | None]:
        """The skill each clause parses to, without executing anything.

        ``None`` for a clause that does not parse (it would move nothing).
        """
        skills: list[str | None] = []
        for clause in self.clauses(utterance):
            intent = self.planner.preview(clause)
            skills.append(None if intent is None else intent.skill)
        return skills

    def needs_confirmation(self, utterance: str) -> bool:
        """Whether a spoken ``utterance`` would move the robot (see
        :func:`requires_confirmation`)."""
        return any(requires_confirmation(skill) for skill in self.planned_skills(utterance))

    def command_with_clarification(
        self,
        utterance: str,
        ask: Callable[[str], str | None],
        notify: Callable[[str], None] | None = None,
        max_questions: int = 3,
    ) -> CommandOutcome:
        """:meth:`command`, asking "which one?" when a referent is ambiguous."""
        base = self.config.robot.base_position
        return run_with_clarification(
            self.command,
            utterance,
            ask,
            scene_provider=lambda: self.runtime.memory.current_scene,
            robot_xy=(float(base[0]), float(base[1])),
            notify=notify,
            max_questions=max_questions,
        )

    def run_text_loop(self, stream: Any = None, max_commands: int | None = None) -> int:
        """Read typed commands until the stream ends."""
        loop = VoiceCommandLoop(TextRecognizer(stream=stream), self.command)
        return loop.run(max_commands=max_commands)

    def run_voice_loop(
        self, recognizer: ISpeechRecognizer, max_commands: int | None = None
    ) -> int:
        """Read spoken commands.

        Voice and text share the same path from the transcript onward, so there is
        only one command pipeline to reason about.
        """
        return VoiceCommandLoop(recognizer, self.command).run(max_commands=max_commands)

    def describe(self) -> dict[str, Any]:
        """Current state: what the robot sees, holds, and can do."""
        scene = self.runtime.memory.current_scene
        return {
            "state": self.planner.machine.state.value,
            "skills": list(self.runtime.skills.names),
            "backends": sorted(self.executors),
            "held_object": self.runtime.memory.get_held_object(),
            "objects": (
                [
                    {"track_id": o.track_id, "label": o.label,
                     "position": o.pose.position.round(4).tolist()}
                    for o in scene.objects.values()
                ]
                if scene is not None
                else []
            ),
        }

    def close(self) -> None:
        if self._gr00t_client is not None:
            self._gr00t_client.close()
        self.runtime.memory.persist()
        self.runtime.close()

    def __enter__(self) -> "Assistant":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

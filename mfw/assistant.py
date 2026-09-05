"""The robotic assistant: the top-level entry point.

Isaac Sim is reached only through :class:`~mfw.simulation.runtime.Runtime`.

Wires the whole stack together::

    text / voice -> intent parser -> task planner -> executor (classical | GR00T)
                                  -> skills -> motion -> PhysX

and exposes one method that matters: :meth:`command`. One utterance in, one action
performed, then the robot waits. Nothing here can chain commands -- the planner
drives a single skill and returns to ``WAIT_FOR_COMMAND``.

The GR00T backend is attached only if it is enabled *and* its server answers.
A configured-but-unreachable policy degrades to the classical backend with a
warning rather than making the robot unusable.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

from mfw.config.schema import FrameworkConfig, load_config
from mfw.core.interfaces import ISkillExecutor
from mfw.language.intent_parser import IIntentParser, LlmIntentParser, RuleBasedIntentParser
from mfw.language.speech import ISpeechRecognizer, TextRecognizer, VoiceCommandLoop
from mfw.planner.task_planner import CommandOutcome, TaskPlanner
from mfw.simulation.runtime import Runtime
from mfw.utils.logging import get_logger

__all__ = ["Assistant", "DEFAULT_CONFIG_PATH"]

_log = get_logger("assistant")

#: The shipped configuration, used when no config is supplied.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


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

        self.runtime = Runtime(config)
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
        _log.info(
            "Assistant ready (backends: %s, default: %s)",
            ", ".join(sorted(self.executors)),
            config.default_executor,
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
        )
        _log.info("GR00T backend attached")

    # ------------------------------------------------------------------

    @staticmethod
    def _split_conjoined(utterance: str) -> list[str]:
        """Split conjoined commands like 'pick up the block and place it in the box'."""
        pattern = r"\b(?:and\s+then|then|after\s+that|and\s+(?=(?:place|put|set|move|go|drop|open|close|rotate|wait|look)))\b"
        parts = re.split(pattern, utterance, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if len(parts) > 1 else [utterance]

    def command(self, utterance: str) -> CommandOutcome:
        """Execute natural-language command(s), supporting conjoined actions."""
        clauses = self._split_conjoined(utterance)
        if len(clauses) > 1:
            total_duration = 0.0
            last_outcome = None
            for clause in clauses:
                self.runtime.memory.update_robot(self.runtime.robot.get_state())
                outcome = self.planner.handle(clause)
                total_duration += outcome.duration_s
                last_outcome = outcome
                if not outcome.ok:
                    outcome.duration_s = total_duration
                    return outcome
            if last_outcome is not None:
                last_outcome.duration_s = total_duration
                return last_outcome

        self.runtime.memory.update_robot(self.runtime.robot.get_state())
        return self.planner.handle(utterance)

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

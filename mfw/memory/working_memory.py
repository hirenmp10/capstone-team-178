"""Working memory and reference resolution.

Pure stdlib + NumPy. This module must never import Isaac Sim.

This is what makes "place it" resolvable. The referent of a pronoun is decided in
a fixed priority order, and where the answer is genuinely ambiguous the memory
says so rather than guessing -- a wrong referent means the robot confidently
manipulates the wrong object.

**Memory stores identities, never poses.** It remembers *which* object is held or
was last discussed, as a ``track_id``; where that object is gets re-perceived
every time. Caching a pose here would mean acting on a remembered position after
the object had moved, which is exactly the class of bug the perception-first rule
exists to prevent.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

from mfw.config.schema import MemoryConfig
from mfw.core.interfaces import IMemory
from mfw.core.types import RobotState, SceneGraph, SkillResult
from mfw.utils.logging import get_logger

__all__ = ["WorkingMemory", "CommandRecord"]

_log = get_logger("memory")


@dataclass
class CommandRecord:
    """One command and what came of it."""

    utterance: str
    skill: str
    params: dict[str, Any]
    status: str = "pending"
    referenced_track_ids: tuple[str, ...] = ()
    wall_time: float = field(default_factory=time.time)

    def to_log(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "skill": self.skill,
            "params": self.params,
            "status": self.status,
            "referenced_track_ids": list(self.referenced_track_ids),
        }


class WorkingMemory(IMemory):
    """Robot and scene state that persists across commands."""

    def __init__(self, config: MemoryConfig) -> None:
        config.validate()
        self.config = config
        self._scene_history: Deque[SceneGraph] = deque(maxlen=config.max_scene_history)
        self._command_history: Deque[CommandRecord] = deque(maxlen=config.max_command_history)
        self._result_history: Deque[SkillResult] = deque(maxlen=config.max_command_history)
        self._held_object: str | None = None
        self._last_referenced: str | None = None
        self._robot_state: RobotState | None = None
        self._pronouns = {word.strip().lower() for word in config.pronoun_words}

    # ------------------------------------------------------------------
    # state updates
    # ------------------------------------------------------------------

    def update_scene(self, scene: SceneGraph | None) -> None:
        if scene is None:
            return
        self._scene_history.append(scene)
        # A held object that has vanished from perception for good must not stay
        # "held" forever, or every later command inherits a phantom payload.
        if self._held_object is not None and scene.get(self._held_object) is None:
            recently_seen = any(
                past.get(self._held_object) is not None
                for past in list(self._scene_history)[-5:]
            )
            if not recently_seen:
                _log.info(
                    "Held object %s has not been perceived recently; clearing",
                    self._held_object,
                )
                self._held_object = None

    def update_robot(self, state: RobotState) -> None:
        self._robot_state = state

    def set_held_object(self, track_id: str | None) -> None:
        if track_id != self._held_object:
            _log.info("Held object: %s -> %s", self._held_object, track_id)
        self._held_object = track_id
        if track_id is not None:
            # The thing just picked up is the most natural referent for "it".
            self._last_referenced = track_id

    def get_held_object(self) -> str | None:
        return self._held_object

    def note_reference(self, track_id: str) -> None:
        """Record that a command referred to this object."""
        self._last_referenced = track_id

    def record_command(self, record: CommandRecord) -> None:
        self._command_history.append(record)

    def record_result(self, result: SkillResult) -> None:
        self._result_history.append(result)
        if self._command_history and self._command_history[-1].status == "pending":
            self._command_history[-1].status = result.status.value

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def current_scene(self) -> SceneGraph | None:
        return self._scene_history[-1] if self._scene_history else None

    @property
    def robot_state(self) -> RobotState | None:
        return self._robot_state

    @property
    def last_referenced(self) -> str | None:
        return self._last_referenced

    def recent_commands(self, count: int = 5) -> list[CommandRecord]:
        return list(self._command_history)[-count:]

    def recent_results(self, count: int = 5) -> list[SkillResult]:
        return list(self._result_history)[-count:]

    # ------------------------------------------------------------------
    # reference resolution
    # ------------------------------------------------------------------

    def is_pronoun(self, phrase: str) -> bool:
        return phrase.strip().lower() in self._pronouns

    def resolve_reference(self, phrase: str) -> str | None:
        """Resolve a natural-language referent to a ``track_id``.

        Priority order, and the reasoning behind it:

        1. **An exact track id** -- already unambiguous.
        2. **A pronoun while holding something** -- if the gripper is full, "it" is
           overwhelmingly what is in the gripper.
        3. **A pronoun otherwise** -- the object most recently referred to.
        4. **A unique class match in the current scene** -- "the can" when there is
           exactly one can.
        5. **Otherwise ``None``** -- including when a class matches several
           objects. Ambiguity must surface as a question, never a guess.
        """
        if not phrase:
            return None
        needle = phrase.strip().lower()
        scene = self.current_scene

        if scene is not None and phrase in scene.objects:
            return phrase

        if self.is_pronoun(needle):
            if self._held_object is not None:
                return self._held_object
            if self._last_referenced is not None and (
                scene is None or scene.get(self._last_referenced) is not None
            ):
                return self._last_referenced
            return None

        if scene is None:
            return None

        # Strip common articles so "the can" matches the class "can".
        for article in ("the ", "a ", "an ", "that ", "this "):
            if needle.startswith(article):
                needle = needle[len(article) :]
                break

        matches = scene.by_label(needle)
        if len(matches) == 1:
            return matches[0].track_id
        if len(matches) > 1:
            _log.debug("%r matches %d objects; refusing to guess", phrase, len(matches))
        return None

    def candidates_for(self, phrase: str) -> list[str]:
        """All track ids a phrase could mean, for asking a disambiguating question."""
        scene = self.current_scene
        if scene is None or not phrase:
            return []
        needle = phrase.strip().lower()
        for article in ("the ", "a ", "an "):
            if needle.startswith(article):
                needle = needle[len(article) :]
                break
        return [obj.track_id for obj in scene.by_label(needle)]

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view of memory, for logging and debugging."""
        scene = self.current_scene
        return {
            "held_object": self._held_object,
            "last_referenced": self._last_referenced,
            "num_scene_observations": len(self._scene_history),
            "current_objects": (
                {track_id: obj.label for track_id, obj in scene.objects.items()}
                if scene is not None
                else {}
            ),
            "gripper": (
                self._robot_state.gripper.to_log() if self._robot_state is not None else None
            ),
            "tcp_pose": (
                self._robot_state.tcp_pose.to_log() if self._robot_state is not None else None
            ),
            "recent_commands": [c.to_log() for c in self.recent_commands()],
        }

    def persist(self, path: str | Path | None = None) -> Path | None:
        """Write the snapshot to disk. Best-effort; never raises into the robot."""
        target = Path(path or self.config.persist_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
            return target
        except OSError as exc:  # pragma: no cover - filesystem dependent
            _log.warning("Could not persist memory to %s: %s", target, exc)
            return None

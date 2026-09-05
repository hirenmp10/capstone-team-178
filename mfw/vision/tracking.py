"""Multi-frame instance tracking.

Pure NumPy + stdlib. This module must never import Isaac Sim.

Tracks give objects a stable identity across observations, which is what the
memory layer resolves pronouns against: "place it" needs *it* to still mean the
same object three commands later, even though every pose has been re-measured
since.

Association is global (Hungarian-style optimal assignment) rather than greedy.
Greedy nearest-neighbour swaps identities when two similar objects sit close
together -- and swapping the identity of the object currently in the gripper is
about the worst failure this layer can produce.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mfw.core.types import ObjectHypothesis

__all__ = ["Track", "ObjectTracker"]


@dataclass
class Track:
    """One tracked object across time."""

    track_id: str
    hypothesis: ObjectHypothesis
    hits: int = 1
    misses: int = 0
    first_seen_step: int = 0
    position_history: list[NDArray[np.float64]] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        """Whether this track has been seen enough times to act on.

        A single-frame detection is often a segmentation artefact; requiring
        corroboration keeps the planner from chasing noise.
        """
        return self.hits >= 2


class ObjectTracker:
    """Associates per-frame detections with persistent track identities.

    Handles the four things the brief requires of a scene that is not static:
    objects may move (association by proximity), rotate (pose is replaced, not
    averaged), disappear (tracks age out), and be added (unmatched detections
    become new tracks).
    """

    def __init__(
        self,
        match_distance: float,
        max_age_steps: int,
        id_prefix: str = "obj",
    ) -> None:
        if match_distance <= 0.0:
            raise ValueError("match_distance must be > 0")
        self.match_distance = float(match_distance)
        self.max_age_steps = int(max_age_steps)
        self._tracks: dict[str, Track] = {}
        self._counter = itertools.count(1)
        self._id_prefix = id_prefix

    @property
    def tracks(self) -> dict[str, Track]:
        return dict(self._tracks)

    def confirmed_objects(self) -> dict[str, ObjectHypothesis]:
        """Hypotheses for tracks stable enough to plan against."""
        return {
            track_id: track.hypothesis
            for track_id, track in self._tracks.items()
            if track.is_confirmed
        }

    def update(
        self, detections: list[ObjectHypothesis], step_index: int
    ) -> dict[str, ObjectHypothesis]:
        """Associate ``detections`` with existing tracks and return the live set.

        Detections arrive with placeholder ``track_id``s; this assigns the real,
        stable ones.
        """
        matches, unmatched_detections = self._associate(detections)
        matches, unmatched_detections = self._reidentify_by_class(
            matches, unmatched_detections, detections
        )

        for track_id, detection_index in matches.items():
            track = self._tracks[track_id]
            detection = detections[detection_index]
            # Replace rather than smooth: a rotated or nudged object must report
            # where it is *now*. Filtering here would lag reality and the arm
            # would arrive at a stale pose.
            detection.track_id = track_id
            track.hypothesis = detection
            track.hits += 1
            track.misses = 0
            track.position_history.append(detection.pose.position.copy())

        for detection_index in unmatched_detections:
            self._spawn_track(detections[detection_index], step_index)

        matched_ids = set(matches)
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            track.misses += 1
            age = step_index - track.hypothesis.last_seen_step
            if age > self.max_age_steps:
                # Objects may legitimately be removed from the scene.
                del self._tracks[track_id]

        return self.confirmed_objects()

    def _reidentify_by_class(
        self,
        matches: dict[str, int],
        unmatched_detections: list[int],
        detections: list[ObjectHypothesis],
    ) -> tuple[dict[str, int], list[int]]:
        """Recover identity for objects that moved further than the distance gate.

        The proximity gate exists to stop identity swaps between neighbours, but
        it is far too tight for the displacements this robot actually causes: a
        pick-and-place moves an object 20-30 cm between two observations, well
        beyond a 6 cm gate, and the object would come back as a stranger --
        breaking "place it" for the very object just manipulated.

        The safe relaxation is uniqueness. If exactly one unmatched track and
        exactly one unmatched detection share a class, no ambiguity exists and
        they must be the same object. Where two candidates of a class remain,
        this deliberately does nothing and lets them become new tracks, because
        a wrong re-identification is worse than a lost one.
        """
        if not unmatched_detections:
            return matches, unmatched_detections

        unmatched_track_ids = [tid for tid in self._tracks if tid not in matches]
        if not unmatched_track_ids:
            return matches, unmatched_detections

        def label_of_track(track_id: str) -> str:
            return self._tracks[track_id].hypothesis.label.strip().lower()

        remaining = list(unmatched_detections)
        for label in {detections[i].label.strip().lower() for i in remaining}:
            if not label:
                continue
            candidate_tracks = [t for t in unmatched_track_ids if label_of_track(t) == label]
            candidate_detections = [
                i for i in remaining if detections[i].label.strip().lower() == label
            ]
            if len(candidate_tracks) == 1 and len(candidate_detections) == 1:
                matches[candidate_tracks[0]] = candidate_detections[0]
                remaining.remove(candidate_detections[0])
                unmatched_track_ids.remove(candidate_tracks[0])

        return matches, remaining

    def _spawn_track(self, detection: ObjectHypothesis, step_index: int) -> None:
        track_id = f"{self._id_prefix}_{next(self._counter):03d}"
        detection.track_id = track_id
        self._tracks[track_id] = Track(
            track_id=track_id,
            hypothesis=detection,
            first_seen_step=step_index,
            position_history=[detection.pose.position.copy()],
        )

    def _associate(
        self, detections: list[ObjectHypothesis]
    ) -> tuple[dict[str, int], list[int]]:
        """Optimally assign detections to tracks under the distance gate."""
        track_ids = list(self._tracks)
        if not track_ids or not detections:
            return {}, list(range(len(detections)))

        cost = np.full((len(track_ids), len(detections)), np.inf)
        for i, track_id in enumerate(track_ids):
            track_position = self._tracks[track_id].hypothesis.pose.position
            for j, detection in enumerate(detections):
                distance = float(np.linalg.norm(detection.pose.position - track_position))
                if distance > self.match_distance:
                    continue
                # Same-class evidence breaks ties between two equidistant
                # candidates; it never overrides the distance gate.
                label_penalty = 0.0 if _labels_match(track_id, self._tracks, detection) else 0.25
                cost[i, j] = distance / self.match_distance + label_penalty

        assignment = _optimal_assignment(cost)
        matches = {track_ids[i]: j for i, j in assignment.items()}
        unmatched = [j for j in range(len(detections)) if j not in set(assignment.values())]
        return matches, unmatched


def _labels_match(
    track_id: str, tracks: dict[str, Track], detection: ObjectHypothesis
) -> bool:
    existing = tracks[track_id].hypothesis.label.strip().lower()
    incoming = detection.label.strip().lower()
    if not existing or not incoming:
        return True
    return existing == incoming


def _optimal_assignment(cost: NDArray[np.float64]) -> dict[int, int]:
    """Minimum-cost assignment, ignoring infinite-cost pairs.

    Uses SciPy's Hungarian solver when available and falls back to a greedy
    sweep otherwise. SciPy ships with Isaac Sim but the fallback keeps this
    module runnable in a bare interpreter, which is the whole point of the
    no-Isaac rule.
    """
    finite_rows = np.any(np.isfinite(cost), axis=1)
    if not np.any(finite_rows):
        return {}

    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        # The solver cannot handle inf; substitute a value that is always worse
        # than any real match, then discard those pairs afterwards.
        finite_values = cost[np.isfinite(cost)]
        big = (float(finite_values.max()) + 1.0) * 10.0 if finite_values.size else 1.0
        padded = np.where(np.isfinite(cost), cost, big)
        rows, cols = linear_sum_assignment(padded)
        return {
            int(r): int(c)
            for r, c in zip(rows, cols)
            if np.isfinite(cost[r, c])
        }
    except ImportError:  # pragma: no cover - SciPy present in Isaac
        assignment: dict[int, int] = {}
        used_cols: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if not np.isfinite(cost[r, c]):
                break
            if r in assignment or c in used_cols:
                continue
            assignment[r] = c
            used_cols.add(c)
        return assignment

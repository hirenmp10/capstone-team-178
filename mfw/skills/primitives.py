"""The atomic skill library.

Isaac Sim is reached only through injected components; this module imports none
of it directly.

Every skill here is independently executable and **ends by returning**. None of
them calls another. In particular:

* ``Pick`` observes, plans a grasp, approaches, closes, verifies, lifts, and then
  **holds**. It does not place.
* ``Place`` finds a destination, lowers, releases, retreats, and then **stops**.
  It does not return home.

Manipulation is pure PhysX throughout: no parenting, no teleporting, no pose
writing, no fake attachment. Objects move only because the fingers push on them.

Object references go through :func:`mfw.language.grounding.resolve_reference`
(colour, class synonyms, size, position, relations; pronouns via memory). It
raises ``ObjectNotFound`` / ``AmbiguousReference`` and never guesses; those
propagate out of the skill (``Skill.execute`` turns them into a FAILED result
carrying ``error_type``/``candidates``/``track_ids``), so the planner asks
"which one?" instead of retrying. Measured on the default scene before this
(logs/e2e/run_a_default.txt, 2026-09-24): "pick the red object", "pick the
large object" and "pick the object" (three in view) all failed with
ObjectNotFound while the robot was looking at them.

Placement (``Place``) implements the parser's place contract: ``{}`` (back
where it was picked), ``{"relation": in|on|next_to|to|left_of|right_of|
in_front_of|behind, "target": phrase}`` and ``{"relation": "direction",
"direction": left|right|forward|back, "distance": m}``. Measured traps it now
handles:

* ``next_to`` was always +Y of the target and unclamped: "place it next to
  the green box" released at y=0.461 (the far edge of reach) and the skill
  reported *success* with "settled 350 mm from the target" (events
  20260924_083548_0ab54f). The side is now chosen from the target's four
  sides: inside the workspace, clear of other objects, nearest the robot.
* The release height beside or on something was ``surface + 0.03`` whatever
  was held; a 125 mm can gripped at its middle was driven 32 mm into the
  table. It is now the held object's own half height, as ``place it`` always
  used.
* The aim correction was measured in the carry orientation and applied after
  the transit re-oriented the wrist (audit probe: a 25 mm offset gave 19/35/50
  mm landing error at 45/90/180 deg of yaw, worse than no correction). The
  object's offset is now stored in the gripper frame at the lift and rotated
  into each candidate release orientation.
* Success was unconditional. A place is now judged from the re-observation:
  inside the container footprint for ``in``, on top for ``on``, within
  ``grasp.place_tolerance_m`` (default 5 cm) otherwise. A miss, or an object
  not seen after release, is a FAILED result with the measured numbers,
  marked ``retryable=False``: the object is already released, so a retry
  could only answer "not holding anything" and bury the measurement.
* "Move the red block onto the green box" released the block 3 mm from where
  it was aimed and it still fell off the box (Isaac, events
  20260924_104645_0cc762 seq 158-162). The aim, the gripper-frame correction
  and the height were right: the *destination centre* was wrong. The green box
  was perceived from the exterior camera alone at (0.6575, 0.3164), 146 x 123
  mm, in every observation of that run, while the one wrist-camera view that
  saw it put it at (0.614, 0.295) and the scene spawns it at (0.62, 0.28), 120
  mm square. The earlier "next to the green box" place in the same run landed
  the can *on* the box's near edge (z 0.540), which is only possible with the
  box's near face at x < 0.564, about 40 mm nearer than perceived. Released
  over the perceived centre, the block sat 21-23 mm inside the real far corner
  on both axes, less than its own 24-30 mm half size, and tipped off onto the
  table 72 mm out. On the sim lane (wrist camera) ``on``/``in`` now look again
  from the pose above the destination and re-centre the release on what that
  closer view measures (:func:`refine_place_target`), within
  :data:`PLACE_REFINE_MAX_SHIFT_M` and only when that view is not truncated.
* ``Pick``/``Place`` look once more before refusing an unknown reference. In
  the same run "pick the blue cube" listed only the red block as visible: the
  can's and box's tracks had aged out while the arm worked elsewhere, and the
  next observation re-detected all three but held two as unconfirmed
  (``ObjectTracker.is_confirmed`` needs two hits; seq 46: 3 raw instances, 1
  object in the scene; seq 49: all three, as new tracks obj_004/obj_005).
* Hardware lane (no gripper feedback): the transit never falls back to an
  unchecked straight line (re-review: 12 of 108 bin crossings drove a held
  cube 37-41 mm into the bin), and an exception between the close and the
  verdict leaves the jaw closed instead of letting a retry open it at height.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mfw.core.errors import ObjectNotFound, SafetyViolation
from mfw.core.types import Frame, ObjectHypothesis, Pose, SceneGraph, SkillResult
from mfw.grasp.generator import generate_grasp_candidates
from mfw.language.grounding import describe, resolve_reference
from mfw.physics.contact import verify_grasp
from mfw.skills.base import Skill
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = [
    "Observe",
    "ScanScene",
    "LookAt",
    "MoveTo",
    "MoveRelative",
    "Pick",
    "Place",
    "OpenGripper",
    "CloseGripper",
    "RotateWrist",
    "GoHome",
    "Wait",
    "Stop",
    "EmergencyStop",
    "ALL_SKILLS",
    "HeldGrasp",
    "PlaceTarget",
    "PlacementCheck",
    "PLACE_RELATIONS",
    "PLACE_DIRECTIONS",
    "DEFAULT_PLACE_TOLERANCE_M",
    "DEFAULT_PLACE_DISTANCE_M",
    "NEXT_TO_GAP_M",
    "normalise_relation",
    "relation_for_to",
    "half_extent_along",
    "footprint_gap",
    "inside_footprint",
    "offset_in_gripper",
    "aim_shift_xy",
    "synthesise_place_target",
    "release_tcp_position",
    "refine_place_target",
    "PLACE_REFINE_MAX_SHIFT_M",
    "PLACE_REFINE_MIN_EXTENT_RATIO",
    "assess_placement",
]

_log = get_logger("skills.primitives")


def _clamp_to_workspace(ctx: Any, position: NDArray[np.float64], margin: float = 0.01) -> NDArray[np.float64]:
    """Clip a target position into the configured workspace.

    Skills must not *command* a pose the controller will reject: a retreat that
    overshoots the ceiling by a millimetre would raise ``SafetyViolation`` and
    abort the whole command, when the intent -- "get clear" -- was perfectly
    achievable a hair lower. The controller's check stays as the real guard; this
    just keeps well-intentioned motions inside it.
    """
    low = np.asarray(ctx.config.scene.workspace_min, dtype=np.float64) + margin
    high = np.asarray(ctx.config.scene.workspace_max, dtype=np.float64) - margin
    return np.clip(np.asarray(position, dtype=np.float64), low, high)


def _robot_xy(ctx: Any) -> tuple[float, float]:
    """The robot base in the world frame: the origin for nearest/front/behind."""
    base = getattr(getattr(ctx.config, "robot", None), "base_position", (0.0, 0.0, 0.0))
    return float(base[0]), float(base[1])


def _resolve_object(
    ctx: Any,
    params: dict[str, Any],
    key: str = "target",
    *,
    scene: SceneGraph | None = None,
    exclude_ids: Iterable[str] = (),
) -> ObjectHypothesis:
    """Resolve a spoken referent to exactly one perceived object.

    An exact track id (what a re-run after "which one?" sends) is taken as is.
    Everything else -- class, colour, size, position, relations, pronouns via
    memory -- goes through :func:`resolve_reference`, whose
    ``ObjectNotFound`` / ``AmbiguousReference`` propagate unchanged: a
    reference error must become a question or a single refusal, and is never
    caught here to fall back to a narrower matcher.
    """
    reference = params.get(key)
    if reference is None or not str(reference).strip():
        raise ObjectNotFound(f"no {key} specified", phrase="")
    reference = str(reference).strip()
    if scene is None:
        scene = ctx.vision.require_fresh_scene()
    excluded = set(exclude_ids)
    if reference in scene.objects and reference not in excluded:
        return scene.objects[reference]
    return resolve_reference(
        reference, scene, memory=ctx.memory, robot_xy=_robot_xy(ctx), exclude_ids=excluded
    )


def _resolve_target(
    ctx: Any,
    params: dict[str, Any],
    key: str = "target",
    *,
    scene: SceneGraph | None = None,
    exclude_ids: Iterable[str] = (),
) -> str:
    """Resolve a target reference to a track id (see :func:`_resolve_object`)."""
    return _resolve_object(ctx, params, key, scene=scene, exclude_ids=exclude_ids).track_id


def _observe_and_resolve(ctx: Any, resolve: Callable[[SceneGraph], Any]) -> tuple[SceneGraph, Any]:
    """Observe, then ``resolve(scene)``; on ``ObjectNotFound`` look once more.

    An object that was out of every camera's view long enough loses its track,
    and the observation that re-detects it holds it as unconfirmed (one hit),
    so it is missing from that scene. Measured: "pick the blue cube" after a
    pick and place answered "currently visible: red block" with the can and
    the box in view (events 20260924_104645_0cc762 seq 46; the next
    observation, seq 49, had all three). The second look confirms such tracks;
    its scene is the one returned, so nothing downstream plans against the
    first. ``AmbiguousReference`` is not retried: it is a question.
    """
    ctx.sim.render_step(2)
    scene = ctx.vision.observe()
    try:
        return scene, resolve(scene)
    except ObjectNotFound as first:
        _log.debug("reference not found (%s); observing once more before refusing", first)
    ctx.sim.render_step(2)
    scene = ctx.vision.observe()
    return scene, resolve(scene)


def _accepts_keyword(func: Any, name: str) -> bool:
    """Whether ``func`` takes keyword ``name`` (the hardware planner's ``check_objects``)."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


# ----------------------------------------------------------------------
# placement geometry (pure; unit-tested in tests/test_place_logic.py)
# ----------------------------------------------------------------------

#: The place relations the parser emits (``mfw.language.intent_parser.PLACE_RELATIONS``).
PLACE_RELATIONS = frozenset(
    {"in", "on", "next_to", "to", "left_of", "right_of", "in_front_of", "behind"}
)

#: Direction words for ``{"relation": "direction"}``, as unit XY vectors in the
#: robot frame: +X forward (away from the base), +Y the robot's left.
PLACE_DIRECTIONS: dict[str, tuple[float, float]] = {
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
}

_DIRECTION_ALIASES = {"backward": "back", "backwards": "back", "forwards": "forward",
                      "ahead": "forward"}

_RELATION_ALIASES = {
    "inside": "in", "into": "in", "within": "in",
    "onto": "on", "on_top_of": "on", "atop": "on", "on_top": "on",
    "beside": "next_to", "next": "next_to", "near": "next_to", "by": "next_to",
    "close_to": "next_to", "nextto": "next_to",
    "front_of": "in_front_of", "infront_of": "in_front_of",
}

#: Default for a place with no ``distance`` (matches the parser's
#: ``DEFAULT_PLACE_OFFSET_M``).
DEFAULT_PLACE_DISTANCE_M = 0.10

#: How far a place may settle from its target and still count, for every
#: relation judged by distance (next to, a side, a direction, back where it was
#: picked, an explicit point). ``grasp.place_tolerance_m`` overrides it when
#: the config defines one. 5 cm: about twice the worst "place it" error
#: measured on the sim lane (8-26 mm), far under the 95-350 mm misses that
#: used to be reported as success.
DEFAULT_PLACE_TOLERANCE_M = 0.05

#: Free gap between the held object and the destination's side for next to /
#: left of / right of / in front of / behind. With a 70 mm can beside the
#: 146 mm green box the centres end 138 mm apart, inside the scene graph's
#: 150 mm ``next_to`` distance, so the result is also *perceived* as next to.
NEXT_TO_GAP_M = 0.03

#: The largest re-centring the close look above an ``on``/``in`` destination
#: may apply. The measured single-view error it corrects was 48 mm (green box,
#: exterior camera 0.6575/0.3164 against the wrist view's 0.614/0.295); a
#: bigger jump is a different object or a bad frame, not a better estimate.
PLACE_REFINE_MAX_SHIFT_M = 0.08

#: The close view is used only when each of its horizontal extents is at least
#: this fraction of the first estimate's (sorted, so yaw does not matter). The
#: held object hangs in front of the wrist camera; a view it truncates shrinks
#: the box from one side and shifts the centre the wrong way.
PLACE_REFINE_MIN_EXTENT_RATIO = 0.75

#: Minimum free gap to every *other* object at a table-level release spot.
PLACE_OBJECT_CLEARANCE_M = 0.01

#: Keep release spots this far inside the workspace walls.
PLACE_EDGE_MARGIN_M = 0.02

#: A direction place that the workspace clamp would move further than this is
#: refused: "10 cm to the left" must not quietly become "4 cm to the left".
DIRECTION_CLAMP_LIMIT_M = 0.01

#: Class words that make "put it to the X" mean *in* X ...
PLACE_CONTAINER_CLASSES = frozenset(
    {"bowl", "bin", "basket", "tray", "container", "bucket", "crate", "pot"}
)
#: ... and those that make it mean *on* X, when X is flat (no taller than wide).
PLACE_SURFACE_CLASSES = frozenset(
    {"box", "block", "brick", "cube", "book", "plate", "board", "platform", "shelf"}
)


def normalise_relation(relation: Any) -> str:
    """Canonical relation name: ``"inside"`` -> ``"in"``, ``"beside"`` -> ``"next_to"``."""
    text = "_".join(str(relation or "").strip().lower().split())
    return _RELATION_ALIASES.get(text, text)


def normalise_direction(direction: Any) -> str:
    """Canonical direction word: ``"backward"`` -> ``"back"``."""
    text = str(direction or "").strip().lower()
    return _DIRECTION_ALIASES.get(text, text)


def _label_words(obj: ObjectHypothesis) -> set[str]:
    return set(str(obj.label or "").lower().replace("_", " ").replace("-", " ").split())


def relation_for_to(destination: ObjectHypothesis) -> str:
    """What "put it to the X" means: ``in`` a container, ``on`` a flat box/block, else ``next_to``."""
    words = _label_words(destination)
    if words & PLACE_CONTAINER_CLASSES:
        return "in"
    extents = np.asarray(destination.bbox.extents, dtype=np.float64)
    if words & PLACE_SURFACE_CLASSES and float(extents[2]) <= float(np.max(extents[:2])):
        return "on"
    return "next_to"


def _footprint_axes(obj: ObjectHypothesis) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The object's box X and Y axes projected onto the table, as unit XY vectors."""
    rotation = obj.bbox.center.rotation_matrix()
    axes = []
    for column, fallback in ((0, (1.0, 0.0)), (1, (0.0, 1.0))):
        axis = np.asarray(rotation[:2, column], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        axes.append(axis / norm if norm > 1e-6 else np.asarray(fallback, dtype=np.float64))
    return axes[0], axes[1]


def half_extent_along(obj: ObjectHypothesis, direction_xy: ArrayLike) -> float:
    """Half the width of ``obj``'s table footprint along a horizontal direction."""
    d = np.asarray(direction_xy, dtype=np.float64)[:2]
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    ax, ay = _footprint_axes(obj)
    ex, ey = (float(e) for e in np.asarray(obj.bbox.extents, dtype=np.float64)[:2])
    return 0.5 * (abs(float(d @ ax)) * ex + abs(float(d @ ay)) * ey)


def _local_xy(obj: ObjectHypothesis, xy: ArrayLike) -> NDArray[np.float64]:
    ax, ay = _footprint_axes(obj)
    delta = np.asarray(xy, dtype=np.float64)[:2] - np.asarray(
        obj.bbox.center.position, dtype=np.float64
    )[:2]
    return np.array([float(delta @ ax), float(delta @ ay)])


def footprint_gap(obj: ObjectHypothesis, xy: ArrayLike, radius: float = 0.0) -> float:
    """Free gap between a disc (``xy``, ``radius``) and ``obj``'s footprint; negative overlaps."""
    local = np.abs(_local_xy(obj, xy))
    half = np.asarray(obj.bbox.extents, dtype=np.float64)[:2] / 2.0
    d = local - half
    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
    inside = min(float(np.max(d)), 0.0)
    return outside + inside - float(radius)


def inside_footprint(obj: ObjectHypothesis, xy: ArrayLike, margin: float = 0.0) -> bool:
    """Whether ``xy`` lies within ``obj``'s table footprint (grown by ``margin``)."""
    local = np.abs(_local_xy(obj, xy))
    half = np.asarray(obj.bbox.extents, dtype=np.float64)[:2] / 2.0 + float(margin)
    return bool(np.all(local <= half))


def _top_of(obj: ObjectHypothesis) -> float:
    return float(obj.bbox.center.position[2] + obj.bbox.extents[2] / 2.0)


def _held_radius(held_extents: ArrayLike | None) -> float:
    """Half the held object's longest horizontal side: its footprint at any yaw it may take."""
    if held_extents is None:
        return 0.03
    return 0.5 * float(np.max(np.asarray(held_extents, dtype=np.float64)[:2]))


def offset_in_gripper(object_centre: ArrayLike, tcp_pose: Pose) -> NDArray[np.float64]:
    """``object_centre - tcp`` expressed in the TCP (gripper) frame."""
    world = np.asarray(object_centre, dtype=np.float64) - np.asarray(
        tcp_pose.position, dtype=np.float64
    )
    return tcp_pose.rotation_matrix().T @ world


def aim_shift_xy(offset_gripper: ArrayLike | None, quat: ArrayLike) -> NDArray[np.float64]:
    """Where the held object's centre sits relative to the TCP, in world XY, at ``quat``."""
    if offset_gripper is None:
        return np.zeros(2)
    world = tf.quat_to_matrix(quat) @ np.asarray(offset_gripper, dtype=np.float64)
    return np.asarray(world[:2], dtype=np.float64)


@dataclass(frozen=True)
class HeldGrasp:
    """What ``Pick`` measured about the object it now holds.

    ``offset_in_gripper`` is the object's centre minus the TCP, in the TCP
    frame, observed right after the verified lift (``None`` when the object
    was not seen or the offset was implausible). It is kept in the gripper
    frame because that is the frame the object is rigidly held in: ``Place``
    rotates it into whichever release orientation it ends up using.
    """

    track_id: str
    offset_in_gripper: tuple[float, float, float] | None
    grasp_quat: tuple[float, float, float, float]
    rest_extents: tuple[float, float, float] | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "offset_in_gripper": None if self.offset_in_gripper is None
            else [round(v, 4) for v in self.offset_in_gripper],
            "grasp_quat": [round(v, 4) for v in self.grasp_quat],
            "rest_extents": None if self.rest_extents is None
            else [round(v, 4) for v in self.rest_extents],
        }


@dataclass(frozen=True)
class PlaceTarget:
    """Where the held object itself should come to rest.

    ``object_xy`` is the object's centre on release; ``surface_z`` the height
    it rests on. ``tcp_z`` fixes the TCP height instead (the verified ``in``
    drop over a rim, and an explicit ``position``). ``check`` names the test
    applied after release: ``"inside"`` (the container's footprint),
    ``"on_top"`` (the destination's footprint, and above its top when heights
    are measured) or ``"distance"`` (within the tolerance of ``object_xy``).
    ``aim_object`` says whether the hand is offset so the *object*, not the
    TCP, lands on ``object_xy``.
    """

    relation: str
    object_xy: tuple[float, float]
    surface_z: float
    description: str
    destination_id: str | None = None
    check: str = "distance"
    tcp_z: float | None = None
    aim_object: bool = True

    def to_log(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "object_xy": [round(v, 4) for v in self.object_xy],
            "surface_z": round(self.surface_z, 4),
            "description": self.description,
            "destination_id": self.destination_id,
            "check": self.check,
            "tcp_z": None if self.tcp_z is None else round(self.tcp_z, 4),
            "aim_object": self.aim_object,
        }


def _xy_bounds(
    workspace_min: ArrayLike, workspace_max: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where the released object's centre may be, inside the walls.

    The workspace bounds the hand, not the object's overhang, so the held
    object's size is *not* subtracted: on the hardware table (x 0.06-0.30 m)
    a 140 mm marker's 70 mm half-length would leave a 6 cm strip, and "5 cm
    forward" was refused as outside the workspace (fake-lane probe).
    """
    low = np.asarray(workspace_min, dtype=np.float64)[:2] + PLACE_EDGE_MARGIN_M
    high = np.asarray(workspace_max, dtype=np.float64)[:2] - PLACE_EDGE_MARGIN_M
    middle = (low + high) / 2.0
    return np.minimum(low, middle), np.maximum(high, middle)


def _blocker(
    xy: NDArray[np.float64],
    radius: float,
    scene: SceneGraph,
    ignore: set[str],
    clearance: float,
) -> ObjectHypothesis | None:
    """The object a release disc at ``xy`` would crowd most, or ``None``."""
    worst: tuple[float, ObjectHypothesis] | None = None
    for obj in scene.objects.values():
        if obj.track_id in ignore:
            continue
        gap = footprint_gap(obj, xy, radius)
        if gap < clearance and (worst is None or gap < worst[0]):
            worst = (gap, obj)
    return None if worst is None else worst[1]


def _side_spot(
    destination: ObjectHypothesis, direction: NDArray[np.float64], radius: float, gap: float
) -> NDArray[np.float64]:
    centre = np.asarray(destination.bbox.center.position, dtype=np.float64)[:2]
    return centre + direction * (half_extent_along(destination, direction) + radius + gap)


def synthesise_place_target(
    relation: str,
    *,
    scene: SceneGraph,
    held_id: str | None,
    support_height: float,
    workspace_min: ArrayLike,
    workspace_max: ArrayLike,
    clearance: float,
    destination: ObjectHypothesis | None = None,
    origin_xy: ArrayLike | None = None,
    origin_description: str = "where it was picked up",
    direction: str | None = None,
    distance: float = DEFAULT_PLACE_DISTANCE_M,
    held_extents: ArrayLike | None = None,
    robot_xy: ArrayLike = (0.0, 0.0),
    gap: float = NEXT_TO_GAP_M,
    reachable: Callable[[NDArray[np.float64]], bool] | None = None,
) -> tuple[PlaceTarget | None, str]:
    """Turn a place relation into a :class:`PlaceTarget`, or ``(None, reason)``.

    Pure geometry. ``destination`` is the resolved target object (every
    relation but ``direction``); ``origin_xy`` is where the held object was
    picked from (``direction``). Table-level spots are clamped into the
    workspace and must stay :data:`PLACE_OBJECT_CLEARANCE_M` clear of every
    object except the held one; a spot that cannot is refused, naming what is
    in the way, rather than released on top of it.

    ``reachable(object_xy)`` (optional; ``Place`` passes an IK test of the
    release and the pose above it) filters ``next_to`` sides and refuses a
    fixed side or direction the arm cannot set the object down at. Without it
    a side was chosen that the hardware arm could transit above but not lower
    to (fake-lane probe: "cannot lower to the release height", object still
    held at 11 cm).
    """
    relation = normalise_relation(relation)
    radius = _held_radius(held_extents)
    ignore = {held_id} if held_id else set()
    robot = np.asarray(robot_xy, dtype=np.float64)[:2]
    low, high = _xy_bounds(workspace_min, workspace_max)

    def can_reach(spot: NDArray[np.float64]) -> bool:
        return reachable is None or bool(reachable(spot))

    if relation == "direction":
        word = normalise_direction(direction)
        if word not in PLACE_DIRECTIONS:
            return None, (
                f"unknown direction {direction!r}; expected one of {sorted(PLACE_DIRECTIONS)}"
            )
        if origin_xy is None:
            return None, "cannot place relative to where it was picked: that position is unknown"
        if not 0.0 < float(distance) <= 0.5:
            return None, f"a place offset of {float(distance) * 100:.0f} cm is out of range (0-50 cm)"
        unit = np.asarray(PLACE_DIRECTIONS[word], dtype=np.float64)
        wanted = np.asarray(origin_xy, dtype=np.float64)[:2] + unit * float(distance)
        spot = np.clip(wanted, low, high)
        where = f"{float(distance) * 100:.0f} cm {word} from {origin_description}"
        if float(np.linalg.norm(spot - wanted)) > DIRECTION_CLAMP_LIMIT_M:
            return None, f"{where} is outside the workspace"
        blocker = _blocker(spot, radius, scene, ignore, PLACE_OBJECT_CLEARANCE_M)
        if blocker is not None:
            return None, f"{where} is taken by the {describe(blocker)}"
        if not can_reach(spot):
            return None, f"{where} is out of the arm's reach"
        return PlaceTarget(
            "direction", (float(spot[0]), float(spot[1])), float(support_height), where
        ), ""

    if destination is None:
        return None, f"place {relation.replace('_', ' ')} needs a destination object"
    name = describe(destination)
    if relation == "to":
        relation = relation_for_to(destination)
    centre = np.asarray(destination.bbox.center.position, dtype=np.float64)

    if relation == "in":
        # Release just above the rim so the object drops in rather than being
        # pressed against the base: the TCP height the sim lane verified
        # ("put it in the bowl" landed inside the bowl, 2026-09-24).
        top = _top_of(destination)
        return PlaceTarget(
            "in", (float(centre[0]), float(centre[1])), top, f"in the {name}",
            destination.track_id, check="inside", tcp_z=top + float(clearance),
        ), ""
    if relation == "on":
        return PlaceTarget(
            "on", (float(centre[0]), float(centre[1])), _top_of(destination), f"on the {name}",
            destination.track_id, check="on_top",
        ), ""

    ignore_all = ignore | {destination.track_id}
    if relation == "next_to":
        options: list[tuple[bool, float, NDArray[np.float64]]] = []
        for axis in _footprint_axes(destination):
            for sign in (1.0, -1.0):
                wanted = _side_spot(destination, sign * axis, radius, gap)
                spot = np.clip(wanted, low, high)
                if footprint_gap(destination, spot, radius) < PLACE_OBJECT_CLEARANCE_M:
                    continue  # the workspace clamp pushed it back into the destination
                if _blocker(spot, radius, scene, ignore_all, PLACE_OBJECT_CLEARANCE_M) is not None:
                    continue
                clamped = float(np.linalg.norm(spot - wanted)) > 1e-6
                options.append((clamped, float(np.linalg.norm(spot - robot)), spot))
        if not options:
            return None, f"there is no free space next to the {name} inside the workspace"
        free = len(options)
        options = [option for option in options if can_reach(option[2])]
        if not options:
            return None, (
                f"the free space next to the {name} ({free} side(s)) is out of the arm's reach"
            )
        # Unclamped sides first, then the one nearest the robot: the side that
        # faces the arm is the most reachable, and the far side is where the
        # old +Y rule released at the edge of reach.
        options.sort(key=lambda option: (option[0], option[1]))
        spot = options[0][2]
        return PlaceTarget(
            "next_to", (float(spot[0]), float(spot[1])), float(support_height),
            f"next to the {name}", destination.track_id,
        ), ""

    if relation in ("left_of", "right_of", "in_front_of", "behind"):
        if relation == "left_of":
            direction_xy = np.array([0.0, 1.0])
        elif relation == "right_of":
            direction_xy = np.array([0.0, -1.0])
        else:
            # "In front of" is between the target and the robot, as grounding
            # reads it (nearer the base); "behind" is the far side.
            toward_robot = robot - centre[:2]
            norm = float(np.linalg.norm(toward_robot))
            toward_robot = toward_robot / norm if norm > 1e-6 else np.array([-1.0, 0.0])
            direction_xy = toward_robot if relation == "in_front_of" else -toward_robot
        spoken = {"left_of": "to the left of", "right_of": "to the right of",
                  "in_front_of": "in front of", "behind": "behind"}[relation]
        where = f"{spoken} the {name}"
        spot = np.clip(_side_spot(destination, direction_xy, radius, gap), low, high)
        on_side = float((spot - centre[:2]) @ direction_xy) > 0.0
        if not on_side or footprint_gap(destination, spot, radius) < PLACE_OBJECT_CLEARANCE_M:
            return None, f"there is no room {where} inside the workspace"
        blocker = _blocker(spot, radius, scene, ignore_all, PLACE_OBJECT_CLEARANCE_M)
        if blocker is not None:
            return None, f"the space {where} is taken by the {describe(blocker)}"
        if not can_reach(spot):
            return None, f"the space {where} is out of the arm's reach"
        return PlaceTarget(
            relation, (float(spot[0]), float(spot[1])), float(support_height), where,
            destination.track_id,
        ), ""

    return None, (
        f"unknown place relation {relation!r}; expected one of "
        f"{sorted(PLACE_RELATIONS | {'direction'})}"
    )


def release_tcp_position(
    target: PlaceTarget,
    quat: ArrayLike,
    *,
    clearance: float,
    held_half_height: float,
    offset_gripper: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Where the TCP must be, at orientation ``quat``, for the object to land on ``target``.

    XY: ``object_xy`` minus the held object's offset from the TCP *at this
    orientation* (only when ``target.aim_object`` and an offset is known).
    Z: ``tcp_z`` when the target fixes it, else the surface plus the clearance
    plus half the held object's height (the TCP sits at about the object's
    middle), so the object is set down rather than dropped or pressed in.
    """
    xy = np.asarray(target.object_xy, dtype=np.float64)
    if target.aim_object and offset_gripper is not None:
        xy = xy - aim_shift_xy(offset_gripper, quat)
    if target.tcp_z is not None:
        z = float(target.tcp_z)
    else:
        z = float(target.surface_z) + float(clearance) + float(held_half_height)
    return np.array([float(xy[0]), float(xy[1]), z])


def refine_place_target(
    target: PlaceTarget,
    before: ObjectHypothesis,
    now: ObjectHypothesis | None,
    *,
    max_shift: float = PLACE_REFINE_MAX_SHIFT_M,
    min_extent_ratio: float = PLACE_REFINE_MIN_EXTENT_RATIO,
) -> tuple[PlaceTarget, float, str]:
    """Re-centre an ``on``/``in`` target on the destination as seen from above it.

    ``before`` is the destination as perceived when the target was made,
    ``now`` as perceived from the pose above it (``None``: not seen now).
    Returns ``(target, shift_m, reason)``: the target moved by the change in
    the destination's centre, or the original one with the reason it was kept.
    Only the XY moves: the heights were right in every measured case, and a
    close view that sees less of the sides would only make them worse.
    """
    if target.check not in ("on_top", "inside"):
        return target, 0.0, "only on/in are centred on the destination"
    if now is None:
        return target, 0.0, "the destination was not seen from above it"
    delta = np.asarray(now.bbox.center.position[:2], dtype=np.float64) - np.asarray(
        before.bbox.center.position[:2], dtype=np.float64
    )
    shift = float(np.linalg.norm(delta))
    if shift > float(max_shift):
        return target, shift, (
            f"the close view moved the destination {shift * 1000:.0f} mm, more than the "
            f"{float(max_shift) * 1000:.0f} mm a re-estimate can be trusted with"
        )
    wide_before = np.sort(np.asarray(before.bbox.extents, dtype=np.float64)[:2])
    wide_now = np.sort(np.asarray(now.bbox.extents, dtype=np.float64)[:2])
    if np.any(wide_now < float(min_extent_ratio) * wide_before):
        return target, shift, (
            "the close view saw only part of the destination "
            f"({np.round(wide_now * 1000).astype(int).tolist()} mm against "
            f"{np.round(wide_before * 1000).astype(int).tolist()} mm)"
        )
    xy = np.asarray(target.object_xy, dtype=np.float64) + delta
    return replace(target, object_xy=(float(xy[0]), float(xy[1]))), shift, ""


@dataclass(frozen=True)
class PlacementCheck:
    """The verdict on a finished place, from the post-release observation."""

    ok: bool
    seen: bool
    offset_m: float
    reason: str

    def to_log(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seen": self.seen,
            "offset_m": None if math.isinf(self.offset_m) else round(self.offset_m, 4),
            "reason": self.reason,
        }


def assess_placement(
    target: PlaceTarget,
    placed: ObjectHypothesis | None,
    destination: ObjectHypothesis | None,
    *,
    tolerance_m: float = DEFAULT_PLACE_TOLERANCE_M,
    vertical_known: bool = True,
    name: str = "object",
) -> PlacementCheck:
    """Did the object end up where it was sent? Never assumed.

    ``placed`` is the released object as re-observed (``None``: not seen in
    that observation). ``destination`` is the container / support as
    perceived *before* the place: after it, the container's own box can
    swallow the object. ``vertical_known`` is false on the depthless lane,
    where every z is pinned to the table and only the footprint is judged.
    """
    if placed is None:
        return PlacementCheck(
            False, False, float("inf"),
            f"the {name} was not seen after release, so where it landed cannot be confirmed",
        )
    landed = np.asarray(placed.bbox.center.position, dtype=np.float64)
    offset = float(np.linalg.norm(landed[:2] - np.asarray(target.object_xy, dtype=np.float64)))
    mm = offset * 1000.0

    if target.check in ("inside", "on_top") and destination is not None:
        dest_name = describe(destination)
        if not inside_footprint(destination, landed[:2]):
            where = "outside" if target.check == "inside" else "off"
            return PlacementCheck(
                False, True, offset,
                f"the {name} settled {mm:.0f} mm from the centre of the {dest_name}, {where} it",
            )
        if target.check == "on_top" and vertical_known and float(landed[2]) < _top_of(destination):
            return PlacementCheck(
                False, True, offset,
                f"the {name} is not on top of the {dest_name}: its centre is at "
                f"z={landed[2]:.3f} m, below the top at {_top_of(destination):.3f} m",
            )
        return PlacementCheck(True, True, offset, "")

    if offset > float(tolerance_m):
        return PlacementCheck(
            False, True, offset,
            f"the {name} settled {mm:.0f} mm from the target, more than the "
            f"{float(tolerance_m) * 1000:.0f} mm allowed",
        )
    return PlacementCheck(True, True, offset, "")


# ----------------------------------------------------------------------
# perception
# ----------------------------------------------------------------------


class Observe(Skill):
    """Perceive the scene and report it. Moves nothing.

    The message names every object with its colour ("observed 3 object(s):
    red block, blue can, green box"), in the words grounding accepts back:
    the audit found colour was perceived but never reported, so an operator
    could not know "the red one" was a valid thing to say.
    """

    skill_name = "observe"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        # Render before capturing: without a rendered frame the annotators hold
        # whatever they last produced.
        self.ctx.sim.render_step(2)
        scene = self.ctx.vision.observe()
        if self.ctx.memory is not None:
            self.ctx.memory.update_scene(scene)

        objects = list(scene.objects.values())
        described = [
            {
                "track_id": o.track_id,
                "label": o.label,
                "color": str((getattr(o, "attributes", None) or {}).get("color", "") or ""),
                "description": describe(o),
                "position": np.round(o.pose.position, 4).tolist(),
                "size": np.round(o.bbox.extents, 4).tolist(),
                "confidence": round(o.confidence, 3),
            }
            for o in objects
        ]
        message = f"observed {len(described)} object(s)"
        if described:
            message += ": " + ", ".join(entry["description"] for entry in described)
        return self._ok(message, objects=described, relations=scene.relations)


class ScanScene(Skill):
    """Sweep the wrist through several viewpoints, observing at each.

    A single viewpoint leaves occlusion shadows. Scanning from a few base
    rotations reveals objects hidden behind others, and each observation feeds the
    same tracker, so identities persist across the sweep.
    """

    skill_name = "scan_scene"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        sweep = float(params.get("sweep", 0.6))
        num_views = int(params.get("views", 3))

        home = np.asarray(self.ctx.config.robot.home_joint_positions, dtype=np.float64)
        offsets = np.linspace(-sweep, sweep, max(1, num_views))
        seen: dict[str, str] = {}

        for offset in offsets:
            target = home.copy()
            target[0] += float(offset)
            scene = self.ctx.vision.require_fresh_scene()
            trajectory = self.ctx.planner.plan_to_joint(
                self.ctx.robot.get_state().joint_state, target, scene
            )
            if trajectory is None:
                continue
            self.ctx.controller.follow_trajectory(trajectory)
            self.ctx.sim.render_step(3)
            observed = self.ctx.vision.observe()
            for track_id, obj in observed.objects.items():
                seen[track_id] = obj.label

        if self.ctx.memory is not None:
            self.ctx.memory.update_scene(self.ctx.vision.last_scene_graph())

        return self._ok(f"scanned {len(offsets)} viewpoint(s), found {len(seen)} object(s)",
                        objects=seen)


class LookAt(Skill):
    """Point the wrist camera at an object without approaching it."""

    skill_name = "look_at"

    def validate(self, params: dict[str, Any]) -> str | None:
        if not params.get("target"):
            return "look_at requires a target"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        scene = self.ctx.vision.require_fresh_scene()
        obj = _resolve_object(self.ctx, params, scene=scene)
        track_id = obj.track_id

        # Stand off above the object looking straight down: the most reliable
        # inspection pose, and the one that occludes least.
        standoff = float(params.get("distance", 0.25))
        view_position = obj.bbox.center.position + np.array([0.0, 0.0, standoff])
        view_pose = Pose(
            view_position,
            tf.matrix_to_quat(np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])),
            Frame.WORLD,
        )

        trajectory = self.ctx.planner.plan_to_pose(
            self.ctx.robot.get_state().joint_state, view_pose, scene
        )
        if trajectory is None:
            return self._infeasible(f"cannot reach a viewing pose for {obj.label!r}")
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted while moving to the viewing pose")

        self.ctx.sim.render_step(3)
        self.ctx.vision.observe()
        return self._ok(f"looking at {obj.label!r}", track_id=track_id)


# ----------------------------------------------------------------------
# motion
# ----------------------------------------------------------------------


class MoveTo(Skill):
    """Move the TCP to an absolute pose or a named object's standoff."""

    skill_name = "move_to"

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("position") is None and not params.get("target"):
            return "move_to requires either a position or a target"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        scene = self.ctx.vision.require_fresh_scene()
        current = self.ctx.robot.tcp_pose()

        if params.get("position") is not None:
            position = np.asarray(params["position"], dtype=np.float64)
        else:
            obj = _resolve_object(self.ctx, params, scene=scene)
            position = obj.bbox.center.position + np.array(
                [0.0, 0.0, float(params.get("standoff", 0.15))]
            )

        goal = Pose(position, np.asarray(params.get("quat", current.quat)), Frame.WORLD)
        trajectory = self.ctx.planner.plan_with_retries(
            self.ctx.robot.get_state().joint_state, goal, scene
        )
        if trajectory is None:
            return self._infeasible(
                f"no collision-free path to {np.round(position, 3).tolist()}"
            )
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted")

        achieved = self.ctx.robot.tcp_pose()
        return self._ok(
            f"moved to {np.round(achieved.position, 3).tolist()}",
            position=achieved.position.tolist(),
            error=achieved.translation_distance(goal),
        )


class MoveRelative(Skill):
    """Move the TCP by a Cartesian delta. "Move left" and nothing else."""

    skill_name = "move_relative"

    #: Direction words mapped to unit vectors in the robot's frame. The base sits
    #: at the origin facing +X, so the operator's "left" is +Y.
    DIRECTIONS = {
        "left": np.array([0.0, 1.0, 0.0]),
        "right": np.array([0.0, -1.0, 0.0]),
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, 0.0, 0.0]),
        "back": np.array([-1.0, 0.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([0.0, 0.0, -1.0]),
    }

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("delta") is None and not params.get("direction"):
            return "move_relative requires a delta or a direction"
        direction = params.get("direction")
        if direction and str(direction).lower() not in self.DIRECTIONS:
            return (
                f"unknown direction {direction!r}; "
                f"expected one of {sorted(self.DIRECTIONS)}"
            )
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        if params.get("delta") is not None:
            delta = np.asarray(params["delta"], dtype=np.float64)
        else:
            distance = float(params.get("distance", 0.10))
            delta = self.DIRECTIONS[str(params["direction"]).lower()] * distance

        current = self.ctx.robot.tcp_pose()
        goal = Pose(current.position + delta, current.quat, Frame.WORLD)
        scene = self.ctx.vision.require_fresh_scene()

        # Straight line: a relative move should go where it was told, not curve.
        trajectory = self.ctx.planner.plan_cartesian_line(
            self.ctx.robot.get_state().joint_state, goal, scene
        )
        if trajectory is None:
            return self._infeasible(
                f"cannot move by {np.round(delta, 3).tolist()}: outside reach or no IK"
            )
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("motion aborted")

        achieved = self.ctx.robot.tcp_pose()
        return self._ok(
            f"moved by {np.round(achieved.position - current.position, 3).tolist()}",
            delta=(achieved.position - current.position).tolist(),
        )


class RotateWrist(Skill):
    """Rotate the last joint. Rotation only, no translation."""

    skill_name = "rotate_wrist"

    def validate(self, params: dict[str, Any]) -> str | None:
        if params.get("angle") is None:
            return "rotate_wrist requires an angle in radians"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        angle = float(params["angle"])
        joints = self.ctx.robot.get_arm_joint_positions().copy()
        joints[-1] += angle

        scene = self.ctx.vision.require_fresh_scene()
        trajectory = self.ctx.planner.plan_to_joint(
            self.ctx.robot.get_state().joint_state, joints, scene
        )
        if trajectory is None:
            return self._infeasible(f"cannot rotate wrist by {angle:.3f} rad (joint limit?)")
        if not self.ctx.controller.follow_trajectory(trajectory):
            return self._fail("rotation aborted")
        return self._ok(f"rotated wrist by {angle:.3f} rad")


class GoHome(Skill):
    """Return to the home posture by planning, not by teleporting."""

    skill_name = "go_home"

    #: Retreats tried, in order, when the arm cannot plan home from where it
    #: stands. Straight up first: it is the shortest escape and the one that
    #: works after an ordinary pick. Then higher. Then up-and-back, which is
    #: what frees the arm when the table edge rather than the height is what
    #: blocks the local configuration space -- the case a pure lift cannot fix,
    #: and the one that left the arm stranded after a place.
    _RETREAT_OFFSETS = (
        np.array([0.0, 0.0, 0.15]),
        np.array([0.0, 0.0, 0.25]),
        np.array([-0.10, 0.0, 0.20]),
        np.array([-0.18, 0.0, 0.10]),
    )

    def _run(self, params: dict[str, Any]) -> SkillResult:
        home = np.asarray(self.ctx.config.robot.home_joint_positions, dtype=np.float64)
        scene = self.ctx.vision.require_fresh_scene()
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        trajectory = planner.plan_to_joint(robot.get_state().joint_state, home, scene)
        if trajectory is None:
            trajectory = self._escape_then_plan_home(home)

        if trajectory is None:
            return self._infeasible(
                "no collision-free path home after "
                f"{len(self._RETREAT_OFFSETS)} retreat attempts"
            )
        if not controller.follow_trajectory(trajectory):
            return self._fail("motion home aborted")

        error = float(np.max(np.abs(robot.get_arm_joint_positions() - home)))
        return self._ok(f"at home posture (max joint error {error:.3f} rad)")

    def _escape_then_plan_home(self, home: NDArray[np.float64]) -> Any:
        """Retreat out of the blocked region, then re-plan the path home.

        Why a ladder rather than one lift. After a pick or place the arm sits
        low and close to the table, where enough of the local configuration
        space is blocked that RRT rejects the start state outright -- and a
        rejected *start* cannot be rescued by re-seeding the tree, only by
        moving somewhere else first. The previous version tried exactly one
        +150 mm lift and, if that single straight line had no IK solution, gave
        up: measured after a place, that left the arm stranded with "no
        collision-free path home" and only a restart to recover from.

        Each offset is attempted twice, straight line first. ``plan_cartesian_line``
        is preferred because it is short and predictable, but it solves IK at
        every waypoint and fails outright when any one of them has no solution.
        RRT can still route around that, so the same target is retried with
        ``plan_with_retries`` -- which also varies the IK branch, the fix that
        made the pregrasp reachable.

        Retreats accumulate: a partial escape that does not yet free the arm
        still leaves it somewhere better for the next offset to work from.
        """
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        for index, offset in enumerate(self._RETREAT_OFFSETS, start=1):
            tcp = robot.tcp_pose()
            target = Pose(
                _clamp_to_workspace(self.ctx, tcp.position + offset),
                tcp.quat,
                Frame.WORLD,
            )
            scene = self.ctx.vision.require_fresh_scene()

            retreat = planner.plan_cartesian_line(
                robot.get_state().joint_state, target, scene
            )
            if retreat is None:
                retreat = planner.plan_with_retries(
                    robot.get_state().joint_state, target, scene
                )
            if retreat is None:
                _log.debug("Retreat %d: no motion to %s", index, offset.tolist())
                continue
            if not controller.follow_trajectory(retreat):
                _log.debug("Retreat %d aborted mid-motion", index)
                continue

            scene = self.ctx.vision.require_fresh_scene()
            trajectory = planner.plan_to_joint(
                robot.get_state().joint_state, home, scene
            )
            if trajectory is not None:
                _log.info("Path home found after retreat %d %s", index, offset.tolist())
                return trajectory

        return None


# ----------------------------------------------------------------------
# gripper
# ----------------------------------------------------------------------


class OpenGripper(Skill):
    """Open the fingers. Opens only -- the arm does not move.

    Releases any held object as a physical consequence of opening, and updates
    memory accordingly, but performs no motion of its own.
    """

    skill_name = "open_gripper"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        width = self.ctx.controller.open_gripper_blocking()
        if self.ctx.memory is not None:
            self.ctx.memory.set_held_object(None)
        self.ctx.held_grasp = None
        return self._ok(f"gripper open at {width * 1000:.1f} mm", width=width)


class CloseGripper(Skill):
    """Close the fingers. Closes only -- no approach, no lift, no verification.

    Reports whether the fingers stalled, which is evidence something is between
    them, but deliberately does not claim a grasp: that requires a lift, and
    lifting is ``Pick``'s job.
    """

    skill_name = "close_gripper"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        width = self.ctx.controller.close_gripper_blocking()
        stalled = width > self.ctx.config.robot.gripper_closed_width + 0.004
        return self._ok(
            f"gripper closed to {width * 1000:.1f} mm"
            + (" (stalled on an object)" if stalled else " (closed on air)"),
            width=width,
            stalled_on_object=stalled,
        )


# ----------------------------------------------------------------------
# pick and place
# ----------------------------------------------------------------------


class Pick(Skill):
    """Pick up an object and **hold** it.

    Sequence: observe, synthesise grasps, choose the best reachable one, plan to
    the pregrasp standoff, approach in a straight line, close with force limiting,
    lift, verify from sensor evidence, then hold and WAIT.

    Does **not** place. Nothing here triggers another skill.
    """

    skill_name = "pick"

    def validate(self, params: dict[str, Any]) -> str | None:
        if not params.get("target"):
            return "pick requires a target"
        if self.ctx.memory is not None and self.ctx.memory.get_held_object() is not None:
            return "already holding something; place or release it first"
        return None

    def _generate(self, obj: Any, scene: Any) -> list[Any]:
        """Grasp candidates for ``obj``, from the injected generator when there is one.

        The sim lane leaves ``ctx.grasp_generator`` unset and keeps the direct
        OBB synthesis. The hardware lane injects a top-down generator: its
        perception has no depth, so the "box" is a size-table entry, and a
        jaw with no wrist roll cannot meet an arbitrary box axis anyway.
        """
        generator = getattr(self.ctx, "grasp_generator", None)
        if generator is not None:
            return list(generator.generate(scene, obj.track_id))
        return generate_grasp_candidates(obj, self.ctx.config.grasp)

    def _run(self, params: dict[str, Any]) -> SkillResult:
        # 1. Perceive. Never act on a remembered pose.
        scene, obj = _observe_and_resolve(
            self.ctx, lambda seen: _resolve_object(self.ctx, params, scene=seen)
        )
        track_id = obj.track_id

        # 2. Synthesise and rank grasps from the perceived geometry.
        candidates = self._generate(obj, scene)
        hardware_generator = getattr(self.ctx, "grasp_generator", None) is not None
        narrow_enough = float(np.min(obj.bbox.extents[:2])) <= self.ctx.config.grasp.max_grasp_width
        if hardware_generator and narrow_enough and (getattr(obj, "attributes", None) or {}).get("yaw_ambiguous"):
            # Depthless perception only (the sim lane never sets it): the pixel
            # box fits neither the along-X nor the across orientation well
            # enough, so both the centre and the chord the candidates were
            # checked against are guesses. A wrong one closes the jaw on air,
            # or drives the fingers onto the body.
            return self._infeasible(
                f"cannot tell precisely which way the {obj.label} is lying (the camera sees it "
                "at an angle), so the jaw could miss it; turn it to point straight at the "
                "arm's base and ask again"
            )
        if not candidates and hardware_generator and narrow_enough:
            # Hardware lane: the jaw cannot rotate about the tool axis, so an
            # object narrow enough overall can still be too wide *across the
            # jaw* the way it is lying. Saying "19 mm, the gripper spans 35 mm"
            # there is a contradiction the student cannot act on.
            return self._infeasible(
                f"the {obj.label} is narrow enough ({np.min(obj.bbox.extents[:2]) * 1000:.0f} mm) "
                "but not lying the way this jaw closes (it cannot rotate); turn it to point "
                "straight at the arm's base and ask again"
            )
        if not candidates:
            return self._infeasible(
                f"{obj.label!r} is {np.min(obj.bbox.extents) * 1000:.0f} mm at its narrowest; "
                f"the gripper spans {self.ctx.config.grasp.max_grasp_width * 1000:.0f} mm"
            )

        state = self.ctx.robot.get_state()
        ranked = self.ctx.grasp_scorer.score(candidates, scene, state)
        if not ranked:
            return self._infeasible(
                f"generated {len(candidates)} grasp(s) for {obj.label!r} but none are "
                "reachable and collision-free"
            )

        self.ctx.emit(
            "pick.grasp_selected",
            {"target": track_id, "label": obj.label, "chosen": ranked[0].to_log(),
             "num_candidates": len(candidates), "num_feasible": len(ranked)},
        )

        # 3. Attempt the best few grasps in order. A failed approach is a normal
        #    outcome, not an error: the next candidate may well work.
        #
        # Every retry re-observes and re-plans from scratch, because a failed
        # attempt usually MOVES the object. Instrumented on the soup can: the
        # first attempt gripped it (fingers stalled at 60.3 mm on a 68 mm can),
        # lost it during the lift, and left it 34 mm from where it started. The
        # remaining candidates -- computed from the original observation --
        # then sent the gripper to empty space 100 mm away, twice, and the skill
        # reported "closed without contact".
        #
        # That message described the stale retries, not the real failure, and it
        # is what made this look like a perception centring problem for far
        # longer than it should have. A retry that plans against a pose the
        # previous attempt invalidated is not a second chance; it is a guaranteed
        # miss that also overwrites the useful error.
        last_error = "no grasp attempted"
        attempts = int(self.ctx.config.motion.max_replan_attempts)

        for attempt in range(attempts):
            if attempt == 0:
                grasp = ranked[0]
            else:
                # Reset before looking, so the arm is not occluding its own view.
                self.ctx.controller.open_gripper_blocking()
                fresh = self.ctx.vision.observe()
                moved = fresh.get(track_id)
                if moved is None:
                    return self._fail(
                        f"{last_error} (and {obj.label!r} is no longer visible to retry)"
                    )

                regenerated = self._generate(moved, fresh)
                reranked = self.ctx.grasp_scorer.score(
                    regenerated, fresh, self.ctx.robot.get_state()
                )
                if not reranked:
                    return self._fail(
                        f"{last_error} (no feasible grasp remains after re-observing)"
                    )
                idx = min(attempt, len(reranked) - 1)
                grasp, obj, scene = reranked[idx], moved, fresh

            outcome = self._attempt_grasp(grasp, obj, scene)
            if outcome.ok:
                return outcome
            if not self.ctx.config.robot.gripper_feedback and (
                outcome.data.get("verification_failed") or outcome.data.get("jaw_closed")
            ):
                # Without gripper feedback the verdict comes from pixels and can
                # be wrong in either direction, and the jaw is closed at lift
                # height. A retry would open it there -- dropping the object if
                # it *is* held -- and its own failure would then replace this
                # message with an unrelated one (review F1/F3). Stop here: the
                # verifier's reason is the answer, and a human decides.
                return self._infeasible(
                    f"{outcome.message}. The jaw is left closed in case "
                    f"{obj.label!r} is held: look at the gripper, then say 'open the gripper' "
                    "to release it",
                    **outcome.data,
                )
            last_error = outcome.message

        return self._fail(f"all grasp attempts failed: {last_error}")

    _MIN_RECENTRE_LIMIT_M = 0.05

    def _recentre_from_standoff(self, grasp: Any) -> Pose | None:
        """Re-measure the target from the pre-grasp and correct the grasp XY.

        Returns ``None`` when the correction should not be applied, which is the
        safe default: the original pose was at least scored for reachability and
        clearance, and a correction derived from a bad observation is worse than
        no correction.

        Declines when the target is no longer tracked -- the arm may have nudged
        it, and a grasp aimed at a stale track is worse than one aimed at a stale
        pose -- or when the shift is larger than the object itself.

        That second bound replaces a fixed 50 mm limit, which was arbitrary and
        rejected exactly the corrections it existed to enable: it fired eight
        times in one run, blocking shifts of 51 to 97 mm on the three objects
        still closing on air. Identity is not what a magnitude test protects
        here anyway -- the detection is fetched by ``track_id``, so the tracker
        has already established this is the same object.

        What is worth testing is physical plausibility. A correction larger than
        the object's own footprint cannot be a re-measurement of that object,
        whatever the tracker says, so the object's largest horizontal extent is
        the natural limit. The floor keeps small objects correctable at all.
        """
        try:
            fresh = self.ctx.vision.observe()
        except Exception as exc:  # noqa: BLE001 - fall back to the planned pose
            _log.debug("Re-observation at the standoff failed: %s", exc)
            return None

        target = fresh.get(grasp.target_track_id)
        if target is None:
            _log.debug(
                "Target %s not tracked from the standoff; keeping the planned grasp",
                grasp.target_track_id,
            )
            return None

        planned = np.asarray(grasp.pose.position, dtype=np.float64)
        corrected = planned.copy()
        corrected[:2] = np.asarray(target.bbox.center.position, dtype=np.float64)[:2]

        shift = float(np.linalg.norm(corrected[:2] - planned[:2]))
        limit = max(
            self._MIN_RECENTRE_LIMIT_M,
            float(np.max(np.asarray(target.bbox.extents, dtype=np.float64)[:2])),
        )
        if shift > limit:
            _log.warning(
                "Re-centring would move the grasp %.0f mm, more than the object's own "
                "%.0f mm footprint; keeping the planned pose",
                shift * 1000.0,
                limit * 1000.0,
            )
            return None

        if shift > 1e-4:
            _log.info("Re-centred the grasp by %.1f mm from the standoff view", shift * 1000.0)
        return Pose(corrected, grasp.pose.quat, Frame.WORLD)

    def _reference_view(self, track_id: str, planning_scene: Any) -> Any:
        """The pre-descent scene the feedback-less verdict compares against.

        Observed now, from the standoff, before the jaw descends. If the arm
        at the standoff already hides the target, the scene the grasp was
        planned from (taken before the arm moved over it) is used instead,
        provided the target was actually detected in it. Either way the
        verifier checks the target was seen *in that very observation*, so a
        track the tracker merely kept alive never becomes a reference.
        """
        try:
            view = self.ctx.vision.observe()
        except Exception as exc:  # noqa: BLE001 - fall back to the planning view
            _log.warning("Pre-descent observation failed (%s); using the planning view", exc)
            return planning_scene
        target = view.get(track_id)
        if target is not None and target.last_seen_step >= view.step_index:
            return view
        planned = planning_scene.get(track_id)
        if planned is not None and planned.last_seen_step >= planning_scene.step_index:
            _log.info("Target hidden from the standoff; the planning view is the pre-descent reference")
            return planning_scene
        return view

    def _lift_prediction(self, scene_before: Any, track_id: str) -> Any:
        """What the target should look like now if it is held (``None``: no predictor)."""
        predictor = getattr(self.ctx.vision, "predict_lift", None)
        target = scene_before.get(track_id) if scene_before is not None else None
        if predictor is None or target is None:
            return None
        try:
            return predictor(target, self.ctx.robot.tcp_pose())
        except Exception as exc:  # noqa: BLE001 - verdict falls back to bare pixel growth
            _log.warning("Lift prediction failed (%s); verifying from pixel growth alone", exc)
            return None

    def _attempt_grasp(self, grasp: Any, obj: Any, scene: Any) -> SkillResult:
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot

        # The target must not be an obstacle while we deliberately close on it.
        planner.exclude_from_collision(grasp.target_track_id)
        try:
            controller.open_gripper_blocking()

            # 4. Plan to the standoff with full collision avoidance.
            approach_plan = planner.plan_with_retries(
                robot.get_state().joint_state, grasp.pregrasp_pose, scene
            )
            if approach_plan is None:
                return self._fail("no path to the pregrasp standoff")
            if not controller.follow_trajectory(approach_plan):
                return self._fail("motion to the pregrasp standoff aborted")

            feedback = bool(self.ctx.config.robot.gripper_feedback)
            before_descent = None
            if not feedback:
                # Without gripper feedback the verdict compares the object's
                # pixel box before and after. Observed after the descent, the
                # arm is parked over the object and a small one is already
                # hidden -- its later absence then proves nothing (review:
                # "before scene observed with the arm over the object"). So
                # the reference is taken here, from the standoff, and the sim
                # lane keeps its post-descent observation below.
                before_descent = self._reference_view(grasp.target_track_id, scene)

            # 4b. Re-observe from the standoff and re-centre the grasp.
            #
            # The pose planned from is measured from wherever the object first
            # became visible, often an oblique exterior view. Measured on the
            # benchmark scene, that carries up to 71 mm of centre error -- more
            # than the soup can is wide -- so the fingers close on empty air
            # beside the object. The errors track visibility exactly:
            # exterior-only objects come in at 71/53/43/24 mm, objects seen by
            # both cameras at 22/22/18/14/3 mm.
            #
            # Here the wrist camera sits ~100 mm directly above the target,
            # looking straight down: minimal occlusion, maximal resolution, no
            # oblique foreshortening. It is the best view this robot will ever
            # get of the object, and it is free -- the arm is already parked.
            #
            # Only the horizontal centre is taken. Height, orientation and grasp
            # width come from the original candidate, because those were what
            # the scorer checked for reachability and finger clearance, and
            # changing them here would invalidate that reasoning. Correcting
            # *where* the fingers close without changing *how* is the whole
            # point.
            grasp_pose = grasp.pose
            if self.ctx.config.grasp.recentre_from_standoff:
                # Off on the hardware lane: one fixed overhead camera sees
                # the same thing from the standoff as it did before.
                recentred = self._recentre_from_standoff(grasp)
                if recentred is not None:
                    grasp_pose = recentred

            # 5. Straight-line approach. A curved path would sweep the fingers
            #    sideways through the object.
            descent = planner.plan_cartesian_line(
                robot.get_state().joint_state, grasp_pose, scene
            )
            if descent is None:
                return self._fail("no straight-line approach to the grasp pose")
            if not controller.follow_trajectory(descent):
                return self._fail("approach aborted")

            if feedback:
                scene_before_lift = self.ctx.vision.observe()
            else:
                scene_before_lift = before_descent

            # 6. Force-limited close. Real contact, real friction. A controller
            #    that can close to a width (hobby servos, no force limit) is
            #    told the chord the candidate was checked against, so the jaw
            #    squeezes the object instead of stalling against it.
            if hasattr(self.ctx.controller, "set_grasp_width"):
                self.ctx.controller.set_grasp_width(grasp.width)
            controller.close_gripper_blocking()

            if feedback:
                return self._lift_and_verify(grasp, obj, scene, scene_before_lift, feedback)
            try:
                return self._lift_and_verify(grasp, obj, scene, scene_before_lift, feedback)
            except SafetyViolation:
                raise
            except Exception as exc:  # noqa: BLE001 - any error here must not reopen the jaw
                # Review F3 (re-review, gv/probe2.py): the jaw is closed on
                # whatever it caught and there is no gripper feedback. An
                # exception here (a detector hiccup on the post-lift observe)
                # used to escape as a FAILED result, the planner retried Pick,
                # and the retry's first act was to open the jaw at lift height
                # -- dropping a marker that really was held -- while the
                # operator read an unrelated "none are reachable". The
                # verdict is simply unknown: stop, jaw closed, say why.
                _log.warning("Grasp verification did not complete: %s: %s", type(exc).__name__, exc)
                self.ctx.emit(
                    "pick.verification_error",
                    {"target": grasp.target_track_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                return self._fail(
                    f"the grasp could not be verified ({type(exc).__name__}: {exc})",
                    jaw_closed=True,
                    verification_failed=True,
                    evidence={"verdict": "unknown", "error": f"{type(exc).__name__}: {exc}"},
                )
        finally:
            planner.include_in_collision(grasp.target_track_id)

    def _lift_and_verify(
        self, grasp: Any, obj: Any, scene: Any, scene_before_lift: Any, feedback: bool
    ) -> SkillResult:
        """Lift the closed jaw, verify from evidence, and hold (steps 7-9)."""
        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot
        # 7. Lift straight up, re-asserting the grip every step. Without
        #    re-assertion the fingers relax and drop the load mid-lift.
        lift_height = float(self.ctx.config.grasp.lift_height)
        grasp_pose_now = robot.tcp_pose()
        lift_target = Pose(
            _clamp_to_workspace(
                self.ctx, grasp_pose_now.position + np.array([0.0, 0.0, lift_height])
            ),
            grasp_pose_now.quat,
            Frame.WORLD,
        )
        lift = planner.plan_cartesian_line(
            robot.get_state().joint_state, lift_target, scene
        )
        if lift is None:
            return self._fail("cannot lift from the grasp pose")

        def hold_grip(_index: int, _positions: NDArray[np.float64]) -> bool:
            controller.maintain_grasp()
            return True

        if not controller.follow_trajectory(lift, on_step=hold_grip):
            if feedback:
                return self._fail("lift aborted")
            # Stopped part-way up with the jaw closed on whatever it
            # holds: the retry's open would drop it (see _run).
            return self._fail("lift aborted", jaw_closed=True)

        # 8. Verify from evidence, not from having issued the commands.
        self.ctx.sim.render_step(3)
        scene_after = self.ctx.vision.observe()
        extra: dict[str, Any] = {}
        prediction = None
        if not feedback:
            prediction = self._lift_prediction(scene_before_lift, grasp.target_track_id)
            extra["lift_prediction"] = prediction
        evidence = verify_grasp(
            robot=robot,
            scene_before=scene_before_lift,
            scene_after=scene_after,
            track_id=grasp.target_track_id,
            closed_width=self.ctx.config.robot.gripper_closed_width,
            expected_height_gain=lift_height,
            gripper_feedback=self.ctx.config.robot.gripper_feedback,
            min_displacement=self.ctx.config.grasp.verify_min_displacement,
            **extra,
        )
        payload = {"target": grasp.target_track_id, **evidence.to_log()}
        if not feedback:
            payload["before_view"] = "planning" if scene_before_lift is scene else "standoff"
        if prediction is not None:
            payload["lift_prediction"] = prediction.to_log()
        self.ctx.emit("pick.verification", payload)

        if not evidence.holding:
            if feedback:
                return self._fail(evidence.reason())
            return self._fail(
                evidence.reason(), verification_failed=True, evidence=evidence.to_log()
            )

        if self.ctx.memory is not None:
            self.ctx.memory.set_held_object(grasp.target_track_id)
        record = self._record_grasp(grasp.target_track_id, obj, scene_after)
        self.ctx.held_grasp = record

        # 9. Hold and WAIT. No place, no return home.
        controller.maintain_grasp()
        return self._ok(
            f"holding {obj.label!r} ({evidence.reason()})",
            track_id=grasp.target_track_id,
            label=obj.label,
            grasp=grasp.to_log(),
            evidence=evidence.to_log(),
            held_grasp=record.to_log(),
        )

    def _record_grasp(self, track_id: str, obj: Any, scene_after: Any) -> HeldGrasp:
        """Where the held object sits in the gripper frame, measured now, at the lift.

        Only on a lane with gripper feedback: without it the held object's
        perceived centre is the table-plane point its *lifted* box maps to,
        parallax-shifted 7-29 mm, so an "offset" would be a camera artefact
        (see ``Place``). An offset larger than
        :attr:`Place._MAX_AIM_CORRECTION_M` is a mis-detection, not a grasp.
        """
        tcp = self.ctx.robot.tcp_pose()
        offset: tuple[float, float, float] | None = None
        held = scene_after.get(track_id) if scene_after is not None else None
        if self.ctx.config.robot.gripper_feedback and held is not None:
            world = np.asarray(held.bbox.center.position, dtype=np.float64) - np.asarray(
                tcp.position, dtype=np.float64
            )
            if float(np.linalg.norm(world[:2])) <= Place._MAX_AIM_CORRECTION_M:
                offset = tuple(float(v) for v in offset_in_gripper(held.bbox.center.position, tcp))
        extents = getattr(getattr(obj, "bbox", None), "extents", None)
        return HeldGrasp(
            track_id=track_id,
            offset_in_gripper=offset,  # type: ignore[arg-type]
            grasp_quat=tuple(float(v) for v in np.asarray(tcp.quat, dtype=np.float64)),  # type: ignore[arg-type]
            rest_extents=None if extents is None else tuple(float(v) for v in extents),  # type: ignore[arg-type]
        )


class Place(Skill):
    """Place the held object and **stop**.

    Sequence: re-observe the scene, work out where the *object* should come to
    rest (:func:`synthesise_place_target`), try release orientations until one
    has a collision-free transit, move above, lower, open, retreat, re-observe
    and judge the result (:func:`assess_placement`), then WAIT.

    Does **not** return home. Requires something to be held: "place it" with an
    empty gripper is a user error, reported as such.

    Parameters (the parser's place contract):

    * ``{}`` -- back where it was picked up (the pick-time scene snapshot).
    * ``{"relation": R, "target": phrase}`` with ``R`` in
      :data:`PLACE_RELATIONS`; ``"to"`` lets the skill choose (``in`` a
      bowl/bin/basket, ``on`` a flat box/block, else ``next_to``). The target
      may carry qualifiers ("the bowl on the left"); it is grounded with the
      held object excluded. A missing ``relation`` keeps the historical ``on``.
    * ``{"relation": "direction", "direction": left|right|forward|back,
      "distance": m}`` -- relative to where the object was picked, robot frame.
    * ``{"position": [x, y, z]}`` -- the TCP release point, as before.

    Orientation policy for placement transit
    ----------------------------------------
    After a pick the arm holds a wrist orientation suited to the grasp it just
    completed. When the destination lies at a significantly different XY
    position (e.g. block at x=0.45 picked with a side-approach, box at x=0.62
    y=0.28), the carry quaternion frequently has no IK solution above the
    destination: the elbow must reconfigure and Lula cannot find a path.

    A top-down approach (TCP +Z pointing world -Z) is the most IK-reachable
    orientation on a tabletop -- it is the configuration the arm is in at the
    home posture. Trying it first, with angled approaches and then the carry
    orientation as fallbacks, covers the overwhelming majority of real
    placements. Because the chosen orientation can differ from the carry one,
    the hand's release point is computed *per orientation* from the object's
    offset in the gripper frame (:func:`release_tcp_position`).
    """

    skill_name = "place"

    # Top-down TCP orientation: approach axis points world -Z (downward).
    # Rotation matrix: X=[-1,0,0], Y=[0,-1,0], Z=[0,0,-1] => quat [0,1,0,0].
    # This is a 180-degree rotation about world X, the standard overhead-grasp
    # orientation for a Franka Panda and the most IK-reachable placement posture.
    _TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    #: Ignore a compensation larger than this. The held object's pose comes from
    #: perception looking past a gripper that occludes it, so a bad frame can
    #: put its "centre" somewhere absurd. Beyond this the offset is not a grasp
    #: offset, it is a mis-detection, and shifting the release by it would throw
    #: the object further than doing nothing at all.
    _MAX_AIM_CORRECTION_M = 0.12

    def validate(self, params: dict[str, Any]) -> str | None:
        if self.ctx.memory is None:
            return "place requires memory to know what is held"
        if self.ctx.memory.get_held_object() is None:
            return "not holding anything to place"
        if params.get("position") is None and params.get("relation") is not None:
            relation = normalise_relation(params.get("relation"))
            if relation == "direction":
                direction = normalise_direction(params.get("direction"))
                if direction not in PLACE_DIRECTIONS:
                    return (
                        f"unknown direction {params.get('direction')!r}; "
                        f"expected one of {sorted(PLACE_DIRECTIONS)}"
                    )
                try:
                    distance = float(params.get("distance", DEFAULT_PLACE_DISTANCE_M))
                except (TypeError, ValueError):
                    return f"place distance {params.get('distance')!r} is not a number"
                if not 0.0 < distance <= 0.5:
                    return f"a place offset of {distance * 100:.0f} cm is out of range (0-50 cm)"
            elif relation not in PLACE_RELATIONS:
                return (
                    f"unknown place relation {params.get('relation')!r}; expected one of "
                    f"{sorted(PLACE_RELATIONS | {'direction'})}"
                )
            elif not params.get("target"):
                return f"place {relation.replace('_', ' ')} needs a destination object"
        return None

    def _run(self, params: dict[str, Any]) -> SkillResult:
        held_id = self.ctx.memory.get_held_object()

        # Re-observe before computing the destination pose. After a pick the arm
        # has moved; the pre-pick scene graph is stale and the target object may
        # have shifted (knocked during approach, or just tracked with drift).
        scene, (target, description) = _observe_and_resolve(
            self.ctx, lambda seen: self._place_target(params, seen, held_id)
        )
        if target is None:
            return self._infeasible(description)
        destination = scene.get(target.destination_id) if target.destination_id else None

        planner = self.ctx.planner
        controller = self.ctx.controller
        robot = self.ctx.robot
        feedback = bool(self.ctx.config.robot.gripper_feedback)
        clearance = float(self.ctx.config.grasp.place_clearance)
        half_height = self._held_half_height(scene, held_id)
        offset = self._held_offset_in_gripper(scene, held_id)

        # The carried object must not be an obstacle to its own placement.
        planner.exclude_from_collision(held_id)
        if hasattr(planner, "set_held_object"):
            # Hardware planner: name the held object so its box is swept along
            # the path instead of inferred from proximity.
            planner.set_held_object(held_id)
        try:
            def hold_grip(_index: int, _positions: NDArray[np.float64]) -> bool:
                controller.maintain_grasp()
                return True

            candidate_quats = self._candidate_quats(target.object_xy)

            transit = None
            release_position: NDArray[np.float64] | None = None
            release_quat: NDArray[np.float64] | None = None
            above_position = None
            any_ik_reachable = False
            line_refused = False

            for cand_quat in candidate_quats:
                release = _clamp_to_workspace(
                    self.ctx,
                    release_tcp_position(
                        target, cand_quat, clearance=clearance,
                        held_half_height=half_height, offset_gripper=offset,
                    ),
                )
                above = _clamp_to_workspace(self.ctx, release + np.array([0.0, 0.0, 0.08]))
                above_position = above
                above_cand = Pose(above, cand_quat, Frame.WORLD)
                if robot.inverse_kinematics(above_cand) is None:
                    continue
                any_ik_reachable = True
                transit = planner.plan_with_retries(robot.get_state().joint_state, above_cand, scene)
                if transit is None:
                    transit, refused = self._transit_line(above_cand, scene)
                    line_refused = line_refused or refused
                if transit is not None:
                    release_position = release
                    release_quat = np.asarray(cand_quat, dtype=np.float64)
                    break

            if transit is None or release_position is None:
                if not any_ik_reachable:
                    why = f"pose {np.round(above_position, 3).tolist()} has no IK solution"
                elif line_refused:
                    why = (
                        "no collision-free path; a straight line there would pass an obstacle "
                        "the planner refused to cross, so it was not taken"
                    )
                else:
                    why = "no collision-free path"
                return self._infeasible(f"cannot reach a position above {description}: {why}")

            if not controller.follow_trajectory(transit, on_step=hold_grip):
                return self._fail(self._motion_failure("transit to the destination aborted"))

            refinement: dict[str, Any] | None = None
            if feedback and destination is not None and target.check in ("on_top", "inside"):
                target, destination, release_position, refinement = self._refine_from_above(
                    target, destination, release_position, release_quat,
                    clearance=clearance, half_height=half_height, offset=offset,
                    hold_grip=hold_grip,
                )

            # Lower straight down.
            lower_target = Pose(release_position, robot.tcp_pose().quat, Frame.WORLD)
            lower = planner.plan_cartesian_line(
                robot.get_state().joint_state, lower_target, scene
            )
            if lower is None:
                lower = planner.plan_with_retries(
                    robot.get_state().joint_state, lower_target, scene
                )
            if lower is None:
                return self._infeasible("cannot lower to the release height")
            if not controller.follow_trajectory(lower, on_step=hold_grip):
                return self._fail(self._motion_failure("lowering aborted"))

            # Release. The object settles under gravity, on its own.
            controller.open_gripper_blocking()
            self.ctx.sim.step(self.ctx.config.simulation.settle_steps)

            # Retreat straight up so the fingers clear what was just placed.
            retreat_target = Pose(
                _clamp_to_workspace(
                    self.ctx, robot.tcp_pose().position + np.array([0.0, 0.0, 0.12])
                ),
                robot.tcp_pose().quat,
                Frame.WORLD,
            )
            retreat = planner.plan_cartesian_line(
                robot.get_state().joint_state, retreat_target, scene
            )
            if retreat is not None:
                controller.follow_trajectory(retreat)

            self.ctx.memory.set_held_object(None)
            self.ctx.held_grasp = None

            # Judge the place from what is seen now, never from having opened.
            self.ctx.sim.render_step(3)
            after = self.ctx.vision.observe()
            placed = self._seen_now(after, held_id)
            name = describe(placed) if placed is not None else self._name_of(scene, held_id)
            tolerance = float(
                getattr(self.ctx.config.grasp, "place_tolerance_m", DEFAULT_PLACE_TOLERANCE_M)
            )
            check = assess_placement(
                target, placed, destination, tolerance_m=tolerance,
                vertical_known=feedback, name=name,
            )
            data: dict[str, Any] = {
                "track_id": held_id,
                "release_position": release_position.tolist(),
                "object_target": list(target.object_xy),
                "place_target": target.to_log(),
                "settled_offset": check.offset_m,
                "settled_offset_axes": "xy",
                "tolerance_m": tolerance,
                "placement": check.to_log(),
                "destination_refinement": refinement,
            }
            self.ctx.emit("place.verification", {"target": held_id, **check.to_log(),
                                                  "place_target": target.to_log()})
            if check.ok:
                return self._ok(
                    f"placed {target.description} (settled {check.offset_m * 1000:.0f} mm "
                    "from the target, horizontally)",
                    **data,
                )
            # Released but not where it was sent: a failure, with the numbers.
            # Not retryable -- the jaw is open and nothing is held, so a retry
            # could only say "not holding anything" and bury this message.
            return self._fail(
                f"place missed {target.description}: {check.reason}; the gripper is open "
                "and nothing is held",
                released=True,
                retryable=False,
                **data,
            )
        finally:
            planner.include_in_collision(held_id)

    # ------------------------------------------------------------------
    # destination
    # ------------------------------------------------------------------

    def _place_target(
        self, params: dict[str, Any], scene: Any, held_id: str
    ) -> tuple[PlaceTarget | None, str]:
        """Resolve the parameters into where the object should come to rest."""
        clearance = float(self.ctx.config.grasp.place_clearance)
        support = float(self.ctx.support_height)
        held_extents = self._held_extents(scene, held_id)

        if params.get("position") is not None:
            position = np.asarray(params["position"], dtype=np.float64)
            return PlaceTarget(
                "position", (float(position[0]), float(position[1])), support,
                f"at {np.round(position, 3).tolist()}", tcp_z=float(position[2]),
                aim_object=False,
            ), ""

        relation = normalise_relation(params.get("relation") or ("on" if params.get("target") else ""))
        half_height = self._held_half_height(scene, held_id)
        offset = self._held_offset_in_gripper(scene, held_id)

        def reachable(object_xy: NDArray[np.float64]) -> bool:
            """IK exists for the release and the pose above it, at some candidate orientation."""
            probe = PlaceTarget("probe", (float(object_xy[0]), float(object_xy[1])), support, "")
            for quat in self._candidate_quats(object_xy):
                release = _clamp_to_workspace(
                    self.ctx,
                    release_tcp_position(probe, quat, clearance=clearance,
                                         held_half_height=half_height, offset_gripper=offset),
                )
                above = _clamp_to_workspace(self.ctx, release + np.array([0.0, 0.0, 0.08]))
                robot = self.ctx.robot
                if (robot.inverse_kinematics(Pose(release, quat, Frame.WORLD)) is not None
                        and robot.inverse_kinematics(Pose(above, quat, Frame.WORLD)) is not None):
                    return True
            return False

        common = dict(
            scene=scene, held_id=held_id, support_height=support,
            workspace_min=self.ctx.config.scene.workspace_min,
            workspace_max=self.ctx.config.scene.workspace_max,
            clearance=clearance, held_extents=held_extents, robot_xy=_robot_xy(self.ctx),
            reachable=reachable,
        )

        if relation == "direction":
            origin_xy, where = self._pick_origin(held_id)
            if origin_xy is None:
                return None, "cannot place relative to where it was picked: that position is unknown"
            return synthesise_place_target(
                "direction", origin_xy=origin_xy, origin_description=where,
                direction=params.get("direction"),
                distance=float(params.get("distance", DEFAULT_PLACE_DISTANCE_M)),
                **common,
            )

        if params.get("target"):
            # Grounded with the held object excluded: "it" / "the can" can
            # never name the thing in the jaw as its own destination. Reference
            # errors propagate (a question, or one refusal).
            destination = _resolve_object(
                self.ctx, params, scene=scene, exclude_ids=(held_id,)
            )
            if destination.track_id == held_id:
                return None, "the destination is the object being held"
            return synthesise_place_target(relation, destination=destination, **common)

        # No destination given: put it back where it was picked up.
        #
        # Releasing at the hand's current XY looks equivalent and is not. The
        # hand is wherever the last command left it -- after "move up 10 cm" it
        # has also drifted a centimetre or two sideways, and after a re-centred
        # grasp it is offset from the object's origin. Measured on the mug: the
        # release landed far enough off that the mug rolled out of the scene
        # entirely and knocked the bottle flat on its way, turning one
        # successful pick into two displaced objects and a "success" log.
        #
        # The pick origin is recovered from the scene memory snapshotted when
        # the object was grasped, so "place it" is the inverse of "pick it up".
        # Height still comes from the support surface rather than the recorded
        # centre: perception measures a held object through a partly occluding
        # gripper, and trusting that z would release it into the table.
        origin_xy, where = self._pick_origin(held_id)
        if origin_xy is None:
            # No usable snapshot (picked before the first observe, or the track
            # was lost). The hand's position is a worse answer, but it is the
            # only one left, and refusing to place would strand the object.
            origin_xy = np.asarray(self.ctx.robot.tcp_pose().position[:2], dtype=np.float64)
            where = "the current position"
        # The TCP goes over the origin itself: "place it" is measured at 8-26
        # mm on the sim lane this way, and is kept exactly as it was.
        return PlaceTarget(
            "origin", (float(origin_xy[0]), float(origin_xy[1])), support, where,
            aim_object=False,
        ), ""

    def _refine_from_above(
        self,
        target: PlaceTarget,
        destination: ObjectHypothesis,
        release_position: NDArray[np.float64],
        release_quat: NDArray[np.float64] | None,
        *,
        clearance: float,
        half_height: float,
        offset: NDArray[np.float64] | None,
        hold_grip: Callable[[int, NDArray[np.float64]], bool],
    ) -> tuple[PlaceTarget, ObjectHypothesis, NDArray[np.float64], dict[str, Any]]:
        """Look at an ``on``/``in`` destination from above it and re-centre the release.

        Sim lane only (the caller checks): it is the lane with a wrist camera,
        which from the pose above the destination sees it from about 20 cm
        instead of the exterior camera's 1.5 m. The hardware lane's one fixed
        overhead camera would only see the arm over the destination.

        Measured (see the module docstring): the exterior-only estimate of the
        green box was 48 mm off the wrist view's, and a block released on it
        tipped off the real corner. Every guard keeps the original release:
        the destination not re-seen this frame, a jump over
        :data:`PLACE_REFINE_MAX_SHIFT_M`, a truncated view, a corrected release
        with no IK, or no straight line to the corrected pose above it.
        """
        self.ctx.sim.render_step(3)
        after = self.ctx.vision.observe()
        seen = self._seen_now(after, destination.track_id)
        refined, shift, reason = refine_place_target(target, destination, seen)
        log: dict[str, Any] = {
            "destination_id": destination.track_id,
            "before_xy": [round(float(v), 4) for v in destination.bbox.center.position[:2]],
            "now_xy": None if seen is None
            else [round(float(v), 4) for v in seen.bbox.center.position[:2]],
            "shift_m": round(shift, 4),
            "applied": False,
            "reason": reason,
        }
        kept = (target, destination, release_position, log)
        if reason or seen is None or release_quat is None:
            self.ctx.emit("place.refine", log)
            return kept
        if shift < 1e-3:
            log["reason"] = "the close view agrees"
            self.ctx.emit("place.refine", log)
            return kept

        robot = self.ctx.robot
        release = _clamp_to_workspace(
            self.ctx,
            release_tcp_position(refined, release_quat, clearance=clearance,
                                 held_half_height=half_height, offset_gripper=offset),
        )
        above = _clamp_to_workspace(self.ctx, release + np.array([0.0, 0.0, 0.08]))
        if (robot.inverse_kinematics(Pose(release, release_quat, Frame.WORLD)) is None
                or robot.inverse_kinematics(Pose(above, release_quat, Frame.WORLD)) is None):
            log["reason"] = "the re-centred release has no IK solution"
            self.ctx.emit("place.refine", log)
            return kept
        # Across at the height above the destination, then the usual lowering.
        across = self.ctx.planner.plan_cartesian_line(
            robot.get_state().joint_state, Pose(above, robot.tcp_pose().quat, Frame.WORLD), after
        )
        if across is None:
            log["reason"] = "no straight line to the pose above the re-centred release"
            self.ctx.emit("place.refine", log)
            return kept
        if not self.ctx.controller.follow_trajectory(across, on_step=hold_grip):
            # The hand is still above the destination and still holding: the
            # usual lowering to the original release goes on from here.
            log["reason"] = "the move across to the re-centred release aborted"
            self.ctx.emit("place.refine", log)
            return kept
        log.update(applied=True, release_position=[round(float(v), 4) for v in release])
        _log.info(
            "Re-centred the release on the %s by %.0f mm from the view above it",
            describe(seen), shift * 1000.0,
        )
        self.ctx.emit("place.refine", log)
        return refined, seen, release, log

    def _candidate_quats(self, object_xy: ArrayLike) -> list[NDArray[np.float64]]:
        """Release orientations to try, in order: top-down, 60 and 45 deg radial, then the carry."""
        bearing = float(np.arctan2(float(object_xy[1]), float(object_xy[0])))
        quats = [self._TOP_DOWN_QUAT]
        for pitch_deg in (60.0, 45.0):
            rad_pitch = np.radians(pitch_deg)
            z_approach = np.array([
                np.cos(bearing) * np.cos(rad_pitch),
                np.sin(bearing) * np.cos(rad_pitch),
                -np.sin(rad_pitch),
            ])
            y_axis = np.array([-np.sin(bearing), np.cos(bearing), 0.0])
            x_axis = np.cross(y_axis, z_approach)
            quats.append(tf.matrix_to_quat(np.stack([x_axis, y_axis, z_approach], axis=1)))
        quats.append(np.asarray(self.ctx.robot.tcp_pose().quat, dtype=np.float64))
        return quats

    def _pick_origin(self, held_id: str) -> tuple[NDArray[np.float64] | None, str]:
        """Where the held object rested before the pick, from memory's snapshot."""
        picked_from = self.ctx.memory.scene_at_pick() if self.ctx.memory is not None else None
        if picked_from is not None:
            resting = picked_from.get(held_id)
            if resting is not None:
                return (
                    np.asarray(resting.bbox.center.position[:2], dtype=np.float64),
                    "where it was picked up",
                )
        return None, ""

    def _held_extents(self, scene: Any, held_id: str) -> NDArray[np.float64] | None:
        """The held object's size: the unoccluded resting box when the pick recorded it."""
        record = self.ctx.held_grasp
        if isinstance(record, HeldGrasp) and record.track_id == held_id and record.rest_extents:
            return np.asarray(record.rest_extents, dtype=np.float64)
        obj = scene.get(held_id) if scene is not None else None
        return None if obj is None else np.asarray(obj.bbox.extents, dtype=np.float64)

    # ------------------------------------------------------------------
    # aiming
    # ------------------------------------------------------------------

    def _held_offset_in_gripper(self, scene: Any, held_id: str) -> NDArray[np.float64] | None:
        """The held object's centre relative to the TCP, in the gripper frame.

        Aiming the OBJECT at the destination, not the hand: measured while
        held, the marker's centre is 22-26 mm from the TCP, mostly sideways
        because a 118 mm marker is gripped near one end, and with the hand over
        the middle of a 172 mm bowl the marker hangs over the rim.

        The offset is kept in the *gripper* frame (the frame the object is
        rigidly held in) and rotated into each candidate release orientation.
        The previous version subtracted the world-frame offset measured in the
        carry orientation, which is only right when the transit keeps the
        wrist's yaw; the top-down orientation Place tries first usually does
        not (audit probe: 25 mm offset, 19/35/50 mm landing error at 45/90/180
        deg of yaw, against 25 mm with no correction at all).

        Source, in order: the pick's own record (measured at the verified
        lift), else the held object as observed now. ``None`` on the
        depthless lane (``robot.gripper_feedback`` false): there the held
        object's perceived centre is the table-plane point its *lifted* box
        maps to, parallax-shifted 7-29 mm, so the "offset" is a camera
        artefact. Measured on the honest fake lane: a marker held dead centre
        read 28 mm off, the release was moved 28 mm the wrong way and the
        lowering had no plan. The top-down grasp there is aimed at the
        object's centre, so no correction is the better estimate.
        """
        if not self.ctx.config.robot.gripper_feedback:
            return None
        record = self.ctx.held_grasp
        if isinstance(record, HeldGrasp) and record.track_id == held_id:
            if record.offset_in_gripper is not None:
                return np.asarray(record.offset_in_gripper, dtype=np.float64)
        held = scene.get(held_id) if scene is not None else None
        if held is None:
            return None
        tcp = self.ctx.robot.tcp_pose()
        world = np.asarray(held.bbox.center.position, dtype=np.float64) - np.asarray(
            tcp.position, dtype=np.float64
        )
        magnitude = float(np.linalg.norm(world[:2]))
        if magnitude > self._MAX_AIM_CORRECTION_M:
            _log.debug(
                "Ignoring %.0f mm aim correction for %s: larger than a plausible "
                "grasp offset, so the held pose is probably a mis-detection",
                magnitude * 1000.0,
                held_id,
            )
            return None
        return offset_in_gripper(held.bbox.center.position, tcp)

    def _held_half_height(self, scene: Any, held_id: str, fallback: float = 0.03) -> float:
        """Half the height of the held object, from perception.

        The TCP sits roughly at the object's middle, so releasing at
        ``surface + clearance`` alone would drop the object from half its own
        height -- enough for a tall object to topple or bounce away from where it
        was meant to go. Using the measured extent releases it just clear of the
        surface instead of dropping it.
        """
        obj = scene.get(held_id) if scene is not None else None
        if obj is None:
            return fallback
        return float(obj.bbox.extents[2]) / 2.0

    # ------------------------------------------------------------------
    # motion guards
    # ------------------------------------------------------------------

    def _transit_line(self, goal: Pose, scene: Any) -> tuple[Any, bool]:
        """The straight-line transit fallback, and whether it was refused.

        Sim lane (``robot.gripper_feedback``): unchanged, a plain Cartesian line.
        Hardware lane: the line must pass the planner's full obstacle check
        (TCP and held box, ``check_objects=True``). Re-review: with the plain
        line, 12 of 108 planner-refused bin crossings with a held cube drove
        it 37-41 mm into the bin, and 20 of 51 waypoints of one put the TCP
        itself inside the bin. A planner without the check gets no fallback.
        """
        planner = self.ctx.planner
        start = self.ctx.robot.get_state().joint_state
        if self.ctx.config.robot.gripper_feedback:
            return planner.plan_cartesian_line(start, goal, scene), False
        if not _accepts_keyword(planner.plan_cartesian_line, "check_objects"):
            return None, True
        line = planner.plan_cartesian_line(start, goal, scene, check_objects=True)
        return line, line is None

    def _motion_failure(self, message: str) -> str:
        """A motion-failure message that says so when the arm went limp mid-motion.

        Hardware lane: a watchdog, host-timeout or serial-loss detach stops
        the motion and relaxes the servos; ``RemoteController`` records it in
        ``last_detach_notice``. Whatever was in the jaw may have dropped, and
        the operator must hear that rather than a bare "aborted".
        """
        notice = getattr(self.ctx.controller, "last_detach_notice", None)
        if notice:
            return (
                f"{message}: the arm went limp ({notice}); what it held may have dropped -- "
                "look before the next command"
            )
        return message

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------

    @staticmethod
    def _seen_now(after: Any, held_id: str) -> Any:
        """The released object only if detected in *this* observation.

        A tracker keeps an unseen track alive for a while with its last pose;
        after a release that pose is the held one, and measuring against it
        would report where the object was, not where it is.
        """
        placed = after.get(held_id) if after is not None else None
        if placed is None:
            return None
        seen_step = getattr(placed, "last_seen_step", None)
        step = getattr(after, "step_index", None)
        if seen_step is not None and step is not None and seen_step < step:
            return None
        return placed

    @staticmethod
    def _name_of(scene: Any, track_id: str) -> str:
        obj = scene.get(track_id) if scene is not None else None
        return describe(obj) if obj is not None else "object"


# ----------------------------------------------------------------------
# control
# ----------------------------------------------------------------------


class Wait(Skill):
    """Do nothing for a while, letting physics settle."""

    skill_name = "wait"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        duration = float(params.get("duration", 1.0))
        steps = max(1, int(duration / self.ctx.config.simulation.physics_dt))
        self.ctx.sim.step(steps)
        return self._ok(f"waited {duration:.2f} s")


class Stop(Skill):
    """Halt motion, keeping the arm actively controlled.

    Does not release: dropping whatever is held would be a new action, and this
    skill was only asked to stop.
    """

    skill_name = "stop"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        self.ctx.controller.stop()
        return self._aborted("stopped; holding position")


class EmergencyStop(Skill):
    """Halt immediately and zero velocities. Still holds, rather than dropping."""

    skill_name = "emergency_stop"

    def _run(self, params: dict[str, Any]) -> SkillResult:
        self.ctx.controller.emergency_stop()
        return self._aborted("emergency stop engaged")


#: Every skill class, for registry construction.
ALL_SKILLS: tuple[type[Skill], ...] = (
    Observe,
    ScanScene,
    LookAt,
    MoveTo,
    MoveRelative,
    Pick,
    Place,
    OpenGripper,
    CloseGripper,
    RotateWrist,
    GoHome,
    Wait,
    Stop,
    EmergencyStop,
)

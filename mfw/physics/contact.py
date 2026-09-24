"""Physics-based grasp verification.

Isaac Sim is imported lazily inside functions; see ``mfw.simulation.app``.

A grasp is verified from **evidence**, never from having issued a close command.
Three independent signals, in increasing strength:

1. **Finger stall** -- the fingers stopped short of fully closed, so something is
   between them. Necessary but not sufficient: they could be jammed on the table.
2. **Object follows the TCP** -- after lifting, the object's perceived position is
   still near the fingertips. This is the signal that actually matters.
3. **Object rose** -- its height increased by roughly the commanded lift, proving
   the arm carried it rather than merely brushing past.

Signal 2 is the one that catches the failure mode that matters most here: with a
fake attachment or a pose write, an object *appears* to be grasped and every log
line reports success while no contact ever occurred. Re-perceiving the object
after the lift cannot be fooled that way, because the evidence comes from the
sensors rather than from the code that commanded the motion.

Without gripper feedback (PWM hobby servos, one overhead webcam, no depth)
---------------------------------------------------------------------------
None of the three signals exists: the "width" is the commanded value and the
homography pins every object to the table plane. The verdict has to come from
the *image*, and the trap measured here is what it must not come from:

* **Table-plane displacement of a lifted object is not evidence.** A carried
  object stays detected (a 140 mm marker protrudes far outside a 45 mm jaw)
  and is re-projected onto the table at a parallax-shifted point. Measured
  through the real estimator: 9-35 mm for the fake lane's oblique camera,
  4-20 mm for the nadir hardware placeholder -- below any displacement gate
  that also rejects a centimetre of detector jitter. The old rule therefore
  reported every genuine lift as "still resting" and then opened the jaw at
  lift height.
* **"Not seen" is not evidence either**, unless it was seen before the
  descent *and* the gripper now covers where a carried object would be but
  not where it rested (see (c)). A small object shadowed by the arm is "not
  seen" whether or not anything was gripped; that is why the skill takes the
  pre-grasp snapshot *before* the descent, from the standoff.

What does discriminate carried / missed / knocked-away, all from the same
pixel boxes the detector already reports:

(a) **Bbox growth.** An object that rose 6 cm toward a camera 0.6 m above the
    table grows by H/(H-h) ~ 11 % (measured 8-15 % across the workspace for
    both cameras above, marker and cube alike); one resting on the table does
    not grow at all. When the camera model is measured
    (``exterior_camera.pose_measured``) the expected growth comes from
    projecting the object's box at rest and at the lift height; otherwise a
    nadir approximation from the configured camera height is used, with a 4 %
    floor so detector jitter never counts as a lift.
(b) **The tool projects inside the object's box and the object appears where
    a carried one would** -- the object's 3-D box is placed at the TCP, projected,
    and mapped through the same homography the observation goes through. Both
    need the camera model; without it (a) alone decides.
(c) **Occlusion**: not seen now, seen before the descent, the projected
    gripper footprint covers where a *carried* object would now be, and it
    does **not** cover the spot where the object rested -> held (inferred,
    and said so). The last condition is the one a literal "the gripper covers
    its last box" rule misses: with the arm lifted straight above the rest
    spot (a nadir camera, a small object) a *missed* object lying there is
    hidden exactly as well as a carried one, so absence proves nothing and
    the verdict is *unknown*. Anything that is not seen and not explained
    this way is *unknown*, which is reported as not holding -- and the skill
    then keeps the jaw closed rather than dropping a possibly held object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mfw.core.types import ObjectHypothesis, Pose, SceneGraph
from mfw.utils.logging import get_logger

__all__ = ["GraspEvidence", "LiftPrediction", "verify_grasp", "read_net_contact_force"]

_log = get_logger("physics.contact")

_MIN_PIXEL_GROWTH = 0.04
"""Linear bbox growth (fraction) below which a "rise" is detector jitter. A
marker 140 px long jitters by 2-3 px on OWL-ViT; 4 % is 5-6 px."""
_GROWTH_FRACTION_REQUIRED = 0.5
"""Share of the *expected* growth the object must show. Half, because the fake
world and a real grasp both lift the object slightly more or less than the
commanded height (grasped below centre, table flex), and the point is to
separate ~10 % from ~0 %, not to measure the lift."""
_PREDICTION_TOLERANCE_M = 0.03
"""How far the observed table-plane estimate may sit from the one a carried
object would produce. Absorbs the few millimetres of grasp-height error the
prediction cannot know about (parallax of ~1 cm of height is ~5 mm here)."""
_TCP_BOX_MARGIN_PX = 12.0
"""Slack around the object's pixel box when asking whether the projected tool
sits inside it; the TCP is the jaw midpoint, so a held object surrounds it."""
_OCCLUSION_COVER_FRACTION = 0.6
"""Share of the *carried* box the projected jaw footprint must cover before
"not seen" can be explained by the jaw hiding it (the fake detector hides at
the same share)."""
_REST_VISIBLE_MAX_COVER = 0.35
"""Most of the *rest* box the jaw footprint may cover for a left-behind object
to count as one the camera would still see. Much lower than the share above
on purpose: whether a real detector finds a half-covered object is not known,
and an absence that a half-covered left-behind object could also produce is
reported as unknown, not as held."""

_HOLDING_VERDICTS = frozenset({"carried", "occluded"})

BoxPx = tuple[float, float, float, float]


@dataclass(frozen=True)
class LiftPrediction:
    """What a *held* object should look like after the lift, from the camera model.

    Built by ``PlanarPerception.predict_lift`` on the hardware lane and handed
    to :func:`verify_grasp`; the sim lane never constructs one. Every field is
    optional because the camera model is optional: with
    ``exterior_camera.pose_measured`` false only ``expected_pixel_scale`` (a
    nadir approximation) can be filled, and the verdict falls back to bbox
    growth alone.
    """

    expected_pixel_scale: float | None = None
    """Linear bbox growth factor if the object rose by the lift height."""
    predicted_xy: tuple[float, float] | None = None
    """Table-plane estimate the observation would produce for the object
    carried at the TCP (the parallax-shifted point, not the rest pose)."""
    predicted_bbox_px: BoxPx | None = None
    """Pixel box of the object's 3-D box placed at the TCP."""
    tcp_px: tuple[float, float] | None = None
    """Where the TCP itself projects."""
    arm_bbox_px: BoxPx | None = None
    """Pixel box of the gripper/wrist footprint above the TCP: the occluder."""
    model_available: bool = False
    """Whether a measured pinhole model produced the pixel predictions."""

    def to_log(self) -> dict[str, Any]:
        """JSON-safe dict for the ``pick.verification`` event."""
        return {
            "expected_pixel_scale": self.expected_pixel_scale,
            "predicted_xy": None if self.predicted_xy is None else list(self.predicted_xy),
            "predicted_bbox_px": None if self.predicted_bbox_px is None else list(self.predicted_bbox_px),
            "tcp_px": None if self.tcp_px is None else list(self.tcp_px),
            "arm_bbox_px": None if self.arm_bbox_px is None else list(self.arm_bbox_px),
            "model_available": self.model_available,
        }


@dataclass(frozen=True)
class GraspEvidence:
    """The measurements behind a grasp verdict.

    Kept as data rather than a bare bool so a failure can be diagnosed from the
    event log without re-running the pick.
    """

    fingers_stalled: bool
    gripper_width: float
    object_tracked: bool
    object_to_tcp_distance: float
    height_gain: float
    expected_height_gain: float
    contact_force: float | None
    #: Smallest measured side of the target. A parallel gripper spans this
    #: axis, so it is what a stall width has to be consistent with.
    object_min_extent: float = 0.0
    width_plausible: bool = True
    gripper_feedback: bool = True
    """Whether the gripper reports a measured width. PWM hobby servos do not:
    their "width" is the commanded value, a stall can never be observed, and
    the verdict has to come from perception alone."""
    object_visible: bool = True
    """Whether the target was re-detected in the post-lift observation itself,
    rather than carried forward by the tracker from an earlier frame."""
    object_displacement: float = 0.0
    """Horizontal distance, metres, between the pre-grasp and post-lift centres."""
    object_displaced: bool = False
    """``object_displacement`` reached the configured minimum. Reported for the
    log; on the feedback-less lane it never decides anything (see module doc)."""
    visible_before: bool = True
    """Whether the target was detected in the pre-descent observation itself."""
    pixel_scale: float | None = None
    """Post-lift / pre-descent bbox diagonal ratio, when both boxes exist."""
    expected_pixel_scale: float | None = None
    """The ratio a lift by the commanded height should produce, if predictable."""
    pixel_grew: bool = False
    """``pixel_scale`` shows at least the required share of the expected growth."""
    tcp_in_object_box: bool | None = None
    """Projected TCP lies inside the post-lift box (``None``: no camera model)."""
    predicted_offset: float | None = None
    """Metres between the observed estimate and the carried-object prediction."""
    on_prediction: bool | None = None
    """``predicted_offset`` within tolerance (``None``: no camera model)."""
    occluded_by_arm: bool = False
    """Not seen now, seen before, the gripper footprint covers where a carried
    object would be, and it does not cover the rest spot."""
    rest_hidden_by_arm: bool | None = None
    """For an object not seen now: whether the gripper footprint also covers
    the spot it rested on, which makes its absence uninformative (``None``:
    not evaluated -- visible, or no camera model)."""
    model_available: bool = False
    """Whether a measured camera model backed the pixel predictions."""
    verdict: str = ""
    """Feedback-less outcome: ``carried``, ``occluded``, ``resting``,
    ``knocked`` or ``unknown``. Empty on the feedback lane."""

    @property
    def holding(self) -> bool:
        """Whether the robot is actually holding the object.

        Three conditions, all necessary: the fingers stalled, the object is
        still near the fingertips, and the stall width is consistent with the
        object's own size.

        The width test is what stops a false positive. Any stall short of fully
        closed used to count, so a gripper that closed to 13.3 mm beside a mug
        whose smallest measured side was 35 mm reported "holding" -- and the
        lift that followed moved the mug from z=0.441 to z=0.444, three
        millimetres, because nothing was between the fingers. Fingers that close
        far past an object are not gripping it, and a pick that reports success
        while the object stays on the table is worse than one that fails.

        Height gain is reported but still not required: a lift can be cut short
        by the workspace ceiling while the grasp is perfectly good.

        **Without gripper feedback** none of the above is measurable and the
        verdict is the one :func:`verify_grasp` reached from pixel evidence:
        ``carried`` (the box grew with the lift and, when a camera model
        exists, the tool projects inside it where a carried object would be)
        or ``occluded`` (seen before the descent, now hidden under the
        projected gripper). ``resting``, ``knocked`` and ``unknown`` are all
        "not holding": a stationary box, a box that slid but did not rise, and
        an absence nothing can attribute to the gripper.
        """
        if not self.gripper_feedback:
            if self.verdict:
                return self.verdict in _HOLDING_VERDICTS
            return self.pixel_grew or self.occluded_by_arm
        return self.fingers_stalled and self.object_tracked and self.width_plausible

    def to_log(self) -> dict[str, Any]:
        return {
            "holding": self.holding,
            "fingers_stalled": self.fingers_stalled,
            "gripper_width": self.gripper_width,
            "object_tracked": self.object_tracked,
            "object_to_tcp_distance": self.object_to_tcp_distance,
            "height_gain": self.height_gain,
            "expected_height_gain": self.expected_height_gain,
            "contact_force": self.contact_force,
            "object_min_extent": self.object_min_extent,
            "width_plausible": self.width_plausible,
            "gripper_feedback": self.gripper_feedback,
            "object_visible": self.object_visible,
            "object_displacement": self.object_displacement,
            "object_displaced": self.object_displaced,
            "visible_before": self.visible_before,
            "pixel_scale": self.pixel_scale,
            "expected_pixel_scale": self.expected_pixel_scale,
            "pixel_grew": self.pixel_grew,
            "tcp_in_object_box": self.tcp_in_object_box,
            "predicted_offset": self.predicted_offset,
            "on_prediction": self.on_prediction,
            "occluded_by_arm": self.occluded_by_arm,
            "rest_hidden_by_arm": self.rest_hidden_by_arm,
            "model_available": self.model_available,
            "verdict": self.verdict,
        }

    def reason(self) -> str:
        """Human-readable explanation, for the skill result message."""
        if not self.gripper_feedback:
            return self._feedbackless_reason()
        if self.holding:
            return (
                f"holding: fingers stalled at {self.gripper_width * 1000:.1f} mm, "
                f"object {self.object_to_tcp_distance * 1000:.0f} mm from the fingertips"
            )
        if not self.fingers_stalled:
            return (
                f"gripper closed to {self.gripper_width * 1000:.1f} mm without stalling: "
                "nothing between the fingers"
            )
        if not self.width_plausible:
            return (
                f"fingers stalled at {self.gripper_width * 1000:.1f} mm, far under the "
                f"object's {self.object_min_extent * 1000:.0f} mm smallest side: "
                "closed past it, not on it"
            )
        return (
            f"fingers stalled at {self.gripper_width * 1000:.1f} mm but the object is "
            f"{self.object_to_tcp_distance * 1000:.0f} mm from the fingertips: not carried"
        )

    def _feedbackless_reason(self) -> str:
        moved = f"{self.object_displacement * 1000:.0f} mm"
        scale = "" if self.pixel_scale is None else f"bbox x{self.pixel_scale:.2f}"
        verdict = self.verdict or ("carried" if self.pixel_grew else "occluded" if self.occluded_by_arm else "unknown")
        if verdict == "carried":
            growth = f"bbox grew {(self.pixel_scale or 1.0) * 100 - 100:.0f} %"
            if self.expected_pixel_scale:
                growth += f" (expected ~{self.expected_pixel_scale * 100 - 100:.0f} % for the lift)"
            tool = "; the tool projects inside its box" if self.tcp_in_object_box else ""
            return f"holding (no gripper feedback): the object rose with the jaw ({growth}{tool})"
        if verdict == "occluded":
            return (
                "holding (no gripper feedback, inferred): the object was seen before the "
                "descent and is now hidden under the gripper's projected footprint; confirm by eye"
            )
        if verdict == "knocked":
            return (
                f"the object moved {moved} across the table but did not rise ({scale}): "
                "knocked, not carried"
            )
        if verdict == "resting":
            return (
                f"the object is still resting where it was ({moved} moved, {scale}) "
                "after the lift: not carried"
            )
        # unknown
        if self.object_visible:
            return (
                "cannot confirm the grasp: the object is seen now but was not detected in the "
                "pre-descent view, so there is no box to compare its size with"
            )
        if not self.visible_before:
            return (
                "cannot confirm the grasp: the object was not seen before the descent, "
                "so its absence now proves nothing"
            )
        if not self.model_available:
            return (
                "cannot confirm the grasp: the object is no longer seen and the camera pose "
                "is not measured (exterior_camera.pose_measured), so the gripper cannot be "
                "shown to be hiding it"
            )
        if self.rest_hidden_by_arm:
            return (
                "cannot confirm the grasp: the object is no longer seen, but the gripper now "
                "also covers the spot where it rested, so an object left behind would be "
                "hidden just the same"
            )
        return (
            "cannot confirm the grasp: the object is no longer seen but the gripper's "
            "projected footprint does not cover where it was"
        )


# ----------------------------------------------------------------------
# pixel helpers
# ----------------------------------------------------------------------


def _bbox_px(obj: ObjectHypothesis | None) -> BoxPx | None:
    """The detector's pixel box carried on a hardware-lane hypothesis, if any."""
    if obj is None:
        return None
    box = obj.attributes.get("bbox_px") if isinstance(obj.attributes, Mapping) else None
    if box is None or len(box) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    if not all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _diagonal(box: BoxPx) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _point_in_box(point: Sequence[float], box: BoxPx, margin: float) -> bool:
    u, v = float(point[0]), float(point[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return False
    pad_u = margin + 0.1 * (box[2] - box[0])
    pad_v = margin + 0.1 * (box[3] - box[1])
    return box[0] - pad_u <= u <= box[2] + pad_u and box[1] - pad_v <= v <= box[3] + pad_v


def _covered_fraction(inner: BoxPx, cover: BoxPx) -> float:
    """Share of ``inner``'s area that ``cover`` overlaps."""
    w = max(0.0, min(inner[2], cover[2]) - max(inner[0], cover[0]))
    h = max(0.0, min(inner[3], cover[3]) - max(inner[1], cover[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return 0.0 if area <= 0.0 else (w * h) / area


def _feedbackless_fields(
    before: ObjectHypothesis | None,
    after: ObjectHypothesis | None,
    object_visible: bool,
    visible_before: bool,
    object_displaced: bool,
    prediction: LiftPrediction | None,
) -> dict[str, Any]:
    """The pixel-evidence fields and the verdict for the feedback-less lane."""
    pred = prediction if prediction is not None else LiftPrediction()
    before_box = _bbox_px(before)
    after_box = _bbox_px(after) if object_visible else None
    fields: dict[str, Any] = {
        "visible_before": visible_before,
        "pixel_scale": None,
        "expected_pixel_scale": pred.expected_pixel_scale,
        "pixel_grew": False,
        "tcp_in_object_box": None,
        "predicted_offset": None,
        "on_prediction": None,
        "occluded_by_arm": False,
        "rest_hidden_by_arm": None,
        "model_available": pred.model_available,
    }

    if not object_visible:
        arm = pred.arm_bbox_px
        if visible_before and before_box is not None and arm is not None:
            rest_hidden = _covered_fraction(before_box, arm) > _REST_VISIBLE_MAX_COVER
            carried_hidden = (
                pred.predicted_bbox_px is not None
                and _covered_fraction(pred.predicted_bbox_px, arm) >= _OCCLUSION_COVER_FRACTION
            )
            fields["rest_hidden_by_arm"] = rest_hidden
            fields["occluded_by_arm"] = bool(carried_hidden and not rest_hidden)
        fields["verdict"] = "occluded" if fields["occluded_by_arm"] else "unknown"
        return fields

    assert after is not None
    if not visible_before or before_box is None or after_box is None:
        # Seen now, but there is no fresh pre-descent box to compare with: a
        # box carried forward by the tracker is from some earlier frame and
        # may predate a nudge. Nothing here can say "carried".
        fields["verdict"] = "unknown"
        return fields
    scale = _diagonal(after_box) / max(_diagonal(before_box), 1e-9)
    fields["pixel_scale"] = scale
    expected = pred.expected_pixel_scale
    required = _MIN_PIXEL_GROWTH
    if expected is not None and expected > 1.0:
        required = max(_MIN_PIXEL_GROWTH, _GROWTH_FRACTION_REQUIRED * (expected - 1.0))
    fields["pixel_grew"] = (scale - 1.0) >= required
    if pred.tcp_px is not None:
        fields["tcp_in_object_box"] = _point_in_box(pred.tcp_px, after_box, _TCP_BOX_MARGIN_PX)
    if pred.predicted_xy is not None:
        offset = float(np.linalg.norm(after.pose.position[:2] - np.asarray(pred.predicted_xy, dtype=np.float64)))
        fields["predicted_offset"] = offset
        fields["on_prediction"] = offset <= _PREDICTION_TOLERANCE_M

    carried = (
        fields["pixel_grew"]
        and fields["tcp_in_object_box"] is not False
        and fields["on_prediction"] is not False
    )
    if carried:
        fields["verdict"] = "carried"
    elif object_displaced:
        fields["verdict"] = "knocked"
    else:
        fields["verdict"] = "resting"
    return fields


# ----------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------


def verify_grasp(
    robot: Any,
    scene_before: SceneGraph,
    scene_after: SceneGraph,
    track_id: str,
    closed_width: float,
    expected_height_gain: float,
    max_object_to_tcp: float = 0.09,
    stall_margin: float = 0.004,
    min_grasp_fraction: float = 0.6,
    gripper_feedback: bool = True,
    min_displacement: float = 0.03,
    lift_prediction: LiftPrediction | None = None,
) -> GraspEvidence:
    """Decide whether ``track_id`` is being held, from sensor evidence only.

    ``scene_before`` supplies the pre-lift height so the gain can be measured
    against what the object actually did, not against where it was expected to be.

    ``min_grasp_fraction`` is the share of the object's smallest measured side
    the fingers must still span to count as holding it. 0.6 was chosen against
    measured cases: a good mug grasp stalls at 74 mm on a 35-65 mm side (ratio
    above 1.0), the documented soup-can grasp stalls at 60.3 mm on 68 mm (0.89),
    and the false positive stalled at 13.3 mm on 35 mm (0.38). Anything between
    0.4 and 0.85 separates them; 0.6 sits in the middle rather than on either
    edge, because perception's extents move by tens of millimetres between
    frames and a threshold hugging a real case would flip with that noise.

    ``gripper_feedback=False`` switches the verdict to the pixel evidence
    described in the module docstring: bbox growth between the ``bbox_px``
    attributes of the pre-descent and post-lift hypotheses, agreement with
    ``lift_prediction`` when a camera model produced one, and arm occlusion
    for an object that is no longer seen. ``scene_before`` must then be the
    observation taken *before* the descent (the object still in view), and an
    object is *visible* in a scene when its ``last_seen_step`` is that
    observation's own step, so a track the tracker merely kept alive does not
    count as seen. ``min_displacement`` only labels a non-rising object that
    slid across the table as ``knocked`` rather than ``resting``; it never
    makes a verdict positive. The stall and width fields are still filled in,
    from whatever the robot reports, so the log shows the commanded width
    alongside the verdict.
    """
    width = robot.get_gripper_width()
    fingers_stalled = width > closed_width + stall_margin

    tcp = robot.tcp_pose()
    after = scene_after.get(track_id)
    before = scene_before.get(track_id)

    object_visible = after is not None and after.last_seen_step >= scene_after.step_index
    visible_before = before is not None and before.last_seen_step >= scene_before.step_index
    object_displacement = 0.0
    if after is not None and before is not None:
        object_displacement = float(
            np.linalg.norm(after.pose.position[:2] - before.pose.position[:2])
        )
    object_displaced = object_displacement >= min_displacement

    extra: dict[str, Any] = {}
    if not gripper_feedback:
        extra = _feedbackless_fields(
            before, after, object_visible, visible_before, object_displaced, lift_prediction
        )

    if after is None:
        # The object is no longer perceived. That is itself informative: a
        # correctly grasped object is usually still visible, but the gripper can
        # occlude a small one entirely, so this is reported rather than
        # interpreted as failure here.
        _log.debug("Grasp verification: %s is no longer perceived", track_id)
        return GraspEvidence(
            fingers_stalled=fingers_stalled,
            gripper_width=width,
            object_tracked=False,
            object_to_tcp_distance=float("inf"),
            height_gain=0.0,
            expected_height_gain=expected_height_gain,
            contact_force=read_net_contact_force(robot),
            gripper_feedback=gripper_feedback,
            object_visible=False,
            object_displacement=0.0,
            object_displaced=False,
            **extra,
        )

    distance = float(np.linalg.norm(after.pose.position - tcp.position))
    height_gain = (
        float(after.pose.position[2] - before.pose.position[2]) if before is not None else 0.0
    )

    min_extent = float(after.bbox.min_extent)

    return GraspEvidence(
        fingers_stalled=fingers_stalled,
        gripper_width=width,
        object_tracked=distance <= max_object_to_tcp,
        object_to_tcp_distance=distance,
        height_gain=height_gain,
        expected_height_gain=expected_height_gain,
        contact_force=read_net_contact_force(robot),
        object_min_extent=min_extent,
        width_plausible=width >= min_extent * min_grasp_fraction,
        gripper_feedback=gripper_feedback,
        object_visible=object_visible,
        object_displacement=object_displacement,
        object_displaced=object_displaced,
        **extra,
    )


def read_net_contact_force(robot: Any) -> float | None:
    """Net contact force magnitude on the fingers, if the robot tracks it.

    Delegates to the persistent contact view the robot builds at initialisation.
    Constructing a view here instead would emit "contact forces cannot be
    retrieved ... unless the RigidPrim is initialized with
    track_contact_forces=True" on every call *and* return nothing useful, since a
    freshly created view has no contact history.

    Returns ``None`` when unavailable rather than zero: "no sensor" and "no
    contact" are different facts, and conflating them would let a missing sensor
    read as a confirmed empty gripper. This enriches the verdict; it never
    decides it.
    """
    reader = getattr(robot, "net_contact_force", None)
    if reader is None:
        return None
    try:
        return reader()
    except Exception as exc:  # pragma: no cover - depends on physics view state
        _log.debug("Net contact force unavailable: %s", exc)
        return None

"""Depthless perception: pixel boxes -> table-plane poses -> tracked scene graph.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers.

The sim lane deprojects depth and fits boxes. This lane has one fixed webcam
and no depth, so geometry comes from two priors instead:

* a **planar homography** ``H`` (``exterior_camera.homography``, row-major
  3x3) mapping a pixel on the *table plane* to table ``(x, y)`` in the robot
  base frame, measured by ``scripts/calibrate_table.py``;
* a **size table** (``hardware.object_sizes``) giving each label's
  ``(x, y, z)`` extents, which supplies the height (``z = support + size_z/2``)
  and the grasp width the depth camera would have measured.

Anchor pixel: the trap that costs centimetres
---------------------------------------------
Only points *on the table plane* map correctly through ``H``. A box's pixel
centroid is the image of a point roughly half the object's height above the
table, and for an oblique overhead camera that lands several centimetres away
from the true centre. The bottom edge of the box, by contrast, lies on the
near face's contact line with the table, which is on the plane. So the anchor
is the **bottom-centre** pixel (``pixel_anchor: bottom_center``), mapped
exactly, and the centre is recovered from it in up to two steps:

1. **First-order step (always).** Half a footprint further along the table
   direction that points *up the image* -- the bbox bottom is the face nearest
   the bottom of the image, so the centre lies the other way. That direction is
   ``-image_down`` of the configured camera orientation. It is *not* the
   horizontal ``look_at - position`` offset: for a near-nadir camera that
   offset is a few millimetres of mounting error pointing anywhere, while the
   face at the bbox bottom is fixed by the camera's roll (``up``). Measured:
   a nadir camera 5 mm off toward +Y flipped the old offset-based step from +X
   to -Y and put the bowl 103-126 mm off with an exact homography. The step
   needs only the camera's orientation to be roughly right (checkable by eye:
   which table direction is "up" in the image?).
2. **Pinhole refinement (only with ``exterior_camera.pose_measured``).** A
   short fixed-point iteration that projects the size-table box at the guessed
   centre through a pinhole model of the configured camera, reads off where
   *its* silhouette's bottom-centre would land on the table, and shifts the
   guess by the residual. With a model that matches the camera this recovers
   a 15 cm bowl to 0-1 mm where the first step alone was 3-4 cm out. With a
   *placeholder* model it is actively harmful -- the review measured a nadir
   camera 10 cm from its configured position turning an exact homography into
   a 100 mm error on every object -- so it runs only when the config says the
   pose and intrinsics were measured. A refinement that still wants to move
   the centre more than :data:`_REFINE_MAX_SHIFT_M` means the "measured" model
   disagrees with the homography; the detection is dropped with a warning
   rather than placed at either answer.

With ``pixel_anchor: center`` (a truly nadir camera) the centroid is used and
no correction is applied.

Yaw: the 0/90 degree choice is observable
-----------------------------------------
The size table gives a label's extents along its own long axis; which way that
axis lies on the table is *not* configuration. A marker lying across the ray
used to be placed 60 mm off (the step walked half its 140 mm length the wrong
way) and still passed the jaw-width check (the chord was computed for the
marker lying along X). The pixel box does distinguish the two: each
orientation predicts its own box (through the pinhole model when the pose is
measured, else the footprint mapped back through ``H``), and the observed box
matches one of them far better. The chosen orientation is reported as an
axis-aligned box -- extents along world X/Y, identity rotation -- because
every consumer downstream (grasp chord, planner obstacles,
``support_height_at``) treats the box that way. When neither orientation
explains the box well enough the hypothesis is marked ``yaw_ambiguous`` and
``Pick`` refuses it with a message, since a wrong chord closes the jaw on air
or on top of the body. "Well enough" is stricter for a long object lying
*across* the image-up direction: there each degree of misalignment moves the
bbox bottom 1.2 mm (marker), so only ~5 deg is accepted; along it, up to ~30
deg costs under 7 mm. Swept in 5 deg steps over the workspace on both
cameras, measured or not, every accepted estimate was within 11 mm of the
truth (``test_every_accepted_marker_estimate_is_within_12_mm``).

Lift prediction
---------------
:meth:`PlanarPerception.predict_lift` tells the feedback-less grasp verifier
(:func:`mfw.physics.contact.verify_grasp`) what a *carried* object should
look like: how much its box grows, where it projects, where the tool and the
gripper's footprint project. Only the growth estimate is available without a
measured pose (a nadir approximation from the configured camera height); see
the contact module for how the verdict uses each field.

Everything downstream is the sim stack unchanged: :class:`ObjectHypothesis`
-> :class:`~mfw.vision.tracking.ObjectTracker` (two-frame corroboration,
class-based re-identification after a 20 cm place) ->
:func:`~mfw.vision.scene_graph.build_scene_graph`. Freshness is judged by the
clock, exactly as :meth:`mfw.vision.manager.VisionManager.require_fresh_scene`
does, so a stale observation is re-taken rather than acted on.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mfw.config.schema import CameraConfig, HardwareConfig, PerceptionConfig
from mfw.core.errors import ConfigurationError, PerceptionError
from mfw.core.interfaces import IPerception, IPoseEstimator
from mfw.core.types import (
    BoundingBox3D,
    CameraFrame,
    Frame,
    ObjectHypothesis,
    Pose,
    SceneGraph,
)
from mfw.physics.contact import LiftPrediction
from mfw.utils.logging import EventLogger, get_logger
from mfw.vision.scene_graph import build_scene_graph
from mfw.vision.tracking import ObjectTracker

__all__ = [
    "FootprintEstimate",
    "HomographyPoseEstimator",
    "PlanarPerception",
    "PinholeModel",
    "homography_from_config",
    "ARM_FOOTPRINT_HALF_WIDTH_M",
    "ARM_FOOTPRINT_HEIGHT_M",
]

_log = get_logger("hardware.perception")

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_HORIZONTAL_EPS = 1e-6
_REFINE_ROUNDS = 3
_REFINE_MAX_SHIFT_M = 0.06
"""Largest centre correction the pinhole refinement may apply. Measured with an
exact model over the fake-lane workspace, the largest legitimate correction is
the bin's (a 10 cm tall body seen obliquely), a few centimetres; the review
measured a 10 cm placeholder-pose error being accepted under the old 0.10 m
bound. Beyond this the "measured" model disagrees with the homography."""

_ELONGATED_RATIO = 1.25
"""Footprint aspect below which the 0/90 degree choice is irrelevant."""
_YAW_MAX_SCORE = 2.5
"""Best orientation's box mismatch (sum of |log size ratios|) above which the
box is explained by neither orientation at all (wrong label or size entry)."""
_YAW_MIN_GAP_MEASURED = 0.65
_YAW_MIN_GAP_FOOTPRINT = 0.7
"""How much better the chosen orientation must explain the box than the other,
with a measured model / footprint-only. The gap falls monotonically with the
object's true misalignment from the 0/90 choice. Swept over the fake-lane and
nadir cameras for the marker and banana (5 deg steps, 20 positions): at 45 deg
off the gap is 0.0-0.61 (a diagonal marker is then placed up to 50 mm off and
meets the jaw on a 27 mm chord), at 30 deg >= 0.55 measured and >= 0.77
footprint-only (chord 19/cos(30) = 22 mm, centre 5-7 mm off: pickable). The
thresholds flag every 45 deg case, at the cost of refusing some 30-35 deg ones
on the measured lane -- the conservative side, since a refusal costs a nudge
of the object and a wrong chord costs a stalled jaw."""

_ACROSS_VIEW_MAX_SCORE_MEASURED = 0.55
_ACROSS_VIEW_MAX_SCORE_FOOTPRINT = 0.40
"""Tighter fit required when the chosen orientation puts the long axis *across*
the up-the-image direction. There a misalignment of d degrees moves the bbox
bottom by half the length times sin(d) -- 1.2 mm per degree for the marker,
along its 19 mm side -- while along the view the same misalignment costs 1-3
mm. Swept (5 deg steps, 40 poses): measured-model score 0.32-0.50 at 5 deg
(error 6-7 mm), 0.59-0.83 at 10 deg (12-13 mm); footprint-only 0.04-0.27 at
0 deg, 0.44-0.54 at 10 deg. The limits keep the centre error under about the
marker's half-width; beyond them the box is reported ``yaw_ambiguous``."""

ARM_FOOTPRINT_HALF_WIDTH_M = 0.0225
"""Half-width of the jaw's occluding volume around the TCP (the kit's 45 mm open
span). Deliberately the jaw only, not the wrist or forearm: a larger occluder
lets "hidden" count as "held" in more places, so the conservative direction is
the smaller one. ``jetson/detector_service.py`` renders the same volume in its
honest mode."""
ARM_FOOTPRINT_HEIGHT_M = 0.05
"""Height of that volume above the TCP (finger length)."""
_ARM_FOOTPRINT_BELOW_M = 0.01


def homography_from_config(values: Sequence[float]) -> NDArray[np.float64]:
    """Row-major 9 numbers -> 3x3, normalised so ``H[2, 2] == 1``.

    Raises :class:`ConfigurationError` on an empty or malformed list, naming
    the calibration script: an uncalibrated homography puts every object at a
    plausible wrong place, which is worse than a refusal.
    """
    flat = [float(v) for v in values]
    if len(flat) != 9:
        raise ConfigurationError(
            "exterior_camera.homography must be 9 numbers (row-major 3x3 pixel -> table); "
            f"got {len(flat)}. Run `py -3.12 scripts/calibrate_table.py --jetson <ip>:5560 "
            "--config configs/hardware.yaml --touch` (or `--image frame.png` and type the "
            "table XY of >= 4 clicked points) to measure it."
        )
    h = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12 or abs(np.linalg.det(h)) < 1e-15:
        raise ConfigurationError("exterior_camera.homography is singular or non-finite")
    return h / h[2, 2]


class PinholeModel:
    """Pinhole in the OpenCV convention, built from a :class:`CameraConfig`.

    Same construction as the scripted detector's ``SyntheticPinhole`` (+Z
    forward along ``look_at - position``, +X = forward x up, +Y = forward x X)
    so the two agree exactly on the fake lane. Used to predict where a box's
    silhouette sits relative to its table contact line and what a lifted
    object looks like; the homography, not this model, does the metric work.

    ``distortion`` (OpenCV ``k1, k2, p1, p2[, k3[, k4, k5, k6]]``, from
    ``scripts/calibrate_camera.py intrinsics``) is applied in :meth:`project`,
    so predicted boxes land in the same *raw* pixel space the detector boxes
    and the homography live in (the homography is fitted to raw pixels by
    both calibration scripts). Empty = ideal pinhole, which every sim and
    fake-lane camera is; the result is then bit-identical to before.
    """

    def __init__(
        self,
        fx: float, fy: float, cx: float, cy: float,
        position: Sequence[float], look_at: Sequence[float], up: Sequence[float],
        distortion: Sequence[float] = (),
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        dist = [float(v) for v in distortion]
        if len(dist) not in CameraConfig.DISTORTION_LENGTHS:
            raise ConfigurationError(
                f"distortion must have {list(CameraConfig.DISTORTION_LENGTHS)} coefficients, got {len(dist)}"
            )
        self.distortion = np.zeros(8) if not dist else np.asarray(dist + [0.0] * (8 - len(dist)), dtype=np.float64)
        self.has_distortion = bool(np.any(self.distortion != 0.0))
        self.position = np.asarray(position, dtype=np.float64)
        forward = np.asarray(look_at, dtype=np.float64) - self.position
        norm = float(np.linalg.norm(forward))
        if norm < 1e-9:
            raise ConfigurationError("camera position and look_at coincide")
        z_axis = forward / norm
        x_axis = np.cross(z_axis, np.asarray(up, dtype=np.float64))
        if float(np.linalg.norm(x_axis)) < 1e-9:
            x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        self.rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

    @classmethod
    def from_config(cls, camera: CameraConfig) -> "PinholeModel | None":
        """``None`` unless the config carries pixel intrinsics and a look-at pose.

        This does not ask whether those values were *measured*; callers that
        use the model metrically gate on ``camera.pose_measured`` themselves.
        """
        if camera.fx <= 0.0 or camera.look_at is None:
            return None
        fx, fy, cx, cy = camera.pixel_intrinsics()
        return cls(fx, fy, cx, cy, camera.position, camera.look_at, camera.up, camera.distortion)

    @classmethod
    def orientation_from_config(cls, camera: CameraConfig) -> "PinholeModel | None":
        """A unit-focal model carrying only the configured orientation, or ``None``.

        Enough to ask which table direction is "down the image"; nothing
        metric may be read from it.
        """
        if camera.look_at is None:
            return None
        return cls(1.0, 1.0, 0.0, 0.0, camera.position, camera.look_at, camera.up)

    @property
    def image_down_on_table(self) -> NDArray[np.float64]:
        """Where 'down the image' points on the table (the camera +Y axis' footprint)."""
        d = self.rotation[:2, 1].copy()
        norm = float(np.linalg.norm(d))
        return d / norm if norm > _HORIZONTAL_EPS else np.zeros(2)

    def project(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """World (N, 3) -> raw pixel (N, 2); NaN for points behind the camera.

        With distortion, also NaN where the lens polynomial has folded over
        (far outside the calibrated field of view, where a point would
        otherwise wrap back into the image at a meaningless place).
        """
        cam = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.position) @ self.rotation
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = cam[:, 0] / z
            y = cam[:, 1] / z
            folded = np.zeros(len(z), dtype=bool)
            if self.has_distortion:
                x, y, folded = self._distort(x, y)
            u = self.fx * x + self.cx
            v = self.fy * y + self.cy
        uv = np.stack([u, v], axis=1)
        uv[(z <= 1e-9) | folded] = np.nan
        return uv

    def _radial(self, r2: NDArray[np.float64]) -> NDArray[np.float64]:
        k1, k2, _p1, _p2, k3, k4, k5, k6 = self.distortion
        return (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))

    def _distort(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """OpenCV's forward lens model on normalised coordinates (same as ``cv2.projectPoints``)."""
        _k1, _k2, p1, p2 = self.distortion[:4]
        r2 = x * x + y * y
        radial = self._radial(r2)
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        # Fold-over: the distorted radius r * radial(r) must still grow with r.
        r = np.sqrt(r2)
        r_next = r * 1.001 + 1e-9
        folded = ~(r_next * self._radial(r_next * r_next) > r * radial) | ~np.isfinite(radial)
        return xd, yd, folded

    def project_box(
        self, centre: Sequence[float], extents: Sequence[float]
    ) -> tuple[float, float, float, float] | None:
        """Pixel bbox of an axis-aligned 3-D box centred at ``centre``; ``None`` if not in front."""
        c = np.asarray(centre, dtype=np.float64)
        half = np.asarray(extents, dtype=np.float64) / 2.0
        corners = np.array(
            [c + np.array([dx, dy, dz]) * half for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
        )
        uv = self.project(corners)
        if not np.all(np.isfinite(uv)):
            return None
        return (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))


@dataclass(frozen=True)
class FootprintEstimate:
    """One detection lifted onto the table, with the orientation evidence."""

    pose: Pose
    extents: NDArray[np.float64]
    """World-axis extents: the size-table entry, swapped when ``yaw_rad`` is 90 deg."""
    yaw_rad: float
    """0 (long axis along world X, as the size table is written) or pi/2."""
    yaw_ambiguous: bool
    """Elongated, and neither orientation explains the pixel box."""
    yaw_scores: tuple[float, float] | None
    """Box mismatch for yaw 0 and yaw 90 (``None``: footprint is square-ish)."""


class HomographyPoseEstimator(IPoseEstimator):
    """Pixel box -> ``(Pose, extents)`` on the table plane.

    ``camera`` decides what the bottom-centre correction may use:

    * a :class:`CameraConfig` -- its orientation drives the first-order step;
      its pinhole model drives the refinement and lift prediction **only if**
      ``pose_measured`` is true;
    * a :class:`PinholeModel` -- taken as measured (tests and scripts that
      built one from a known camera);
    * ``None`` -- only valid with ``pixel_anchor: center`` or an explicit
      ``view_direction`` (the table direction pointing up the image). Without
      either, a bottom-centre anchor would silently return the near face as
      the centre (half a footprint off), so construction is refused.
    """

    def __init__(
        self,
        homography: Sequence[float] | NDArray[np.float64],
        object_sizes: Mapping[str, Sequence[float]],
        default_size: Sequence[float],
        support_height: float = 0.0,
        pixel_anchor: str = "bottom_center",
        camera: CameraConfig | PinholeModel | None = None,
        view_direction: Sequence[float] | None = None,
    ) -> None:
        if pixel_anchor not in HardwareConfig.PIXEL_ANCHORS:
            raise ConfigurationError(
                f"pixel_anchor must be one of {list(HardwareConfig.PIXEL_ANCHORS)}, got {pixel_anchor!r}"
            )
        self.h = homography_from_config(np.asarray(homography, dtype=np.float64).reshape(-1))
        self.h_inv = np.linalg.inv(self.h)
        self.object_sizes = {
            str(k).strip().lower(): np.asarray(v, dtype=np.float64) for k, v in object_sizes.items()
        }
        self.default_size = np.asarray(default_size, dtype=np.float64)
        self.support_height = float(support_height)
        self.pixel_anchor = pixel_anchor
        self.nominal_camera_height: float | None = None
        """Configured camera height above the support (placeholder or not);
        feeds only the nadir growth approximation."""

        if isinstance(camera, CameraConfig):
            self.pose_measured = bool(camera.pose_measured)
            self.camera: PinholeModel | None = PinholeModel.from_config(camera) if self.pose_measured else None
            self._view_dir = self._view_direction_from_config(camera)
            self.nominal_camera_height = float(camera.position[2]) - self.support_height
        elif isinstance(camera, PinholeModel):
            self.pose_measured = True
            self.camera = camera
            self._view_dir = -camera.image_down_on_table
            self.nominal_camera_height = float(camera.position[2]) - self.support_height
        else:
            self.pose_measured = False
            self.camera = None
            self._view_dir = np.zeros(2)
        if view_direction is not None:
            d = np.asarray(view_direction, dtype=np.float64).reshape(-1)[:2]
            norm = float(np.linalg.norm(d))
            if norm < _HORIZONTAL_EPS:
                raise ConfigurationError("view_direction must be a non-zero table direction")
            self._view_dir = d / norm

        if self.pixel_anchor == "bottom_center":
            if float(np.linalg.norm(self._view_dir)) < _HORIZONTAL_EPS:
                raise ConfigurationError(
                    "pixel_anchor bottom_center needs the camera orientation (a CameraConfig with "
                    "look_at/up, a PinholeModel, or view_direction) to know which face the bbox "
                    "bottom is; without it every centre would be the near face, half a footprint off"
                )
            if self.camera is None:
                _log.warning(
                    "exterior_camera.pose_measured is false: the centre is the bbox bottom-centre "
                    "mapped through the homography plus half a footprint along %s (the table "
                    "direction up the image); the pinhole refinement and the lift prediction are "
                    "off. Expect up to a few cm of error on tall objects seen obliquely; measure "
                    "fx/fy/cx/cy, position and look_at and set pose_measured: true to enable them.",
                    np.round(self._view_dir, 3).tolist(),
                )

    @staticmethod
    def _view_direction_from_config(camera: CameraConfig) -> NDArray[np.float64]:
        """Table direction pointing *up the image*: from the bbox-bottom face toward the centre.

        Taken from the configured orientation as ``-image_down`` projected on
        the table, never from the horizontal ``look_at - position`` offset: for
        a near-nadir camera that offset is mounting error and points anywhere,
        while the face at the bbox bottom is fixed by the roll (``up``). For an
        oblique camera the two agree. Without ``look_at`` the horizontal part
        of ``up`` (the image-up direction of a nadir camera) is used.
        """
        model = PinholeModel.orientation_from_config(camera)
        if model is not None:
            d = -model.image_down_on_table
            if float(np.linalg.norm(d)) > _HORIZONTAL_EPS:
                return d
        up = np.asarray(camera.up, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(up))
        return up / norm if norm > _HORIZONTAL_EPS else np.zeros(2)

    # ------------------------------------------------------------------

    @property
    def view_direction(self) -> NDArray[np.float64]:
        """Unit table direction of the first-order step (copy)."""
        return self._view_dir.copy()

    def size_for(self, label: str) -> NDArray[np.float64]:
        """Configured ``(x, y, z)`` extents for ``label``, or the default."""
        return self.object_sizes.get(label.strip().lower(), self.default_size).copy()

    def pixel_to_table(self, u: float, v: float) -> NDArray[np.float64]:
        """One pixel on the table plane -> ``(x, y)``."""
        p = self.h @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-12:
            raise PerceptionError(f"pixel ({u}, {v}) maps to infinity through the homography")
        return p[:2] / p[2]

    def table_to_pixel(self, points_xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Table ``(N, 2)`` -> pixel ``(N, 2)`` through ``H^-1``; NaN where it diverges."""
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ self.h_inv.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = hom[:, :2] / hom[:, 2:3]
        uv[np.abs(hom[:, 2]) < 1e-12] = np.nan
        return uv

    def anchor_pixel(self, bbox_px: Sequence[float]) -> tuple[float, float]:
        """The pixel the homography is applied to, per ``pixel_anchor``."""
        x0, y0, x1, y1 = (float(v) for v in bbox_px)
        u = 0.5 * (x0 + x1)
        if self.pixel_anchor == "center":
            return u, 0.5 * (y0 + y1)
        return u, y1

    def _first_guess(
        self, anchor_xy: NDArray[np.float64], half_extents: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Step from the near-face contact point to the footprint centre.

        Half an extent along the up-the-image direction, split over the box's
        world X and Y axes by the squared cosines of that direction. The
        extents are world-axis extents, i.e. already swapped for the observed
        orientation (see :meth:`estimate_footprint`).
        """
        if self.pixel_anchor != "bottom_center":
            return anchor_xy
        d = self._view_dir
        weights = d * d  # squared cosines; sum to 1
        return anchor_xy + np.sign(d) * weights * half_extents[:2]

    def _predicted_anchor_xy(
        self, centre_xy: NDArray[np.float64], size: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Where this object's silhouette anchor would map to, if it sat at ``centre_xy``."""
        assert self.camera is not None
        box = self.camera.project_box(
            (centre_xy[0], centre_xy[1], self.support_height + float(size[2]) / 2.0), size
        )
        if box is None:
            return None
        u, v = self.anchor_pixel(box)
        return self.pixel_to_table(u, v)

    def _refine(
        self, centre_xy: NDArray[np.float64], anchor_xy: NDArray[np.float64], size: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fixed-point correction of the centre through the measured camera model.

        Shifts the guess by (observed anchor - predicted anchor) on the table
        until the box placed at the guess would produce the anchor that was
        seen. Contracts quickly because the map is nearly a translation.
        Raises :class:`PerceptionError` when the correction exceeds
        :data:`_REFINE_MAX_SHIFT_M`: the "measured" model and the homography
        disagree, and neither answer should be acted on.
        """
        if self.camera is None:
            return centre_xy
        centre = centre_xy.copy()
        for _ in range(_REFINE_ROUNDS):
            predicted = self._predicted_anchor_xy(centre, size)
            if predicted is None:
                return centre_xy
            residual = anchor_xy - predicted
            if float(np.linalg.norm(residual)) < 1e-4:
                break
            centre = centre + residual
        shift = float(np.linalg.norm(centre - centre_xy))
        if shift > _REFINE_MAX_SHIFT_M:
            raise PerceptionError(
                f"the camera model wants to move this object {shift * 1000.0:.0f} mm from the "
                f"first-order estimate (limit {_REFINE_MAX_SHIFT_M * 1000.0:.0f} mm): "
                "exterior_camera position/look_at/fx disagree with the homography. Re-measure "
                "them, or set exterior_camera.pose_measured: false"
            )
        return centre

    def _centre_for(
        self, anchor_xy: NDArray[np.float64], extents: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        centre = self._first_guess(anchor_xy, extents / 2.0)
        if self.pixel_anchor == "bottom_center":
            centre = self._refine(centre, anchor_xy, extents)
        return centre

    def _predicted_box(
        self, centre_xy: NDArray[np.float64], extents: NDArray[np.float64]
    ) -> tuple[float, float, float, float] | None:
        """The pixel box this orientation should produce at ``centre_xy``.

        With a measured model: the full 3-D box projected. Without: only the
        footprint, mapped back through ``H^-1`` (exact for the table plane,
        but it cannot know how far the top face extends the box).
        """
        if self.camera is not None:
            return self.camera.project_box(
                (centre_xy[0], centre_xy[1], self.support_height + float(extents[2]) / 2.0), extents
            )
        hx, hy = float(extents[0]) / 2.0, float(extents[1]) / 2.0
        corners = np.array([[centre_xy[0] + dx, centre_xy[1] + dy] for dx in (-hx, hx) for dy in (-hy, hy)])
        uv = self.table_to_pixel(corners)
        if not np.all(np.isfinite(uv)):
            return None
        return (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))

    def _box_mismatch(
        self, predicted: tuple[float, float, float, float] | None, observed: Sequence[float]
    ) -> float:
        """How badly a predicted box explains the observed one (0 = exactly).

        Sum over width and height of |log(predicted / observed)|. Footprint-only
        predictions (no measured model) must fit *inside* the observed box,
        which also contains the top face: a footprint wider than the box is
        penalised double, one narrower than it only half.
        """
        if predicted is None:
            return math.inf
        pw, ph = predicted[2] - predicted[0], predicted[3] - predicted[1]
        ow, oh = float(observed[2]) - float(observed[0]), float(observed[3]) - float(observed[1])
        if min(pw, ph, ow, oh) <= 0.0:
            return math.inf
        score = 0.0
        for p, o in ((pw, ow), (ph, oh)):
            r = math.log(p / o)
            if self.camera is not None:
                score += abs(r)
            else:
                score += 2.0 * r if r > 0.0 else -0.5 * r
        return score

    def estimate_footprint(self, detection: Mapping[str, Any]) -> FootprintEstimate | None:
        """Pose, world-axis extents and orientation evidence for one wire detection.

        ``None`` when the box is unusable; :class:`PerceptionError` when the
        measured camera model contradicts the homography.
        """
        box = detection.get("bbox_px")
        if box is None or len(box) != 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in box)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        label = str(detection.get("label", ""))
        size = self.size_for(label)
        u, v = self.anchor_pixel(box)
        anchor_xy = self.pixel_to_table(u, v)

        long_ratio = max(size[0], size[1]) / max(min(size[0], size[1]), 1e-9)
        if self.pixel_anchor != "bottom_center" or long_ratio < _ELONGATED_RATIO:
            centre = self._centre_for(anchor_xy, size)
            yaw, ambiguous, scores, extents = 0.0, False, None, size
        else:
            options: list[tuple[float, NDArray[np.float64], NDArray[np.float64] | None]] = []
            for extents_try in (size, size[[1, 0, 2]]):
                try:
                    centre_try: NDArray[np.float64] | None = self._centre_for(anchor_xy, extents_try)
                    score = self._box_mismatch(self._predicted_box(centre_try, extents_try), (x0, y0, x1, y1))
                except PerceptionError:
                    centre_try, score = None, math.inf
                options.append((score, extents_try, centre_try))
            scores = (float(options[0][0]), float(options[1][0]))
            best = 0 if scores[0] <= scores[1] else 1
            if options[best][2] is None:
                # Both orientations need an implausible correction: surface it.
                self._centre_for(anchor_xy, size)
            yaw = 0.0 if best == 0 else math.pi / 2.0
            extents = options[best][1]
            centre = options[best][2]
            assert centre is not None
            gap = abs(scores[0] - scores[1])
            min_gap = _YAW_MIN_GAP_MEASURED if self.camera is not None else _YAW_MIN_GAP_FOOTPRINT
            max_score = _YAW_MAX_SCORE
            long_axis = np.array([1.0, 0.0]) if extents[0] >= extents[1] else np.array([0.0, 1.0])
            if abs(float(long_axis @ self._view_dir)) < math.sqrt(0.5):
                max_score = (
                    _ACROSS_VIEW_MAX_SCORE_MEASURED if self.camera is not None else _ACROSS_VIEW_MAX_SCORE_FOOTPRINT
                )
            ambiguous = bool(scores[best] > max_score or gap < min_gap)
            if ambiguous:
                _log.info(
                    "%s: pixel box %s fits neither orientation well (mismatch yaw0=%.2f yaw90=%.2f)",
                    label, [round(x0), round(y0), round(x1), round(y1)], scores[0], scores[1],
                )
        position = np.array([centre[0], centre[1], self.support_height + float(extents[2]) / 2.0])
        return FootprintEstimate(
            pose=Pose(position, _IDENTITY_QUAT.copy(), Frame.WORLD),
            extents=np.asarray(extents, dtype=np.float64).copy(),
            yaw_rad=float(yaw),
            yaw_ambiguous=ambiguous,
            yaw_scores=scores,
        )

    def estimate(
        self, frame: CameraFrame | None, detection: Mapping[str, Any]
    ) -> tuple[Pose, NDArray[np.float64]] | None:
        """``(pose, extents)`` for one wire detection, or ``None`` if unusable.

        ``frame`` is accepted for interface symmetry and unused: the geometry
        comes from the homography, not from pixels. Extents are world-axis
        (see :meth:`estimate_footprint`).
        """
        result = self.estimate_footprint(detection)
        return None if result is None else (result.pose, result.extents)

    def xy_for_box(self, bbox_px: Sequence[float], extents: Sequence[float]) -> NDArray[np.float64]:
        """Table ``(x, y)`` the estimator produces for ``bbox_px`` with known world-axis extents."""
        u, v = self.anchor_pixel(bbox_px)
        return self._centre_for(self.pixel_to_table(u, v), np.asarray(extents, dtype=np.float64))


class PlanarPerception(IPerception):
    """``observe()`` for a fixed overhead camera and a remote detector."""

    def __init__(
        self,
        clock: Any,
        detector: Any,
        config: PerceptionConfig,
        hardware: HardwareConfig,
        homography: Sequence[float],
        support_height: float = 0.0,
        workspace_min: Sequence[float] | None = None,
        workspace_max: Sequence[float] | None = None,
        camera: CameraConfig | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        config.validate()
        hardware.validate()
        self.config = config
        self.hardware = hardware
        self._clock = clock
        self._detector = detector
        self._events = event_logger
        self._support = float(support_height)
        self.estimator = HomographyPoseEstimator(
            homography=homography,
            object_sizes=hardware.object_sizes,
            default_size=hardware.default_object_size,
            support_height=support_height,
            pixel_anchor=hardware.pixel_anchor,
            camera=camera,
        )
        self._vocabulary = {str(l).strip().lower() for l in hardware.labels}
        self._ws_min = None if workspace_min is None else np.asarray(workspace_min, dtype=np.float64)[:2]
        self._ws_max = None if workspace_max is None else np.asarray(workspace_max, dtype=np.float64)[:2]
        self._tracker = ObjectTracker(
            match_distance=config.track_match_distance,
            max_age_steps=config.track_max_age_steps,
        )
        self._last_scene: SceneGraph | None = None
        self._out_of_reach: list[dict[str, Any]] = []

    #: A detection further than this from the robot base is background (a
    #: marker on the next desk), not an object the operator put down for the
    #: arm; nearer ones outside the workspace are reported as out of reach.
    OUT_OF_REACH_REPORT_M = 0.6

    @property
    def last_out_of_reach(self) -> list[dict[str, Any]]:
        """Objects the latest observe saw but dropped as outside the workspace.

        ``[{"label", "position": [x, y], "confidence", "distance_m"}]``. Skills
        use it to say "the cube is out of reach" instead of "there is no cube
        in view" (audit: the old wording told the operator to look for an
        object that was plainly on the table).
        """
        return [dict(entry) for entry in self._out_of_reach]

    # ------------------------------------------------------------------
    # IPerception

    def observe(self) -> SceneGraph:
        """Ask the detector, lift every box onto the table, track, build the graph."""
        started = time.perf_counter()
        try:
            detections = self._detector.detect(None)
        except PerceptionError:
            raise
        except Exception as exc:  # noqa: BLE001 - the detector is a network peer
            raise PerceptionError(f"detector failed: {type(exc).__name__}: {exc}") from exc

        hypotheses = self._hypotheses(detections)
        objects = self._tracker.update(hypotheses, int(self._clock.step_index))
        scene = build_scene_graph(
            objects=objects,
            sim_time=float(self._clock.sim_time),
            step_index=int(self._clock.step_index),
        )
        self._last_scene = scene
        if self._events is not None:
            self._events.emit(
                "perception.observe",
                {
                    "duration_s": time.perf_counter() - started,
                    "cameras": [self.hardware_camera_name],
                    "raw_detections": len(detections),
                    "detections": len(hypotheses),
                    "out_of_reach": self.last_out_of_reach,
                    "scene": scene.to_log(),
                },
            )
        _log.debug(
            "Observed %d object(s) from %d detection(s) in %.1f ms",
            len(objects), len(detections), (time.perf_counter() - started) * 1000.0,
        )
        return scene

    hardware_camera_name = "exterior_camera"

    def last_scene_graph(self) -> SceneGraph | None:
        return self._last_scene

    def require_fresh_scene(self) -> SceneGraph:
        """Re-observe when the cached graph is older than ``max_scene_graph_age_s``."""
        scene = self._last_scene
        if scene is None:
            return self.observe()
        age = float(self._clock.sim_time) - scene.sim_time
        if age > self.config.max_scene_graph_age_s:
            _log.debug("Scene graph is %.2fs old; re-observing", age)
            return self.observe()
        return scene

    def support_height_at(self, position: ArrayLike, default: float) -> float:
        """Top of the tallest perceived object under ``position``, else ``default``."""
        scene = self._last_scene
        if scene is None:
            return default
        best = default
        p = np.asarray(position, dtype=np.float64)[:2]
        for obj in scene.objects.values():
            half = obj.bbox.extents[:2] / 2.0
            if np.all(np.abs(obj.bbox.center.position[:2] - p) <= half):
                best = max(best, float(obj.bbox.center.position[2] + obj.bbox.extents[2] / 2.0))
        return best

    # ------------------------------------------------------------------
    # grasp verification support

    @property
    def pose_measured(self) -> bool:
        """Whether the camera model may be used metrically (``exterior_camera.pose_measured``)."""
        return self.estimator.camera is not None

    def predict_lift(self, target: ObjectHypothesis, tcp: Pose | ArrayLike) -> LiftPrediction:
        """What ``target`` should look like if it is now held at ``tcp``.

        The carried object's centre is taken to be the TCP (the jaw closes on
        the middle of it). With a measured camera every field is filled from
        the pinhole model: the expected bbox growth (box at rest vs box at the
        TCP), the carried box and its table-plane estimate through this very
        estimator, the TCP pixel, and the gripper's occluding footprint.
        Without one only the growth is predicted, as a nadir approximation
        ``d / (d - rise)`` from the configured camera height -- the verifier
        floors the required growth anyway, so a placeholder height only
        changes the threshold, never the direction of the verdict.
        """
        tcp_p = np.asarray(tcp.position if isinstance(tcp, Pose) else tcp, dtype=np.float64).reshape(3)
        rest = np.asarray(target.pose.position, dtype=np.float64)
        extents = np.asarray(target.bbox.extents, dtype=np.float64)
        rise = float(tcp_p[2] - rest[2])
        model = self.estimator.camera
        if model is None:
            expected = None
            height = self.estimator.nominal_camera_height
            if height is not None:
                d_rest = height - (rest[2] - self._support)
                if rise > 0.0 and d_rest - rise > 0.05:
                    expected = d_rest / (d_rest - rise)
            return LiftPrediction(expected_pixel_scale=expected, model_available=False)

        rest_box = model.project_box(rest, extents)
        carried_box = model.project_box(tcp_p, extents)
        expected = None
        predicted_xy = None
        if rest_box is not None and carried_box is not None:
            d_rest = math.hypot(rest_box[2] - rest_box[0], rest_box[3] - rest_box[1])
            d_carried = math.hypot(carried_box[2] - carried_box[0], carried_box[3] - carried_box[1])
            expected = d_carried / max(d_rest, 1e-9)
            try:
                xy = self.estimator.xy_for_box(carried_box, extents)
                predicted_xy = (float(xy[0]), float(xy[1]))
            except PerceptionError as exc:
                _log.debug("no table-plane prediction for the carried box: %s", exc)
        tcp_uv = model.project(tcp_p.reshape(1, 3))[0]
        tcp_px = (float(tcp_uv[0]), float(tcp_uv[1])) if np.all(np.isfinite(tcp_uv)) else None
        arm_centre = tcp_p + np.array([0.0, 0.0, (ARM_FOOTPRINT_HEIGHT_M - _ARM_FOOTPRINT_BELOW_M) / 2.0])
        arm_extents = np.array(
            [2.0 * ARM_FOOTPRINT_HALF_WIDTH_M, 2.0 * ARM_FOOTPRINT_HALF_WIDTH_M,
             ARM_FOOTPRINT_HEIGHT_M + _ARM_FOOTPRINT_BELOW_M]
        )
        return LiftPrediction(
            expected_pixel_scale=expected,
            predicted_xy=predicted_xy,
            predicted_bbox_px=carried_box,
            tcp_px=tcp_px,
            arm_bbox_px=model.project_box(arm_centre, arm_extents),
            model_available=True,
            rest_bbox_px=rest_box,
        )

    # ------------------------------------------------------------------
    # internals

    def _in_workspace(self, xy: NDArray[np.float64]) -> bool:
        if self._ws_min is None or self._ws_max is None:
            return True
        margin = float(self.hardware.workspace_margin_xy)
        return bool(np.all(xy >= self._ws_min - margin) and np.all(xy <= self._ws_max + margin))

    def _hypotheses(self, detections: Sequence[Mapping[str, Any]]) -> list[ObjectHypothesis]:
        out: list[ObjectHypothesis] = []
        out_of_reach: list[dict[str, Any]] = []
        self._out_of_reach = out_of_reach
        now, step = float(self._clock.sim_time), int(self._clock.step_index)
        for det in detections:
            label = str(det.get("label", "")).strip().lower()
            if not label or (self._vocabulary and label not in self._vocabulary):
                continue
            confidence = float(det.get("confidence", 0.0))
            if confidence < self.hardware.detection_min_score or confidence < self.config.min_confidence:
                continue
            try:
                estimate = self.estimator.estimate_footprint(det)
            except PerceptionError as exc:
                _log.warning("dropping %s: %s", label, exc)
                continue
            if estimate is None:
                continue
            pose, extents = estimate.pose, estimate.extents
            if not self._in_workspace(pose.position[:2]):
                xy = np.asarray(pose.position[:2], dtype=np.float64)
                distance = float(np.linalg.norm(xy))
                if distance <= self.OUT_OF_REACH_REPORT_M:
                    _log.debug("%s at %s is out of reach (outside the workspace); not tracked",
                               label, np.round(pose.position, 3))
                    out_of_reach.append({
                        "label": label,
                        "position": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
                        "confidence": round(confidence, 3),
                        "distance_m": round(distance, 4),
                    })
                else:
                    _log.debug("dropping %s at %s: %.2f m from the base, background",
                               label, np.round(pose.position, 3), distance)
                continue
            x0, y0, x1, y1 = (float(v) for v in det["bbox_px"])
            out.append(
                ObjectHypothesis(
                    track_id="",
                    label=label,
                    pose=pose,
                    bbox=BoundingBox3D(center=pose, extents=extents),
                    confidence=confidence,
                    num_points=int(max(1.0, (x1 - x0) * (y1 - y0))),
                    last_seen_sim_time=now,
                    last_seen_step=step,
                    seg_id=None,
                    attributes={
                        "bbox_px": [x0, y0, x1, y1],
                        "anchor_px": list(self.estimator.anchor_pixel(det["bbox_px"])),
                        "yaw_rad": estimate.yaw_rad,
                        "yaw_ambiguous": estimate.yaw_ambiguous,
                        "cameras": [self.hardware_camera_name],
                        "primary_camera": self.hardware_camera_name,
                        "color": "",
                    },
                )
            )
        return out

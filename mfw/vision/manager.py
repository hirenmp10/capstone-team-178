"""VisionManager -- the only component permitted to touch cameras.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

Pipeline per observation:

    cameras -> RGB/depth/instance-seg -> deproject -> per-instance clouds
            -> filter -> 6-DoF pose + OBB -> track association -> SceneGraph

Design commitments:

* **Multi-camera fusion by instance id.** Points from the wrist and exterior
  views are merged per segmentation instance before pose fitting, so an object
  half-occluded in one view is still measured from the other. Fusing before
  fitting (rather than fitting per view and averaging poses) avoids averaging
  two biased partial-view estimates into a third wrong one.
* **No ground truth anywhere.** Nothing here can read a spawn pose; there is no
  accessor that returns one.
* **Detection is segmentation-driven, not a learned detector.** Isaac's instance
  annotator already provides exact masks and per-instance identity. Bolting a
  bounding-box detector on top would add latency and error to information the
  renderer supplies perfectly.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import PerceptionConfig
from mfw.core.errors import PerceptionError
from mfw.core.interfaces import ICamera, IPerception
from mfw.core.types import (
    BoundingBox3D,
    CameraFrame,
    Frame,
    ObjectHypothesis,
    Pose,
    SceneGraph,
)
from mfw.utils import transforms as tf
from mfw.utils.logging import EventLogger, get_logger
from mfw.vision.camera import deproject_to_point_cloud
from mfw.vision.geometry import (
    UNLIT_FRAME_VALUE,
    dominant_color_name,
    fit_oriented_bbox,
    frame_brightness,
    is_unlit_frame,
    remove_statistical_outliers,
    voxel_downsample,
)
from mfw.vision.scene_graph import build_scene_graph
from mfw.vision.tracking import ObjectTracker

__all__ = ["VisionManager", "object_color_signature"]

_log = get_logger("vision.manager")

# Segmentation entries that are never objects.
_NON_OBJECT_LABELS = {"BACKGROUND", "UNLABELLED", "UNLABELED", ""}
# Prims that are scene furniture rather than manipulable objects.
_SUPPORT_LABELS = {"table", "ground", "groundplane", "defaultgroundplane"}
# Every manipulable object is spawned under this scope.
_OBJECT_SCOPE = "/World/objects/"


def object_color_signature(frame: CameraFrame) -> dict[str, NDArray[np.float64]] | None:
    """Mean RGB of each object's pixels in one frame, keyed by prim path.

    What start-up warm-up polls to decide that object colours have stopped
    changing. Per object rather than per frame, because the objects are a few
    percent of the image: a block going from unrendered black to red moves the
    frame mean by a couple of levels, which is inside render noise, while its
    own mean moves by a hundred.

    Returns ``None`` for a frame with no usable colour (unlit, or no RGB), so
    such a frame can never count as settled. A frame with no object pixels (no
    segmentation, or nothing in view) falls back to the whole-frame mean under
    the key ``""``.
    """
    rgb = frame.rgb
    if rgb is None or np.asarray(rgb).size == 0 or is_unlit_frame(rgb):
        return None
    pixels = np.asarray(rgb)[..., :3].astype(np.float64)

    signature: dict[str, NDArray[np.float64]] = {}
    seg = frame.segmentation
    if seg is not None and np.asarray(seg).shape[:2] == pixels.shape[:2]:
        seg = np.asarray(seg)
        for seg_id, prim in frame.seg_id_to_prim.items():
            if not str(prim).startswith(_OBJECT_SCOPE):
                continue
            mask = seg == int(seg_id)
            if np.any(mask):
                signature[str(prim)] = pixels[mask].mean(axis=0)
    if not signature:
        signature[""] = pixels.reshape(-1, 3).mean(axis=0)
    return signature


class VisionManager(IPerception):
    """Turns camera frames into a tracked, structured scene graph."""

    def __init__(
        self,
        sim: Any,
        cameras: dict[str, ICamera],
        config: PerceptionConfig,
        event_logger: EventLogger | None = None,
        support_height: float | None = None,
    ) -> None:
        config.validate()
        if not cameras:
            raise PerceptionError("VisionManager requires at least one camera")
        self.config = config
        self._sim = sim
        self._cameras = dict(cameras)
        self._events = event_logger
        self._tracker = ObjectTracker(
            match_distance=config.track_match_distance,
            max_age_steps=config.track_max_age_steps,
        )
        self._last_scene: SceneGraph | None = None
        # Per-camera frame brightness from the latest capture; see _capture_all.
        self._frame_brightness: dict[str, float] = {}
        # Height of the surface objects rest on. Defaults to the configured
        # ground plane, which is right only when they are on the floor.
        self._support_height = (
            float(support_height) if support_height is not None else float(config.ground_plane_z)
        )

    # ------------------------------------------------------------------
    # IPerception
    # ------------------------------------------------------------------

    def observe(self) -> SceneGraph:
        """Acquire fresh sensor data and return an updated scene graph.

        Every skill calls this before acting. Nothing in the framework may act
        on a scene graph it did not just request.
        """
        started = time.perf_counter()
        frames = self._capture_all()
        instance_clouds = self._accumulate_instances(frames)
        detections = self._fit_hypotheses(instance_clouds)

        objects = self._tracker.update(detections, self._sim.step_index)
        scene = build_scene_graph(
            objects=objects,
            sim_time=self._sim.sim_time,
            step_index=self._sim.step_index,
        )
        self._last_scene = scene

        if self._events is not None:
            self._events.emit(
                "perception.observe",
                {
                    "duration_s": time.perf_counter() - started,
                    "cameras": sorted(frames),
                    "raw_instances": len(instance_clouds),
                    "detections": len(detections),
                    # Evidence for the colour path: a camera listed as unlit
                    # contributed geometry but no colour to this observation.
                    "rgb_brightness": {
                        name: round(value, 1) for name, value in self._frame_brightness.items()
                    },
                    "unlit_cameras": self._unlit_cameras(),
                    "scene": scene.to_log(),
                },
            )
        _log.debug(
            "Observed %d objects from %d cameras in %.1f ms",
            len(objects),
            len(frames),
            (time.perf_counter() - started) * 1000.0,
        )
        return scene

    def last_scene_graph(self) -> SceneGraph | None:
        return self._last_scene

    def colors_settled(self) -> bool:
        """Whether every confirmed object reports a colour the latest frame agreed with.

        Start-up priming polls this so the operator's first "what do you see"
        is answered from colours that have been measured, not from a frame the
        renderer had not finished. See :meth:`ObjectTracker.colors_settled`.
        """
        return self._tracker.colors_settled()

    def require_fresh_scene(self) -> SceneGraph:
        """Return a scene graph, re-observing if the cached one is stale.

        Staleness is the quiet killer in manipulation: acting on a
        second-old pose sends the gripper to where the object *was*. Skills call
        this rather than trusting :meth:`last_scene_graph`.
        """
        scene = self._last_scene
        if scene is None:
            return self.observe()
        age = self._sim.sim_time - scene.sim_time
        if age > self.config.max_scene_graph_age_s:
            _log.debug("Scene graph is %.2fs old; re-observing", age)
            return self.observe()
        return scene

    # ------------------------------------------------------------------
    # pipeline stages
    # ------------------------------------------------------------------

    def _capture_all(self) -> dict[str, CameraFrame]:
        """Capture every camera. A single failing sensor must not blind the robot.

        Renders first. ``SimulationContext.step`` only renders when the app is
        windowed, and physics-only stepping is what every settle loop does, so a
        capture that follows one reads a render product that was never refreshed.

        The failure is asymmetric and that is what makes it nasty: depth and
        instance segmentation still come back correct, because they are derived
        from geometry and prim identity rather than from accumulated samples.
        Only **RGB** goes black. Detection, tracking, poses and bounding boxes
        are therefore all fine, and the sole visible symptom is that every
        object's colour attribute becomes "black" -- so ``the red mug`` stops
        resolving while ``the mug`` still works.

        Measured on this scene: identical code paths returned a mug median of
        [16, 4, 3] with a render before capture and [1, 1, 1] without.
        """
        if self.config.render_frames_before_capture > 0:
            self._sim.render_step(self.config.render_frames_before_capture)

        frames: dict[str, CameraFrame] = {}
        for name, camera in self._cameras.items():
            try:
                frames[name] = camera.capture()
            except PerceptionError as exc:
                _log.warning("Camera %s failed to capture: %s", name, exc)
        if not frames:
            raise PerceptionError("No camera produced a frame; perception is blind")

        # Measured once here and consulted by _accumulate_instances. A camera
        # whose RGB is unlit still contributes geometry -- depth and
        # segmentation do not depend on the colour buffer -- but its pixels
        # must not vote on colour, or an unwritten buffer names every object
        # it sees "black".
        self._frame_brightness = {
            name: frame_brightness(frame.rgb) for name, frame in frames.items()
        }
        unlit = self._unlit_cameras()
        if unlit:
            _log.info(
                "Camera(s) %s returned an unlit RGB frame (bright end %s); "
                "their colour is ignored for this observation",
                ", ".join(unlit),
                ", ".join(f"{self._frame_brightness[n]:.0f}" for n in unlit),
            )
        return frames

    def _unlit_cameras(self) -> list[str]:
        return sorted(
            name for name, value in self._frame_brightness.items() if value <= UNLIT_FRAME_VALUE
        )

    def _accumulate_instances(
        self, frames: dict[str, CameraFrame]
    ) -> dict[str, dict[str, Any]]:
        """Collect world-frame points per object instance, **kept separate per camera**.

        Keyed by prim path, which is the only identifier stable across two
        cameras: segmentation *ids* are assigned per render product and mean
        different things in different views.

        The per-camera separation is deliberate. Concatenating the two clouds
        and fitting once is the obvious approach and it measurably *degrades*
        the result: each camera sees a different face, so the two partial
        surfaces sit tens of millimetres apart, and the union spans far more
        than the object. Measured on this scene, per-camera fits recovered a
        0.05 m cube as 0.058/0.050/0.051 while the fused fit reported 0.095 --
        nearly double, and grasp width comes straight from that number.
        Combining partial views correctly needs ICP or TSDF integration; until
        then :meth:`_fit_hypotheses` picks the best-supported single view.
        """
        instances: dict[str, dict[str, Any]] = {}

        for name, frame in frames.items():
            if frame.depth is None or frame.segmentation is None:
                continue
            brightness = self._frame_brightness.get(name)
            color_usable = (
                not is_unlit_frame(frame.rgb)
                if brightness is None
                else brightness > UNLIT_FRAME_VALUE
            )

            cloud = deproject_to_point_cloud(frame, self.config, to_world=True)
            if len(cloud) == 0 or cloud.seg_ids is None:
                continue

            for seg_id in np.unique(cloud.seg_ids):
                seg_id = int(seg_id)
                prim_path = frame.seg_id_to_prim.get(seg_id, "")
                label = frame.seg_id_to_label.get(seg_id, "")

                if self._is_non_object(label, prim_path):
                    continue

                mask = cloud.seg_ids == seg_id
                points = cloud.points[mask]
                if points.shape[0] == 0:
                    continue

                key = prim_path or f"{name}:{seg_id}"
                entry = instances.setdefault(
                    key, {"per_camera": {}, "label": label, "seg_id": seg_id, "colors": []}
                )
                entry["per_camera"][name] = points
                # Colour is what lets an operator say "the green box" rather than
                # having to know the class vocabulary. Collected here because this
                # is the only place per-point RGB is still aligned with the mask.
                if cloud.colors is not None and color_usable:
                    entry["colors"].append(cloud.colors[mask])

        return instances

    @staticmethod
    def _is_non_object(label: str, prim_path: str) -> bool:
        """Filter out background and support surfaces.

        Matches on the semantic class, not on any specific object name -- the
        framework must never branch on "is this the can".
        """
        if label.upper() in _NON_OBJECT_LABELS:
            return True
        if label.strip().lower() in _SUPPORT_LABELS:
            return True
        if prim_path and not prim_path.startswith(_OBJECT_SCOPE):
            # Robot links and scene furniture live outside the objects scope.
            return True
        return False

    def _fit_hypotheses(
        self, instances: dict[str, dict[str, Any]]
    ) -> list[ObjectHypothesis]:
        """Estimate a 6-DoF pose and OBB per instance from its best-supported view.

        Each camera's cloud is fitted independently; the view with the most
        surviving points wins. Cross-view agreement is then used only to adjust
        confidence -- never to average the geometry, since averaging two
        partial-surface estimates produces a third estimate that matches
        neither.

        A cross-view *centre* estimator was tried here and removed. Taking the
        horizontal centre from the midpoint of both clouds' trimmed extents is
        sound in principle -- two views bracket the object, where one is biased
        toward the camera -- and it measurably improved accuracy, halving the
        cracker box's centre error from 47 mm to 14 mm. It still made the robot
        worse: the pick rate fell from 3/8 to 2/8 and the mug regressed from a
        clean +118 mm lift to a failure.

        Worth stating plainly, because the instinct is to keep the more accurate
        number: a better-centred box is not automatically a better *graspable*
        box. Grasp candidates are generated from this pose and filtered on it,
        so shifting a centre relocates every pre-grasp standoff, and a
        standoff that was reachable may stop being so. Accuracy against ground
        truth is not the objective function; lifting the object is.
        """
        hypotheses: list[ObjectHypothesis] = []

        for key, entry in instances.items():
            fits: dict[str, tuple[NDArray, NDArray, NDArray, int]] = {}

            for camera_name, raw_points in entry["per_camera"].items():
                fit = self._fit_single_view(raw_points)
                if fit is not None:
                    fits[camera_name] = fit

            if not fits:
                continue

            best_camera = max(fits, key=lambda name: fits[name][3])
            center, quat, extents, num_points = fits[best_camera]

            agreement = self._view_agreement(fits)
            confidence = self._confidence(num_points, len(fits), agreement)
            if confidence < self.config.min_confidence:
                continue

            color = ""
            if entry.get("colors"):
                color = dominant_color_name(np.vstack(entry["colors"]))

            pose = Pose(center, quat, Frame.WORLD)
            hypotheses.append(
                ObjectHypothesis(
                    # Placeholder; the tracker assigns the stable identity.
                    track_id="",
                    label=entry["label"],
                    pose=pose,
                    bbox=BoundingBox3D(center=pose, extents=extents),
                    confidence=confidence,
                    num_points=num_points,
                    last_seen_sim_time=self._sim.sim_time,
                    last_seen_step=self._sim.step_index,
                    seg_id=entry["seg_id"],
                    attributes={
                        "prim_path": key,
                        "cameras": sorted(fits),
                        "primary_camera": best_camera,
                        "view_agreement_m": agreement,
                        "color": color,
                    },
                )
            )

        return hypotheses

    def _fit_single_view(
        self, raw_points: NDArray[np.float64]
    ) -> tuple[NDArray, NDArray, NDArray, int] | None:
        """Downsample, denoise and box-fit one camera's view of one instance."""
        if raw_points.shape[0] < self.config.min_points_per_object:
            return None

        # Drop points at or below the surface the object rests on.
        #
        # Correct in principle: an instance mask carries a fringe of edge pixels
        # whose depth interpolates onto the background, landing as a skirt of
        # points across the support. Filtering against ``ground_plane_z`` (the
        # floor, 0.0) never removed them for a table-top object 400 mm up.
        #
        # Honest note on what this did NOT fix. It was added to chase a centre
        # error of up to 71 mm on the benchmark objects, and it did not move
        # that number at all -- the soup can measured 71.3 mm off before and
        # 71.4 mm after. The skirt was not the cause. The residual bias is
        # inherent to fitting a box to a **single view**: the points describe
        # the visible surface, not the volume, so the fitted centre sits
        # roughly half an object-depth toward the camera. No amount of
        # filtering recovers geometry that was never observed.
        support = self._support_height + self.config.ground_clearance
        above = raw_points[:, 2] > support
        if int(above.sum()) >= self.config.min_points_per_object:
            raw_points = raw_points[above]

        points, _ = voxel_downsample(raw_points, self.config.voxel_size)
        if points.shape[0] < 4:
            return None

        # The neighbour filter is O(n^2) in memory; cap the input so a large
        # segment cannot allocate a multi-gigabyte distance matrix mid-loop.
        if points.shape[0] > self.config.max_points_for_outlier_filter:
            stride = int(np.ceil(points.shape[0] / self.config.max_points_for_outlier_filter))
            points = points[::stride]

        keep = remove_statistical_outliers(points, std_ratio=self.config.outlier_std_ratio)
        filtered = points[keep] if int(keep.sum()) >= 4 else points

        try:
            center, quat, extents = fit_oriented_bbox(
                filtered,
                gravity_aligned=True,
                trim_percentile=self.config.bbox_trim_percentile,
            )
        except ValueError:
            return None

        return center, quat, extents, int(filtered.shape[0])

    @staticmethod
    def _view_agreement(fits: dict[str, tuple[NDArray, NDArray, NDArray, int]]) -> float:
        """Largest centre disagreement between views, in metres.

        Each camera measures the surface it can see, so some disagreement is
        expected and normal. A large value means the views are describing
        different things -- a partially occluded object, or a segmentation leak
        onto the background.
        """
        centers = [fit[0] for fit in fits.values()]
        if len(centers) < 2:
            return 0.0
        return float(
            max(
                np.linalg.norm(a - b)
                for i, a in enumerate(centers)
                for b in centers[i + 1 :]
            )
        )

    def _confidence(self, num_points: int, num_views: int, agreement: float) -> float:
        """Score an observation by how much evidence supports it.

        Point count saturating at 10x the minimum, a bonus for corroboration by
        a second camera, and a penalty when those cameras disagree about where
        the object is.
        """
        point_score = min(1.0, num_points / max(1.0, self.config.min_points_per_object * 10.0))
        multi_view_bonus = 0.15 if num_views > 1 else 0.0
        disagreement_penalty = min(0.3, agreement * 2.0) if num_views > 1 else 0.0
        return float(
            np.clip(0.35 + 0.55 * point_score + multi_view_bonus - disagreement_penalty, 0.0, 1.0)
        )

    # ------------------------------------------------------------------
    # helpers for skills
    # ------------------------------------------------------------------

    def support_height_at(self, position: NDArray[np.float64], default: float) -> float:
        """Height of the surface beneath ``position``.

        Used by Place to find where to release. Derived from the tallest
        perceived object whose footprint contains the point, falling back to the
        configured plane when nothing is underneath.
        """
        scene = self._last_scene
        if scene is None:
            return default

        best = default
        for obj in scene.objects.values():
            half = obj.bbox.extents[:2] / 2.0
            delta = np.abs(obj.bbox.center.position[:2] - np.asarray(position)[:2])
            if np.all(delta <= half):
                top = float(obj.bbox.center.position[2] + obj.bbox.extents[2] / 2.0)
                best = max(best, top)
        return best

"""Calibrated RGB-D + segmentation camera.

Isaac Sim is imported lazily inside methods; see ``mfw.simulation.app``.

This is the only place in the framework that talks to a render product. The
planner receives a :class:`~mfw.core.types.SceneGraph` and never sees pixels --
if planning could reach the camera it would eventually start making perception
decisions, and the two layers would fuse.

Frame convention
----------------
Isaac's camera prims look down their local **-Z** with **+Y** up, which is the
USD/OpenGL convention. Computer-vision maths (and the pinhole projection used
here) assumes **+Z forward, +X right, +Y down** -- the OpenCV convention.
:meth:`Camera.get_extrinsics` returns the OpenCV-convention pose, applying the
180-degree flip about X once, here, rather than leaving every downstream
consumer to remember it. Getting this wrong mirrors the point cloud vertically,
which looks subtly plausible and ruins every grasp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import CameraConfig, PerceptionConfig
from mfw.core.errors import PerceptionError
from mfw.core.interfaces import ICamera
from mfw.core.types import CameraFrame, CameraIntrinsics, Frame, PointCloud, Pose
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = ["Camera", "deproject_to_point_cloud"]

_log = get_logger("vision.camera")

# Rotation from USD camera convention (-Z forward, +Y up) to OpenCV
# (+Z forward, +Y down): 180 degrees about the local X axis.
_USD_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


class Camera(ICamera):
    """One calibrated sensor producing synchronised RGB, depth and segmentation."""

    def __init__(self, sim: Any, config: CameraConfig, perception_config: PerceptionConfig) -> None:
        config.validate()
        self.config = config
        self.perception_config = perception_config
        self._sim = sim
        self._camera: Any = None
        self._initialized = False
        self._intrinsics: CameraIntrinsics | None = None
        self._label_cache: dict[str, str] = {}
        self._spawn()

    def _spawn(self) -> None:
        from isaacsim.sensors.camera import Camera as IsaacCamera  # noqa: PLC0415

        width, height = self.config.resolution
        # Deliberately no position/orientation here.
        #
        # Isaac's Camera interprets pose arguments in its *own* axis convention,
        # which defaults to camera_axes="world" (+Z up, +X forward) -- not the
        # USD camera convention (-Z forward, +Y up) that this framework's
        # look-at maths produces. Passing a USD-convention quaternion through
        # that layer silently rotates the camera to point somewhere else
        # entirely: measured here, it swung an intended table view onto the
        # horizon, producing depth full of NaN and an empty segmentation while
        # every pose query still reported the value we asked for.
        #
        # Writing the USD xform ourselves keeps exactly one convention in play
        # and makes the transform auditable on the prim.
        self._camera = IsaacCamera(
            prim_path=self.config.prim_path,
            name=self.config.name,
            resolution=(int(width), int(height)),
        )
        self._set_local_transform()

    def _set_local_transform(self) -> None:
        """Write the camera's transform directly, in USD camera convention.

        For a parented camera (wrist) this is a local offset that rides with the
        hand. For an unparented camera the local transform is the world
        transform, so the same code path serves both.
        """
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        stage = self._sim.world.stage
        prim = stage.GetPrimAtPath(self.config.prim_path)
        if not prim or not prim.IsValid():
            raise PerceptionError(f"Camera prim {self.config.prim_path} was not created")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        matrix = tf.make_transform(self.config.position, self.config.resolved_quat())
        gf_matrix = Gf.Matrix4d(*matrix.T.flatten().tolist())
        xform.AddTransformOp().Set(gf_matrix)

    def initialize(self) -> None:
        """Create render products and enable the annotators.

        Must run after ``World.reset()``; the render product does not exist
        before then and annotator attachment silently no-ops.
        """
        if self._initialized:
            return

        self._camera.initialize()
        self._camera.set_focal_length(float(self.config.focal_length))
        self._camera.set_horizontal_aperture(float(self.config.horizontal_aperture))
        self._camera.set_vertical_aperture(float(self.config.vertical_aperture))
        self._camera.set_clipping_range(*[float(v) for v in self.config.clipping_range])

        if self.config.enable_depth:
            self._camera.add_distance_to_image_plane_to_frame()
            """Distance to the image *plane*, not to the camera centre: this is
            true pinhole Z depth, which is what deprojection needs. The radial
            'distance_to_camera' annotator would bow the point cloud outward."""
        if self.config.enable_segmentation:
            self._camera.add_instance_segmentation_to_frame()

        self._initialized = True
        _log.info(
            "Camera %s initialised at %dx%d (depth=%s, seg=%s)",
            self.config.name,
            *self.config.resolution,
            self.config.enable_depth,
            self.config.enable_segmentation,
        )

    @property
    def name(self) -> str:
        return self.config.name

    def get_intrinsics(self) -> CameraIntrinsics:
        """Pixel-space intrinsics, derived from the physical lens parameters.

        Isaac parameterises cameras by focal length and aperture in millimetres;
        this converts to the pixel units the projection maths uses. Cached
        because the lens does not change at runtime.
        """
        if self._intrinsics is not None:
            return self._intrinsics

        width, height = (int(v) for v in self.config.resolution)
        fx = self.config.focal_length * width / self.config.horizontal_aperture
        # fy comes from vertical_aperture, NOT from fx.
        #
        # The apertures do not match the resolution's aspect ratio -- 3.896 mm
        # across 640 px implies 2.922 mm across 480 px, while the config says
        # 2.453 -- so these are not square pixels and fx != fy. That looks like
        # a bug and is not one: forcing fy = fx was tried and measurably
        # degraded reconstruction, adding vertical errors of up to 43 mm to
        # objects whose Z had been accurate to within 5 mm. Isaac's renderer
        # uses the configured vertical aperture, so deprojection must too.
        fy = self.config.focal_length * height / self.config.vertical_aperture
        self._intrinsics = CameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )
        return self._intrinsics

    def get_extrinsics(self) -> Pose:
        """Camera pose in world, in the OpenCV optical convention.

        Re-read on every call: a wrist camera moves with the arm, and a cached
        extrinsic would silently reconstruct the point cloud at the pose the
        camera held on the previous frame.
        """
        self._require_init()
        from isaacsim.core.utils.xforms import get_world_pose  # noqa: PLC0415

        # Read the raw USD prim transform rather than Camera.get_world_pose(),
        # which returns its result in Isaac's configurable camera_axes
        # convention (default +X forward). Going straight to USD keeps a single,
        # documented convention across the whole perception path.
        position, quat = get_world_pose(self.config.prim_path)

        usd_matrix = tf.make_transform(np.asarray(position), np.asarray(quat))
        optical = np.eye(4)
        optical[:3, :3] = _USD_TO_OPENCV
        return Pose.from_matrix(usd_matrix @ optical, Frame.WORLD)

    def capture(self) -> CameraFrame:
        """Return the most recent rendered frame.

        Does not step the simulation. The caller must have rendered at least one
        frame since the last scene change, or this returns stale pixels.
        """
        self._require_init()
        frame = self._camera.get_current_frame()

        rgba = self._camera.get_rgba()
        if rgba is None or rgba.size == 0:
            raise PerceptionError(
                f"Camera {self.config.name} produced no RGB data. Step the simulation "
                "with render=True at least once before capturing."
            )
        rgb = np.asarray(rgba[..., :3], dtype=np.uint8)

        depth = None
        if self.config.enable_depth:
            raw_depth = frame.get("distance_to_image_plane")
            if raw_depth is not None:
                depth = np.asarray(raw_depth, dtype=np.float32)
                # The renderer reports misses as inf; NaN is a friendlier
                # sentinel because it propagates through arithmetic instead of
                # producing enormous finite coordinates.
                depth[~np.isfinite(depth)] = np.nan

        segmentation = None
        seg_labels: dict[int, str] = {}
        seg_prims: dict[int, str] = {}
        if self.config.enable_segmentation:
            seg_data = frame.get("instance_segmentation")
            if seg_data is not None:
                segmentation = np.asarray(seg_data["data"], dtype=np.int32)
                info = seg_data.get("info", {}) or {}
                for key, value in (info.get("idToLabels", {}) or {}).items():
                    try:
                        seg_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    raw = str(value)
                    seg_prims[seg_id] = raw
                    seg_labels[seg_id] = self._resolve_class_label(raw)

        return CameraFrame(
            camera_name=self.config.name,
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            seg_id_to_label=seg_labels,
            seg_id_to_prim=seg_prims,
            intrinsics=self.get_intrinsics(),
            extrinsics=self.get_extrinsics(),
            sim_time=self._sim.sim_time,
            step_index=self._sim.step_index,
        )

    def _resolve_class_label(self, raw_label: str) -> str:
        """Map an instance-segmentation entry to a semantic class name.

        Isaac's instance annotator identifies each instance by its USD prim path
        ("/World/objects/blue_can"), not by the semantic class we tagged it with
        ("can"). Language grounding matches on the class, so resolve the path
        back to its ``SemanticsLabelsAPI`` label.

        Cached: the mapping is fixed for the life of a prim, and this would
        otherwise run a USD query per segment per frame inside the perception
        loop. Non-path entries (BACKGROUND, UNLABELLED) pass through unchanged.
        """
        if raw_label in self._label_cache:
            return self._label_cache[raw_label]

        resolved = raw_label
        if raw_label.startswith("/"):
            try:
                from isaacsim.core.utils.semantics import get_labels  # noqa: PLC0415

                prim = self._sim.world.stage.GetPrimAtPath(raw_label)
                if prim and prim.IsValid():
                    labels = get_labels(prim) or {}
                    values = labels.get("class") or []
                    if values:
                        resolved = str(values[0])
            except Exception as exc:  # pragma: no cover - version dependent
                _log.debug("Could not resolve semantic label for %s: %s", raw_label, exc)

        self._label_cache[raw_label] = resolved
        return resolved

    def _require_init(self) -> None:
        if not self._initialized:
            raise PerceptionError(
                f"Camera {self.config.name}.initialize() has not been called (must follow World.reset())"
            )


def deproject_to_point_cloud(
    frame: CameraFrame,
    perception_config: PerceptionConfig,
    to_world: bool = True,
) -> PointCloud:
    """Turn an RGB-D frame into a point cloud.

    Fully vectorised: a 640x480 frame is 307k pixels and this runs before every
    action, so a per-pixel loop would dominate the perception budget.

    Invalid depth (NaN from the renderer, or outside the configured working
    range) is dropped rather than clamped. Clamping would invent a surface at
    the range limit and the grasp planner would happily plan against it.
    """
    if frame.depth is None:
        raise PerceptionError(
            f"Camera {frame.camera_name} has no depth; cannot build a point cloud"
        )

    intrinsics = frame.intrinsics
    depth = frame.depth
    height, width = depth.shape[:2]

    valid = (
        np.isfinite(depth)
        & (depth > perception_config.depth_min)
        & (depth < perception_config.depth_max)
    )
    if not np.any(valid):
        return PointCloud(
            points=np.empty((0, 3), dtype=np.float64),
            frame=Frame.WORLD if to_world else Frame.WRIST_CAMERA,
        )

    vs, us = np.nonzero(valid)
    zs = depth[vs, us].astype(np.float64)

    xs = (us.astype(np.float64) - intrinsics.cx) * zs / intrinsics.fx
    ys = (vs.astype(np.float64) - intrinsics.cy) * zs / intrinsics.fy
    points = np.stack([xs, ys, zs], axis=1)

    colors = None
    if frame.rgb is not None and frame.rgb.shape[:2] == (height, width):
        colors = frame.rgb[vs, us]

    seg_ids = None
    if frame.segmentation is not None and frame.segmentation.shape[:2] == (height, width):
        seg_ids = frame.segmentation[vs, us].astype(np.int32)

    if to_world:
        points = tf.transform_points(frame.extrinsics.as_matrix(), points)
        out_frame = Frame.WORLD
    else:
        out_frame = Frame.WRIST_CAMERA

    return PointCloud(points=points, frame=out_frame, colors=colors, seg_ids=seg_ids)

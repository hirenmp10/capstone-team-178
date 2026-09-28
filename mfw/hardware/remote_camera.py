"""``ICamera`` over the Jetson robot server's ``get_frame`` endpoint.

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers; ``cv2`` is imported lazily for JPEG decoding.

This camera is **not** what perception plans from. On the hardware lane the
detector service owns the webcam and only detections cross the LAN;
:class:`RemoteCamera` exists for the table-homography calibration script
(click points on a live frame) and for the Florence fallback, where the
laptop needs pixels. ``depth`` and ``segmentation`` are always ``None``: a
webcam has neither, and pretending otherwise would let the sim perception
stack run against garbage.

Intrinsics come from ``CameraConfig.pixel_intrinsics()`` (measured fx/fy/cx/cy
when set, Isaac's aperture formula otherwise) but the image size comes from
the *frame*: the Jetson's V4L2 mode is whatever its calibration says, and a
mismatch is logged once rather than papered over.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import CameraConfig
from mfw.core.errors import PerceptionError
from mfw.core.interfaces import ICamera
from mfw.core.types import CameraFrame, CameraIntrinsics, Frame, Pose
from mfw.hardware.jetson_client import JetsonClient, RpcError
from mfw.utils import transforms as tf
from mfw.utils.logging import get_logger

__all__ = ["RemoteCamera", "decode_jpeg"]

_log = get_logger("hardware.remote_camera")

# USD camera (-Z forward, +Y up) -> OpenCV optical (+Z forward, +Y down): a
# half-turn about X. Same constant the sim camera applies.
_USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def decode_jpeg(data: bytes) -> NDArray[np.uint8]:
    """JPEG bytes -> RGB uint8 (H, W, 3). Raises :class:`PerceptionError` on failure."""
    try:
        import cv2
    except ImportError as exc:
        raise PerceptionError(
            "opencv-python is required to decode camera frames; "
            "`py -3.12 -m pip install opencv-python`"
        ) from exc
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PerceptionError("camera frame could not be decoded as JPEG")
    return np.ascontiguousarray(bgr[:, :, ::-1])


class RemoteCamera(ICamera):
    """A fixed webcam on the Jetson, read on demand."""

    def __init__(self, client: JetsonClient, config: CameraConfig, clock: Any) -> None:
        config.validate()
        self._client = client
        self.config = config
        self._clock = clock
        self._size_warned = False

    @property
    def name(self) -> str:
        return self.config.name

    def get_intrinsics(self, width: int | None = None, height: int | None = None) -> CameraIntrinsics:
        """Pixel intrinsics; ``width``/``height`` default to the configured resolution."""
        fx, fy, cx, cy = self.config.pixel_intrinsics()
        w = int(self.config.resolution[0]) if width is None else int(width)
        h = int(self.config.resolution[1]) if height is None else int(height)
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)

    def get_extrinsics(self) -> Pose:
        """Configured pose in the OpenCV optical convention (fixed camera)."""
        position = np.asarray(self.config.position, dtype=np.float64)
        if self.config.look_at is not None:
            quat = tf.look_at_quat(position, np.asarray(self.config.look_at), self.config.up)
        else:
            quat = np.asarray(self.config.quat, dtype=np.float64)
        usd = tf.make_transform(position, quat)
        optical = np.eye(4)
        optical[:3, :3] = _USD_TO_OPENCV
        return Pose.from_matrix(usd @ optical, Frame.WORLD)

    def capture(self) -> CameraFrame:
        """Fetch and decode the latest JPEG. Raises :class:`PerceptionError`."""
        try:
            reply = self._client.get_frame()
        except RpcError as exc:
            raise PerceptionError(f"camera {self.name!r}: {exc}") from exc
        rgb = decode_jpeg(reply["jpeg"])
        h, w = rgb.shape[:2]
        if (w, h) != tuple(int(v) for v in self.config.resolution) and not self._size_warned:
            self._size_warned = True
            _log.warning(
                "camera %s delivers %dx%d but the config says %s; intrinsics assume the config",
                self.name, w, h, list(self.config.resolution),
            )
        return CameraFrame(
            camera_name=self.name,
            rgb=rgb,
            depth=None,
            segmentation=None,
            seg_id_to_label={},
            seg_id_to_prim={},
            intrinsics=self.get_intrinsics(w, h),
            extrinsics=self.get_extrinsics(),
            sim_time=float(self._clock.sim_time),
            step_index=int(self._clock.step_index),
        )

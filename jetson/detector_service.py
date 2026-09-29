"""Object-detection service for the hardware lane (TCP newline-JSON, port 5558).

The server, the scripted backend and the frame plumbing shared by
``scripts/serve_detector.py`` -- **the** detector on :5558 (Florence-2 ONNX
FP16 on the Jetson). This file is imported there and runs standalone only for
the scripted fake, so it must never import ``mfw``: the Jetson deployment is
this file plus numpy, pyzmq/msgpack for frames, and opencv *or* pillow for
JPEG decoding. Python 3.10 compatible.

Protocol (one JSON object per line, reply per request, many requests per
connection):

    {"cmd": "ping"}
        -> {"ok": true, "backend": "scripted|florence-onnx|florence|nanoowl"}
    {"cmd": "detect", "labels": [...], "min_score": 0.1[, "jpeg": "<base64>"]}
        -> {"ok": true, "t": <unix s>, "width": W, "height": H,
            "objects": [{"label": str, "confidence": float,
                         "bbox_px": [x0, y0, x1, y1]}],
            "frames": {...}}        # model backends only: which frames voted

Model backends (Florence) run behind a :class:`FramePipeline`: frames come
**only** from ``robot_server.py``'s ``get_frame`` (the process that owns the
webcam -- two processes opening /dev/video0 is the EBUSY trap the audit
found), a frame older than ``max_frame_age_s`` is refused rather than
detected on, a request that carries ``jpeg`` is refused, and an object is
reported only when it was detected in ``required`` of the last ``window``
distinct *frames* (:class:`FrameVoter`; not requests, and never the warm-up
frame). The scripted backend keeps the legacy path (``jpeg`` accepted and
ignored, no frame provider). Geometry (pixel -> table) is deliberately *not*
here; it lives in ``mfw/hardware/perception.py``.

NanoOWL (:class:`NanoOwlDetector`) is **not the simulation's model; do not
deploy it**. It stays for reference only; ``--backend`` defaults to
``scripted`` so it is never started by accident.

Why a scripted backend ships in the same file: the whole laptop side has to
run end-to-end before servos or a webcam exist. :class:`SyntheticPinhole` is an
analytic camera whose pixel -> table homography is known exactly, and
:class:`ScriptedDetector` renders boxes for a scripted scene through it. With
``--world-from HOST:PORT`` the scene is read live from a
``robot_server.py --driver fake --fake-world ...`` process, so two standalone
fakes agree about what was picked up.

The scripted camera is *honest* by default: a lifted object keeps being
reported, its raised 3-D box projected like any other (so it grows and shifts
in the image, exactly as a real overhead webcam sees a carried marker), and an
object is hidden only where the gripper's footprint, projected from the fake
world's TCP, covers most of its box. The old rule -- "anything whose base is
above :data:`HIDE_ABOVE_Z` vanishes" -- is not something any camera does; the
review found the fake lane passing only because of it (the verifier read
"vanished" as "carried") while the real-camera path failed. It survives as an
explicit legacy mode (``hide_above_z=...`` / ``--hide-above 0.04``) for tests
that show what the verifier does with an unexplained absence.

P3, server half: :func:`warm_up` runs one detect on a black frame at startup
(NanoOWL builds its TensorRT context and encodes the vocabulary lazily, 10-40
s on the Orin) and logs how long it took, so the first real request is not
the one that pays for the load.

Measured trap recorded here so nobody re-learns it: the pixel bbox of a real
object in an oblique overhead view has its *bottom edge* on the near face's
contact line with the table, not under the centroid. The detector reports the
box honestly (this file projects the full 3-D box, not just the footprint);
the estimator on the laptop owns the half-depth correction.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import socketserver
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HIDE_ABOVE_Z",
    "ARM_FOOTPRINT_HALF_WIDTH_M",
    "ARM_FOOTPRINT_HEIGHT_M",
    "OCCLUSION_COVER_FRACTION",
    "ARM_OCCLUDES_BELOW_Z",
    "DEFAULT_FOOTPRINTS",
    "DEFAULT_FOOTPRINT",
    "DEFAULT_SYNONYMS",
    "SyntheticPinhole",
    "ScriptedDetector",
    "RobotWorldScene",
    "FrameVoter",
    "LabelRule",
    "Frame",
    "FrameUnavailable",
    "StaleFrameError",
    "RobotFrameSource",
    "FramePipeline",
    "box_iou",
    "NanoOwlDetector",
    "CameraLoop",
    "DetectorServer",
    "decode_jpeg",
    "parse_scene",
    "format_homography",
    "add_common_arguments",
    "build_scripted",
    "build_parser",
    "warm_up",
    "main",
]

_log = logging.getLogger("jetson.detector")

DEFAULT_PORT = 5558

HIDE_ABOVE_Z = 0.04
"""Legacy-mode threshold: objects whose base z exceeds this are not reported.

Not physics -- no camera loses an object because it rose 4 cm. Used only by
``ScriptedDetector(hide_above_z=HIDE_ABOVE_Z)`` / ``--hide-above 0.04``; the
default (honest) mode keeps reporting lifted objects and hides them only
under the projected gripper footprint."""

ARM_FOOTPRINT_HALF_WIDTH_M = 0.0225
"""Honest-mode occluder: half-width of the jaw volume around the TCP (the kit's
45 mm open span). Mirrors ``mfw.hardware.perception.ARM_FOOTPRINT_HALF_WIDTH_M``
(this file may not import ``mfw``); keep the two equal."""
ARM_FOOTPRINT_HEIGHT_M = 0.05
"""Honest-mode occluder: height of the jaw volume above the TCP (finger length).
A solid 60 x 60 x 90 mm block was tried first: its projected *rectangle*
covered 70-75 % of a carried marker's box, so a 140 mm marker held in a 45 mm
jaw -- whose ends plainly stick out -- vanished, which is the old
hide-above-z artefact by another route."""
_ARM_FOOTPRINT_BELOW_M = 0.01
ARM_OCCLUDES_BELOW_Z = 0.20
"""Honest mode models the gripper *at work*: it occludes only while the TCP is
below this height (the configured workspace ceiling). The scripted camera sits
behind and above the base, right where the parked home pose (TCP ~0.34 m
high, ~0.10 m behind the base) would blank part of the table; the real overhead camera must be mounted so the
parked arm does not (a day-1 check), and the fake lane assumes it was."""
OCCLUSION_COVER_FRACTION = 0.6
"""Share of an object's pixel box the jaw footprint must cover to hide it; less
than that and a detector still finds the object from what sticks out. With the
fake camera a carried marker is covered 47-57 % (stays visible) and a carried
50 mm cube 86-92 % (hidden)."""

DEFAULT_FOOTPRINTS: dict[str, tuple[float, float, float]] = {
    # (x, y, z) extents in metres, long axis along world +X. Mirrors
    # hardware.object_sizes in configs/hardware.yaml.
    "marker": (0.140, 0.019, 0.019),
    "banana": (0.190, 0.036, 0.036),
    "box": (0.100, 0.070, 0.050),
    "cube": (0.050, 0.050, 0.050),
    "bowl": (0.150, 0.150, 0.060),
    "bin": (0.180, 0.120, 0.100),
}
DEFAULT_FOOTPRINT: tuple[float, float, float] = (0.04, 0.04, 0.04)

DEFAULT_SYNONYMS: dict[str, tuple[str, ...]] = {
    # Two to three phrasings per label. OWL-ViT is a phrase matcher, and the
    # bare word "bin" or "marker" scores poorly against a photo; the phrases are
    # encoded once and any of them counts as the label.
    "marker": ("a marker pen", "a whiteboard marker", "a felt tip pen"),
    "banana": ("a banana", "a yellow banana"),
    "box": ("a small cardboard box", "a box", "a carton"),
    "cube": ("a toy cube", "a wooden block", "a small cube"),
    "bowl": ("a bowl", "a small bowl", "a dish"),
    "bin": ("a small bin", "a small container", "a plastic tub"),
}


# ---------------------------------------------------------------------------
# Synthetic camera
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticPinhole:
    """An ideal pinhole camera with a known pose, in the OpenCV convention.

    Camera axes: +Z forward (along the view ray), +X right, +Y down. World +Z
    is up. Defaults are an oblique overhead view of the hobby arm's table:
    1920x1080, f = 1400 px, at (-0.10, 0, 0.60) looking at (0.18, 0, 0).

    Everything is analytic, so :meth:`homography_pixel_to_table` is exact and
    a test can check that projecting table points and mapping them back
    round-trips to well under a millimetre.
    """

    fx: float = 1400.0
    fy: float = 1400.0
    cx: float = 960.0
    cy: float = 540.0
    width: int = 1920
    height: int = 1080
    position: tuple[float, float, float] = (-0.10, 0.0, 0.60)
    look_at: tuple[float, float, float] = (0.18, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return (int(self.width), int(self.height))

    def intrinsic_matrix(self) -> NDArray[np.float64]:
        """The 3x3 ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def rotation_world_from_camera(self) -> NDArray[np.float64]:
        """Columns are the camera's x/y/z axes expressed in world coordinates."""
        eye = np.asarray(self.position, dtype=np.float64)
        target = np.asarray(self.look_at, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        forward = target - eye
        norm = float(np.linalg.norm(forward))
        if norm < 1e-9:
            raise ValueError("SyntheticPinhole: position and look_at coincide")
        z_axis = forward / norm
        x_axis = np.cross(z_axis, up)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-9:
            # Looking straight down: ``up`` cannot fix roll. Pick world +X as
            # the image-up direction, matching hardware.yaml's ``up: [1,0,0]``.
            x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
            x_norm = float(np.linalg.norm(x_axis))
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def project(self, xyz: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """World point(s) -> pixel ``(u, v)``. Accepts ``(3,)`` or ``(N, 3)``.

        Points behind the camera project to NaN rather than to a mirrored
        location, so a caller cannot mistake them for real image evidence.
        """
        pts = np.asarray(xyz, dtype=np.float64)
        single = pts.ndim == 1
        pts = pts.reshape(-1, 3)
        rot = self.rotation_world_from_camera()
        cam = (pts - np.asarray(self.position, dtype=np.float64)) @ rot
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * cam[:, 0] / z + self.cx
            v = self.fy * cam[:, 1] / z + self.cy
        uv = np.stack([u, v], axis=1)
        uv[z <= 1e-9] = np.nan
        return uv[0] if single else uv

    def homography_table_to_pixel(self) -> NDArray[np.float64]:
        """3x3 map from table ``(x, y, 1)`` (z = 0 plane) to pixel ``(u, v, 1)``."""
        rot_cw = self.rotation_world_from_camera().T
        t = -rot_cw @ np.asarray(self.position, dtype=np.float64)
        plane = np.stack([rot_cw[:, 0], rot_cw[:, 1], t], axis=1)
        return self.intrinsic_matrix() @ plane

    def homography_pixel_to_table(self) -> NDArray[np.float64]:
        """3x3 map from pixel ``(u, v, 1)`` to table ``(x, y, 1)``, normalised so ``H[2, 2] == 1``.

        Apply as ``p = H @ [u, v, 1]; x, y = p[0] / p[2], p[1] / p[2]``. This is
        the matrix ``exterior_camera.homography`` stores, row-major.
        """
        h = np.linalg.inv(self.homography_table_to_pixel())
        return h / h[2, 2]

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float]:
        """Convenience: one pixel to table ``(x, y)`` through the exact homography."""
        p = self.homography_pixel_to_table() @ np.array([u, v, 1.0])
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def in_image(self, u: float, v: float) -> bool:
        return bool(0.0 <= u < self.width and 0.0 <= v < self.height)


def format_homography(h: NDArray[np.float64]) -> str:
    """Render a 3x3 as the one-line YAML list ``exterior_camera.homography`` takes."""
    flat = [float(v) for v in np.asarray(h, dtype=np.float64).reshape(-1)]
    return "[" + ", ".join(repr(v) for v in flat) + "]"


# ---------------------------------------------------------------------------
# Scripted backend
# ---------------------------------------------------------------------------

SceneDict = Mapping[str, Sequence[float]]
SceneSource = SceneDict | Callable[[], SceneDict]


class ScriptedDetector:
    """Detections for a scripted scene, rendered through a :class:`SyntheticPinhole`.

    ``scene`` maps label -> ``(x, y)``, ``(x, y, z)`` or ``(x, y, z, yaw)`` in
    the table frame (metres, radians; ``z`` is the object's *base*), or is a
    zero-argument callable returning such a mapping so a fake world can move
    objects between calls. Each object is a box of the label's footprint
    (long axis along +X at ``yaw = 0``, rotated about Z by ``yaw``); its pixel
    bbox is the projection of all eight corners, clipped to the image.

    Rules:

    * **honest mode** (default, ``hide_above_z=None``): every object in view is
      reported wherever it is, lifted or not; an object is hidden only when
      the gripper footprint (a box :data:`ARM_FOOTPRINT_HALF_WIDTH_M` either
      side of the TCP, up to :data:`ARM_FOOTPRINT_HEIGHT_M` above it,
      projected) covers at least :data:`OCCLUSION_COVER_FRACTION` of its box.
      The TCP comes from ``arm``, or from ``scene.tcp()`` when the scene
      source has one (:class:`RobotWorldScene` does); without either nothing
      is occluded;
    * **legacy mode** (``hide_above_z`` a number): an object whose base is
      above it is not reported, and nothing else is occluded;
    * an object entirely outside the image is not reported;
    * ``labels`` restricts the vocabulary (empty means everything in the scene);
    * every detection carries ``confidence`` (default 0.9) and is dropped when
      it is below ``min_score``.

    Real-detector imperfections, off by default, so the grasp verifier can be
    tested against the boxes that fooled it (re-review, perception lens) --
    the exact boxes above never do:

    * ``box_scale``: every box is scaled about its centre (a detector that
      boxes loosely; vary it between calls to model jitter);
    * ``merge_jaw``: a box the jaw footprint overlaps is widened to take in
      the jaw -- NanoOWL boxing the fingers together with a marker left
      behind under them;
    * ``partial_occlusion``: a box the jaw covers partly (below the hiding
      share) is cut down to its largest visible piece -- a jaw splitting a
      marker and the detector boxing one end.

    A spin needs nothing here: give the object a yaw (the fake world's
    ``(x, y, z, yaw)``) and its axis-aligned box changes shape as a real one does.
    """

    name = "scripted"

    def __init__(
        self,
        scene: SceneSource,
        camera: SyntheticPinhole | None = None,
        footprints: Mapping[str, Sequence[float]] | None = None,
        confidence: float = 0.9,
        hide_above_z: float | None = None,
        default_footprint: Sequence[float] = DEFAULT_FOOTPRINT,
        arm: Callable[[], Sequence[float] | None] | None = None,
        box_scale: float = 1.0,
        merge_jaw: bool = False,
        partial_occlusion: bool = False,
    ) -> None:
        self._scene = scene
        self.camera = camera if camera is not None else SyntheticPinhole()
        merged: dict[str, tuple[float, float, float]] = dict(DEFAULT_FOOTPRINTS)
        for label, size in (footprints or {}).items():
            merged[label.strip().lower()] = _as_size(size, f"footprints[{label!r}]")
        self.footprints = merged
        self.default_footprint = _as_size(default_footprint, "default_footprint")
        self.confidence = float(confidence)
        self.hide_above_z = None if hide_above_z is None else float(hide_above_z)
        if not box_scale > 0.0:
            raise ValueError(f"box_scale must be > 0, got {box_scale}")
        self.box_scale = float(box_scale)
        self.merge_jaw = bool(merge_jaw)
        self.partial_occlusion = bool(partial_occlusion)
        if arm is None:
            tcp_reader = getattr(scene, "tcp", None)
            arm = tcp_reader if callable(tcp_reader) else None
        self._arm = arm

    @property
    def honest(self) -> bool:
        """True unless the legacy hide-above-z rule is on."""
        return self.hide_above_z is None

    @property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)`` the boxes are expressed in."""
        return self.camera.image_size

    def current_scene(self) -> dict[str, tuple[float, float, float, float]]:
        """The scene right now, normalised to ``label -> (x, y, z, yaw)``."""
        raw = self._scene() if callable(self._scene) else self._scene
        scene: dict[str, tuple[float, float, float, float]] = {}
        for label, xyz in raw.items():
            vals = [float(v) for v in xyz]
            while len(vals) < 4 and len(vals) >= 2:
                vals.append(0.0)
            if len(vals) != 4:
                raise ValueError(f"scene[{label!r}] must be (x, y[, z[, yaw]]), got {xyz!r}")
            scene[str(label).strip().lower()] = (vals[0], vals[1], vals[2], vals[3])
        return scene

    def arm_tcp(self) -> NDArray[np.float64] | None:
        """The fake world's TCP, if an arm source exists and answers."""
        if self._arm is None:
            return None
        tcp = self._arm()
        if tcp is None:
            return None
        vals = np.asarray([float(v) for v in tcp], dtype=np.float64).reshape(-1)
        return vals[:3] if vals.size >= 3 and np.all(np.isfinite(vals[:3])) else None

    def occluder(self) -> tuple[float, float, float, float] | None:
        """The honest-mode occluder right now: the gripper box while it works near the table."""
        if not self.honest:
            return None
        tcp = self.arm_tcp()
        if tcp is None or tcp[2] > ARM_OCCLUDES_BELOW_Z:
            return None
        return self.arm_box(tcp)

    def arm_box(self, tcp: Sequence[float] | None = None) -> tuple[float, float, float, float] | None:
        """Unclipped pixel box of the gripper footprint at ``tcp`` (default: the live TCP)."""
        centre = self.arm_tcp() if tcp is None else np.asarray(tcp, dtype=np.float64)
        if centre is None:
            return None
        r = ARM_FOOTPRINT_HALF_WIDTH_M
        corners = np.array(
            [
                [centre[0] + dx, centre[1] + dy, centre[2] + dz]
                for dx in (-r, r)
                for dy in (-r, r)
                for dz in (-_ARM_FOOTPRINT_BELOW_M, ARM_FOOTPRINT_HEIGHT_M)
            ]
        )
        uv = self.camera.project(corners)
        if not np.all(np.isfinite(uv)):
            return None
        return (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))

    def footprint_for(self, label: str) -> tuple[float, float, float]:
        return self.footprints.get(label.strip().lower(), self.default_footprint)

    def box_for(
        self,
        label: str,
        xyz: Sequence[float],
        occluder: tuple[float, float, float, float] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Pixel bbox of one object, or ``None`` if hidden or off-image.

        ``xyz`` is ``(x, y, z[, yaw])`` with ``z`` the base. ``occluder`` is a
        pixel box (honest mode: the gripper footprint) that hides the object
        when it covers at least :data:`OCCLUSION_COVER_FRACTION` of its box.
        """
        vals = [float(v) for v in xyz]
        x, y, z = vals[0], vals[1], vals[2]
        yaw = vals[3] if len(vals) > 3 else 0.0
        if self.hide_above_z is not None and z > self.hide_above_z:
            return None
        sx, sy, sz = self.footprint_for(label)
        hx, hy = sx / 2.0, sy / 2.0
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        corners = np.array(
            [
                [x + c * dx - s * dy, y + s * dx + c * dy, z + dz]
                for dx in (-hx, hx)
                for dy in (-hy, hy)
                for dz in (0.0, sz)
            ]
        )
        uv = self.camera.project(corners)
        if not np.all(np.isfinite(uv)):
            return None
        w, h = self.camera.image_size
        x0 = max(0.0, float(uv[:, 0].min()))
        y0 = max(0.0, float(uv[:, 1].min()))
        x1 = min(float(w - 1), float(uv[:, 0].max()))
        y1 = min(float(h - 1), float(uv[:, 1].max()))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        if occluder is not None:
            ow = max(0.0, min(x1, occluder[2]) - max(x0, occluder[0]))
            oh = max(0.0, min(y1, occluder[3]) - max(y0, occluder[1]))
            if ow * oh >= OCCLUSION_COVER_FRACTION * (x1 - x0) * (y1 - y0):
                return None
            if ow * oh > 0.0 and self.partial_occlusion:
                x0, y0, x1, y1 = _largest_visible_piece((x0, y0, x1, y1), occluder)
            elif ow * oh > 0.0 and self.merge_jaw:
                x0, y0 = min(x0, occluder[0]), min(y0, occluder[1])
                x1, y1 = max(x1, occluder[2]), max(y1, occluder[3])
        if self.box_scale != 1.0:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            hw, hh = (x1 - x0) / 2.0 * self.box_scale, (y1 - y0) / 2.0 * self.box_scale
            x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w - 1), x1), min(float(h - 1), y1)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def detect(
        self,
        rgb: NDArray[np.uint8] | None,
        labels: Sequence[str],
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Boxes for every visible scripted object the vocabulary asks for.

        ``rgb`` is accepted for protocol symmetry and ignored: the scene, not the
        pixels, is the ground truth here.
        """
        wanted = {str(l).strip().lower() for l in labels if str(l).strip()}
        if self.confidence < float(min_score):
            return []
        out: list[dict[str, Any]] = []
        scene = self.current_scene()  # read first: a RobotWorldScene refreshes its TCP here
        occluder = self.occluder()
        for label, xyz in scene.items():
            if wanted and label not in wanted:
                continue
            box = self.box_for(label, xyz, occluder)
            if box is None:
                continue
            out.append({"label": label, "confidence": self.confidence, "bbox_px": list(box)})
        return out


def _largest_visible_piece(
    box: tuple[float, float, float, float], cover: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """The largest axis-aligned part of ``box`` left of, right of, above or
    below ``cover`` (``box`` itself when nothing is left)."""
    x0, y0, x1, y1 = box
    pieces = [
        (x0, y0, min(x1, cover[0]), y1),
        (max(x0, cover[2]), y0, x1, y1),
        (x0, y0, x1, min(y1, cover[1])),
        (x0, max(y0, cover[3]), x1, y1),
    ]
    best = max(pieces, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    if (best[2] - best[0]) < 1.0 or (best[3] - best[1]) < 1.0:
        return box
    return best


class RobotWorldScene:
    """Scene source that reads a fake robot server's ``get_fake_world``.

    A fresh ZMQ REQ socket per call (msgpack framing, as the robot server
    speaks): simple, and immune to the stuck-REQ problem after a timeout.
    When the server does not answer, the last scene seen (initially
    ``fallback``) is returned and the failure is logged once, so the detector
    keeps serving instead of taking the whole lane down with it.
    """

    def __init__(self, host: str, port: int, fallback: SceneDict | None = None, timeout_s: float = 1.0) -> None:
        self.host, self.port = str(host), int(port)
        self.timeout_s = float(timeout_s)
        self._last: dict[str, tuple[float, ...]] = {
            str(k): tuple(float(x) for x in v) for k, v in (fallback or {}).items()
        }
        self._warned = False
        self._context: Any = None
        self._tcp: tuple[float, ...] | None = None

    def tcp(self) -> tuple[float, ...] | None:
        """The fake world's TCP from the last successful read (the honest-mode occluder)."""
        return self._tcp

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __call__(self) -> dict[str, tuple[float, ...]]:
        try:
            import msgpack  # noqa: PLC0415
            import msgpack_numpy  # noqa: PLC0415
            import zmq  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("--world-from needs pyzmq, msgpack and msgpack-numpy") from exc
        # A private context, never the process-global instance: that one is
        # terminated at interpreter exit and blocks while any socket is open.
        if self._context is None:
            self._context = zmq.Context()
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        socket.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        try:
            socket.connect(self.endpoint)
            socket.send(msgpack.packb({"endpoint": "get_fake_world", "data": {}}, default=msgpack_numpy.encode, use_bin_type=True))
            reply = msgpack.unpackb(socket.recv(), object_hook=msgpack_numpy.decode, raw=False)
            if not isinstance(reply, dict) or reply.get("error"):
                raise RuntimeError(reply.get("error") if isinstance(reply, dict) else "malformed reply")
            self._last = {str(k): tuple(float(x) for x in v) for k, v in reply["objects"].items()}
            tcp = reply.get("tcp")
            self._tcp = None if tcp is None else tuple(float(x) for x in tcp)
            self._warned = False
        except Exception as exc:  # noqa: BLE001 - keep serving on a dead peer
            if not self._warned:
                _log.warning("fake world at %s unavailable (%s); serving the last scene", self.endpoint, exc)
                self._warned = True
        finally:
            socket.close()
        return dict(self._last)


def _as_size(size: Sequence[float], what: str) -> tuple[float, float, float]:
    vals = tuple(float(v) for v in size)
    if len(vals) != 3 or any(v <= 0.0 for v in vals):
        raise ValueError(f"{what} must be a positive (x, y, z) triple, got {size!r}")
    return vals  # type: ignore[return-value]


def parse_scene(text: str) -> dict[str, tuple[float, ...]]:
    """Parse ``"marker:0.18,0.05 bowl:0.15,-0.12"`` into ``{label: (x, y[, z])}``."""
    scene: dict[str, tuple[float, ...]] = {}
    for token in text.replace(";", " ").split():
        if ":" not in token:
            raise ValueError(f"scene token {token!r} must look like label:x,y")
        label, coords = token.split(":", 1)
        parts = [p for p in coords.split(",") if p != ""]
        if len(parts) not in (2, 3):
            raise ValueError(f"scene token {token!r} needs x,y or x,y,z")
        scene[label.strip().lower()] = tuple(float(p) for p in parts)
    return scene


# ---------------------------------------------------------------------------
# NanoOWL backend
# ---------------------------------------------------------------------------


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes (0 for degenerate ones)."""
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class FrameVoter:
    """Report an object only when ``required`` of the last ``window`` distinct camera frames saw it.

    Florence-2 grounding draws a box for every phrase it is given, present or
    not, and a real detector misses a thin marker on some frames. A box is
    reported when at least ``required`` of the last ``window`` frames hold a
    same-label box overlapping it (IoU >= ``min_iou``); the reported box is
    the newest one of that cluster, so one dropped frame neither loses the
    object nor moves it, and a phantom box that lands somewhere different on
    each frame never gathers votes. Two same-label objects are two clusters.

    Frames are keyed by their camera ``frame_id`` (``robot_server``'s
    ``seq``): recording the same frame twice replaces it, it never votes
    twice. The audit found the old voter counted *requests* -- the same
    stale frame could confirm itself -- and was warmed up on a black frame
    that then voted "nothing" (the first real observe reported nothing).
    :meth:`decide` also says which labels are still *undecided* (could still
    reach ``required`` with the frames not yet seen), so a caller can stop
    grabbing frames as soon as every label is settled.

    ``required_views`` (default 1: off): each frame may carry a *view* tag
    (the Florence backend tags the caption order it used), and a box is
    confirmed only when its supporting frames span that many distinct views.
    On a still scene consecutive frames are near-identical and greedy
    decoding is deterministic, so two frames alone prove little: a phantom
    that one caption order produces every time would confirm itself; two
    caption orders agreeing is the real test (measured on sim renders).

    Pure logic, no model; unit-tested on the laptop.
    """

    def __init__(self, window: int = 3, required: int = 2, min_iou: float = 0.3, required_views: int = 1) -> None:
        if window < 1 or required < 1 or required > window:
            raise ValueError(f"need 1 <= required <= window, got {required}/{window}")
        if not 0.0 <= float(min_iou) <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if not 1 <= int(required_views) <= int(required):
            raise ValueError(f"need 1 <= required_views <= required, got {required_views}/{required}")
        self.window = int(window)
        self.required = int(required)
        self.min_iou = float(min_iou)
        self.required_views = int(required_views)
        self._frames: deque[tuple[Any, float, list[dict[str, Any]], Any]] = deque(maxlen=self.window)
        self._auto_id = 0

    def reset(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frame_ids(self) -> list[Any]:
        """Recorded frame ids, oldest first."""
        return [f[0] for f in self._frames]

    def has_frame(self, frame_id: Any) -> bool:
        return any(f[0] == frame_id for f in self._frames)

    def record(
        self, frame_id: Any, detections: Sequence[Mapping[str, Any]], t_capture: float = 0.0, view: Any = None
    ) -> bool:
        """Store one frame's detections; ``False`` when ``frame_id`` was already there (replaced, not added)."""
        dets = [dict(d) for d in detections]
        for i, f in enumerate(self._frames):
            if f[0] == frame_id:
                self._frames[i] = (frame_id, float(t_capture), dets, view)
                return False
        self._frames.append((frame_id, float(t_capture), dets, view))
        return True

    def prune(self, older_than: float) -> int:
        """Forget frames captured before ``older_than``; returns how many went."""
        keep = [f for f in self._frames if f[1] >= float(older_than)]
        dropped = len(self._frames) - len(keep)
        if dropped:
            self._frames = deque(keep, maxlen=self.window)
        return dropped

    def decide(
        self, min_score: float = 0.0, current: Any = None
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """``(confirmed boxes, undecided labels)`` over the recorded frames.

        ``current``: the frame ids grabbed by the request being answered
        (``None``: every recorded frame counts as current). Frames from an
        earlier request may add votes, but a cluster is confirmed only when
        at least one *current* frame is in it: two older frames must never
        outvote the newest one and report an object where it *was* (the
        review measured a lifted marker reported at its old place 1 s later).
        Undecided is judged on current frames plus the frames still to grab
        (``window`` minus the current ones): each new frame evicts the oldest,
        so history support cannot outlast them.
        """
        frames = [
            [d for d in f[2] if float(d.get("confidence", 1.0)) >= float(min_score)]
            for f in self._frames
        ]
        views = [f[3] for f in self._frames]
        if current is None:
            is_current = [True] * len(frames)
        else:
            current_ids = set(current)
            is_current = [f[0] in current_ids for f in self._frames]
        remaining = self.window - sum(is_current)
        used: set[tuple[int, int]] = set()
        confirmed: list[dict[str, Any]] = []
        undecided: set[str] = set()
        for fi in range(len(frames) - 1, -1, -1):  # newest frame first
            for di, det in enumerate(frames[fi]):
                if (fi, di) in used:
                    continue
                label = str(det.get("label", ""))
                box = det.get("bbox_px", det.get("bbox"))
                support: set[int] = set()
                for fj, other_frame in enumerate(frames):
                    for dj, other in enumerate(other_frame):
                        if (fj, dj) in used or str(other.get("label", "")) != label:
                            continue
                        if box_iou(box, other.get("bbox_px", other.get("bbox"))) >= self.min_iou:
                            support.add(fj)
                            used.add((fj, dj))
                seen_views = {views[fj] for fj in support}
                n_current = sum(1 for fj in support if is_current[fj])
                if len(support) >= self.required and len(seen_views) >= self.required_views and n_current:
                    confirmed.append(dict(det))
                elif remaining > 0 and n_current + remaining >= self.required:
                    undecided.add(label)
        return confirmed, undecided

    def update(self, detections: Sequence[Mapping[str, Any]], frame_id: Any = None) -> list[dict[str, Any]]:
        """Record one frame and return the confirmed set (``frame_id=None``: a new frame per call)."""
        if frame_id is None:
            self._auto_id += 1
            frame_id = ("auto", self._auto_id)
        self.record(frame_id, detections)
        return self.decide()[0]


@dataclass(frozen=True)
class LabelRule:
    """How one grounding label is asked for and which of its boxes are believed.

    Used by ``scripts/serve_detector.py`` (defined here, in an importable
    module, because a dataclass in a script loaded by path breaks).
    """

    prompt: str | None = None
    """Phrase put in the grounding caption (default: the label itself)."""
    synonyms: tuple[str, ...] = ()
    """Other phrases Florence may echo back that mean this label."""
    min_score: float = 0.5
    """Minimum box score (ONNX decoder coordinate confidence; the PyTorch backend's fixed score)."""
    min_side_px: float = 4.0
    """A box thinner than this is a sliver along the image edge, not an object."""
    max_area_frac: float = 0.15
    """A box covering more of the image than this is the table or the whole frame."""

    def names(self, label: str) -> list[str]:
        return [label] + ([self.prompt] if self.prompt else []) + list(self.synonyms)


# ---------------------------------------------------------------------------
# Frames from robot_server.py (the webcam's only owner)
# ---------------------------------------------------------------------------


class FrameUnavailable(RuntimeError):
    """``robot_server`` gave no frame: down, timed out, no camera, or an undecodable JPEG."""


class StaleFrameError(FrameUnavailable):
    """A frame arrived but was captured too long ago (stalled webcam, late reply)."""


@dataclass(frozen=True)
class Frame:
    """One decoded webcam frame; ``t_capture`` is on *this* host's clock."""

    rgb: NDArray[np.uint8]
    seq: int
    t_capture: float
    received: float

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.rgb.shape[1]), int(self.rgb.shape[0]))


def _zmq_rpc(endpoint_url: str, context_holder: dict[str, Any]) -> Callable[[str, Mapping[str, Any], float], Any]:
    """A robot_server REQ call: fresh socket per call (a timed-out REQ socket is stuck for good)."""

    def call(endpoint: str, data: Mapping[str, Any], timeout_s: float) -> Any:
        try:
            import msgpack  # noqa: PLC0415
            import zmq  # noqa: PLC0415
        except ImportError as exc:
            raise FrameUnavailable("frames from robot_server need pyzmq and msgpack (pip install pyzmq msgpack)") from exc
        try:
            import msgpack_numpy  # noqa: PLC0415

            encode, decode = msgpack_numpy.encode, msgpack_numpy.decode
        except ImportError:  # get_frame / ping carry no arrays; plain msgpack is enough
            encode, decode = None, None
        if context_holder.get("ctx") is None:
            context_holder["ctx"] = zmq.Context()
        sock = context_holder["ctx"].socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_s * 1000)))
        sock.setsockopt(zmq.SNDTIMEO, max(1, int(timeout_s * 1000)))
        try:
            sock.connect(endpoint_url)
            sock.send(msgpack.packb({"endpoint": endpoint, "data": dict(data)}, default=encode, use_bin_type=True))
            raw = sock.recv()
        except zmq.Again as exc:
            raise FrameUnavailable(f"{endpoint!r} to {endpoint_url} got no reply within {timeout_s:.1f} s") from exc
        except zmq.ZMQError as exc:
            raise FrameUnavailable(f"{endpoint!r} to {endpoint_url} failed: {exc}") from exc
        finally:
            sock.close()
        return msgpack.unpackb(raw, object_hook=decode, raw=False) if decode else msgpack.unpackb(raw, raw=False)

    return call


class RobotFrameSource:
    """Grabs frames from ``robot_server.py``'s ``get_frame`` (ZMQ REQ, msgpack) -- mfw-free.

    A frame's age comes from the reply's ``age_s`` (robot_server measures it
    on its monotonic clock; see :meth:`grab`), so the two hosts' clocks never
    have to agree -- the Jetson has no RTC battery. Only for a reply without
    ``age_s`` is ``t_capture`` (the robot server's wall clock) used, with the
    offset estimated from ``ping``'s ``t`` (NTP-style, the smallest of a few
    round trips) and refreshed every ``resync_s``.

    Note for the robot server's owner: every request here (``ping``,
    ``get_frame``) refreshes its host-liveness clock. This source only asks
    while the laptop is waiting on a ``detect``, so a dead laptop is still
    detected; do not add a background poller here.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 2.0,
        quality: int | None = None,
        resync_s: float = 60.0,
        rpc: Callable[[str, Mapping[str, Any], float], Any] | None = None,
        clock: Callable[[], float] = time.time,
        clock_sync_pings: int = 5,
        max_age_s: float | None = None,
    ) -> None:
        self.host, self.port = str(host), int(port)
        self.timeout_s = float(timeout_s)
        self.max_age_s = None if max_age_s is None else float(max_age_s)
        """Forwarded to ``get_frame``: robot_server refuses an older frame itself."""
        self.quality = None if quality is None else int(quality)
        self.resync_s = float(resync_s)
        self.clock_sync_pings = int(clock_sync_pings)
        self._ctx: dict[str, Any] = {}
        self._rpc = rpc if rpc is not None else _zmq_rpc(self.endpoint, self._ctx)
        self._clock = clock
        self.offset_s: float | None = None
        """Robot-server clock minus this host's clock, seconds."""
        self.rtt_s: float | None = None
        self._synced_at = -math.inf
        self._reports_age: bool | None = None
        """Whether the server's replies carry ``age_s`` (then no clock sync is needed); unknown at first."""

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def _call(self, endpoint: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        reply = self._rpc(endpoint, dict(data or {}), self.timeout_s)
        if not isinstance(reply, dict):
            raise FrameUnavailable(f"{endpoint!r} reply from {self.endpoint} is not a dict")
        if reply.get("error"):
            raise FrameUnavailable(f"robot_server {endpoint!r} error: {reply['error']}")
        return reply

    @property
    def same_host(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    def sync_clock(self) -> float:
        """Estimate the robot server's clock offset; returns it.

        Same host (the Jetson MVP): one clock, offset 0, nothing measured.
        Otherwise the ``ping`` with the smallest round trip out of
        ``clock_sync_pings`` wins (NTP's rule): the first request on a new
        connection is slow on one leg only (measured on the laptop: a 1.3 s
        first round trip put the midpoint 0.65 s off and every frame then
        read as 0.65 s stale).
        """
        if self.same_host:
            self.offset_s, self.rtt_s = 0.0, None
            self._synced_at = self._clock()
            return 0.0
        best: tuple[float, float] | None = None
        for _ in range(max(1, self.clock_sync_pings)):
            t0 = self._clock()
            reply = self._call("ping")
            t1 = self._clock()
            server_t = reply.get("t")
            if server_t is None:
                _log.warning("robot_server ping carries no 't'; assuming its clock equals ours")
                best = (0.0, max(0.0, t1 - t0))
                break
            sample = (float(server_t) - (t0 + t1) / 2.0, max(0.0, t1 - t0))
            if best is None or sample[1] < best[1]:
                best = sample
        assert best is not None
        self.offset_s, self.rtt_s = best
        self._synced_at = self._clock()
        if abs(self.offset_s) > 1.0:
            _log.info("robot_server clock is %+.3f s from ours (RTT %.0f ms); frame ages are corrected",
                      self.offset_s, self.rtt_s * 1000.0)
        return self.offset_s

    def grab(self) -> Frame:
        """The robot server's latest frame, decoded, its capture time on our clock.

        ``robot_server.py`` replies with ``age_s`` (the frame's age on its own
        monotonic clock when it built the reply): then no clock is compared
        at all, and the capture time is put at ``request sent - age_s`` -- the
        oldest it can be, so a late reply counts as staleness rather than
        hiding it. ``max_age_s`` is forwarded so the robot server refuses a
        stalled camera's last frame itself. Only a reply without ``age_s``
        (an older server) falls back to ``t_capture`` and the ping clock
        offset (synced before the first grab and every ``resync_s`` until the
        server is seen to report ``age_s``; never for ``127.0.0.1``).
        """
        if self._reports_age is not True and (
            self.offset_s is None or self._clock() - self._synced_at > self.resync_s
        ):
            self.sync_clock()  # before the grab, so the pings' time is not counted as frame age
        data: dict[str, Any] = {}
        if self.quality is not None:
            data["quality"] = self.quality
        if self.max_age_s is not None:
            data["max_age_s"] = self.max_age_s
        sent = self._clock()
        try:
            reply = self._call("get_frame", data)
        except FrameUnavailable as exc:
            if "stale" in str(exc).lower():
                raise StaleFrameError(str(exc)) from exc
            raise
        received = self._clock()
        payload = reply.get("jpeg")
        if not isinstance(payload, (bytes, bytearray)):
            raise FrameUnavailable("get_frame reply carries no JPEG bytes")
        try:
            rgb = decode_jpeg(bytes(payload))
        except (ValueError, RuntimeError) as exc:
            raise FrameUnavailable(f"get_frame JPEG undecodable: {exc}") from exc
        age = reply.get("age_s")
        self._reports_age = age is not None
        if age is not None:
            t_capture = sent - max(0.0, float(age))
        else:
            t_server = float(reply.get("t_capture", received + (self.offset_s or 0.0)))
            t_capture = t_server - float(self.offset_s or 0.0)
        return Frame(rgb=rgb, seq=int(reply.get("seq", -1)), t_capture=t_capture, received=received)

    def close(self) -> None:
        ctx = self._ctx.pop("ctx", None)
        if ctx is not None:
            ctx.term()


class FramePipeline:
    """A model backend behind fresh frames and the frame vote -- what a model ``detect`` request runs.

    Per request: refuse the newest frame if it is older than
    ``max_frame_age_s`` (after ``fresh_attempts`` grabs), run the backend on
    it, and while any label is still undecided grab the next *distinct*
    frame (new ``seq``) and run again -- normally two inferences, three when
    the first two disagree, one when frames from the last
    ``history_max_age_s`` are still in the vote and agree with this
    request's frame (an earlier request's frames add votes but never confirm
    a box on their own: every reported box includes a frame grabbed by this
    request, so an object moved since is reported where it is now or not at
    all). It stops early rather than
    run past ``request_budget_s`` (the laptop's detect timeout is 5 s), and
    an undecided label is then not reported: a miss, never a phantom.
    """

    def __init__(
        self,
        backend: Any,
        source: Any,
        voter: FrameVoter | None = None,
        max_frame_age_s: float = 0.5,
        history_max_age_s: float = 2.0,
        new_frame_wait_s: float = 0.5,
        request_budget_s: float = 4.0,
        fresh_attempts: int = 2,
        poll_s: float = 0.02,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_frame_age_s <= 0 or new_frame_wait_s < 0 or request_budget_s <= 0 or fresh_attempts < 1:
            raise ValueError("max_frame_age_s/request_budget_s must be > 0, new_frame_wait_s >= 0, "
                             "fresh_attempts >= 1")
        self.backend = backend
        self.source = source
        self.voter = voter if voter is not None else FrameVoter()
        self.max_frame_age_s = float(max_frame_age_s)
        self.history_max_age_s = float(history_max_age_s)
        self.new_frame_wait_s = float(new_frame_wait_s)
        self.request_budget_s = float(request_budget_s)
        self.fresh_attempts = int(fresh_attempts)
        self.poll_s = float(poll_s)
        self._clock = clock
        self._sleep = sleep
        self._vocab: tuple[str, ...] | None = None
        self._last_inference_s = 0.0
        self.inferences = 0

    def _fresh(self, frame: Frame) -> bool:
        return self._clock() - frame.t_capture <= self.max_frame_age_s

    def _first_frame(self) -> Frame:
        last_age = 0.0
        for attempt in range(self.fresh_attempts):
            frame = self.source.grab()
            if self._fresh(frame):
                return frame
            last_age = self._clock() - frame.t_capture
            if attempt + 1 < self.fresh_attempts:
                self._sleep(self.poll_s)
        raise StaleFrameError(
            f"the newest frame from robot_server is {last_age:.2f} s old (max {self.max_frame_age_s:.2f} s, "
            f"{self.fresh_attempts} grabs): the webcam stalled or its replies are late; refusing to detect on it"
        )

    def _next_frame(self, after: set[int], deadline: float) -> Frame | None:
        while True:
            try:
                frame = self.source.grab()
            except FrameUnavailable as exc:
                _log.warning("extra frame for the vote unavailable (%s); deciding with what we have", exc)
                return None
            if frame.seq not in after and not self.voter.has_frame(frame.seq) and self._fresh(frame):
                return frame
            if self._clock() >= deadline:
                return None
            self._sleep(self.poll_s)

    def _run(self, frame: Frame, labels: Sequence[str]) -> None:
        started = time.perf_counter()
        objects = self.backend.detect(frame.rgb, labels, 0.0)
        self._last_inference_s = time.perf_counter() - started
        self.inferences += 1
        self.voter.record(frame.seq, [_clean_object(o) for o in objects], frame.t_capture,
                          getattr(self.backend, "last_view", None))

    def detect(self, labels: Sequence[str], min_score: float = 0.0) -> dict[str, Any]:
        """``{width, height, objects, frames}`` for one request."""
        start = self._clock()
        vocabulary = tuple(sorted({str(l).strip().lower() for l in labels if str(l).strip()}))
        if vocabulary != self._vocab:
            self.voter.reset()
            self._vocab = vocabulary
        self.voter.prune(start - self.history_max_age_s)
        frame = self._first_frame()
        width, height = frame.size
        newest_age = self._clock() - frame.t_capture
        seen = {frame.seq}
        ran = 0
        if not self.voter.has_frame(frame.seq):
            self._run(frame, vocabulary)
            ran += 1
        confirmed, undecided = self.voter.decide(min_score, current=seen)
        stopped = ""
        while undecided:
            elapsed = self._clock() - start
            if elapsed + 1.1 * self._last_inference_s > self.request_budget_s:
                stopped = f"request budget {self.request_budget_s:.1f} s"
                break
            nxt = self._next_frame(seen, min(start + self.request_budget_s, self._clock() + self.new_frame_wait_s))
            if nxt is None:
                stopped = "no new frame"
                break
            seen.add(nxt.seq)
            self._run(nxt, vocabulary)
            ran += 1
            confirmed, undecided = self.voter.decide(min_score, current=seen)
        if undecided:
            _log.info("not reporting %s: undecided after %d frame(s) (%s)", sorted(undecided), len(self.voter), stopped)
        return {
            "width": width,
            "height": height,
            "objects": confirmed,
            "frames": {
                "newest_seq": frame.seq,
                "newest_age_s": round(newest_age, 4),
                "voted_over": self.voter.frame_ids,
                "inferences": ran,
                "undecided": sorted(undecided),
                "elapsed_s": round(self._clock() - start, 4),
            },
        }


class NanoOwlDetector:
    """OWL-ViT B/32 through NanoOWL's TensorRT image encoder (Jetson only).

    **NOT THE SIMULATION'S MODEL; DO NOT DEPLOY.** The hardware lane runs the
    sim's own detector, Florence-2-base (``scripts/serve_detector.py
    --backend florence-onnx``). Kept for reference; untested on hardware; its
    voter still counts one frame per request (the camera loop has no seq).

    ``nanoowl`` is imported lazily on first :meth:`detect`, so this module
    imports anywhere. Text embeddings for a vocabulary are computed once and
    cached per label set; per-label thresholds default to ``default_threshold``
    and are overridden by ``thresholds``; the result passes through a
    :class:`FrameVoter` (3 frames, 2 required by default).
    """

    name = "nanoowl"

    def __init__(
        self,
        model: str = "google/owlvit-base-patch32",
        image_encoder_engine: str | None = "/opt/nanoowl/data/owl_image_encoder_patch32.engine",
        synonyms: Mapping[str, Sequence[str]] | None = None,
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = 0.1,
        vote_window: int = 3,
        vote_required: int = 2,
    ) -> None:
        self.model = model
        self.image_encoder_engine = image_encoder_engine
        self.synonyms: dict[str, tuple[str, ...]] = dict(DEFAULT_SYNONYMS)
        for label, phrases in (synonyms or {}).items():
            self.synonyms[label.strip().lower()] = tuple(str(p) for p in phrases)
        self.thresholds = {k.strip().lower(): float(v) for k, v in (thresholds or {}).items()}
        self.default_threshold = float(default_threshold)
        self.voter = FrameVoter(vote_window, vote_required)
        self._predictor: Any = None
        self._vocab_key: tuple[str, ...] | None = None
        self._texts: list[str] = []
        self._text_labels: list[str] = []
        self._text_encodings: Any = None

    def phrases_for(self, label: str) -> tuple[str, ...]:
        """The 2-3 phrasings encoded for ``label`` (``"a <label>"`` when unknown)."""
        key = label.strip().lower()
        return self.synonyms.get(key, (f"a {key}",))

    def threshold_for(self, label: str, min_score: float) -> float:
        return max(float(min_score), self.thresholds.get(label.strip().lower(), self.default_threshold))

    def _ensure_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        try:
            from nanoowl.owl_predictor import OwlPredictor  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "nanoowl is not importable. This backend runs inside the "
                "jetson-containers image: `jetson-containers run $(autotag nanoowl) "
                "python3 /work/jetson/detector_service.py --backend nanoowl` "
                "(dustynv/nanoowl:r36.4.0). On the laptop use "
                "`scripts/serve_detector.py --backend florence|scripted` instead."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.image_encoder_engine:
            kwargs["image_encoder_engine"] = self.image_encoder_engine
        self._predictor = OwlPredictor(self.model, **kwargs)
        _log.info("NanoOWL loaded: %s (engine=%s)", self.model, self.image_encoder_engine)
        return self._predictor

    def _ensure_vocabulary(self, labels: Sequence[str]) -> None:
        key = tuple(sorted({l.strip().lower() for l in labels if l.strip()}))
        if key == self._vocab_key and self._text_encodings is not None:
            return
        texts: list[str] = []
        text_labels: list[str] = []
        for label in key:
            for phrase in self.phrases_for(label):
                texts.append(phrase)
                text_labels.append(label)
        predictor = self._ensure_predictor()
        self._text_encodings = predictor.encode_text(texts)
        self._texts, self._text_labels, self._vocab_key = texts, text_labels, key
        self.voter.reset()
        _log.info("Encoded %d phrases for %d labels", len(texts), len(key))

    def detect(
        self,
        rgb: NDArray[np.uint8] | None,
        labels: Sequence[str],
        min_score: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Run OWL-ViT on one RGB frame and return the temporally voted boxes."""
        if rgb is None:
            raise RuntimeError("NanoOwlDetector needs a frame: send 'jpeg' or run with a camera loop")
        self._ensure_vocabulary(labels)
        if not self._texts:
            return []
        from PIL import Image  # noqa: PLC0415

        predictor = self._ensure_predictor()
        image = Image.fromarray(np.ascontiguousarray(rgb))
        floor = min(self.threshold_for(l, min_score) for l in self._vocab_key or ())
        output = predictor.predict(
            image=image,
            text=self._texts,
            text_encodings=self._text_encodings,
            threshold=floor,
            pad_square=False,
        )
        raw: list[dict[str, Any]] = []
        for idx, score, box in zip(
            _to_list(output.labels), _to_list(output.scores), _to_list(output.boxes)
        ):
            label = self._text_labels[int(idx)]
            conf = float(score)
            if conf < self.threshold_for(label, min_score):
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
            raw.append({"label": label, "confidence": conf, "bbox_px": [x0, y0, x1, y1]})
        return self.voter.update(raw)


def _to_list(x: Any) -> list[Any]:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x)


class CameraLoop:
    """Background V4L2 grabber that keeps only the latest RGB frame.

    Used as ``frame_provider`` so a ``detect`` request without ``jpeg`` runs on
    the newest webcam frame instead of a stale buffered one.
    """

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480) -> None:
        self.source = source
        self.width, self.height = int(width), int(height)
        self._latest: NDArray[np.uint8] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "CameraLoop":
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for the camera loop (pip install opencv-python)") from exc
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.source!r}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def run() -> None:
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                with self._lock:
                    self._latest = rgb
            cap.release()

        self._thread = threading.Thread(target=run, name="camera-loop", daemon=True)
        self._thread.start()
        return self

    def latest(self) -> NDArray[np.uint8] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def decode_jpeg(data: bytes) -> NDArray[np.uint8]:
    """JPEG bytes -> RGB uint8 array (OpenCV decodes BGR; flipped here, once).

    Pillow is the fallback when opencv is absent (the Florence ONNX venv on
    the Jetson has Pillow for preprocessing anyway).
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        cv2 = None
    if cv2 is not None:
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode JPEG payload")
        return np.ascontiguousarray(bgr[:, :, ::-1])
    try:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("opencv-python or pillow is required to decode JPEG frames") from exc
    try:
        with Image.open(io.BytesIO(bytes(data))) as image:
            return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    except Exception as exc:  # noqa: BLE001 - PIL raises several types for junk
        raise ValueError(f"could not decode JPEG payload: {exc}") from exc


class DetectorServer:
    """Threaded TCP newline-JSON server around one detector backend.

    ``backend`` implements ``detect(rgb, labels, min_score) -> list[dict]`` and
    has a ``name``. ``frame_provider`` is a zero-argument callable returning the
    latest RGB frame (or ``None``) for requests that carry no ``jpeg``.

    With a ``pipeline`` (:class:`FramePipeline`, the model backends) every
    ``detect`` runs through it instead: frames only from robot_server, age-
    checked and voted, and a request carrying ``jpeg`` is refused.
    """

    def __init__(
        self,
        host: str,
        port: int,
        backend: Any,
        frame_provider: Callable[[], NDArray[np.uint8] | None] | None = None,
        pipeline: FramePipeline | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.backend = backend
        self.frame_provider = frame_provider
        self.pipeline = pipeline
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.requests_served = 0

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "name", type(self.backend).__name__.lower()))

    @property
    def bound_port(self) -> int:
        """Port actually bound (differs from ``port`` when it was 0)."""
        if self._server is None:
            return self.port
        return int(self._server.server_address[1])

    # -- request handling (pure, testable) ---------------------------------

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Turn one decoded request into its reply dict. Never raises."""
        try:
            cmd = str(request.get("cmd", "")).lower()
            if cmd == "ping":
                return {"ok": True, "backend": self.backend_name, "t": time.time()}
            if cmd == "detect":
                return self._detect(request)
            return {"ok": False, "error": f"unknown cmd {cmd!r}; expected 'ping' or 'detect'"}
        except FrameUnavailable as exc:  # expected on a stalled camera: no traceback spam
            _log.warning("detect refused: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - the wire must always get a reply
            _log.exception("detect failed")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _detect(self, request: Mapping[str, Any]) -> dict[str, Any]:
        labels = [str(l) for l in request.get("labels", [])]
        min_score = float(request.get("min_score", 0.0))
        jpeg_b64 = request.get("jpeg")
        if self.pipeline is not None:
            if jpeg_b64:
                return {"ok": False, "error": (
                    f"the {self.backend_name} detector takes frames only from robot_server get_frame "
                    "(--jetson HOST:PORT); send detect without 'jpeg'")}
            with self._lock:
                result = self.pipeline.detect(labels, min_score)
            self.requests_served += 1
            return {
                "ok": True,
                "t": time.time(),
                "backend": self.backend_name,
                "width": int(result["width"]),
                "height": int(result["height"]),
                "objects": [_clean_object(o) for o in result["objects"]],
                "frames": result.get("frames", {}),
            }
        rgb: NDArray[np.uint8] | None = None
        if jpeg_b64:
            rgb = decode_jpeg(base64.b64decode(jpeg_b64))
        elif self.frame_provider is not None:
            rgb = self.frame_provider()
        if rgb is not None:
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
        else:
            width, height = tuple(getattr(self.backend, "image_size", (0, 0)))
        with self._lock:
            # Backends with temporal state (voting) are not re-entrant.
            objects = self.backend.detect(rgb, labels, min_score)
        self.requests_served += 1
        return {
            "ok": True,
            "t": time.time(),
            "backend": self.backend_name,
            "width": width,
            "height": height,
            "objects": [_clean_object(o) for o in objects],
        }

    # -- lifecycle -----------------------------------------------------------

    def _bind(self, port: int | None) -> None:
        if self._server is not None:
            return
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                for raw in self.rfile:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        request = json.loads(line)
                        if not isinstance(request, dict):
                            raise ValueError("request must be a JSON object")
                    except ValueError as exc:
                        reply: dict[str, Any] = {"ok": False, "error": f"bad request: {exc}"}
                    else:
                        reply = owner.handle_request(request)
                    self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
                    self.wfile.flush()

        class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((self.host, self.port if port is None else int(port)), Handler)
        _log.info("Detector server (%s) listening on %s:%d", self.backend_name, self.host, self.bound_port)

    def serve_forever(self) -> None:
        """Bind (if needed) and block serving requests until :meth:`stop`."""
        self._bind(None)
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.2)

    def serve_in_thread(self, port: int | None = None) -> tuple["DetectorServer", int]:
        """Bind and serve on a daemon thread; return ``(self, bound_port)``.

        ``port=0`` picks a free ephemeral port (what tests want); ``None`` keeps
        the constructor's port.
        """
        self._bind(port)
        self._thread = threading.Thread(target=self.serve_forever, name="detector-server", daemon=True)
        self._thread.start()
        return self, self.bound_port

    def stop(self) -> None:
        """Stop serving and release the socket. Safe to call twice."""
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "DetectorServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _clean_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    box = obj.get("bbox_px", obj.get("bbox"))
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    return {
        "label": str(obj.get("label", "")),
        "confidence": float(obj.get("confidence", 0.0)),
        "bbox_px": [x0, y0, x1, y1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by this CLI and ``scripts/serve_detector.py``."""
    parser.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 on the Jetson)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--scene",
        default="marker:0.18,0.05 bowl:0.15,-0.12",
        help='scripted scene, e.g. "marker:0.18,0.05 bowl:0.15,-0.12" (x,y[,z] metres)',
    )
    parser.add_argument(
        "--print-homography",
        action="store_true",
        help="print the synthetic camera's pixel->table homography (exterior_camera.homography) and exit",
    )
    parser.add_argument(
        "--world-from",
        default=None,
        metavar="HOST:PORT",
        help="(scripted) read the live scene from a robot_server.py --driver fake --fake-world process; "
        "--scene is then only the fallback while it is unreachable",
    )
    parser.add_argument(
        "--hide-above",
        type=float,
        default=None,
        help="(scripted) LEGACY: stop reporting objects whose base z exceeds this, e.g. "
        f"{HIDE_ABOVE_Z}. Default: honest mode -- lifted objects stay visible and only the "
        "gripper footprint (from the fake world's TCP) hides them",
    )
    parser.add_argument("--log-level", default="INFO")


def build_scripted(
    scene_text: str, world_from: str | None = None, hide_above_z: float | None = None
) -> ScriptedDetector:
    """The scripted backend used by both CLIs and the end-to-end tests.

    Honest mode unless ``hide_above_z`` is given; with ``world_from`` the
    gripper occluder follows the robot server's fake-world TCP.
    """
    scene: SceneSource = parse_scene(scene_text)
    if world_from:
        host, _, port_text = world_from.partition(":")
        scene = RobotWorldScene(host or "127.0.0.1", int(port_text or 5560), fallback=scene)
        _log.info("Scripted scene follows the fake world at %s", scene.endpoint)
    return ScriptedDetector(scene, SyntheticPinhole(), hide_above_z=hide_above_z)


def warm_up(
    backend: Any,
    labels: Sequence[str] | None = None,
    image_size: tuple[int, int] = (640, 480),
) -> float | None:
    """Run one detect on a black frame so the model is loaded before serving.

    NanoOWL (and Florence behind ``serve_detector.py``) load weights, build
    the TensorRT context and encode the vocabulary on the first ``detect``:
    10-40 s on the Orin. Paid here, at startup, it appears once in this log
    with its duration; paid on the first request, it looked like a dead
    service to the laptop (review P3). Returns the seconds taken, or ``None``
    when the backend failed -- logged, not raised, because the server must
    still come up and report the error per request.
    """
    width, height = (int(v) for v in image_size)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    vocabulary = list(labels) if labels is not None else list(DEFAULT_SYNONYMS)
    started = time.perf_counter()
    try:
        backend.detect(frame, vocabulary, 0.99)
    except Exception as exc:  # noqa: BLE001 - report and keep serving
        _log.error(
            "detector warm-up failed after %.1f s: %s: %s",
            time.perf_counter() - started, type(exc).__name__, exc,
        )
        return None
    finally:
        # The black frame must never vote (audit: it did, so the first real
        # observe reported nothing).
        voter = getattr(backend, "voter", None)
        if voter is not None and hasattr(voter, "reset"):
            voter.reset()
    elapsed = time.perf_counter() - started
    _log.info(
        "detector backend %s warm in %.1f s (dummy detect on a black %dx%d frame); ready",
        getattr(backend, "name", type(backend).__name__), elapsed, width, height,
    )
    return elapsed


def build_parser() -> argparse.ArgumentParser:
    """This file's CLI: the scripted fake by default; NanoOWL only on explicit request."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--backend", choices=("scripted", "nanoowl"), default="scripted",
                        help="scripted (default, the fake lane). nanoowl is NOT the sim model; do not deploy it: "
                             "the real detector is scripts/serve_detector.py --backend florence-onnx")
    parser.add_argument("--model", default="google/owlvit-base-patch32")
    parser.add_argument("--engine", default="/opt/nanoowl/data/owl_image_encoder_patch32.engine")
    parser.add_argument("--threshold", type=float, default=0.1, help="default per-label score threshold")
    parser.add_argument("--camera", default="0", help="V4L2 index or device for the continuous loop")
    parser.add_argument("--no-camera", action="store_true", help="only serve requests that carry a jpeg")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    if args.print_homography:
        print(format_homography(SyntheticPinhole().homography_pixel_to_table()))
        return 0

    loop: CameraLoop | None = None
    if args.backend == "scripted":
        backend: Any = build_scripted(args.scene, args.world_from, args.hide_above)
        provider = None
    else:
        _log.warning("NanoOWL is NOT the simulation's model; do not deploy it. The hardware detector is "
                     "scripts/serve_detector.py --backend florence-onnx. NanoOWL also opens the webcam "
                     "robot_server.py already owns.")
        backend = NanoOwlDetector(model=args.model, image_encoder_engine=args.engine,
                                  default_threshold=args.threshold)
        provider = None
        if not args.no_camera:
            source: int | str = int(args.camera) if args.camera.isdigit() else args.camera
            loop = CameraLoop(source).start()
            provider = loop.latest

    if args.backend != "scripted":
        # Load the model now, not on the first student request (review P3).
        warm_up(backend)
    server = DetectorServer(args.host, args.port, backend, frame_provider=provider)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if loop is not None:
            loop.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

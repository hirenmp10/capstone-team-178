"""Phase 9: the overhead camera's calibration tool (``scripts/calibrate_camera.py``).

Everything runs on SYNTHETIC images of known cameras, so the numbers can be
checked against the truth the bench never has:

* a renderer that images the printed sheet through a pinhole with OpenCV's
  lens model (inverse-mapped per pixel, box-filtered, then blurred, noised
  and JPEG'd like a webcam frame), cross-checked against ``cv2.projectPoints``;
* intrinsics recovered within 2 % from 20 views, a bowed sheet refused,
  views captured while the board moves rejected, a smeared view dropped;
* the camera pose recovered within 5 mm / 1 deg from a sheet at a given
  robot-frame offset and yaw, and the homography that follows within 2 mm
  over the workspace;
* the whole chain through a real ``robot_server`` ``get_frame`` whose webcam
  drops reads and stalls;
* the YAML splice leaving every other line of ``configs/hardware.yaml``
  byte-identical;
* and the point of it all (audit F1): a config written by the tool makes
  the feedback-less grasp verdict reach ``carried`` through the production
  ``PlanarPerception``, where the shipped placeholder config cannot.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import struct
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

from mfw.config.schema import CameraConfig, ConfigError, load_config  # noqa: E402
from mfw.hardware.perception import PinholeModel, PlanarPerception  # noqa: E402

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_YAML = REPO_ROOT / "configs" / "hardware.yaml"


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load_script("calibrate_camera")
BOARD = cc.BoardSpec()

# A C270-class webcam at the robot server's 640x480, with a mild barrel lens.
W, H = 640, 480
K_TRUE = np.array([[812.0, 0.0, 322.5], [0.0, 809.0, 236.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([0.06, -0.15, 0.0008, -0.0006, 0.0])


# ----------------------------------------------------------------------
# synthetic imaging
# ----------------------------------------------------------------------


def _undistort_normalized(xd: np.ndarray, yd: np.ndarray, dist: np.ndarray, iters: int = 30):
    d = np.zeros(8)
    d[: len(dist)] = dist
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


_RAYS: dict[bytes, np.ndarray] = {}


def _pixel_rays(k: np.ndarray, dist: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Undistorted ``(x, y, 1)`` ray of every pixel centre; cached per camera (it is the slow part)."""
    key = np.asarray(k, dtype=np.float64).tobytes() + np.asarray(dist, dtype=np.float64).tobytes() + bytes(str(size), "ascii")
    if key not in _RAYS:
        w_img, h_img = size
        u, v = np.meshgrid(np.arange(w_img, dtype=np.float64), np.arange(h_img, dtype=np.float64))
        x, y = _undistort_normalized((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.asarray(dist, dtype=np.float64))
        _RAYS[key] = np.stack([x, y, np.ones_like(x)], axis=-1)
    return _RAYS[key]


def _tri(u: np.ndarray) -> np.ndarray:
    """Integral of the square wave that is +1 on [0, 1) (mod 2) and -1 on [1, 2)."""
    return 1.0 - np.abs(np.mod(u, 2.0) - 1.0)


def _filtered_square(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-6)
    return (_tri(u + w / 2) - _tri(u - w / 2)) / w


def _filtered_box(u: np.ndarray, w: np.ndarray, a: float, b: float) -> np.ndarray:
    w = np.maximum(w, 1e-6)
    return np.clip((np.minimum(b, u + w / 2) - np.maximum(a, u - w / 2)) / w, 0.0, 1.0)


def render_sheet(
    k: np.ndarray, dist: np.ndarray, r_cb: np.ndarray, t_cb: np.ndarray, *,
    spec=BOARD, size=(W, H), blur_sigma: float = 0.7, noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None, bow_m: float = 0.0, jpeg_quality: int | None = 90,
    paper: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """BGR image of the printed sheet (board frame ``r_cb, t_cb`` -> camera) through ``k, dist``.

    Per pixel: undistort, intersect the ray with the sheet, box-filter the
    checker over the pixel's footprint on the sheet (so edges are
    anti-aliased exactly), then lens blur, sensor noise and JPEG. ``bow_m``
    bends the sheet into a cylinder of that sag across its long side (paper
    that is not taped flat).
    """
    w_img, h_img = size
    s = spec.square
    lay = cc.board_layout(spec, dpi=300.0)
    if paper is None:
        x_l = -lay["u0"] / lay["ppm"]
        x_r = (lay["width"] - lay["u0"]) / lay["ppm"]
        y_t = lay["v0"] / lay["ppm"]
        y_b = (lay["v0"] - lay["height"]) / lay["ppm"]
        paper = (x_l, x_r, y_b, y_t)
    rays = _pixel_rays(k, dist, size) @ r_cb  # camera -> board frame directions
    c_b = -r_cb.T @ t_cb
    xm = spec.cols * s / 2.0
    half = (spec.cols + 1) * s / 2.0

    def surface(px: np.ndarray) -> np.ndarray:
        if bow_m == 0.0:
            return np.zeros_like(px)
        return bow_m * (1.0 - ((px - xm) / half) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (0.0 - c_b[2]) / rays[..., 2]
        for _ in range(4 if bow_m else 1):
            px = c_b[0] + lam * rays[..., 0]
            lam = (surface(px) - c_b[2]) / rays[..., 2]
        bx = c_b[0] + lam * rays[..., 0]
        by = c_b[1] + lam * rays[..., 1]
    valid = np.isfinite(lam) & (lam > 0)
    bx = np.where(valid, bx, 1e3)
    by = np.where(valid, by, 1e3)
    gxu, gxv = np.gradient(bx, axis=1), np.gradient(bx, axis=0)
    gyu, gyv = np.gradient(by, axis=1), np.gradient(by, axis=0)
    fw_x = (np.abs(gxu) + np.abs(gxv)) / s
    fw_y = (np.abs(gyu) + np.abs(gyv)) / s
    ux, uy = bx / s, by / s
    checker = 0.5 * (1.0 + _filtered_square(ux, fw_x) * _filtered_square(uy, fw_y))
    inside = _filtered_box(ux, fw_x, -1.0, spec.cols) * _filtered_box(uy, fw_y, -1.0, spec.rows)
    paper_cov = (_filtered_box(bx / s, fw_x, paper[0] / s, paper[1] / s)
                 * _filtered_box(by / s, fw_y, paper[2] / s, paper[3] / s))
    table, white, black = 70.0, 225.0, 30.0
    img = table + (white - table) * paper_cov - (white - black) * checker * inside
    img = np.where(valid, img, table)
    img = img.astype(np.float32)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    rng = rng if rng is not None else np.random.default_rng(0)
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    gray = np.clip(np.round(img), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        assert ok
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return bgr


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


FACE_CAMERA = np.diag([1.0, -1.0, -1.0])
"""Board -> camera rotation of a sheet held square to the camera, printed side toward it."""


def board_pose(centre_cam, tilt_x=0.0, tilt_y=0.0, roll=0.0, spec=BOARD):
    """``(r_cb, t_cb)`` for a sheet whose checker centre sits at ``centre_cam`` (camera frame)."""
    r = _rot("z", roll) @ _rot("x", tilt_x) @ _rot("y", tilt_y) @ FACE_CAMERA
    mid = np.array([(spec.cols - 1) * spec.square / 2.0, (spec.rows - 1) * spec.square / 2.0, 0.0])
    return r, np.asarray(centre_cam, dtype=np.float64) - r @ mid


def truth_corners(k, dist, r_cb, t_cb, spec=BOARD) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(r_cb)
    uv, _ = cv2.projectPoints(spec.object_points().reshape(-1, 1, 3), rvec, t_cb, k, dist)
    return uv.reshape(-1, 2)


def _in_view(uv: np.ndarray, margin: float = 25.0) -> bool:
    return bool(np.all((uv[:, 0] > margin) & (uv[:, 0] < W - margin) & (uv[:, 1] > margin) & (uv[:, 1] < H - margin)))


def calibration_poses(n: int, seed: int = 1, k=K_TRUE, dist=D_TRUE):
    """``n`` varied, fully visible sheet poses: near/far, tilted, spread over the image."""
    rng = np.random.default_rng(seed)
    poses = []
    while len(poses) < n:
        z = rng.uniform(0.38, 0.62)
        x = rng.uniform(-0.16, 0.16) * z / 0.5
        y = rng.uniform(-0.11, 0.11) * z / 0.5
        r, t = board_pose((x, y, z), rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-25, 25))
        if _in_view(truth_corners(k, dist, r, t), margin=30.0):
            poses.append((r, t))
    return poses


@pytest.fixture(scope="module")
def calibration_views():
    """20 rendered views of a C270-class camera and the corners detected in them."""
    poses = calibration_poses(20)
    rng = np.random.default_rng(7)
    images = [render_sheet(K_TRUE, D_TRUE, r, t, rng=rng) for r, t in poses]
    detections = [cc.find_board_corners(img, BOARD) for img in images]
    return poses, images, detections


# ----------------------------------------------------------------------
# the renderer itself (so the other numbers mean something)
# ----------------------------------------------------------------------


class TestSyntheticImaging:
    def test_undistort_inverts_the_lens_model(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-0.4, 0.4, (200, 2))
        xd = cv2.projectPoints(np.c_[pts, np.ones(200)].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                               np.eye(3), D_TRUE)[0].reshape(-1, 2)
        x, y = _undistort_normalized(xd[:, 0], xd[:, 1], D_TRUE)
        assert np.max(np.abs(np.c_[x, y] - pts)) < 1e-9

    def test_detected_corners_match_projectpoints(self, calibration_views):
        poses, _images, detections = calibration_views
        errs = []
        for (r, t), det in zip(poses, detections):
            assert det is not None
            errs.append(np.linalg.norm(det.corners - truth_corners(K_TRUE, D_TRUE, r, t), axis=1))
        errs = np.concatenate(errs)
        # Blur + noise + JPEG, found by the production detector: sub-pixel.
        assert float(np.sqrt(np.mean(errs ** 2))) < 0.15, float(np.sqrt(np.mean(errs ** 2)))
        assert float(errs.max()) < 0.6


# ----------------------------------------------------------------------
# the board and its detection
# ----------------------------------------------------------------------


class TestBoard:
    @pytest.mark.parametrize("cols,rows", [(7, 7), (8, 6), (7, 5)])
    def test_boards_symmetric_under_a_half_turn_are_refused(self, cols, rows):
        with pytest.raises(ValueError, match="half turn"):
            cc.BoardSpec(cols=cols, rows=rows).validate()

    def test_make_board_writes_a4_at_300_dpi(self, tmp_path):
        out = tmp_path / "board.png"
        assert cc.main(["make-board", "--out", str(out)]) == 0
        data = out.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (3508, 2480)  # A4 landscape at 300 dpi
        i = data.index(b"pHYs")
        ppx, ppy, unit = struct.unpack(">IIB", data[i + 4:i + 13])
        assert (ppx, ppy, unit) == (11811, 11811, 1)  # 300 dpi, so "actual size" prints true
        page = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert page.shape == (2480, 3508)
        # The printed squares are 25 mm: detect the board on the page itself.
        small = cv2.resize(page, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        det = cc.find_board_corners(small, BOARD)
        assert det is not None and det.origin_resolved
        # Back to full-page pixel centres (a 4x4 INTER_AREA block's centre is at +1.5).
        g = det.corners.reshape(BOARD.rows, BOARD.cols, 2) * 4.0 + 1.5
        spacing_mm = np.linalg.norm(np.diff(g, axis=1), axis=2).mean() / (300 / 25.4)
        assert spacing_mm == pytest.approx(25.0, abs=0.05)
        # Canonical order: +x runs right on the page, +y runs UP the page.
        assert g[0, -1, 0] > g[0, 0, 0] and g[-1, 0, 1] < g[0, 0, 1]
        lay = cc.board_layout(BOARD)  # u0/v0 are edge coordinates: pixel centres sit at -0.5
        assert g[0, 0] == pytest.approx([lay["u0"] - 0.5, lay["v0"] - 0.5], abs=0.6)
        # The 100 mm scale bar the operator is told to measure is really printed,
        # 100 mm long, inside the printable area, clear of the squares.
        mm = lay["ppm"] / 1000.0
        sq = BOARD.square * lay["ppm"]
        board_top = int(round(lay["v0"] - BOARD.rows * sq))
        best = 0
        for row in page[:board_top]:
            dark = np.concatenate([[False], row < 128, [False]]).astype(np.int8)
            edges = np.flatnonzero(np.diff(dark))
            if len(edges):
                best = max(best, int(np.max(edges[1::2] - edges[0::2])))
        assert best / mm == pytest.approx(100.0, abs=0.5)
        ink = np.argwhere(page < 128)
        margin = int(4 * mm)
        assert ink[:, 0].min() >= margin and ink[:, 1].min() >= margin
        assert ink[:, 0].max() < page.shape[0] - margin and ink[:, 1].max() < page.shape[1] - margin
        above = page[board_top - int(8.5 * mm):board_top - 2]
        assert above.min() > 128, "annotation inside the board's quiet zone"

    def test_printed_page_through_a_camera_resolves_the_origin(self, tmp_path):
        """The real PNG (title, arrows, scale bar in the margins) warped into a
        camera view: still detected, and the origin corner is the marked one."""
        page = cc.make_board_image(BOARD, dpi=150.0)
        # ~4 texels per camera pixel: pre-filter so the bilinear warp does not alias.
        page = cv2.GaussianBlur(page.astype(np.float32), (0, 0), 1.8)
        lay = cc.board_layout(BOARD, dpi=150.0)
        for roll in (0.0, 180.0, 90.0):
            r, t = board_pose((0.01, -0.005, 0.5), 10.0, -12.0, roll)
            # Map each camera pixel to a page pixel (ideal lens for the warp).
            u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            rays = np.stack([(u - K_TRUE[0, 2]) / K_TRUE[0, 0], (v - K_TRUE[1, 2]) / K_TRUE[1, 1],
                             np.ones_like(u)], axis=-1) @ r
            c_b = -r.T @ t
            lam = -c_b[2] / rays[..., 2]
            bx, by = c_b[0] + lam * rays[..., 0], c_b[1] + lam * rays[..., 1]
            map_u = (lay["u0"] + bx * lay["ppm"]).astype(np.float32)
            map_v = (lay["v0"] - by * lay["ppm"]).astype(np.float32)
            img = cv2.remap(page, map_u, map_v, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=70)
            img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.6), 0, 255).astype(np.uint8)
            det = cc.find_board_corners(img, BOARD)
            assert det is not None, roll
            assert det.origin_resolved, (roll, det.contrast)
            truth = truth_corners(K_TRUE, np.zeros(5), r, t)
            assert np.max(np.linalg.norm(det.corners - truth, axis=1)) < 0.6, roll

    @pytest.mark.parametrize("roll", [0.0, 90.0, 180.0, 270.0, 150.0])
    def test_corner_order_is_canonical_whatever_the_sheet_rotation(self, roll):
        r, t = board_pose((0.0, 0.0, 0.5), 15.0, 10.0, roll)
        det = cc.find_board_corners(render_sheet(K_TRUE, D_TRUE, r, t), BOARD)
        assert det is not None and det.origin_resolved
        assert np.max(np.linalg.norm(det.corners - truth_corners(K_TRUE, D_TRUE, r, t), axis=1)) < 0.5

    def test_sheet_partly_out_of_view_is_not_detected(self):
        r, t = board_pose((0.20, 0.0, 0.45), 0.0, 0.0, 0.0)
        assert cc.find_board_corners(render_sheet(K_TRUE, D_TRUE, r, t), BOARD) is None

    def test_origin_square_hidden_leaves_the_order_unresolved(self):
        """A hand over the origin corner: fine for intrinsics, not for pose."""
        r, t = board_pose((0.0, 0.0, 0.5), 5.0, 5.0, 0.0)
        img = render_sheet(K_TRUE, D_TRUE, r, t)
        truth = truth_corners(K_TRUE, D_TRUE, r, t)
        # Paint the origin's outside square light-grey (a thumb on it).
        g = truth.reshape(BOARD.rows, BOARD.cols, 2)
        o = g[0, 0] - (g[0, 1] - g[0, 0]) * 0.5 - (g[1, 0] - g[0, 0]) * 0.5
        thumb = int(0.55 * np.linalg.norm(g[0, 1] - g[0, 0]))  # ~20 mm across
        cv2.circle(img, (int(o[0]), int(o[1])), thumb, (225, 225, 225), -1)
        det = cc.find_board_corners(img, BOARD)
        assert det is not None and not det.origin_resolved


# ----------------------------------------------------------------------
# intrinsics
# ----------------------------------------------------------------------


class TestIntrinsics:
    def test_recovers_the_camera_within_two_percent(self, calibration_views):
        _poses, _images, detections = calibration_views
        res = cc.calibrate_intrinsics([d.corners for d in detections], (W, H), BOARD)
        for got, want in ((res.fx, K_TRUE[0, 0]), (res.fy, K_TRUE[1, 1]),
                          (res.cx, K_TRUE[0, 2]), (res.cy, K_TRUE[1, 2])):
            assert abs(got / want - 1.0) < 0.02, (got, want)
        # In fact far better than the requirement.
        assert abs(res.fx / K_TRUE[0, 0] - 1.0) < 0.005
        assert res.rms_px < 0.3
        assert len(res.distortion) == 4
        assert res.distortion[0] == pytest.approx(D_TRUE[0], abs=0.02)
        # The recovered lens reprojects the true corners of an unseen pose within a pixel.
        r, t = board_pose((0.05, 0.03, 0.55), 20.0, -20.0, 10.0)
        truth = truth_corners(K_TRUE, D_TRUE, r, t)
        rvec, _ = cv2.Rodrigues(r)
        mine, _ = cv2.projectPoints(BOARD.object_points().reshape(-1, 1, 3), rvec, t, res.camera_matrix,
                                    res.distortion)
        assert np.max(np.linalg.norm(mine.reshape(-1, 2) - truth, axis=1)) < 1.0

    def test_a_bowed_sheet_is_refused(self):
        """Paper not taped flat: the model cannot fit it and nothing is saved."""
        poses = calibration_poses(16, seed=5)
        rng = np.random.default_rng(11)
        views = []
        for r, t in poses:
            det = cc.find_board_corners(render_sheet(K_TRUE, D_TRUE, r, t, rng=rng, bow_m=0.012), BOARD)
            if det is not None:
                views.append(det.corners)
        assert len(views) >= 15
        with pytest.raises(cc.CalibrationError, match="exceeds 1.0 px"):
            cc.calibrate_intrinsics(views, (W, H), BOARD)

    def test_fewer_than_fifteen_views_are_refused(self, calibration_views):
        _poses, _images, detections = calibration_views
        with pytest.raises(cc.CalibrationError, match="need >= 15"):
            cc.calibrate_intrinsics([d.corners for d in detections[:14]], (W, H), BOARD)

    def test_a_view_smeared_by_motion_is_dropped(self, calibration_views):
        """A board that moved during the exposure: each corner smeared along
        the motion by a different amount (1.5 px RMS), depending on its edge
        directions. That view is dropped; the rest calibrate clean."""
        _poses, _images, detections = calibration_views
        views = [d.corners.copy() for d in detections]
        smear = np.random.default_rng(5).normal(0.0, 1.5, BOARD.count)[:, None] * np.array([[0.93, 0.37]])
        views[4] = views[4] + smear
        res = cc.calibrate_intrinsics(views, (W, H), BOARD)
        assert 4 in res.dropped and res.rms_px < 0.3
        assert abs(res.fx / K_TRUE[0, 0] - 1.0) < 0.005


class ScriptedSource:
    """Frames the way a hand-held board arrives: moving, then held, repeated
    frames, dropped reads and a frame of the wrong size."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.frames)

    def read(self):
        if self.i >= len(self.frames):
            return None
        f = self.frames[self.i]
        self.i += 1
        return f

    def close(self):
        pass


class TestViewCollection:
    def test_only_still_new_views_are_kept(self):
        rng = np.random.default_rng(2)
        poses = calibration_poses(3, seed=9)
        frames = []
        key = 0

        def add(img, same_key=False):
            nonlocal key
            if not same_key:
                key += 1
            frames.append(cc.Frame(image=img, key=("seq", key)))

        for n, (r, t) in enumerate(poses):
            # Moving in: three frames 4 mm apart each.
            for step in (3, 2, 1):
                add(render_sheet(K_TRUE, D_TRUE, r, t + np.array([0.004 * step, 0.0, 0.0]), rng=rng))
            frames.append(None)  # a dropped read
            still = render_sheet(K_TRUE, D_TRUE, r, t, rng=rng)
            add(still)
            add(still, same_key=True)  # the server handed out the same frame twice
            add(render_sheet(K_TRUE, D_TRUE, r, t, rng=rng))  # held: new noise, same pose
            add(render_sheet(K_TRUE, D_TRUE, r, t, rng=rng))  # still held: same view again
        add(np.zeros((240, 320, 3), dtype=np.uint8))  # wrong resolution
        collector = cc.collect_views(ScriptedSource(frames), BOARD, want=10, need_still=True)
        assert len(collector.views) == 3
        assert collector.counts["moving"] >= 3 * 3
        assert collector.counts["duplicate-frame"] == 3
        assert collector.counts["similar"] == 3
        assert collector.counts["size-mismatch"] == 1
        for view, (r, t) in zip(collector.views, poses):
            assert np.max(np.linalg.norm(view - truth_corners(K_TRUE, D_TRUE, r, t), axis=1)) < 0.6


# ----------------------------------------------------------------------
# pose
# ----------------------------------------------------------------------

CAM_POS = np.array([0.175, 0.012, 0.585])
CAM_LOOK = np.array([0.19, -0.004, 0.0])
CAM_UP = np.array([1.0, 0.03, 0.0])
SHEET_ORIGIN = (0.075, -0.115)
SHEET_YAW = 25.0


def robot_camera(position=CAM_POS, look_at=CAM_LOOK, up=CAM_UP):
    """``(r_rc, position)``: camera axes in the robot frame (PinholeModel's construction)."""
    m = PinholeModel(1, 1, 0, 0, position, look_at, up)
    return m.rotation, np.asarray(position, dtype=np.float64)


def sheet_extrinsics(r_rc, cam_pos, origin=SHEET_ORIGIN, yaw=SHEET_YAW, table_z=0.0):
    """Board -> camera ``(r_cb, t_cb)`` for a sheet lying on the table at ``origin``/``yaw``."""
    r_rb = _rot("z", yaw)
    t_rb = np.array([origin[0], origin[1], table_z])
    r_cb = r_rc.T @ r_rb
    t_cb = r_rc.T @ (t_rb - cam_pos)
    return r_cb, t_cb


def _angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


@pytest.fixture(scope="module")
def calibrated(calibration_views):
    _poses, _images, detections = calibration_views
    return cc.calibrate_intrinsics([d.corners for d in detections], (W, H), BOARD)


def _sheet_frames(n=6, seed=4, k=K_TRUE, dist=D_TRUE, **kw):
    r_rc, pos = robot_camera()
    r_cb, t_cb = sheet_extrinsics(r_rc, pos, **kw)
    rng = np.random.default_rng(seed)
    return [render_sheet(k, dist, r_cb, t_cb, rng=rng) for _ in range(n)]


class TestPose:
    def test_camera_pose_from_a_sheet_at_a_known_robot_offset(self, calibrated):
        frames = [cc.find_board_corners(img, BOARD) for img in _sheet_frames()]
        assert all(f is not None and f.origin_resolved for f in frames)
        pose = cc.estimate_sheet_pose([f.corners for f in frames], calibrated.camera_matrix, calibrated.distortion,
                                      BOARD, SHEET_ORIGIN, SHEET_YAW)
        r_rc, pos = robot_camera()
        assert np.linalg.norm(pose.position - pos) * 1000 < 5.0, pose.position
        assert _angle_deg(pose.rotation, r_rc) < 1.0
        # look_at / up rebuild the same rotation through the production model.
        rebuilt = PinholeModel(1, 1, 0, 0, pose.position, pose.look_at, pose.up).rotation
        assert np.allclose(rebuilt, pose.rotation, atol=1e-9)
        assert abs(pose.look_at[2]) < 1e-9  # on the table
        assert pose.rms_px < 0.3

    def test_workspace_homography_matches_the_true_camera(self, calibrated):
        frames = [cc.find_board_corners(img, BOARD).corners for img in _sheet_frames()]
        pose = cc.estimate_sheet_pose(frames, calibrated.camera_matrix, calibrated.distortion, BOARD,
                                      SHEET_ORIGIN, SHEET_YAW)
        lo, hi = (0.03, -0.27), (0.33, 0.27)
        hm, rms_mm, max_mm, n = cc.fit_workspace_homography(calibrated.camera_matrix, calibrated.distortion,
                                                           pose, lo, hi, (W, H))
        assert n > 1000 and max_mm < 1.5
        # Truth: raw pixels of table points through the TRUE camera -> H -> table.
        r_rc, pos = robot_camera()
        xs, ys = np.meshgrid(np.linspace(0.06, 0.30, 13), np.linspace(-0.2, 0.2, 17))
        pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
        rvec, _ = cv2.Rodrigues(r_rc.T)
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, -r_rc.T @ pos, K_TRUE, D_TRUE)
        uv = uv.reshape(-1, 2)
        ok = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
        xy = cc._apply_h(hm, uv[ok])
        err = np.linalg.norm(xy - pts[ok, :2], axis=1) * 1000
        assert ok.sum() > 150 and err.max() < 2.0, err.max()

    def test_one_frame_disturbed_by_the_arm_is_dropped(self, calibrated):
        frames = [cc.find_board_corners(img, BOARD).corners for img in _sheet_frames(n=5)]
        frames[2] = frames[2] + np.array([4.0, -2.0])  # sheet nudged for one frame
        pose = cc.estimate_sheet_pose(frames, calibrated.camera_matrix, calibrated.distortion, BOARD,
                                      SHEET_ORIGIN, SHEET_YAW)
        assert pose.frames_rejected == 1
        assert np.linalg.norm(pose.position - CAM_POS) * 1000 < 5.0

    def test_a_sheet_that_keeps_moving_is_refused(self, calibrated):
        frames = [cc.find_board_corners(img, BOARD).corners for img in _sheet_frames(n=4)]
        frames = [f + np.array([3.0 * i, 0.0]) for i, f in enumerate(frames)]
        with pytest.raises(cc.CalibrationError, match="moved between frames"):
            cc.estimate_sheet_pose(frames, calibrated.camera_matrix, calibrated.distortion, BOARD,
                                   SHEET_ORIGIN, SHEET_YAW)

    def test_frames_at_another_resolution_than_the_intrinsics_are_refused(self, tmp_path):
        """Intrinsics measured at 1280x720, frames arriving at 640x480. A plane
        alone cannot expose wrong intrinsics (the pose fit absorbs them), so the
        resolution itself is checked, and nothing is written."""
        cfg_path = tmp_path / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")
        cc.write_camera_values(cfg_path, {"resolution": [1280, 720], "fx": 1624.0, "fy": 1618.0, "cx": 645.0,
                                          "cy": 472.0, "distortion": [0.06, -0.15, 0.0, 0.0],
                                          "intrinsics_rms_px": 0.2})
        sheet = tmp_path / "sheet"
        sheet.mkdir()
        cv2.imwrite(str(sheet / "s0.png"), _sheet_frames(n=1)[0])
        before = cfg_path.read_bytes()
        assert cc.main(["pose", "--images", str(sheet), "--config", str(cfg_path), "--origin",
                        *map(str, SHEET_ORIGIN), "--yaw", str(SHEET_YAW)]) == 2
        assert cfg_path.read_bytes() == before


class _FakeTouchProbe:
    """What ``calibrate_table.TouchProbe.measure`` returns on a real arm: the
    TCP its OWN kinematics report once the jaw tip is on the corner. The arm
    believes it is ``bias`` further than it is (link lengths off), the
    operator lands within ~1 mm, and may skip a corner."""

    def __init__(self, truth_xy, bias=(0.0, 0.0), jitter_mm=0.8, skip=(), mistouch=None, seed=0):
        self.truth_xy = [np.asarray(v, dtype=np.float64) for v in truth_xy]
        self.bias = np.asarray(bias, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter_mm / 1000.0
        self.skip = set(skip)
        self.mistouch = mistouch
        self.closed = False
        self.asked = []

    def measure(self, index, uv):
        self.asked.append((index, uv))
        if index - 1 in self.skip:
            return None
        xy = self.truth_xy[index - 1] + self.bias + self.rng.normal(0.0, self.jitter, 2)
        if self.mistouch is not None and index - 1 == self.mistouch[0]:
            xy = xy + np.asarray(self.mistouch[1])
        return float(xy[0]), float(xy[1])

    def close(self):
        self.closed = True


class TestTouchPlacement:
    def test_rigid_fit_recovers_origin_and_yaw(self):
        board = cc.touch_corner_board_xy(BOARD)
        truth = cc.sheet_corners_robot(BOARD, SHEET_ORIGIN, SHEET_YAW)
        touched = np.array([truth[0, :2], truth[BOARD.cols - 1, :2], truth[-1, :2],
                            truth[(BOARD.rows - 1) * BOARD.cols, :2]])
        origin, yaw, worst = cc.sheet_placement_from_points(board, touched)
        assert origin == pytest.approx(SHEET_ORIGIN, abs=1e-12) and yaw == pytest.approx(SHEET_YAW, abs=1e-9)
        assert worst < 1e-9

    def _run(self, monkeypatch, **probe_kw):
        truth = cc.sheet_corners_robot(BOARD, SHEET_ORIGIN, SHEET_YAW)
        corners = [truth[0, :2], truth[BOARD.cols - 1, :2], truth[-1, :2], truth[(BOARD.rows - 1) * BOARD.cols, :2]]
        probe = _FakeTouchProbe(corners, **probe_kw)
        monkeypatch.setattr(cc, "_make_touch_probe", lambda jetson, config: probe)
        import argparse

        det = cc.find_board_corners(_sheet_frames(n=1)[0], BOARD)
        args = argparse.Namespace(jetson="127.0.0.1:5581", config=str(HARDWARE_YAML))
        return cc._touch_sheet(args, BOARD, det.corners), probe

    def test_touching_places_the_sheet_in_the_arms_own_frame(self, monkeypatch):
        """Arm kinematics biased 4 mm in x: the placement carries the same
        bias, which is what makes the arm's picks land (it goes where it
        believes the object is, and the camera now believes the same)."""
        (origin, yaw), probe = self._run(monkeypatch, bias=(0.004, 0.0))
        assert probe.closed and [i for i, _ in probe.asked] == [1, 2, 3, 4]
        assert origin[0] == pytest.approx(SHEET_ORIGIN[0] + 0.004, abs=0.0015)
        assert origin[1] == pytest.approx(SHEET_ORIGIN[1], abs=0.0015)
        assert yaw == pytest.approx(SHEET_YAW, abs=0.5)

    def test_one_skipped_corner_still_places_the_sheet(self, monkeypatch):
        (origin, yaw), _probe = self._run(monkeypatch, skip={2})
        assert origin == pytest.approx(SHEET_ORIGIN, abs=0.0015) and yaw == pytest.approx(SHEET_YAW, abs=0.6)

    def test_a_mistouched_corner_is_refused(self, monkeypatch):
        with pytest.raises(cc.CalibrationError, match="disagree with the printed geometry"):
            self._run(monkeypatch, mistouch=(1, (0.0, 0.010)))

    def test_three_skipped_corners_are_refused(self, monkeypatch):
        with pytest.raises(cc.CalibrationError, match="at least two touched corners"):
            self._run(monkeypatch, skip={0, 1, 2})

    def test_two_touches_place_the_sheet_but_warn_a_mistouch_is_invisible(self, monkeypatch, capsys):
        """With two corners the rigid fit has no redundancy: a 10 mm mis-touch
        is NOT refused (it cannot be seen), so the operator is told to check."""
        (origin, _yaw), _ = self._run(monkeypatch, skip={1, 3}, mistouch=(2, (0.0, 0.010)))
        assert "mis-touch cannot be detected from two" in capsys.readouterr().out
        assert np.linalg.norm(np.subtract(origin, SHEET_ORIGIN)) > 0.003  # the error went through

    def test_pose_cli_demands_a_placement(self, capsys):
        with pytest.raises(SystemExit):
            cc.main(["pose", "--images", "x"])
        assert "--origin X Y and --yaw DEG" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            cc.main(["pose", "--images", "x", "--touch"])


# ----------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------


def _block_range(lines: list[str], key: str = "exterior_camera") -> tuple[int, int]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{key}:"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
               and not lines[i].startswith("#"))
    return start, end


class TestYamlWrite:
    INTRINSICS = {"resolution": [640, 480], "fx": 812.34567, "fy": 809.1, "cx": 322.5, "cy": 236.25,
                  "distortion": [0.061, -0.149, 0.0008, -1.0e-05], "intrinsics_rms_px": 0.1234,
                  "pose_measured": False}

    def _pose_values(self):
        return {"position": [0.175, 0.012, 0.585], "look_at": [0.19, -0.004, 0.0], "up": [0.99955, 0.03, 0.0],
                "homography": [1e-05, -0.00056, 0.52, -0.00057, 3.2e-20, 0.55, 0.0, 0.0004, 1.0],
                "pose_measured": True}

    def test_round_trip_keeps_every_other_line_byte_identical(self, tmp_path):
        cfg_path = tmp_path / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")
        original = cfg_path.read_bytes()
        cc.write_camera_values(cfg_path, self.INTRINSICS, comments={"fx": "calibrated"})
        cc.write_camera_values(cfg_path, self._pose_values())
        new = cfg_path.read_bytes()
        assert (tmp_path / "hardware.yaml.bak").exists()
        old_lines = original.decode().split("\n")
        new_lines = new.decode().split("\n")
        s0, e0 = _block_range(old_lines)
        s1, e1 = _block_range(new_lines)
        assert old_lines[:s0 + 1] == new_lines[:s1 + 1]  # everything above, byte for byte
        assert old_lines[e0:] == new_lines[e1:]  # everything below, byte for byte
        # Inside the block, every comment line survives, in order.
        old_comments = [ln for ln in old_lines[s0:e0] if ln.strip().startswith("#")]
        new_comments = [ln for ln in new_lines[s1:e1] if ln.strip().startswith("#")]
        assert old_comments == new_comments
        cfg = load_config(cfg_path)
        cam = cfg.exterior_camera
        assert (cam.fx, cam.fy, cam.cx, cam.cy) == (812.3457, 809.1, 322.5, 236.25)
        assert cam.distortion == (0.061, -0.149, 0.0008, -1.0e-05)
        assert cam.intrinsics_rms_px == 0.1234 and cam.pose_measured is True
        assert cam.position == (0.175, 0.012, 0.585) and cam.up == (0.99955, 0.03, 0.0)
        assert cam.homography[0] == 1e-05 and cam.homography[4] == 3.2e-20
        # The unrelated sections load exactly as before.
        before = load_config(HARDWARE_YAML)
        assert cfg.hardware == before.hardware and cfg.robot == before.robot and cfg.scene == before.scene
        assert "fx: 812.3457  # calibrated" in new.decode()

    def test_crlf_file_stays_crlf_and_missing_keys_land_inside_the_block(self, tmp_path):
        text = ("# top comment\r\nextends: default.yaml\r\n\r\nexterior_camera:\r\n  # measure me\r\n"
                "  fx: 600.0\r\n  homography: []\r\n\r\n# about perception\r\nperception:\r\n"
                "  ground_plane_z: 0.0\r\n")
        path = tmp_path / "cam.yaml"
        path.write_bytes(text.encode())
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")
        cc.write_camera_values(path, {"fx": 700.0, "intrinsics_rms_px": 0.2, "look_at": [0.1, 0.0, 0.0]})
        raw = path.read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n")
        lines = raw.decode().split("\r\n")
        assert lines[:5] == ["# top comment", "extends: default.yaml", "", "exterior_camera:", "  # measure me"]
        assert lines[5] == "  fx: 700.0000"
        assert lines[6] == "  homography: []"
        assert lines[7:9] == ["  intrinsics_rms_px: 0.2000", "  look_at: [0.10000, 0.00000, 0.00000]"]
        assert lines[9:] == ["", "# about perception", "perception:", "  ground_plane_z: 0.0", ""]

    def test_multiline_and_block_lists_are_replaced_whole(self):
        text = ("exterior_camera:\n  homography: [1, 0, 0,\n               0, 1, 0,\n               0, 0, 1]\n"
                "  distortion:\n    - 0.1\n    - 0.2\n    - 0.0\n    - 0.0\n  fx: 1.0\nrobot:\n  name: x\n")
        new = cc.update_camera_block(text, {"homography": [2.0] * 9, "distortion": [0.5, 0.0, 0.0, 0.0]})
        data = yaml.safe_load(new)
        assert data["exterior_camera"] == {"homography": [2.0] * 9, "distortion": [0.5, 0.0, 0.0, 0.0], "fx": 1.0}
        assert new.endswith("  fx: 1.0\nrobot:\n  name: x\n")

    def test_a_value_the_schema_rejects_restores_the_file(self, tmp_path):
        cfg_path = tmp_path / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")
        original = cfg_path.read_bytes()
        with pytest.raises(ConfigError, match="distortion"):
            cc.write_camera_values(cfg_path, {"distortion": [0.1, 0.2, 0.3]})
        assert cfg_path.read_bytes() == original

    @pytest.mark.parametrize("value", [1e-05, -3.2e-20, 12345678901.0, 0.0, 5.0])
    def test_floats_are_written_so_yaml_reads_floats(self, value):
        line = cc._format_value("homography", [value] * 9, "")
        parsed = yaml.safe_load("\n".join(line))["homography"]
        assert all(isinstance(v, float) for v in parsed) and parsed[0] == value

    def test_shipped_hardware_yaml_has_no_measured_intrinsics(self):
        cam = load_config(HARDWARE_YAML).exterior_camera
        assert cam.intrinsics_rms_px == 0.0 and cam.pose_measured is False

    def test_negative_rms_is_a_config_error(self):
        with pytest.raises(ConfigError, match="intrinsics_rms_px"):
            CameraConfig(intrinsics_rms_px=-1.0).validate()


# ----------------------------------------------------------------------
# perception uses the calibration
# ----------------------------------------------------------------------


class TestPinholeDistortion:
    @pytest.mark.parametrize("dist", [D_TRUE[:4], D_TRUE, np.array([0.1, -0.2, 0.001, 0.002, 0.05, 0.01, -0.02, 0.03])])
    def test_projection_matches_cv2(self, dist):
        r_rc, pos = robot_camera()
        model = PinholeModel(K_TRUE[0, 0], K_TRUE[1, 1], K_TRUE[0, 2], K_TRUE[1, 2], pos, CAM_LOOK, CAM_UP,
                             distortion=tuple(dist))
        rng = np.random.default_rng(0)
        pts = np.c_[rng.uniform(0.02, 0.34, 300), rng.uniform(-0.25, 0.25, 300), rng.uniform(0.0, 0.1, 300)]
        rvec, _ = cv2.Rodrigues(r_rc.T)
        want, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, -r_rc.T @ pos, K_TRUE, dist)
        assert np.max(np.abs(model.project(pts) - want.reshape(-1, 2))) < 1e-6

    def test_no_distortion_is_the_old_ideal_model(self):
        model = PinholeModel(600, 600, 320, 240, (0.18, 0, 0.6), (0.18, 0, 0), (1, 0, 0))
        assert not model.has_distortion
        p = np.array([[0.2, 0.05, 0.0]])
        cam = (p - model.position) @ model.rotation
        assert np.allclose(model.project(p), [[600 * cam[0, 0] / cam[0, 2] + 320, 600 * cam[0, 1] / cam[0, 2] + 240]])

    def test_points_past_the_lens_fold_are_not_projected(self):
        """Strong barrel lens: a point far outside the field of view would wrap
        back into the image; it must come out NaN instead."""
        model = PinholeModel(600, 600, 320, 240, (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (1, 0, 0),
                             distortion=(-0.4, 0.0, 0.0, 0.0))
        uv = model.project(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [5.0, 0.0, 0.0]]))
        assert np.all(np.isfinite(uv[:2])) and np.all(np.isnan(uv[2]))

    def test_bad_coefficient_count_is_refused(self):
        from mfw.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            PinholeModel(600, 600, 320, 240, (0, 0, 0.5), (0, 0, 0), (1, 0, 0), distortion=(0.1, 0.2))


class _Clock:
    step_index = 0
    sim_time = 0.0


class _VerifyRobot:
    def __init__(self, width, tcp):
        self._width, self._tcp = width, tcp

    def get_gripper_width(self):
        return self._width

    def tcp_pose(self):
        return self._tcp


# The team's camera: 640x480, clamp stand over the table, roughly nadir.
NADIR = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480,
             position=(0.18, 0.0, 0.60), look_at=(0.18, 0.0, 0.0), up=(1.0, 0.0, 0.0))


def _nadir_views(n=20, seed=21):
    """Calibration views for the NADIR camera (ideal lens, like the scripted detector)."""
    k = np.array([[NADIR["fx"], 0, NADIR["cx"]], [0, NADIR["fy"], NADIR["cy"]], [0, 0, 1.0]])
    rng = np.random.default_rng(seed)
    views = []
    for r, t in calibration_poses(n, seed=seed, k=k, dist=np.zeros(5)):
        det = cc.find_board_corners(render_sheet(k, np.zeros(5), r, t, rng=rng), BOARD)
        assert det is not None
        views.append(det.corners)
    return k, views


class TestCalibratedCameraConfirmsCarries:
    """Audit F1: the tool's output is what turns 'cannot confirm the grasp' into 'carried'."""

    @pytest.fixture(scope="class")
    def configs(self, tmp_path_factory):
        from jetson.detector_service import SyntheticPinhole

        tmp = tmp_path_factory.mktemp("camcal")
        cfg_path = tmp / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp / "default.yaml")
        truth = SyntheticPinhole(**NADIR)

        k, views = _nadir_views()
        intr = cc.calibrate_intrinsics(views, (640, 480), BOARD)
        cc.write_camera_values(cfg_path, cc.intrinsics_values(intr))
        only_intrinsics = load_config(cfg_path)

        r_rc = truth.rotation_world_from_camera()
        pos = np.asarray(truth.position, dtype=np.float64)
        r_cb, t_cb = sheet_extrinsics(r_rc, pos, origin=(0.10, -0.12), yaw=0.0)
        rng = np.random.default_rng(3)
        frames = []
        for _ in range(5):
            det = cc.find_board_corners(render_sheet(k, np.zeros(5), r_cb, t_cb, rng=rng), BOARD)
            assert det is not None and det.origin_resolved
            frames.append(det.corners)
        cam = only_intrinsics.exterior_camera
        pose = cc.estimate_sheet_pose(frames, intr.camera_matrix, intr.distortion, BOARD, (0.10, -0.12), 0.0)
        lo, hi = (0.03, -0.27), (0.33, 0.27)
        hm, _rms, max_mm, _n = cc.fit_workspace_homography(intr.camera_matrix, intr.distortion, pose, lo, hi,
                                                          tuple(cam.resolution))
        assert max_mm < 1.0
        cc.write_camera_values(cfg_path, cc.pose_values(pose, hm, measured=True))
        return truth, only_intrinsics, load_config(cfg_path), pose

    def _verdict(self, cfg, truth, xy, lift=True):
        from jetson.detector_service import ScriptedDetector
        from mfw.core.types import BoundingBox3D, ObjectHypothesis, SceneGraph
        from mfw.grasp.generator import build_grasp_pose
        from mfw.physics.contact import verify_grasp

        cam = cfg.exterior_camera
        perception = PlanarPerception(clock=_Clock(), detector=None, config=cfg.perception, hardware=cfg.hardware,
                                      homography=cam.homography, camera=cam)
        detector = ScriptedDetector({}, truth)
        lift_h = cfg.grasp.lift_height
        half_h = perception.estimator.size_for("marker")[2] / 2.0
        tcp = np.array([xy[0], xy[1], half_h + lift_h])

        def seen(base, step, occluder=None):
            box = detector.box_for("marker", base, occluder)
            est = perception.estimator.estimate_footprint({"label": "marker", "bbox_px": list(box)})
            return ObjectHypothesis(
                track_id="m", label="marker", pose=est.pose,
                bbox=BoundingBox3D(center=est.pose, extents=est.extents), confidence=0.9, num_points=1,
                last_seen_sim_time=0.0, last_seen_step=step,
                attributes={"bbox_px": [float(v) for v in box], "yaw_rad": est.yaw_rad,
                            "yaw_ambiguous": est.yaw_ambiguous})

        before = seen((xy[0], xy[1], 0.0), 0)
        after = seen((xy[0], xy[1], lift_h) if lift else (xy[0], xy[1], 0.0), 40)
        bearing = math.atan2(xy[1], xy[0])
        tcp_pose = build_grasp_pose(tcp, np.array([-math.sin(bearing), math.cos(bearing), 0.0]),
                                    np.array([0.0, 0.0, -1.0]))
        evidence = verify_grasp(
            robot=_VerifyRobot(cfg.robot.gripper_closed_width, tcp_pose),
            scene_before=SceneGraph(objects={"m": before}, sim_time=0.0, step_index=0),
            scene_after=SceneGraph(objects={"m": after}, sim_time=0.0, step_index=40),
            track_id="m", closed_width=cfg.robot.gripper_closed_width, expected_height_gain=lift_h,
            gripper_feedback=False, min_displacement=cfg.grasp.verify_min_displacement,
            lift_prediction=perception.predict_lift(before, tcp),
        )
        return evidence, perception

    def test_the_measured_pose_is_the_true_one(self, configs):
        truth, _only, cfg, pose = configs
        assert np.linalg.norm(pose.position - np.asarray(truth.position)) * 1000 < 5.0
        assert _angle_deg(pose.rotation, truth.rotation_world_from_camera()) < 1.0
        assert cfg.exterior_camera.pose_measured is True and cfg.exterior_camera.intrinsics_rms_px > 0

    POSITIONS = [(0.15, 0.0), (0.18, 0.05), (0.20, -0.10), (0.25, 0.12), (0.22, -0.15)]

    def test_a_real_carry_reads_carried_with_the_calibrated_config(self, configs):
        truth, _only, cfg, _pose = configs
        for xy in self.POSITIONS:
            evidence, perception = self._verdict(cfg, truth, xy)
            assert perception.pose_measured
            assert evidence.verdict == "carried" and evidence.holding, (xy, evidence.reason())

    def test_a_missed_marker_is_still_not_carried(self, configs):
        truth, _only, cfg, _pose = configs
        for xy in self.POSITIONS:
            evidence, _ = self._verdict(cfg, truth, xy, lift=False)
            assert not evidence.holding, (xy, evidence.reason())

    def test_without_the_pose_step_the_same_carry_is_unknown(self, configs):
        """Intrinsics alone reset pose_measured: the verdict stays conservative.
        (The homography is borrowed from the finished config so perception can run.)"""
        truth, only, cfg, _pose = configs
        assert only.exterior_camera.pose_measured is False
        from dataclasses import replace

        cam = replace(only.exterior_camera, homography=cfg.exterior_camera.homography,
                      position=cfg.exterior_camera.position, look_at=cfg.exterior_camera.look_at,
                      up=cfg.exterior_camera.up)
        evidence, perception = self._verdict(replace(only, exterior_camera=cam), truth, (0.18, 0.05))
        assert not perception.pose_measured
        assert evidence.verdict == "unknown" and not evidence.holding

    def test_objects_are_placed_within_a_few_mm(self, configs):
        truth, _only, cfg, _pose = configs
        from jetson.detector_service import ScriptedDetector

        cam = cfg.exterior_camera
        perception = PlanarPerception(clock=_Clock(), detector=None, config=cfg.perception, hardware=cfg.hardware,
                                      homography=cam.homography, camera=cam)
        detector = ScriptedDetector({}, truth)
        for label in ("cube", "box", "bowl"):
            for xy in self.POSITIONS:
                box = detector.box_for(label, (xy[0], xy[1], 0.0))
                est = perception.estimator.estimate_footprint({"label": label, "bbox_px": list(box)})
                err = np.linalg.norm(est.pose.position[:2] - np.asarray(xy)) * 1000
                assert err < 4.0, (label, xy, err)


# ----------------------------------------------------------------------
# end to end through robot_server's get_frame
# ----------------------------------------------------------------------


class _FlakyCheckerCapture:
    """A ``cv2.VideoCapture`` stand-in for the webcam robot_server owns.

    The sheet lies on the table; frames carry fresh sensor noise; every 7th
    read fails (USB hiccup); and for a stretch the camera stalls completely,
    so ``get_frame`` refuses the stale frame, as it does on the Jetson.
    """

    def __init__(self, images, stall=(0.25, 0.9)):
        self.images = images
        self.reads = 0
        self.t0 = time.monotonic()
        self.stall = stall

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.005)
        self.reads += 1
        t = time.monotonic() - self.t0
        if self.stall[0] <= t < self.stall[1]:
            time.sleep(0.05)
            return False, None
        if self.reads % 7 == 0:
            return False, None
        return True, self.images[self.reads % len(self.images)].copy()

    def set(self, *_a):
        return True

    def release(self):
        pass


class TestThroughRobotServer:
    def test_pose_from_get_frame_with_dropped_reads_and_a_stall(self, calibrated, monkeypatch):
        from jetson import robot_server as rs
        from mfw.hardware.jetson_client import JetsonClient

        images = _sheet_frames(n=4, seed=8)
        capture = _FlakyCheckerCapture(images)
        grabber = rs.CameraGrabber(rs.CameraSettings(device="0", width=W, height=H),
                                   capture_factory=lambda _dev: capture)
        monkeypatch.setattr(rs, "DEFAULT_MAX_FRAME_AGE_S", 0.2)
        cal = rs.ServoCalibration()
        server = rs.RobotServer("127.0.0.1", 0, rs.FakeDriver(cal), camera=grabber, calibration=cal,
                                follower_sleep=lambda _s: None, host_timeout_s=0.0)
        grabber.start()
        _, port = server.serve_in_thread(port=0)
        assert port not in range(5555, 5561)
        client = JetsonClient("127.0.0.1", port, request_timeout_s=2.0)
        try:
            client.connect()
            source = cc.JetsonFrameSource("127.0.0.1", port, client=client, max_consecutive_errors=500)
            time.sleep(0.3)  # into the stall
            frames, size, counts = cc.collect_sheet_frames(source, BOARD, want=6, timeout_s=20.0, idle_sleep_s=0.01)
        finally:
            client.close()
            server.stop()
            grabber.stop()
        assert size == (W, H) and len(frames) == 6, counts
        assert source.errors >= 1, "the stall must have been seen (and survived)"
        assert counts.get("duplicate-frame", 0) >= 0
        pose = cc.estimate_sheet_pose(frames, calibrated.camera_matrix, calibrated.distortion, BOARD,
                                      SHEET_ORIGIN, SHEET_YAW)
        assert np.linalg.norm(pose.position - CAM_POS) * 1000 < 5.0

    def test_a_dead_server_is_reported_not_waited_on(self):
        class Dead:
            def get_frame(self, _q=None):
                raise TimeoutError("no reply in 3.0 s")

        source = cc.JetsonFrameSource("x", 1, client=Dead(), max_consecutive_errors=3)
        assert source.read() is None and source.read() is None
        with pytest.raises(cc.CalibrationError, match="3 frame requests in a row failed"):
            source.read()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


class TestCli:
    def test_intrinsics_then_pose_then_check_from_saved_images(self, tmp_path, calibration_views, capsys):
        _poses, images, _dets = calibration_views
        views_dir = tmp_path / "views"
        views_dir.mkdir()
        for i, img in enumerate(images):
            cv2.imwrite(str(views_dir / f"v{i:02d}.png"), img)
        sheet_dir = tmp_path / "sheet"
        sheet_dir.mkdir()
        for i, img in enumerate(_sheet_frames(n=4)):
            cv2.imwrite(str(sheet_dir / f"s{i}.png"), img)
        cfg_path = tmp_path / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")

        # Pose before intrinsics: refused, nothing written.
        before = cfg_path.read_bytes()
        assert cc.main(["pose", "--images", str(sheet_dir), "--config", str(cfg_path),
                        "--origin", *map(str, SHEET_ORIGIN), "--yaw", str(SHEET_YAW)]) == 2
        assert cfg_path.read_bytes() == before
        assert cc.main(["intrinsics", "--images", str(views_dir), "--config", str(cfg_path)]) == 0
        cfg = load_config(cfg_path)
        assert 0 < cfg.exterior_camera.intrinsics_rms_px < 0.3 and cfg.exterior_camera.pose_measured is False
        assert abs(cfg.exterior_camera.fx / K_TRUE[0, 0] - 1) < 0.02
        assert cc.main(["pose", "--images", str(sheet_dir), "--config", str(cfg_path),
                        "--origin", *map(str, SHEET_ORIGIN), "--yaw", str(SHEET_YAW)]) == 0
        cam = load_config(cfg_path).exterior_camera
        assert cam.pose_measured is True and len(cam.homography) == 9
        assert np.linalg.norm(np.asarray(cam.position) - CAM_POS) * 1000 < 5.0
        out = tmp_path / "check.png"
        code = cc.main(["check", "--image", str(sheet_dir / "s0.png"), "--config", str(cfg_path),
                        "--origin", *map(str, SHEET_ORIGIN), "--yaw", str(SHEET_YAW), "--out", str(out)])
        text = capsys.readouterr().out
        assert code == 0, text
        assert out.exists() and cv2.imread(str(out)).shape == (H, W, 3)
        assert "sheet corners, homography" in text and "sheet corners, camera model" in text
        # A sheet claimed at the wrong place fails the check.
        code = cc.main(["check", "--image", str(sheet_dir / "s0.png"), "--config", str(cfg_path),
                        "--origin", str(SHEET_ORIGIN[0] + 0.01), str(SHEET_ORIGIN[1]), "--yaw", str(SHEET_YAW),
                        "--out", str(out)])
        assert code == 1

    def test_intrinsics_from_too_few_images_writes_nothing(self, tmp_path, calibration_views):
        _poses, images, _dets = calibration_views
        d = tmp_path / "few"
        d.mkdir()
        for i, img in enumerate(images[:10]):
            cv2.imwrite(str(d / f"v{i}.png"), img)
        cfg_path = tmp_path / "hardware.yaml"
        shutil.copy2(HARDWARE_YAML, cfg_path)
        shutil.copy2(REPO_ROOT / "configs" / "default.yaml", tmp_path / "default.yaml")
        before = cfg_path.read_bytes()
        assert cc.main(["intrinsics", "--images", str(d), "--config", str(cfg_path)]) == 2
        assert cfg_path.read_bytes() == before

"""Measure the fixed overhead camera: intrinsics, pose in the robot frame, homography.

    py -3.12 scripts/calibrate_camera.py make-board --out board.png
    py -3.12 scripts/calibrate_camera.py intrinsics --jetson 192.168.1.50:5560 --config configs/hardware.yaml
    py -3.12 scripts/calibrate_camera.py pose --jetson 192.168.1.50:5560 --origin 0.10 -0.12 --yaw 0
    py -3.12 scripts/calibrate_camera.py check --jetson 192.168.1.50:5560 --origin 0.10 -0.12 --yaw 0

Why this exists (hardware audit F1): with ``exterior_camera.pose_measured:
false`` the feedback-less grasp check (``mfw/physics/contact.py``) can never
say ``carried`` -- growth of the object's pixel box alone is what a spun
object or detector jitter also produce -- so every real pick ended "cannot
confirm the grasp". The lift prediction that *can* confirm a carry needs a
pinhole model of the real camera: checkerboard intrinsics plus the camera's
pose in the robot base frame. This script measures both and writes them, with
the table-plane homography that follows from them, into the config.

Subcommands
-----------
``make-board``
    A4 landscape PNG at 300 dpi (``pHYs`` chunk set, so "actual size" prints
    true): 9 x 6 inner corners, 25 mm squares, the origin corner and the +x /
    +y directions marked, a 100 mm scale bar. PRINT AT 100 % (no "fit to
    page"), then measure the bar; pass the real square size with
    ``--square-mm`` if the printer scaled it. Tape the sheet flat: a bowed
    sheet is the most common reason for a bad calibration.
``intrinsics``
    Collects >= 15 distinct, *still* checkerboard views (hold the sheet
    still for a moment in each pose; tilt it up to ~40 deg; cover the corners
    of the image), from the robot server's ``get_frame`` (``--jetson``), a
    local webcam (``--device N``) or saved images (``--images DIR``). Runs
    ``cv2.findChessboardCorners`` + ``cornerSubPix`` + ``cv2.calibrateCamera``,
    drops views that disagree with the rest, and REFUSES to save when the
    RMS reprojection error is above 1.0 px. Writes ``fx fy cx cy distortion
    resolution intrinsics_rms_px`` and resets ``pose_measured`` to false (a
    pose fitted with the old intrinsics is no longer valid).
``pose``
    The same sheet lying FLAT on the table at a KNOWN place in the robot base
    frame: ``--origin X Y`` is where the origin inner corner (the one the
    arrows start from) is, metres; ``--yaw DEG`` is the angle of the sheet's
    +x arrow from the robot's +x, counter-clockwise seen from above; or
    ``--touch`` (with ``--jetson``): after the frames are taken, jog the jaw
    tip onto the sheet's four outer inner-corners and the placement is
    fitted in the arm's own frame (preferred: a ruler error of 5 mm or 1 deg
    moves every object by about as much). Averages
    several frames, ``cv2.solvePnP`` (IPPE + LM refinement), then writes the
    camera ``position``, ``look_at`` (where the optical axis meets the table),
    ``up`` (image-up), the pixel -> table ``homography`` fitted over the
    workspace, and ``pose_measured: true`` -- the last only when the
    intrinsics came from ``intrinsics`` (``intrinsics_rms_px`` in (0, 1]).
``check``
    Overlays the robot-frame grid, the workspace box, the reach band and the
    base axes on a live frame through the *production* ``PinholeModel``
    (so it checks exactly what perception will use), and with ``--origin`` /
    ``--yaw`` prints the reprojection error (px) and table error (mm) of the
    sheet corners. Exit status 1 when the table error exceeds ``--tol-mm``.

Conventions and traps
---------------------
* Sheet frame: origin at the marked inner corner, +x along the long side,
  +y along the short side, +z out of the printed face (up, on the table).
  Corner order from OpenCV is ambiguous up to a 180 deg turn of the board;
  the board has an even x odd square count, so its two ends differ in colour
  and the origin is resolved from the image (a view where it cannot be is
  not used for ``pose``).
* All pixels are RAW (distorted) pixels, the ones the detector boxes are in.
  The homography is fitted to raw pixels over the workspace (a homography
  cannot model the lens; the residual is printed, typically < 1.5 mm for a
  C270-class lens) and ``PinholeModel`` applies the distortion when it
  projects. ``scripts/calibrate_table.py`` also fits raw pixels, so the two
  tools write the same kind of matrix; running it after ``pose`` replaces
  the homography (``check`` then reports how far the two disagree).
* The table plane is ``perception.ground_plane_z`` (``--table-z`` overrides):
  the sheet is assumed to lie on the same plane objects stand on.
* The resolution calibrated is the resolution the frames came in at; the
  detector must see frames of that same size (robot_server serves 640x480).
* Writing the YAML is textual and local to the ``exterior_camera:`` block:
  every other line of the file (comments, ``extends:``, other sections) is
  kept byte for byte, a ``.bak`` is written first, and the result must load
  through ``load_config`` or the backup is restored.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__all__ = [
    "BoardSpec",
    "BoardDetection",
    "CalibrationError",
    "IntrinsicsResult",
    "PoseResult",
    "ViewCollector",
    "MIN_VIEWS",
    "MAX_INTRINSICS_RMS_PX",
    "MAX_POSE_RMS_PX",
    "make_board_image",
    "save_png",
    "find_board_corners",
    "calibrate_intrinsics",
    "estimate_sheet_pose",
    "sheet_corners_robot",
    "sheet_placement_from_points",
    "touch_corner_board_xy",
    "fit_workspace_homography",
    "camera_model_from_config",
    "check_against_sheet",
    "draw_check_overlay",
    "update_camera_block",
    "write_camera_values",
    "intrinsics_values",
    "pose_values",
    "JetsonFrameSource",
    "CaptureFrameSource",
    "ImageFrameSource",
    "collect_views",
    "collect_sheet_frames",
    "main",
]

MIN_VIEWS = 15
"""Fewest distinct views ``intrinsics`` will calibrate from."""
MAX_INTRINSICS_RMS_PX = 1.0
"""Above this RMS reprojection error the intrinsics are not saved."""
MAX_POSE_RMS_PX = 2.0
"""Above this the sheet fit is refused: a bowed sheet, a wrong board size or
intrinsics that do not belong to this camera (or this resolution)."""
MIN_ORIGIN_CONTRAST = 25.0
"""Grey levels by which the origin's dark square must beat its light neighbours
before a view's 180-degree ambiguity counts as resolved."""
DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_SQUARE_MM = 9, 6, 25.0
A4_LANDSCAPE_MM = (297.0, 210.0)


class CalibrationError(RuntimeError):
    """A calibration that must not be saved (or cannot be computed)."""


def _cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment
        raise RuntimeError("opencv-python is required: py -3.12 -m pip install opencv-python") from exc
    return cv2


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardSpec:
    """A checkerboard of ``cols`` x ``rows`` *inner* corners, ``square`` metres apart.

    Board frame: corner ``(i, j)`` (i along x, j along y) is at
    ``(i * square, j * square, 0)``; the square just outside the origin corner
    (x, y in [-square, 0]) is dark, and a square is dark when ``i + j`` is even
    (its lower-left corner index). ``cols + rows`` must be odd so the far end
    of the board has the opposite colour -- that is what resolves OpenCV's
    180-degree corner-order ambiguity.
    """

    cols: int = DEFAULT_COLS
    rows: int = DEFAULT_ROWS
    square: float = DEFAULT_SQUARE_MM / 1000.0

    def validate(self) -> None:
        if self.cols < 3 or self.rows < 3:
            raise ValueError("the board needs at least 3 x 3 inner corners")
        if self.cols == self.rows or (self.cols + self.rows) % 2 == 0:
            raise ValueError(
                f"{self.cols} x {self.rows} inner corners is symmetric under a half turn: use an odd + even "
                "pair (e.g. 9 x 6) so the origin can be told from the opposite corner"
            )
        if not 0.003 <= self.square <= 0.2:
            raise ValueError(f"square size {self.square * 1000:.1f} mm is implausible")

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    @property
    def count(self) -> int:
        return self.cols * self.rows

    def object_points(self) -> NDArray[np.float64]:
        """``(cols*rows, 3)`` board-frame corners in canonical order (j-major)."""
        j, i = np.mgrid[0:self.rows, 0:self.cols]
        return np.stack([i.ravel() * self.square, j.ravel() * self.square, np.zeros(self.count)], axis=1)

    @property
    def size_m(self) -> tuple[float, float]:
        """Printed checker extent (all squares, including the outer ring), metres."""
        return ((self.cols + 1) * self.square, (self.rows + 1) * self.square)


def _png_bytes(gray: NDArray[np.uint8], dpi: float) -> bytes:
    """A greyscale PNG with a ``pHYs`` chunk (cv2 cannot write one)."""
    h, w = gray.shape
    raw = b"".join(b"\x00" + gray[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ppm = int(round(dpi / 0.0254))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def save_png(path: str | Path, gray: NDArray[np.uint8], dpi: float = 300.0) -> Path:
    """Write ``gray`` as a PNG that prints at ``dpi``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_png_bytes(np.ascontiguousarray(gray, dtype=np.uint8), dpi))
    return p


def board_layout(spec: BoardSpec, dpi: float = 300.0,
                 paper_mm: tuple[float, float] = A4_LANDSCAPE_MM) -> dict[str, Any]:
    """Page geometry: pixels per metre, page size, and the board origin's page pixel.

    Page pixel ``(pu, pv)`` <-> board ``(x, y)``: ``x = (pu - u0) / ppm``,
    ``y = (v0 - pv) / ppm`` (board +y is up the page).
    """
    ppm = dpi / 0.0254
    width = int(round(paper_mm[0] / 1000.0 * ppm))
    height = int(round(paper_mm[1] / 1000.0 * ppm))
    bw, bh = spec.size_m
    if bw * 1000.0 > paper_mm[0] - 30.0 or bh * 1000.0 > paper_mm[1] - 24.0:
        raise ValueError(
            f"a {spec.cols} x {spec.rows} board of {spec.square * 1000:.1f} mm squares "
            f"({bw * 1000:.0f} x {bh * 1000:.0f} mm) leaves no margin on {paper_mm[0]:.0f} x {paper_mm[1]:.0f} mm paper"
        )
    left = (width - bw * ppm) / 2.0
    bottom = (height + bh * ppm) / 2.0
    # The origin corner is one square in from the board's lower-left edge.
    return {
        "ppm": ppm, "width": width, "height": height,
        "u0": left + spec.square * ppm, "v0": bottom - spec.square * ppm,
        "paper_m": (paper_mm[0] / 1000.0, paper_mm[1] / 1000.0),
    }


def make_board_image(spec: BoardSpec, dpi: float = 300.0,
                     paper_mm: tuple[float, float] = A4_LANDSCAPE_MM) -> NDArray[np.uint8]:
    """The printable page, greyscale uint8 (white paper, black squares, annotations)."""
    cv2 = _cv2()
    spec.validate()
    lay = board_layout(spec, dpi, paper_mm)
    ppm, u0, v0 = lay["ppm"], lay["u0"], lay["v0"]
    page = np.full((lay["height"], lay["width"]), 255, dtype=np.uint8)
    s = spec.square * ppm
    for j in range(-1, spec.rows):
        for i in range(-1, spec.cols):
            if (i + j) % 2 != 0:
                continue
            ua, ub = int(round(u0 + i * s)), int(round(u0 + (i + 1) * s))
            va, vb = int(round(v0 - (j + 1) * s)), int(round(v0 - j * s))
            page[va:vb, ua:ub] = 0

    mm = ppm / 1000.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.09 * mm  # ~3 mm cap height
    thick = max(1, int(round(0.3 * mm)))
    board_top = v0 - spec.rows * s
    board_bottom = v0 + s
    board_left = u0 - s
    # Top margin, one line: the title on the left, a 100 mm scale bar
    # right-aligned to the board. Both kept >= 9 mm off the squares
    # (findChessboardCorners wants a white quiet zone round the board) and
    # >= 5 mm off the paper edge (printers do not print the outer ~4 mm; a
    # 9x6 board of 25 mm squares leaves a 17.5 mm top margin on A4).
    title = (f"mfw calibration board  {spec.cols}x{spec.rows} inner corners  "
             f"square = {spec.square * 1000:.1f} mm  PRINT AT 100% (no fit-to-page)")
    y_text = int(board_top - 9.5 * mm)
    (title_w, _), _ = cv2.getTextSize(title, font, scale, thick)
    cv2.putText(page, title, (int(board_left), y_text), font, scale, 0, thick, cv2.LINE_AA)
    bar_label = "100 mm"
    (label_w, _), _ = cv2.getTextSize(bar_label, font, scale, thick)
    bar_y = int(board_top - 10.5 * mm)
    bar_x1 = int(round(board_left + spec.size_m[0] * ppm))
    bar_x0 = bar_x1 - int(round(100 * mm))
    label_x = bar_x0 - int(3 * mm) - label_w
    if label_x < board_left + title_w + 3 * mm or bar_y - 1.5 * mm < 4 * mm:
        # A board this wide leaves no room beside the title: bar on its own line above.
        bar_y = int(board_top - 15.0 * mm)
    if bar_y - 1.5 * mm >= 4 * mm and label_x >= 4 * mm:
        cv2.line(page, (bar_x0, bar_y), (bar_x1, bar_y), 0, thick, cv2.LINE_AA)
        for x in (bar_x0, bar_x1):
            cv2.line(page, (x, bar_y - int(1.5 * mm)), (x, bar_y + int(1.5 * mm)), 0, thick, cv2.LINE_AA)
        cv2.putText(page, bar_label, (label_x, bar_y + int(1.2 * mm)), font, scale, 0, thick, cv2.LINE_AA)
    # Bottom margin, under the origin: +x arrow and the origin label.
    y_arrow = int(board_bottom + 9.5 * mm)
    x_origin = int(round(u0))
    cv2.arrowedLine(page, (x_origin, y_arrow), (x_origin + int(40 * mm), y_arrow), 0, thick, cv2.LINE_AA,
                    tipLength=0.12)
    cv2.putText(page, "+x", (x_origin + int(42 * mm), y_arrow + int(1.5 * mm)), font, scale, 0, thick, cv2.LINE_AA)
    cv2.line(page, (x_origin, y_arrow - int(2 * mm)), (x_origin, y_arrow + int(2 * mm)), 0, thick, cv2.LINE_AA)
    cv2.putText(page, "ORIGIN = inner corner above this tick; +y is up the page, +z out of the paper",
                (x_origin + int(60 * mm), y_arrow + int(1.5 * mm)), font, scale * 0.8, 0, thick, cv2.LINE_AA)
    # Left margin, level with the origin: +y arrow.
    x_arrow = int(board_left - 11 * mm)
    y_origin = int(round(v0))
    if x_arrow > int(5 * mm):
        cv2.arrowedLine(page, (x_arrow, y_origin), (x_arrow, y_origin - int(40 * mm)), 0, thick, cv2.LINE_AA,
                        tipLength=0.12)
        cv2.putText(page, "+y", (x_arrow - int(3 * mm), y_origin - int(43 * mm)), font, scale, 0, thick, cv2.LINE_AA)
    return page


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardDetection:
    """Corners in canonical order (``BoardSpec.object_points`` order), raw pixels."""

    corners: NDArray[np.float64]
    origin_resolved: bool
    """True when the dark origin square was seen, so the order is certain; False
    means the order may be the half-turned one (fine for intrinsics only)."""
    contrast: float
    """Weakest light minus darkest dark grey level round the origin (nan: out of view)."""


def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    cv2 = _cv2()
    img = np.asarray(image)
    if img.ndim == 2:
        return np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {img.shape}")


def _sample_mean(gray: NDArray[np.uint8], pts: NDArray[np.float64]) -> float:
    h, w = gray.shape
    vals = []
    for u, v in pts:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        iu, iv = int(round(u)), int(round(v))
        if 0 <= iu < w and 0 <= iv < h:
            vals.append(float(gray[iv, iu]))
    return float(np.mean(vals)) if len(vals) >= max(3, len(pts) // 2) else math.nan


def _origin_contrast(gray: NDArray[np.uint8], grid: NDArray[np.float64], spec: BoardSpec) -> tuple[float, float]:
    """``(mean light - mean dark, min light - max dark)`` round the origin for this corner order.

    Samples the two squares that must be dark (outside the origin corner and
    the first inner one) and the two that must be light (their neighbours).
    The first number ranks the candidate orders; the second must clear
    :data:`MIN_ORIGIN_CONTRAST` for the order to count as resolved, so a hand
    or the arm over any one of the four squares leaves it unresolved rather
    than guessed. ``nan`` when a square is out of the image.
    """
    cv2 = _cv2()
    j, i = np.mgrid[0:spec.rows, 0:spec.cols]
    src = np.stack([i.ravel(), j.ravel()], axis=1).astype(np.float64)
    h_bi, _ = cv2.findHomography(src, grid.reshape(-1, 2), 0)
    if h_bi is None:
        return math.nan, math.nan
    offsets = np.array([[a, b] for a in (-0.25, 0.0, 0.25) for b in (-0.25, 0.0, 0.25)])

    def square(ci: float, cj: float) -> float:
        pts = np.array([[ci, cj]]) + offsets
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h_bi.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = hom[:, :2] / hom[:, 2:3]
        return _sample_mean(gray, uv)

    dark = [square(-0.5, -0.5), square(0.5, 0.5)]
    light = [square(0.5, -0.5), square(-0.5, 0.5)]
    if any(math.isnan(v) for v in dark + light):
        return math.nan, math.nan
    return float(np.mean(light) - np.mean(dark)), float(min(light) - max(dark))


def find_board_corners(image: NDArray[np.uint8], spec: BoardSpec) -> BoardDetection | None:
    """Detect the board; ``None`` when it is not (fully) in view.

    ``findChessboardCorners`` (adaptive threshold, normalised) with
    ``cornerSubPix`` refinement, falling back to ``findChessboardCornersSB``.
    The corner order is then made canonical: of the four orders a planar grid
    admits, the two mirror images are rejected by handedness (the printed face
    is seen from above), and the half turn by the origin's colour.
    """
    cv2 = _cv2()
    gray = _to_gray(image)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, spec.pattern_size, flags=flags)
    refined = False
    if found:
        pts = corners.reshape(-1, 2).astype(np.float32)
        grid0 = pts.reshape(spec.rows, spec.cols, 2)
        spacing = min(
            float(np.min(np.linalg.norm(np.diff(grid0, axis=1), axis=2))),
            float(np.min(np.linalg.norm(np.diff(grid0, axis=0), axis=2))),
        )
        win = int(np.clip(spacing / 3.0, 2, 11))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
        pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (win, win), (-1, -1), criteria)
        corners = pts
        refined = True
    elif hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found, corners = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=sb_flags)
        refined = bool(found)
    if not found or corners is None or not refined:
        return None
    grid = np.asarray(corners, dtype=np.float64).reshape(spec.rows, spec.cols, 2)
    candidates = [grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1]]
    kept: list[tuple[float, float, NDArray[np.float64]]] = []
    for g in candidates:
        a = np.mean(g[:, 1:] - g[:, :-1], axis=(0, 1))  # along board +x
        b = np.mean(g[1:, :] - g[:-1, :], axis=(0, 1))  # along board +y
        if a[0] * b[1] - a[1] * b[0] < 0.0:  # printed face seen from above
            mean_diff, margin = _origin_contrast(gray, g, spec)
            kept.append((mean_diff if np.isfinite(mean_diff) else -math.inf, margin, g))
    if not kept:
        return None
    kept.sort(key=lambda t: t[0], reverse=True)
    _rank, margin, best = kept[0]
    resolved = bool(np.isfinite(margin) and margin >= MIN_ORIGIN_CONTRAST)
    return BoardDetection(corners=best.reshape(-1, 2).copy(), origin_resolved=resolved,
                          contrast=float(margin) if np.isfinite(margin) else math.nan)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicsResult:
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    rms_px: float
    per_view_rms_px: list[float]
    used: list[int]
    dropped: list[int]
    image_size: tuple[int, int]
    coverage: float
    warnings: list[str] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def report(self) -> str:
        lines = [
            f"views used {len(self.used)}  dropped {len(self.dropped)} {self.dropped if self.dropped else ''}",
            f"RMS reprojection {self.rms_px:.3f} px (limit {MAX_INTRINSICS_RMS_PX:.1f})",
            f"fx {self.fx:.2f}  fy {self.fy:.2f}  cx {self.cx:.2f}  cy {self.cy:.2f}  "
            f"at {self.image_size[0]}x{self.image_size[1]}",
            "distortion " + ", ".join(f"{v:+.5f}" for v in self.distortion),
            f"image coverage by corners {self.coverage * 100:.0f} %",
            "per-view RMS px: " + " ".join(f"{v:.2f}" for v in self.per_view_rms_px),
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _coverage(views: Sequence[NDArray[np.float64]], image_size: tuple[int, int], cells: tuple[int, int] = (8, 6)) -> float:
    w, h = image_size
    hit = np.zeros((cells[1], cells[0]), dtype=bool)
    for v in views:
        cu = np.clip((v[:, 0] / w * cells[0]).astype(int), 0, cells[0] - 1)
        cv = np.clip((v[:, 1] / h * cells[1]).astype(int), 0, cells[1] - 1)
        hit[cv, cu] = True
    return float(hit.mean())


def _calibrate_once(views: Sequence[NDArray[np.float64]], spec: BoardSpec, image_size: tuple[int, int],
                    fix_k3: bool) -> tuple[float, NDArray[np.float64], NDArray[np.float64], list[float]]:
    cv2 = _cv2()
    obj = spec.object_points().astype(np.float32)
    objs = [obj.reshape(-1, 1, 3) for _ in views]
    imgs = [np.asarray(v, dtype=np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, image_size, None, None, flags=flags,
                                                  criteria=criteria)
    per_view = []
    for o, im, rv, tv in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, k, d)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
    return float(rms), np.asarray(k, dtype=np.float64), np.asarray(d, dtype=np.float64).reshape(-1), per_view


def calibrate_intrinsics(
    views: Sequence[NDArray[np.float64]],
    image_size: tuple[int, int],
    spec: BoardSpec,
    fix_k3: bool = True,
    min_views: int = MIN_VIEWS,
    max_rms_px: float = MAX_INTRINSICS_RMS_PX,
) -> IntrinsicsResult:
    """``cv2.calibrateCamera`` over canonical-order corner sets, with one outlier pass.

    A view whose own RMS exceeds ``max(max_rms_px / 2, 3 x median)`` (a board that
    moved during the exposure, a mis-ordered detection) is dropped and the
    calibration repeated, as long as ``min_views`` remain. Raises
    :class:`CalibrationError` with too few views or an RMS above
    ``max_rms_px`` -- such a result must not be saved.
    """
    spec.validate()
    views = [np.asarray(v, dtype=np.float64).reshape(-1, 2) for v in views]
    if len(views) < min_views:
        raise CalibrationError(
            f"only {len(views)} usable checkerboard views; need >= {min_views}. Show the board in more "
            "places and tilts (hold it still in each)."
        )
    for v in views:
        if len(v) != spec.count:
            raise CalibrationError(f"a view has {len(v)} corners, the board has {spec.count}")
    idx = list(range(len(views)))
    rms, k, d, per_view = _calibrate_once(views, spec, image_size, fix_k3)
    dropped: list[int] = []
    limit = max(0.5 * max_rms_px, 3.0 * float(np.median(per_view)))
    bad = [i for i, e in zip(idx, per_view) if e > limit]
    if bad and len(views) - len(bad) >= min_views:
        dropped = bad
        idx = [i for i in idx if i not in bad]
        rms, k, d, per_view = _calibrate_once([views[i] for i in idx], spec, image_size, fix_k3)
    coverage = _coverage([views[i] for i in idx], image_size)
    if fix_k3 and len(d) >= 5:
        d = d[:4]
    else:
        d = d[:5]
    result = IntrinsicsResult(
        camera_matrix=k, distortion=d, rms_px=rms, per_view_rms_px=per_view, used=idx,
        dropped=dropped, image_size=(int(image_size[0]), int(image_size[1])), coverage=coverage,
    )
    w, h = image_size
    if coverage < 0.6:
        result.warnings.append(
            f"the corners covered only {coverage * 100:.0f} % of the image; the lens model is extrapolated "
            "near the edges. Show the board close to each image corner too."
        )
    if abs(result.fx / result.fy - 1.0) > 0.05:
        result.warnings.append(f"fx/fy = {result.fx / result.fy:.3f}; webcam pixels are square, expect ~1.0")
    if abs(result.cx - w / 2.0) > 0.15 * w or abs(result.cy - h / 2.0) > 0.15 * h:
        result.warnings.append("the principal point is far from the image centre; views may be too uniform")
    if not np.isfinite(rms) or rms > max_rms_px:
        raise CalibrationError(
            f"RMS reprojection error {rms:.3f} px exceeds {max_rms_px:.1f} px; NOT saved. Usual causes: the "
            "sheet is not flat (tape it to a board), the board moved during capture, or the square size / "
            "corner count is wrong.\n" + result.report()
        )
    return result


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    image: NDArray[np.uint8]
    """BGR (or greyscale) uint8."""
    key: Any
    """Identity of the frame (Jetson ``seq``, a counter, a file name): equal keys = same exposure."""


class JetsonFrameSource:
    """Frames from ``robot_server``'s ``get_frame`` (the webcam's only owner on the Jetson).

    Transient failures -- a stale-frame refusal while the camera stalls, a
    request timeout -- return ``None`` and are counted; ``max_consecutive_errors``
    of them in a row raise, so a dead server is reported rather than waited on
    forever. Repeated ``seq`` numbers are passed through (as the same ``key``)
    for the caller to skip.
    """

    def __init__(self, host: str, port: int, quality: int = 95, request_timeout_s: float = 3.0,
                 max_consecutive_errors: int = 20, client: Any = None) -> None:
        self.quality = int(quality)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.errors = 0
        self.consecutive_errors = 0
        self.last_error = ""
        if client is None:
            from mfw.hardware.jetson_client import JetsonClient  # noqa: PLC0415

            client = JetsonClient(host, int(port), request_timeout_s=request_timeout_s)
            client.connect()
        self.client = client

    def read(self) -> Frame | None:
        from mfw.hardware.remote_camera import decode_jpeg  # noqa: PLC0415

        try:
            reply = self.client.get_frame(self.quality)
            rgb = decode_jpeg(bytes(reply["jpeg"]))
        except Exception as exc:  # noqa: BLE001 - a network peer; counted and bounded
            self.errors += 1
            self.consecutive_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise CalibrationError(
                    f"{self.consecutive_errors} frame requests in a row failed; last: {self.last_error}"
                ) from exc
            return None
        self.consecutive_errors = 0
        return Frame(image=np.ascontiguousarray(rgb[:, :, ::-1]), key=("seq", int(reply.get("seq", -1))))

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()


class CaptureFrameSource:
    """Frames from a ``cv2.VideoCapture``-like object (``--device N``)."""

    def __init__(self, capture: Any, max_consecutive_errors: int = 30) -> None:
        self.capture = capture
        self.count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = int(max_consecutive_errors)

    @classmethod
    def open_device(cls, index: int, width: int | None = None, height: int | None = None) -> "CaptureFrameSource":
        cv2 = _cv2()
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            raise CalibrationError(f"could not open camera device {index}")
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        return cls(cap)

    def read(self) -> Frame | None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise CalibrationError(f"{self.consecutive_errors} camera reads in a row failed")
            return None
        self.consecutive_errors = 0
        self.count += 1
        return Frame(image=np.ascontiguousarray(frame), key=("n", self.count))

    def close(self) -> None:
        release = getattr(self.capture, "release", None)
        if release is not None:
            release()


class ImageFrameSource:
    """Frames from image files (``--images DIR`` / ``--image PATH``); ``None`` once exhausted."""

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, paths: Iterable[str | Path]) -> None:
        self.paths = [Path(p) for p in paths]
        self._i = 0

    @classmethod
    def from_dir(cls, directory: str | Path) -> "ImageFrameSource":
        d = Path(directory)
        if not d.is_dir():
            raise CalibrationError(f"{d} is not a directory")
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in cls.EXTENSIONS)
        if not paths:
            raise CalibrationError(f"no images in {d}")
        return cls(paths)

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self.paths)

    def read(self) -> Frame | None:
        cv2 = _cv2()
        while self._i < len(self.paths):
            p = self.paths[self._i]
            self._i += 1
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return Frame(image=img, key=("file", p.name))
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# View collection (intrinsics)
# ---------------------------------------------------------------------------


def _view_descriptor(corners: NDArray[np.float64], spec: BoardSpec, image_size: tuple[int, int]) -> NDArray[np.float64]:
    """Where, how big and how tilted the board is: what makes views informative."""
    cv2 = _cv2()
    w, h = image_size
    g = corners.reshape(spec.rows, spec.cols, 2)
    c = corners.mean(axis=0)
    hull = cv2.convexHull(corners.astype(np.float32))
    area = float(cv2.contourArea(hull))
    top = np.linalg.norm(g[0, -1] - g[0, 0])
    bottom = np.linalg.norm(g[-1, -1] - g[-1, 0])
    left = np.linalg.norm(g[-1, 0] - g[0, 0])
    right = np.linalg.norm(g[-1, -1] - g[0, -1])
    return np.array([
        c[0] / w, c[1] / h, math.sqrt(max(area, 1.0) / (w * h)),
        math.log(max(bottom, 1e-6) / max(top, 1e-6)), math.log(max(right, 1e-6) / max(left, 1e-6)),
    ])


class ViewCollector:
    """Keeps a view only when the board is detected, held still, and new.

    *Still*: its corners moved less than ``still_px`` (mean) since the
    previous *different* frame -- a board in motion is smeared along the
    motion and ``cornerSubPix`` then lands off the true corner. *New*: its
    descriptor (image position, size, tilt) is at least ``min_novelty`` from
    every view already kept, so twenty frames of one pose do not count as
    twenty views.
    """

    def __init__(self, spec: BoardSpec, want: int, still_px: float = 0.6, min_novelty: float = 0.1,
                 need_still: bool = True) -> None:
        spec.validate()
        self.spec = spec
        self.want = int(want)
        self.still_px = float(still_px)
        self.min_novelty = float(min_novelty)
        self.need_still = bool(need_still)
        self.views: list[NDArray[np.float64]] = []
        self.images: list[NDArray[np.uint8]] = []
        self.descriptors: list[NDArray[np.float64]] = []
        self.image_size: tuple[int, int] | None = None
        self.counts: dict[str, int] = {}
        self._last_key: Any = None
        self._last_corners: NDArray[np.float64] | None = None

    @property
    def done(self) -> bool:
        return len(self.views) >= self.want

    def _count(self, status: str) -> str:
        self.counts[status] = self.counts.get(status, 0) + 1
        return status

    def offer(self, frame: Frame) -> str:
        """``accepted`` | ``duplicate-frame`` | ``size-mismatch`` | ``no-board`` | ``moving`` | ``similar``."""
        if frame.key is not None and frame.key == self._last_key:
            return self._count("duplicate-frame")
        self._last_key = frame.key
        h, w = frame.image.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)
        elif self.image_size != (w, h):
            return self._count("size-mismatch")
        det = find_board_corners(frame.image, self.spec)
        if det is None:
            self._last_corners = None
            return self._count("no-board")
        previous, self._last_corners = self._last_corners, det.corners
        if self.need_still:
            if previous is None:
                return self._count("moving")
            if float(np.mean(np.linalg.norm(det.corners - previous, axis=1))) > self.still_px:
                # Also catches a half-turn re-ordering between frames.
                return self._count("moving")
        desc = _view_descriptor(det.corners, self.spec, (w, h))
        if self.descriptors and min(float(np.linalg.norm(desc - d)) for d in self.descriptors) < self.min_novelty:
            return self._count("similar")
        self.views.append(det.corners)
        self.images.append(frame.image)
        self.descriptors.append(desc)
        return self._count("accepted")


def collect_views(source: Any, spec: BoardSpec, want: int = 20, *, timeout_s: float = 300.0,
                  still_px: float = 0.6, min_novelty: float = 0.1, need_still: bool = True,
                  on_status: Callable[[str, int], None] | None = None,
                  clock: Callable[[], float] = time.monotonic, idle_sleep_s: float = 0.0) -> ViewCollector:
    """Pull frames from ``source`` until ``want`` views are kept, it runs dry, or ``timeout_s``."""
    collector = ViewCollector(spec, want, still_px=still_px, min_novelty=min_novelty, need_still=need_still)
    deadline = clock() + float(timeout_s)
    while not collector.done and clock() < deadline:
        frame = source.read()
        if frame is None:
            if getattr(source, "exhausted", False):
                break
            if idle_sleep_s:
                time.sleep(idle_sleep_s)
            continue
        status = collector.offer(frame)
        if on_status is not None:
            on_status(status, len(collector.views))
    return collector


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def _rot_z(yaw_rad: float) -> NDArray[np.float64]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sheet_corners_robot(spec: BoardSpec, origin_xy: Sequence[float], yaw_deg: float,
                        table_z: float = 0.0) -> NDArray[np.float64]:
    """Board corners in the robot base frame for a sheet at ``origin_xy`` turned ``yaw_deg``."""
    r = _rot_z(math.radians(float(yaw_deg)))
    t = np.array([float(origin_xy[0]), float(origin_xy[1]), float(table_z)])
    return spec.object_points() @ r.T + t


TOUCH_CORNERS = ("origin", "+x end", "far", "+y end")
"""Sheet corners ``pose --touch`` asks for, in this order."""
MAX_TOUCH_RESIDUAL_MM = 3.0
"""A touched corner further than this from where the other touches (and the
printed geometry) put it means a mis-touch or a wrongly scaled print."""


def touch_corner_board_xy(spec: BoardSpec) -> NDArray[np.float64]:
    """Board-frame ``(x, y)`` of :data:`TOUCH_CORNERS`."""
    x, y = (spec.cols - 1) * spec.square, (spec.rows - 1) * spec.square
    return np.array([[0.0, 0.0], [x, 0.0], [x, y], [0.0, y]])


def sheet_placement_from_points(board_xy: ArrayLike, robot_xy: ArrayLike) -> tuple[tuple[float, float], float, float]:
    """``(origin_xy, yaw_deg, worst_residual_mm)``: the sheet placement that best maps ``board_xy`` onto ``robot_xy``.

    A 2-D rigid fit (rotation + translation, no scale: the print's scale is
    known) over >= 2 point pairs, e.g. corners touched with the jaw tip in the
    arm's own frame -- which, like ``calibrate_table.py --touch``, makes any
    link-length or zero-offset bias of the arm cancel out of the final pick.
    """
    b = np.asarray(board_xy, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(robot_xy, dtype=np.float64).reshape(-1, 2)
    if len(b) != len(r) or len(b) < 2:
        raise CalibrationError("need >= 2 matching sheet points to place the sheet")
    bc, rc = b.mean(axis=0), r.mean(axis=0)
    m = (b - bc).T @ (r - rc)
    yaw = math.atan2(m[0, 1] - m[1, 0], m[0, 0] + m[1, 1])
    rot = _rot_z(yaw)[:2, :2]
    origin = rc - rot @ bc
    residual = np.linalg.norm((b @ rot.T + origin) - r, axis=1)
    return (float(origin[0]), float(origin[1])), math.degrees(yaw), float(residual.max() * 1000.0)


@dataclass
class PoseResult:
    position: NDArray[np.float64]
    """Camera centre, robot base frame, metres."""
    rotation: NDArray[np.float64]
    """Columns: camera x (right), y (down), z (forward) in the robot frame."""
    look_at: NDArray[np.float64]
    up: NDArray[np.float64]
    rms_px: float
    max_px: float
    frames_used: int
    frames_rejected: int
    spread_mm: float
    """Std of the per-frame camera centres (noise estimate; not the ruler error)."""
    table_z: float

    def report(self) -> str:
        tilt = math.degrees(math.acos(float(np.clip(-self.rotation[2, 2], -1.0, 1.0))))
        return "\n".join([
            f"frames used {self.frames_used}  rejected {self.frames_rejected}  "
            f"per-frame spread {self.spread_mm:.2f} mm",
            f"sheet fit RMS {self.rms_px:.3f} px  max {self.max_px:.3f} px (limit {MAX_POSE_RMS_PX:.1f})",
            "camera position (robot frame) " + np.array2string(self.position, precision=4),
            f"height above table {self.position[2] - self.table_z:.4f} m  optical axis {tilt:.1f} deg from vertical",
            "look_at " + np.array2string(self.look_at, precision=4) + "  up " + np.array2string(self.up, precision=4),
        ])


def estimate_sheet_pose(
    frames: Sequence[NDArray[np.float64]],
    camera_matrix: ArrayLike,
    distortion: ArrayLike,
    spec: BoardSpec,
    origin_xy: Sequence[float],
    yaw_deg: float,
    table_z: float = 0.0,
    max_rms_px: float = MAX_POSE_RMS_PX,
) -> PoseResult:
    """Camera pose in the robot frame from canonical-order sheet corners (one or more frames).

    Frames are combined by the per-corner median after dropping frames more
    than 1 px (mean) from it -- the arm or a hand passing over the sheet,
    a nudge. ``solvePnP`` (IPPE for a plane, then LM) gives board -> camera;
    the known sheet placement turns that into the robot frame.
    """
    cv2 = _cv2()
    spec.validate()
    if not frames:
        raise CalibrationError("no frame showed the whole sheet with its origin resolved")
    stack = np.stack([np.asarray(f, dtype=np.float64).reshape(-1, 2) for f in frames])
    median = np.median(stack, axis=0)
    dev = np.mean(np.linalg.norm(stack - median, axis=2), axis=1)
    keep = dev <= 1.0
    if int(keep.sum()) < max(1, (len(stack) + 1) // 2):
        raise CalibrationError(
            f"the sheet moved between frames (only {int(keep.sum())}/{len(stack)} agree within 1 px); "
            "keep hands and the arm off it and retry"
        )
    corners = np.median(stack[keep], axis=0)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    obj = spec.object_points()

    def pnp(img: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), k, d, flags=flag)
        if not ok:
            raise CalibrationError("solvePnP failed on the sheet corners")
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), k, d, rvec, tvec)
        return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)

    rvec, tvec = pnp(corners)
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, k, d)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners, axis=1)
    rms, mx = float(np.sqrt(np.mean(err ** 2))), float(err.max())
    if rms > max_rms_px:
        raise CalibrationError(
            f"the sheet fit is {rms:.2f} px RMS (limit {max_rms_px:.1f}): the sheet is not flat, the "
            "square size / corner count is wrong, or the intrinsics are not this camera's at this resolution"
        )
    r_cb, _ = cv2.Rodrigues(rvec)
    r_rb = _rot_z(math.radians(float(yaw_deg)))
    t_rb = np.array([float(origin_xy[0]), float(origin_xy[1]), float(table_z)])

    def centre(rv: NDArray[np.float64], tv: NDArray[np.float64]) -> NDArray[np.float64]:
        rm, _ = cv2.Rodrigues(rv)
        return r_rb @ (-rm.T @ tv) + t_rb

    position = centre(rvec, tvec)
    rotation = r_rb @ r_cb.T  # robot <- camera
    forward = rotation[:, 2]
    if forward[2] > -0.2:
        raise CalibrationError(
            "the recovered camera does not look down at the table (optical axis "
            f"{np.round(forward, 3).tolist()}); check --yaw / --origin and that the sheet lies flat face up"
        )
    height = float(position[2] - table_z)
    if height < 0.1:
        raise CalibrationError(f"the recovered camera is {height:.3f} m above the table; that is not an overhead camera")
    s = (table_z - position[2]) / forward[2]
    look_at = position + s * forward
    up = -rotation[:, 1]
    per_frame = []
    for f in stack[keep]:
        try:
            per_frame.append(centre(*pnp(f)))
        except CalibrationError:
            continue
    spread = float(np.sqrt(np.mean(np.sum((np.asarray(per_frame) - position) ** 2, axis=1)))) * 1000.0 \
        if len(per_frame) > 1 else 0.0
    return PoseResult(position=position, rotation=rotation, look_at=look_at, up=up, rms_px=rms, max_px=mx,
                      frames_used=int(keep.sum()), frames_rejected=int((~keep).sum()), spread_mm=spread,
                      table_z=float(table_z))


def fit_workspace_homography(
    camera_matrix: ArrayLike, distortion: ArrayLike, pose: PoseResult,
    xy_min: Sequence[float], xy_max: Sequence[float], image_size: tuple[int, int],
    step_m: float = 0.005,
) -> tuple[NDArray[np.float64], float, float, int]:
    """Raw-pixel -> table homography fitted over the workspace; ``(H, rms_mm, max_mm, n_points)``.

    Table points on a ``step_m`` grid are projected through the full lens
    model; the ones that land inside the image are fitted. The residual is
    what a homography cannot express (lens distortion), in millimetres.
    """
    cv2 = _cv2()
    xs = np.arange(float(xy_min[0]), float(xy_max[0]) + 1e-9, step_m)
    ys = np.arange(float(xy_min[1]), float(xy_max[1]) + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, pose.table_z)], axis=1)
    r_cr = pose.rotation.T
    rvec, _ = cv2.Rodrigues(r_cr)
    tvec = -r_cr @ pose.position
    cam_z = (pts - pose.position) @ pose.rotation[:, 2]
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, np.asarray(camera_matrix, dtype=np.float64),
                              np.asarray(distortion, dtype=np.float64))
    uv = uv.reshape(-1, 2)
    w, h = image_size
    inside = (cam_z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    if int(inside.sum()) < 20:
        raise CalibrationError(
            f"only {int(inside.sum())} workspace grid points are in view; the camera does not see the workspace"
        )
    hm, _ = cv2.findHomography(uv[inside], pts[inside, :2], 0)
    if hm is None:
        raise CalibrationError("homography fit failed")
    hm = hm / hm[2, 2]
    hom = np.concatenate([uv[inside], np.ones((int(inside.sum()), 1))], axis=1) @ hm.T
    xy = hom[:, :2] / hom[:, 2:3]
    err = np.linalg.norm(xy - pts[inside, :2], axis=1) * 1000.0
    return hm, float(np.sqrt(np.mean(err ** 2))), float(err.max()), int(inside.sum())


def _apply_h(h: NDArray[np.float64], uv: NDArray[np.float64]) -> NDArray[np.float64]:
    hom = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ np.asarray(h, dtype=np.float64).reshape(3, 3).T
    return hom[:, :2] / hom[:, 2:3]


def _homography_disagreement_mm(h_old: NDArray[np.float64], h_new: NDArray[np.float64], camera_matrix: ArrayLike,
                                distortion: ArrayLike, pose: PoseResult, xy_min: Sequence[float],
                                xy_max: Sequence[float], image_size: tuple[int, int]) -> float | None:
    """Worst table distance (mm) between two homographies over the in-view workspace pixels."""
    cv2 = _cv2()
    xs = np.arange(float(xy_min[0]), float(xy_max[0]) + 1e-9, 0.01)
    ys = np.arange(float(xy_min[1]), float(xy_max[1]) + 1e-9, 0.01)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, pose.table_z)], axis=1)
    r_cr = pose.rotation.T
    rvec, _ = cv2.Rodrigues(r_cr)
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, -r_cr @ pose.position,
                              np.asarray(camera_matrix, dtype=np.float64), np.asarray(distortion, dtype=np.float64))
    uv = uv.reshape(-1, 2)
    w, h = image_size
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not np.any(ok):
        return None
    return float(np.max(np.linalg.norm(_apply_h(h_old, uv[ok]) - _apply_h(h_new, uv[ok]), axis=1)) * 1000.0)


def _fmt_float(v: float, spec: str) -> str:
    """A float YAML will read back as a float (PyYAML needs a '.' before an exponent)."""
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"refusing to write non-finite value {x}")
    s = format(x, spec)
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        if "." not in mant:
            mant += ".0"
        s = f"{mant}e{exp}"
    elif "." not in s and "n" not in s:
        s += ".0"
    return s


def intrinsics_values(result: IntrinsicsResult) -> dict[str, Any]:
    """Config values the ``intrinsics`` step writes."""
    return {
        "resolution": [int(result.image_size[0]), int(result.image_size[1])],
        "fx": result.fx, "fy": result.fy, "cx": result.cx, "cy": result.cy,
        "distortion": [float(v) for v in result.distortion],
        "intrinsics_rms_px": float(result.rms_px),
        "pose_measured": False,
    }


def pose_values(pose: PoseResult, homography: NDArray[np.float64] | None, measured: bool) -> dict[str, Any]:
    """Config values the ``pose`` step writes."""
    values: dict[str, Any] = {
        "position": [float(v) for v in pose.position],
        "look_at": [float(v) for v in pose.look_at],
        "up": [float(v) for v in pose.up],
    }
    if homography is not None:
        values["homography"] = [float(v) for v in np.asarray(homography, dtype=np.float64).reshape(-1)]
    values["pose_measured"] = bool(measured)
    return values


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def camera_model_from_config(camera: Any) -> Any:
    """The production :class:`mfw.hardware.perception.PinholeModel` for ``camera`` (``None`` if it has no pose)."""
    from mfw.hardware.perception import PinholeModel  # noqa: PLC0415

    return PinholeModel.from_config(camera)


def check_against_sheet(camera: Any, corners_px: NDArray[np.float64], spec: BoardSpec,
                        origin_xy: Sequence[float], yaw_deg: float, table_z: float = 0.0) -> dict[str, Any]:
    """Errors of the configured camera against a sheet seen at a known place.

    ``reproj_px``: detected corners vs the configured ``PinholeModel``'s
    projection of the known corners. ``table_mm``: the configured homography
    applied to the detected corners vs the known corners -- the error an
    object placed there would get.
    """
    truth = sheet_corners_robot(spec, origin_xy, yaw_deg, table_z)
    out: dict[str, Any] = {}
    model = camera_model_from_config(camera)
    if model is not None:
        proj = model.project(truth)
        e = np.linalg.norm(proj - corners_px, axis=1)
        out["reproj_px_rms"] = float(np.sqrt(np.nanmean(e ** 2)))
        out["reproj_px_max"] = float(np.nanmax(e))
    if len(camera.homography) == 9:
        xy = _apply_h(np.asarray(camera.homography, dtype=np.float64), corners_px)
        e = np.linalg.norm(xy - truth[:, :2], axis=1) * 1000.0
        out["table_mm_rms"] = float(np.sqrt(np.mean(e ** 2)))
        out["table_mm_max"] = float(e.max())
    return out


def homography_vs_model_mm(camera: Any, xy_min: Sequence[float], xy_max: Sequence[float],
                           table_z: float = 0.0, step_m: float = 0.01) -> float | None:
    """Worst disagreement (mm) over the workspace between the configured homography and pinhole model."""
    model = camera_model_from_config(camera)
    if model is None or len(camera.homography) != 9:
        return None
    xs = np.arange(float(xy_min[0]), float(xy_max[0]) + 1e-9, step_m)
    ys = np.arange(float(xy_min[1]), float(xy_max[1]) + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, table_z)], axis=1)
    uv = model.project(pts)
    w, h = (int(v) for v in camera.resolution)
    ok = np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not np.any(ok):
        return None
    xy = _apply_h(np.asarray(camera.homography, dtype=np.float64), uv[ok])
    return float(np.max(np.linalg.norm(xy - pts[ok, :2], axis=1)) * 1000.0)


def draw_check_overlay(image: NDArray[np.uint8], camera: Any, xy_min: Sequence[float], xy_max: Sequence[float],
                       reach: tuple[float, float] | None = None, table_z: float = 0.0,
                       grid_m: float = 0.05, sheet_px: NDArray[np.float64] | None = None) -> NDArray[np.uint8]:
    """BGR copy of ``image`` with the robot-frame grid drawn through the configured camera model."""
    cv2 = _cv2()
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    model = camera_model_from_config(camera)
    if model is None:
        raise CalibrationError("exterior_camera has no fx / look_at: nothing to project")

    def poly(points: NDArray[np.float64], colour: tuple[int, int, int], thick: int = 1) -> None:
        uv = model.project(points)
        good = np.all(np.isfinite(uv), axis=1)
        for a, b, ga, gb in zip(uv[:-1], uv[1:], good[:-1], good[1:]):
            if ga and gb:
                cv2.line(out, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))),
                         colour, thick, cv2.LINE_AA)

    x0, y0 = float(xy_min[0]), float(xy_min[1])
    x1, y1 = float(xy_max[0]), float(xy_max[1])
    for x in np.arange(math.ceil(x0 / grid_m) * grid_m, x1 + 1e-9, grid_m):
        poly(np.array([[x, y, table_z] for y in np.linspace(y0, y1, 40)]), (200, 200, 0))
    for y in np.arange(math.ceil(y0 / grid_m) * grid_m, y1 + 1e-9, grid_m):
        poly(np.array([[x, y, table_z] for x in np.linspace(x0, x1, 40)]), (200, 200, 0))
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    for a, b in zip(box[:-1], box[1:]):
        poly(np.array([[*(a + (b - a) * t), table_z] for t in np.linspace(0, 1, 30)]), (0, 200, 255), 2)
    if reach is not None:
        for radius in reach:
            th = np.linspace(-math.pi / 2, math.pi / 2, 90)
            poly(np.stack([radius * np.cos(th), radius * np.sin(th), np.full(th.size, table_z)], axis=1),
                 (255, 0, 255), 1)
    poly(np.array([[0.0, 0.0, table_z], [0.05, 0.0, table_z]]), (0, 0, 255), 2)
    poly(np.array([[0.0, 0.0, table_z], [0.0, 0.05, table_z]]), (0, 255, 0), 2)
    if sheet_px is not None:
        for u, v in sheet_px:
            cv2.circle(out, (int(round(u)), int(round(v))), 2, (0, 0, 255), -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# YAML write
# ---------------------------------------------------------------------------

_TOP_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*):\s*(#.*)?$")

_FLOAT_FORMATS = {
    "fx": ".4f", "fy": ".4f", "cx": ".4f", "cy": ".4f",
    "distortion": ".8g", "intrinsics_rms_px": ".4f",
    "position": ".5f", "look_at": ".5f", "up": ".6f",
    "homography": ".17g",
}


def _format_value(key: str, value: Any, indent: str) -> list[str]:
    """``key: value`` line(s) in YAML flow style."""
    fmt = _FLOAT_FORMATS.get(key, ".10g")
    head = f"{indent}{key}: "
    if isinstance(value, bool):
        return [head + ("true" if value else "false")]
    if isinstance(value, (int, np.integer)):
        return [head + str(int(value))]
    if isinstance(value, (float, np.floating)):
        return [head + _fmt_float(float(value), fmt)]
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [str(int(v)) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                 else _fmt_float(float(v), fmt) for v in value]
        if key == "homography" and len(items) == 9:
            pad = " " * len(head + "[")
            rows = [", ".join(items[i:i + 3]) for i in (0, 3, 6)]
            return [f"{head}[{rows[0]},", f"{pad}{rows[1]},", f"{pad}{rows[2]}]"]
        return [head + "[" + ", ".join(items) + "]"]
    raise TypeError(f"cannot write {key}={value!r}")


def _block_extent(lines: list[str], camera_key: str) -> tuple[int, int] | None:
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{camera_key}:") and _TOP_KEY_RE.match(ln)), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i][0].isspace() and not lines[i].lstrip().startswith("#"):
            end = i
            break
    return start, end


def _entry_extent(lines: list[str], i: int, end: int, entry_indent: str) -> int:
    """Last line index of the value starting on line ``i``."""
    rest = lines[i].split(":", 1)[1].split("#", 1)[0].strip()
    last = i
    if rest.startswith("["):
        depth = 0
        for j in range(i, end):
            body = lines[j].split("#", 1)[0] if j > i else lines[j].split(":", 1)[1].split("#", 1)[0]
            depth += body.count("[") - body.count("]")
            last = j
            if depth <= 0:
                break
    elif rest == "":
        # Block sequence: `- v` items at or below the key's indentation.
        for j in range(i + 1, end):
            s = lines[j].strip()
            ind = len(lines[j]) - len(lines[j].lstrip())
            if s == "" or s.startswith("#"):
                continue
            if (s == "-" or s.startswith("- ")) and ind >= len(entry_indent):
                last = j
                continue
            break
    return last


def update_camera_block(text: str, values: Mapping[str, Any], camera_key: str = "exterior_camera",
                        comments: Mapping[str, str] | None = None) -> str:
    """Set ``values`` inside the top-level ``camera_key`` block of YAML ``text``.

    Existing entries are replaced in place (their own trailing comment kept
    unless ``comments`` gives a new one); missing ones are appended after the
    block's last entry. Every line outside the replaced entries is returned
    unchanged. A file without the block gets a new block at the end.
    """
    comments = dict(comments or {})
    lines = text.split("\n")
    trailing_newline = text.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]
    extent = _block_extent(lines, camera_key)
    if extent is None:
        block = [f"{camera_key}:"]
        for key, value in values.items():
            new = _format_value(key, value, "  ")
            if key in comments:
                new[0] += f"  # {comments[key]}"
            block += new
        tail = [] if not lines or lines[-1].strip() == "" else [""]
        return "\n".join(lines + tail + block) + "\n"
    start, end = extent
    indent = "  "
    for i in range(start + 1, end):
        stripped = lines[i].lstrip()
        if stripped and not stripped.startswith("#"):
            indent = lines[i][: len(lines[i]) - len(stripped)]
            break
    pending = dict(values)
    out = lines[: start + 1]
    i = start + 1
    last_entry_out = len(out) - 1
    while i < end:
        line = lines[i]
        m = re.match(rf"^{re.escape(indent)}([A-Za-z_][\w-]*):(\s|$)", line)
        if m is not None and m.group(1) in pending:
            key = m.group(1)
            last = _entry_extent(lines, i, end, indent)
            old_comment = ""
            if "#" in lines[i].split(":", 1)[1] and last == i:
                old_comment = "  #" + lines[i].split(":", 1)[1].split("#", 1)[1]
            new = _format_value(key, pending.pop(key), indent)
            if key in comments:
                new[-1] += f"  # {comments[key]}"
            elif old_comment:
                new[-1] += old_comment
            out += new
            last_entry_out = len(out) - 1
            i = last + 1
            continue
        out.append(line)
        if m is not None or (line.startswith(indent) and line.strip() and not line.strip().startswith("#")):
            last_entry_out = len(out) - 1
        i += 1
    if pending:
        extra: list[str] = []
        for key, value in pending.items():
            new = _format_value(key, value, indent)
            if key in comments:
                new[-1] += f"  # {comments[key]}"
            extra += new
        out = out[: last_entry_out + 1] + extra + out[last_entry_out + 1:]
    out += lines[end:]
    return "\n".join(out) + ("\n" if trailing_newline else "")


def write_camera_values(yaml_path: str | Path, values: Mapping[str, Any], camera_key: str = "exterior_camera",
                        comments: Mapping[str, str] | None = None, backup: bool = True,
                        validate: bool = True) -> Path:
    """Splice ``values`` into ``yaml_path``, verify, write (with ``.bak``), and re-load it.

    Checks before writing: the result parses, ``camera_key`` holds exactly
    the new values, every other key of that block and every other top-level
    section parses to what it was. After writing, the file must pass
    ``load_config``; if it does not, the backup is restored and the error
    re-raised. Line endings (LF / CRLF) are preserved.
    """
    import yaml  # noqa: PLC0415

    path = Path(yaml_path)
    raw = path.read_bytes() if path.exists() else b""
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")
    before = yaml.safe_load(text) if text.strip() else {}
    before = before or {}
    if not isinstance(before, dict):
        raise ValueError(f"{path} does not hold a YAML mapping")
    new_text = update_camera_block(text, values, camera_key, comments)
    after = yaml.safe_load(new_text)
    if not isinstance(after, dict) or not isinstance(after.get(camera_key), dict):
        raise RuntimeError(f"the edited {path} lost its {camera_key} block (this is a bug)")
    for key in set(before) | set(after):
        if key == camera_key:
            continue
        if before.get(key) != after.get(key):
            raise RuntimeError(f"the edit would change top-level key {key!r} (this is a bug)")
    old_cam = before.get(camera_key) or {}
    for key in set(old_cam) | set(after[camera_key]):
        if key in values:
            expected = yaml.safe_load("\n".join(_format_value(key, values[key], "")))[key]
            if after[camera_key].get(key) != expected:
                raise RuntimeError(f"{camera_key}.{key} did not round-trip (this is a bug)")
        elif old_cam.get(key) != after[camera_key].get(key):
            raise RuntimeError(f"the edit would change {camera_key}.{key} (this is a bug)")
    bak = path.with_name(path.name + ".bak")
    if backup and path.exists():
        shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8", newline=newline) as fh:
        fh.write(new_text)
    if validate:
        try:
            from mfw.config.schema import load_config  # noqa: PLC0415

            load_config(str(path))
        except Exception:
            if backup and bak.exists():
                shutil.copy2(bak, path)
            else:
                path.write_bytes(raw)
            raise
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _board_from_args(args: argparse.Namespace) -> BoardSpec:
    spec = BoardSpec(cols=int(args.cols), rows=int(args.rows), square=float(args.square_mm) / 1000.0)
    spec.validate()
    return spec


def _open_source(args: argparse.Namespace, resolution: tuple[int, int] | None = None) -> Any:
    if getattr(args, "images", None):
        return ImageFrameSource.from_dir(args.images)
    if getattr(args, "image", None):
        return ImageFrameSource([args.image])
    if getattr(args, "device", None) is not None:
        w, h = resolution if resolution else (None, None)
        return CaptureFrameSource.open_device(int(args.device), w, h)
    if getattr(args, "jetson", None):
        host, _, port = str(args.jetson).partition(":")
        return JetsonFrameSource(host or "127.0.0.1", int(port or 5560))
    raise CalibrationError("give a frame source: --jetson HOST[:PORT], --device N or --images DIR / --image PATH")


def _add_source_args(p: argparse.ArgumentParser, many: bool) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--jetson", help="robot_server HOST[:PORT] (get_frame)")
    g.add_argument("--device", type=int, help="local webcam index")
    if many:
        g.add_argument("--images", help="directory of saved frames")
    g.add_argument("--image", help="one saved frame")


def _add_board_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cols", type=int, default=DEFAULT_COLS, help="inner corners along the sheet's x")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="inner corners along the sheet's y")
    p.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM, help="MEASURED printed square size")


def _load(config_path: str) -> Any:
    from mfw.config.schema import load_config  # noqa: PLC0415

    return load_config(config_path)


def _table_z(cfg: Any, override: float | None) -> float:
    if override is not None:
        return float(override)
    if not cfg.scene.add_table:
        return float(cfg.perception.ground_plane_z)
    return float(cfg.scene.table_position[2] + cfg.scene.table_scale[2] / 2.0)


def _workspace(cfg: Any) -> tuple[list[float], list[float]]:
    m = float(cfg.hardware.workspace_margin_xy)
    lo = [float(cfg.scene.workspace_min[0]) - m, float(cfg.scene.workspace_min[1]) - m]
    hi = [float(cfg.scene.workspace_max[0]) + m, float(cfg.scene.workspace_max[1]) + m]
    return lo, hi


def _cmd_make_board(args: argparse.Namespace) -> int:
    spec = _board_from_args(args)
    page = make_board_image(spec, dpi=float(args.dpi))
    path = save_png(args.out, page, dpi=float(args.dpi))
    bw, bh = spec.size_m
    print(f"wrote {path}: A4 landscape at {args.dpi:g} dpi, {spec.cols}x{spec.rows} inner corners, "
          f"{spec.square * 1000:.1f} mm squares ({bw * 1000:.0f} x {bh * 1000:.0f} mm checker).")
    print("Print at 100 % / actual size. Measure the 100 mm bar; if it is not 100 mm, pass "
          "--square-mm <measured square> to every other subcommand. Tape it flat to something rigid.")
    return 0


def _cmd_intrinsics(args: argparse.Namespace) -> int:
    spec = _board_from_args(args)
    live = not (args.images or args.image)
    source = _open_source(args)
    shown = {"t": 0.0}

    def status(s: str, n: int) -> None:
        if s == "accepted":
            print(f"  view {n}/{args.views} kept")
        elif time.monotonic() - shown["t"] > 2.0 and live:
            shown["t"] = time.monotonic()
            print(f"  ({s}) {n}/{args.views} -- move the board to a new pose and hold it still")

    try:
        print(f"Collecting {args.views} views ({'live' if live else 'from files'}); hold the board still in "
              "each pose, vary distance and tilt, and reach every image corner.")
        collector = collect_views(source, spec, want=args.views, timeout_s=args.timeout_s,
                                  still_px=args.still_px, need_still=live, min_novelty=args.min_novelty if live else 0.0,
                                  on_status=status, idle_sleep_s=0.02 if live else 0.0)
    finally:
        source.close()
    print("frame outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(collector.counts.items())))
    if args.save_dir and collector.images:
        cv2 = _cv2()
        d = Path(args.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(collector.images):
            cv2.imwrite(str(d / f"view_{i:02d}.png"), img)
        print(f"saved {len(collector.images)} views to {d} (re-run with --images {d})")
    if collector.image_size is None:
        raise CalibrationError("no frames were received")
    result = calibrate_intrinsics(collector.views, collector.image_size, spec, fix_k3=not args.free_k3,
                                  min_views=args.min_views)
    print(result.report())
    if args.no_write:
        print("--no-write: config not changed")
        return 0
    cfg = _load(args.config)
    old = tuple(int(v) for v in cfg.exterior_camera.resolution)
    if old != result.image_size:
        print(f"NOTE: exterior_camera.resolution changes {old} -> {result.image_size}; the detector must see "
              "frames of this size.")
    stamp = time.strftime("%Y-%m-%d")
    write_camera_values(args.config, intrinsics_values(result), comments={
        "fx": f"calibrate_camera.py intrinsics {stamp}: {len(result.used)} views, RMS {result.rms_px:.3f} px",
        "pose_measured": "reset by calibrate_camera.py intrinsics; run its `pose` step",
    })
    print(f"wrote intrinsics to {args.config}; pose_measured is now false -- run the `pose` step next.")
    return 0


def collect_sheet_frames(source: Any, spec: BoardSpec, want: int, timeout_s: float = 60.0,
                         clock: Callable[[], float] = time.monotonic,
                         idle_sleep_s: float = 0.0) -> tuple[list[NDArray[np.float64]], tuple[int, int] | None, dict[str, int]]:
    """Up to ``want`` distinct frames with the whole sheet and a resolved origin."""
    frames: list[NDArray[np.float64]] = []
    counts: dict[str, int] = {}
    size: tuple[int, int] | None = None
    last_key: Any = None
    deadline = clock() + float(timeout_s)
    while len(frames) < want and clock() < deadline:
        f = source.read()
        if f is None:
            if getattr(source, "exhausted", False):
                break
            counts["read-failed"] = counts.get("read-failed", 0) + 1
            if idle_sleep_s:
                time.sleep(idle_sleep_s)
            continue
        if f.key is not None and f.key == last_key:
            counts["duplicate-frame"] = counts.get("duplicate-frame", 0) + 1
            continue
        last_key = f.key
        h, w = f.image.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            counts["size-mismatch"] = counts.get("size-mismatch", 0) + 1
            continue
        det = find_board_corners(f.image, spec)
        if det is None:
            counts["no-board"] = counts.get("no-board", 0) + 1
            continue
        if not det.origin_resolved:
            counts["origin-unresolved"] = counts.get("origin-unresolved", 0) + 1
            continue
        frames.append(det.corners)
        counts["used"] = counts.get("used", 0) + 1
    return frames, size, counts


def _cmd_pose(args: argparse.Namespace) -> int:
    spec = _board_from_args(args)
    cfg = _load(args.config)
    cam = cfg.exterior_camera
    table_z = _table_z(cfg, args.table_z)
    fx, fy, cx, cy = cam.pixel_intrinsics()
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    d = np.asarray(cam.distortion if cam.distortion else (0.0, 0.0, 0.0, 0.0), dtype=np.float64)
    intrinsics_ok = 0.0 < float(cam.intrinsics_rms_px) <= MAX_INTRINSICS_RMS_PX
    if not intrinsics_ok and not args.allow_unmeasured_intrinsics:
        raise CalibrationError(
            f"{args.config}: exterior_camera.intrinsics_rms_px is {cam.intrinsics_rms_px}: the intrinsics were not "
            "measured by `calibrate_camera.py intrinsics`. Run that first (or pass "
            "--allow-unmeasured-intrinsics to write the pose with pose_measured left false)."
        )
    source = _open_source(args)
    try:
        frames, size, counts = collect_sheet_frames(source, spec, int(args.frames), timeout_s=args.timeout_s,
                                                    idle_sleep_s=0.02)
    finally:
        source.close()
    print("frame outcomes: " + ", ".join(f"{k_}: {v}" for k_, v in sorted(counts.items())))
    if size is not None and size != tuple(int(v) for v in cam.resolution):
        raise CalibrationError(
            f"frames are {size[0]}x{size[1]} but the intrinsics are for {cam.resolution[0]}x{cam.resolution[1]}; "
            "calibrate the intrinsics at the resolution the detector uses"
        )
    if args.touch:
        if not frames:
            raise CalibrationError("no frame showed the whole sheet with its origin resolved")
        origin, yaw = _touch_sheet(args, spec, np.median(np.stack(frames), axis=0))
    else:
        origin, yaw = (float(args.origin[0]), float(args.origin[1])), float(args.yaw)
    args.origin, args.yaw = list(origin), yaw
    pose = estimate_sheet_pose(frames, k, d, spec, origin, yaw, table_z)
    print(pose.report())
    lo, hi = _workspace(cfg)
    hm, rms_mm, max_mm, n = fit_workspace_homography(k, d, pose, lo, hi, tuple(int(v) for v in cam.resolution))
    print(f"homography over the workspace ({n} grid points in view): lens residual RMS {rms_mm:.2f} mm, "
          f"max {max_mm:.2f} mm")
    if len(cam.homography) == 9:
        diff = _homography_disagreement_mm(np.asarray(cam.homography, dtype=np.float64), hm, k, d, pose, lo, hi,
                                           tuple(int(v) for v in cam.resolution))
        if diff is not None:
            print(f"the previous homography and this one disagree by up to {diff:.1f} mm over the workspace"
                  + (" (previous kept: --keep-homography)" if args.keep_homography else " (replaced)"))
    if max_mm > 5.0:
        print("WARNING: the lens bends straight lines more than a homography can follow over this workspace "
              f"({max_mm:.1f} mm); objects near the image edge will be placed that far off.")
    measured = bool(intrinsics_ok)
    values = pose_values(pose, None if args.keep_homography else hm, measured)
    if args.no_write:
        print("--no-write: config not changed")
        return 0
    stamp = time.strftime("%Y-%m-%d")
    write_camera_values(args.config, values, comments={
        "pose_measured": (f"calibrate_camera.py pose {stamp}: sheet at ({args.origin[0]:.4f}, {args.origin[1]:.4f}) "
                          f"yaw {args.yaw:g} deg, fit {pose.rms_px:.2f} px") if measured
        else "pose written from UNMEASURED intrinsics; run `intrinsics` then `pose`",
    })
    print(f"wrote pose{'' if args.keep_homography else ' and homography'} to {args.config}; "
          f"pose_measured: {'true' if measured else 'false'}. Verify with the `check` subcommand.")
    return 0


def _make_touch_probe(jetson: str, config_path: str) -> Any:
    """``calibrate_table.TouchProbe`` (jog REPL, workspace/floor guards, heartbeat) -- reused, not copied."""
    import importlib.util  # noqa: PLC0415

    path = _ROOT / "scripts" / "calibrate_table.py"
    spec_mod = importlib.util.spec_from_file_location("_calibrate_table_for_touch", path)
    assert spec_mod is not None and spec_mod.loader is not None
    table_mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(table_mod)
    return table_mod.TouchProbe(jetson, config_path)


def _touch_sheet(args: argparse.Namespace, spec: BoardSpec,
                 corners_px: NDArray[np.float64]) -> tuple[tuple[float, float], float]:
    """Jog the jaw tip onto the sheet's four outer inner-corners (any may be skipped, >= 2 needed; >= 3
    to catch a mis-touch) and fit the sheet placement in the arm's frame."""
    grid = corners_px.reshape(spec.rows, spec.cols, 2)
    pixels = [grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]]
    print("Frames are captured; now touch the four outer inner-corners of the sheet with the closed jaw tip (a hair above the "
          "paper). The arm may cover the sheet from here on.")
    probe = _make_touch_probe(args.jetson, args.config)
    touched: list[tuple[float, float]] = []
    used: list[int] = []
    try:
        for i, (name, uv) in enumerate(zip(TOUCH_CORNERS, pixels)):
            print(f"-- sheet {name} corner")
            xy = probe.measure(i + 1, (float(uv[0]), float(uv[1])))
            if xy is not None:
                touched.append(xy)
                used.append(i)
    finally:
        probe.close()
    if len(touched) < 2:
        raise CalibrationError("need at least two touched corners to place the sheet")
    origin, yaw, worst = sheet_placement_from_points(touch_corner_board_xy(spec)[used], touched)
    print(f"sheet placed by touch: origin ({origin[0]:+.4f}, {origin[1]:+.4f}) yaw {yaw:+.2f} deg, "
          f"worst touch residual {worst:.1f} mm")
    if len(touched) == 2:
        print("WARNING: only two corners touched -- a mis-touch cannot be detected from two; check the result "
              "with the `check` subcommand before trusting it")
    if len(touched) >= 3 and worst > MAX_TOUCH_RESIDUAL_MM:
        # With 3+ points a bad touch shows up as residual (with 2 it cannot).
        raise CalibrationError(
            f"the touched corners disagree with the printed geometry by {worst:.1f} mm "
            f"(limit {MAX_TOUCH_RESIDUAL_MM:.0f}): a mis-touch, or --square-mm is not the printed size"
        )
    return origin, yaw


def _cmd_check(args: argparse.Namespace) -> int:
    cv2 = _cv2()
    spec = _board_from_args(args)
    cfg = _load(args.config)
    cam = cfg.exterior_camera
    table_z = _table_z(cfg, args.table_z)
    source = _open_source(args)
    try:
        frame = None
        deadline = time.monotonic() + 15.0
        while frame is None and time.monotonic() < deadline:
            frame = source.read()
            if frame is None and getattr(source, "exhausted", False):
                break
    finally:
        source.close()
    if frame is None:
        raise CalibrationError("no frame received")
    lo, hi = _workspace(cfg)
    det = find_board_corners(frame.image, spec)
    overlay = draw_check_overlay(
        frame.image, cam, lo, hi, reach=(float(cfg.scene.robot_min_reach_m), float(cfg.scene.robot_reach_m)),
        table_z=table_z, sheet_px=None if det is None else det.corners,
    )
    status = 0
    print(f"pose_measured: {cam.pose_measured}  intrinsics_rms_px: {cam.intrinsics_rms_px}")
    disagree = homography_vs_model_mm(cam, lo, hi, table_z)
    if disagree is not None:
        print(f"homography vs camera model over the workspace: up to {disagree:.1f} mm apart")
    if args.origin is not None:
        if det is None or not det.origin_resolved:
            print("the sheet (with its origin) is not visible in this frame; overlay only")
            status = 1
        else:
            res = check_against_sheet(cam, det.corners, spec, args.origin, args.yaw, table_z)
            if "reproj_px_rms" in res:
                print(f"sheet corners, camera model: RMS {res['reproj_px_rms']:.2f} px  max {res['reproj_px_max']:.2f} px")
            if "table_mm_rms" in res:
                print(f"sheet corners, homography:   RMS {res['table_mm_rms']:.2f} mm  max {res['table_mm_max']:.2f} mm "
                      f"(tolerance {args.tol_mm:.1f} mm)")
                if res["table_mm_max"] > args.tol_mm:
                    status = 1
    cv2.imwrite(str(args.out), overlay)
    print(f"overlay written to {args.out}: cyan = 5 cm robot grid, orange = workspace, magenta = reach band, "
          "red / green = robot +x / +y, red dots = detected sheet corners")
    if args.show:  # pragma: no cover - GUI
        cv2.imshow("calibrate_camera check", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-board", help="write the printable A4 checkerboard PNG")
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=float, default=300.0)
    _add_board_args(p)

    p = sub.add_parser("intrinsics", help="checkerboard views -> fx fy cx cy distortion")
    _add_source_args(p, many=True)
    _add_board_args(p)
    p.add_argument("--config", default=str(_ROOT / "configs" / "hardware.yaml"))
    p.add_argument("--views", type=int, default=20, help="views to collect (>= 15 are required)")
    p.add_argument("--min-views", type=int, default=MIN_VIEWS)
    p.add_argument("--still-px", type=float, default=0.6, help="max corner motion between frames for a still view")
    p.add_argument("--min-novelty", type=float, default=0.1, help="how different a new view must be")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--free-k3", action="store_true", help="also fit k3 (default: k1 k2 p1 p2)")
    p.add_argument("--save-dir", help="save the kept views here")
    p.add_argument("--no-write", action="store_true")

    p = sub.add_parser("pose", help="sheet at a known place on the table -> camera pose + homography")
    _add_source_args(p, many=True)
    _add_board_args(p)
    p.add_argument("--config", default=str(_ROOT / "configs" / "hardware.yaml"))
    p.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="robot-frame position of the sheet's origin corner, metres (ruler)")
    p.add_argument("--yaw", type=float, default=None, help="sheet +x from robot +x, degrees CCW from above")
    p.add_argument("--touch", action="store_true",
                   help="instead of --origin/--yaw: touch four sheet corners with the jaw (needs --jetson)")
    p.add_argument("--table-z", type=float, default=None, help="default: perception.ground_plane_z")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--keep-homography", action="store_true", help="write the pose only")
    p.add_argument("--allow-unmeasured-intrinsics", action="store_true")
    p.add_argument("--no-write", action="store_true")

    p = sub.add_parser("check", help="overlay the robot grid on a live frame; sheet corner errors")
    _add_source_args(p, many=False)
    _add_board_args(p)
    p.add_argument("--config", default=str(_ROOT / "configs" / "hardware.yaml"))
    p.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--table-z", type=float, default=None)
    p.add_argument("--tol-mm", type=float, default=3.0)
    p.add_argument("--out", default="camera_check.png")
    p.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pose":
        if args.touch and not args.jetson:
            parser.error("pose --touch needs --jetson HOST[:PORT] (the arm lives behind the robot server)")
        if not args.touch and (args.origin is None or args.yaw is None):
            parser.error("pose needs --origin X Y and --yaw DEG (ruler), or --touch")
    handlers = {"make-board": _cmd_make_board, "intrinsics": _cmd_intrinsics, "pose": _cmd_pose,
                "check": _cmd_check}
    try:
        return handlers[args.command](args)
    except (CalibrationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

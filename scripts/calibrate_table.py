"""Measure the overhead camera's pixel -> table homography and write it to a config.

    py -3.12 scripts/calibrate_table.py --jetson 192.168.1.50:5560 --config configs/hardware.yaml
    py -3.12 scripts/calibrate_table.py --image frame.png --config configs/hardware.yaml
    py -3.12 scripts/calibrate_table.py --jetson 192.168.1.50:5560 --config configs/hardware.yaml --touch
    py -3.12 scripts/calibrate_table.py --from-csv points.csv --config configs/hardware.yaml   # re-fit, no window

Why a homography and not a camera pose: the arm has one fixed webcam and no
depth, so the only geometry perception can do is "this pixel lies on the table
plane, therefore it is at table (x, y)". That map is a 3x3 projective
transform, measured directly from >= 4 point pairs, and it absorbs the lens
pose, focal length and mounting error in one go. ``mfw/hardware/perception.py``
applies it as ``p = H @ [u, v, 1]; x, y = p[0]/p[2], p[1]/p[2]``; the matrix is
stored row-major under ``exterior_camera.homography``.

Two ways to get the table (x, y) of a clicked pixel:

* **typed**: measure with a ruler from the robot base (x forward, y left,
  metres) and type it. Always available; the whole script works this way.
* **``--touch``**: jog the arm until the jaw tip touches the point, and record
  the TCP position the kinematics report. Measures in the *arm's* frame, so
  any link-length or zero-offset bias in ``hardware.arm`` cancels out of the
  final pick -- the arm goes where it thinks the object is, and the camera was
  calibrated in the same belief. This needs ``--jetson`` and the arm powered;
  a heartbeat keeps the arm held while you think at the prompt.

Frames come from ``robot_server.py``'s ``get_frame`` (``--jetson``): the robot
server is the only process that opens the webcam.

Recorded traps:

* Click the point where the object *meets the table* (a tape cross on the
  surface), never a raised feature; ``pixel_anchor: bottom_center`` assumes
  every calibration pixel is on the plane.
* Spread the points over the whole workspace and off the two diagonals. Four
  points in a 5 cm square give a homography that is exact there and 3 cm off
  at the far edge. Eight to twelve points, RANSAC on, is the habit.
* ``findHomography`` with RANSAC needs the error threshold in *pixels* on the
  destination side; here the destination is metres, so the fit is done
  pixel -> table with the threshold converted through the local scale. The
  per-point report prints both metres (what the arm cares about) and pixels.
* Writing the YAML is textual: the matrix is spliced into the existing
  ``exterior_camera:`` block so the file's comments (the MEASURE notes) and
  its ``extends:`` line survive. ``yaml.safe_load`` validates before and
  after; ``yaml.safe_dump`` is only the fallback for a file with no such
  block, and a ``.bak`` copy is written first either way.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__all__ = [
    "MIN_POINTS",
    "fit_homography",
    "apply_homography",
    "reprojection_errors",
    "pixel_errors",
    "format_report",
    "write_homography",
    "read_points_csv",
    "write_points_csv",
    "parse_xy",
    "grab_frame",
    "collect_clicks",
    "TouchProbe",
    "TOUCH_MAX_NUDGE_M",
    "TOUCH_PROBE_MARGIN_M",
    "touch_target_refusal",
    "main",
]

MIN_POINTS = 4
DEFAULT_RANSAC_PX = 3.0


# ---------------------------------------------------------------------------
# Pure maths (unit-tested in tests/test_hardware_perception.py)
# ---------------------------------------------------------------------------


def _as_points(points: ArrayLike, what: str) -> NDArray[np.float64]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{what} must be an (N, 2) array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains non-finite values")
    return arr


def apply_homography(h: ArrayLike, pixels: ArrayLike) -> NDArray[np.float64]:
    """Pixels ``(N, 2)`` -> table ``(N, 2)`` through ``h`` (3x3)."""
    hm = np.asarray(h, dtype=np.float64).reshape(3, 3)
    uv = _as_points(pixels, "pixels")
    hom = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ hm.T
    w = hom[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("a pixel maps to infinity through this homography")
    return hom[:, :2] / w


def fit_homography(
    pixels: ArrayLike,
    table_xy: ArrayLike,
    ransac_threshold_px: float = DEFAULT_RANSAC_PX,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fit pixel -> table with ``cv2.findHomography(RANSAC)``.

    Returns ``(H, inliers)`` with ``H`` normalised so ``H[2, 2] == 1`` and
    ``inliers`` a boolean mask over the input points. With exactly four points
    the fit is exact and every point is an inlier by construction; RANSAC only
    starts rejecting from five up.

    The RANSAC threshold is given in pixels, which is the unit a user can
    judge on the screen. OpenCV measures the threshold on the *destination*
    side (metres here), so it is converted with the median local scale of a
    first least-squares fit.
    """
    uv = _as_points(pixels, "pixels")
    xy = _as_points(table_xy, "table_xy")
    if len(uv) != len(xy):
        raise ValueError(f"{len(uv)} pixels but {len(xy)} table points")
    if len(uv) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} point pairs, got {len(uv)}")
    if float(ransac_threshold_px) <= 0.0:
        raise ValueError("ransac_threshold_px must be > 0")
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("opencv-python is required: py -3.12 -m pip install opencv-python") from exc

    src = uv.reshape(-1, 1, 2)
    dst = xy.reshape(-1, 1, 2)
    if len(uv) == MIN_POINTS:
        h0, _ = cv2.findHomography(src, dst, 0)
        if h0 is None:
            raise ValueError("the four points are degenerate (collinear or repeated)")
        return h0 / h0[2, 2], np.ones(len(uv), dtype=bool)

    # Metres per pixel around the points, from a plain least-squares fit, to
    # express the pixel threshold on the metric side.
    h0, _ = cv2.findHomography(src, dst, 0)
    if h0 is None:
        raise ValueError("points are degenerate (collinear or repeated)")
    scale = _median_metres_per_pixel(h0, uv)
    threshold_m = float(ransac_threshold_px) * scale
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold_m)
    if h is None:
        raise ValueError("RANSAC found no consistent set of points")
    inliers = np.asarray(mask, dtype=bool).reshape(-1)
    if int(inliers.sum()) < MIN_POINTS:
        raise ValueError(f"only {int(inliers.sum())} inliers; re-measure the points")
    return h / h[2, 2], inliers


def _median_metres_per_pixel(h: NDArray[np.float64], uv: NDArray[np.float64]) -> float:
    base = apply_homography(h, uv)
    du = apply_homography(h, uv + np.array([1.0, 0.0])) - base
    dv = apply_homography(h, uv + np.array([0.0, 1.0])) - base
    scale = float(np.median(np.hypot(np.linalg.norm(du, axis=1), np.linalg.norm(dv, axis=1)) / math.sqrt(2.0)))
    return max(scale, 1e-9)


def reprojection_errors(h: ArrayLike, pixels: ArrayLike, table_xy: ArrayLike) -> NDArray[np.float64]:
    """Per-point ``|H(pixel) - table_xy|`` in metres: the error the arm will see."""
    predicted = apply_homography(h, pixels)
    return np.linalg.norm(predicted - _as_points(table_xy, "table_xy"), axis=1)


def pixel_errors(h: ArrayLike, pixels: ArrayLike, table_xy: ArrayLike) -> NDArray[np.float64]:
    """Per-point ``|H^-1(table_xy) - pixel|`` in pixels: the error on the screen."""
    hm = np.asarray(h, dtype=np.float64).reshape(3, 3)
    back = apply_homography(np.linalg.inv(hm), table_xy)
    return np.linalg.norm(back - _as_points(pixels, "pixels"), axis=1)


def format_report(
    pixels: ArrayLike,
    table_xy: ArrayLike,
    h: ArrayLike,
    inliers: ArrayLike | None = None,
) -> str:
    """Human-readable per-point table plus RMS / max, metres and pixels."""
    uv = _as_points(pixels, "pixels")
    xy = _as_points(table_xy, "table_xy")
    err_m = reprojection_errors(h, uv, xy)
    err_px = pixel_errors(h, uv, xy)
    mask = np.ones(len(uv), dtype=bool) if inliers is None else np.asarray(inliers, dtype=bool)
    lines = ["  #   pixel (u, v)        table (x, y) m      err mm   err px  "]
    for i, ((u, v), (x, y), em, ep, ok) in enumerate(zip(uv, xy, err_m, err_px, mask), start=1):
        flag = "" if ok else "  OUTLIER"
        lines.append(f"{i:3d}   ({u:7.1f}, {v:7.1f})   ({x:+.4f}, {y:+.4f})   {em * 1000.0:7.2f}   {ep:6.2f}{flag}")
    used = err_m[mask]
    rms = float(np.sqrt(np.mean(used ** 2))) if len(used) else float("nan")
    lines.append(
        f"inliers {int(mask.sum())}/{len(uv)}   RMS {rms * 1000.0:.2f} mm   "
        f"max {float(used.max()) * 1000.0 if len(used) else float('nan'):.2f} mm"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML write
# ---------------------------------------------------------------------------

_TOP_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*):\s*(#.*)?$")


def _format_rows(indent: str, flat: Sequence[float]) -> list[str]:
    key = f"{indent}homography: ["
    pad = " " * len(key)
    rows = [", ".join(repr(float(v)) for v in flat[i:i + 3]) for i in (0, 3, 6)]
    return [f"{key}{rows[0]},", f"{pad}{rows[1]},", f"{pad}{rows[2]}]"]


def _splice_homography(text: str, camera_key: str, flat: Sequence[float]) -> str:
    """Replace or insert ``homography:`` inside the top-level ``camera_key`` block.

    Everything outside the entry (comments included) is untouched. Appends a
    new block when the key is absent.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{camera_key}:")
                  and _TOP_KEY_RE.match(ln)), None)
    if start is None:
        tail = [] if not lines or lines[-1].strip() == "" else [""]
        return "\n".join(lines + tail + [f"{camera_key}:"] + _format_rows("  ", flat)) + "\n"

    # Block extent: up to the next top-level key.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i][0].isspace() and not lines[i].lstrip().startswith("#"):
            end = i
            break
    indent = "  "
    for i in range(start + 1, end):
        stripped = lines[i].lstrip()
        if stripped and not stripped.startswith("#"):
            indent = lines[i][: len(lines[i]) - len(stripped)]
            break

    key_re = re.compile(r"^(\s+)homography:\s*(.*)$")
    for i in range(start + 1, end):
        m = key_re.match(lines[i])
        if m is None or lines[i].lstrip().startswith("#"):
            continue
        entry_indent, rest = m.group(1), m.group(2).split("#", 1)[0].strip()
        last = i
        if rest.startswith("["):
            depth = 0
            for j in range(i, end):
                depth += lines[j].count("[") - lines[j].count("]")
                last = j
                if depth <= 0:
                    break
        elif rest == "":
            # Block sequence of `- v` items, more indented than the key.
            for j in range(i + 1, end):
                s = lines[j].strip()
                if s.startswith("- ") or s == "" or s.startswith("#"):
                    if s.startswith("- ") and len(lines[j]) - len(lines[j].lstrip()) <= len(entry_indent):
                        break
                    last = j
                else:
                    break
            while last > i and lines[last].strip() == "":
                last -= 1
        return "\n".join(lines[:i] + _format_rows(entry_indent, flat) + lines[last + 1:]) + "\n"

    return "\n".join(lines[: start + 1] + _format_rows(indent, flat) + lines[start + 1:]) + "\n"


def pose_measured_note(config_path: str | Path, camera_key: str = "exterior_camera") -> str | None:
    """Return a warning when the config still treats the camera pose as a placeholder.

    Review GEO-2: this script measures only the homography, but perception also
    reads the camera's position/look_at/up and fx/fy/cx/cy. With
    ``pose_measured: false`` (the shipped hardware.yaml) only the first-order
    near-face step runs, along the configured look_at/up; the pinhole
    refinement and the lift prediction that confirms a carry stay off. The
    note tells the student what is still unmeasured instead of letting a
    fresh homography suggest the camera is fully calibrated. Returns ``None``
    when the pose is flagged as measured or the config cannot be read.
    """
    try:
        from mfw.config.schema import load_config  # noqa: PLC0415

        cfg = load_config(str(config_path))
        camera = getattr(cfg, camera_key)
    except Exception:  # noqa: BLE001 - advisory only; the write already succeeded
        return None
    if bool(getattr(camera, "pose_measured", False)):
        return None
    return (
        f"NOTE: {camera_key}.pose_measured is false. The homography is now measured, but the "
        "camera pose is not: bottom_center positions use only the first-order near-face step "
        "along the configured look_at/up, and the pinhole refinement and lift prediction "
        "(carry confirmation from the camera model) are off. Measure position/look_at/up with a "
        "tape and fx/fy/cx/cy with a checkerboard, then set pose_measured: true."
    )


def write_homography(
    yaml_path: str | Path,
    h: ArrayLike,
    camera_key: str = "exterior_camera",
    backup: bool = True,
) -> Path:
    """Write ``h`` (3x3, normalised) as ``<camera_key>.homography`` into ``yaml_path``.

    Validates the file with ``yaml.safe_load`` before and after, keeps every
    other key (``extends:`` first among them), the file's comments and its
    line-ending style (Python's text mode would otherwise turn an LF file into
    CRLF on Windows and make every line show up in a diff), and leaves
    ``<name>.bak`` beside it. Falls back to ``yaml.safe_dump`` of the parsed
    mapping only if the textual splice does not round-trip.
    """
    import yaml  # noqa: PLC0415

    path = Path(yaml_path)
    hm = np.asarray(h, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(hm)) or abs(hm[2, 2]) < 1e-12:
        raise ValueError("homography is non-finite or not normalisable")
    flat = [float(v) for v in (hm / hm[2, 2]).reshape(-1)]

    raw = path.read_bytes() if path.exists() else b""
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")
    data = yaml.safe_load(text) if text.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a YAML mapping")

    def _ok(candidate: str) -> bool:
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError:
            return False
        if not isinstance(parsed, dict) or not isinstance(parsed.get(camera_key), dict):
            return False
        got = parsed[camera_key].get("homography")
        if not isinstance(got, list) or len(got) != 9:
            return False
        if not np.allclose(np.asarray(got, dtype=np.float64), flat, rtol=0.0, atol=1e-15):
            return False
        return all(k in parsed for k in data) and parsed.get("extends") == data.get("extends")

    new_text = _splice_homography(text, camera_key, flat)
    if not _ok(new_text):
        merged = dict(data)
        camera = dict(merged.get(camera_key) or {})
        camera["homography"] = flat
        merged[camera_key] = camera
        new_text = yaml.safe_dump(merged, sort_keys=False, default_flow_style=None, width=120)
        if not _ok(new_text):
            raise RuntimeError(f"could not produce a valid {path} (this is a bug)")

    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    with path.open("w", encoding="utf-8", newline=newline) as fh:
        fh.write(new_text)
    return path


# ---------------------------------------------------------------------------
# Point files
# ---------------------------------------------------------------------------


def write_points_csv(path: str | Path, pixels: ArrayLike, table_xy: ArrayLike) -> Path:
    """``u,v,x,y`` rows so a fit can be repeated without the window."""
    uv, xy = _as_points(pixels, "pixels"), _as_points(table_xy, "table_xy")
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["u_px", "v_px", "x_m", "y_m"])
        for (u, v), (x, y) in zip(uv, xy):
            writer.writerow([f"{u:.3f}", f"{v:.3f}", f"{x:.5f}", f"{y:.5f}"])
    return p


def read_points_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`write_points_csv`; header optional."""
    uv: list[tuple[float, float]] = []
    xy: list[tuple[float, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                u, v, x, y = (float(c) for c in row[:4])
            except ValueError:
                continue  # header
            uv.append((u, v))
            xy.append((x, y))
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2), np.asarray(xy, dtype=np.float64).reshape(-1, 2)


def parse_xy(text: str) -> tuple[float, float]:
    """``"0.18 0.05"`` / ``"0.18, 0.05"`` -> ``(0.18, 0.05)`` metres."""
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise ValueError(f"expected two numbers (x y in metres), got {text!r}")
    return float(parts[0]), float(parts[1])


# ---------------------------------------------------------------------------
# Frames, clicks, touch (interactive; not unit-tested)
# ---------------------------------------------------------------------------


def grab_frame(image: str | None = None, jetson: str | None = None) -> NDArray[np.uint8]:
    """RGB frame from ``--image`` or the robot server's webcam."""
    if image:
        import cv2  # noqa: PLC0415

        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {image!r}")
        return np.ascontiguousarray(bgr[:, :, ::-1])
    if jetson:
        from mfw.hardware.jetson_client import JetsonClient  # noqa: PLC0415
        from mfw.hardware.remote_camera import decode_jpeg  # noqa: PLC0415

        host, _, port_text = jetson.partition(":")
        with JetsonClient(host or "127.0.0.1", int(port_text or 5560)) as client:
            client.connect()
            reply = client.get_frame()
        return decode_jpeg(bytes(reply["jpeg"]))
    raise ValueError("give --image PATH or --jetson HOST[:PORT]")


def collect_clicks(rgb: NDArray[np.uint8], min_points: int = MIN_POINTS,
                   window: str = "calibrate_table") -> list[tuple[float, float]]:
    """Click table points in a cv2 window. Enter/Space finishes, ``u`` undoes, Esc aborts."""
    import cv2  # noqa: PLC0415

    base = np.ascontiguousarray(rgb[:, :, ::-1])
    points: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Click >= {min_points} points where objects meet the table. Enter = done, u = undo, Esc = abort.")
    try:
        while True:
            canvas = base.copy()
            for i, (u, v) in enumerate(points, start=1):
                cv2.circle(canvas, (int(u), int(v)), 6, (0, 0, 255), 2)
                cv2.putText(canvas, str(i), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(canvas, f"{len(points)} point(s); Enter=done u=undo Esc=abort", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32) and len(points) >= min_points:
                break
            if key in (ord("u"), ord("U")) and points:
                points.pop()
            if key == 27:
                raise KeyboardInterrupt("calibration aborted")
    finally:
        cv2.destroyWindow(window)
    return points


TOUCH_PROBE_MARGIN_M = 0.0
"""How far below the table plane (and below ``scene.workspace_min[2]``) a
touch target may go: none. It used to be 5 mm "to press the tip onto a
mis-levelled table", but the table plane here is the *model's* z = 0, which
before ``calibrate_servos.py`` zero/limit can sit centimetres below the real
surface -- and since the keepalive the Jetson held that press indefinitely
while the student lined up pixels (review, motion lens). Bring the tip to a
hair above the table; the homography needs XY, not contact force. Run
``calibrate_servos.py`` ``zero``/``limit`` before ``--touch``."""
TOUCH_MAX_NUDGE_M = 0.02
"""Largest single ``x/y/z`` nudge, metres. Review P2: a student meaning a 5 mm
descent typed ``z -50`` and IK happily solved 5 cm *into* the table."""


TOUCH_JAW_WIDTH_M = 0.002
"""Jaw width the probe closes to for a defined tip: 2 mm, not 0. Width 0 is
the calibrated closed end point (``pulse_closed_us``); an MG90S driven onto
its own jaws stalls for as long as it is held (review, finding 2)."""
TOUCH_HOLD_MIN_Z_M = 0.005
"""Below this model height (5 mm) the probe stops its heartbeat. The model's
z = 0 can sit millimetres under the real table (placeholder link lengths, a
flexing PLA arm), so a tip "at the table" may really be pressing into it with
the shoulder/elbow stalled. Without beats the robot server's host timeout
(5 s) relaxes the arm; ``ok`` still records the commanded pose and the next
move re-attaches slowly from it."""
TOUCH_MAX_HOLD_S = 90.0
"""Longest the probe keeps the arm energised while nobody types at the
``touch>`` prompt; after that the heartbeat stops (a hung ``input()`` or a
student who walked away must not hold five servos forever)."""


def touch_target_refusal(
    target: Sequence[float],
    current: Sequence[float] | None,
    workspace_min: Sequence[float],
    workspace_max: Sequence[float],
    nudge: bool = False,
) -> str | None:
    """Why a ``--touch`` move to ``target`` must not be sent, or ``None`` if it may.

    Refused, in this order: a nudge longer than :data:`TOUCH_MAX_NUDGE_M`
    (``current`` is the TCP before it); a target outside the workspace in X or
    Y; a target below ``max(workspace_min[2] - margin, -margin)`` with
    :data:`TOUCH_PROBE_MARGIN_M` (0) as the margin, i.e. never below the
    model's table plane whatever the configured floor. Pure, so it is unit
    tested without an arm.
    """
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    lo = np.asarray(workspace_min, dtype=np.float64).reshape(-1)
    hi = np.asarray(workspace_max, dtype=np.float64).reshape(-1)
    if t.shape[0] != 3 or lo.shape[0] != 3 or hi.shape[0] != 3:
        return "touch target and workspace bounds must be (x, y, z)"
    if not np.all(np.isfinite(t)):
        return "touch target is not a finite point"
    if nudge and current is not None:
        step = float(np.linalg.norm(t - np.asarray(current, dtype=np.float64).reshape(-1)))
        if step > TOUCH_MAX_NUDGE_M + 1e-9:
            return (
                f"refused: a single nudge of {step * 1000.0:.0f} mm exceeds the "
                f"{TOUCH_MAX_NUDGE_M * 1000.0:.0f} mm cap (did you mean millimetres? "
                "'z -5' is 5 mm); nudge in smaller steps or use 'go X Y Z'"
            )
    for axis, name in ((0, "x"), (1, "y")):
        if not lo[axis] <= t[axis] <= hi[axis]:
            return (
                f"refused: {name} = {t[axis]:+.3f} m is outside the workspace "
                f"[{lo[axis]:+.3f}, {hi[axis]:+.3f}] (scene.workspace_min/max)"
            )
    floor = max(float(lo[2]) - TOUCH_PROBE_MARGIN_M, -TOUCH_PROBE_MARGIN_M)
    if t[2] < floor - 1e-9:
        return (
            f"refused: z = {t[2] * 1000.0:+.1f} mm is below the touch floor {floor * 1000.0:+.1f} mm "
            "(the jaw would press into the table and stall the shoulder/elbow servos); bring the "
            "tip to a hair ABOVE the table -- the homography needs XY, not contact"
        )
    return None


class TouchProbe:
    """``--touch`` hook: jog the jaw tip onto a point and read the TCP from FK.

    Talks to ``robot_server.py`` through ``JetsonClient`` and to the arm model
    through ``PlanarKinematics`` (both imported lazily), so the typed path
    never needs pyzmq. One point at a time, a tiny REPL:

        go X Y [Z]     move the TCP to metres (Z default: hover height)
        x +5 / y -3 / z -2   nudge by millimetres (at most 20 mm per nudge)
        ok             record the current TCP (x, y) for this pixel
        type X Y       give up on touching and type the table point
        skip           drop this pixel

    Every move is screened by :func:`touch_target_refusal` first (review P2):
    nothing below the table floor, nothing outside the workspace XY, no nudge
    over 20 mm. Refusals are printed and nothing is sent to the arm.

    Heartbeat. The Jetson relaxes an arm nobody has talked to for
    ``host_timeout_s`` (5 s), and a student lining up a pixel at the
    ``touch>`` prompt is silent for far longer than that. Without a heartbeat
    the arm went limp at the prompt and the *next nudge* re-attached it right
    next to the table (audit P2). So while the probe is open -- including
    every wait at the prompt -- a :class:`~mfw.hardware.jetson_client.Heartbeat`
    sends ``get_state`` every ``hardware.heartbeat_s`` (1 s) from its own
    thread. It is started only once the bridge is known to have a position
    (so the refusal below never leaves a thread behind) and stopped by
    :meth:`close`; a crashed script stops beating and the server still
    relaxes the arm. The hold is bounded (review, finding 2): the model's
    table plane is not the real table, so once the commanded TCP is below
    :data:`TOUCH_HOLD_MIN_Z_M` (5 mm) the heartbeat stops and the host
    timeout relaxes an arm that may be pressing into the table; and after
    :data:`TOUCH_MAX_HOLD_S` without a typed command it stops too. The next
    move re-attaches slowly from the commanded pose and, when it ends above
    5 mm, beats again. The jaw closes to :data:`TOUCH_JAW_WIDTH_M` (2 mm),
    never onto its end point. ``ok`` records the *commanded* pose.
    Refuses to start while the bridge knows no
    position (right after ``robot_server.py`` starts): run
    ``calibrate_servos.py`` ``home`` with the arm hand-posed at home first.
    """

    HOVER_Z = 0.03
    STEP_S = 1.5

    def __init__(
        self,
        jetson: str,
        config_path: str | Path,
        heartbeat_s: float | None = None,
        max_hold_s: float = TOUCH_MAX_HOLD_S,
        clock: Any = time.monotonic,
    ) -> None:
        """``heartbeat_s`` overrides ``hardware.heartbeat_s`` (tests use a
        short period; ``0`` disables the heartbeat -- only a test wants that).
        ``max_hold_s``/``clock``: the idle bound on holding the arm."""
        from mfw.config.schema import load_config  # noqa: PLC0415
        from mfw.hardware.jetson_client import JetsonClient  # noqa: PLC0415
        from mfw.hardware.kinematics import PlanarKinematics  # noqa: PLC0415

        cfg = load_config(config_path)
        self.cfg = cfg
        self.heartbeat = None
        self.max_hold_s = float(max_hold_s)
        self._clock = clock
        self._last_activity = clock()
        self.hold_stopped = ""
        """Why the hold bound last stopped the heartbeat ("" while holding)."""
        host, _, port_text = jetson.partition(":")
        host, port = host or "127.0.0.1", int(port_text or 5560)
        self.client = JetsonClient(host, port,
                                   cfg.hardware.request_timeout_s, cfg.hardware.trajectory_timeout_margin_s)
        self.client.connect()
        self.kin = PlanarKinematics(cfg.hardware.arm, cfg.robot.arm_joint_names)
        self.max_v = float(cfg.motion.max_joint_velocity)
        state = self.client.get_state()
        if state.get("bridge_position_known") is False:
            self.client.close()
            raise RuntimeError(
                "the servo bridge has no known position (the Uno resets whenever robot_server.py "
                "opens its port): its first frame would snap every servo to the target. Hand-pose "
                "the arm at home and run `py -3.12 scripts/calibrate_servos.py --jetson "
                f"{jetson}` -> `home` first, then start --touch again"
            )
        self._host, self._port = host, port
        self._period = float(cfg.hardware.heartbeat_s if heartbeat_s is None else heartbeat_s)
        try:
            self.q = np.asarray(state["q"], dtype=np.float64)
            self.tcp = np.asarray(self.kin.fk(self.q).position, dtype=np.float64)
            self._update_hold()
            self.client.set_gripper(TOUCH_JAW_WIDTH_M)  # nearly closed jaw = a defined tip, no stall
        except BaseException:
            self.close()
            raise

    def _start_heartbeat(self) -> None:
        from mfw.hardware.jetson_client import Heartbeat  # noqa: PLC0415

        if self._period <= 0.0 or (self.heartbeat is not None and self.heartbeat.running):
            return
        self._last_activity = self._clock()
        self.hold_stopped = ""
        self.heartbeat = Heartbeat(self._host, self._port, period_s=self._period, on_state=self._on_beat,
                                   request_timeout_s=min(float(self.cfg.hardware.request_timeout_s), 2.0)).start()

    def _stop_heartbeat(self, why: str) -> None:
        heartbeat = self.heartbeat
        if heartbeat is not None:
            heartbeat.stop()
            self.heartbeat = None
            self.hold_stopped = why
            print(f"  {why}: heartbeat stopped, the robot server relaxes the arm within its host "
                  "timeout (5 s); type ok now if the tip is on the point -- the next move "
                  "re-attaches the arm slowly from the commanded pose")

    def _on_beat(self, _state: dict[str, Any]) -> None:
        """Heartbeat thread: stop beating once nobody has typed for ``max_hold_s``."""
        heartbeat = self.heartbeat
        if heartbeat is not None and self._clock() - self._last_activity > self.max_hold_s:
            self.hold_stopped = f"idle for over {self.max_hold_s:.0f} s at the prompt"
            heartbeat.stop()  # from its own thread: the loop ends after this beat

    def _update_hold(self) -> None:
        """Hold (beat) only while the commanded TCP is clear of the table."""
        if float(self.tcp[2]) < TOUCH_HOLD_MIN_Z_M:
            self._stop_heartbeat(f"TCP {self.tcp[2] * 1000.0:+.1f} mm is within "
                                 f"{TOUCH_HOLD_MIN_Z_M * 1000.0:.0f} mm of the model table")
        else:
            self._start_heartbeat()

    def close(self) -> None:
        """Stop the heartbeat (the server relaxes the arm 5 s later), then
        close the connection. Idempotent."""
        heartbeat = getattr(self, "heartbeat", None)
        if heartbeat is not None:
            heartbeat.stop()
            self.heartbeat = None
        self.client.close()

    def _pose_down(self, position: NDArray[np.float64], tilt: float):
        from mfw.grasp.generator import build_grasp_pose  # noqa: PLC0415
        from mfw.hardware.grasp import TopDownGraspGenerator  # noqa: PLC0415

        approach, closing = TopDownGraspGenerator._frame(math.atan2(position[1], position[0]), tilt, self.kin.jaw_axis)
        return build_grasp_pose(position, closing, approach)

    def move_to(self, position: Sequence[float], nudge: bool = False) -> bool:
        target = np.asarray(position, dtype=np.float64)
        refusal = touch_target_refusal(
            target, self.tcp, self.cfg.scene.workspace_min, self.cfg.scene.workspace_max, nudge=nudge
        )
        if refusal is not None:
            print(f"  {refusal}")
            return False
        q = None
        for tilt_deg in (0.0, 15.0, 30.0, 45.0):
            q = self.kin.ik(self._pose_down(target, math.radians(tilt_deg)), seed=self.q)
            if q is not None:
                break
        if q is None:
            print(f"  unreachable: {np.round(target, 3)}")
            return False
        duration = max(self.STEP_S, float(np.max(np.abs(q - self.q))) / max(self.max_v, 1e-6))
        reply = self.client.follow_trajectory(np.vstack([self.q, q]), duration)
        self.q = np.asarray(reply.get("q", q), dtype=np.float64)
        self.tcp = np.asarray(self.kin.fk(self.q).position, dtype=np.float64)
        print(f"  TCP now ({self.tcp[0]:+.4f}, {self.tcp[1]:+.4f}, {self.tcp[2]:+.4f})")
        self._update_hold()
        return True

    def measure(self, index: int, uv: tuple[float, float]) -> tuple[float, float] | None:
        print(f"\nPoint {index} at pixel ({uv[0]:.0f}, {uv[1]:.0f}). Jog the jaw tip onto it. "
              "Commands: go X Y [Z] | x/y/z +mm | ok | type X Y | skip")
        while True:
            try:
                line = input("touch> ").strip().lower()
            except EOFError:
                return None
            self._last_activity = self._clock()
            if not line:
                continue
            parts = line.split()
            try:
                if parts[0] == "ok":
                    return float(self.tcp[0]), float(self.tcp[1])
                if parts[0] == "skip":
                    return None
                if parts[0] == "type":
                    return parse_xy(" ".join(parts[1:]))
                if parts[0] == "go" and len(parts) in (3, 4):
                    z = float(parts[3]) if len(parts) == 4 else self.HOVER_Z
                    self.move_to((float(parts[1]), float(parts[2]), z))
                    continue
                if parts[0] in ("x", "y", "z") and len(parts) == 2:
                    delta = np.zeros(3)
                    delta["xyz".index(parts[0])] = float(parts[1]) / 1000.0
                    self.move_to(self.tcp + delta, nudge=True)
                    continue
            except ValueError as exc:
                print(f"  {exc}")
                continue
            print("  unknown command")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help="calibrate on a saved frame")
    src.add_argument("--from-csv", help="skip the window; re-fit u,v,x,y rows from this file")
    parser.add_argument("--jetson", metavar="HOST[:PORT]", help="robot_server.py: frame source (and the arm for --touch)")
    parser.add_argument("--config", default="configs/hardware.yaml", help="YAML to write exterior_camera.homography into")
    parser.add_argument("--camera-key", default="exterior_camera")
    parser.add_argument("--touch", action="store_true", help="measure table XY by touching each point with the jaw (needs --jetson)")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS)
    parser.add_argument("--ransac-px", type=float, default=DEFAULT_RANSAC_PX)
    parser.add_argument("--save-csv", default=None, help="where to keep the point pairs (default: <config>.points.csv)")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, do not write the YAML")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = _ROOT / args.config
    if args.touch and not args.jetson:
        print("--touch needs --jetson HOST[:PORT] (the arm lives behind the robot server)")
        return 2

    if args.from_csv:
        pixels, table = read_points_csv(args.from_csv)
    else:
        try:
            rgb = grab_frame(args.image, args.jetson)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc))
            return 2
        try:
            clicks = collect_clicks(rgb, args.min_points)
        except KeyboardInterrupt:
            print("aborted; nothing written")
            return 1
        probe = TouchProbe(args.jetson, config_path) if args.touch else None
        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        try:
            for i, uv in enumerate(clicks, start=1):
                if probe is not None:
                    xy = probe.measure(i, uv)
                else:
                    xy = None
                    while xy is None:
                        text = input(f"point {i} at pixel ({uv[0]:.0f}, {uv[1]:.0f}) -> table x y in metres (or skip): ").strip()
                        if text.lower() == "skip":
                            break
                        try:
                            xy = parse_xy(text)
                        except ValueError as exc:
                            print(f"  {exc}")
                if xy is not None:
                    pairs.append((uv, xy))
        finally:
            if probe is not None:
                probe.close()
        if len(pairs) < args.min_points:
            print(f"only {len(pairs)} usable point(s); need {args.min_points}. Nothing written.")
            return 1
        pixels = np.array([p for p, _ in pairs])
        table = np.array([t for _, t in pairs])
        csv_path = Path(args.save_csv) if args.save_csv else config_path.with_name(config_path.name + ".points.csv")
        write_points_csv(csv_path, pixels, table)
        print(f"point pairs saved to {csv_path}")

    try:
        h, inliers = fit_homography(pixels, table, args.ransac_px)
    except (ValueError, RuntimeError) as exc:
        print(f"fit failed: {exc}")
        return 1
    print(format_report(pixels, table, h, inliers))
    print("homography (row-major):", ", ".join(repr(float(v)) for v in h.reshape(-1)))
    note = pose_measured_note(config_path, camera_key=args.camera_key)
    if args.dry_run:
        if note:
            print(note)
        return 0
    out = write_homography(config_path, h, camera_key=args.camera_key)
    print(f"wrote {args.camera_key}.homography into {out} (backup: {out.name}.bak)")
    if note:
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())

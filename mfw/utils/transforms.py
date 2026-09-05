"""Rigid-body transform math.

Pure NumPy. This module must never import Isaac Sim, so it can be unit-tested
in any interpreter.

Conventions (chosen to match Isaac Sim, and converted at the GR00T boundary):

* Quaternions are **scalar-first** ``(w, x, y, z)``. Isaac Sim's
  ``isaacsim.core`` APIs use this ordering throughout; SciPy and many ROS
  message types use scalar-last, so conversions are explicit and named.
* Rotation matrices are row-major 3x3, applied to column vectors: ``p' = R @ p``.
* A "pose" is a position ``(3,)`` plus a quaternion ``(4,)``, always expressed
  in a named frame. See :class:`mfw.core.types.Pose`.
* ``rot6d`` is the continuous 6-D rotation representation of Zhou et al. (2019):
  the first two *columns* of the rotation matrix, flattened. GR00T's
  ``eef_9d`` state is ``[xyz(3), rot6d(6)]``, so this is a hard external
  contract, not a stylistic choice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "quat_to_matrix",
    "matrix_to_quat",
    "quat_multiply",
    "quat_conjugate",
    "quat_rotate",
    "quat_scalar_first_to_last",
    "quat_scalar_last_to_first",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "make_transform",
    "invert_transform",
    "transform_points",
    "pose_to_matrix",
    "matrix_to_pose",
    "quat_angular_distance",
    "orthonormalize",
    "look_at_quat",
]

_EPS = 1e-12


def _as_vec(x: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} elements, got shape {np.shape(x)}")
    return arr


def quat_to_matrix(quat: ArrayLike) -> NDArray[np.float64]:
    """Convert a scalar-first quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    w, x, y, z = _as_vec(quat, 4, "quat")
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < _EPS:
        raise ValueError("Cannot convert a zero-norm quaternion to a rotation matrix")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: ArrayLike) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion ``(w, x, y, z)``.

    Uses Shepperd's branch-selection method: pick the largest of the four
    candidate denominators so the division is always well-conditioned. The naive
    ``w``-only formula loses precision near 180-degree rotations, which is
    exactly where wrist-flip grasps live.

    The returned quaternion is canonicalised to ``w >= 0`` so that the two
    double-cover representations of the same rotation compare equal.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3, got {m.shape}")

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quat_multiply(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` of two scalar-first quaternions.

    Composition order matches matrix multiplication: ``quat_to_matrix(a*b) ==
    quat_to_matrix(a) @ quat_to_matrix(b)``.
    """
    aw, ax, ay, az = _as_vec(a, 4, "a")
    bw, bx, by, bz = _as_vec(b, 4, "b")
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conjugate(quat: ArrayLike) -> NDArray[np.float64]:
    """Conjugate of a scalar-first quaternion; the inverse for unit quaternions."""
    w, x, y, z = _as_vec(quat, 4, "quat")
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_rotate(quat: ArrayLike, vec: ArrayLike) -> NDArray[np.float64]:
    """Rotate a 3-vector by a scalar-first quaternion."""
    return quat_to_matrix(quat) @ _as_vec(vec, 3, "vec")


def quat_scalar_first_to_last(quat: ArrayLike) -> NDArray[np.float64]:
    """``(w, x, y, z)`` -> ``(x, y, z, w)`` for SciPy / ROS interop."""
    w, x, y, z = _as_vec(quat, 4, "quat")
    return np.array([x, y, z, w], dtype=np.float64)


def quat_scalar_last_to_first(quat: ArrayLike) -> NDArray[np.float64]:
    """``(x, y, z, w)`` -> ``(w, x, y, z)`` for SciPy / ROS interop."""
    x, y, z, w = _as_vec(quat, 4, "quat")
    return np.array([w, x, y, z], dtype=np.float64)


def orthonormalize(matrix: ArrayLike) -> NDArray[np.float64]:
    """Project a near-rotation 3x3 matrix onto SO(3) via SVD.

    Numerical drift accumulates when rotations are composed over a long episode,
    and ``rot6d`` decoded from a neural network is never exactly orthonormal.
    Both paths funnel through here. The determinant is forced positive so a
    reflection can never be returned as a rotation.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3, got {m.shape}")
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def matrix_to_rot6d(matrix: ArrayLike) -> NDArray[np.float64]:
    """Flatten the first two *columns* of a rotation matrix into GR00T's rot6d.

    Column-major ordering is what makes :func:`rot6d_to_matrix` the exact
    inverse; getting this backwards silently transposes every orientation the
    policy sees, which shows up as a plausible-looking but consistently wrong
    wrist angle.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3, got {m.shape}")
    return np.concatenate([m[:, 0], m[:, 1]])


def rot6d_to_matrix(rot6d: ArrayLike) -> NDArray[np.float64]:
    """Decode GR00T's rot6d into a proper rotation matrix by Gram-Schmidt.

    Network outputs are unconstrained, so the two 3-vectors are neither unit
    length nor orthogonal. This reconstructs the third column via cross product,
    which is what makes the representation continuous and singularity-free.
    """
    v = _as_vec(rot6d, 6, "rot6d")
    a1, a2 = v[:3], v[3:]

    n1 = np.linalg.norm(a1)
    if n1 < _EPS:
        raise ValueError("rot6d first basis vector is degenerate (zero length)")
    b1 = a1 / n1

    a2_perp = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_perp)
    if n2 < _EPS:
        raise ValueError("rot6d basis vectors are collinear; cannot form a rotation")
    b2 = a2_perp / n2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def make_transform(position: ArrayLike, quat: ArrayLike) -> NDArray[np.float64]:
    """Build a 4x4 homogeneous transform from a position and scalar-first quaternion."""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_to_matrix(quat)
    t[:3, 3] = _as_vec(position, 3, "position")
    return t


def invert_transform(transform: ArrayLike) -> NDArray[np.float64]:
    """Invert a 4x4 rigid transform analytically (``R^T``, ``-R^T t``).

    Cheaper and numerically better behaved than ``np.linalg.inv`` because it
    exploits orthonormality instead of doing a general LU solve.
    """
    m = np.asarray(transform, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {m.shape}")
    r = m[:3, :3]
    t = m[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def transform_points(transform: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points.

    Kept vectorised because it runs on full point clouds (~300k points per
    depth frame) inside the perception loop.
    """
    m = np.asarray(transform, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {m.shape}")
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {p.shape}")
    return p @ m[:3, :3].T + m[:3, 3]


def pose_to_matrix(position: ArrayLike, quat: ArrayLike) -> NDArray[np.float64]:
    """Alias of :func:`make_transform`, named for call sites that read as poses."""
    return make_transform(position, quat)


def matrix_to_pose(transform: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a 4x4 transform into ``(position, scalar-first quaternion)``.

    The rotation block is re-orthonormalised first so that accumulated drift
    cannot produce an invalid quaternion.
    """
    m = np.asarray(transform, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {m.shape}")
    return m[:3, 3].copy(), matrix_to_quat(orthonormalize(m[:3, :3]))


def look_at_quat(
    eye: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)
) -> NDArray[np.float64]:
    """Orientation for a USD camera at ``eye`` aimed at ``target``.

    Returns a scalar-first quaternion in the **USD camera convention**: the
    camera looks down its local **-Z** with **+Y** up. (The OpenCV convention
    used by the projection maths is +Z forward / +Y down; the two differ by a
    180-degree flip about X, applied once in
    :meth:`mfw.vision.camera.Camera.get_extrinsics`.)

    Placing a camera by a hand-written quaternion is guesswork, and a camera
    aimed slightly wrong produces an image full of empty floor -- which reads
    downstream as "the detector found nothing" rather than "the camera is
    pointing the wrong way". Deriving orientation from a look-at target makes
    the intent explicit and checkable.
    """
    eye_v = _as_vec(eye, 3, "eye")
    target_v = _as_vec(target, 3, "target")
    up_v = _as_vec(up, 3, "up")

    forward = target_v - eye_v
    norm = np.linalg.norm(forward)
    if norm < _EPS:
        raise ValueError("look_at_quat: eye and target coincide, direction is undefined")
    forward /= norm

    # USD cameras look along -Z, so the camera's +Z axis points backwards.
    z_axis = -forward

    x_axis = np.cross(up_v, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        # Looking straight up or down: the chosen up vector is parallel to the
        # view direction and cannot disambiguate roll. Fall back to a world axis
        # that is guaranteed independent.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(fallback, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis /= x_norm

    y_axis = np.cross(z_axis, x_axis)
    return matrix_to_quat(np.stack([x_axis, y_axis, z_axis], axis=1))


def quat_angular_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Geodesic angle in radians between two orientations, in ``[0, pi]``.

    Uses ``|<a, b>|`` so the quaternion double cover is handled: ``q`` and
    ``-q`` are the same rotation and must report zero distance. Used by grasp
    scoring and by the skill layer's success checks.
    """
    qa = _as_vec(a, 4, "a")
    qb = _as_vec(b, 4, "b")
    qa = qa / max(float(np.linalg.norm(qa)), _EPS)
    qb = qb / max(float(np.linalg.norm(qb)), _EPS)
    dot = abs(float(np.dot(qa, qb)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

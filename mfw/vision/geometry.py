"""Point-cloud geometry for perception.

Pure NumPy. This module must never import Isaac Sim, so the geometry that
grasping depends on is unit-testable without a simulator.

The central operation is fitting an oriented bounding box to a cluster of
points. Everything downstream -- grasp width, approach axes, placement
clearance -- is derived from that box, so its conventions matter:

* Boxes are **gravity-aligned** by default: local +Z is world up, and only the
  yaw about Z is fitted from the data. Objects on a table are supported by
  gravity, and a full 3-DoF PCA on a partial, single-viewpoint cloud routinely
  tilts the box by 10-20 degrees, which tilts every grasp derived from it. Yaw
  is the only rotation that is actually observable from a top-down-ish view.
* Extents are **full side lengths**, not half-extents.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mfw.utils import transforms as tf

__all__ = [
    "voxel_downsample",
    "remove_statistical_outliers",
    "fit_oriented_bbox",
    "principal_axes",
    "cluster_by_euclidean_distance",
    "points_above_plane",
]

_EPS = 1e-12


def voxel_downsample(
    points: NDArray[np.float64], voxel_size: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reduce a cloud to one point per occupied voxel.

    Returns ``(downsampled_points, representative_indices)`` so per-point
    attributes (colour, segmentation id) can be carried through by indexing.

    A depth frame is ~300k points and perception runs before every action;
    without this, outlier removal and clustering dominate the cycle. Uses an
    exact integer-key grouping rather than a spatial tree -- it is O(n) and has
    no tuning parameters beyond the leaf size.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be > 0")
    if pts.shape[0] == 0:
        return pts.copy(), np.empty(0, dtype=np.int64)

    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    return pts[first_indices], first_indices


def remove_statistical_outliers(
    points: NDArray[np.float64], k: int = 8, std_ratio: float = 2.0
) -> NDArray[np.bool_]:
    """Flag points whose mean distance to their ``k`` nearest neighbours is anomalous.

    Returns a boolean keep-mask.

    Depth sensors produce "flying pixels" at depth discontinuities -- points
    that interpolate across an object's silhouette and land in empty space
    between the object and the background. Left in, they inflate the bounding
    box and produce grasps that close on nothing.

    Uses a brute-force distance matrix, which is fine because this runs on
    already-clustered, already-downsampled segments (hundreds of points), not on
    the full frame.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n <= k + 1:
        # Too few points for the statistic to mean anything; keeping them all is
        # safer than deleting a small but real object.
        return np.ones(n, dtype=bool)

    diff = pts[:, None, :] - pts[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(distances, np.inf)

    neighbour_distances = np.sort(distances, axis=1)[:, :k]
    mean_distances = neighbour_distances.mean(axis=1)

    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    return mean_distances <= threshold


def principal_axes(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(eigenvectors_as_columns, eigenvalues)`` sorted by descending variance."""
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order], eigenvalues[order]


def fit_oriented_bbox(
    points: NDArray[np.float64],
    gravity_aligned: bool = True,
    trim_percentile: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit an oriented bounding box to a point cluster.

    Returns ``(center, quat, extents)`` where ``quat`` is scalar-first and
    ``extents`` are full side lengths along the box's own axes.

    With ``gravity_aligned`` (the default) only the yaw about world +Z is
    fitted. See the module docstring for why full 3-DoF PCA is the wrong choice
    on single-viewpoint tabletop data.

    ``trim_percentile`` sizes the box from the ``[p, 100-p]`` percentile range
    of the projected coordinates instead of the raw min/max. This is not
    cosmetic smoothing -- it is what makes the box usable at all. Instance masks
    include a halo of edge pixels whose depth interpolates between the object
    surface and the background, producing a "skirt" of points trailing from the
    silhouette onto the support surface. Measured on this scene, min/max sizing
    inflated a 0.12 m box to 0.22 m; since grasp width comes straight from these
    extents, that difference is the gap between a grasp that fits and one the
    gripper cannot span. Set to 0 for exact min/max.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a box, got {pts.shape[0]}")
    if not 0.0 <= trim_percentile < 50.0:
        raise ValueError(f"trim_percentile must be in [0, 50), got {trim_percentile}")

    if gravity_aligned:
        rotation = _yaw_only_rotation(pts, trim_percentile=trim_percentile)
    else:
        eigenvectors, _ = principal_axes(pts)
        rotation = tf.orthonormalize(eigenvectors)

    local = (pts - pts.mean(axis=0)) @ rotation

    if trim_percentile > 0.0 and pts.shape[0] >= 20:
        local_min = np.percentile(local, trim_percentile, axis=0)
        local_max = np.percentile(local, 100.0 - trim_percentile, axis=0)
    else:
        # Too few points for percentiles to be meaningful; a small cluster is
        # mostly signal anyway.
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)

    extents = local_max - local_min
    local_center = (local_min + local_max) / 2.0
    center = pts.mean(axis=0) + rotation @ local_center

    return center, tf.matrix_to_quat(rotation), extents


def _yaw_only_rotation(
    points: NDArray[np.float64], trim_percentile: float = 2.0, num_angles: int = 90
) -> NDArray[np.float64]:
    """Yaw about world +Z, chosen to **minimise the footprint area**.

    Not PCA. PCA picks the direction of greatest variance, which is the right
    answer for an elongated object and the wrong one for a square: a square
    footprint has a near-degenerate covariance, so the "dominant" direction is
    noise and typically lands near 45 degrees -- where the measured side length
    is ``s * sqrt(2)``, a 41% over-estimate. Measured here, that inflated a
    0.12 m box to 0.21 m, and grasp width is taken straight from these extents.

    A minimum-area rectangle has no such degeneracy: for a square, every
    orientation that aligns with a side is a true minimum. Because the box is
    gravity-aligned, only yaw is free, and a rectangle is symmetric under 90-degree
    rotation, so sweeping ``[0, 90)`` is exhaustive.
    """
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    if xy.shape[0] < 3 or not np.all(np.isfinite(xy)) or np.allclose(xy, 0.0):
        # Degenerate footprint (a vertical column of points): yaw is
        # unobservable, so any consistent choice is as good as another.
        return np.eye(3)

    angles = np.linspace(0.0, np.pi / 2.0, num_angles, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    # Project every point onto every candidate axis pair at once: (A, N).
    projected_u = np.outer(cos, xy[:, 0]) + np.outer(sin, xy[:, 1])
    projected_v = -np.outer(sin, xy[:, 0]) + np.outer(cos, xy[:, 1])

    if trim_percentile > 0.0 and xy.shape[0] >= 20:
        # Same halo rejection as the extents themselves; without it a few skirt
        # points steer the chosen angle.
        lo, hi = trim_percentile, 100.0 - trim_percentile
        width = np.percentile(projected_u, hi, axis=1) - np.percentile(projected_u, lo, axis=1)
        height = np.percentile(projected_v, hi, axis=1) - np.percentile(projected_v, lo, axis=1)
    else:
        width = projected_u.max(axis=1) - projected_u.min(axis=1)
        height = projected_v.max(axis=1) - projected_v.min(axis=1)

    best = int(np.argmin(width * height))
    angle = float(angles[best])

    # Canonicalise so the longer side is local +X. Without this a near-square
    # object can swap its axes between frames, which the tracker would read as a
    # 90-degree rotation.
    if width[best] < height[best]:
        angle += np.pi / 2.0

    x_axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


#: Hue ranges in degrees, mapped to the words operators actually use. Ordered
#: bands rather than colour exemplars, because hue is what survives lighting.
_HUE_BANDS: tuple[tuple[float, float, str], ...] = (
    (345.0, 360.0, "red"),
    (0.0, 15.0, "red"),
    (15.0, 45.0, "orange"),
    (45.0, 70.0, "yellow"),
    (70.0, 165.0, "green"),
    (165.0, 200.0, "cyan"),
    (200.0, 255.0, "blue"),
    (255.0, 290.0, "purple"),
    (290.0, 345.0, "pink"),
)

#: Below this saturation an object has no meaningful hue and is named by
#: lightness instead. 0.15 sits above sensor noise on a grey surface and well
#: below a lit coloured one (measured on this scene: 0.28-0.38).
_NEUTRAL_SATURATION = 0.15


def dominant_color_name(colors: NDArray[np.uint8] | None) -> str:
    """Name the dominant colour of a set of per-point RGB samples.

    Classifies by **hue**, not by distance to RGB exemplars. That distinction is
    the whole difficulty: rendered surfaces are lifted toward white by lighting,
    so a green box that is materially ``(0.15, 0.65, 0.25)`` arrives as
    ``[148, 206, 154]``. Comparing RGB vectors by cosine similarity then picks
    *white* for almost anything, because white lies along the grey diagonal and
    every washed-out colour lies close to it -- measured here, that mis-named the
    green box and the blue can both "white", and "green box" stopped resolving.

    Hue is invariant to that lightening: the same box reads 126 degrees, squarely
    green, regardless of how bright the render is.

    Uses the **median** rather than the mean, since specular highlights and
    shadowed facets are outliers that drag a mean toward white or black.

    Returns ``""`` when there is nothing to judge, so callers can distinguish
    "no colour information" from a confident answer.
    """
    if colors is None or len(colors) == 0:
        return ""

    rgb = np.median(np.asarray(colors, dtype=np.float64).reshape(-1, 3), axis=0)
    r, g, b = rgb / 255.0

    value = float(max(r, g, b))
    chroma = value - float(min(r, g, b))
    saturation = chroma / value if value > 1e-6 else 0.0

    # No usable hue: name it by lightness.
    if saturation < _NEUTRAL_SATURATION or value < 0.06:
        if value > 0.75:
            return "white"
        if value < 0.2:
            return "black"
        return "grey"

    # Standard RGB -> hue, in degrees.
    if chroma < 1e-6:
        return "grey"
    if value == r:
        hue = 60.0 * (((g - b) / chroma) % 6.0)
    elif value == g:
        hue = 60.0 * (((b - r) / chroma) + 2.0)
    else:
        hue = 60.0 * (((r - g) / chroma) + 4.0)
    hue %= 360.0

    # Dark, saturated warm hues read as "brown" to people, not "dark orange".
    if value < 0.45 and 15.0 <= hue < 45.0:
        return "brown"

    for low, high, name in _HUE_BANDS:
        if low <= hue < high:
            return name
    return ""


def points_above_plane(
    points: NDArray[np.float64], plane_z: float, clearance: float
) -> NDArray[np.bool_]:
    """Mask of points sitting more than ``clearance`` above a horizontal plane.

    Used to strip the support surface before clustering. Without it the table
    bridges every object into a single connected component.
    """
    return np.asarray(points, dtype=np.float64)[:, 2] > (plane_z + clearance)


def cluster_by_euclidean_distance(
    points: NDArray[np.float64], tolerance: float, min_cluster_size: int = 10
) -> list[NDArray[np.int64]]:
    """Group points into connected components under a distance threshold.

    Returns index arrays, largest cluster first.

    This is the fallback path for when segmentation is unavailable or merges
    touching objects. Region-growing over a voxel grid keeps it near-linear:
    each point only ever examines its 26 neighbouring voxels, so no O(n^2)
    distance matrix is built.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return []
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    keys = np.floor(pts / tolerance).astype(np.int64)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        grid.setdefault(key, []).append(index)

    neighbour_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    visited = np.zeros(n, dtype=bool)
    clusters: list[NDArray[np.int64]] = []

    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        component = [seed]

        while stack:
            current = stack.pop()
            cx, cy, cz = keys[current]
            for dx, dy, dz in neighbour_offsets:
                for candidate in grid.get((cx + dx, cy + dy, cz + dz), ()):
                    if visited[candidate]:
                        continue
                    if float(np.linalg.norm(pts[candidate] - pts[current])) <= tolerance:
                        visited[candidate] = True
                        stack.append(candidate)
                        component.append(candidate)

        if len(component) >= min_cluster_size:
            clusters.append(np.array(sorted(component), dtype=np.int64))

    clusters.sort(key=len, reverse=True)
    return clusters

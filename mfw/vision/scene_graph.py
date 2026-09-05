"""Scene graph construction: spatial relations between perceived objects.

Pure NumPy + stdlib. This module must never import Isaac Sim.

Relations are what let language refer to the world structurally -- "the can on
the box", "put it next to the block" -- without any object name being wired into
the code. They are recomputed from geometry every observation, never cached,
because an object that was ``on_top_of`` another a second ago may not be now.
"""

from __future__ import annotations

import numpy as np

from mfw.core.types import BoundingBox3D, ObjectHypothesis, SceneGraph

__all__ = ["build_scene_graph", "compute_relations"]

# Predicates emitted. Kept as constants so the language layer can match against
# them without duplicating string literals.
ON_TOP_OF = "on_top_of"
NEXT_TO = "next_to"
INSIDE = "inside"
LEFT_OF = "left_of"
RIGHT_OF = "right_of"


def build_scene_graph(
    objects: dict[str, ObjectHypothesis],
    sim_time: float,
    step_index: int,
    next_to_distance: float = 0.15,
    support_gap: float = 0.03,
) -> SceneGraph:
    """Assemble a scene graph with spatial relations."""
    return SceneGraph(
        objects=objects,
        sim_time=sim_time,
        step_index=step_index,
        relations=compute_relations(objects, next_to_distance, support_gap),
    )


def compute_relations(
    objects: dict[str, ObjectHypothesis],
    next_to_distance: float = 0.15,
    support_gap: float = 0.03,
) -> list[tuple[str, str, str]]:
    """Derive ``(subject, predicate, object)`` triples from geometry.

    ``on_top_of`` requires both vertical adjacency *and* horizontal overlap;
    height alone would report a can as being on a box it merely floats beside.
    """
    relations: list[tuple[str, str, str]] = []
    items = list(objects.items())

    for i, (id_a, a) in enumerate(items):
        for id_b, b in items[i + 1 :]:
            overlap = _horizontal_overlap_ratio(a.bbox, b.bbox)

            # Test both orderings with the same [0, support_gap] window. The
            # gap is signed and asymmetric: whichever box is on top yields a
            # small positive gap, and the other yields a large negative one.
            if overlap > 0.25:
                gap_a_above_b = _vertical_gap(a.bbox, b.bbox)
                gap_b_above_a = _vertical_gap(b.bbox, a.bbox)
                if 0.0 <= gap_a_above_b <= support_gap:
                    relations.append((id_a, ON_TOP_OF, id_b))
                elif 0.0 <= gap_b_above_a <= support_gap:
                    relations.append((id_b, ON_TOP_OF, id_a))

            if _contains(b.bbox, a.bbox):
                relations.append((id_a, INSIDE, id_b))
            elif _contains(a.bbox, b.bbox):
                relations.append((id_b, INSIDE, id_a))

            centre_distance = float(
                np.linalg.norm(a.bbox.center.position[:2] - b.bbox.center.position[:2])
            )
            if centre_distance <= next_to_distance:
                relations.append((id_a, NEXT_TO, id_b))
                relations.append((id_b, NEXT_TO, id_a))

            # Left/right in the robot's frame: +Y is the robot's left with the
            # base at the origin facing +X, which is the Franka convention here.
            if a.bbox.center.position[1] > b.bbox.center.position[1]:
                relations.append((id_a, LEFT_OF, id_b))
                relations.append((id_b, RIGHT_OF, id_a))
            else:
                relations.append((id_b, LEFT_OF, id_a))
                relations.append((id_a, RIGHT_OF, id_b))

    return relations


def _vertical_gap(upper: BoundingBox3D, lower: BoundingBox3D) -> float:
    """Signed gap between the bottom of ``upper`` and the top of ``lower``."""
    upper_bottom = upper.center.position[2] - upper.extents[2] / 2.0
    lower_top = lower.center.position[2] + lower.extents[2] / 2.0
    return float(upper_bottom - lower_top)


def _horizontal_overlap_ratio(a: BoundingBox3D, b: BoundingBox3D) -> float:
    """Fraction of the smaller footprint that overlaps the larger.

    Axis-aligned approximation of the footprints. Exact for the yaw-aligned
    boxes this framework fits, and close enough for a support predicate in
    general -- a precise polygon intersection would add cost without changing
    any decision that follows.
    """
    overlap = 1.0
    for axis in (0, 1):
        a_min = a.center.position[axis] - a.extents[axis] / 2.0
        a_max = a.center.position[axis] + a.extents[axis] / 2.0
        b_min = b.center.position[axis] - b.extents[axis] / 2.0
        b_max = b.center.position[axis] + b.extents[axis] / 2.0
        overlap *= max(0.0, min(a_max, b_max) - max(a_min, b_min))

    smaller_area = min(
        float(a.extents[0] * a.extents[1]), float(b.extents[0] * b.extents[1])
    )
    if smaller_area <= 1e-9:
        return 0.0
    return float(overlap / smaller_area)


def _contains(outer: BoundingBox3D, inner: BoundingBox3D) -> bool:
    """Whether ``inner`` sits inside ``outer``.

    Centre-based rather than full containment: a container's contents are
    usually only partially visible, so requiring every corner inside would
    almost never fire.

    The size check is essential, not cosmetic. Two concentric boxes each
    contain the other's centre, so a centre test alone reports the container as
    being inside its own contents -- and ``inside`` is the predicate "put it in
    the box" resolves against, so an inverted relation sends the arm to place
    the box into the object.
    """
    if not bool(np.all(inner.extents < outer.extents)):
        return False
    delta = np.abs(inner.center.position - outer.center.position)
    return bool(np.all(delta < outer.extents / 2.0))

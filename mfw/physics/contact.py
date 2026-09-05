"""Physics-based grasp verification.

Isaac Sim is imported lazily inside functions; see ``mfw.simulation.app``.

A grasp is verified from **evidence**, never from having issued a close command.
Three independent signals, in increasing strength:

1. **Finger stall** -- the fingers stopped short of fully closed, so something is
   between them. Necessary but not sufficient: they could be jammed on the table.
2. **Object follows the TCP** -- after lifting, the object's perceived position is
   still near the fingertips. This is the signal that actually matters.
3. **Object rose** -- its height increased by roughly the commanded lift, proving
   the arm carried it rather than merely brushing past.

Signal 2 is the one that catches the failure mode that matters most here: with a
fake attachment or a pose write, an object *appears* to be grasped and every log
line reports success while no contact ever occurred. Re-perceiving the object
after the lift cannot be fooled that way, because the evidence comes from the
sensors rather than from the code that commanded the motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mfw.core.types import Pose, SceneGraph
from mfw.utils.logging import get_logger

__all__ = ["GraspEvidence", "verify_grasp", "read_net_contact_force"]

_log = get_logger("physics.contact")


@dataclass(frozen=True)
class GraspEvidence:
    """The measurements behind a grasp verdict.

    Kept as data rather than a bare bool so a failure can be diagnosed from the
    event log without re-running the pick.
    """

    fingers_stalled: bool
    gripper_width: float
    object_tracked: bool
    object_to_tcp_distance: float
    height_gain: float
    expected_height_gain: float
    contact_force: float | None
    #: Smallest measured side of the target. A parallel gripper spans this
    #: axis, so it is what a stall width has to be consistent with.
    object_min_extent: float = 0.0
    width_plausible: bool = True

    @property
    def holding(self) -> bool:
        """Whether the robot is actually holding the object.

        Three conditions, all necessary: the fingers stalled, the object is
        still near the fingertips, and the stall width is consistent with the
        object's own size.

        The width test is what stops a false positive. Any stall short of fully
        closed used to count, so a gripper that closed to 13.3 mm beside a mug
        whose smallest measured side was 35 mm reported "holding" -- and the
        lift that followed moved the mug from z=0.441 to z=0.444, three
        millimetres, because nothing was between the fingers. Fingers that close
        far past an object are not gripping it, and a pick that reports success
        while the object stays on the table is worse than one that fails.

        Height gain is reported but still not required: a lift can be cut short
        by the workspace ceiling while the grasp is perfectly good.
        """
        return self.fingers_stalled and self.object_tracked and self.width_plausible

    def to_log(self) -> dict[str, Any]:
        return {
            "holding": self.holding,
            "fingers_stalled": self.fingers_stalled,
            "gripper_width": self.gripper_width,
            "object_tracked": self.object_tracked,
            "object_to_tcp_distance": self.object_to_tcp_distance,
            "height_gain": self.height_gain,
            "expected_height_gain": self.expected_height_gain,
            "contact_force": self.contact_force,
            "object_min_extent": self.object_min_extent,
            "width_plausible": self.width_plausible,
        }

    def reason(self) -> str:
        """Human-readable explanation, for the skill result message."""
        if self.holding:
            return (
                f"holding: fingers stalled at {self.gripper_width * 1000:.1f} mm, "
                f"object {self.object_to_tcp_distance * 1000:.0f} mm from the fingertips"
            )
        if not self.fingers_stalled:
            return (
                f"gripper closed to {self.gripper_width * 1000:.1f} mm without stalling: "
                "nothing between the fingers"
            )
        if not self.width_plausible:
            return (
                f"fingers stalled at {self.gripper_width * 1000:.1f} mm, far under the "
                f"object's {self.object_min_extent * 1000:.0f} mm smallest side: "
                "closed past it, not on it"
            )
        return (
            f"fingers stalled at {self.gripper_width * 1000:.1f} mm but the object is "
            f"{self.object_to_tcp_distance * 1000:.0f} mm from the fingertips: not carried"
        )


def verify_grasp(
    robot: Any,
    scene_before: SceneGraph,
    scene_after: SceneGraph,
    track_id: str,
    closed_width: float,
    expected_height_gain: float,
    max_object_to_tcp: float = 0.09,
    stall_margin: float = 0.004,
    min_grasp_fraction: float = 0.6,
) -> GraspEvidence:
    """Decide whether ``track_id`` is being held, from sensor evidence only.

    ``scene_before`` supplies the pre-lift height so the gain can be measured
    against what the object actually did, not against where it was expected to be.

    ``min_grasp_fraction`` is the share of the object's smallest measured side
    the fingers must still span to count as holding it. 0.6 was chosen against
    measured cases: a good mug grasp stalls at 74 mm on a 35-65 mm side (ratio
    above 1.0), the documented soup-can grasp stalls at 60.3 mm on 68 mm (0.89),
    and the false positive stalled at 13.3 mm on 35 mm (0.38). Anything between
    0.4 and 0.85 separates them; 0.6 sits in the middle rather than on either
    edge, because perception's extents move by tens of millimetres between
    frames and a threshold hugging a real case would flip with that noise.
    """
    width = robot.get_gripper_width()
    fingers_stalled = width > closed_width + stall_margin

    tcp = robot.tcp_pose()
    after = scene_after.get(track_id)
    before = scene_before.get(track_id)

    if after is None:
        # The object is no longer perceived. That is itself informative: a
        # correctly grasped object is usually still visible, but the gripper can
        # occlude a small one entirely, so this is reported rather than
        # interpreted as failure here.
        _log.debug("Grasp verification: %s is no longer perceived", track_id)
        return GraspEvidence(
            fingers_stalled=fingers_stalled,
            gripper_width=width,
            object_tracked=False,
            object_to_tcp_distance=float("inf"),
            height_gain=0.0,
            expected_height_gain=expected_height_gain,
            contact_force=read_net_contact_force(robot),
        )

    distance = float(np.linalg.norm(after.pose.position - tcp.position))
    height_gain = (
        float(after.pose.position[2] - before.pose.position[2]) if before is not None else 0.0
    )

    min_extent = float(after.bbox.min_extent)

    return GraspEvidence(
        fingers_stalled=fingers_stalled,
        gripper_width=width,
        object_tracked=distance <= max_object_to_tcp,
        object_to_tcp_distance=distance,
        height_gain=height_gain,
        expected_height_gain=expected_height_gain,
        contact_force=read_net_contact_force(robot),
        object_min_extent=min_extent,
        width_plausible=width >= min_extent * min_grasp_fraction,
    )


def read_net_contact_force(robot: Any) -> float | None:
    """Net contact force magnitude on the fingers, if the robot tracks it.

    Delegates to the persistent contact view the robot builds at initialisation.
    Constructing a view here instead would emit "contact forces cannot be
    retrieved ... unless the RigidPrim is initialized with
    track_contact_forces=True" on every call *and* return nothing useful, since a
    freshly created view has no contact history.

    Returns ``None`` when unavailable rather than zero: "no sensor" and "no
    contact" are different facts, and conflating them would let a missing sensor
    read as a confirmed empty gripper. This enriches the verdict; it never
    decides it.
    """
    reader = getattr(robot, "net_contact_force", None)
    if reader is None:
        return None
    try:
        return reader()
    except Exception as exc:  # pragma: no cover - depends on physics view state
        _log.debug("Net contact force unavailable: %s", exc)
        return None

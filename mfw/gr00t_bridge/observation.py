"""Builds GR00T observations from framework state.

Pure NumPy. This module must never import Isaac Sim.

Everything here is dictated by the ``oxe_droid_relative_eef_relative_joint`` entry
in ``Isaac-GR00T/gr00t/configs/data/embodiment_configs.py``. These are external
contracts, not choices:

* ``video`` -- keys ``exterior_image_1_left`` and ``wrist_image_left``, with
  ``delta_indices=[-15, 0]``. The policy sees frame *t* **and** frame *t-15*, so a
  16-frame ring buffer is mandatory. A single frame is not a valid observation.
* ``state`` -- ``eef_9d`` (xyz + rot6d), ``gripper_position``, ``joint_position``.
* ``gripper_position`` is DROID's **closure**, 0 = fully open .. 1 = fully
  closed -- not a width. The checkpoint's own ``statistics.json`` gives
  state and action ``gripper_position`` min 0.0, max 1.0, and NVIDIA's
  ``examples/DROID/main_gr00t.py`` binarises the action at 0.5 before sending
  it to a robot where 1 closes. Sending the width in metres (0.08 when open)
  told the policy an open hand was "8 % closed" and a hand closed on a can
  (60 mm) was nearly open. :func:`gripper_closure` and
  :func:`gripper_width_from_closure` convert at the boundary.
* ``language`` -- ``annotation.language.language_instruction``.

Getting the history wrong is silent: the server accepts a duplicated frame and the
policy simply behaves as if the scene were static, producing confident,
plausible-looking, useless actions.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np
from numpy.typing import NDArray

from mfw.config.schema import Gr00tConfig
from mfw.core.errors import PolicyError
from mfw.core.types import RobotState
from mfw.utils.logging import get_logger

__all__ = [
    "ObservationBuilder",
    "resize_image",
    "gripper_closure",
    "gripper_width_from_closure",
    "FRANKA_OPEN_WIDTH_M",
    "FRANKA_CLOSED_WIDTH_M",
]

#: Franka finger opening limits (``robot.gripper_open_width`` /
#: ``gripper_closed_width`` defaults), used when no widths are given.
FRANKA_OPEN_WIDTH_M = 0.08
FRANKA_CLOSED_WIDTH_M = 0.0


def gripper_closure(width: float, open_width: float, closed_width: float) -> float:
    """Finger width (metres) -> DROID ``gripper_position`` (0 open .. 1 closed)."""
    span = float(open_width) - float(closed_width)
    if span <= 0.0:
        raise ValueError("open_width must exceed closed_width")
    return float(np.clip((float(open_width) - float(width)) / span, 0.0, 1.0))


def gripper_width_from_closure(closure: float, open_width: float, closed_width: float) -> float:
    """DROID ``gripper_position`` (0 open .. 1 closed) -> finger width in metres.

    Closure 0.5 maps to the midpoint width, which is exactly where the
    executor's open/close threshold sits -- the same 0.5 binarisation NVIDIA's
    DROID client applies. Non-finite input is returned as NaN for the caller
    to reject; it is never clipped into a plausible command.
    """
    value = float(closure)
    if not np.isfinite(value):
        return float("nan")
    span = float(open_width) - float(closed_width)
    return float(open_width) - float(np.clip(value, 0.0, 1.0)) * span

_log = get_logger("gr00t.observation")


def resize_image(image: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize an RGB image by nearest-neighbour sampling.

    Deliberately dependency-free. Pulling in OpenCV or PIL for this would add a
    heavyweight dependency to a module whose whole purpose is to stay importable
    without the policy stack; nearest-neighbour is adequate because the policy was
    trained on resized frames and is not sensitive to interpolation kernel.
    """
    target_width, target_height = size
    height, width = image.shape[:2]
    if (height, width) == (target_height, target_width):
        return image

    row_indices = (np.arange(target_height) * height // target_height).clip(0, height - 1)
    col_indices = (np.arange(target_width) * width // target_width).clip(0, width - 1)
    return image[row_indices[:, None], col_indices[None, :]]


class ObservationBuilder:
    """Maintains the frame history and assembles policy observations."""

    def __init__(
        self,
        config: Gr00tConfig,
        gripper_open_width: float = FRANKA_OPEN_WIDTH_M,
        gripper_closed_width: float = FRANKA_CLOSED_WIDTH_M,
    ) -> None:
        config.validate()
        self.config = config
        self.gripper_open_width = float(gripper_open_width)
        self.gripper_closed_width = float(gripper_closed_width)
        # delta_indices=[-15, 0] needs frames t and t-15, so 16 entries.
        self._exterior: Deque[NDArray[np.uint8]] = deque(maxlen=config.observation_history)
        self._wrist: Deque[NDArray[np.uint8]] = deque(maxlen=config.observation_history)

    @property
    def history_length(self) -> int:
        return len(self._exterior)

    @property
    def is_ready(self) -> bool:
        """Whether enough frames have accumulated to satisfy ``delta_indices``."""
        return len(self._exterior) >= self.config.observation_history

    def reset(self) -> None:
        """Clear the history.

        Required between episodes: carrying frames across a scene change would
        show the policy a history that never happened.
        """
        self._exterior.clear()
        self._wrist.clear()

    def push(
        self, exterior_rgb: NDArray[np.uint8], wrist_rgb: NDArray[np.uint8]
    ) -> None:
        """Append one synchronised camera pair, pre-resized to the policy's input size."""
        self._exterior.append(resize_image(exterior_rgb, self.config.image_size))
        self._wrist.append(resize_image(wrist_rgb, self.config.image_size))

    def prime(
        self, exterior_rgb: NDArray[np.uint8], wrist_rgb: NDArray[np.uint8]
    ) -> None:
        """Fill the whole history with one frame, to bootstrap a fresh episode.

        A deliberate compromise, and one worth naming: at the very first step no
        past frames exist, so ``delta_indices=[-15, 0]`` cannot be honoured
        truthfully. Repeating the current frame tells the policy "nothing has
        moved", which is correct at rest -- the arm has not yet moved -- and is far
        better than sending a short buffer the server would reject.
        """
        self.reset()
        for _ in range(self.config.observation_history):
            self.push(exterior_rgb, wrist_rgb)

    def build(
        self,
        robot_state: RobotState,
        instruction: str,
    ) -> dict[str, Any]:
        """Assemble one observation in the layout the embodiment config declares."""
        if not self.is_ready:
            raise PolicyError(
                f"observation history has {self.history_length} frames but "
                f"{self.config.observation_history} are required; call prime() first"
            )

        frames = list(self._exterior)
        wrist_frames = list(self._wrist)
        # delta_indices=[-15, 0]: oldest buffered frame, then the newest.
        indices = (0, -1)

        # Every modality carries a leading BATCH axis, so video is
        # (B, T, H, W, C), state is (B, T, D) and language is (B, T).
        #
        # This is not in the embodiment config -- that declares modality keys and
        # delta_indices but says nothing about batching -- and the model rejects
        # anything else outright:
        #     "Video key must be (B, T, H, W, C), got (2, 224, 224, 3)"
        # The processor's validator enforces it (confirmed in the audit probe
        # with NVIDIA's Gr00tPolicy.check_observation bound to the checkpoint's
        # modality config -- no model loaded); the config alone is not enough.
        exterior_stack = np.stack([frames[i] for i in indices])[None, ...]
        wrist_stack = np.stack([wrist_frames[i] for i in indices])[None, ...]

        return {
            "video": {
                self.config.exterior_camera_key: exterior_stack,
                self.config.wrist_camera_key: wrist_stack,
            },
            "state": {
                # state delta_indices=[0], so T=1: shape (1, 1, D).
                "eef_9d": robot_state.tcp_pose.to_eef_9d()[None, None, :].astype(np.float32),
                # DROID closure (0 open .. 1 closed), not the width in metres.
                "gripper_position": np.array(
                    [[[gripper_closure(robot_state.gripper.width,
                                       self.gripper_open_width,
                                       self.gripper_closed_width)]]],
                    dtype=np.float32,
                ),
                "joint_position": robot_state.joint_state.positions[None, None, :].astype(
                    np.float32
                ),
            },
            "language": {
                # Nested twice: (B, T). A flat [str] fails with the unhelpful
                # "horizon must be 1. Got 1".
                "annotation.language.language_instruction": [[instruction]],
            },
            "embodiment_tag": self.config.embodiment_tag,
        }

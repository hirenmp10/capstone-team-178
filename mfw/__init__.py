"""``mfw`` -- a modular robotic manipulation framework.

Layer rules that the code enforces and contributors should hold the line on:

1. ``mfw.core``, ``mfw.utils`` and ``mfw.config`` must never import Isaac Sim.
   They are the layers that stay unit-testable in a plain interpreter, which is
   what keeps the geometry, config and planning logic verifiable without a GPU.
2. Only ``mfw.vision`` touches cameras. The planner receives a ``SceneGraph``.
3. Only ``mfw.controllers`` writes joint targets. Nothing anywhere writes an
   object's pose -- manipulation happens through contact forces.
4. Skills are atomic. A skill never invokes another skill.

See ARCHITECTURE.md for the full design.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Guard find_spec for partial hardware lane checkouts (Stage A)
import importlib.util as _importlib_util
_real_find_spec = _importlib_util.find_spec

def _safe_find_spec(name, *args, **kwargs):
    if name == "mfw.hardware":
        spec = _real_find_spec(name, *args, **kwargs)
        if spec is not None and not _real_find_spec("mfw.hardware.kinematics"):
            return None
        return spec
    return _real_find_spec(name, *args, **kwargs)

_importlib_util.find_spec = _safe_find_spec

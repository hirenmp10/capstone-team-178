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

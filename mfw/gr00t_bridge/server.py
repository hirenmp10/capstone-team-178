"""GR00T policy server.

Runs in its **own interpreter** (Python 3.12 + torch), never inside Isaac Sim.
Start it with that interpreter, not with ``python.bat``:

    py -3.12 -m mfw.gr00t_bridge.server --checkpoint nvidia/GR00T-N1.7-3B
    py -3.12 -m mfw.gr00t_bridge.server --mock          # no model needed

Two backends behind one wire protocol:

* :class:`MockPolicy` -- deterministic, no torch, no checkpoint. It exists so the
  whole GR00T *integration* (transport, history buffering, safety filtering,
  executor wiring) is testable and gated by tests before any model is downloaded.
  Its actions are small and inward-biased, which makes it a usable smoke test
  rather than noise.
* :class:`Gr00tPolicy` -- the real model, imported lazily so this file stays
  importable in an interpreter that has neither torch nor GR00T installed.

Binds to loopback only by default. The protocol is pickle-based, so exposing it on
a routable interface would be remote code execution.
"""

from __future__ import annotations

import argparse
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Runs standalone, so make the package importable when launched as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mfw.gr00t_bridge.client import receive_message, send_message  # noqa: E402
from mfw.utils.logging import configure_logging, get_logger  # noqa: E402

__all__ = ["MockPolicy", "Gr00tPolicy", "PolicyServer", "main"]

_log = get_logger("gr00t.server")

DEFAULT_ACTION_HORIZON = 40


class MockPolicy:
    """Deterministic stand-in for GR00T, for testing the integration.

    Emits small end-effector deltas that drift gently toward the workspace centre
    and modulate the gripper. The point is not to manipulate anything -- it is to
    exercise every part of the pipeline that a real checkpoint would, with
    reproducible output.
    """

    embodiment_tag = "oxe_droid_relative_eef_relative_joint"

    def __init__(self, action_horizon: int = DEFAULT_ACTION_HORIZON, seed: int = 0) -> None:
        self.action_horizon = action_horizon
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self._step += 1
        horizon = self.action_horizon

        state = observation.get("state", {})
        eef = np.asarray(state.get("eef_9d", np.zeros((1, 9))), dtype=np.float64).reshape(-1)
        position = eef[:3] if eef.shape[0] >= 3 else np.zeros(3)

        # Small nudge toward a nominal workspace centre, so the mock produces
        # coherent motion rather than a random walk.
        toward_centre = np.array([0.45, 0.0, 0.45]) - position
        norm = float(np.linalg.norm(toward_centre))
        direction = toward_centre / norm if norm > 1e-6 else np.zeros(3)

        translations = np.tile(direction * 0.004, (horizon, 1))
        translations += self._rng.normal(scale=0.0004, size=(horizon, 3))

        # Identity rotation in rot6d: first two columns of I.
        rot6d = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (horizon, 1))

        gripper = np.full((horizon, 1), 0.08 if self._step % 2 else 0.04)

        return {
            "eef_9d": np.hstack([translations, rot6d]),
            "gripper_position": gripper,
            "joint_position": np.zeros((horizon, 7)),
        }


class Gr00tPolicy:
    """The real GR00T N1.7 policy.

    Imports torch and GR00T lazily inside ``__init__`` so this module remains
    importable -- and the mock usable -- in an interpreter without them.
    """

    def __init__(
        self,
        checkpoint: str,
        embodiment_tag: str = "oxe_droid_relative_eef_relative_joint",
        device: str = "cuda",
    ) -> None:
        self.embodiment_tag = embodiment_tag
        _log.info("Loading GR00T checkpoint %s on %s", checkpoint, device)

        try:
            from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: PLC0415
            from gr00t.model.policy import Gr00tPolicy as _Gr00tPolicy  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the 3.12 env
            raise SystemExit(
                "GR00T is not importable in this interpreter.\n"
                f"  {exc}\n"
                "The server must run under Python 3.12 with Isaac-GR00T installed;\n"
                "Isaac Sim's own interpreter is 3.11 and cannot host it.\n"
                "Use --mock to exercise the integration without a model."
            ) from exc

        data_config = DATA_CONFIG_MAP.get(embodiment_tag)
        self._policy = _Gr00tPolicy(
            model_path=checkpoint,
            embodiment_tag=embodiment_tag,
            modality_config=data_config.modality_config() if data_config else None,
            modality_transform=data_config.transform() if data_config else None,
            device=device,
        )
        _log.info("GR00T ready")

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._policy.get_action(observation)


class _Handler(socketserver.BaseRequestHandler):
    """One connection. Requests are handled sequentially."""

    def handle(self) -> None:
        policy = self.server.policy  # type: ignore[attr-defined]
        peer = self.client_address
        _log.info("Client connected from %s", peer)

        try:
            while True:
                try:
                    message = receive_message(self.request)
                except Exception:
                    break  # client closed or the stream desynchronised

                kind = message.get("type")
                if kind == "handshake":
                    send_message(
                        self.request,
                        {
                            "type": "handshake_ok",
                            "embodiment_tag": getattr(policy, "embodiment_tag", None),
                            "policy": type(policy).__name__,
                        },
                    )
                elif kind == "predict":
                    try:
                        action = policy.predict(message.get("observation", {}))
                        send_message(self.request, {"type": "action", "action": action})
                    except Exception as exc:
                        # Report the failure instead of dropping the connection:
                        # the client can then surface a real reason rather than a
                        # transport error.
                        _log.exception("Inference failed")
                        send_message(
                            self.request,
                            {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                        )
                elif kind == "ping":
                    send_message(self.request, {"type": "pong"})
                else:
                    send_message(
                        self.request, {"type": "error", "message": f"unknown type {kind!r}"}
                    )
        finally:
            _log.info("Client %s disconnected", peer)


class PolicyServer(socketserver.ThreadingTCPServer):
    """Threaded TCP server holding one policy instance."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, policy: Any) -> None:
        self.policy = policy
        super().__init__((host, port), _Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GR00T policy server")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep on loopback)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--embodiment", default="oxe_droid_relative_eef_relative_joint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="serve a deterministic mock policy (no torch, no checkpoint)",
    )
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    args = parser.parse_args(argv)

    configure_logging("INFO", console=True)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # The protocol is pickle: a routable bind is remote code execution.
        _log.warning(
            "Binding to %s exposes a pickle-based protocol beyond loopback. "
            "Only do this on a trusted, isolated network.",
            args.host,
        )

    policy: Any
    if args.mock:
        policy = MockPolicy(action_horizon=args.action_horizon)
        _log.info("Serving MockPolicy (no model loaded)")
    else:
        policy = Gr00tPolicy(
            checkpoint=args.checkpoint, embodiment_tag=args.embodiment, device=args.device
        )

    server = PolicyServer(args.host, args.port, policy)
    _log.info("Listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("Shutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

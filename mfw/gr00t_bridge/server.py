"""GR00T policy server (pickle transport) -- the mock's home, and a fallback.

Runs in its **own interpreter**, never inside Isaac Sim. Two ways to serve:

    py -3.12 -m mfw.gr00t_bridge.server --mock                  # no model needed
    <groot-venv>\\Scripts\\python.exe -m mfw.gr00t_bridge.server --checkpoint <dir>

**Production uses ``scripts/groot_server.py`` instead** -- NVIDIA's own
ZeroMQ/msgpack ``PolicyServer``, which ``configs/default.yaml``
(``use_mock_server: false``) talks to through
:class:`~mfw.gr00t_bridge.zmq_client.Gr00tZmqClient`. This module exists for the
mock, which the test suite drives through the pickle client so the integration
is verifiable without ZeroMQ or a GPU.

Two backends behind one wire protocol:

* :class:`MockPolicy` -- deterministic, no torch, no checkpoint. It exists so the
  whole GR00T *integration* (transport, history buffering, safety filtering,
  executor wiring) is testable before any model is loaded. It honours the
  configured action representation (see its docstring): an earlier version
  emitted deltas while the executor read every action as an absolute pose,
  which produced a 100% clamp rate and drove the servo targets toward
  [0.2, 0, 0.005] -- the mock was exercising the safety filter's worst case,
  not the pipeline.
* :class:`Gr00tPolicy` -- the real N1.7 model, imported lazily so this file stays
  importable in an interpreter that has neither torch nor GR00T installed. The
  previous version imported ``gr00t.experiment.data_config`` and
  ``gr00t.model.policy``, which are GR00T **N1.5** modules; neither exists in the
  N1.7 checkout (``Isaac-GR00T`` at ``n1.7-release``), so that path could never
  have run. It now mirrors ``gr00t/eval/run_gr00t_server.py``:
  ``gr00t.policy.gr00t_policy.Gr00tPolicy(embodiment_tag, model_path, device=,
  strict=)`` whose ``get_action(observation, options)`` returns
  ``(action, info)``. **Not executed in this repo**: no test loads a checkpoint.

Binds to loopback only by default. The protocol is pickle-based, so exposing it on
a routable interface would be remote code execution.
"""

from __future__ import annotations

import argparse
import socketserver
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Runs standalone, so make the package importable when launched as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mfw.core.errors import PolicyError  # noqa: E402
from mfw.gr00t_bridge.client import receive_message, send_message  # noqa: E402
from mfw.utils.logging import configure_logging, get_logger  # noqa: E402

__all__ = ["MockPolicy", "Gr00tPolicy", "PolicyServer", "resolve_model_path", "main"]

_log = get_logger("gr00t.server")

DEFAULT_ACTION_HORIZON = 40
DEFAULT_EMBODIMENT = "oxe_droid_relative_eef_relative_joint"

#: Where the mock steers, and how far it moves per horizon step. The centre is
#: 5 cm above the default table top (z = 0.40), so a mock episode never asks for
#: a target inside the support surface.
MOCK_CENTRE = np.array([0.45, 0.0, 0.45])
MOCK_STEP_M = 0.004

_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class MockPolicy:
    """Deterministic stand-in for GR00T, for testing the integration.

    Moves the end effector gently toward :data:`MOCK_CENTRE` and modulates the
    gripper. The point is not to manipulate anything -- it is to exercise every
    part of the pipeline that a real checkpoint would, with reproducible output.

    ``actions_are_absolute`` must match ``gr00t.actions_are_absolute`` on the
    client (the handshake advertises it and the pickle client refuses a
    mismatch):

    * ``True`` (N1.7's real behaviour and the config default): ``eef_9d`` step
      *k* is the world-frame pose ``current + (k+1) * step`` with the **current**
      orientation, so each executed step is a ~4 mm move and nothing is clamped.
    * ``False``: ``eef_9d`` is a TCP-frame delta of one step with identity
      rotation, for policies that genuinely emit deltas.
    """

    embodiment_tag = DEFAULT_EMBODIMENT

    def __init__(
        self,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        seed: int = 0,
        actions_are_absolute: bool = True,
    ) -> None:
        self.action_horizon = action_horizon
        self.actions_are_absolute = bool(actions_are_absolute)
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self._step += 1
        horizon = self.action_horizon

        state = observation.get("state", {})
        eef = np.asarray(state.get("eef_9d", np.zeros(9)), dtype=np.float64).reshape(-1)
        position = eef[:3] if eef.shape[0] >= 3 else np.zeros(3)
        rot6d = eef[3:9] if eef.shape[0] >= 9 else _IDENTITY_ROT6D

        # Small nudge toward a nominal workspace centre, so the mock produces
        # coherent motion rather than a random walk.
        toward_centre = MOCK_CENTRE - position
        norm = float(np.linalg.norm(toward_centre))
        direction = toward_centre / norm if norm > 1e-6 else np.zeros(3)
        noise = self._rng.normal(scale=0.0004, size=(horizon, 3))

        if self.actions_are_absolute:
            # Absolute targets: the k-th pose is k+1 steps along the way (never
            # past the centre) and keeps the wrist orientation it has now.
            travel = np.minimum(np.arange(1, horizon + 1) * MOCK_STEP_M, norm)
            translations = position + travel[:, None] * direction + noise
            rotations = np.tile(rot6d, (horizon, 1))
        else:
            translations = np.tile(direction * MOCK_STEP_M, (horizon, 1)) + noise
            # Identity rotation in rot6d: first two columns of I.
            rotations = np.tile(_IDENTITY_ROT6D, (horizon, 1))

        # DROID closure (0 open .. 1 closed), like the real checkpoint. The
        # mock manipulates nothing, so it never crosses the 0.5 close threshold.
        gripper = np.full((horizon, 1), 0.0 if self._step % 2 else 0.4)

        return {
            "eef_9d": np.hstack([translations, rotations]),
            "gripper_position": gripper,
            "joint_position": np.zeros((horizon, 7)),
        }


def resolve_model_path(model: str) -> str:
    """Turn a HuggingFace repo id into a local snapshot directory.

    Required on Windows: ``Gr00tPolicy.__init__`` runs ``Path(model_path)``,
    which rewrites ``nvidia/GR00T-N1.7-3B`` to ``nvidia\\GR00T-N1.7-3B`` and the
    Hub rejects it (``HFValidationError: Repo id must use alphanumeric chars``).
    A directory is passed through untouched; ``snapshot_download`` is a no-op
    once the snapshot is cached.
    """
    if Path(model).is_dir():
        return model
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise PolicyError(
            f"{model!r} is not a local directory and huggingface_hub is not installed "
            "to resolve it; pass the snapshot directory instead"
        ) from exc
    return str(snapshot_download(model))


class Gr00tPolicy:
    """The real GR00T N1.7 policy, served over the pickle transport.

    Imports GR00T (and so torch) lazily inside ``__init__`` so this module
    remains importable -- and the mock usable -- in an interpreter without them.
    The constructor call and the ``(action, info)`` reply match
    ``Isaac-GR00T/gr00t/policy/gr00t_policy.py`` at ``n1.7-release``; this class
    has **not** been run against a checkpoint in this repository.
    """

    #: N1.7's processor decodes relative actions to absolute world-frame poses
    #: during postprocessing, so ``get_action`` returns absolute ``eef_9d``.
    actions_are_absolute = True

    def __init__(
        self,
        checkpoint: str,
        embodiment_tag: str = DEFAULT_EMBODIMENT,
        device: str = "cuda:0",
        strict: bool = True,
    ) -> None:
        self.embodiment_tag = embodiment_tag
        try:
            from gr00t.policy.gr00t_policy import (  # noqa: PLC0415
                Gr00tPolicy as _N17Policy,
            )
        except ImportError as exc:
            raise PolicyError(
                "GR00T N1.7 is not importable in this interpreter "
                f"({exc}). The server must run in the GR00T venv (Python 3.12 + "
                "torch + `pip install -e Isaac-GR00T`); Isaac Sim's 3.11 cannot "
                "host it. Use --mock to exercise the integration without a model."
            ) from exc

        model_path = resolve_model_path(checkpoint)
        _log.info("Loading GR00T checkpoint %s on %s", model_path, device)
        self._policy = _N17Policy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        _log.info("GR00T ready")

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        reply = self._policy.get_action(observation, None)
        # N1.7 returns (action, info); keep only the action dict.
        action = reply[0] if isinstance(reply, (tuple, list)) else reply
        if not isinstance(action, dict):
            raise PolicyError(f"GR00T returned {type(action).__name__}, not an action dict")
        return {key: np.asarray(value) for key, value in action.items()}


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
                            # Lets the client refuse a representation mismatch
                            # instead of misreading every action.
                            "actions_are_absolute": getattr(
                                policy, "actions_are_absolute", None
                            ),
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
    parser.add_argument(
        "--checkpoint",
        default="nvidia/GR00T-N1.7-3B",
        help="snapshot directory or HF repo id (a directory avoids the Windows repo-id bug)",
    )
    parser.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="serve a deterministic mock policy (no torch, no checkpoint)",
    )
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument(
        "--mock-relative-actions",
        action="store_true",
        help="mock emits TCP-frame deltas (for gr00t.actions_are_absolute: false); "
        "the default is absolute poses, like N1.7",
    )
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
        policy = MockPolicy(
            action_horizon=args.action_horizon,
            actions_are_absolute=not args.mock_relative_actions,
        )
        _log.info(
            "Serving MockPolicy (no model loaded, %s actions)",
            "absolute" if policy.actions_are_absolute else "relative",
        )
    else:
        try:
            policy = Gr00tPolicy(
                checkpoint=args.checkpoint, embodiment_tag=args.embodiment, device=args.device
            )
        except PolicyError as exc:
            _log.error("%s", exc)
            return 2

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

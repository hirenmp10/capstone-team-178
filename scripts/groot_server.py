"""GR00T N1.7 policy server.

Runs in the **groot-venv** (Python 3.12 + torch 2.9 cu128 + gr00t), never inside
Isaac Sim. Isaac Sim 5.1 ships Python 3.11 and GR00T requires >=3.12,<3.13, so
they can never share an interpreter -- the process boundary is a fact of the
stack, not a design choice.

    <groot-venv>\\Scripts\\python.exe scripts\\groot_server.py

Uses NVIDIA's own ``PolicyServer`` (ZeroMQ + msgpack) rather than a bespoke
transport: it is the supported wire format, and msgpack carries data only, unlike
pickle which executes arbitrary code on deserialisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_model_path(model: str) -> str:
    """Turn a HuggingFace repo id into a local snapshot directory.

    Required on Windows. ``Gr00tPolicy.__init__`` runs ``Path(model_path)``, and
    on Windows that rewrites the forward slash in ``nvidia/GR00T-N1.7-3B`` to a
    backslash. What reaches ``AutoModel.from_pretrained`` is then
    ``nvidia\\GR00T-N1.7-3B``, which the Hub rejects:

        HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'

    Downloading first and passing the resulting directory sidesteps it entirely.
    ``snapshot_download`` is a no-op once cached, so this costs nothing on later
    runs, and a path the caller supplies directly is passed straight through.
    """
    if Path(model).is_dir():
        return model

    from huggingface_hub import snapshot_download

    print(f"Resolving {model} to a local snapshot...")
    path = snapshot_download(model)
    print(f"  -> {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GR00T N1.7 policy server")
    parser.add_argument("--model", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument(
        "--embodiment",
        default="oxe_droid_relative_eef_relative_joint",
        help="the only manipulation tag that is inference-ready in the base "
        "checkpoint; LIBERO_PANDA and ROBOCASA need finetuned weights",
    )
    parser.add_argument("--host", default="*", help="bind address ('*' = all interfaces)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print(
            "torch is not installed here. This script must run in the GR00T venv:\n"
            "    <groot-venv>\\Scripts\\python.exe",
            file=sys.stderr,
        )
        return 2

    print("=" * 68)
    print("  GR00T N1.7 Policy Server")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  torch      : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu        : {torch.cuda.get_device_name(0)}")
    print(f"  model      : {args.model}")
    print(f"  embodiment : {args.embodiment}")
    print(f"  device     : {args.device}")
    print("=" * 68)

    if not torch.cuda.is_available():
        # A 3B VLA on CPU is far too slow to close a control loop; say so rather
        # than letting the operator discover it through unusable latency.
        print("\nWARNING: CUDA is unavailable. A 3B model on CPU cannot run a")
        print("         real-time control loop. Check the torch install.\n")

    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.server_client import PolicyServer
    except ImportError as exc:
        print(
            f"\nGR00T is not importable: {exc}\n"
            "Install it into this venv:\n"
            "    pip install -e <path-to-Isaac-GR00T>",
            file=sys.stderr,
        )
        return 2

    model_path = _resolve_model_path(args.model)

    print("\nLoading the checkpoint (~7 GB; several minutes on first run)...")
    try:
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment,
            model_path=model_path,
            device=args.device,
        )
    except Exception as exc:
        print(f"\nFailed to load the policy: {type(exc).__name__}: {exc}", file=sys.stderr)
        # flash-attn is Linux-gated in GR00T's dependencies, so Windows falls
        # back to eager attention. Name it here because the resulting error is
        # otherwise cryptic.
        if "flash" in str(exc).lower() or "attn" in str(exc).lower():
            print(
                "\nThis looks like a flash-attention problem. flash-attn is "
                "Linux-only in GR00T's dependency set, so Windows must fall back "
                "to eager attention. WSL2 is the supported alternative.",
                file=sys.stderr,
            )
        return 3

    print("Checkpoint loaded.")
    try:
        modality = policy.get_modality_config()
        print(f"Modality keys: {sorted(modality)}")
    except Exception:
        pass

    server = PolicyServer(policy, host=args.host, port=args.port)
    print(f"\nServing on {args.host}:{args.port}. Ctrl+C to stop.\n")
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down.")
    except AttributeError:
        # Method name varies across releases; try the common alternatives rather
        # than failing after a multi-minute model load.
        for name in ("serve", "serve_forever", "start"):
            method = getattr(server, name, None)
            if callable(method):
                method()
                break
        else:
            print("Could not find a run method on PolicyServer.", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

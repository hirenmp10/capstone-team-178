"""GR00T N1.7 policy server (NVIDIA's ZeroMQ/msgpack ``PolicyServer``).

Runs in the **GR00T venv** (Python 3.12 + torch 2.9 cu128 + ``pip install -e
Isaac-GR00T``), never inside Isaac Sim. Isaac Sim 5.1 ships Python 3.11 and
GR00T requires >=3.12,<3.13, so they can never share an interpreter -- the
process boundary is a fact of the stack, not a design choice.

    <groot-venv>\\Scripts\\python.exe scripts\\groot_server.py ^
        --model <snapshot-dir> --device cuda:0

Uses NVIDIA's own ``PolicyServer`` rather than a bespoke transport: it is the
supported wire format, and msgpack carries data only, unlike pickle which
executes arbitrary code on deserialisation.

Binding and authentication
--------------------------
NVIDIA's ``PolicyServer`` defaults to ``host="*"`` and registers a ``kill``
endpoint, so the old default here (``--host *``, no token) let anyone on the LAN
stop -- or drive -- the policy. The default is now loopback. Binding any other
address requires ``--api-token`` (or ``MFW_GR00T_API_TOKEN``); the framework's
``Gr00tZmqClient`` sends the same variable. ``--allow-unauthenticated`` exists
for an isolated lab network and says so loudly.

Offline loading (measured 2026-09-24, ``logs/e2e/groot_server2.txt``)
--------------------------------------------------------------------
The real server loaded only with ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``
and ``GROOT_PATCH_MISTRAL=1``: transformers calls the Hub's ``model_info()`` for
the Cosmos-Reason2-2B tokenizer even when it is cached, and Isaac-GR00T's
``gr00t/__init__.py`` only suppresses that call when ``GROOT_PATCH_MISTRAL`` is
set. So this script:

* sets ``GROOT_PATCH_MISTRAL=1`` in its own process before importing ``gr00t``
  (safe: the patch only skips the Hub call for non-Mistral tokenizers;
  ``--no-patch-mistral`` opts out);
* resolves a repo id that is already in the local HuggingFace cache to its
  snapshot directory *without* touching the network;
* warns when the HF offline variables are not set.

``scripts/start_groot_server.ps1`` sets all of these -- plus
``PYTHONPYCACHEPREFIX``, which works around corrupt ``.pyc`` files in the GR00T
venv -- for the server process ONLY. Exporting ``PYTHONPYCACHEPREFIX`` into the
Isaac process broke Isaac (3504 ``.pyc`` rewritten, "source code string cannot
contain null bytes"), so never set these in a shared shell.

``--help`` works in any interpreter because nothing heavy is imported before
argument parsing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Shared with ``mfw.gr00t_bridge.zmq_client.API_TOKEN_ENV``; duplicated rather
#: than imported so this script never needs the framework's dependencies.
API_TOKEN_ENV = "MFW_GR00T_API_TOKEN"

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


DEFAULT_REPO_ID = "nvidia/GR00T-N1.7-3B"

#: Values of an environment flag that mean "off" (matches Isaac-GR00T's rule).
_FALSEY = frozenset({"", "0", "false", "no", "off"})

#: Set before ``import gr00t`` so its ``_patch_mistral`` runs.
PATCH_MISTRAL_ENV = "GROOT_PATCH_MISTRAL"
OFFLINE_ENVS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _flag(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() not in _FALSEY


def hf_hub_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """The HuggingFace hub cache directory, by huggingface_hub's own precedence.

    ``HF_HUB_CACHE`` > ``HUGGINGFACE_HUB_CACHE`` > ``HF_HOME``/hub >
    ``~/.cache/huggingface/hub``. Pure stdlib: no Hub import, no network.
    """
    env = os.environ if env is None else env
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(name):
            return Path(env[name]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_snapshot(repo_id: str, cache_dir: Path | str | None = None) -> Path | None:
    """The local snapshot directory for ``repo_id``, or ``None`` if not cached.

    Follows the cache layout ``models--<org>--<name>/refs/main`` -> commit ->
    ``snapshots/<commit>``. When ``refs/main`` is missing or stale, the most
    recently modified snapshot that has a ``config.json`` is used. A snapshot
    without ``config.json`` is an interrupted download and does not count.
    """
    if "/" not in repo_id:
        return None
    cache = Path(cache_dir) if cache_dir is not None else hf_hub_cache_dir()
    root = cache / ("models--" + repo_id.replace("/", "--"))
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = root / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if (candidate / "config.json").is_file():
            return candidate
    complete = [d for d in snapshots.iterdir() if (d / "config.json").is_file()]
    if not complete:
        return None
    return max(complete, key=lambda d: d.stat().st_mtime)


def configure_environment(
    env: MutableMapping[str, str] | None = None, patch_mistral: bool = True
) -> list[str]:
    """Prepare this process's environment for loading GR00T; return warnings.

    Must run before ``import gr00t``: its ``__init__`` reads
    ``GROOT_PATCH_MISTRAL`` at import time. Only this process is affected --
    which is the point: these variables must never reach Isaac Sim.
    """
    env = os.environ if env is None else env
    warnings: list[str] = []
    if patch_mistral:
        env.setdefault(PATCH_MISTRAL_ENV, "1")
    else:
        # Isaac-GR00T tests the raw string for truthiness, so "0" would still
        # enable it; removing the variable is the only real opt-out.
        env.pop(PATCH_MISTRAL_ENV, None)
    unset = [name for name in OFFLINE_ENVS if not _flag(env, name)]
    if unset:
        warnings.append(
            f"{' and '.join(unset)} not set: transformers may contact the HuggingFace Hub "
            "while loading (the Cosmos-Reason2-2B tokenizer check), which fails or hangs "
            "offline. scripts/start_groot_server.ps1 sets both for the server only."
        )
    return warnings


def _resolve_model_path(model: str, env: Mapping[str, str] | None = None) -> str:
    """Turn a HuggingFace repo id into a local snapshot directory.

    Required on Windows. ``Gr00tPolicy.__init__`` runs ``Path(model_path)``, and
    on Windows that rewrites the forward slash in ``nvidia/GR00T-N1.7-3B`` to a
    backslash. What reaches ``AutoModel.from_pretrained`` is then
    ``nvidia\\GR00T-N1.7-3B``, which the Hub rejects:

        HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'

    Order: an existing directory is passed straight through; a repo id already
    in the local cache resolves to its snapshot with no network access at all;
    only an uncached id reaches ``snapshot_download`` -- and not when
    ``HF_HUB_OFFLINE`` is set, where that could only fail.
    """
    env = os.environ if env is None else env
    if Path(model).is_dir():
        return model

    local = cached_snapshot(model, hf_hub_cache_dir(env))
    if local is not None:
        print(f"Resolved {model} from the local cache -> {local}")
        return str(local)

    if _flag(env, "HF_HUB_OFFLINE"):
        raise FileNotFoundError(
            f"{model!r} is not a directory and is not in the HuggingFace cache at "
            f"{hf_hub_cache_dir(env)}, and HF_HUB_OFFLINE is set. Pass the snapshot "
            "directory with --model, or download it once with HF_HUB_OFFLINE unset."
        )

    from huggingface_hub import snapshot_download

    print(f"Resolving {model} to a local snapshot (download)...")
    path = snapshot_download(model)
    print(f"  -> {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T N1.7 policy server (ZeroMQ/msgpack)")
    parser.add_argument(
        "--model",
        default=DEFAULT_REPO_ID,
        help="snapshot directory, or an HF repo id (resolved from the local cache "
        "first, with no network access)",
    )
    parser.add_argument(
        "--no-patch-mistral",
        action="store_true",
        help=f"do not set {PATCH_MISTRAL_ENV}=1 before importing gr00t (set by default: "
        "it only skips a Hub call for non-Mistral tokenizers)",
    )
    parser.add_argument(
        "--embodiment",
        default="oxe_droid_relative_eef_relative_joint",
        help="the only manipulation tag that is inference-ready in the base "
        "checkpoint; LIBERO_PANDA and ROBOCASA need finetuned weights",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default loopback). Anything else requires --api-token",
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--api-token",
        default=os.environ.get(API_TOKEN_ENV),
        help=f"shared secret clients must send (default: ${API_TOKEN_ENV}); "
        "set the same variable for run_assistant",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="permit a non-loopback bind without a token (isolated networks only: "
        "the server exposes a 'kill' endpoint)",
    )
    return parser


def check_bind(host: str, api_token: str | None, allow_unauthenticated: bool) -> str | None:
    """Return an error message if this bind would expose an open server."""
    if host in LOOPBACK_HOSTS or api_token or allow_unauthenticated:
        return None
    return (
        f"refusing to bind {host!r} without an API token: NVIDIA's PolicyServer "
        "accepts get_action and kill from any peer that can reach it.\n"
        f"Pass --api-token <secret> (or set {API_TOKEN_ENV}) and give clients the "
        "same variable, or use --allow-unauthenticated on an isolated network."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    problem = check_bind(args.host, args.api_token, args.allow_unauthenticated)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.host not in LOOPBACK_HOSTS and not args.api_token:
        print(f"WARNING: serving {args.host}:{args.port} with NO authentication.", file=sys.stderr)

    # Before torch/transformers/gr00t are imported: gr00t reads the patch flag
    # at import time.
    for warning in configure_environment(patch_mistral=not args.no_patch_mistral):
        print(f"WARNING: {warning}", file=sys.stderr)

    try:
        model_path = _resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError as exc:
        print(
            f"torch is not importable here ({exc}). This script must run in the GR00T venv:\n"
            "    <groot-venv>\\Scripts\\python.exe scripts\\groot_server.py\n"
            "If torch IS installed there and this still fails with 'bad marshal data',\n"
            "the venv has corrupted .pyc caches -- see the GR00T repair notes.",
            file=sys.stderr,
        )
        return 2

    print("=" * 68)
    print("  GR00T N1.7 Policy Server")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  torch      : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu        : {torch.cuda.get_device_name(0)}")
    print(f"  model      : {model_path}")
    print(
        f"  offline    : HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '')!r} "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE', '')!r} "
        f"{PATCH_MISTRAL_ENV}={os.environ.get(PATCH_MISTRAL_ENV, '')!r}"
    )
    print(f"  pycache    : {sys.pycache_prefix or '(beside sources)'}")
    print(f"  embodiment : {args.embodiment}")
    print(f"  device     : {args.device}")
    print(f"  bind       : {args.host}:{args.port}  token={'yes' if args.api_token else 'no'}")
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

    print("\nLoading the checkpoint (~7 GB; several minutes on first run)...")
    try:
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment,
            model_path=model_path,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - report any load failure, then exit
        print(f"\nFailed to load the policy: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print(
                "\nThe GPU is full. Budget (estimated): GR00T N1.7-3B in bf16 needs\n"
                "~7-9 GB at peak; Isaac Sim takes ~10 GB. Stop Canary / the Qwen LLM\n"
                "worker before starting this server.",
                file=sys.stderr,
            )
        return 3

    print("Checkpoint loaded.")
    try:
        modality = policy.get_modality_config()
        print(f"Modality keys: {sorted(modality)}")
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"(could not list modality config: {exc})")

    server = PolicyServer(policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"\nServing on {args.host}:{args.port}. Ctrl+C to stop.\n")
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        close = getattr(server, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

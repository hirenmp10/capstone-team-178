"""Start the speech server in exactly ONE process.

    <base-python-3.11> scripts\\serve_speech.py --asr canary --device cuda --mic 1 --serve

``<base-python-3.11>`` is the *base* interpreter the canary venv was created
from (on the development machine,
``%LOCALAPPDATA%\\Programs\\Python\\Python311\\python.exe``) -- not the venv's own
``python.exe``. Every argument except ``--nemo-site-packages`` is passed through
to ``speech_worker.py``.

Where NeMo is found
-------------------
``--nemo-site-packages DIR``, else ``$MFW_NEMO_SITE_PACKAGES``, else the
canary venv under the home directory (``~/canary-venv/Lib/site-packages`` on
Windows, ``~/canary-venv/lib/pythonX.Y/site-packages`` elsewhere). On the
development machine the default resolves to exactly the path this script used
to hardcode (``C:\\Users\\adyan\\canary-venv\\Lib\\site-packages``), so behaviour
there is unchanged; on any other machine it no longer points into one person's
profile.

Why this exists, rather than running ``speech_worker.py`` under the venv:

``canary-venv\\Scripts\\python.exe`` is a launcher stub that re-executes the base
interpreter, so one command produced *two* processes. Both ran the server, both
loaded Canary (~11 GB of VRAM instead of 5.5), and both opened microphone 1.
The parent won the audio device and the child bound the port, so the process
actually serving transcripts held a dead stream: calibration reported
``ambient 1e-05, peak 1e-05`` and no amount of speaking ever crossed the
threshold. The microphone was fine -- it measured 0.995 standalone.

The package layout is the reason this file appends rather than prepends:

    venv : torch 2.11.0+cu128, no torchvision
    base : torch 2.12.0+cu130, torchvision 0.27.0+cu130

NeMo pulls in torchvision through torchmetrics, so torch and torchvision must
come from the *same* CUDA line. Putting the venv first gives cu128 torch with
cu130 torchvision and NeMo refuses to import. Appending lets the base pair win
and takes only ``nemo`` from the venv -- which is exactly the mix the forked
child stumbled into and ran successfully.

The catch (measured 2026-09-24): the base torch is a cu130 build, and on a
driver older than the R580 line (577.03 here) it reports CUDA unavailable, so
``--asr canary --device cuda`` cannot put Canary on the GPU through this
launcher. It therefore refuses to start in that case (exit status 4) and prints
the command that does work -- ``speech_worker.py`` under the canary venv's own
python (torch 2.11+cu128). ``speech_worker.py --serve`` binds its port before
loading anything, so a duplicate process from the launcher stub now exits at
once instead of loading a second model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Overrides the default NeMo site-packages location.
SITE_PACKAGES_ENV = "MFW_NEMO_SITE_PACKAGES"

_HERE = os.path.dirname(os.path.abspath(__file__))


def default_site_packages() -> Path:
    """The canary venv's site-packages under the home directory."""
    venv = Path.home() / "canary-venv"
    if os.name == "nt":
        return venv / "Lib" / "site-packages"
    matches = sorted((venv / "lib").glob("python3*/site-packages"))
    if matches:
        return matches[-1]
    return venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def resolve_site_packages(flag: str | None) -> Path:
    """Flag, then environment variable, then :func:`default_site_packages`."""
    if flag:
        return Path(flag)
    from_env = os.environ.get(SITE_PACKAGES_ENV)
    if from_env:
        return Path(from_env)
    return default_site_packages()


#: Exit status when ``--device cuda`` was asked for and this torch has no GPU.
EXIT_NO_CUDA = 4


def _requested_option(rest: list[str], name: str, default: str) -> str:
    """The value of ``--name X`` / ``--name=X`` among the pass-through args."""
    value = default
    for index, token in enumerate(rest):
        if token == name and index + 1 < len(rest):
            value = rest[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def canary_venv_python() -> Path:
    """The canary venv's own interpreter (the one with the working cu128 torch)."""
    venv = Path.home() / "canary-venv"
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


def cuda_unavailable_message(torch_module, worker_args: list[str]) -> str:
    """Why ``--device cuda`` cannot work under this interpreter, and what does."""
    version = getattr(torch_module, "__version__", "unknown")
    built_for = getattr(getattr(torch_module, "version", None), "cuda", None) or "no CUDA"
    worker = Path(_HERE) / "speech_worker.py"
    shown_args = " ".join(worker_args) or "--serve --asr canary --device cuda --mic 1"
    return (
        f"--device cuda was requested, but the torch this interpreter imports "
        f"({version}, built for CUDA {built_for}) reports CUDA unavailable.\n"
        "\n"
        "This launcher deliberately lets the BASE interpreter's torch win (see the\n"
        "module docstring). On the development machine that is torch 2.12+cu130,\n"
        "and a CUDA 13 build needs an NVIDIA driver from the R580 line or newer;\n"
        "the installed driver (577.03 when this was measured) tops out at CUDA 12.9,\n"
        "so torch sees no GPU. Canary would then load on the CPU -- or not at all.\n"
        "\n"
        "Working command (measured: 5/5 commands transcribed on the GPU): run the\n"
        "worker with the canary venv's OWN python, whose torch is 2.11+cu128:\n"
        f"    {canary_venv_python()} {worker} {shown_args}\n"
        "\n"
        "speech_worker.py --serve now claims its port before loading anything, so a\n"
        "duplicate process from the venv launcher stub exits at once instead of\n"
        "loading a second Canary. Or update the driver, or pass --device cpu."
    )


def cuda_guard(rest: list[str], import_torch=None) -> str | None:
    """An error message when ``--device cuda`` cannot be honoured, else ``None``.

    Only Canary runs on torch; whisper (CTranslate2) and parakeet (onnxruntime)
    bring their own CUDA runtimes, so torch's view says nothing about them. A
    torch that does not import at all is left for the worker to report.
    """
    if _requested_option(rest, "--device", "cpu") != "cuda":
        return None
    if _requested_option(rest, "--asr", "whisper") != "canary":
        return None
    try:
        torch = import_torch() if import_torch is not None else __import__("torch")
    except ImportError:
        return None
    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken CUDA probe is "unavailable"
        available = False
    if available:
        return None
    return cuda_unavailable_message(torch, rest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run speech_worker.py with NeMo appended from the canary venv. "
        "All other arguments go to speech_worker.py (see its --help).",
        add_help=False,
    )
    parser.add_argument(
        "--nemo-site-packages",
        default=None,
        help=f"site-packages holding nemo (default: ${SITE_PACKAGES_ENV}, else "
        f"{default_site_packages()})",
    )
    parser.add_argument("-h", "--help", action="store_true")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    if args.help:
        parser.print_help()
        print("\nspeech_worker.py options follow:\n")
        rest = ["--help"]
    else:
        site_packages = resolve_site_packages(args.nemo_site_packages)
        if not site_packages.is_dir():
            print(
                f"NeMo site-packages not found at {site_packages}.\n"
                f"Point at the canary venv with --nemo-site-packages DIR or set "
                f"{SITE_PACKAGES_ENV}=DIR (the folder that contains 'nemo').",
                file=sys.stderr,
            )
            return 2
        # Appended, never inserted. See the module docstring: prepending swaps
        # torch for the cu128 build while torchvision stays cu130, and NeMo will
        # not load.
        if str(site_packages) not in sys.path:
            sys.path.append(str(site_packages))

        # Checked with the torch this process will actually use (the base one,
        # since the venv is appended). Without this, CanaryEngine logs a quiet
        # "falling back to CPU" and the operator finds out from the latency.
        problem = cuda_guard(rest)
        if problem is not None:
            print(problem, file=sys.stderr)
            return EXIT_NO_CUDA

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from speech_worker import main as worker_main  # noqa: PLC0415

    try:
        return worker_main(rest)
    except SystemExit as exc:  # argparse --help / usage errors inside the worker
        return int(exc.code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

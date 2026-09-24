"""Start the speech server in exactly ONE process.

    "C:\\Users\\adyan\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" ^
        scripts\\serve_speech.py --asr canary --device cuda --mic 1 --serve

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
"""

from __future__ import annotations

import os
import sys

#: Where NeMo lives. Everything else deliberately resolves from the base
#: interpreter, so torch and torchvision stay on the same CUDA line.
VENV_SITE_PACKAGES = r"C:\Users\adyan\canary-venv\Lib\site-packages"

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    if not os.path.isdir(VENV_SITE_PACKAGES):
        print(
            f"NeMo site-packages not found at {VENV_SITE_PACKAGES}.\n"
            "Edit VENV_SITE_PACKAGES in this file if the virtualenv moved.",
            file=sys.stderr,
        )
        return 2

    # Appended, never inserted. See the module docstring: prepending swaps torch
    # for the cu128 build while torchvision stays cu130, and NeMo will not load.
    if VENV_SITE_PACKAGES not in sys.path:
        sys.path.append(VENV_SITE_PACKAGES)

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from speech_worker import main as worker_main  # noqa: PLC0415

    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthesise a spoken command to a 16 kHz mono WAV with Windows SAPI.

    py -3.12 scripts\\tts_to_wav.py "pick up the red block" logs\\voice\\pick.wav
    <voice-python> scripts\\speech_worker.py --asr whisper --model small.en ^
        --wav logs\\voice\\pick.wav

Why: the voice pipeline's ASR half had never been exercised without a live
microphone, so no saved run shows audio going through a model. SAPI
(``System.Speech``) ships with Windows, needs no pip install and loads no ML
model, and writing 16 kHz / 16-bit / mono directly matches what every engine in
``speech_worker.py`` consumes, so ``--wav`` does no resampling on these files.

A synthetic voice is cleaner than a person at a laptop microphone, so a correct
transcript here proves the plumbing and the engine, not robustness to room
noise or accents. Windows only; exits 2 elsewhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def build_command(text: str, output: Path, rate: int = 0) -> list[str]:
    """PowerShell invocation that writes ``text`` to ``output`` as 16 kHz PCM."""
    safe_text = text.replace("'", "''")
    safe_path = str(output).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {int(rate)}; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{safe_path}', $f); "
        f"$s.Speak('{safe_text}'); "
        "$s.Dispose()"
    )
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="the command to speak, e.g. 'pick up the red block'")
    parser.add_argument("output", help="WAV file to write (parent folders are created)")
    parser.add_argument("--rate", type=int, default=0, help="SAPI speaking rate, -10..10")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("tts_to_wav.py uses Windows SAPI and runs only on Windows.", file=sys.stderr)
        return 2
    if not -10 <= args.rate <= 10:
        print("--rate must be between -10 and 10", file=sys.stderr)
        return 2

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        build_command(args.text, output, args.rate), capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0 or not output.is_file():
        print(f"SAPI synthesis failed:\n{completed.stderr.strip()}", file=sys.stderr)
        return 1

    with wave.open(str(output), "rb") as handle:
        seconds = handle.getnframes() / handle.getframerate()
        layout = f"{handle.getframerate()} Hz, {handle.getnchannels()} ch, {8 * handle.getsampwidth()}-bit"
    print(f"wrote {output} ({seconds:.2f} s, {layout}): {args.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

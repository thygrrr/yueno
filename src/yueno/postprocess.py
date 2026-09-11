"""Optional ffmpeg conversion of the generated WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FORMATS = ("wav", "flac", "mp3")
_CODEC_ARGS = {"flac": ["-c:a", "flac"], "mp3": ["-c:a", "libmp3lame", "-q:a", "0"]}


def convert(wav: Path, fmt: str) -> Path:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
    if fmt == "wav":
        return wav
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not on PATH; keep --format wav or install ffmpeg")
    target = wav.with_suffix(f".{fmt}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), *_CODEC_ARGS[fmt], str(target)], check=True)
    return target

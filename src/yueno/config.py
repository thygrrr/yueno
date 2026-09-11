"""Project paths and pinned upstream identifiers. Environment variables override the defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

AUDIOCPP_REPO = "https://github.com/0xShug0/audio.cpp"
# dev HEAD on 2026-09-11 ("Fix UI for YuE2 (#509)"); YuE2 support is not in any tagged release yet.
AUDIOCPP_PINNED_SHA = "fbe3eedbf6c504e45189e2cdcf1b257740a28863"
AUDIOCPP_PRESET = "windows-cuda-release"
HF_REPO = "audio-cpp/Yue2-3B-GGUF"
MODEL_DIR_NAME = "Yue2-3B-GGUF"


def project_root() -> Path:
    env = os.environ.get("YUENO_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Paths:
    root: Path
    audiocpp_src: Path
    build_dir: Path
    exe: Path
    models: Path
    outputs: Path

    @classmethod
    def resolve(cls) -> "Paths":
        root = project_root()
        src = root / "third_party" / "audio.cpp"
        build_dir = src / "build" / AUDIOCPP_PRESET
        exe_env = os.environ.get("YUENO_AUDIOCPP_EXE")
        exe = Path(exe_env).expanduser().resolve() if exe_env else build_dir / "bin" / "audiocpp_cli.exe"
        models_env = os.environ.get("YUENO_MODELS_DIR")
        models = Path(models_env).expanduser().resolve() if models_env else root / "models" / MODEL_DIR_NAME
        return cls(root=root, audiocpp_src=src, build_dir=build_dir, exe=exe, models=models, outputs=root / "outputs")

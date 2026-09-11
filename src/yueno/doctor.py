"""Environment checks for building audio.cpp and running YuE2 generation."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import audiocpp, models
from .config import Paths
from .vram import query_gpu

VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True  # required for `generate`; build-only tools are not


def _tool_version(name: str, args: tuple[str, ...] = ("--version",)) -> str | None:
    exe = shutil.which(name)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return exe
    first = (out.stdout or out.stderr).strip().splitlines()
    return first[0][:80] if first else exe


def _visual_studio() -> str | None:
    if not VSWHERE.is_file():
        return None
    out = subprocess.run([str(VSWHERE), "-latest", "-products", "*", "-property", "displayName"],
                         capture_output=True, text=True, timeout=15).stdout.strip()
    return out or None


def run_checks(paths: Paths, quant: str = models.DEFAULT_QUANT, vae: str = models.DEFAULT_VAE) -> list[Check]:
    checks: list[Check] = []

    gpu = query_gpu()
    if gpu:
        checks.append(Check("GPU", True, f"{gpu.name}, driver {gpu.driver}, {gpu.free_mib}/{gpu.total_mib} MiB free"))
    else:
        checks.append(Check("GPU", False, "nvidia-smi not found or no NVIDIA GPU"))

    cuda = audiocpp.find_cuda_toolkit()
    cuda_detail = str(cuda) if cuda else f"no v{audiocpp.MIN_CUDA_MAJOR}.x toolkit (needed to build)"
    checks.append(Check("CUDA toolkit", cuda is not None, cuda_detail, required=False))
    vs = _visual_studio()
    checks.append(Check("Visual Studio", vs is not None, vs or "vswhere found no installation", required=False))
    for tool in ("cmake", "ninja", "git"):
        ver = _tool_version(tool)
        checks.append(Check(tool, ver is not None, ver or "not on PATH", required=False))
    ff = _tool_version("ffmpeg", ("-version",))
    checks.append(Check("ffmpeg", ff is not None, ff or "not on PATH (only needed for --format flac/mp3)",
                        required=False))

    if paths.exe.is_file():
        ok = audiocpp.supports_yue2(paths.exe)
        checks.append(Check("audiocpp_cli", ok, str(paths.exe) + ("" if ok else " (no yue2 loader; rebuild)")))
    else:
        checks.append(Check("audiocpp_cli", False, f"{paths.exe} missing; run 'yueno build'"))

    missing = models.missing_files(paths.models, quant, vae)
    detail = str(paths.models) if not missing else f"missing {len(missing)} file(s); run 'yueno models pull'"
    checks.append(Check(f"models ({quant}, vae {vae})", not missing, detail))
    return checks

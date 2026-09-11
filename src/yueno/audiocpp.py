"""Clone, build, and run the audio.cpp CLI (Windows, CUDA preset)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import AUDIOCPP_PINNED_SHA, AUDIOCPP_PRESET, AUDIOCPP_REPO, Paths

CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
MIN_CUDA_MAJOR = 13  # Visual Studio 2026 needs CUDA >= 13.2; the audio.cpp CUDA release builds use 13.3.


class BuildError(RuntimeError):
    pass


def find_cuda_toolkit() -> Path | None:
    """Prefer CUDA_PATH when it is a v13+ toolkit, else the newest v13+ directory under the NVIDIA root."""
    env = os.environ.get("CUDA_PATH")
    if env and _is_usable(Path(env)):
        return Path(env)
    if not CUDA_ROOT.is_dir():
        return None
    candidates = [p for p in CUDA_ROOT.glob("v*") if _is_usable(p)]
    return max(candidates, key=_cuda_version, default=None)


def _is_usable(path: Path) -> bool:
    return _cuda_version(path)[0] >= MIN_CUDA_MAJOR and (path / "bin" / "nvcc.exe").is_file()


def _cuda_version(path: Path) -> tuple[int, ...]:
    name = path.name.lstrip("v")
    try:
        return tuple(int(x) for x in name.split("."))
    except ValueError:
        return (0,)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None,
         log: Callable[[str], None] = print) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise BuildError(f"command failed with exit code {proc.returncode}: {cmd[0]}")


def checkout(paths: Paths, sha: str = AUDIOCPP_PINNED_SHA, *, log: Callable[[str], None] = print) -> None:
    src = paths.audiocpp_src
    if not (src / ".git").is_dir():
        src.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", AUDIOCPP_REPO, str(src)], log=log)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    if head != sha:
        have = subprocess.run(["git", "cat-file", "-e", sha + "^{commit}"], cwd=src, capture_output=True)
        if have.returncode != 0:
            _run(["git", "fetch", "origin", sha], cwd=src, log=log)
        _run(["git", "checkout", "--detach", sha], cwd=src, log=log)


def build(paths: Paths, *, sha: str = AUDIOCPP_PINNED_SHA, jobs: int = 0, clean: bool = False,
          cuda_arch: str = "86-real", log: Callable[[str], None] = print) -> Path:
    cuda = find_cuda_toolkit()
    if cuda is None:
        raise BuildError(
            f"No CUDA toolkit v{MIN_CUDA_MAJOR}.x found under {CUDA_ROOT} or CUDA_PATH. Install CUDA 13.4.1 from "
            "https://developer.nvidia.com/cuda-downloads and rerun 'yueno build'."
        )
    for tool in ("git", "cmake", "ninja", "powershell"):
        if not shutil.which(tool):
            raise BuildError(f"{tool} is not on PATH")
    checkout(paths, sha, log=log)
    script = paths.audiocpp_src / "scripts" / "build_windows.ps1"
    if not script.is_file():
        raise BuildError(f"build script missing: {script}")
    env = dict(os.environ, CUDA_PATH=str(cuda), CUDAToolkit_ROOT=str(cuda))
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
           "-Preset", AUDIOCPP_PRESET, "-Target", "audiocpp_cli", "-CudaArchitectures", cuda_arch]
    if jobs > 0:
        cmd += ["-Jobs", str(jobs)]
    if clean:
        cmd.append("-Clean")
    log(f"Using CUDA toolkit at {cuda}")
    _run(cmd, cwd=paths.audiocpp_src, env=env, log=log)
    exe = paths.build_dir / "bin" / "audiocpp_cli.exe"
    if not exe.is_file():
        raise BuildError(f"build finished but {exe} is missing")
    return exe


def runtime_env() -> dict[str, str]:
    """PATH with the CUDA toolkit DLL directories first; the build does not copy cudart/cublas next to the exe."""
    env = dict(os.environ)
    cuda = find_cuda_toolkit()
    if cuda:
        dirs = [str(cuda / "bin" / "x64"), str(cuda / "bin")]
        env["PATH"] = os.pathsep.join([*dirs, env.get("PATH", "")])
    return env


def list_loaders(exe: Path) -> str:
    proc = subprocess.run([str(exe), "--list-loaders"], capture_output=True, text=True, cwd=exe.parent, timeout=60,
                          env=runtime_env())
    return proc.stdout + proc.stderr


def supports_yue2(exe: Path) -> bool:
    try:
        return "yue2" in list_loaders(exe).lower()
    except (OSError, subprocess.SubprocessError):
        return False


def source_sha(paths: Paths) -> str | None:
    if not (paths.audiocpp_src / ".git").is_dir():
        return None
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.audiocpp_src, capture_output=True, text=True)
    return proc.stdout.strip() or None


def run_streaming(cmd: list[str], *, cwd: Path, on_line: Callable[[str], None]) -> tuple[int, str]:
    """Run a command, forwarding merged stdout/stderr line by line; returns (exit code, full text)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=runtime_env())
    lines: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            lines.append(line)
            on_line(line)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        proc.stdout.close()
    return proc.wait(), "\n".join(lines)

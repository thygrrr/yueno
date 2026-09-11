"""GPU memory queries via nvidia-smi and low-VRAM heuristics."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

from .families import Family

HEADROOM_MIB = 512

_OOM_PATTERNS = (
    "out of memory",
    "cudaErrorMemoryAllocation",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "failed to allocate",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    total_mib: int
    used_mib: int
    free_mib: int


def parse_gpu_csv(line: str) -> GpuInfo:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        raise ValueError(f"unexpected nvidia-smi output: {line!r}")
    return GpuInfo(parts[0], parts[1], int(float(parts[2])), int(float(parts[3])), int(float(parts[4])))


def query_gpu(index: int = 0) -> GpuInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_gpu_csv(out.splitlines()[0]) if out else None


def vram_warning(free_mib: int, family: Family, variant: str) -> str | None:
    """Warn when the variant's measured peak is unlikely to fit; silent for variants never measured."""
    need = family.peak_mib.get(variant)
    if need is None or free_mib >= need + HEADROOM_MIB:
        return None
    fits = [v for v in family.variants if v != variant and free_mib >= family.peak_mib.get(v, 10**9) + HEADROOM_MIB]
    hint = f" Consider --package {fits[0]}." if fits else " Shorten the song or free GPU memory."
    return f"{family.name} {variant} peaks near {need} MiB but only {free_mib} MiB VRAM is free.{hint}"


def looks_like_oom(text: str) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in _OOM_PATTERNS) or bool(re.search(r"\boom\b", low))


class PeakSampler:
    """Samples nvidia-smi memory.used in a background thread and keeps the maximum."""

    def __init__(self, interval: float = 2.0, index: int = 0) -> None:
        self.interval = interval
        self.index = index
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            info = query_gpu(self.index)
            if info:
                self.peak_mib = max(self.peak_mib, info.used_mib)
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakSampler":
        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1)

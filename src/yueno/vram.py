"""GPU memory queries via nvidia-smi and 10GB-card heuristics."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

# Process VRAM at peak, rounded up. q8_0 measured 7.2 GiB on an RTX 3080 for a 147 s song (8055 MiB total with
# the desktop's ~900 MiB); q4_0/bf16 offsets follow audio.cpp's 5090 measurements (7755 / 12535 vs 8867 MiB).
PEAK_MIB = {"q8_0": 7700, "q4_0": 6600, "bf16": 11400}
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


def vram_warning(free_mib: int, quant: str) -> str | None:
    """Return a warning when the chosen quantization is unlikely to fit in the free VRAM."""
    need = PEAK_MIB.get(quant)
    if need is None or free_mib >= need + HEADROOM_MIB:
        return None
    fits = [q for q in ("q8_0", "q4_0") if free_mib >= PEAK_MIB[q] + HEADROOM_MIB]
    hint = f" Consider --quant {fits[0]}." if fits else " Lower --max-semantic-tokens or free GPU memory."
    return f"{quant} peaks near {need} MiB but only {free_mib} MiB VRAM is free.{hint}"


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

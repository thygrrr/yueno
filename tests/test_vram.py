import pytest

from yueno.families import MINIMAX_MUSIC3, YUE2
from yueno.vram import GpuInfo, PeakSampler, looks_like_oom, parse_gpu_csv, vram_warning


def test_parse_gpu_csv():
    info = parse_gpu_csv("NVIDIA GeForce RTX 3080, 616.92, 10240, 887, 9353")
    assert info == GpuInfo("NVIDIA GeForce RTX 3080", "616.92", 10240, 887, 9353)


def test_parse_gpu_csv_rejects_garbage():
    with pytest.raises(ValueError):
        parse_gpu_csv("nope")


def test_vram_warning_thresholds():
    assert vram_warning(8769, YUE2, "q8_0") is None  # RTX 3080 with the desktop running
    w = vram_warning(8000, YUE2, "q8_0")
    assert w is not None and "may fail or run slowly" in w
    w = vram_warning(6000, YUE2, "q8_0")
    assert w is not None and "shared system memory" in w and "Shorten" not in w
    assert vram_warning(8769, YUE2, "bf16") is not None
    assert vram_warning(20000, YUE2, "bf16") is None
    assert vram_warning(1000, MINIMAX_MUSIC3, "q8_0") is None  # never measured: stay silent
    assert vram_warning(9000, MINIMAX_MUSIC3, "q4_0") is not None


@pytest.mark.parametrize("text,expected", [
    ("ggml_cuda_host_malloc: failed to allocate 512 MiB", True),
    ("CUDA error: out of memory", True),
    ("error: OOM during prefill", True),
    ("wall 12.3s rtf 0.61 realtime 1.6x", False),
    ("room for improvement", False),
])
def test_looks_like_oom(text, expected):
    assert looks_like_oom(text) is expected


def test_process_peaks_use_simultaneous_totals_and_preserve_zero():
    sampler = PeakSampler()
    assert sampler.process_shared_mib is None
    sampler._record_process_memory(100 * 2**20, 0)
    assert sampler.process_shared_mib == 0
    sampler._record_process_memory(100 * 2**20, 500 * 2**20)
    sampler._record_process_memory(600 * 2**20, 0)
    assert sampler.process_dedicated_mib == 600
    assert sampler.process_shared_mib == 500
    assert sampler.process_total_mib == 600


@pytest.mark.parametrize("available", [True, False])
def test_sampler_falls_back_and_closes_counters(monkeypatch, available):
    import yueno.vram as vram

    sampler = PeakSampler()
    closed = []

    class Memory:
        def __init__(self, pid):
            assert pid == 123
            if not available:
                raise OSError("counters unavailable")

        def sample(self):
            return (100 * 2**20, 20 * 2**20)

        def close(self):
            closed.append(True)

    def gpu(index):
        sampler._stop.set()
        return GpuInfo("RTX", "1", 6144, 200, 5944)

    monkeypatch.setattr(vram, "ProcessGpuMemory", Memory)
    monkeypatch.setattr(vram, "query_gpu", gpu)
    with sampler:
        sampler.start(123)
        sampler._thread.join(timeout=2)
    assert sampler.peak_mib == 200
    assert sampler.process_shared_mib == (20 if available else None)
    assert closed == ([True] if available else [])

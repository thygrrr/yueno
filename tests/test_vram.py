import pytest

from yueno.vram import GpuInfo, looks_like_oom, parse_gpu_csv, vram_warning


def test_parse_gpu_csv():
    info = parse_gpu_csv("NVIDIA GeForce RTX 3080, 616.92, 10240, 887, 9353")
    assert info == GpuInfo("NVIDIA GeForce RTX 3080", "616.92", 10240, 887, 9353)


def test_parse_gpu_csv_rejects_garbage():
    with pytest.raises(ValueError):
        parse_gpu_csv("nope")


def test_vram_warning_thresholds():
    assert vram_warning(8769, "q8_0") is None  # RTX 3080 with the desktop running
    w = vram_warning(8000, "q8_0")
    assert w is not None and "--quant q4_0" in w
    w = vram_warning(6000, "q8_0")
    assert w is not None and "--max-semantic-tokens" in w
    assert vram_warning(8769, "bf16") is not None
    assert vram_warning(20000, "bf16") is None
    assert vram_warning(8769, "unknown") is None


@pytest.mark.parametrize("text,expected", [
    ("ggml_cuda_host_malloc: failed to allocate 512 MiB", True),
    ("CUDA error: out of memory", True),
    ("error: OOM during prefill", True),
    ("wall 12.3s rtf 0.61 realtime 1.6x", False),
    ("room for improvement", False),
])
def test_looks_like_oom(text, expected):
    assert looks_like_oom(text) is expected

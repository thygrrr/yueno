import pytest

from yueno import models
from yueno.config import Paths
from yueno.families import MINIMAX_MUSIC3, YUE2, FamilyError


def _paths(tmp_path, monkeypatch) -> Paths:
    monkeypatch.setenv("YUENO_HOME", str(tmp_path))
    return Paths.resolve()


def test_spec_for_falls_back_and_model_dir(tmp_path, monkeypatch):
    paths = _paths(tmp_path, monkeypatch)
    spec = models.spec_for(paths, YUE2)
    assert spec.repo == "audio-cpp/Yue2-3B-GGUF"
    assert models.model_dir(paths, spec) == tmp_path / "models" / "Yue2-3B-GGUF"
    assert models.model_dir(paths, models.spec_for(paths, MINIMAX_MUSIC3)).name == "MiniMax-Music3-GGUF"


def test_required_files_default():
    files = models.required_files(YUE2, YUE2.fallback_spec, "q8_0")
    assert files[0] == "yue2-3b-q8_0.gguf" and "yue2-vae-f16.gguf" in files and len(files) == 6


@pytest.mark.parametrize("variant,settings", [("q9", {}), ("q8_0", {"vae": "f8"})])
def test_required_files_rejects_unknown(variant, settings):
    with pytest.raises(FamilyError):
        models.required_files(YUE2, YUE2.fallback_spec, variant, settings)


def test_missing_and_inventory(tmp_path):
    spec = YUE2.fallback_spec
    (tmp_path / "sidecars").mkdir()
    (tmp_path / "yue2-3b-q4_0.gguf").write_bytes(b"x" * 10)
    for rel in spec.package("yue2_vae_f16").files[1:]:
        (tmp_path / rel).write_text("{}")
    assert models.missing_files(tmp_path, models.required_files(YUE2, spec, "q4_0")) == ["yue2-vae-f16.gguf"]
    assert "yue2-3b-q8_0.gguf" in models.missing_files(tmp_path, models.required_files(YUE2, spec, "q8_0"))
    inv = {f.relpath: f for f in models.inventory(tmp_path, spec)}
    assert inv["yue2-3b-q4_0.gguf"].size == 10 and inv["yue2-3b-q4_0.gguf"].present
    assert not inv["yue2-3b-bf16.gguf"].present
    assert len(inv) == 9  # 3 main + 2 vae + 4 sidecars, deduplicated


def test_pull_skips_present_files(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, revision, local_dir):
        assert repo_id == "audio-cpp/Yue2-3B-GGUF" and revision == "main"
        calls.append(filename)
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    (tmp_path / "yue2-vae-f16.gguf").write_bytes(b"v")
    files = models.required_files(YUE2, YUE2.fallback_spec, "q4_0")
    got = models.pull(tmp_path, YUE2.fallback_spec, files, log=lambda s: None)
    assert "yue2-vae-f16.gguf" not in calls
    assert "yue2-3b-q4_0.gguf" in calls and len(got) == 5
    assert models.missing_files(tmp_path, files) == []

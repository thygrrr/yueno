import pytest

from yueno import models


def test_required_files_default():
    files = models.required_files("q8_0", "f16")
    assert files[0] == "yue2-3b-q8_0.gguf" and files[1] == "yue2-vae-f16.gguf"
    assert len(files) == 2 + len(models.SIDECARS)


@pytest.mark.parametrize("quant,vae", [("q9", "f16"), ("q8_0", "f8")])
def test_required_files_rejects_unknown(quant, vae):
    with pytest.raises(ValueError):
        models.required_files(quant, vae)


def test_missing_and_inventory(tmp_path):
    (tmp_path / "sidecars").mkdir()
    (tmp_path / "yue2-3b-q4_0.gguf").write_bytes(b"x" * 10)
    for rel in models.SIDECARS:
        (tmp_path / rel).write_text("{}")
    assert models.missing_files(tmp_path, "q4_0", "f16") == ["yue2-vae-f16.gguf"]
    assert "yue2-3b-q8_0.gguf" in models.missing_files(tmp_path, "q8_0", "f16")
    inv = {f.relpath: f for f in models.inventory(tmp_path)}
    assert inv["yue2-3b-q4_0.gguf"].size == 10 and inv["yue2-3b-q4_0.gguf"].present
    assert not inv["yue2-3b-bf16.gguf"].present


def test_pull_skips_present_files(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, local_dir):
        calls.append(filename)
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    (tmp_path / "yue2-vae-f16.gguf").write_bytes(b"v")
    got = models.pull(tmp_path, "q4_0", "f16", log=lambda s: None)
    assert "yue2-vae-f16.gguf" not in calls
    assert "yue2-3b-q4_0.gguf" in calls and len(got) == 1 + len(models.SIDECARS)
    assert models.missing_files(tmp_path, "q4_0", "f16") == []

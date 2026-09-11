from pathlib import Path

import pytest

from yueno.families import FAMILIES, MINIMAX_MUSIC3, YUE2, FamilyError, get_family
from yueno.request import RequestError, SongRequest


def make(**kw) -> SongRequest:
    base = dict(style="lofi hip hop, mellow", lyrics="[Verse]\nla la la\n", seed=42)
    base.update(kw)
    return SongRequest(**base)


def test_registry():
    assert set(FAMILIES) == {"yue2", "minimax_music3"}
    assert get_family("yue2") is YUE2
    with pytest.raises(FamilyError, match="unknown model"):
        get_family("nope")


def test_yue2_packages_and_files():
    assert YUE2.packages("q8_0", {}) == ["yue2_main_q8_0", "yue2_vae_f16"]
    assert YUE2.packages("q4_0", {"vae": "f32"}) == ["yue2_main_q4_0", "yue2_vae_f32"]
    files = YUE2.component_files(YUE2.fallback_spec, "q8_0", {})
    assert files[0] == "yue2-3b-q8_0.gguf" and "yue2-vae-f16.gguf" in files and len(files) == 6
    with pytest.raises(FamilyError):
        YUE2.packages("q2", {})
    with pytest.raises(FamilyError):
        YUE2.packages("q8_0", {"vae": "f8"})


def test_yue2_build_args_shape(tmp_path):
    req = make(extra={"cot": "off", "semantic_max_tokens": 1500}, guidance=1.5, steps=16)
    args = YUE2.build_args(req, exe=Path("C:/x/audiocpp_cli.exe"), model_dir=tmp_path, spec=None, variant="q4_0",
                           settings={"vae": "f32"}, threads=8, out=tmp_path / "o.wav")
    assert args[:7] == ["C:\\x\\audiocpp_cli.exe", "--task", "gen", "--family", "yue2", "--model", str(tmp_path)]
    assert "yue2.model_gguf=yue2-3b-q4_0.gguf" in args and "yue2.vae_gguf=yue2-vae-f32.gguf" in args
    assert args[args.index("--lyrics") + 1] == req.lyrics
    for opt in ("style=lofi hip hop, mellow", "cot=off", "semantic_max_tokens=1500", "cfg_scale=1.5",
                "num_inference_steps=16"):
        assert opt in args
    assert args[args.index("--seed") + 1] == "42"
    assert "--metrics" in args and "--log" in args and "--json" not in args
    assert args[args.index("--out") + 1] == str(tmp_path / "o.wav")


def test_yue2_build_args_json_and_device(tmp_path):
    args = YUE2.build_args(make(), exe=Path("a.exe"), model_dir=tmp_path, spec=None, variant="q8_0", settings={},
                           threads=4, out=tmp_path / "o.wav", json_output=True, device=1)
    assert args[args.index("--device") + 1] == "1" and args[-1] == "--json"


def test_build_args_requires_resolved_seed(tmp_path):
    with pytest.raises(RequestError, match="unresolved"):
        YUE2.build_args(make(seed=None), exe=Path("a.exe"), model_dir=tmp_path, spec=None, variant="q8_0",
                        settings={}, threads=4, out=tmp_path / "o.wav")


def test_minimax_packages_and_files():
    assert MINIMAX_MUSIC3.packages("q4_0", {}) == ["minimax_music3_q4_0"]
    files = MINIMAX_MUSIC3.component_files(MINIMAX_MUSIC3.fallback_spec, "q8_0", {})
    assert "language_model_q8_0.gguf" in files and "transformer_q8_0.gguf" in files and "vocoder.gguf" in files
    assert "rvq_depth_decoder_q8_0.gguf" in files and len(files) == 13
    assert "rvq_depth_decoder_bf16.gguf" in MINIMAX_MUSIC3.component_files(MINIMAX_MUSIC3.fallback_spec, "bf16", {})


def test_minimax_build_args_shape(tmp_path):
    req = make(duration=45, guidance=2.0, extra={"ar_guidance_scale": 1.2, "cot": "off"})
    args = MINIMAX_MUSIC3.build_args(req, exe=Path("a.exe"), model_dir=tmp_path, spec=None, variant="q4_0",
                                     settings={}, threads=8, out=tmp_path / "o.wav")
    assert args[3:5] == ["--family", "minimax_music3"]
    assert "minimax_music3.language_model_gguf=language_model_q4_0.gguf" in args
    assert "minimax_music3.rvq_depth_decoder_gguf=rvq_depth_decoder_q8_0.gguf" in args
    assert "minimax_music3.flow_transformer_gguf=transformer_q4_0.gguf" in args
    assert args[args.index("--text") + 1] == req.style
    assert f"lyrics={req.lyrics}" in args
    for opt in ("duration_sec=45", "guidance_scale=2.0", "ar_guidance_scale=1.2"):
        assert opt in args
    assert not any(a.startswith("cot=") for a in args)
    assert args[args.index("--seed") + 1] == "42"


def test_minimax_validation_uses_spec_ranges():
    with pytest.raises(FamilyError, match="ar_guidance_scale"):
        MINIMAX_MUSIC3.validate(make(extra={"ar_guidance_scale": -1}), None)
    MINIMAX_MUSIC3.validate(make(extra={"ar_guidance_scale": 0}), None)

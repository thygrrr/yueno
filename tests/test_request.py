import json
import random
from pathlib import Path

import pytest

from yueno.request import RequestError, SongRequest, build_generate_args


def make(**kw) -> SongRequest:
    base = dict(style="lofi hip hop, mellow", lyrics="[Verse]\nla la la\n")
    base.update(kw)
    return SongRequest(**base)


def test_valid_default_request():
    make().validate()


@pytest.mark.parametrize("kw,fragment", [
    (dict(style=" "), "style"),
    (dict(lyrics=""), "lyrics"),
    (dict(cot="plan"), "cot"),
    (dict(cot="off", abc="X:1"), "ABC"),
    (dict(cfg_scale=25.0), "cfg_scale"),
    (dict(steps=0), "steps"),
    (dict(semantic_max_tokens=-1), "semantic_max_tokens"),
    (dict(abc="X:1", abc_file="a.abc"), "either"),
    (dict(seed="7"), "seed"),
    (dict(seed=True), "seed"),
])
def test_invalid_requests(kw, fragment):
    with pytest.raises(RequestError, match=fragment):
        make(**kw).validate()


def test_abc_file_must_exist(tmp_path):
    with pytest.raises(RequestError, match="not found"):
        make(abc_file=str(tmp_path / "missing.abc")).validate()


def test_from_json_rejects_unknown_fields(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "bogus": 1}), encoding="utf-8")
    with pytest.raises(RequestError, match="bogus"):
        SongRequest.from_json(p)


def test_from_json_resolves_relative_abc_file(tmp_path):
    (tmp_path / "score.abc").write_text("X:1\n", encoding="utf-8")
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "cot": "melody", "abc_file": "score.abc"}), encoding="utf-8")
    req = SongRequest.from_json(p)
    assert Path(req.abc_file) == (tmp_path / "score.abc").resolve()
    req.validate()


def test_seed_defaults_to_none_and_resolves_randomly():
    req = make()
    assert req.seed is None
    a = req.resolve_seed(random.Random(1)).seed
    b = req.resolve_seed(random.Random(2)).seed
    assert isinstance(a, int) and 0 <= a < 2**31 and a != b
    assert make(seed=5).resolve_seed().seed == 5


def test_build_generate_args_requires_resolved_seed(tmp_path):
    with pytest.raises(RequestError, match="unresolved"):
        build_generate_args(make(), exe=Path("a.exe"), model_dir=tmp_path, model_gguf="m", vae_gguf="v",
                            threads=4, out=tmp_path / "o.wav")


def test_with_overrides_ignores_none():
    req = make(seed=7).with_overrides(seed=None, cot="off")
    assert req.seed == 7 and req.cot == "off"


def test_request_options():
    opts = make(cot="off", cfg_scale=1.5, steps=16, semantic_max_tokens=1500).request_options()
    assert opts == {"style": "lofi hip hop, mellow", "cot": "off", "cfg_scale": "1.5",
                    "num_inference_steps": "16", "semantic_max_tokens": "1500"}


def test_build_generate_args_shape(tmp_path):
    req = make(seed=42)
    args = build_generate_args(req, exe=Path("C:/x/audiocpp_cli.exe"), model_dir=tmp_path, model_gguf="m.gguf",
                               vae_gguf="v.gguf", threads=8, out=tmp_path / "o.wav")
    assert args[:7] == ["C:\\x\\audiocpp_cli.exe", "--task", "gen", "--family", "yue2", "--model", str(tmp_path)]
    assert "--session-option" in args and "yue2.model_gguf=m.gguf" in args and "yue2.vae_gguf=v.gguf" in args
    assert args[args.index("--lyrics") + 1] == req.lyrics
    assert "style=lofi hip hop, mellow" in args and "cot=full" in args
    assert args[args.index("--seed") + 1] == "42"
    assert "--metrics" in args and "--log" in args and "--json" not in args
    assert args[args.index("--out") + 1] == str(tmp_path / "o.wav")


def test_build_generate_args_json_and_device(tmp_path):
    args = build_generate_args(make(seed=1), exe=Path("a.exe"), model_dir=tmp_path, model_gguf="m", vae_gguf="v",
                               threads=4, out=tmp_path / "o.wav", json_output=True, device=1)
    assert args[args.index("--device") + 1] == "1"
    assert args[-1] == "--json"

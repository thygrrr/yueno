import json
import random
from pathlib import Path

import pytest

from yueno.families import MINIMAX_MUSIC3, YUE2
from yueno.request import RequestError, SongRequest, parse_models


def make(**kw) -> SongRequest:
    base = dict(style="lofi hip hop, mellow", lyrics="[Verse]\nla la la\n")
    base.update(kw)
    return SongRequest(**base)


def test_valid_default_request():
    assert make().validate() == []
    assert make().models == ("yue2",)


@pytest.mark.parametrize("kw,fragment", [
    (dict(style=" "), "style"),
    (dict(lyrics=""), "lyrics"),
    (dict(models=()), "model"),
    (dict(guidance=25.0), "guidance"),
    (dict(steps=0), "steps"),
    (dict(duration=-1), "duration"),
    (dict(seed="7"), "seed"),
    (dict(seed=True), "seed"),
    (dict(extra={"cot": "plan"}), "cot"),
    (dict(extra={"cot": "off", "abc": "X:1"}), "ABC"),
    (dict(extra={"semantic_max_tokens": -1}), "semantic_max_tokens"),
    (dict(extra={"abc": "X:1", "abc_file": "a.abc"}), "either"),
    (dict(extra={"bogus": 1}), "bogus"),
])
def test_invalid_requests(kw, fragment):
    with pytest.raises(RequestError, match=fragment):
        make(**kw).validate()


def test_abc_file_must_exist(tmp_path):
    with pytest.raises(RequestError, match="not found"):
        make(extra={"abc_file": str(tmp_path / "missing.abc")}).validate()


def test_from_json_rejects_unknown_fields(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "bogus": 1}), encoding="utf-8")
    with pytest.raises(RequestError, match="bogus"):
        SongRequest.from_json(p).validate()


def test_from_json_resolves_relative_abc_file(tmp_path):
    (tmp_path / "score.abc").write_text("X:1\n", encoding="utf-8")
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "cot": "melody", "abc_file": "score.abc"}), encoding="utf-8")
    req = SongRequest.from_json(p)
    assert Path(req.extra["abc_file"]) == (tmp_path / "score.abc").resolve()
    req.validate()


def test_from_json_legacy_cfg_scale_and_model_forms(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "cfg_scale": 1.5, "model": "minimax_music3"}), encoding="utf-8")
    req = SongRequest.from_json(p)
    assert req.guidance == 1.5 and "cfg_scale" not in req.extra
    assert req.models == ("minimax_music3",)
    p.write_text(json.dumps({"style": "s", "lyrics": "l", "model": ["yue2", "minimax_music3", "yue2"]}), encoding="utf-8")
    assert SongRequest.from_json(p).models == ("yue2", "minimax_music3")


@pytest.mark.parametrize("value", [42, ["yue2", 3], "nope", ["nope"]])
def test_parse_models_rejects_bad_values(value):
    with pytest.raises(RequestError):
        parse_models(value)


def test_parse_models_defaults():
    assert parse_models(None) == ("yue2",)
    assert parse_models([]) == ("yue2",)
    assert parse_models(" minimax_music3 ") == ("minimax_music3",)


def test_multi_model_skips_keys_per_family_but_rejects_typos():
    req = make(models=("yue2", "minimax_music3"), duration=90, extra={"cot": "off", "ar_guidance_scale": 1.2})
    notes = req.validate()
    assert any("duration" in n and "yue2" in n for n in notes)
    assert any("cot" in n and "minimax_music3" in n for n in notes)
    with pytest.raises(RequestError, match="typo_key"):
        make(models=("yue2", "minimax_music3"), extra={"typo_key": 1}).validate()
    # A yue2-only key on a minimax-only request is skipped, not rejected: the file may be reused across models.
    notes = make(models=("minimax_music3",), extra={"cot": "off"}).validate()
    assert notes == ["skipping cot for minimax_music3"]


def test_family_request_options_mapping():
    req = make(steps=16, guidance=1.5, duration=45, extra={"cot": "off", "semantic_max_tokens": 1500})
    opts, skipped = YUE2.request_options(req, None)
    assert opts == {"num_inference_steps": "16", "cfg_scale": "1.5", "cot": "off", "semantic_max_tokens": "1500"}
    assert skipped == ["duration"]
    opts, skipped = MINIMAX_MUSIC3.request_options(req, None)
    assert opts == {"num_inference_steps": "16", "guidance_scale": "1.5", "duration_sec": "45"}
    assert sorted(skipped) == ["cot", "semantic_max_tokens"]


def test_seed_defaults_to_none_and_resolves_randomly():
    req = make()
    assert req.seed is None
    a = req.resolve_seed(random.Random(1)).seed
    b = req.resolve_seed(random.Random(2)).seed
    assert isinstance(a, int) and 0 <= a < 2**31 and a != b
    assert make(seed=5).resolve_seed().seed == 5


def test_with_overrides_and_extra_ignore_none():
    req = make(seed=7).with_overrides(seed=None, duration=30).with_extra(cot=None, abc="X:1")
    assert req.seed == 7 and req.duration == 30 and req.extra == {"abc": "X:1"}


def test_to_dict_lists_models():
    assert make(models=("yue2",)).to_dict()["models"] == ["yue2"]

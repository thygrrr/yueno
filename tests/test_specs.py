import json
from pathlib import Path

import pytest

from yueno.config import Paths
from yueno.families import FAMILIES
from yueno.specs import OptionSpec, SpecError, load_spec, parse_spec

SPEC_DIR = Path(__file__).resolve().parents[1] / "third_party" / "audio.cpp" / "model_specs"

SAMPLE = {
    "family": "demo", "display_name": "Demo",
    "package_defaults": {"download": {"repo": "org/Demo-GGUF", "revision": "main"}},
    "packages": [
        {"id": "demo_q4", "precision": "q4_0", "target_directory": "Demo", "files": ["a.gguf", "cfg.json"], "default": True},
        {"id": "demo_q8", "precision": "q8_0", "target_directory": "Demo", "files": ["b.gguf", "cfg.json"]},
    ],
    "options": {
        "request": [{"name": "lyrics", "type": "string", "required": True},
                    {"name": "steps", "type": "int", "min": 1, "default": 30},
                    {"name": "mode", "type": "enum", "values": ["a", "b"], "default": "a"}],
        "session": [{"name": "mem_saver", "type": "bool", "default": True}],
    },
}


def test_parse_spec_sample():
    spec = parse_spec(SAMPLE)
    assert spec.repo == "org/Demo-GGUF" and spec.target_directory == "Demo"
    assert spec.package("demo_q8").files == ("b.gguf", "cfg.json")
    assert spec.request_options["steps"].minimum == 1 and spec.request_options["lyrics"].required
    assert spec.session_options["mem_saver"].type == "bool"
    with pytest.raises(SpecError):
        spec.package("demo_q2")
    with pytest.raises(SpecError):
        parse_spec({"packages": [{"id": "x"}]})


@pytest.mark.parametrize("opt,value,ok", [
    (OptionSpec("n", "int", minimum=1), 3, True),
    (OptionSpec("n", "int", minimum=1), 0, False),
    (OptionSpec("n", "int"), 1.5, False),
    (OptionSpec("n", "int"), "abc", False),
    (OptionSpec("f", "float", minimum=0, maximum=20), 20, True),
    (OptionSpec("f", "float", minimum=0, maximum=20), 20.5, False),
    (OptionSpec("m", "enum", values=("a", "b")), "b", True),
    (OptionSpec("m", "enum", values=("a", "b")), "c", False),
    (OptionSpec("b", "bool"), "true", True),
    (OptionSpec("b", "bool"), "maybe", False),
    (OptionSpec("s", "string"), "anything", True),
])
def test_option_check(opt, value, ok):
    assert (opt.check(value) is None) is ok


def test_load_spec_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("YUENO_HOME", str(tmp_path))
    assert load_spec(Paths.resolve(), "yue2") is None


@pytest.mark.skipif(not SPEC_DIR.is_dir(), reason="audio.cpp checkout not present")
@pytest.mark.parametrize("name", list(FAMILIES))
def test_real_specs_match_fallbacks(name):
    real = parse_spec(json.loads((SPEC_DIR / f"{name}.json").read_text(encoding="utf-8")))
    fallback = FAMILIES[name].fallback_spec
    assert real.repo == fallback.repo and real.target_directory == fallback.target_directory
    real_ids = {p.id: set(p.files) for p in real.packages}
    for pkg in fallback.packages:
        assert pkg.id in real_ids, f"{pkg.id} missing from the real spec"
        assert set(pkg.files) == real_ids[pkg.id], f"{pkg.id} file list drifted"
    for key in fallback.request_options:
        assert key in real.request_options, f"{key} not a real request option"

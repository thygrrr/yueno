import json
from pathlib import Path

from typer.testing import CliRunner

from yueno.cli import app

runner = CliRunner()
EXAMPLE = Path(__file__).resolve().parents[1] / "requests" / "example.json"


def _env(tmp_path):
    return {"YUENO_HOME": str(tmp_path), "YUENO_AUDIOCPP_EXE": str(tmp_path / "audiocpp_cli.exe")}


def _cmds(result) -> list[str]:
    """Split dry-run output into one string per command; commands span lines because lyrics contain newlines."""
    marker = "audiocpp_cli.exe --task gen"
    parts = result.stdout.split(marker)
    return [marker + p for p in parts[1:]]


def test_generate_dry_run_from_request(tmp_path):
    result = runner.invoke(app, ["generate", "--request", str(EXAMPLE), "--dry-run", "--quant", "q4_0",
                                 "--max-semantic-tokens", "1500", "--cot", "off"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmds = _cmds(result)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert "--family yue2" in cmd
    assert "yue2.model_gguf=yue2-3b-q4_0.gguf" in cmd
    assert "cot=off" in cmd and "semantic_max_tokens=1500" in cmd
    assert "late_trains-831001.wav" in cmd
    assert str(tmp_path / "models" / "Yue2-3B-GGUF") in cmd


def test_generate_dry_run_multi_model_from_file(tmp_path):
    req = tmp_path / "two.json"
    req.write_text(json.dumps({"id": "duo", "style": "s", "lyrics": "[Verse]\nl", "seed": 5,
                               "model": ["yue2", "minimax_music3"], "cot": "off", "duration_sec": 40}), encoding="utf-8")
    result = runner.invoke(app, ["generate", "-r", str(req), "--dry-run"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmds = _cmds(result)
    assert len(cmds) == 2
    assert "--family yue2" in cmds[0] and "duo-yue2-5.wav" in cmds[0] and "cot=off" in cmds[0]
    assert "--family minimax_music3" in cmds[1] and "duo-minimax_music3-5.wav" in cmds[1]
    assert "duration_sec=40" in cmds[1] and "cot=" not in cmds[1]
    assert "skipping" in result.output


def test_generate_model_flag_overrides_file_and_out_needs_single(tmp_path):
    result = runner.invoke(app, ["generate", "-r", str(EXAMPLE), "--dry-run", "-m", "minimax_music3", "--duration", "30"],
                           env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmds = _cmds(result)
    assert len(cmds) == 1 and "--family minimax_music3" in cmds[0] and "late_trains-831001.wav" in cmds[0]
    result = runner.invoke(app, ["generate", "-r", str(EXAMPLE), "--dry-run", "-m", "yue2", "-m", "minimax_music3",
                                 "--out", str(tmp_path / "x.wav")], env=_env(tmp_path))
    assert result.exit_code == 2


def test_generate_opt_passthrough_and_typo(tmp_path):
    base = ["generate", "--style", "s", "--lyrics", "[Verse]\nl", "--dry-run"]
    ok = runner.invoke(app, base + ["--opt", "semantic_top_k=20"], env=_env(tmp_path))
    assert ok.exit_code == 0 and "semantic_top_k=20" in ok.stdout
    bad = runner.invoke(app, base + ["--opt", "semantic_topk=20"], env=_env(tmp_path))
    assert bad.exit_code == 2 and "semantic_topk" in bad.output


def test_generate_dry_run_draws_random_seed(tmp_path):
    args = ["generate", "--style", "jazz", "--lyrics", "[Verse]\nhi", "--id", "rnd", "--dry-run"]
    first = runner.invoke(app, args, env=_env(tmp_path))
    second = runner.invoke(app, args, env=_env(tmp_path))
    assert first.exit_code == 0 and second.exit_code == 0
    seeds = [r.stdout.split("--seed ")[1].split()[0] for r in (first, second)]
    assert all(s.isdigit() for s in seeds) and seeds[0] != seeds[1]
    assert f"rnd-{seeds[0]}.wav" in first.stdout


def test_generate_requires_style_and_lyrics(tmp_path):
    result = runner.invoke(app, ["generate", "--style", "jazz", "--dry-run"], env=_env(tmp_path))
    assert result.exit_code == 2


def test_generate_rejects_abc_with_cot_off(tmp_path):
    abc = tmp_path / "s.abc"
    abc.write_text("X:1\n")
    result = runner.invoke(app, ["generate", "--style", "jazz", "--lyrics", "[Verse]\nhi", "--cot", "off",
                                 "--abc-file", str(abc), "--dry-run"], env=_env(tmp_path))
    assert result.exit_code == 2


def test_generate_without_binary_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["generate", "--style", "jazz", "--lyrics", "[Verse]\nhi"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "yueno build" in result.output


def test_models_list_runs(tmp_path):
    result = runner.invoke(app, ["models", "list"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "yue2-3b-q8_0.gguf" in result.stdout and "language_model_q4_0.gguf" in result.stdout
    only = runner.invoke(app, ["models", "list", "--model", "minimax_music3"], env=_env(tmp_path))
    assert only.exit_code == 0 and "yue2-3b-q8_0.gguf" not in only.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "yueno" in result.stdout


def test_parse_metrics():
    from yueno.cli import _parse_metrics
    text = "[TIMING ts=1] yue2.nar_ms 8301\nfamily=yue2\nmetrics.wall_ms=20003.1\nmetrics.rtf=0.333393\nmetrics.channels=2"
    assert _parse_metrics(text) == {"wall_ms": "20003.1", "rtf": "0.333393", "channels": "2"}


def test_was_truncated():
    from yueno.cli import _was_truncated
    marker = "yue2.semantic.truncated 1"
    assert _was_truncated("[TIMING ts=1] yue2.semantic.tokens 3500\n[TIMING ts=1] yue2.semantic.truncated 1\n", marker)
    assert not _was_truncated("[TIMING ts=1] yue2.semantic.truncated 0\n", marker)
    assert not _was_truncated("[TIMING ts=1] yue2.semantic.truncated 1\n", None)


def test_help_command_root_and_nested():
    root = runner.invoke(app, ["help"])
    assert root.exit_code == 0 and "generate" in root.stdout and "models" in root.stdout
    gen = runner.invoke(app, ["help", "generate"])
    assert gen.exit_code == 0 and "--guidance" in gen.stdout and "--model" in gen.stdout
    pull = runner.invoke(app, ["help", "models", "pull"])
    assert pull.exit_code == 0 and "--package" in pull.stdout
    bad = runner.invoke(app, ["help", "nope"])
    assert bad.exit_code == 2

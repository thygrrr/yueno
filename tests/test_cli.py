from pathlib import Path

from typer.testing import CliRunner

from yueno.cli import app

runner = CliRunner()
EXAMPLE = Path(__file__).resolve().parents[1] / "requests" / "example.json"


def _env(tmp_path):
    return {"YUENO_HOME": str(tmp_path), "YUENO_AUDIOCPP_EXE": str(tmp_path / "audiocpp_cli.exe")}


def test_generate_dry_run_from_request(tmp_path):
    result = runner.invoke(app, ["generate", "--request", str(EXAMPLE), "--dry-run", "--quant", "q4_0",
                                 "--max-semantic-tokens", "1500", "--cot", "off"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmd = result.stdout
    assert "--family yue2" in cmd
    assert "yue2.model_gguf=yue2-3b-q4_0.gguf" in cmd
    assert "cot=off" in cmd and "semantic_max_tokens=1500" in cmd
    assert "late_trains-831001.wav" in cmd


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
    assert "yue2-3b-q8_0.gguf" in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "yueno" in result.stdout


def test_parse_metrics():
    from yueno.cli import _parse_metrics
    text = "[TIMING ts=1] yue2.nar_ms 8301\nfamily=yue2\nmetrics.wall_ms=20003.1\nmetrics.rtf=0.333393\nmetrics.channels=2"
    assert _parse_metrics(text) == {"wall_ms": "20003.1", "rtf": "0.333393", "channels": "2"}


def test_was_truncated():
    from yueno.cli import _was_truncated
    assert _was_truncated("[TIMING ts=1] yue2.semantic.tokens 3500\n[TIMING ts=1] yue2.semantic.truncated 1\n")
    assert not _was_truncated("[TIMING ts=1] yue2.semantic.truncated 0\n")


def test_help_command_root_and_nested():
    root = runner.invoke(app, ["help"])
    assert root.exit_code == 0 and "generate" in root.stdout and "models" in root.stdout
    gen = runner.invoke(app, ["help", "generate"])
    assert gen.exit_code == 0 and "--cfg-scale" in gen.stdout
    pull = runner.invoke(app, ["help", "models", "pull"])
    assert pull.exit_code == 0 and "--quant" in pull.stdout
    bad = runner.invoke(app, ["help", "nope"])
    assert bad.exit_code == 2

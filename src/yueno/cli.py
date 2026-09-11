"""yueno command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, audiocpp, models, postprocess
from .config import AUDIOCPP_PINNED_SHA, Paths
from .doctor import run_checks
from .request import COT_MODES, RequestError, SongRequest, build_generate_args
from .vram import PeakSampler, looks_like_oom, query_gpu, vram_warning

app = typer.Typer(help="Generate music with YuE2 on a consumer GPU via audio.cpp GGUF inference.",
                  no_args_is_help=True, add_completion=False)
models_app = typer.Typer(help="Download and inspect GGUF weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")

out = Console()
err = Console(stderr=True)

QUANT_HELP = f"Model quantization: {', '.join(models.QUANTS)}"
VAE_HELP = f"VAE precision: {', '.join(models.VAES)}"


def _version(value: bool) -> None:
    if value:
        out.print(f"yueno {__version__}")
        raise typer.Exit()


@app.callback()
def _root(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)) -> None:
    pass


@app.command("help")
def help_command(command: list[str] = typer.Argument(None, help="Command path, e.g. generate or models pull")) -> None:
    """Show help for yueno or one of its commands (an alternative to --help that uv does not intercept)."""
    import click

    root = typer.main.get_command(app)
    ctx = click.Context(root, info_name="yueno")
    cmd = root
    for name in command or []:
        sub = cmd.get_command(ctx, name) if hasattr(cmd, "get_command") else None
        if sub is None:
            _fail(f"unknown command: {' '.join(command)}", 2)
        ctx = click.Context(sub, info_name=name, parent=ctx)
        cmd = sub
    out.print(cmd.get_help(ctx), markup=False, highlight=False)


def _fail(message: str, code: int = 1) -> None:
    err.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _check_variant(quant: str, vae: str) -> None:
    if quant not in models.QUANTS:
        _fail(f"unknown --quant {quant!r}; choose from {', '.join(models.QUANTS)}", 2)
    if vae not in models.VAES:
        _fail(f"unknown --vae {vae!r}; choose from {', '.join(models.VAES)}", 2)


def _print_checks(checks) -> bool:
    table = Table(title="yueno doctor", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    healthy = True
    for c in checks:
        if c.ok:
            status = "[green]ok[/green]"
        elif c.required:
            status = "[red]missing[/red]"
            healthy = False
        else:
            status = "[yellow]missing[/yellow]"
        table.add_row(c.name, status, c.detail)
    out.print(table)
    return healthy


@app.command()
def doctor(quant: str = typer.Option(models.DEFAULT_QUANT, help=QUANT_HELP),
           vae: str = typer.Option(models.DEFAULT_VAE, help=VAE_HELP)) -> None:
    """Report GPU, toolchain, audiocpp_cli, and model status."""
    _check_variant(quant, vae)
    healthy = _print_checks(run_checks(Paths.resolve(), quant, vae))
    if not healthy:
        raise typer.Exit(1)


@app.command()
def build(sha: str = typer.Option(AUDIOCPP_PINNED_SHA, help="audio.cpp commit to build"),
          jobs: int = typer.Option(0, help="Parallel build jobs (0 = script default)"),
          clean: bool = typer.Option(False, help="Clean the build directory first"),
          cuda_arch: str = typer.Option("86-real", help="CMAKE_CUDA_ARCHITECTURES (86 = RTX 30xx, 'auto' = detect)")) -> None:
    """Clone audio.cpp at the pinned commit and build audiocpp_cli.exe with CUDA."""
    paths = Paths.resolve()
    try:
        exe = audiocpp.build(paths, sha=sha, jobs=jobs, clean=clean, cuda_arch=cuda_arch, log=err.print)
    except audiocpp.BuildError as exc:
        _fail(str(exc))
    if not audiocpp.supports_yue2(exe):
        _fail(f"{exe} built, but --list-loaders does not report yue2; check the pinned commit")
    out.print(f"[green]built[/green] {exe}")


@models_app.command("pull")
def models_pull(quant: str = typer.Option(models.DEFAULT_QUANT, help=QUANT_HELP),
                vae: str = typer.Option(models.DEFAULT_VAE, help=VAE_HELP)) -> None:
    """Download the selected GGUF weights and sidecar files from Hugging Face."""
    _check_variant(quant, vae)
    paths = Paths.resolve()
    got = models.pull(paths.models, quant, vae, log=err.print)
    out.print(f"[green]ready[/green] {paths.models} ({len(got)} file(s) downloaded)")


def _human_size(size: int) -> str:
    return f"{size / 2**20:,.0f} MiB" if size >= 2**20 else f"{size / 2**10:,.0f} KiB"


@models_app.command("list")
def models_list() -> None:
    """Show which weight variants are present locally."""
    paths = Paths.resolve()
    table = Table(title=str(paths.models))
    table.add_column("file")
    table.add_column("size", justify="right")
    for f in models.inventory(paths.models):
        size = _human_size(f.size) if f.present else "[dim]absent[/dim]"
        table.add_row(f.relpath, size)
    out.print(table)


@app.command()
def setup(quant: str = typer.Option(models.DEFAULT_QUANT, help=QUANT_HELP),
          vae: str = typer.Option(models.DEFAULT_VAE, help=VAE_HELP),
          jobs: int = typer.Option(0, help="Parallel build jobs"),
          sha: str = typer.Option(AUDIOCPP_PINNED_SHA, help="audio.cpp commit to build")) -> None:
    """First-time setup: build audiocpp_cli, download weights, then run doctor."""
    _check_variant(quant, vae)
    paths = Paths.resolve()
    if paths.exe.is_file() and audiocpp.supports_yue2(paths.exe):
        err.print(f"audiocpp_cli already built: {paths.exe}")
    else:
        try:
            audiocpp.build(paths, sha=sha, jobs=jobs, log=err.print)
        except audiocpp.BuildError as exc:
            _fail(str(exc))
    models.pull(paths.models, quant, vae, log=err.print)
    if not _print_checks(run_checks(paths, quant, vae)):
        raise typer.Exit(1)


def _load_request(request: Path | None, style: str | None, lyrics: str | None, lyrics_file: Path | None) -> SongRequest:
    if lyrics and lyrics_file:
        raise RequestError("give either --lyrics or --lyrics-file, not both")
    if lyrics_file is not None:
        try:
            lyrics = lyrics_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise RequestError(f"cannot read lyrics file: {exc}") from exc
    if request is not None:
        req = SongRequest.from_json(request)
        return req.with_overrides(style=style, lyrics=lyrics)
    if style is None or lyrics is None:
        raise RequestError("need --request, or both --style and --lyrics/--lyrics-file")
    return SongRequest(style=style, lyrics=lyrics)


def _default_threads() -> int:
    cpus = os.cpu_count() or 8
    return max(4, min(16, cpus // 2))


@app.command()
def generate(
    request: Path | None = typer.Option(None, "--request", "-r", help="JSON request file (style, lyrics, cot, seed, ...)"),
    style: str | None = typer.Option(None, help="Style prompt: genre, instruments, vocal, language, tempo"),
    lyrics: str | None = typer.Option(None, help="Lyrics text with [Verse]/[Chorus] section tags"),
    lyrics_file: Path | None = typer.Option(None, help="Read lyrics from a UTF-8 text file"),
    song_id: str | None = typer.Option(None, "--id", help="Song id used for the default output name"),
    cot: str | None = typer.Option(None, help=f"Planning mode: {', '.join(COT_MODES)} (default full)"),
    seed: int | None = typer.Option(None, help="Random seed (default: a fresh random seed, printed and recorded)"),
    abc_file: Path | None = typer.Option(None, help="ABC score to condition on (needs cot full or melody)"),
    cfg_scale: float | None = typer.Option(None, help="Classifier-free guidance, 0..20"),
    steps: int | None = typer.Option(None, help="NAR ODE steps (default 32)"),
    max_semantic_tokens: int | None = typer.Option(None, help="Cap on semantic tokens, i.e. song length (default 9000)"),
    quant: str = typer.Option(models.DEFAULT_QUANT, help=QUANT_HELP),
    vae: str = typer.Option(models.DEFAULT_VAE, help=VAE_HELP),
    threads: int = typer.Option(_default_threads(), help="CPU threads for audiocpp_cli"),
    output: Path | None = typer.Option(None, "--out", "-o", help="Output WAV path (default outputs/<id>-<seed>.wav)"),
    fmt: str = typer.Option("wav", "--format", help=f"Final format: {', '.join(postprocess.FORMATS)} (ffmpeg)"),
    device: int | None = typer.Option(None, help="CUDA device index"),
    overwrite: bool = typer.Option(False, help="Replace an existing output file"),
    dry_run: bool = typer.Option(False, help="Print the audiocpp_cli command and exit"),
    json_output: bool = typer.Option(False, "--json", help="Ask audiocpp_cli for machine-readable output"),
    skip_vram_check: bool = typer.Option(False, help="Do not warn about free VRAM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show audiocpp_cli [TIMING]/[TRACE] lines"),
) -> None:
    """Generate a song from a style prompt and lyrics."""
    _check_variant(quant, vae)
    if fmt not in postprocess.FORMATS:
        _fail(f"unknown --format {fmt!r}; choose from {', '.join(postprocess.FORMATS)}", 2)
    try:
        req = _load_request(request, style, lyrics, lyrics_file)
        req = req.with_overrides(id=song_id, cot=cot, seed=seed, cfg_scale=cfg_scale, steps=steps,
                                 semantic_max_tokens=max_semantic_tokens,
                                 abc_file=str(abc_file.resolve()) if abc_file else None)
        req.validate()
        req = req.resolve_seed()
    except RequestError as exc:
        _fail(str(exc), 2)

    paths = Paths.resolve()
    wav = (output or paths.outputs / f"{req.id}-{req.seed}.wav").resolve()
    if wav.suffix.lower() != ".wav":
        _fail("--out must end in .wav; use --format to convert afterwards", 2)
    cmd = build_generate_args(req, exe=paths.exe, model_dir=paths.models, model_gguf=models.QUANTS[quant],
                              vae_gguf=models.VAES[vae], threads=threads, out=wav, json_output=json_output, device=device)
    if dry_run:
        out.print(subprocess.list2cmdline(cmd))
        raise typer.Exit()

    if not paths.exe.is_file():
        _fail(f"{paths.exe} not found; run 'yueno build' (or 'yueno setup')")
    missing = models.missing_files(paths.models, quant, vae)
    if missing:
        _fail(f"missing model files: {', '.join(missing)}; run 'yueno models pull --quant {quant} --vae {vae}'")
    if wav.exists() and not overwrite:
        _fail(f"{wav} exists; pass --overwrite or choose --out")
    wav.parent.mkdir(parents=True, exist_ok=True)

    gpu = query_gpu(device or 0)
    if gpu and not skip_vram_check:
        warning = vram_warning(gpu.free_mib, quant)
        if warning:
            err.print(f"[yellow]warning:[/yellow] {warning}")

    err.print(f"[bold]yueno[/bold] {req.id} seed={req.seed} cot={req.cot} quant={quant} vae={vae} -> {wav}")
    started = time.monotonic()
    with PeakSampler(index=device or 0) as sampler:
        code, log_text = audiocpp.run_streaming(cmd, cwd=paths.exe.parent, on_line=_line_printer(verbose))
    elapsed = time.monotonic() - started

    metrics = _parse_metrics(log_text)
    truncated = _was_truncated(log_text)
    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": req.to_dict(),
        "quant": quant,
        "vae": vae,
        "audiocpp_sha": audiocpp.source_sha(paths),
        "command": cmd,
        "exit_code": code,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_mib": sampler.peak_mib or None,
        "metrics": metrics,
        "truncated": truncated,
    }
    wav.with_suffix(".request.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    if code != 0:
        if looks_like_oom(log_text):
            err.print("[red]CUDA ran out of memory.[/red] Retry with --quant q4_0 or a lower "
                      "--max-semantic-tokens, and close other GPU programs.")
        _fail(f"audiocpp_cli exited with code {code}", code)
    if not wav.is_file():
        _fail(f"audiocpp_cli exited 0 but {wav} was not written")

    if truncated:
        err.print("[yellow]warning:[/yellow] the song hit --max-semantic-tokens and was cut off before its natural "
                  "ending; raise the cap or shorten the lyrics.")
    final = wav
    if fmt != "wav":
        try:
            final = postprocess.convert(wav, fmt)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            _fail(f"conversion failed: {exc}")
    peak = f", peak VRAM {sampler.peak_mib} MiB" if sampler.peak_mib else ""
    audio_s = metrics.get("audio_duration_ms")
    length = f", {float(audio_s) / 1000:.0f}s audio" if audio_s else ""
    rtf = f", RTF {float(metrics['rtf']):.2f}" if "rtf" in metrics else ""
    out.print(f"[green]done[/green] {final} ({elapsed:.0f}s{length}{rtf}{peak})")


def _line_printer(verbose: bool):
    def on_line(line: str) -> None:
        if verbose or not line.startswith(("[TIMING", "[TRACE")):
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
    return on_line


def _was_truncated(log_text: str) -> bool:
    """audio.cpp logs 'yue2.semantic.truncated 1' when generation stopped at the token cap."""
    return any(line.rstrip().endswith("yue2.semantic.truncated 1") for line in log_text.splitlines())


def _parse_metrics(log_text: str) -> dict[str, str]:
    """Collect audiocpp's trailing 'metrics.<key>=<value>' summary lines."""
    metrics: dict[str, str] = {}
    for line in log_text.splitlines():
        if line.startswith("metrics.") and "=" in line:
            key, value = line[len("metrics."):].split("=", 1)
            metrics[key.strip()] = value.strip()
    return metrics

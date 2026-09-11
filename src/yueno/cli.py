"""yueno command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, audiocpp, models, postprocess
from .config import AUDIOCPP_PINNED_SHA, Paths
from .doctor import run_checks
from .families import FAMILIES, YUE2, YUE2_COT_MODES, YUE2_VAES, Family, FamilyError, get_family
from .request import RequestError, SongRequest, parse_models
from .specs import ModelSpec, SpecError
from .vram import PeakSampler, looks_like_oom, query_gpu, vram_warning

app = typer.Typer(help="Generate music with YuE2, MiniMax Music 3, and other audio.cpp GGUF models on a consumer GPU.",
                  no_args_is_help=True, add_completion=False)
models_app = typer.Typer(help="Download and inspect GGUF weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")

out = Console()
err = Console(stderr=True)

MODEL_HELP = f"Model family: {', '.join(FAMILIES)} (default from the request file, else yue2)"
PACKAGE_HELP = "Weight variant for the family, e.g. q8_0, q4_0, bf16 (default per family)"


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


def _family(name: str) -> Family:
    try:
        return get_family(name)
    except FamilyError as exc:
        _fail(str(exc), 2)


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
def doctor() -> None:
    """Report GPU, toolchain, audiocpp_cli, and per-model weight status."""
    if not _print_checks(run_checks(Paths.resolve())):
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
    loaders = audiocpp.gen_families(exe)
    missing = sorted(set(FAMILIES) - loaders)
    if missing:
        _fail(f"{exe} built, but --list-loaders lacks {', '.join(missing)}; check the pinned commit")
    out.print(f"[green]built[/green] {exe} (gen loaders: {', '.join(sorted(loaders))})")


@dataclass(frozen=True)
class _Weights:
    family: Family
    spec: ModelSpec
    directory: Path
    variant: str
    settings: dict[str, str]
    files: list[str]


def _resolve_weights(paths: Paths, family: Family, variant: str | None, settings: dict[str, str]) -> _Weights:
    try:
        spec = models.spec_for(paths, family)
        chosen = variant or family.default_variant
        family.check_variant(chosen)
        files = models.required_files(family, spec, chosen, settings)
    except (SpecError, FamilyError, ValueError) as exc:
        _fail(str(exc), 2)
    return _Weights(family, spec, models.model_dir(paths, spec), chosen, settings, files)


@models_app.command("pull")
def models_pull(model: str = typer.Option(YUE2.name, "--model", help=MODEL_HELP),
                package: str | None = typer.Option(None, "--package", help=PACKAGE_HELP),
                quant: str | None = typer.Option(None, help="Alias of --package for yue2"),
                vae: str = typer.Option("f16", help=f"yue2 VAE precision: {', '.join(YUE2_VAES)}")) -> None:
    """Download a family's weights (one variant plus its shared files) from Hugging Face."""
    paths = Paths.resolve()
    w = _resolve_weights(paths, _family(model), package or quant, {"vae": vae})
    try:
        got = models.pull(w.directory, w.spec, w.files, log=err.print)
    except ValueError as exc:
        _fail(str(exc))
    out.print(f"[green]ready[/green] {w.family.name} {w.variant} in {w.directory} ({len(got)} file(s) downloaded)")


def _human_size(size: int) -> str:
    if size >= 2**20:
        return f"{size / 2**20:,.0f} MiB"
    return f"{size / 2**10:,.0f} KiB" if size >= 2**10 else f"{size} B"


@models_app.command("list")
def models_list(model: str | None = typer.Option(None, "--model", help="Only this family")) -> None:
    """Show which weight files are present locally, per family."""
    paths = Paths.resolve()
    names = [model] if model else list(FAMILIES)
    for name in names:
        fam = _family(name)
        try:
            spec = models.spec_for(paths, fam)
        except (SpecError, ValueError) as exc:
            _fail(str(exc))
        directory = models.model_dir(paths, spec)
        table = Table(title=f"{fam.display_name} ({fam.name}): {directory}")
        table.add_column("file")
        table.add_column("size", justify="right")
        for f in models.inventory(directory, spec):
            table.add_row(f.relpath, _human_size(f.size) if f.present else "[dim]absent[/dim]")
        out.print(table)


@app.command()
def setup(model: str = typer.Option(YUE2.name, "--model", help=MODEL_HELP),
          package: str | None = typer.Option(None, "--package", help=PACKAGE_HELP),
          jobs: int = typer.Option(0, help="Parallel build jobs"),
          sha: str = typer.Option(AUDIOCPP_PINNED_SHA, help="audio.cpp commit to build")) -> None:
    """First-time setup: build audiocpp_cli, download one family's weights, then run doctor."""
    paths = Paths.resolve()
    fam = _family(model)
    if paths.exe.is_file() and audiocpp.supports_family(paths.exe, fam.name):
        err.print(f"audiocpp_cli already built: {paths.exe}")
    else:
        try:
            audiocpp.build(paths, sha=sha, jobs=jobs, log=err.print)
        except audiocpp.BuildError as exc:
            _fail(str(exc))
    w = _resolve_weights(paths, fam, package, {"vae": "f16"})
    models.pull(w.directory, w.spec, w.files, log=err.print)
    if not _print_checks(run_checks(paths)):
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


def _parse_opts(opts: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in opts:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise RequestError(f"--opt expects key=value; got {item!r}")
        parsed[key.strip()] = value
    return parsed


def _default_threads() -> int:
    cpus = os.cpu_count() or 8
    return max(4, min(16, cpus // 2))


@app.command()
def generate(
    request: Path | None = typer.Option(None, "--request", "-r", help="JSON request file (style, lyrics, model, seed, ...)"),
    model: list[str] = typer.Option(None, "--model", "-m", help=MODEL_HELP + "; repeat for several"),
    style: str | None = typer.Option(None, help="Style prompt: genre, instruments, vocal, language, tempo"),
    lyrics: str | None = typer.Option(None, help="Lyrics text with [Verse]/[Chorus] section tags"),
    lyrics_file: Path | None = typer.Option(None, help="Read lyrics from a UTF-8 text file"),
    song_id: str | None = typer.Option(None, "--id", help="Song id used for the default output name"),
    seed: int | None = typer.Option(None, help="Random seed (default: a fresh random seed, printed and recorded)"),
    steps: int | None = typer.Option(None, help="Inference steps (yue2 NAR ODE steps 32, minimax flow steps 30)"),
    guidance: float | None = typer.Option(None, help="Guidance scale (yue2 cfg_scale, minimax guidance_scale)"),
    cfg_scale: float | None = typer.Option(None, help="Alias of --guidance"),
    duration: float | None = typer.Option(None, help="Target length in seconds for families that take one (minimax)"),
    cot: str | None = typer.Option(None, help=f"yue2 planning mode: {', '.join(YUE2_COT_MODES)} (default full)"),
    abc_file: Path | None = typer.Option(None, help="yue2 ABC score to condition on (needs cot full or melody)"),
    max_semantic_tokens: int | None = typer.Option(None, help="yue2 cap on semantic tokens, 25 per second (default 9000)"),
    opt: list[str] = typer.Option(None, "--opt", help="Any spec-listed request option as key=value; repeatable"),
    package: str | None = typer.Option(None, "--package", help=PACKAGE_HELP),
    quant: str | None = typer.Option(None, help="Alias of --package for yue2"),
    vae: str = typer.Option("f16", help=f"yue2 VAE precision: {', '.join(YUE2_VAES)}"),
    threads: int = typer.Option(_default_threads(), help="CPU threads for audiocpp_cli"),
    output: Path | None = typer.Option(None, "--out", "-o", help="Output WAV path (single model only; default outputs/<id>-<seed>.wav)"),
    fmt: str = typer.Option("wav", "--format", help=f"Final format: {', '.join(postprocess.FORMATS)} (ffmpeg)"),
    device: int | None = typer.Option(None, help="CUDA device index"),
    overwrite: bool = typer.Option(False, help="Replace existing output files"),
    dry_run: bool = typer.Option(False, help="Print the audiocpp_cli command(s) and exit"),
    json_output: bool = typer.Option(False, "--json", help="Ask audiocpp_cli for machine-readable output"),
    skip_vram_check: bool = typer.Option(False, help="Do not warn about free VRAM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show audiocpp_cli [TIMING]/[TRACE] lines"),
) -> None:
    """Generate a song from a style prompt and lyrics with one or more model families."""
    if fmt not in postprocess.FORMATS:
        _fail(f"unknown --format {fmt!r}; choose from {', '.join(postprocess.FORMATS)}", 2)
    if guidance is not None and cfg_scale is not None:
        _fail("give either --guidance or --cfg-scale", 2)
    paths = Paths.resolve()
    try:
        req = _load_request(request, style, lyrics, lyrics_file)
        req = req.with_overrides(id=song_id, seed=seed, steps=steps, duration=duration,
                                 guidance=guidance if guidance is not None else cfg_scale,
                                 models=parse_models(list(model)) if model else None)
        req = req.with_extra(cot=cot, abc_file=str(abc_file.resolve()) if abc_file else None,
                             semantic_max_tokens=max_semantic_tokens, **_parse_opts(list(opt or [])))
        families = [get_family(name) for name in req.models]
        specs = {f.name: models.spec_for(paths, f) for f in families}
        for note in req.validate(families, specs):
            err.print(f"[dim]note:[/dim] {note}")
        req = req.resolve_seed()
    except (RequestError, FamilyError, SpecError, ValueError) as exc:
        _fail(str(exc), 2)

    multi = len(families) > 1
    if output is not None and multi:
        _fail("--out applies to a single model; with several models outputs are named <id>-<model>-<seed>.wav", 2)
    settings = {"vae": vae}
    variant_for = {f.name: (package or (quant if f is YUE2 else None)) for f in families}
    plan: list[tuple[_Weights, Path, list[str]]] = []
    for fam in families:
        w = _resolve_weights(paths, fam, variant_for[fam.name], settings)
        stem = f"{req.id}-{fam.name}-{req.seed}" if multi else f"{req.id}-{req.seed}"
        wav = (output or paths.outputs / f"{stem}.wav").resolve()
        if wav.suffix.lower() != ".wav":
            _fail("--out must end in .wav; use --format to convert afterwards", 2)
        try:
            cmd = fam.build_args(req, exe=paths.exe, model_dir=w.directory, spec=w.spec, variant=w.variant,
                                 settings=settings, threads=threads, out=wav, json_output=json_output, device=device)
        except (FamilyError, RequestError, SpecError) as exc:
            _fail(str(exc), 2)
        plan.append((w, wav, cmd))

    if dry_run:
        for _, _, cmd in plan:
            typer.echo(subprocess.list2cmdline(cmd))  # plain echo: no line wrapping of long commands
        raise typer.Exit()

    if not paths.exe.is_file():
        _fail(f"{paths.exe} not found; run 'yueno build' (or 'yueno setup')")
    for w, wav, _ in plan:
        missing = models.missing_files(w.directory, w.files)
        if missing:
            _fail(f"{w.family.name}: missing model files: {', '.join(missing)}; run "
                  f"'yueno models pull --model {w.family.name} --package {w.variant}'")
        if wav.exists() and not overwrite:
            _fail(f"{wav} exists; pass --overwrite or choose --out")
    paths.outputs.mkdir(parents=True, exist_ok=True)

    failures = 0
    for w, wav, cmd in plan:
        if not _run_one(paths, req, w, wav, cmd, fmt=fmt, device=device, skip_vram_check=skip_vram_check,
                        verbose=verbose):
            failures += 1
    if failures:
        _fail(f"{failures} of {len(plan)} model run(s) failed", 1)


def _run_one(paths: Paths, req: SongRequest, w: _Weights, wav: Path, cmd: list[str], *, fmt: str,
             device: int | None, skip_vram_check: bool, verbose: bool) -> bool:
    fam = w.family
    gpu = query_gpu(device or 0)
    if gpu and not skip_vram_check:
        warning = vram_warning(gpu.free_mib, fam, w.variant)
        if warning:
            err.print(f"[yellow]warning:[/yellow] {warning}")

    err.print(f"[bold]yueno[/bold] {req.id} model={fam.name} package={w.variant} seed={req.seed} -> {wav}")
    started = time.monotonic()
    with PeakSampler(index=device or 0) as sampler:
        code, log_text = audiocpp.run_streaming(cmd, cwd=paths.exe.parent, on_line=_line_printer(verbose))
    elapsed = time.monotonic() - started

    metrics = _parse_metrics(log_text)
    truncated = _was_truncated(log_text, fam.truncation_marker)
    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": req.to_dict(),
        "model": fam.name,
        "package": w.variant,
        "settings": w.settings,
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
            lighter = [v for v in fam.variants if fam.peak_mib.get(v, 0) < fam.peak_mib.get(w.variant, 0)]
            hint = f"--package {lighter[0]}" if lighter else "a lighter --package"
            err.print(f"[red]CUDA ran out of memory.[/red] Retry with {hint} or a shorter song, "
                      "and close other GPU programs.")
        err.print(f"[red]error:[/red] {fam.name}: audiocpp_cli exited with code {code}")
        return False
    if not wav.is_file():
        err.print(f"[red]error:[/red] {fam.name}: audiocpp_cli exited 0 but {wav} was not written")
        return False

    if truncated:
        err.print("[yellow]warning:[/yellow] the song hit its token cap and was cut off before its natural "
                  "ending; raise the cap or shorten the lyrics.")
    final = wav
    if fmt != "wav":
        try:
            final = postprocess.convert(wav, fmt)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            err.print(f"[red]error:[/red] conversion failed: {exc}")
            return False
    peak = f", peak VRAM {sampler.peak_mib} MiB" if sampler.peak_mib else ""
    audio_s = metrics.get("audio_duration_ms")
    length = f", {float(audio_s) / 1000:.0f}s audio" if audio_s else ""
    rtf = f", RTF {float(metrics['rtf']):.2f}" if "rtf" in metrics else ""
    out.print(f"[green]done[/green] {fam.name}: {final} ({elapsed:.0f}s{length}{rtf}{peak})")
    return True


def _line_printer(verbose: bool):
    def on_line(line: str) -> None:
        if verbose or not line.startswith(("[TIMING", "[TRACE")):
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
    return on_line


def _was_truncated(log_text: str, marker: str | None) -> bool:
    """True when the family's 'hit the token cap' log line appears (e.g. 'yue2.semantic.truncated 1')."""
    if not marker:
        return False
    return any(line.rstrip().endswith(marker) for line in log_text.splitlines())


def _parse_metrics(log_text: str) -> dict[str, str]:
    """Collect audiocpp's trailing 'metrics.<key>=<value>' summary lines."""
    metrics: dict[str, str] = {}
    for line in log_text.splitlines():
        if line.startswith("metrics.") and "=" in line:
            key, value = line[len("metrics."):].split("=", 1)
            metrics[key.strip()] = value.strip()
    return metrics

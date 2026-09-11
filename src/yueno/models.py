"""Weight packages: which files a family variant needs, what is present, and downloading from Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Paths
from .families import Family
from .specs import ModelSpec, load_spec


@dataclass(frozen=True)
class ModelFile:
    relpath: str
    size: int | None

    @property
    def present(self) -> bool:
        return self.size is not None


def spec_for(paths: Paths, family: Family) -> ModelSpec:
    """The audio.cpp spec when the checkout has it, else the adapter's built-in fallback."""
    spec = load_spec(paths, family.name) or family.fallback_spec
    if spec is None:
        raise ValueError(f"no model spec available for {family.name}")
    return spec


def model_dir(paths: Paths, spec: ModelSpec) -> Path:
    return paths.models / spec.target_directory


def required_files(family: Family, spec: ModelSpec, variant: str, settings: dict[str, str] | None = None) -> list[str]:
    return family.component_files(spec, variant, settings or {})


def missing_files(directory: Path, files: list[str]) -> list[str]:
    return [rel for rel in files if not (directory / rel).is_file()]


def inventory(directory: Path, spec: ModelSpec) -> list[ModelFile]:
    seen: list[str] = []
    for pkg in spec.packages:
        for rel in pkg.files:
            if rel not in seen:
                seen.append(rel)
    return [ModelFile(rel, (directory / rel).stat().st_size if (directory / rel).is_file() else None) for rel in seen]


def pull(directory: Path, spec: ModelSpec, files: list[str], *, log=print) -> list[Path]:
    """Download files from the spec's Hugging Face repo into directory, skipping files already present."""
    from huggingface_hub import hf_hub_download

    if not spec.repo:
        raise ValueError(f"{spec.family} spec has no download repo")
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for rel in files:
        if (directory / rel).is_file():
            log(f"present  {rel}")
            continue
        log(f"download {rel}")
        got = hf_hub_download(repo_id=spec.repo, filename=rel, revision=spec.revision, local_dir=str(directory))
        downloaded.append(Path(got))
    return downloaded

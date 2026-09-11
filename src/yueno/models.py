"""GGUF weight variants and downloads from the audio-cpp/Yue2-3B-GGUF Hugging Face repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import HF_REPO

QUANTS = {
    "q8_0": "yue2-3b-q8_0.gguf",
    "q4_0": "yue2-3b-q4_0.gguf",
    "bf16": "yue2-3b-bf16.gguf",
}
VAES = {
    "f16": "yue2-vae-f16.gguf",
    "f32": "yue2-vae-f32.gguf",
}
SIDECARS = (
    "sidecars/yue2-model-config.json",
    "sidecars/yue2-generation-config.json",
    "sidecars/yue2-qwen.tiktoken",
    "sidecars/yue2-vae-config.json",
)
DEFAULT_QUANT = "q8_0"
DEFAULT_VAE = "f16"


@dataclass(frozen=True)
class ModelFile:
    relpath: str
    size: int | None

    @property
    def present(self) -> bool:
        return self.size is not None


def required_files(quant: str, vae: str) -> list[str]:
    if quant not in QUANTS:
        raise ValueError(f"unknown quant {quant!r}; choose from {', '.join(QUANTS)}")
    if vae not in VAES:
        raise ValueError(f"unknown vae {vae!r}; choose from {', '.join(VAES)}")
    return [QUANTS[quant], VAES[vae], *SIDECARS]


def missing_files(model_dir: Path, quant: str, vae: str) -> list[str]:
    return [rel for rel in required_files(quant, vae) if not (model_dir / rel).is_file()]


def inventory(model_dir: Path) -> list[ModelFile]:
    files = []
    for rel in [*QUANTS.values(), *VAES.values(), *SIDECARS]:
        path = model_dir / rel
        files.append(ModelFile(rel, path.stat().st_size if path.is_file() else None))
    return files


def pull(model_dir: Path, quant: str, vae: str, *, log=print) -> list[Path]:
    """Download the selected weights and all sidecars into model_dir, skipping files already present."""
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for rel in required_files(quant, vae):
        target = model_dir / rel
        if target.is_file():
            log(f"present  {rel}")
            continue
        log(f"download {rel}")
        got = hf_hub_download(repo_id=HF_REPO, filename=rel, local_dir=str(model_dir))
        downloaded.append(Path(got))
    return downloaded

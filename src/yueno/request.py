"""Song request model, validation, and translation into an audiocpp_cli command line."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

COT_MODES = ("full", "melody", "off")
SEED_BITS = 31  # audio.cpp --seed is a signed 32-bit int

class RequestError(ValueError):
    pass


@dataclass
class SongRequest:
    style: str
    lyrics: str
    id: str = "song"
    cot: str = "full"
    seed: int | None = None  # None = draw a random seed at generation time
    abc: str | None = None
    abc_file: str | None = None
    cfg_scale: float | None = None
    steps: int | None = None
    semantic_max_tokens: int | None = None
    extra_request_options: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "SongRequest":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RequestError(f"cannot read request {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RequestError(f"request {path} must be a JSON object")
        known = {f for f in cls.__dataclass_fields__ if f != "extra_request_options"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise RequestError(f"unknown request fields in {path}: {', '.join(unknown)}")
        if "style" not in data or "lyrics" not in data:
            raise RequestError(f"request {path} needs 'style' and 'lyrics'")
        req = cls(**data)
        if req.abc_file and not os.path.isabs(req.abc_file):
            req.abc_file = str((path.parent / req.abc_file).resolve())
        return req

    def with_overrides(self, **overrides: object) -> "SongRequest":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def validate(self) -> None:
        if not self.style or not self.style.strip():
            raise RequestError("style must not be empty")
        if not self.lyrics or not self.lyrics.strip():
            raise RequestError("lyrics must not be empty")
        if self.cot not in COT_MODES:
            raise RequestError(f"cot must be one of {', '.join(COT_MODES)}; got {self.cot!r}")
        if self.abc and self.abc_file:
            raise RequestError("give either abc or abc_file, not both")
        if (self.abc or self.abc_file) and self.cot == "off":
            raise RequestError("an ABC score requires cot=full or cot=melody")
        if self.abc_file and not Path(self.abc_file).is_file():
            raise RequestError(f"abc_file not found: {self.abc_file}")
        if self.cfg_scale is not None and not (0.0 <= self.cfg_scale <= 20.0):
            raise RequestError("cfg_scale must be within [0, 20]")
        if self.steps is not None and self.steps <= 0:
            raise RequestError("steps must be positive")
        if self.semantic_max_tokens is not None and self.semantic_max_tokens <= 0:
            raise RequestError("semantic_max_tokens must be positive")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise RequestError("seed must be an integer")

    def resolve_seed(self, rng: random.Random | None = None) -> "SongRequest":
        """Return a copy with a concrete seed, drawing a random one if none was given."""
        if self.seed is not None:
            return self
        return replace(self, seed=(rng or random).getrandbits(SEED_BITS))

    def request_options(self) -> dict[str, str]:
        opts = {"style": self.style, "cot": self.cot}
        if self.abc:
            opts["abc"] = self.abc
        if self.abc_file:
            opts["abc_file"] = str(Path(self.abc_file).resolve())
        if self.cfg_scale is not None:
            opts["cfg_scale"] = repr(float(self.cfg_scale))
        if self.steps is not None:
            opts["num_inference_steps"] = str(self.steps)
        if self.semantic_max_tokens is not None:
            opts["semantic_max_tokens"] = str(self.semantic_max_tokens)
        opts.update(self.extra_request_options)
        return opts

    def to_dict(self) -> dict:
        return asdict(self)


def build_generate_args(
    req: SongRequest,
    *,
    exe: Path,
    model_dir: Path,
    model_gguf: str,
    vae_gguf: str,
    threads: int,
    out: Path,
    json_output: bool = False,
    backend: str = "cuda",
    device: int | None = None,
) -> list[str]:
    args = [
        str(exe), "--task", "gen", "--family", "yue2", "--model", str(model_dir), "--backend", backend,
        "--threads", str(threads),
        "--session-option", f"yue2.model_gguf={model_gguf}",
        "--session-option", f"yue2.vae_gguf={vae_gguf}",
    ]
    if device is not None:
        args += ["--device", str(device)]
    args += ["--lyrics", req.lyrics]
    for key, value in req.request_options().items():
        args += ["--request-option", f"{key}={value}"]
    if req.seed is None:
        raise RequestError("seed is unresolved; call resolve_seed() first")
    args += ["--seed", str(req.seed), "--metrics", "--log", "--out", str(out)]
    if json_output:
        args.append("--json")
    return args

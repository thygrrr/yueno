"""Song request model: shared fields for every model family plus family-specific extras."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .families import DEFAULT_FAMILY, FAMILIES, FamilyError, get_family

if TYPE_CHECKING:
    from .families import Family
    from .specs import ModelSpec

SEED_BITS = 31  # audio.cpp --seed is a signed 32-bit int
SHARED_FIELDS = ("id", "model", "style", "lyrics", "seed", "steps", "guidance", "duration")
LEGACY_KEYS = {"cfg_scale": "guidance"}  # old request keys mapped onto shared fields
PATH_KEYS = ("abc_file",)  # extras resolved relative to the request file


class RequestError(ValueError):
    pass


@dataclass
class SongRequest:
    style: str
    lyrics: str
    id: str = "song"
    models: tuple[str, ...] = (DEFAULT_FAMILY,)
    seed: int | None = None  # None = draw a random seed at generation time
    steps: int | None = None
    guidance: float | None = None
    duration: float | None = None
    extra: dict[str, object] = field(default_factory=dict)  # family-specific request options, passed by name

    @classmethod
    def from_json(cls, path: Path) -> "SongRequest":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RequestError(f"cannot read request {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RequestError(f"request {path} must be a JSON object")
        if "style" not in data or "lyrics" not in data:
            raise RequestError(f"request {path} needs 'style' and 'lyrics'")
        data = dict(data)
        for old, new in LEGACY_KEYS.items():
            if old in data:
                data.setdefault(new, data.pop(old))
        models = parse_models(data.pop("model", None))
        extra = {k: data.pop(k) for k in list(data) if k not in SHARED_FIELDS}
        for key in PATH_KEYS:
            value = extra.get(key)
            if isinstance(value, str) and value and not os.path.isabs(value):
                extra[key] = str((path.parent / value).resolve())
        return cls(models=models, extra=extra, **data)

    def with_overrides(self, **overrides: object) -> "SongRequest":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def with_extra(self, **extra: object) -> "SongRequest":
        merged = {**self.extra, **{k: v for k, v in extra.items() if v is not None}}
        return replace(self, extra=merged)

    def validate(self, families: "list[Family] | None" = None,
                 specs: "dict[str, ModelSpec | None] | None" = None) -> list[str]:
        """Check shared fields, then each family's view of the extras. Returns notes about skipped keys."""
        if not self.style or not self.style.strip():
            raise RequestError("style must not be empty")
        if not self.lyrics or not self.lyrics.strip():
            raise RequestError("lyrics must not be empty")
        if not self.models:
            raise RequestError("at least one model is required")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise RequestError("seed must be an integer")
        if self.steps is not None and (isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0):
            raise RequestError("steps must be a positive integer")
        if self.guidance is not None and not (0.0 <= float(self.guidance) <= 20.0):
            raise RequestError("guidance must be within [0, 20]")
        if self.duration is not None and float(self.duration) <= 0:
            raise RequestError("duration must be positive")
        families = families if families is not None else [get_family(m) for m in self.models]
        specs = specs or {}
        notes: list[str] = []
        # A key no registered family knows is a typo. A key only other families know is skipped with a note,
        # so a request written for yue2 can still be run with --model minimax_music3.
        unknown_everywhere = [k for k in self.extra
                              if not any(f.knows(k, specs.get(f.name)) for f in FAMILIES.values())]
        if unknown_everywhere:
            raise RequestError(f"unknown request option(s): {', '.join(unknown_everywhere)}")
        for fam in families:
            try:
                fam.validate(self, specs.get(fam.name))
            except FamilyError as exc:
                raise RequestError(str(exc)) from exc
            _, skipped = fam.request_options(self, specs.get(fam.name))
            if skipped:
                notes.append(f"skipping {', '.join(skipped)} for {fam.name}")
        return notes

    def resolve_seed(self, rng: random.Random | None = None) -> "SongRequest":
        """Return a copy with a concrete seed, drawing a random one if none was given."""
        if self.seed is not None:
            return self
        return replace(self, seed=(rng or random).getrandbits(SEED_BITS))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["models"] = list(self.models)
        return data


def parse_models(value: object) -> tuple[str, ...]:
    """Accept a family name or a list of names; empty means the default family."""
    if value is None:
        return (DEFAULT_FAMILY,)
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        names = list(value)
    else:
        raise RequestError("model must be a string or a list of strings")
    names = [n.strip() for n in names if n.strip()]
    if not names:
        return (DEFAULT_FAMILY,)
    seen: list[str] = []
    for name in names:
        try:
            get_family(name)
        except FamilyError as exc:
            raise RequestError(str(exc)) from exc
        if name not in seen:
            seen.append(name)
    return tuple(seen)

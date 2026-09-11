"""Reader for audio.cpp's machine-readable model specs (third_party/audio.cpp/model_specs/<family>.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Paths


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Package:
    id: str
    precision: str
    target_directory: str
    files: tuple[str, ...]
    default: bool = False


@dataclass(frozen=True)
class OptionSpec:
    name: str
    type: str = "string"
    default: object = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[str, ...] = ()

    def check(self, value: object) -> str | None:
        """Return an error message when value violates the spec, else None."""
        if self.type in ("int", "float"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return f"{self.name} must be a number"
            if self.type == "int" and (isinstance(value, bool) or number != int(number)):
                return f"{self.name} must be an integer"
            if self.minimum is not None and number < self.minimum:
                return f"{self.name} must be >= {self.minimum:g}"
            if self.maximum is not None and number > self.maximum:
                return f"{self.name} must be <= {self.maximum:g}"
        elif self.type == "enum" and self.values and str(value) not in self.values:
            return f"{self.name} must be one of {', '.join(self.values)}; got {value!r}"
        elif self.type == "bool" and not isinstance(value, bool) and str(value).lower() not in ("true", "false"):
            return f"{self.name} must be true or false"
        return None


@dataclass(frozen=True)
class ModelSpec:
    family: str
    display_name: str
    repo: str
    revision: str = "main"
    packages: tuple[Package, ...] = ()
    request_options: dict[str, OptionSpec] = field(default_factory=dict)
    session_options: dict[str, OptionSpec] = field(default_factory=dict)

    def package(self, package_id: str) -> Package:
        for pkg in self.packages:
            if pkg.id == package_id:
                return pkg
        raise SpecError(f"{self.family} has no package {package_id!r}; known: {', '.join(p.id for p in self.packages)}")

    @property
    def target_directory(self) -> str:
        return self.packages[0].target_directory if self.packages else self.family


def spec_path(paths: Paths, family: str) -> Path:
    return paths.audiocpp_src / "model_specs" / f"{family}.json"


def _option(raw: dict) -> OptionSpec:
    return OptionSpec(
        name=raw["name"], type=raw.get("type", "string"), default=raw.get("default"),
        required=bool(raw.get("required", False)), minimum=raw.get("min"), maximum=raw.get("max"),
        values=tuple(raw.get("values", ())),
    )


def parse_spec(data: dict) -> ModelSpec:
    try:
        download = data.get("package_defaults", {}).get("download", {})
        packages = tuple(
            Package(id=p["id"], precision=p.get("precision", ""), target_directory=p["target_directory"],
                    files=tuple(p["files"]), default=bool(p.get("default", False)))
            for p in data.get("packages", [])
        )
        options = data.get("options", {})
        return ModelSpec(
            family=data["family"], display_name=data.get("display_name", data["family"]),
            repo=download.get("repo", ""), revision=download.get("revision", "main") or "main", packages=packages,
            request_options={o["name"]: _option(o) for o in options.get("request", [])},
            session_options={o["name"]: _option(o) for o in options.get("session", [])},
        )
    except (KeyError, TypeError) as exc:
        raise SpecError(f"malformed model spec: {exc}") from exc


def load_spec(paths: Paths, family: str) -> ModelSpec | None:
    """Parse the audio.cpp spec for a family; None when the checkout does not have it yet."""
    path = spec_path(paths, family)
    if not path.is_file():
        return None
    try:
        return parse_spec(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc

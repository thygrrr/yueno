"""Model-family adapters: how a SongRequest maps onto each audio.cpp generation family."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .specs import ModelSpec, OptionSpec, Package

if TYPE_CHECKING:
    from .request import SongRequest


class FamilyError(ValueError):
    pass


@dataclass(frozen=True)
class Family:
    name: str
    display_name: str
    variants: tuple[str, ...]
    default_variant: str
    steps_option: str | None
    guidance_option: str | None
    duration_option: str | None
    peak_mib: dict[str, int] = field(default_factory=dict)  # process VRAM per variant, measured
    truncation_marker: str | None = None
    fallback_spec: ModelSpec | None = None  # used when the audio.cpp checkout has no spec file

    # ---- request options -------------------------------------------------------------------------------------

    def shared_option_names(self) -> dict[str, str | None]:
        return {"steps": self.steps_option, "guidance": self.guidance_option, "duration": self.duration_option}

    def request_option_specs(self, spec: ModelSpec | None) -> dict[str, OptionSpec]:
        return (spec or self.fallback_spec or ModelSpec(self.name, self.name, "")).request_options

    def knows(self, key: str, spec: ModelSpec | None) -> bool:
        return key in self.request_option_specs(spec)

    def request_options(self, req: "SongRequest", spec: ModelSpec | None) -> tuple[dict[str, str], list[str]]:
        """Map shared fields and extras onto audio.cpp request options; returns (options, skipped keys)."""
        specs = self.request_option_specs(spec)
        opts: dict[str, str] = {}
        skipped: list[str] = []
        for key, target in self.shared_option_names().items():
            value = getattr(req, key)
            if value is None:
                continue
            if target is None:
                skipped.append(key)
            else:
                opts[target] = _fmt(value)
        for key, value in req.extra.items():
            if key in specs:
                opts[key] = _fmt(value)
            else:
                skipped.append(key)
        return opts, skipped

    def validate(self, req: "SongRequest", spec: ModelSpec | None) -> None:
        specs = self.request_option_specs(spec)
        opts, _ = self.request_options(req, spec)
        for key, value in opts.items():
            if key in specs:
                problem = specs[key].check(req.extra.get(key, value))
                if problem:
                    raise FamilyError(f"{self.name}: {problem}")
        self.validate_extra(req, spec)

    def validate_extra(self, req: "SongRequest", spec: ModelSpec | None) -> None:  # noqa: ARG002
        return None

    # ---- weights ---------------------------------------------------------------------------------------------

    def check_variant(self, variant: str) -> None:
        if variant not in self.variants:
            raise FamilyError(f"{self.name} has no variant {variant!r}; choose from {', '.join(self.variants)}")

    def packages(self, variant: str, settings: dict[str, str]) -> list[str]:  # noqa: ARG002
        raise NotImplementedError

    def component_files(self, spec: ModelSpec, variant: str, settings: dict[str, str]) -> list[str]:
        files: list[str] = []
        for package_id in self.packages(variant, settings):
            for rel in spec.package(package_id).files:
                if rel not in files:
                    files.append(rel)
        return files

    # ---- command line ----------------------------------------------------------------------------------------

    def build_args(self, req: "SongRequest", *, exe: Path, model_dir: Path, spec: ModelSpec | None, variant: str,
                   settings: dict[str, str], threads: int, out: Path, json_output: bool = False,
                   backend: str = "cuda", device: int | None = None) -> list[str]:
        raise NotImplementedError

    def _common_head(self, *, exe: Path, model_dir: Path, threads: int, backend: str, device: int | None) -> list[str]:
        args = [str(exe), "--task", "gen", "--family", self.name, "--model", str(model_dir), "--backend", backend,
                "--threads", str(threads)]
        if device is not None:
            args += ["--device", str(device)]
        return args

    @staticmethod
    def _common_tail(req: "SongRequest", out: Path, json_output: bool) -> list[str]:
        from .request import RequestError

        if req.seed is None:
            raise RequestError("seed is unresolved; call resolve_seed() first")
        args = ["--seed", str(req.seed), "--metrics", "--log", "--out", str(out)]
        if json_output:
            args.append("--json")
        return args


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _opt(name: str, type_: str = "string", default: object = None, required: bool = False,
         minimum: float | None = None, maximum: float | None = None, values: tuple[str, ...] = ()) -> OptionSpec:
    return OptionSpec(name, type_, default, required, minimum, maximum, values)


# ---- YuE2 -------------------------------------------------------------------------------------------------------

YUE2_SIDECARS = ("sidecars/yue2-model-config.json", "sidecars/yue2-generation-config.json",
                 "sidecars/yue2-qwen.tiktoken", "sidecars/yue2-vae-config.json")
YUE2_COT_MODES = ("full", "melody", "off")
YUE2_VAES = ("f16", "f32")

_YUE2_FALLBACK = ModelSpec(
    family="yue2", display_name="YuE2", repo="audio-cpp/Yue2-3B-GGUF",
    packages=tuple(
        [Package(f"yue2_main_{q}", q, "Yue2-3B-GGUF", (f"yue2-3b-{q}.gguf", *YUE2_SIDECARS), q == "q8_0")
         for q in ("q8_0", "q4_0", "bf16")]
        + [Package(f"yue2_vae_{v}", v, "Yue2-3B-GGUF", (f"yue2-vae-{v}.gguf", *YUE2_SIDECARS)) for v in YUE2_VAES]
    ),
    request_options={o.name: o for o in (
        _opt("style", required=True), _opt("lyrics", required=True),
        _opt("cot", "enum", "full", values=YUE2_COT_MODES), _opt("abc"), _opt("abc_file"),
        _opt("cfg_scale", "float", minimum=0.0, maximum=20.0), _opt("seed", "int", 831001, minimum=0),
        _opt("num_inference_steps", "int", 32, minimum=1),
        _opt("abc_max_tokens", "int", minimum=1), _opt("abc_temperature", "float", minimum=0.0, maximum=5.0),
        _opt("semantic_max_tokens", "int", minimum=1), _opt("semantic_min_tokens", "int", minimum=0),
        _opt("semantic_temperature", "float", minimum=0.0, maximum=5.0), _opt("semantic_top_p", "float", minimum=0.0, maximum=1.0),
        _opt("semantic_top_k", "int", minimum=1), _opt("semantic_repetition_penalty", "float", minimum=0.001),
    )},
)


@dataclass(frozen=True)
class YuE2Family(Family):
    def packages(self, variant: str, settings: dict[str, str]) -> list[str]:
        self.check_variant(variant)
        vae = settings.get("vae", "f16")
        if vae not in YUE2_VAES:
            raise FamilyError(f"yue2 has no VAE {vae!r}; choose from {', '.join(YUE2_VAES)}")
        return [f"yue2_main_{variant}", f"yue2_vae_{vae}"]

    def validate_extra(self, req: "SongRequest", spec: ModelSpec | None) -> None:  # noqa: ARG002
        cot = req.extra.get("cot", "full")
        abc, abc_file = req.extra.get("abc"), req.extra.get("abc_file")
        if abc and abc_file:
            raise FamilyError("yue2: give either abc or abc_file, not both")
        if (abc or abc_file) and cot == "off":
            raise FamilyError("yue2: an ABC score requires cot=full or cot=melody")
        if abc_file and not Path(str(abc_file)).is_file():
            raise FamilyError(f"yue2: abc_file not found: {abc_file}")

    def build_args(self, req, *, exe, model_dir, spec, variant, settings, threads, out, json_output=False,
                   backend="cuda", device=None) -> list[str]:
        self.check_variant(variant)
        vae = settings.get("vae", "f16")
        args = self._common_head(exe=exe, model_dir=model_dir, threads=threads, backend=backend, device=device)
        args += ["--session-option", f"yue2.model_gguf=yue2-3b-{variant}.gguf",
                 "--session-option", f"yue2.vae_gguf=yue2-vae-{vae}.gguf",
                 "--lyrics", req.lyrics]
        opts, _ = self.request_options(req, spec)
        opts = {"style": req.style, **opts}
        if "abc_file" in opts:
            opts["abc_file"] = str(Path(opts["abc_file"]).resolve())
        for key, value in opts.items():
            args += ["--request-option", f"{key}={value}"]
        return args + self._common_tail(req, out, json_output)


# ---- MiniMax Music 3 --------------------------------------------------------------------------------------------

_MM3_COMMON = ("config.json", "config/language_model.json", "config/rvq_depth_decoder.json",
               "config/condition_encoder.json", "config/transformer.json", "config/vocoder.json",
               "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "condition_encoder.gguf", "vocoder.gguf")

_MM3_FALLBACK = ModelSpec(
    family="minimax_music3", display_name="MiniMax Music 3", repo="audio-cpp/MiniMax-Music3-GGUF",
    packages=tuple(
        Package(f"minimax_music3_{p}", p, "MiniMax-Music3-GGUF",
                (*_MM3_COMMON, f"language_model_{p}.gguf",
                 f"rvq_depth_decoder_{'bf16' if p == 'bf16' else 'q8_0'}.gguf", f"transformer_{p}.gguf"),
                p == "q4_0")
        for p in ("q4_0", "q8_0", "bf16")
    ),
    request_options={o.name: o for o in (
        _opt("lyrics", required=True), _opt("duration_sec", "float", 20.0, minimum=0.1),
        _opt("num_inference_steps", "int", 30, minimum=1), _opt("guidance_scale", "float", 1.7, minimum=0.0),
        _opt("ar_guidance_scale", "float", 1.5, minimum=0.0), _opt("top_k", "int", 50, minimum=1),
        _opt("seed", "int", 0, minimum=0),
    )},
)

_MM3_COMPONENTS = {"language_model": "language_model_gguf", "rvq_depth_decoder": "rvq_depth_decoder_gguf",
                   "transformer": "flow_transformer_gguf"}


@dataclass(frozen=True)
class MiniMaxMusic3Family(Family):
    def packages(self, variant: str, settings: dict[str, str]) -> list[str]:  # noqa: ARG002
        self.check_variant(variant)
        return [f"minimax_music3_{variant}"]

    def build_args(self, req, *, exe, model_dir, spec, variant, settings, threads, out, json_output=False,
                   backend="cuda", device=None) -> list[str]:
        self.check_variant(variant)
        package = (spec or _MM3_FALLBACK).package(f"minimax_music3_{variant}")
        args = self._common_head(exe=exe, model_dir=model_dir, threads=threads, backend=backend, device=device)
        for prefix, option in _MM3_COMPONENTS.items():
            gguf = next((f for f in package.files if f.startswith(prefix + "_") and f.endswith(".gguf")), None)
            if gguf is None:
                raise FamilyError(f"minimax_music3 package {package.id} has no {prefix} GGUF")
            args += ["--session-option", f"minimax_music3.{option}={gguf}"]
        args += ["--text", req.style, "--request-option", f"lyrics={req.lyrics}"]
        opts, _ = self.request_options(req, spec)
        for key, value in opts.items():
            args += ["--request-option", f"{key}={value}"]
        return args + self._common_tail(req, out, json_output)


# ---- registry ---------------------------------------------------------------------------------------------------

YUE2 = YuE2Family(
    name="yue2", display_name="YuE2 3B", variants=("q8_0", "q4_0", "bf16"), default_variant="q8_0",
    steps_option="num_inference_steps", guidance_option="cfg_scale", duration_option=None,
    # q8_0 measured 7.2 GiB process VRAM on an RTX 3080 (147 s song); the others follow audio.cpp's 5090 offsets.
    peak_mib={"q8_0": 7700, "q4_0": 6600, "bf16": 11400},
    truncation_marker="yue2.semantic.truncated 1", fallback_spec=_YUE2_FALLBACK,
)

MINIMAX_MUSIC3 = MiniMaxMusic3Family(
    name="minimax_music3", display_name="MiniMax Music 3", variants=("q4_0", "q8_0", "bf16"), default_variant="q4_0",
    steps_option="num_inference_steps", guidance_option="guidance_scale", duration_option="duration_sec",
    # q4_0 measured 9807 MiB total on an RTX 3080 for a 20 s budget (~8900 MiB process); longer budgets need more.
    peak_mib={"q4_0": 8900}, truncation_marker=None, fallback_spec=_MM3_FALLBACK,
)

FAMILIES: dict[str, Family] = {f.name: f for f in (YUE2, MINIMAX_MUSIC3)}
DEFAULT_FAMILY = YUE2.name


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise FamilyError(f"unknown model {name!r}; choose from {', '.join(FAMILIES)}") from None

# yueno

Command-line music generation on a consumer NVIDIA GPU. yueno drives
[audio.cpp](https://github.com/0xShug0/audio.cpp) with GGUF weights, so models whose official pipelines need
24 GB of VRAM fit an RTX 3080 10 GB. Supported model families:

| `--model` | Model | Default weights | Inputs |
|---|---|---|---|
| `yue2` (default) | [YuE2 3B](https://github.com/multimodal-art-projection/YuE) | Q8_0 main + F16 VAE, 4.3 GB | style, lyrics, optional ABC score |
| `minimax_music3` | [MiniMax Music 3](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) | Q4_0 components, 7.9 GB | caption (style), lyrics, duration |

Generation runs on the GPU via CUDA. Adding a family is one adapter class in `src/yueno/families.py`; weights,
file lists, and option schemas come from audio.cpp's own `model_specs/<family>.json`.

## Prerequisites (Windows)

- NVIDIA driver 580 or newer and an Ampere or newer GPU (compute capability 7.5+).
- CUDA Toolkit 13.x (13.4.1 tested target). Visual Studio 2026 requires CUDA 13.2 or newer.
- Visual Studio 2022/2026 with the C++ desktop workload, CMake, Ninja, git.
- [uv](https://docs.astral.sh/uv/). Optional: ffmpeg for `--format flac|mp3`.

## Setup

```powershell
uv sync
uv run yueno doctor          # check GPU, toolchain, binary, per-model weights
uv run yueno setup           # clone + build audio.cpp (pinned commit), download YuE2 Q8_0 weights, doctor
uv run yueno models pull --model minimax_music3     # optional second family (Q4_0 package)
```

`setup` is `yueno build` followed by `yueno models pull`. Weights land in `models/<package>/`, the binary in
`third_party/audio.cpp/build/windows-cuda-release/bin/`. Both directories are gitignored.

## Generate

`uv run yueno help` and `uv run yueno help generate` print usage without `--help`, which uv may claim for itself.

```powershell
uv run yueno generate --request requests/example.json
uv run yueno generate --style "dark synthwave, male vocal, 110 BPM" --lyrics-file lyrics.txt --cot off --seed 7
uv run yueno generate -r song.json --package q4_0 --format flac --out outputs/take2.wav
uv run yueno generate -r song.json -m minimax_music3 --duration 90      # same song, another model
uv run yueno generate -r requests/compare.json                          # "model": [...] renders every family
```

With several models the request runs each family in turn on the same style, lyrics, and seed, and names the
outputs `<id>-<model>-<seed>.wav`. Keys a family does not understand are skipped with a note, so one request file
serves every model.

Key options:

| Option | Meaning |
|---|---|
| `--model NAME`, `-m` | Family to run; repeat for several. Overrides `model` in the request file. |
| `--package q8_0\|q4_0\|bf16` | Weight variant of the family (`--quant` is the yue2 alias; `--vae f16\|f32` picks the yue2 VAE). |
| `--steps N`, `--guidance F` | Shared knobs mapped per family: yue2 `num_inference_steps` (32) and `cfg_scale`; minimax `num_inference_steps` (30) and `guidance_scale` (1.7). `--cfg-scale` is an alias. |
| `--duration S` | Length target for families that take one (minimax `duration_sec`, an AR budget, default 20). Skipped for yue2. |
| `--cot full\|melody\|off` | yue2 symbolic planning: melody+chords, melody only, or none (default `full`). |
| `--abc-file score.abc` | yue2: condition on an ABC score (needs `full` or `melody`). |
| `--max-semantic-tokens N` | yue2: caps song length at 25 tokens per second (default 9000). |
| `--opt key=value` | Any request option a family's spec lists, e.g. `--opt ar_guidance_scale=1.2`. Repeatable. |
| `--seed N` | Reproducibility. Without a seed in the request file or on the command line, a random one is drawn, printed, and saved in the sidecar JSON. |
| `--dry-run` | Print the `audiocpp_cli` command without running it. |
| `-v` / `--verbose` | Show audio.cpp `[TIMING]`/`[TRACE]` lines. |

Each run writes `<out>.request.json` next to the audio with the effective request, model, package, command, elapsed
time, peak VRAM, and metric lines.

### Measured on an RTX 3080 10 GB (CUDA 13.4, driver 616.92)

| Run | Audio | Wall time | RTF | Peak GPU memory (whole card) |
|---|---|---|---|---|
| yue2 `--cot off --max-semantic-tokens 1500`, q8_0 | 60 s | 20 s | 0.33 | 6.4 GB |
| yue2 `--cot full`, default tokens, q8_0 | 147 s | 66 s | 0.45 | 8.1 GB |
| minimax_music3 `--duration 20`, q4_0 | 20 s | 42 s | 2.1 | 9.8 GB |
| minimax_music3 `--duration 60`, q4_0 | 48 s | 146 s | 3.0 | 9.9 GB |

MiniMax Music 3 is a much larger model (its Q4_0 language model alone is 6 GB). On a 10 GB card it runs only
because audio.cpp's `mem_saver` loads one stage at a time, which is why it is 3x slower than real time and sits
within 0.4 GB of the VRAM limit. Keep `--duration` at 60 or below and close other GPU programs; a budget that
does not fit fails with a CUDA out-of-memory error rather than degrading.

Semantic tokens map to audio at 25 tokens per second, so the default cap of 9000 allows up to six minutes.

Start with `q8_0`. If audio.cpp reports a CUDA out-of-memory error, retry with `--quant q4_0` (about 1.1 GB less)
or a lower `--max-semantic-tokens` (the KV cache and NAR graph grow with song length), and close other GPU-heavy
programs. Two renders at once on a 10 GB card will run out of memory or crawl. `yueno generate` warns before
running when free VRAM looks too small for the chosen variant. Pass `-v` to see audio.cpp's per-stage timing lines.

The `yue2.*_arena_mb` session options that audio.cpp exposes size host-side graph descriptor pools, not GPU
buffers, so they are not a VRAM lever and yueno does not expose them.

## Controlling song length

`--max-semantic-tokens` is a hard cut, not a target: the song stops mid-phrase when it hits the cap, and yueno
prints a warning (the sidecar JSON records `"truncated": true`). To get a song of a chosen length that ends
properly, make the model plan a shorter song instead:

1. **Lyrics decide the length.** The model sings everything you give it, plus intros, breaks, and an outro.
   Measured with the default planner: a verse, a chorus, and a one-line outro gave 91 s; two verses and two
   choruses gave more than 140 s. For about 1:40, use roughly two short sections plus an outro, or one verse,
   one chorus, and a half chorus.
2. **Score decides the length exactly.** With `cot: full` and an `abc_file`, the model realizes the score and
   ends when it ends. A 40-bar score at 96 BPM (100 s nominal) rendered as a 110 s song. Bars must contain
   a melody; a chord-only skeleton of rests does not constrain length. See `requests/len-b-40bars.abc` and
   `requests/len-b.json`.
3. **Keep the cap as a safety net**, comfortably above the target (25 tokens per second, so 3500 for a
   2:20 ceiling).

## Request file

```json
{
  "id": "my_song",
  "model": ["yue2", "minimax_music3"],
  "style": "genre, instruments, vocal character, language, tempo",
  "lyrics": "[Verse]\n...\n\n[Chorus]\n...",
  "seed": 831001,
  "cot": "full",
  "duration": 60
}
```

`model` is a family name or a list (default `yue2`). Shared optional fields: `seed`, `steps`, `guidance`,
`duration`. Anything else is passed to the family by its audio.cpp option name (`cot`, `abc_file`,
`semantic_max_tokens`, `ar_guidance_scale`, ...); keys unknown to every family are rejected as typos. The old
`cfg_scale` key still works as `guidance`. Command-line flags override the file. Omit `seed` to get a different
song on every run.

## Development

```powershell
uv run pytest
```

## Licenses

yueno and audio.cpp are Apache 2.0. The YuE2 weights (and their GGUF conversions) are **CC BY-NC 4.0**:
non-commercial use only.

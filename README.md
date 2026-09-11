# yueno

Command-line music generation with [YuE2](https://github.com/multimodal-art-projection/YuE) on a consumer NVIDIA
GPU. The official Python pipeline needs 24 GB of VRAM. yueno instead drives
[audio.cpp](https://github.com/0xShug0/audio.cpp) with Q8/Q4 GGUF weights, which peak at roughly 8.9 GB (Q8_0) or
7.8 GB (Q4_0), so it fits an RTX 3080 10 GB. Generation runs on the GPU via CUDA.

## Prerequisites (Windows)

- NVIDIA driver 580 or newer and an Ampere or newer GPU (compute capability 7.5+).
- CUDA Toolkit 13.x (13.4.1 tested target). Visual Studio 2026 requires CUDA 13.2 or newer.
- Visual Studio 2022/2026 with the C++ desktop workload, CMake, Ninja, git.
- [uv](https://docs.astral.sh/uv/). Optional: ffmpeg for `--format flac|mp3`.

## Setup

```powershell
uv sync
uv run yueno doctor          # check GPU, toolchain, binary, models
uv run yueno setup           # clone + build audio.cpp (pinned commit), download Q8_0 weights, doctor
```

`setup` is `yueno build` followed by `yueno models pull`. Weights land in `models/Yue2-3B-GGUF/`, the binary in
`third_party/audio.cpp/build/windows-cuda-release/bin/`. Both directories are gitignored.

## Generate

`uv run yueno help` and `uv run yueno help generate` print usage without `--help`, which uv may claim for itself.

```powershell
uv run yueno generate --request requests/example.json
uv run yueno generate --style "dark synthwave, male vocal, 110 BPM" --lyrics-file lyrics.txt --cot off --seed 7
uv run yueno generate -r song.json --quant q4_0 --format flac --out outputs/take2.wav
```

Key options:

| Option | Meaning |
|---|---|
| `--cot full\|melody\|off` | Symbolic planning: melody+chords, melody only, or none (default `full`). |
| `--abc-file score.abc` | Condition on an ABC score (needs `full` or `melody`). |
| `--quant q8_0\|q4_0\|bf16`, `--vae f16\|f32` | Weight variants; `q8_0`+`f16` is the default. |
| `--max-semantic-tokens N` | Caps song length (default 9000). Use a small value for quick smoke tests. |
| `--steps N` | NAR ODE steps (default 32). |
| `--cfg-scale F` | Classifier-free guidance on the semantic stage, 0..20. Default 1.0 with planning, 1.01 with `--cot off`. 1.0 means no guidance; higher values push harder toward the style and lyrics at the cost of variety. |
| `--seed N` | Reproducibility. Without a seed in the request file or on the command line, a random one is drawn, printed, and saved in the sidecar JSON. |
| `--dry-run` | Print the `audiocpp_cli` command without running it. |
| `-v` / `--verbose` | Show audio.cpp `[TIMING]`/`[TRACE]` lines. |

Each run writes `<out>.request.json` next to the audio with the effective request, command, elapsed time, peak VRAM,
and metric lines.

### Measured on an RTX 3080 10 GB (CUDA 13.4, driver 616.92)

| Run | Audio | Wall time | RTF | Peak GPU memory (whole card) |
|---|---|---|---|---|
| `--cot off --max-semantic-tokens 1500`, q8_0 | 60 s | 20 s | 0.33 | 6.4 GB |
| `--cot full`, default tokens, q8_0 | 147 s | 66 s | 0.45 | 8.1 GB |

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
  "style": "genre, instruments, vocal character, language, tempo",
  "lyrics": "[Verse]\n...\n\n[Chorus]\n...",
  "cot": "full",
  "seed": 831001
}
```

Optional fields: `seed`, `abc`, `abc_file`, `cfg_scale`, `steps`, `semantic_max_tokens`. Command-line flags override
the file. Omit `seed` to get a different song on every run.

## Development

```powershell
uv run pytest
```

## Licenses

yueno and audio.cpp are Apache 2.0. The YuE2 weights (and their GGUF conversions) are **CC BY-NC 4.0**:
non-commercial use only.

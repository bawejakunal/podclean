# 🎧 PodClean — Podcast Ad Remover

Automatically detect and remove spoken advertisements from podcast episodes using AI.

PodClean transcribes your podcast audio, uses an LLM (Google Gemini) to identify ad segments, and produces a clean audio file with smooth transitions — locally on **macOS** (Apple Silicon via MLX) or **Linux** (faster-whisper on CPU, or CUDA when available).

## How It Works

```
Audio File → Transcribe (Whisper) → Detect Ads (Gemini AI) → Remove & Export → Clean MP3 ✨
```

1. **Transcribe** — Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon or [faster-whisper](https://github.com/SYSTRAN/faster-whisper) elsewhere to generate a timestamped transcript with word-level precision
2. **Detect** — Sends the transcript to Google Gemini to identify sponsor reads, mid-rolls, self-promotions, and other ad segments
3. **Remove** — Cuts the ad segments from the audio using [pydub](https://github.com/jiaaro/pydub) with smooth crossfade transitions
4. **Export** — Outputs a clean MP3 file ready to listen to

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (or your distro equivalent)
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works great)

### Installation

Install PodClean as a standalone command so `podclean` lands on your `PATH` (typically `~/.local/bin`). The package gets its own isolated environment; you do not need a project venv to run it.

```bash
# from a clone of this repo
uv tool install .
```

For development against a checkout (the installed command tracks your local edits):

```bash
uv tool install -e .
```

Upgrade or remove that install later:

```bash
uv tool upgrade podclean
uv tool uninstall podclean
```

`pipx` is the same idea if you prefer it:

```bash
pipx install .
pipx upgrade podclean
pipx uninstall podclean
```

If `podclean` is not found after install, `~/.local/bin` is probably not on `PATH`. Fix that with:

```bash
uv tool update-shell
# then restart the shell, or: source ~/.bashrc / ~/.zshrc
```

The install pulls the Whisper stack that matches this machine. You do **not** need both backends.

| Platform | Package installed automatically |
|----------|----------------------------------|
| macOS Apple Silicon (`darwin` + `arm64`) | `mlx-whisper` |
| Linux, Intel Mac, and other non-MLX hosts | `faster-whisper` (>= 1.1, the first release that knows `large-v3-turbo`) |

To force a backend (in addition to, or instead of, the automatic extra):

```bash
# faster-whisper on any platform (Linux/CPU default; also useful on a Mac)
uv tool install ".[cpu]"

# mlx-whisper (Apple Silicon). Will not install usefully on Linux.
uv tool install ".[mlx]"
```

If you force the non-default extra, set `WHISPER_BACKEND` to match (see below).

### Configuration

PodClean reads process environment variables first, then layers `.env` files. The first file that defines a key wins; later files only fill keys that are still unset. Search order, highest priority first:

1. `PODCLEAN_ENV` — explicit path to a file, if set
2. `.env` in the current working directory
3. `$XDG_CONFIG_HOME/podclean/.env` (defaults to `~/.config/podclean/.env`)
4. `~/.podclean/.env`
5. A checkout-root `.env` when running from an editable/source install

A tool install does **not** keep reading `.env` from the git clone unless you run the command from that directory or copy the file to one of the paths above. After `uv tool install` / `pipx install`, put your API key in the user config file:

```bash
mkdir -p ~/.config/podclean
cp .env.example ~/.config/podclean/.env
# Edit ~/.config/podclean/.env and add your Gemini API key
```

A project-local `.env` in the directory you run `podclean` from still works and overrides matching keys in the user config file.

Leave `WHISPER_MODEL` unset unless you want to override the backend default. Model IDs are **not** interchangeable:

| Backend | Default model | Example `--model` / `WHISPER_MODEL` |
|---------|---------------|--------------------------------------|
| `mlx` | `mlx-community/whisper-large-v3-turbo` | HuggingFace repo id |
| `faster-whisper` | `large-v3-turbo` | Size/name (`tiny`, `small`, `large-v3`, `large-v3-turbo`) |

### Usage

#### Process a local audio file
```bash
podclean file episode.mp3
```

#### Process from an RSS feed (e.g., The Daily Stoic)
```bash
podclean feed "https://rss.art19.com/the-daily-stoic"
```

#### List episodes from a feed
```bash
podclean list "https://rss.art19.com/the-daily-stoic"
```

#### Process a specific episode (e.g., #3 from the feed)
```bash
podclean feed "https://rss.art19.com/the-daily-stoic" -n 3
```

#### Preview detected ads without processing
```bash
podclean file episode.mp3 --preview
```

#### Delete this run's local files after upload
`--cleanup` deletes only the files that the same `file` / `feed` run created:

- the downloaded episode audio (`feed`, unless it was already cached)
- the transcript JSON (unless a cached transcript was reused)
- the cleaned output audio, **only once it has been uploaded** with `--upload`; otherwise it is kept and PodClean says so

Your input file, earlier runs' output, and anything on S3 are never touched.

```bash
podclean feed "https://rss.art19.com/the-daily-stoic" --upload --cleanup
podclean file episode.mp3 --upload --cleanup
```

Cleanup also runs if the command fails partway. If a file cannot be deleted, PodClean prints the count and exits with status 1 (unless the command already failed, in which case its own error is reported).

#### Use a different Whisper model
```bash
# faster-whisper (Linux / CPU)
podclean file episode.mp3 --model small
podclean file episode.mp3 --model large-v3

# mlx-whisper (Apple Silicon)
podclean file episode.mp3 --model mlx-community/whisper-large-v3-turbo
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Whisper model id for the **active** backend (HF repo on MLX; size/name on faster-whisper) | backend default |
| `--output` | Output file path | `./output/<name>_clean.mp3` |
| `--preview` | Show detected ads without processing | off |
| `--api-key` | Gemini API key (or set in .env) | — |
| `--cleanup` | Delete files this run created: the download, the transcript, and the cleaned output once uploaded | off |

## Configuration (`.env`)

See [Configuration](#configuration) above for where these files are read from (`~/.config/podclean/.env` after a tool install, or `.env` in the working directory).

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *(required)* |
| `WHISPER_BACKEND` | `mlx` or `faster-whisper` | Darwin → `mlx`, else `faster-whisper` (or the installed backend) |
| `WHISPER_MODEL` | Whisper model id (format depends on backend) | `mlx-community/whisper-large-v3-turbo` or `large-v3-turbo` |
| `WHISPER_DEVICE` | Inference device | `gpu` (mlx) / `cpu` (faster-whisper; `cuda` when available) |
| `WHISPER_COMPUTE_TYPE` | faster-whisper compute type | `int8` |
| `OUTPUT_FORMAT` | Output format (mp3/wav/m4a) | `mp3` |
| `OUTPUT_BITRATE` | Output audio bitrate | `192k` |
| `CROSSFADE_MS` | Crossfade duration between cuts | `300` |

Runtime backend selection: `WHISPER_BACKEND` → whichever of `mlx-whisper` / `faster-whisper` is installed (platform default if both) → Darwin → mlx, otherwise faster-whisper. If the chosen backend is missing, the CLI errors with the exact `pip install` command to fix it.

Switching backends also switches the model and device defaults, so a machine configured for MLX can run faster-whisper (and vice versa) without hand-editing `WHISPER_MODEL`. Anything you set explicitly is always kept.

## Performance

Typical processing times for a 1-hour episode:

| Platform | Whisper Model | Time | Accuracy |
|----------|---------------|------|----------|
| Apple Silicon (MLX GPU) | `mlx-community/whisper-large-v3-turbo` | ~3–8 min | Best |
| Linux CPU (faster-whisper, `int8`) | `large-v3-turbo` | longer; hardware-dependent | Best |
| Either backend | `small` | faster | Good |

CUDA is optional. Linux works on CPU with the default `WHISPER_DEVICE=cpu` / `WHISPER_COMPUTE_TYPE=int8`.

## Development

For contributors who want a project virtualenv instead of `uv tool install -e .`:

```bash
cd ~/podclean
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Force a backend extra the same way: `pip install -e ".[cpu]"` or `pip install -e ".[mlx]"`. If you force the non-default extra, set `WHISPER_BACKEND` to match.

A local `.env` in the checkout root is picked up automatically (see Configuration above). Copy `.env.example` to `.env` and add your Gemini API key.

Run the test suite from the checkout:

```bash
python -m unittest discover -s tests
```

## Project Structure

```
podclean/
├── pyproject.toml          # Project config & dependencies
├── .env.example            # Template for ~/.config/podclean/.env or ./.env
├── .env                    # Local API key (not committed; optional)
├── podclean/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration management
│   ├── fetcher.py          # RSS feed parsing & audio download
│   ├── transcriber.py      # Whisper audio transcription
│   ├── whisper_backend.py  # mlx vs faster-whisper selection
│   ├── detector.py         # Gemini-based ad detection
│   ├── processor.py        # Audio cutting & reassembly
│   └── models.py           # Data models
└── output/                 # Cleaned audio files
```

## License

MIT

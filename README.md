# 🎧 PodClean — Podcast Ad Remover

Automatically detect and remove spoken advertisements from podcast episodes using AI.

PodClean transcribes your podcast audio, uses an LLM (Google Gemini) to identify ad segments, and produces a clean audio file with smooth transitions — all locally on your Mac.

## How It Works

```
Audio File → Transcribe (Whisper) → Detect Ads (Gemini AI) → Remove & Export → Clean MP3 ✨
```

1. **Transcribe** — Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) to generate a timestamped transcript with word-level precision
2. **Detect** — Sends the transcript to Google Gemini to identify sponsor reads, mid-rolls, self-promotions, and other ad segments
3. **Remove** — Cuts the ad segments from the audio using [pydub](https://github.com/jiaaro/pydub) with smooth crossfade transitions
4. **Export** — Outputs a clean MP3 file ready to listen to

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works great)

### Installation

```bash
cd ~/podclean
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

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

#### Use a different Whisper model
```bash
# Faster (less accurate)
podclean file episode.mp3 --model small

# More accurate (slower)
podclean file episode.mp3 --model large-v3
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Whisper model: tiny, base, small, medium, large-v3 | large-v3 |
| `--output` | Output file path | `./output/<name>_clean.mp3` |
| `--preview` | Show detected ads without processing | off |
| `--api-key` | Gemini API key (or set in .env) | — |

## Configuration (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *(required)* |
| `WHISPER_MODEL` | Whisper model size | `large-v3` |
| `OUTPUT_FORMAT` | Output format (mp3/wav/m4a) | `mp3` |
| `OUTPUT_BITRATE` | Output audio bitrate | `192k` |
| `CROSSFADE_MS` | Crossfade duration between cuts | `300` |

## Performance

Typical processing times on Apple Silicon Mac (M1/M2/M3):

| Whisper Model | 1-hour Episode | Accuracy |
|---------------|----------------|----------|
| `small` | ~3–5 min | Good |
| `medium` | ~8–12 min | Better |
| `large-v3` | ~15–25 min | Best |

## Project Structure

```
podclean/
├── pyproject.toml          # Project config & dependencies
├── .env                    # Your API key (not committed)
├── podclean/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration management
│   ├── fetcher.py          # RSS feed parsing & audio download
│   ├── transcriber.py      # Whisper audio transcription
│   ├── detector.py         # Gemini-based ad detection
│   ├── processor.py        # Audio cutting & reassembly
│   └── models.py           # Data models
└── output/                 # Cleaned audio files
```

## License

MIT

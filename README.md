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
| `RSS_MAX_ITEMS` | Newest episodes kept in the S3 RSS file (`0` = unlimited) | `300` |

## S3 RSS feed size

`--upload` writes cleaned audio plus a single `podclean/output/rss.xml` object to S3. Apple Podcasts (iTunes) and YouTube Music download that whole XML file on every refresh — there is no paging.

S3 will serve a large object without trouble. The constraint is the player: many aggregators time out or throttle around **512 KiB**, and Apple Podcasts only **displays the latest 2,000 episodes**. Timeouts are download/parse time, not S3 object size. Static S3 hosting is already the fast path (no PHP generation); S3 also supports the HTTP `HEAD` and byte-range requests Apple requires.

PodClean episode entries are small (no transcripts or long show notes), so **300 newest items stay well under 512 KiB**. Older episodes drop out of the feed; the MP3 objects remain in the bucket. Set `RSS_MAX_ITEMS=0` to publish the full catalog. Prefer a CloudFront distribution if you want gzip — do not upload a gzip-only `rss.xml` to S3, because S3 does not negotiate `Accept-Encoding`.

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

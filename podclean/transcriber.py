"""Audio transcription using mlx-whisper.

Provides word-level timestamps for precise ad boundary detection.
Optimised for Apple Silicon (M1/M2/M3/M4) via MLX framework.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from podclean.config import get_config
from podclean.models import TranscriptSegment, Word

console = Console()


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


class Transcriber:
    """Transcribe audio files using mlx-whisper with word-level timestamps.

    The transcriber leverages Apple's MLX framework for GPU-accelerated
    transcription on Apple Silicon and produces :class:`TranscriptSegment`
    objects that carry per-word timing data.

    Parameters
    ----------
    model_size:
        Whisper model HuggingFace repo path (e.g. 
        ``mlx-community/whisper-large-v3-turbo``). Falls back to the value in
        :func:`podclean.config.get_config` when *None*.
    """

    def __init__(self, model_size: str | None = None) -> None:
        config = get_config()
        self.model_size = model_size or config.whisper_model
        self.word_timestamps = config.word_timestamps

        console.print(
            f"  [green]✓[/green] MLX framework ready. Model will be loaded on demand: [bold]{self.model_size}[/bold]"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe an audio file and return word-level segments."""
        import mlx_whisper

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        console.print(
            f"\n[bold]Transcribing[/bold] [cyan]{audio_path.name}[/cyan]…"
        )
        start_time = time.perf_counter()

        try:
            with console.status("[bold blue]Running MLX-Whisper inference (GPU)…"):
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=self.model_size,
                    word_timestamps=self.word_timestamps,
                )
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for {audio_path.name}: {exc}"
            ) from exc

        language = result.get("language", "unknown")
        
        console.print(f"  [dim]Language:[/dim] {language}")

        segments: list[TranscriptSegment] = []
        raw_segments = result.get("segments", [])

        with console.status("[bold blue]Processing segments…"):
            for raw_seg in raw_segments:
                words = self._extract_words(raw_seg)
                segment = TranscriptSegment(
                    start=float(raw_seg["start"]),
                    end=float(raw_seg["end"]),
                    text=str(raw_seg.get("text", "")).strip(),
                    words=words,
                )
                segments.append(segment)

        elapsed = time.perf_counter() - start_time
        total_words = sum(len(seg.words) for seg in segments)
        
        console.print(
            f"  [green]✓[/green] Transcription complete — "
            f"[bold]{len(segments)}[/bold] segments, "
            f"[bold]{total_words:,}[/bold] words "
            f"in [bold]{elapsed:.1f}s[/bold]"
        )

        return segments

    def _extract_words(self, raw_segment: dict) -> list[Word]:
        raw_words = raw_segment.get("words", [])
        if not raw_words:
            return []

        return [
            Word(
                text=str(w.get("word", "")).strip(),
                start=float(w["start"]),
                end=float(w["end"]),
                probability=float(w.get("probability", 1.0)),
            )
            for w in raw_words
        ]


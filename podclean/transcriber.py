"""Audio transcription with word-level timestamps.

Selects mlx-whisper (Apple Silicon / macOS) or faster-whisper (Linux and
other non-MLX environments) so callers always receive
``list[TranscriptSegment]``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from rich.console import Console

from podclean.config import get_config
from podclean.models import TranscriptSegment, Word
from podclean.whisper_backend import (
    BackendName,
    missing_backend_error,
    resolve_whisper_backend,
)

try:
    import mlx_whisper
except ImportError:  # pragma: no cover - optional extra
    mlx_whisper = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional extra
    WhisperModel = None

console = Console()


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


class _WhisperEngine(Protocol):
    def transcribe(self, audio_path: Path) -> tuple[str, list[TranscriptSegment]]:
        """Return ``(language, segments)`` for *audio_path*."""


class _MlxEngine:
    """mlx-whisper implementation (Apple Silicon GPU)."""

    def __init__(self, model_size: str, word_timestamps: bool) -> None:
        if mlx_whisper is None:
            raise TranscriptionError(missing_backend_error("mlx"))
        self.model_size = model_size
        self.word_timestamps = word_timestamps

    def transcribe(self, audio_path: Path) -> tuple[str, list[TranscriptSegment]]:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_size,
            word_timestamps=self.word_timestamps,
        )
        language = str(result.get("language", "unknown"))
        segments: list[TranscriptSegment] = []
        for raw_seg in result.get("segments", []) or []:
            segments.append(
                TranscriptSegment(
                    start=float(raw_seg["start"]),
                    end=float(raw_seg["end"]),
                    text=str(raw_seg.get("text", "")).strip(),
                    words=_words_from_mapping(raw_seg.get("words", []) or []),
                )
            )
        return language, segments


class _FasterWhisperEngine:
    """faster-whisper implementation (CPU or CUDA)."""

    def __init__(
        self,
        model_size: str,
        word_timestamps: bool,
        device: str,
        compute_type: str,
        vad_filter: bool,
    ) -> None:
        if WhisperModel is None:
            raise TranscriptionError(missing_backend_error("faster-whisper"))
        self.model_size = model_size
        self.word_timestamps = word_timestamps
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> tuple[str, list[TranscriptSegment]]:
        model = self._get_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
        )
        language = str(getattr(info, "language", "unknown"))
        segments: list[TranscriptSegment] = []
        for raw_seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=float(raw_seg.start),
                    end=float(raw_seg.end),
                    text=str(raw_seg.text or "").strip(),
                    words=_words_from_objects(getattr(raw_seg, "words", None) or []),
                )
            )
        return language, segments


def _words_from_mapping(raw_words: list[dict]) -> list[Word]:
    return [
        Word(
            text=str(w.get("word", "")).strip(),
            start=float(w["start"]),
            end=float(w["end"]),
            probability=float(w.get("probability", 1.0)),
        )
        for w in raw_words
    ]


def _words_from_objects(raw_words: list[object]) -> list[Word]:
    result: list[Word] = []
    for w in raw_words:
        result.append(
            Word(
                text=str(getattr(w, "word", "")).strip(),
                start=float(w.start),
                end=float(w.end),
                probability=float(getattr(w, "probability", 1.0)),
            )
        )
    return result


class Transcriber:
    """Transcribe audio files with word-level timestamps.

    The public API is backend-agnostic: :meth:`transcribe` always returns
    :class:`TranscriptSegment` objects. The engine is chosen by
    :func:`resolve_whisper_backend` (env override → installed package →
    platform default).

    Parameters
    ----------
    model_size:
        Whisper model id. MLX expects a HuggingFace repo
        (e.g. ``mlx-community/whisper-large-v3-turbo``); faster-whisper
        expects a size/name (e.g. ``large-v3-turbo``). Falls back to
        :func:`podclean.config.get_config` when *None*.
    backend:
        Optional ``mlx`` or ``faster-whisper`` override. When omitted,
        uses the resolved config backend.
    """

    def __init__(
        self,
        model_size: str | None = None,
        backend: str | None = None,
    ) -> None:
        config = get_config()
        self.backend: BackendName = resolve_whisper_backend(
            backend or config.whisper_backend
        )
        settings = config.whisper_settings_for(self.backend)
        self.model_size = model_size or settings.model
        self.word_timestamps = config.word_timestamps
        self.device = settings.device
        self.compute_type = settings.compute_type
        self._engine = self._make_engine(config.vad_filter)

        console.print(
            f"  [green]✓[/green] {self._backend_label()} ready. "
            f"Model: [bold]{self.model_size}[/bold] "
            f"([dim]{self.device}/{self.compute_type}[/dim])"
        )
        self._warn_if_model_looks_mismatched()

    def _backend_label(self) -> str:
        return "mlx-whisper" if self.backend == "mlx" else "faster-whisper"

    def _make_engine(self, vad_filter: bool) -> _WhisperEngine:
        if self.backend == "mlx":
            return _MlxEngine(self.model_size, self.word_timestamps)
        return _FasterWhisperEngine(
            self.model_size,
            self.word_timestamps,
            self.device,
            self.compute_type,
            vad_filter,
        )

    def _warn_if_model_looks_mismatched(self) -> None:
        model = self.model_size
        if self.backend == "faster-whisper" and model.startswith("mlx-community/"):
            console.print(
                "  [yellow]Warning:[/yellow] "
                f"{model!r} looks like an MLX HuggingFace repo, but the "
                "active backend is faster-whisper. Use a size/name such as "
                "large-v3-turbo, or set WHISPER_BACKEND=mlx."
            )
        if self.backend == "mlx" and "/" not in model:
            console.print(
                "  [yellow]Warning:[/yellow] "
                f"{model!r} looks like a faster-whisper size name, but the "
                "active backend is mlx. Use a HuggingFace repo such as "
                "mlx-community/whisper-large-v3-turbo, or set "
                "WHISPER_BACKEND=faster-whisper."
            )

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe an audio file and return word-level segments."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        console.print(f"\n[bold]Transcribing[/bold] [cyan]{audio_path.name}[/cyan]…")
        start_time = time.perf_counter()

        status = (
            "Running MLX-Whisper inference (GPU)…"
            if self.backend == "mlx"
            else "Running faster-whisper inference…"
        )
        try:
            with console.status(f"[bold blue]{status}"):
                language, segments = self._engine.transcribe(audio_path)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for {audio_path.name}: {exc}"
            ) from exc

        console.print(f"  [dim]Language:[/dim] {language}")

        elapsed = time.perf_counter() - start_time
        total_words = sum(len(seg.words) for seg in segments)

        console.print(
            f"  [green]✓[/green] Transcription complete — "
            f"[bold]{len(segments)}[/bold] segments, "
            f"[bold]{total_words:,}[/bold] words "
            f"in [bold]{elapsed:.1f}s[/bold]"
        )

        return segments

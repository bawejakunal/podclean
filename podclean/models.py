"""Data models for PodClean pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    """A single word with its timestamp boundaries."""

    text: str
    start: float  # seconds
    end: float  # seconds
    probability: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Word:
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            probability=float(data.get("probability", 1.0)),
        )

    def __repr__(self) -> str:
        return f"Word({self.text!r}, {self.start:.2f}s–{self.end:.2f}s)"


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with word-level detail."""

    start: float  # seconds
    end: float  # seconds
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptSegment:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            words=[Word.from_dict(w) for w in data.get("words", [])],
        )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Segment({self.start:.1f}s–{self.end:.1f}s, {preview!r})"


@dataclass
class AdRegion:
    """A detected advertisement region in the audio."""

    start: float  # seconds
    end: float  # seconds
    confidence: float  # 0.0–1.0
    reason: str  # e.g. "Sponsor read for BetterHelp"
    transcript_excerpt: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: AdRegion) -> bool:
        """Check if this region overlaps with another."""
        return self.start < other.end and other.start < self.end

    def merge(self, other: AdRegion) -> AdRegion:
        """Merge two overlapping regions into one."""
        return AdRegion(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            confidence=max(self.confidence, other.confidence),
            reason=f"{self.reason}; {other.reason}",
            transcript_excerpt=self.transcript_excerpt or other.transcript_excerpt,
        )

    def __repr__(self) -> str:
        return (
            f"Ad({self.start:.1f}s–{self.end:.1f}s, "
            f"{self.duration:.0f}s, {self.confidence:.0%}, {self.reason!r})"
        )


@dataclass
class EpisodeInfo:
    """Metadata about a podcast episode."""

    title: str
    published: str = ""
    duration: str = ""
    audio_url: str = ""
    description: str = ""

    def __repr__(self) -> str:
        return f"Episode({self.title!r}, {self.published})"


@dataclass
class ProcessingResult:
    """Result of processing a podcast episode."""

    original_duration: float  # seconds
    cleaned_duration: float  # seconds
    ads_removed: int
    ad_regions: list[AdRegion]
    output_path: str
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def time_saved(self) -> float:
        return self.original_duration - self.cleaned_duration

    @property
    def time_saved_pct(self) -> float:
        if self.original_duration == 0:
            return 0.0
        return (self.time_saved / self.original_duration) * 100

    def summary(self) -> str:
        """Human-readable summary of processing results."""

        def fmt_time(seconds: float) -> str:
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h}h {m:02d}m {s:02d}s"
            return f"{m}m {s:02d}s"

        lines = [
            f"Original duration:  {fmt_time(self.original_duration)}",
            f"Cleaned duration:   {fmt_time(self.cleaned_duration)}",
            f"Time saved:         {fmt_time(self.time_saved)} ({self.time_saved_pct:.1f}%)",
            f"Ads removed:        {self.ads_removed}",
            f"Output:             {self.output_path}",
        ]
        return "\n".join(lines)

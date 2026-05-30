"""Audio processing — cut ads and produce clean podcast audio.

This module takes an original audio file and a list of detected ad regions,
removes the ad segments, and exports a clean audio file with smooth
crossfade transitions between the remaining content segments.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydub import AudioSegment
from rich.console import Console

from podclean.config import get_config
from podclean.models import AdRegion, ProcessingResult

console = Console()


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as a human-readable ``Xm YYs`` / ``Xh YYm ZZs`` string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _merge_ad_regions(
    regions: list[AdRegion],
    buffer_seconds: float = 0.0,
) -> list[AdRegion]:
    """Sort ad regions by start time and merge any that overlap or are adjacent.

    Parameters
    ----------
    regions:
        Raw detected ad regions (may overlap or be unsorted).
    buffer_seconds:
        Extra padding (in seconds) added around each ad region before the
        overlap check.  A buffer of 0.5 s means two regions that are within
        1 s of each other will be merged.

    Returns
    -------
    list[AdRegion]
        A sorted, non-overlapping list of merged ad regions.
    """
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r.start)
    merged: list[AdRegion] = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        prev = merged[-1]
        # Expand both edges by the buffer before checking overlap.
        if region.start - buffer_seconds <= prev.end + buffer_seconds:
            merged[-1] = prev.merge(region)
        else:
            merged.append(region)

    return merged


def _resolve_output_path(
    audio_path: Path,
    output_path: Path | None,
    output_format: str,
    output_dir: Path,
) -> Path:
    """Determine the output file path.

    Priority:
    1. Explicit *output_path* provided by the caller.
    2. ``<output_dir>/<stem>_clean.<output_format>``.
    """
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_name = f"{audio_path.stem}_clean.{output_format}"
    return output_dir / clean_name


class AudioProcessor:
    """Cut detected ad regions from an audio file and export clean audio.

    The processor loads the full audio into memory via *pydub*, slices out
    the non-ad segments, stitches them together with configurable crossfade
    durations, and writes the result to disk.

    Usage::

        processor = AudioProcessor()
        result = processor.process(Path("episode.mp3"), ad_regions)
        print(result.summary())
    """

    def __init__(self) -> None:
        self._cfg = get_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio_path: Path,
        ad_regions: list[AdRegion],
        output_path: Path | None = None,
    ) -> ProcessingResult:
        """Remove ads from *audio_path* and export a clean audio file.

        Parameters
        ----------
        audio_path:
            Path to the source audio file (any format *pydub* supports).
        ad_regions:
            Detected ad regions to remove.
        output_path:
            Optional explicit output path.  When ``None`` the file is written
            to ``<config.output_dir>/<stem>_clean.<format>``.

        Returns
        -------
        ProcessingResult
            Metadata about the processing run including durations and the
            output file location.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        """
        t_start = time.perf_counter()

        # -- Validate input ------------------------------------------------
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # -- Load audio ----------------------------------------------------
        console.print(f"[bold blue]Loading[/] {audio_path.name} …")
        audio = AudioSegment.from_file(str(audio_path))
        original_duration_s = len(audio) / 1000.0
        console.print(
            f"  Loaded {_fmt_time(original_duration_s)} of audio "
            f"({audio.frame_rate} Hz, {audio.channels} ch)"
        )

        # -- Fast path: no ads detected ------------------------------------
        if not ad_regions:
            console.print("[green]No ads to remove — exporting original audio.[/]")
            resolved = _resolve_output_path(
                audio_path,
                output_path,
                self._cfg.output_format,
                self._cfg.output_dir,
            )
            self._export(audio, resolved)
            return ProcessingResult(
                original_duration=original_duration_s,
                cleaned_duration=original_duration_s,
                ads_removed=0,
                ad_regions=[],
                output_path=str(resolved),
            )

        # -- Merge overlapping / adjacent regions --------------------------
        merged = _merge_ad_regions(ad_regions, self._cfg.buffer_seconds)
        console.print(
            f"[bold blue]Removing[/] {len(merged)} ad region(s) "
            f"({len(ad_regions)} raw → {len(merged)} merged)"
        )
        for i, r in enumerate(merged, 1):
            console.print(
                f"  {i}. {_fmt_time(r.start)}–{_fmt_time(r.end)} "
                f"({r.duration:.0f}s) — {r.reason}"
            )

        # -- Extract non-ad segments ---------------------------------------
        segments = self._extract_segments(audio, merged)

        if not segments:
            console.print(
                "[yellow]Warning:[/] All audio was classified as ads. "
                "Exporting silence."
            )
            segments = [AudioSegment.silent(duration=0)]

        # -- Stitch with crossfades ----------------------------------------
        clean_audio = self._stitch_segments(segments)
        cleaned_duration_s = len(clean_audio) / 1000.0

        # -- Export --------------------------------------------------------
        resolved = _resolve_output_path(
            audio_path,
            output_path,
            self._cfg.output_format,
            self._cfg.output_dir,
        )
        self._export(clean_audio, resolved)

        elapsed = time.perf_counter() - t_start
        console.print(
            f"[bold green]Done![/] Processed in {elapsed:.1f}s  ·  "
            f"saved {_fmt_time(original_duration_s - cleaned_duration_s)} "
            f"({(1 - cleaned_duration_s / original_duration_s) * 100:.1f}%)"
        )

        return ProcessingResult(
            original_duration=original_duration_s,
            cleaned_duration=cleaned_duration_s,
            ads_removed=len(merged),
            ad_regions=merged,
            output_path=str(resolved),
        )

    def preview(
        self,
        audio_path: Path,
        ad_regions: list[AdRegion],
    ) -> ProcessingResult:
        """Dry-run: compute what *process()* would produce **without** audio I/O.

        This is useful for showing the user what will happen before they
        commit to the (potentially slow) processing step.

        Parameters
        ----------
        audio_path:
            Path to the source audio file.
        ad_regions:
            Detected ad regions.

        Returns
        -------
        ProcessingResult
            A result whose ``output_path`` is set to the *planned* destination
            (the file is **not** created).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = AudioSegment.from_file(str(audio_path))
        original_duration_s = len(audio) / 1000.0

        if not ad_regions:
            resolved = _resolve_output_path(
                audio_path, None, self._cfg.output_format, self._cfg.output_dir
            )
            return ProcessingResult(
                original_duration=original_duration_s,
                cleaned_duration=original_duration_s,
                ads_removed=0,
                ad_regions=[],
                output_path=str(resolved),
            )

        merged = _merge_ad_regions(ad_regions, self._cfg.buffer_seconds)
        total_ad_duration = sum(r.duration for r in merged)
        cleaned_duration_s = max(original_duration_s - total_ad_duration, 0.0)

        resolved = _resolve_output_path(
            audio_path, None, self._cfg.output_format, self._cfg.output_dir
        )

        console.print("[bold cyan]Preview[/] (no audio will be written)")
        console.print(f"  Original duration : {_fmt_time(original_duration_s)}")
        console.print(f"  Ads to remove     : {len(merged)}")
        console.print(f"  Ad time           : {_fmt_time(total_ad_duration)}")
        console.print(f"  Cleaned duration  : ~{_fmt_time(cleaned_duration_s)}")
        console.print(f"  Output path       : {resolved}")

        for i, r in enumerate(merged, 1):
            console.print(
                f"  {i}. {_fmt_time(r.start)}–{_fmt_time(r.end)} "
                f"({r.duration:.0f}s) — {r.reason}"
            )

        return ProcessingResult(
            original_duration=original_duration_s,
            cleaned_duration=cleaned_duration_s,
            ads_removed=len(merged),
            ad_regions=merged,
            output_path=str(resolved),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_segments(
        self,
        audio: AudioSegment,
        merged_ads: list[AdRegion],
    ) -> list[AudioSegment]:
        """Return the non-ad slices of *audio*.

        Each returned segment has fade-in / fade-out applied at its edges so
        that the subsequent crossfade sounds natural even without perfectly
        aligned cut-points.
        """
        total_ms = len(audio)
        segments: list[AudioSegment] = []
        cursor_ms = 0  # playhead position in ms

        for ad in merged_ads:
            ad_start_ms = int(ad.start * 1000)
            ad_end_ms = int(ad.end * 1000)

            # Clamp to audio bounds
            ad_start_ms = max(0, min(ad_start_ms, total_ms))
            ad_end_ms = max(0, min(ad_end_ms, total_ms))

            # Content *before* this ad
            if cursor_ms < ad_start_ms:
                seg = audio[cursor_ms:ad_start_ms]
                seg = self._apply_fades(seg)
                segments.append(seg)

            cursor_ms = ad_end_ms

        # Content *after* the last ad
        if cursor_ms < total_ms:
            seg = audio[cursor_ms:total_ms]
            seg = self._apply_fades(seg)
            segments.append(seg)

        return segments

    def _apply_fades(self, segment: AudioSegment) -> AudioSegment:
        """Apply fade-in and fade-out if the segment is long enough."""
        duration_ms = len(segment)
        fade_in = min(self._cfg.fade_in_ms, duration_ms // 2)
        fade_out = min(self._cfg.fade_out_ms, duration_ms // 2)

        if fade_in > 0:
            segment = segment.fade_in(fade_in)
        if fade_out > 0:
            segment = segment.fade_out(fade_out)
        return segment

    def _stitch_segments(
        self,
        segments: list[AudioSegment],
    ) -> AudioSegment:
        """Join *segments* with crossfades, handling edge cases gracefully.

        If a segment is shorter than the crossfade duration the segments are
        simply concatenated without a crossfade to avoid pydub errors.
        """
        if not segments:
            return AudioSegment.silent(duration=0)

        result = segments[0]
        crossfade_ms = self._cfg.crossfade_ms

        for seg in segments[1:]:
            # pydub requires both sides to be at least as long as the crossfade
            min_len = min(len(result), len(seg))
            if min_len <= 0:
                # Skip truly empty segments
                continue
            if crossfade_ms > 0 and min_len > crossfade_ms:
                result = result.append(seg, crossfade=crossfade_ms)
            else:
                # Segment too short for crossfade — hard-join instead
                result = result + seg

        return result

    def _export(self, audio: AudioSegment, path: Path) -> None:
        """Export *audio* to *path* using configured format and bitrate."""
        fmt = self._cfg.output_format
        path.parent.mkdir(parents=True, exist_ok=True)

        export_kwargs: dict[str, str] = {"format": fmt}
        if fmt != "wav":
            export_kwargs["bitrate"] = self._cfg.output_bitrate

        console.print(
            f"[bold blue]Exporting[/] → {path} ({fmt}, {self._cfg.output_bitrate})"
        )
        audio.export(str(path), **export_kwargs)
        console.print(f"  Wrote {path.stat().st_size / (1024 * 1024):.1f} MB")

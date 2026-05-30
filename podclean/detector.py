"""LLM-based advertisement detection using Google Gemini.

Sends a timestamped podcast transcript to the Gemini API and parses
structured JSON output to identify ad segments.  Supports chunked
processing for long episodes, overlapping-region merging, an optional
verification pass, and configurable confidence filtering.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai.errors import APIError
from rich.console import Console

from podclean.config import get_config
from podclean.models import AdRegion, TranscriptSegment

console = Console()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DETECTION_SYSTEM_PROMPT = """\
You are an expert podcast ad detector.  Your job is to analyze a timestamped
podcast transcript and identify every advertisement, sponsorship, or
promotional segment.

Detect the following categories:
1. **Host-read sponsor messages** – phrases such as "brought to you by",
   "use code", "thanks to our sponsor", "special offer", "go to [url]",
   "promo code", "sponsor of today's episode".
2. **Pre-roll / post-roll ads** – ad reads at the very beginning or end
   of the episode.
3. **Mid-roll ad breaks** – ad reads inserted in the middle of the episode,
   often preceded/followed by transition phrases ("we'll be right back",
   "and now a word from", "let's get back to the show").
4. **Self-promotion** – Patreon plugs, merchandise pushes, promos for the
   host's other shows/books, newsletter sign-ups, or event tickets.
   Watch for "Daily Stoic store", "dailystoic.com/…", "Ryan Holiday",
   book promotions, and medallion/coin/journal promotions.
5. **Transition phrases** that bookend ads – include these in the ad region
   so cuts sound clean.

Rules:
- Return ONLY a JSON array of objects.  No commentary before or after.
- Each object must have these exact keys:
    "start"              – float, start time in seconds
    "end"                – float, end time in seconds
    "confidence"         – float 0.0–1.0, how confident you are this is an ad
    "reason"             – string, short description (e.g. "Sponsor read for BetterHelp")
    "transcript_excerpt" – string, a short verbatim excerpt (≤80 chars)
- If there are NO ads, return an empty array: []
- Be precise with timestamps – use the [MM:SS] markers in the transcript.
- Prefer slightly wider boundaries over cutting into real content.
"""

_VERIFICATION_SYSTEM_PROMPT = """\
You are reviewing ad detections made by another system on a podcast
transcript.  Your goals:

1. Confirm or reject each proposed ad region.
2. Identify any ads that were MISSED.
3. Adjust start/end timestamps if they are slightly off.

Return a JSON array of the final, corrected ad regions using the same
schema:
    "start", "end", "confidence", "reason", "transcript_excerpt"

If all original detections are correct and nothing was missed, return
them unchanged.  Return ONLY the JSON array.
"""


class AdDetector:
    """Detects advertisement segments in a podcast transcript via Gemini."""

    def __init__(self) -> None:
        config = get_config()
        if not config.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set.  "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.llm_model
        self._confidence_threshold = config.detection_confidence_threshold
        self._chunk_duration = config.chunk_duration_minutes * 60  # seconds
        self._chunk_overlap = config.chunk_overlap_minutes * 60  # seconds
        self._verification_pass = config.verification_pass
        self._buffer_seconds = config.buffer_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_ads(
        self, segments: list[TranscriptSegment]
    ) -> list[AdRegion]:
        """Detect ads in *segments* and return a sorted list of ad regions.

        Pipeline:
        1. Format transcript with ``[MM:SS]`` timestamps.
        2. Split into overlapping chunks for long episodes.
        3. Send each chunk to Gemini for detection.
        4. Parse structured JSON responses.
        5. Merge overlapping detections across chunks.
        6. (Optional) Run a verification pass.
        7. Apply confidence-threshold filter.
        8. Add buffer padding for cleaner cuts.
        """
        if not segments:
            return []

        total_duration = segments[-1].end
        console.print(
            f"[bold blue]Ad Detection[/] – analysing "
            f"{_fmt_time(total_duration)} of transcript "
            f"({len(segments)} segments)"
        )

        # 1 & 2 — format + chunk
        chunks = self._build_chunks(segments)
        console.print(
            f"  Split into [cyan]{len(chunks)}[/] chunk(s) for processing"
        )

        # 3 & 4 — detect per chunk
        all_regions: list[AdRegion] = []
        for idx, (chunk_text, chunk_offset) in enumerate(chunks, 1):
            with console.status(
                f"  [yellow]Chunk {idx}/{len(chunks)}[/] – querying Gemini…"
            ):
                regions = self._detect_chunk(chunk_text, chunk_offset)
            console.print(
                f"  Chunk {idx}: found [magenta]{len(regions)}[/] candidate region(s)"
            )
            all_regions.extend(regions)

        # 5 — merge overlapping
        merged = self._merge_regions(all_regions)
        console.print(
            f"  Merged to [cyan]{len(merged)}[/] distinct region(s)"
        )

        # 6 — optional verification
        if self._verification_pass and merged:
            with console.status("  [yellow]Running verification pass…[/]"):
                merged = self._verify(segments, merged)
            console.print(
                f"  Verified: [cyan]{len(merged)}[/] region(s) confirmed"
            )

        # 7 — confidence filter
        filtered = [
            r for r in merged if r.confidence >= self._confidence_threshold
        ]
        if len(filtered) < len(merged):
            console.print(
                f"  Filtered out [dim]{len(merged) - len(filtered)}[/] "
                f"low-confidence region(s) "
                f"(threshold={self._confidence_threshold:.0%})"
            )

        # 8 — add buffer padding
        padded = self._apply_buffer(filtered, total_duration)

        console.print(
            f"[bold green]✓[/] Detected [bold]{len(padded)}[/] ad region(s) "
            f"totalling [bold]{_fmt_time(sum(r.duration for r in padded))}[/]"
        )
        return padded

    # ------------------------------------------------------------------
    # Transcript formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transcript(segments: list[TranscriptSegment]) -> str:
        """Render segments as ``[MM:SS] text`` lines."""
        lines: list[str] = []
        for seg in segments:
            minutes, seconds = divmod(int(seg.start), 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {seg.text.strip()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _build_chunks(
        self, segments: list[TranscriptSegment]
    ) -> list[tuple[str, float]]:
        """Split *segments* into overlapping text chunks.

        Returns a list of ``(chunk_text, chunk_start_offset)`` tuples.
        *chunk_start_offset* is the timestamp (seconds) where this chunk
        begins so that parsed timestamps can be mapped back to episode
        time.
        """
        if not segments:
            return []

        total_duration = segments[-1].end
        if total_duration <= self._chunk_duration:
            return [(self._format_transcript(segments), 0.0)]

        chunks: list[tuple[str, float]] = []
        chunk_start = 0.0

        while chunk_start < total_duration:
            chunk_end = chunk_start + self._chunk_duration

            # Gather segments whose time range intersects this chunk
            chunk_segments = [
                s for s in segments if s.end > chunk_start and s.start < chunk_end
            ]
            if chunk_segments:
                # Format with timestamps relative to episode start (absolute)
                chunk_text = self._format_transcript(chunk_segments)
                chunks.append((chunk_text, chunk_start))

            # Advance by (chunk_duration - overlap) to create overlap
            chunk_start += self._chunk_duration - self._chunk_overlap

        return chunks

    # ------------------------------------------------------------------
    # Gemini interaction
    # ------------------------------------------------------------------

    def _detect_chunk(
        self, transcript_text: str, _offset: float
    ) -> list[AdRegion]:
        """Send a single chunk to Gemini and parse the response."""
        prompt = (
            f"{_DETECTION_SYSTEM_PROMPT}\n\n"
            f"--- TRANSCRIPT ---\n{transcript_text}\n--- END ---"
        )
        raw = self._call_gemini(prompt)
        return self._parse_regions(raw)

    def _verify(
        self,
        segments: list[TranscriptSegment],
        regions: list[AdRegion],
    ) -> list[AdRegion]:
        """Run a verification pass: ask Gemini to confirm / fix detections."""
        transcript_text = self._format_transcript(segments)
        detections_json = json.dumps(
            [
                {
                    "start": r.start,
                    "end": r.end,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "transcript_excerpt": r.transcript_excerpt,
                }
                for r in regions
            ],
            indent=2,
        )
        prompt = (
            f"{_VERIFICATION_SYSTEM_PROMPT}\n\n"
            f"--- TRANSCRIPT ---\n{transcript_text}\n--- END ---\n\n"
            f"--- PROPOSED DETECTIONS ---\n{detections_json}\n--- END ---"
        )
        raw = self._call_gemini(prompt)
        verified = self._parse_regions(raw)
        # Fall back to originals if verification returned nothing useful
        return verified if verified else regions

    def _call_gemini(self, prompt: str) -> str:
        """Make a single Gemini API call and return the text response with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                return response.text or ""
            except APIError as exc:
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    console.print(f"[yellow]Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{max_retries}...[/]")
                    time.sleep(wait_time)
                    continue
                console.print(f"[bold red]Gemini API error:[/] {exc}")
                raise
            except Exception as exc:
                console.print(f"[bold red]Gemini API error:[/] {exc}")
                raise
        return ""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_regions(raw_text: str) -> list[AdRegion]:
        """Extract a list of :class:`AdRegion` from Gemini's text output.

        Handles both raw JSON arrays and JSON wrapped in markdown
        code-fences (````json … ``` ``).
        """
        # Try to pull JSON from a fenced code block first
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL
        )
        json_text = fence_match.group(1) if fence_match else raw_text.strip()

        try:
            data: Any = json.loads(json_text)
        except json.JSONDecodeError:
            # Last resort: look for the first '[' … last ']'
            start_idx = json_text.find("[")
            end_idx = json_text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    data = json.loads(json_text[start_idx : end_idx + 1])
                except json.JSONDecodeError:
                    console.print(
                        "[yellow]Warning:[/] Could not parse Gemini response as JSON"
                    )
                    return []
            else:
                console.print(
                    "[yellow]Warning:[/] Could not parse Gemini response as JSON"
                )
                return []

        if not isinstance(data, list):
            data = [data]

        regions: list[AdRegion] = []
        for item in data:
            try:
                regions.append(
                    AdRegion(
                        start=float(item["start"]),
                        end=float(item["end"]),
                        confidence=float(item.get("confidence", 0.8)),
                        reason=str(item.get("reason", "Detected ad")),
                        transcript_excerpt=str(
                            item.get("transcript_excerpt", "")
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                console.print(
                    f"[yellow]Warning:[/] Skipping malformed detection: {exc}"
                )

        return regions

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_regions(regions: list[AdRegion]) -> list[AdRegion]:
        """Merge overlapping or adjacent ad regions.

        Sorts by start time, then combines any pair whose time ranges
        overlap.
        """
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda r: r.start)
        merged: list[AdRegion] = [sorted_regions[0]]

        for current in sorted_regions[1:]:
            if merged[-1].overlaps(current):
                merged[-1] = merged[-1].merge(current)
            else:
                merged.append(current)

        return merged

    def _apply_buffer(
        self, regions: list[AdRegion], episode_duration: float
    ) -> list[AdRegion]:
        """Add a small buffer before and after each region for cleaner cuts.

        Clamps to ``[0, episode_duration]`` so we never exceed the audio
        boundaries.
        """
        buffered: list[AdRegion] = []
        for r in regions:
            buffered.append(
                AdRegion(
                    start=max(0.0, r.start - self._buffer_seconds),
                    end=min(episode_duration, r.end + self._buffer_seconds),
                    confidence=r.confidence,
                    reason=r.reason,
                    transcript_excerpt=r.transcript_excerpt,
                )
            )
        # Re-merge in case buffer padding caused new overlaps
        return self._merge_regions(buffered)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``Xm YYs`` or ``Xh YYm ZZs``."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

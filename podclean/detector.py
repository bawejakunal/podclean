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
You are an expert podcast ad detector. Analyze a timestamped transcript and
identify every advertisement, sponsorship, or promotional segment.

A product or service ad read is a **contiguous commercial pitch**, usually
30–180 seconds (often 60–120 for host-reads). It is not a single brand-name
utterance. Conversational interview podcasts (The Knowledge Project,
Megaphone mid-rolls, and similar) use full host-reads; mark the entire pitch.

Typical host-read structure:
1. **Transition / bookend in** — "we'll be right back", "quick break",
   "thanks to our partners", "this episode is brought to you by…",
   "support for the show comes from…", "before we continue…",
   "and now a word from…".
2. **Sponsor intro** — brand name plus category (mattress, CRM, AI tool,
   vacuum, electrolyte drink, clothing, therapy app, etc.).
3. **Problem framing** — "if you're tired of…", "most teams struggle with…",
   "I used to waste hours on…".
4. **Product pitch** — features, benefits, who it is for, how the host uses
   it, social proof.
5. **Offer / CTA** — discount, free trial, "go to …", "visit … slash …",
   spelled-out URLs ("heygen dot com slash knowledge"), promo codes
   ("use code KNOWLEDGE"), "link in the show notes".
6. **Transition / bookend out** — "okay, back to the conversation",
   "now back to my chat with…", "thanks again to …", "let's get back to it".

Mark from the first transition or sponsor intro through the CTA and the
return-to-show line. Include bookend transitions. Prefer slightly wider
boundaries over cutting mid-sentence into real interview content.

Transcript lines look like `[MM:SS] spoken words…`. An ad is a **run of
consecutive lines** that stay in commercial register, not the one line
where the brand first appears. Example shape (illustrative):

```
[12:04] we'll take a quick break and be right back
[12:08] this episode is brought to you by Acme Widgets
[12:12] if you've been looking for a simpler way to keep your home clean
[12:20] Acme's new robot vacuum maps your floor and empties itself
[12:35] I've been using it for months and it's honestly been a game changer
[12:48] listeners get twenty percent off at acme dot com slash knowledge
[12:55] use code KNOWLEDGE for that deal
[13:02] okay, thanks Acme — now back to my conversation with Tobi
[13:08] so you were saying about decision making…
```

Correct region: start ≈ 724 (12:04), end ≈ 785 (13:05) — the whole break.
Wrong: start=728, end=730 because "Acme" appeared once, or start=12, end=13
because you copied the minute numbers instead of converting to seconds.

Multiple sponsors back-to-back in one break: emit **one region per sponsor
block**, or one merged region covering the whole break if they share the
same bookends. Each region must last the full pitch. Never return a
1–2 second stub whose `reason` lists every brand.

Placement:
- Pre-roll: often right after the cold open / theme, before the interview.
- Mid-roll: inserted mid-episode with bookend transitions.
- Post-roll: after wrap-up, before credits.

Categories to detect:
1. **Host-read sponsor messages** — full product/service pitches.
2. **Pre-roll / post-roll ads**.
3. **Mid-roll ad breaks**.
4. **Self-promotion** — newsletter, Patreon, merch store, host's
   book/course/tour, "my new book", ticket links. Include the full plug,
   not just the title mentioned inside interview discussion. Watch for
   "Daily Stoic store", "dailystoic.com/…", "Ryan Holiday", book
   promotions, and medallion/coin/journal promotions.
5. **Transition phrases** that bookend ads — include them in the region.

Lexical / transcript cues (Whisper output is messy — still count these):
- Sponsor framing: brought to you by, sponsored by, partner, today's
  sponsor, thanks to, support comes from, ad break, commercial break.
- CTA / offer: promo code, discount code, use code, percent off, free
  trial, free month, limited time, exclusive offer for listeners.
- URL / destination: go to, visit, head to, check out, dot com,
  slash [show name], .com/, http-ish phrases, "link in the
  description/show notes".
- Brand + category in commercial tone (not a casual namedrop): mattress,
  vacuum, avatar, CRM, VPN, credit card, meal kit, skincare, etc. paired
  with pitch language.

What is NOT an ad:
- Guest mentioning a company as part of the interview ("when I was at
  Shopify…").
- Host naming a product once without pitch or CTA.
- Brief thanks without a commercial block.
- Discussion of advertising or business models as an interview topic.

Timestamps:
- Transcript labels are `[MM:SS]` (or `[H:MM:SS]`). JSON `start` and `end`
  MUST be floats in **total seconds from episode start**.
- [12:04] → 724.0; [00:17] → 17.0; [1:02:08] → 3728.0.
- [17:42] is 1062 seconds, NOT 17. Do not copy the minute field as seconds.

Hard rules:
- Host-read product/service ads are almost never 1–5 seconds. If you name
  a sponsor and confidence is high, the region should usually be **at
  least ~20–30s** and typically **45–120s+** unless the transcript clearly
  ends the pitch sooner.
- Do not collapse four sponsors into a 2-second span with a combined
  reason — split or expand to the true boundaries.
- Include bookend transitions in the region.
- Prefer slightly wider boundaries over cutting into real content.

Output:
- Return ONLY a JSON array of objects. No commentary before or after.
- Each object must have these exact keys:
    "start"              – float, start time in total seconds
    "end"                – float, end time in total seconds
    "confidence"         – float 0.0–1.0, how confident you are this is an ad
    "reason"             – string, short description (e.g. "Sponsor read for BetterHelp")
    "transcript_excerpt" – string, a short verbatim excerpt (≤80 chars)
- If there are NO ads, return an empty array: []
"""

_VERIFICATION_SYSTEM_PROMPT = """\
You are reviewing ad detections made by another system on a podcast
transcript. Proposed detections use the same JSON schema and **total
seconds** from episode start.

Goals:
1. Confirm or reject each proposed ad region.
2. **Expand stub regions** that clearly name a sponsor but are far too
   short (about 1–5s, or well under ~20–30s). Grow them to the full
   contiguous pitch: bookend in → intro → problem → features → CTA →
   bookend out. Typical host-reads are 45–120s+ (often 60–120).
3. **Recover missed full reads**, especially conversational interview-
   podcast host-reads (Knowledge Project / Megaphone style) that look like
   a long commercial block rather than a short cue phrase.
4. **Reject collapsing multiple distinct mid-rolls into one tiny window.**
   If several brands are jammed into a 1–5s span (or a combined `reason`
   lists Apple Oven, Matic Vacuum, HeyGen, Element, etc.), split them
   into full-length regions — or one merged region covering the whole
   break if they share the same bookends. Never keep a 2-second stub.
5. Adjust start/end if they are slightly off. Include bookend transitions.
   Prefer slightly wider boundaries over cutting mid-sentence.

A 2-second region whose reason lists several sponsors is almost always
wrong. Look at the surrounding `[MM:SS]` lines, keep commercial-register
lines, and convert those labels to total seconds.

Timestamps:
- Transcript labels are `[MM:SS]`. JSON `start`/`end` are total seconds.
- [17:42] → 1062, not 17. Do not rewrite a correct ~1000s region into
  17–19 just because the transcript shows [17:…].
- Host-read ads are almost never 1–5 seconds when confidence is high.

What is NOT an ad: interview namedrops ("when I was at Shopify…"), a
one-off product mention without pitch/CTA, brief thanks without a
commercial block, or discussion of advertising as a topic.

Return a JSON array of the final, corrected ad regions using the same
schema:
    "start", "end", "confidence", "reason", "transcript_excerpt"

If all original detections are already full-length, correctly converted
to seconds, and nothing was missed, return them unchanged. Return ONLY
the JSON array.
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
        code-fences (````json … ``` ``). ``start`` / ``end`` may be numeric
        seconds or ``MM:SS`` / ``H:MM:SS`` transcript labels.
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
                        start=timestamp_label_to_seconds(item["start"]),
                        end=timestamp_label_to_seconds(item["end"]),
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


def timestamp_label_to_seconds(value: float | int | str) -> float:
    """Normalize a Gemini start/end value to episode seconds.

    Accepts numeric seconds (``724``, ``724.0``, ``"724"``) and transcript
    labels such as ``12:04``, ``[12:04]``, or ``1:02:08``.
    """
    if isinstance(value, bool):
        raise TypeError("boolean is not a timestamp")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().strip("[]")
    if not text:
        raise ValueError("empty timestamp")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unrecognized timestamp: {value!r}")


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``Xm YYs`` or ``Xh YYm ZZs``."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

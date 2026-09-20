"""Tests for Gemini ad-detection prompts and timestamp helpers."""

from __future__ import annotations

from unittest import TestCase

from podclean.detector import (
    AdDetector,
    _DETECTION_SYSTEM_PROMPT,
    _VERIFICATION_SYSTEM_PROMPT,
    timestamp_label_to_seconds,
)
from podclean.models import TranscriptSegment


class TimestampLabelToSecondsTest(TestCase):
    def test_numeric_seconds_pass_through(self) -> None:
        self.assertEqual(timestamp_label_to_seconds(724), 724.0)
        self.assertEqual(timestamp_label_to_seconds(724.5), 724.5)
        self.assertEqual(timestamp_label_to_seconds("1062"), 1062.0)

    def test_mm_ss_labels_convert_to_total_seconds(self) -> None:
        self.assertEqual(timestamp_label_to_seconds("12:04"), 724.0)
        self.assertEqual(timestamp_label_to_seconds("[17:42]"), 1062.0)
        self.assertEqual(timestamp_label_to_seconds("00:17"), 17.0)
        self.assertEqual(timestamp_label_to_seconds("1:02:08"), 3728.0)

    def test_rejects_unrecognized_values(self) -> None:
        with self.assertRaises(ValueError):
            timestamp_label_to_seconds("not-a-time")
        with self.assertRaises(ValueError):
            timestamp_label_to_seconds("")
        with self.assertRaises(TypeError):
            timestamp_label_to_seconds(True)


class ParseRegionsTest(TestCase):
    def test_parses_numeric_and_mmss_start_end(self) -> None:
        raw = """
        [
          {
            "start": 724,
            "end": 785,
            "confidence": 0.95,
            "reason": "Host-read for Acme Widgets",
            "transcript_excerpt": "brought to you by Acme"
          },
          {
            "start": "17:42",
            "end": "[19:05]",
            "confidence": 0.9,
            "reason": "Mid-roll for HeyGen",
            "transcript_excerpt": "heygen dot com slash knowledge"
          }
        ]
        """
        regions = AdDetector._parse_regions(raw)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].start, 724.0)
        self.assertEqual(regions[0].end, 785.0)
        self.assertEqual(regions[1].start, 1062.0)
        self.assertEqual(regions[1].end, 1145.0)


class FormatTranscriptTest(TestCase):
    def test_lines_use_mmss_labels(self) -> None:
        segments = [
            TranscriptSegment(start=724.0, end=728.0, text="brought to you by Acme"),
            TranscriptSegment(start=3728.0, end=3732.0, text="use code KNOWLEDGE"),
        ]
        text = AdDetector._format_transcript(segments)
        self.assertEqual(
            text,
            "[12:04] brought to you by Acme\n[62:08] use code KNOWLEDGE",
        )


def _collapsed(text: str) -> str:
    return " ".join(text.split())


class DetectionPromptGuidanceTest(TestCase):
    def test_teaches_full_host_read_span_not_brand_stubs(self) -> None:
        prompt = _collapsed(_DETECTION_SYSTEM_PROMPT)
        required = (
            "contiguous commercial pitch",
            "30–180 seconds",
            "we'll be right back",
            "brought to you by",
            "use code",
            "dot com",
            "back to the conversation",
            "when I was at",
            "Shopify",
            "at least ~20–30s",
            "45–120s+",
            "Do not collapse four sponsors",
            "total seconds",
            "[MM:SS]",
            "Acme Widgets",
            "Daily Stoic store",
            "transcript_excerpt",
            "Return ONLY a JSON array",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)

    def test_example_region_uses_seconds_not_minute_copy(self) -> None:
        prompt = _collapsed(_DETECTION_SYSTEM_PROMPT)
        self.assertIn("start ≈ 724 (12:04)", prompt)
        self.assertIn("end ≈ 785 (13:05)", prompt)
        self.assertIn("[17:42] is 1062 seconds, NOT 17", prompt)


class VerificationPromptGuidanceTest(TestCase):
    def test_expands_stubs_recovers_misses_and_rejects_collapsed_midrolls(self) -> None:
        prompt = _collapsed(_VERIFICATION_SYSTEM_PROMPT)
        required = (
            "Expand stub regions",
            "Recover missed full reads",
            "Reject collapsing multiple distinct mid-rolls into one tiny window",
            "Knowledge Project",
            "total seconds",
            "[17:42] → 1062, not 17",
            "Apple Oven",
            "2-second stub",
            "transcript_excerpt",
            "Return ONLY",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)

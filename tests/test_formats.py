"""Guard against per-module copies of the supported audio suffix list."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import TestCase

from podclean import cleanup, fetcher, rss
from podclean.config import Config
from podclean.formats import (
    AUDIO_MIME_TYPES,
    OUTPUT_FORMATS,
    SUPPORTED_AUDIO_EXTENSIONS,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "podclean"


class UnifiedAudioFormatsTest(TestCase):
    def test_output_formats_are_supported_audio(self) -> None:
        for fmt in OUTPUT_FORMATS:
            self.assertIn(f".{fmt}", SUPPORTED_AUDIO_EXTENSIONS)

    def test_every_supported_suffix_has_a_mime_type(self) -> None:
        self.assertEqual(set(AUDIO_MIME_TYPES), set(SUPPORTED_AUDIO_EXTENSIONS))
        for suffix in SUPPORTED_AUDIO_EXTENSIONS:
            self.assertEqual(rss.audio_mime_type(f"ep{suffix}"), AUDIO_MIME_TYPES[suffix])

    def test_modules_share_the_same_suffix_set(self) -> None:
        self.assertIs(fetcher.SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS)
        self.assertIs(cleanup.SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS)
        self.assertIs(rss.AUDIO_MIME_TYPES, AUDIO_MIME_TYPES)

    def test_config_rejects_unlisted_output_format(self) -> None:
        cfg = Config(gemini_api_key="k")
        cfg.output_format = "ogg"
        self.assertTrue(any("Choose from: mp3, wav, m4a" in e for e in cfg.validate()))

    def test_no_other_module_hardcodes_a_suffix_collection(self) -> None:
        suffixes = {s.lstrip(".") for s in SUPPORTED_AUDIO_EXTENSIONS}
        offenders: list[str] = []
        for path in sorted(_PACKAGE_DIR.glob("*.py")):
            if path.name == "formats.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
                    continue
                elts = node.keys if isinstance(node, ast.Dict) else node.elts
                literals = {
                    e.value.lstrip(".").lower()
                    for e in elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                if len(literals & suffixes) >= 2:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "Import audio suffixes from podclean.formats")

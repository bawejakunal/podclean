"""Audio formats PodClean reads, writes, uploads, and cleans up.

This is the single source of truth for audio file suffixes; other modules
import from here instead of keeping their own lists.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

AUDIO_MIME_TYPES: Final = MappingProxyType(
    {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }
)
"""Supported audio suffixes mapped to their podcast enclosure MIME type."""

SUPPORTED_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset(AUDIO_MIME_TYPES)
"""Suffixes (with leading dot) that PodClean accepts as input audio."""

OUTPUT_FORMATS: Final[tuple[str, ...]] = ("mp3", "wav", "m4a")
"""Formats the exporter can write (``OUTPUT_FORMAT``); a subset of the above."""

DEFAULT_AUDIO_MIME_TYPE: Final = AUDIO_MIME_TYPES[".mp3"]

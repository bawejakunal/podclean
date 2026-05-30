"""RSS feed parser and audio downloader for PodClean.

This module provides :class:`PodcastFetcher`, the entry-point for acquiring
podcast audio.  It can:

* Parse an RSS feed and list available episodes.
* Download an episode's audio with a ``rich`` progress bar.
* Accept a local audio file path as a pass-through.
* Cache downloads so the same episode is never fetched twice.

Example
-------
>>> from podclean.fetcher import PodcastFetcher
>>> fetcher = PodcastFetcher()
>>> episodes = fetcher.list_episodes("https://rss.art19.com/the-daily-stoic")
>>> path = fetcher.download_episode(episodes[0])
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import feedparser  # type: ignore[import-untyped]
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from podclean.config import get_config
from podclean.models import EpisodeInfo

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".m4a", ".wav", ".ogg"})
"""File extensions that PodClean knows how to process."""

_AUDIO_MIME_PREFIXES: tuple[str, ...] = ("audio/",)

_DOWNLOAD_CHUNK_SIZE: int = 64 * 1024  # 64 KiB
"""Chunk size used when streaming downloads."""

_DEFAULT_USER_AGENT: str = "PodClean/0.1 (https://github.com/podclean)"

_REQUEST_TIMEOUT: int = 30  # seconds
"""Timeout for HTTP requests (connect + read)."""


# ── Exceptions ─────────────────────────────────────────────────────────────────

class FetcherError(Exception):
    """Base exception for fetcher-related errors."""


class FeedParseError(FetcherError):
    """Raised when an RSS feed cannot be parsed or is empty."""


class DownloadError(FetcherError):
    """Raised when an audio file fails to download."""


class UnsupportedFormatError(FetcherError):
    """Raised when a file's format is not in :data:`SUPPORTED_AUDIO_EXTENSIONS`."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    """Return a short, filesystem-safe hash of *url*."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extension_for_url(url: str) -> str:
    """Best-effort extraction of the audio extension from a URL.

    Falls back to ``".mp3"`` when the extension cannot be determined.
    """
    path = urlparse(url).path
    # Strip query params that sometimes cling to the path component
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_AUDIO_EXTENSIONS:
        return ext
    return ".mp3"


def _extension_for_content_type(content_type: str | None) -> str:
    """Derive the file extension from an HTTP ``Content-Type`` header."""
    if not content_type:
        return ".mp3"
    mime = content_type.split(";")[0].strip()
    ext = mimetypes.guess_extension(mime)
    if ext and ext in SUPPORTED_AUDIO_EXTENSIONS:
        return ext
    return ".mp3"


def _parse_duration(raw: str | int | None) -> str:
    """Normalise a duration value into a human-friendly ``HH:MM:SS`` string.

    Handles both raw seconds (int / numeric string) and ``HH:MM:SS`` /
    ``MM:SS`` formatted strings that podcast feeds commonly provide.
    """
    if raw is None:
        return ""
    if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
        total = int(raw)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    return str(raw)


def _best_audio_link(entry: feedparser.FeedParserDict) -> str | None:
    """Extract the best audio enclosure URL from a feed entry.

    Prefers enclosures that have an ``audio/*`` MIME type; falls back to the
    first enclosure with a recognized audio extension.
    """
    enclosures: list[dict[str, str]] = getattr(entry, "enclosures", [])
    # First pass – explicit audio MIME type
    for enc in enclosures:
        mime = enc.get("type", "")
        if any(mime.startswith(p) for p in _AUDIO_MIME_PREFIXES):
            return enc.get("href") or enc.get("url")

    # Second pass – recognised audio extension
    for enc in enclosures:
        href = enc.get("href") or enc.get("url", "")
        ext = Path(urlparse(href).path).suffix.lower()
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            return href

    # Some feeds put the link directly on the entry
    for link in getattr(entry, "links", []):
        if link.get("rel") == "enclosure":
            return link.get("href")

    return None


# ── PodcastFetcher ─────────────────────────────────────────────────────────────

class PodcastFetcher:
    """Fetch podcast episodes from RSS feeds or local files.

    Parameters
    ----------
    session:
        An optional :class:`requests.Session` to reuse across calls.
        A default session with a ``User-Agent`` header is created if *None*.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or self._default_session()
        self._config = get_config()

    # ── Public API ─────────────────────────────────────────────────────────

    def list_episodes(
        self,
        rss_url: str,
        limit: int = 20,
    ) -> list[EpisodeInfo]:
        """Parse an RSS feed and return episode metadata.

        Parameters
        ----------
        rss_url:
            URL of the podcast RSS feed.
        limit:
            Maximum number of episodes to return (most-recent first).

        Raises
        ------
        FeedParseError
            If the feed cannot be fetched or contains no entries.
        """
        logger.info("Parsing RSS feed: %s", rss_url)
        feed = feedparser.parse(rss_url)

        if feed.bozo and not feed.entries:
            exc = feed.get("bozo_exception")
            raise FeedParseError(
                f"Failed to parse RSS feed at {rss_url}: {exc}"
            )

        if not feed.entries:
            raise FeedParseError(f"RSS feed at {rss_url} contains no episodes.")

        episodes: list[EpisodeInfo] = []
        for entry in feed.entries[:limit]:
            audio_url = _best_audio_link(entry) or ""
            if not audio_url:
                logger.debug("Skipping entry without audio: %s", entry.get("title"))
                continue

            published = getattr(entry, "published", "")
            duration_raw = entry.get("itunes_duration") or entry.get("duration")

            episodes.append(
                EpisodeInfo(
                    title=entry.get("title", "Untitled Episode"),
                    published=published,
                    duration=_parse_duration(duration_raw),
                    audio_url=audio_url,
                    description=entry.get("summary", ""),
                )
            )

        logger.info("Found %d episodes in feed.", len(episodes))
        return episodes

    def download_episode(
        self,
        episode: EpisodeInfo,
        output_dir: Path | None = None,
    ) -> Path:
        """Download the audio for *episode*, returning the local file path.

        If the file has already been downloaded (cache hit), the cached path is
        returned immediately.

        Parameters
        ----------
        episode:
            Episode metadata that includes a non-empty ``audio_url``.
        output_dir:
            Directory to save the file.  Defaults to ``config.cache_dir``.

        Raises
        ------
        DownloadError
            If the download fails for any reason (network, HTTP status, I/O).
        ValueError
            If ``episode.audio_url`` is empty.
        """
        if not episode.audio_url:
            raise ValueError("Episode has no audio URL – cannot download.")

        dest_dir = output_dir or self._config.cache_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename from URL hash + extension
        ext = _extension_for_url(episode.audio_url)
        filename = f"{_url_hash(episode.audio_url)}{ext}"
        dest = dest_dir / filename

        if dest.exists():
            logger.info("Cache hit: %s → %s", episode.title, dest)
            return dest

        logger.info("Downloading: %s", episode.title)
        try:
            response = self._session.get(
                episode.audio_url,
                stream=True,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(
                f"Failed to download {episode.audio_url!r}: {exc}"
            ) from exc

        # Refine extension using Content-Type if the URL was ambiguous
        if ext == ".mp3":
            ct_ext = _extension_for_content_type(
                response.headers.get("Content-Type")
            )
            if ct_ext != ext:
                filename = f"{_url_hash(episode.audio_url)}{ct_ext}"
                dest = dest_dir / filename
                if dest.exists():
                    logger.info("Cache hit (content-type): %s → %s", episode.title, dest)
                    return dest

        total_size = int(response.headers.get("Content-Length", 0))

        try:
            with self._progress_bar() as progress:
                label = _truncate(episode.title, 40)
                task = progress.add_task(label, total=total_size or None)

                with dest.open("wb") as fp:
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        fp.write(chunk)
                        progress.advance(task, len(chunk))
        except OSError as exc:
            # Clean up partial download
            dest.unlink(missing_ok=True)
            raise DownloadError(
                f"I/O error while writing {dest}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            dest.unlink(missing_ok=True)
            raise DownloadError(
                f"Network error while downloading {episode.audio_url!r}: {exc}"
            ) from exc

        logger.info("Saved: %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def get_audio_path(self, source: str) -> Path:
        """Resolve *source* to a local audio file path.

        *source* can be:

        * A **local file path** – validated and returned directly.
        * An **RSS feed URL** – the most recent episode is downloaded and
          its path returned.

        Parameters
        ----------
        source:
            A local path to an audio file, or an RSS feed URL.

        Raises
        ------
        FileNotFoundError
            If *source* looks like a local path but does not exist.
        UnsupportedFormatError
            If the file extension is not in :data:`SUPPORTED_AUDIO_EXTENSIONS`.
        FeedParseError
            If *source* is a URL but the feed cannot be parsed.
        DownloadError
            If the episode download fails.
        """
        # ── Local file ────────────────────────────────────────────────────
        local_path = Path(source)
        if local_path.exists():
            self._validate_audio_extension(local_path)
            logger.info("Using local file: %s", local_path)
            return local_path.resolve()

        # ── URL ───────────────────────────────────────────────────────────
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            episodes = self.list_episodes(source, limit=1)
            if not episodes:
                raise FeedParseError(
                    f"No episodes with audio found in feed: {source}"
                )
            return self.download_episode(episodes[0])

        # ── Nothing matched ───────────────────────────────────────────────
        raise FileNotFoundError(
            f"Source not found: {source!r}.  "
            "Provide a valid local file path or an RSS feed URL."
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _default_session() -> requests.Session:
        """Create a :class:`requests.Session` with default headers."""
        session = requests.Session()
        session.headers.update({"User-Agent": _DEFAULT_USER_AGENT})
        return session

    @staticmethod
    def _validate_audio_extension(path: Path) -> None:
        """Raise :class:`UnsupportedFormatError` if *path* is not a known audio type."""
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported audio format {path.suffix!r}.  "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

    @staticmethod
    def _progress_bar() -> Progress:
        """Create a ``rich`` download progress bar."""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters, adding ``…`` if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

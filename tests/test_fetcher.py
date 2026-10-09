"""Tests for the fetcher's download-cache lookup."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from podclean.fetcher import PodcastFetcher, _url_hash
from podclean.models import EpisodeInfo

_EPISODE = EpisodeInfo(title="Ep", audio_url="https://cdn.example.com/ep.mp3")


class FindCachedTest(TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self.session = MagicMock()
        cfg = SimpleNamespace(cache_dir=self.cache)
        with patch("podclean.fetcher.get_config", return_value=cfg):
            self.fetcher = PodcastFetcher(session=self.session)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_returns_none_when_not_downloaded(self) -> None:
        self.assertIsNone(self.fetcher.find_cached(_EPISODE))

    def test_finds_file_saved_under_content_type_extension(self) -> None:
        cached = self.cache / f"{_url_hash(_EPISODE.audio_url)}.m4a"
        cached.write_bytes(b"audio")

        self.assertEqual(self.fetcher.find_cached(_EPISODE), cached)

    def test_download_episode_reuses_cache_without_network(self) -> None:
        cached = self.cache / f"{_url_hash(_EPISODE.audio_url)}.m4a"
        cached.write_bytes(b"audio")

        self.assertEqual(self.fetcher.download_episode(_EPISODE), cached)
        self.session.get.assert_not_called()

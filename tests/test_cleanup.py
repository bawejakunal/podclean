"""Tests for per-run cleanup (``--cleanup`` on ``file`` / ``feed``)."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from click.testing import CliRunner

from podclean import config as config_mod
from podclean.cleanup import CleanupSummary, RunArtifacts, format_bytes, remove_files
from podclean.cli import cli
from podclean.config import Config
from podclean.models import AdRegion, EpisodeInfo, ProcessingResult, TranscriptSegment

_WHISPER_ENV = (
    "WHISPER_BACKEND",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "GEMINI_API_KEY",
)

_AD = AdRegion(start=1.0, end=2.0, confidence=0.9, reason="Sponsor read")


def _clear_whisper_env() -> dict[str, str]:
    saved = {key: os.environ[key] for key in _WHISPER_ENV if key in os.environ}
    for key in _WHISPER_ENV:
        os.environ.pop(key, None)
    return saved


def _restore_env(saved: dict[str, str]) -> None:
    for key in _WHISPER_ENV:
        os.environ.pop(key, None)
    os.environ.update(saved)


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


class FormatBytesTest(TestCase):
    def test_units(self) -> None:
        self.assertEqual(format_bytes(0), "0 bytes")
        self.assertEqual(format_bytes(512), "512 bytes")
        self.assertEqual(format_bytes(2048), "2.0 KB")
        self.assertEqual(format_bytes(3 * 1024 * 1024), "3.0 MB")


class RunArtifactsTest(TestCase):
    def test_output_removable_only_once_uploaded(self) -> None:
        artifacts = RunArtifacts(scratch=[Path("t.json")], output=Path("o.mp3"))
        self.assertEqual(artifacts.removable(), [Path("t.json")])
        self.assertEqual(artifacts.kept_output, Path("o.mp3"))

        artifacts.uploaded = True
        self.assertEqual(artifacts.removable(), [Path("t.json"), Path("o.mp3")])
        self.assertIsNone(artifacts.kept_output)

    def test_no_output_means_nothing_kept(self) -> None:
        self.assertEqual(RunArtifacts().removable(), [])
        self.assertIsNone(RunArtifacts().kept_output)


class RemoveFilesTest(TestCase):
    def test_removes_listed_files_and_skips_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write(root / "a.json", "12345")
            b = _write(root / "b.mp3", b"123")
            untouched = _write(root / "c.mp3", b"keep")

            summary = remove_files([a, b, a, root / "missing.mp3"])

            self.assertEqual(summary, CleanupSummary(files_removed=2, bytes_freed=8))
            self.assertFalse(a.exists())
            self.assertFalse(b.exists())
            self.assertTrue(untouched.exists())

    def test_counts_errors_without_raising(self) -> None:
        with TemporaryDirectory() as tmp:
            blocked = _write(Path(tmp) / "blocked.mp3", b"x")
            fine = _write(Path(tmp) / "fine.mp3", b"y")
            real_unlink = Path.unlink

            def unlink(path: Path, missing_ok: bool = False) -> None:
                if path.name == blocked.name:
                    raise PermissionError(13, "Permission denied", str(path))
                real_unlink(path, missing_ok=missing_ok)

            with patch.object(Path, "unlink", unlink):
                summary = remove_files([blocked, fine])

            self.assertEqual(summary.files_removed, 1)
            self.assertEqual(summary.errors, 1)
            self.assertTrue(blocked.exists())


class _FakeFetcher:
    """Stands in for PodcastFetcher with a single episode and a local cache."""

    def __init__(self, cache: Path) -> None:
        self.path = cache / "abc123.mp3"
        self.episode = EpisodeInfo(title="Ep One", audio_url="https://x/ep.mp3")

    def list_episodes(self, rss_url: str) -> list[EpisodeInfo]:
        return [self.episode]

    def find_cached(self, episode: EpisodeInfo) -> Path | None:
        return self.path if self.path.exists() else None

    def download_episode(self, episode: EpisodeInfo) -> Path:
        if not self.path.exists():
            _write(self.path, b"downloaded")
        return self.path


class CleanupCliTest(TestCase):
    def setUp(self) -> None:
        self._saved_env = _clear_whisper_env()
        config_mod._config = None
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / "output"
        self.cache = self.root / "cache"
        self.cfg = Config(
            gemini_api_key="test-key",
            output_dir=self.output,
            cache_dir=self.cache,
            models_dir=self.root / "models",
        )
        self.runner = CliRunner()
        self.upload_result: tuple[str | None, str | None] = (
            "https://s3/rss.xml",
            "https://s3/ep.mp3",
        )

    def tearDown(self) -> None:
        config_mod._config = None
        self._tmp.cleanup()
        _restore_env(self._saved_env)

    def _process(
        self, audio_path: Path, ad_regions: list[AdRegion], output_path: Path | None
    ) -> ProcessingResult:
        out = output_path or self.output / f"{audio_path.stem}_clean.mp3"
        _write(out, b"cleaned-audio")
        return ProcessingResult(10.0, 9.0, 1, ad_regions, str(out))

    def _preview(
        self, audio_path: Path, ad_regions: list[AdRegion]
    ) -> ProcessingResult:
        planned = self.output / f"{audio_path.stem}_clean.mp3"
        return ProcessingResult(10.0, 9.0, 1, ad_regions, str(planned))

    def _invoke(self, args: list[str], *, detect_error: Exception | None = None):
        fake_fetcher = _FakeFetcher(self.cache)
        with (
            patch("podclean.cli.get_config", return_value=self.cfg),
            patch("podclean.cli.Transcriber") as transcriber,
            patch("podclean.cli.AdDetector") as detector,
            patch("podclean.cli.AudioProcessor") as processor,
            patch("podclean.cli.upload_to_s3", return_value=self.upload_result),
            patch("podclean.cli.PodcastFetcher", return_value=fake_fetcher),
        ):
            transcriber.return_value.transcribe.return_value = [
                TranscriptSegment(start=0.0, end=3.0, text="hello")
            ]
            if detect_error is not None:
                detector.return_value.detect_ads.side_effect = detect_error
            else:
                detector.return_value.detect_ads.return_value = [_AD]
            processor.return_value.process.side_effect = self._process
            processor.return_value.preview.side_effect = self._preview
            return self.runner.invoke(cli, args)

    def test_help_documents_cleanup_flag(self) -> None:
        for command in ("file", "feed"):
            result = self.runner.invoke(cli, [command, "--help"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("--cleanup", result.output)
            self.assertIn("files this run created", result.output)

    def test_no_standalone_cleanup_mode(self) -> None:
        self.assertNotEqual(self.runner.invoke(cli, ["--cleanup"]).exit_code, 0)
        self.assertNotEqual(self.runner.invoke(cli, ["cleanup"]).exit_code, 0)

    def test_file_upload_cleanup_deletes_only_this_runs_files(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")
        earlier = _write(self.output / "earlier_clean.mp3", b"previous run")
        state = _write(self.output / "processed.json", "{}")

        result = self._invoke(["file", str(source), "--upload", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(source.exists())
        self.assertFalse((self.root / "episode.json").exists())
        self.assertFalse((self.output / "episode_clean.mp3").exists())
        self.assertTrue(earlier.exists())
        self.assertTrue(state.exists())
        self.assertIn("Removed 2 file(s)", result.output)

    def test_file_cleanup_without_upload_keeps_output(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")

        result = self._invoke(["file", str(source), "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((self.output / "episode_clean.mp3").exists())
        self.assertFalse((self.root / "episode.json").exists())
        self.assertIn("not uploaded", result.output)

    def test_file_cleanup_keeps_output_when_upload_fails(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")
        self.upload_result = (None, None)

        result = self._invoke(["file", str(source), "--upload", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((self.output / "episode_clean.mp3").exists())

    def test_file_cleanup_keeps_reused_transcript(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")
        transcript = _write(
            self.root / "episode.json", '[{"start": 0, "end": 1, "text": "hi"}]'
        )

        result = self._invoke(["file", str(source), "--upload", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(transcript.exists())

    def test_file_without_cleanup_deletes_nothing_extra(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")

        with patch("podclean.cli.remove_files") as remove:
            result = self._invoke(["file", str(source), "--upload"])

        self.assertEqual(result.exit_code, 0, result.output)
        remove.assert_not_called()
        self.assertTrue((self.root / "episode.json").exists())
        self.assertTrue((self.output / "episode_clean.mp3").exists())

    def test_pipeline_error_still_cleans_and_propagates(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")

        result = self._invoke(
            ["file", str(source), "--cleanup"], detect_error=RuntimeError("boom")
        )

        self.assertIsInstance(result.exception, RuntimeError)
        self.assertFalse((self.root / "episode.json").exists())
        self.assertTrue(source.exists())

    def test_deletion_error_exits_nonzero(self) -> None:
        source = _write(self.root / "episode.mp3", b"only copy")

        with patch(
            "podclean.cli.remove_files", return_value=CleanupSummary(errors=1)
        ):
            result = self._invoke(["file", str(source), "--upload", "--cleanup"])

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Could not delete 1 file(s)", result.output)

    def test_feed_preview_cleanup_deletes_new_download_and_transcript(self) -> None:
        result = self._invoke(["feed", "https://x/rss", "--preview", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse((self.cache / "abc123.mp3").exists())
        self.assertFalse((self.cache / "abc123.json").exists())

    def test_feed_preview_cleanup_keeps_previously_cached_files(self) -> None:
        cached_audio = _write(self.cache / "abc123.mp3", b"from an earlier run")
        cached_transcript = _write(
            self.cache / "abc123.json", '[{"start": 0, "end": 1, "text": "hi"}]'
        )

        result = self._invoke(["feed", "https://x/rss", "--preview", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(cached_audio.exists())
        self.assertTrue(cached_transcript.exists())
        self.assertIn("Nothing to clean", result.output)

    def test_feed_upload_cleanup_deletes_uploaded_output(self) -> None:
        result = self._invoke(["feed", "https://x/rss", "--upload", "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse((self.output / "Ep_One_clean.mp3").exists())
        self.assertFalse((self.cache / "abc123.mp3").exists())

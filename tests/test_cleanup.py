"""Tests for local leftover cleanup (CLI --cleanup / cleanup command)."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from click.testing import CliRunner

from podclean import config as config_mod
from podclean.cleanup import cleanup_local_artifacts, format_bytes
from podclean.cli import cli
from podclean.config import Config

_WHISPER_ENV = (
    "WHISPER_BACKEND",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "GEMINI_API_KEY",
)


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


class CleanupLocalArtifactsTest(TestCase):
    def test_deletes_expected_leftovers(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            cache = root / "cache"
            output.mkdir()
            cache.mkdir()

            leftovers = [
                _write(output / "Daily_Stoic_clean.mp3", b"clean" * 20),
                _write(output / "episode.wav", b"wav" * 10),
                _write(output / "cut.tmp", b"tmp"),
                _write(output / "podclean.log", b"log"),
                _write(cache / "abc123def4567890.mp3", b"orig" * 50),
                _write(cache / "abc123def4567890.json", '[{"text": "hi"}]'),
            ]
            expected_bytes = sum(path.stat().st_size for path in leftovers)

            summary = cleanup_local_artifacts(output_dir=output, cache_dir=cache)

            self.assertEqual(summary.files_removed, 6)
            self.assertEqual(summary.bytes_freed, expected_bytes)
            for path in leftovers:
                self.assertFalse(path.exists())
            self.assertTrue(output.is_dir())
            self.assertTrue(cache.is_dir())

    def test_preserves_processed_and_config_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            cache = root / "cache"
            output.mkdir()
            cache.mkdir()

            leftover = _write(output / "show_clean.mp3", b"xx")
            processed = _write(output / "processed.json", '{"ids": []}')
            processed_alt = _write(cache / "processed_episodes.json", "{}")
            feeds = _write(output / "feeds.json", '["https://example.com/rss"]')
            env_file = _write(output / ".env", "GEMINI_API_KEY=secret")
            env_example = _write(cache / ".env.example", "GEMINI_API_KEY=")
            unrelated = _write(root / "keep_me.mp3", b"outside")

            summary = cleanup_local_artifacts(output_dir=output, cache_dir=cache)

            self.assertEqual(summary.files_removed, 1)
            self.assertFalse(leftover.exists())
            self.assertTrue(processed.exists())
            self.assertEqual(processed.read_text(encoding="utf-8"), '{"ids": []}')
            self.assertTrue(processed_alt.exists())
            self.assertTrue(feeds.exists())
            self.assertTrue(env_file.exists())
            self.assertTrue(env_example.exists())
            self.assertTrue(unrelated.exists())

    def test_noop_when_nothing_to_clean(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            cache = root / "cache"
            output.mkdir()
            cache.mkdir()
            _write(output / "processed.json", "{}")
            _write(output / "feeds.json", "[]")

            summary = cleanup_local_artifacts(output_dir=output, cache_dir=cache)
            self.assertEqual(summary.files_removed, 0)
            self.assertEqual(summary.bytes_freed, 0)
            self.assertEqual(summary.errors, 0)
            self.assertTrue((output / "processed.json").exists())

            # Missing dirs are also a no-op.
            missing = cleanup_local_artifacts(
                output_dir=root / "no-output",
                cache_dir=root / "no-cache",
            )
            self.assertEqual(missing.files_removed, 0)
            self.assertEqual(missing.bytes_freed, 0)

            # Idempotent: a second sweep of an already-clean tree is still a no-op.
            again = cleanup_local_artifacts(output_dir=output, cache_dir=cache)
            self.assertEqual(again.files_removed, 0)
            self.assertEqual(again.bytes_freed, 0)


class CleanupCliTest(TestCase):
    def setUp(self) -> None:
        self._saved_env = _clear_whisper_env()
        config_mod._config = None
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / "output"
        self.cache = self.root / "cache"
        self.models = self.root / "models"
        self.output.mkdir()
        self.cache.mkdir()
        self.models.mkdir()
        self.cfg = Config(
            gemini_api_key="test-key",
            output_dir=self.output,
            cache_dir=self.cache,
            models_dir=self.models,
        )
        self.runner = CliRunner()

    def tearDown(self) -> None:
        config_mod._config = None
        self._tmp.cleanup()
        _restore_env(self._saved_env)

    def _patch_config(self):
        return patch("podclean.cli.get_config", return_value=self.cfg)

    def test_help_documents_cleanup_flag(self) -> None:
        root_help = self.runner.invoke(cli, ["--help"])
        self.assertEqual(root_help.exit_code, 0)
        self.assertIn("--cleanup", root_help.output)
        self.assertIn("cleanup", root_help.output)

        file_help = self.runner.invoke(cli, ["file", "--help"])
        self.assertEqual(file_help.exit_code, 0)
        self.assertIn("--cleanup", file_help.output)

        feed_help = self.runner.invoke(cli, ["feed", "--help"])
        self.assertEqual(feed_help.exit_code, 0)
        self.assertIn("--cleanup", feed_help.output)

    def test_cleanup_flag_only_deletes_leftovers(self) -> None:
        leftover = _write(self.output / "ep_clean.mp3", b"audio-bytes")
        cached = _write(self.cache / "deadbeefcafebabe.mp3", b"orig")
        processed = _write(self.output / "processed.json", "{}")

        with self._patch_config():
            result = self.runner.invoke(cli, ["--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Removed", result.output)
        self.assertIn("2 file", result.output)
        self.assertFalse(leftover.exists())
        self.assertFalse(cached.exists())
        self.assertTrue(processed.exists())

    def test_cleanup_command_is_noop_when_empty(self) -> None:
        _write(self.output / "processed.json", "{}")
        with self._patch_config():
            result = self.runner.invoke(cli, ["cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Nothing to clean", result.output)
        self.assertTrue((self.output / "processed.json").exists())

    def test_file_cleanup_runs_after_pipeline(self) -> None:
        audio = _write(self.root / "episode.mp3", b"fake-audio")
        leftover = _write(self.output / "episode_clean.mp3", b"generated")
        processed = _write(self.output / "processed.json", "{}")

        with (
            self._patch_config(),
            patch("podclean.cli._run_pipeline", return_value=None) as pipeline,
        ):
            result = self.runner.invoke(cli, ["file", str(audio), "--cleanup"])

        self.assertEqual(result.exit_code, 0, result.output)
        pipeline.assert_called_once()
        self.assertFalse(leftover.exists())
        self.assertTrue(processed.exists())
        self.assertTrue(audio.exists())
        self.assertIn("Removed", result.output)

    def test_file_without_cleanup_keeps_leftovers(self) -> None:
        audio = _write(self.root / "episode.mp3", b"fake-audio")
        leftover = _write(self.output / "episode_clean.mp3", b"generated")

        with (
            self._patch_config(),
            patch("podclean.cli._run_pipeline", return_value=None),
            patch("podclean.cli._run_cleanup") as sweep,
        ):
            result = self.runner.invoke(cli, ["file", str(audio)])

        self.assertEqual(result.exit_code, 0, result.output)
        sweep.assert_not_called()
        self.assertTrue(leftover.exists())

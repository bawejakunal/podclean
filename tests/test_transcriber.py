"""Tests for Whisper backend selection and the Transcriber facade."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from podclean import config as config_mod
from podclean.cli import WHISPER_MODEL_HELP
from podclean.config import Config
from pathlib import Path as _PathForToml
from podclean.transcriber import Transcriber, TranscriptionError
from podclean.whisper_backend import (
    FASTER_WHISPER_DEFAULT_MODEL,
    MLX_DEFAULT_MODEL,
    backend_defaults,
    missing_backend_error,
    platform_default_backend,
    resolve_whisper_backend,
)

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


class ResolveBackendTest(TestCase):
    def test_explicit_override_wins_even_if_not_installed(self) -> None:
        self.assertEqual(
            resolve_whisper_backend(
                "mlx",
                installed=set(),
                platform_default="faster-whisper",
            ),
            "mlx",
        )
        self.assertEqual(
            resolve_whisper_backend(
                "faster-whisper",
                installed=set(),
                platform_default="mlx",
            ),
            "faster-whisper",
        )

    def test_aliases_and_underscores(self) -> None:
        self.assertEqual(resolve_whisper_backend("mlx-whisper"), "mlx")
        self.assertEqual(resolve_whisper_backend("faster_whisper"), "faster-whisper")
        self.assertEqual(resolve_whisper_backend("cpu"), "faster-whisper")

    def test_invalid_override_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_whisper_backend("openai")
        self.assertIn("mlx", str(ctx.exception))
        self.assertIn("faster-whisper", str(ctx.exception))

    def test_env_override_used_when_argument_omitted(self) -> None:
        saved = _clear_whisper_env()
        try:
            os.environ["WHISPER_BACKEND"] = "faster-whisper"
            self.assertEqual(
                resolve_whisper_backend(
                    None,
                    installed={"mlx"},
                    platform_default="mlx",
                ),
                "faster-whisper",
            )
        finally:
            _restore_env(saved)

    def test_prefers_platform_default_when_both_installed(self) -> None:
        self.assertEqual(
            resolve_whisper_backend(
                None,
                installed={"mlx", "faster-whisper"},
                platform_default="mlx",
            ),
            "mlx",
        )
        self.assertEqual(
            resolve_whisper_backend(
                None,
                installed={"mlx", "faster-whisper"},
                platform_default="faster-whisper",
            ),
            "faster-whisper",
        )

    def test_uses_the_other_installed_backend(self) -> None:
        self.assertEqual(
            resolve_whisper_backend(
                None,
                installed={"faster-whisper"},
                platform_default="mlx",
            ),
            "faster-whisper",
        )
        self.assertEqual(
            resolve_whisper_backend(
                None,
                installed={"mlx"},
                platform_default="faster-whisper",
            ),
            "mlx",
        )

    def test_none_installed_falls_back_to_platform_default(self) -> None:
        self.assertEqual(
            resolve_whisper_backend(
                None,
                installed=set(),
                platform_default="faster-whisper",
            ),
            "faster-whisper",
        )

    def test_platform_default_darwin_is_mlx(self) -> None:
        with patch("podclean.whisper_backend.sys") as sys_mod:
            sys_mod.platform = "darwin"
            self.assertEqual(platform_default_backend(), "mlx")

    def test_platform_default_linux_is_faster_whisper(self) -> None:
        with patch("podclean.whisper_backend.sys") as sys_mod:
            sys_mod.platform = "linux"
            self.assertEqual(platform_default_backend(), "faster-whisper")


class BackendDefaultsTest(TestCase):
    def test_mlx_defaults(self) -> None:
        defaults = backend_defaults("mlx")
        self.assertEqual(defaults.model, MLX_DEFAULT_MODEL)
        self.assertEqual(defaults.device, "gpu")
        self.assertEqual(defaults.compute_type, "int8")

    def test_faster_whisper_defaults_cpu_without_cuda(self) -> None:
        with patch("podclean.whisper_backend.cuda_available", return_value=False):
            defaults = backend_defaults("faster-whisper")
        self.assertEqual(defaults.model, FASTER_WHISPER_DEFAULT_MODEL)
        self.assertEqual(defaults.device, "cpu")
        self.assertEqual(defaults.compute_type, "int8")

    def test_faster_whisper_defaults_cuda_when_available(self) -> None:
        with patch("podclean.whisper_backend.cuda_available", return_value=True):
            defaults = backend_defaults("faster-whisper")
        self.assertEqual(defaults.device, "cuda")

    def test_missing_backend_error_includes_pip_commands(self) -> None:
        mlx_msg = missing_backend_error("mlx")
        self.assertIn('pip install ".[mlx]"', mlx_msg)
        self.assertIn("pip install mlx-whisper", mlx_msg)

        cpu_msg = missing_backend_error("faster-whisper")
        self.assertIn('pip install ".[cpu]"', cpu_msg)
        self.assertIn("pip install faster-whisper", cpu_msg)


class ConfigWhisperDefaultsTest(TestCase):
    def setUp(self) -> None:
        self._saved = _clear_whisper_env()
        config_mod._config = None

    def tearDown(self) -> None:
        config_mod._config = None
        _restore_env(self._saved)

    def test_mlx_constructor_override_fills_backend_defaults(self) -> None:
        cfg = Config(whisper_backend="mlx")
        self.assertEqual(cfg.whisper_backend, "mlx")
        self.assertEqual(cfg.whisper_model, MLX_DEFAULT_MODEL)
        self.assertEqual(cfg.whisper_device, "gpu")
        self.assertEqual(cfg.whisper_compute_type, "int8")

    def test_faster_whisper_constructor_override_fills_backend_defaults(self) -> None:
        with patch("podclean.whisper_backend.cuda_available", return_value=False):
            cfg = Config(whisper_backend="faster-whisper")
        self.assertEqual(cfg.whisper_backend, "faster-whisper")
        self.assertEqual(cfg.whisper_model, FASTER_WHISPER_DEFAULT_MODEL)
        self.assertEqual(cfg.whisper_device, "cpu")
        self.assertEqual(cfg.whisper_compute_type, "int8")

    def test_whisper_model_env_overrides_backend_default(self) -> None:
        os.environ["WHISPER_MODEL"] = "small"
        cfg = Config(whisper_backend="faster-whisper")
        self.assertEqual(cfg.whisper_model, "small")

    def test_device_and_compute_env_overrides(self) -> None:
        os.environ["WHISPER_DEVICE"] = "cuda"
        os.environ["WHISPER_COMPUTE_TYPE"] = "float16"
        cfg = Config(whisper_backend="faster-whisper")
        self.assertEqual(cfg.whisper_device, "cuda")
        self.assertEqual(cfg.whisper_compute_type, "float16")

    def test_invalid_backend_is_reported_by_validate(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["WHISPER_BACKEND"] = "nope"
        cfg = Config()
        errors = cfg.validate()
        self.assertTrue(any("WHISPER_BACKEND" in err for err in errors))


class _FakeMlx:
    last_call: dict | None = None

    @staticmethod
    def transcribe(path: str, path_or_hf_repo: str, word_timestamps: bool) -> dict:
        _FakeMlx.last_call = {
            "path": path,
            "path_or_hf_repo": path_or_hf_repo,
            "word_timestamps": word_timestamps,
        }
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": " hello world ",
                    "words": [
                        {
                            "word": " hello",
                            "start": 0.0,
                            "end": 0.5,
                            "probability": 0.9,
                        },
                        {"word": "world", "start": 0.5, "end": 1.2},
                    ],
                }
            ],
        }


class _FakeWord:
    def __init__(self, word: str, start: float, end: float, probability: float = 1.0):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSegment:
    def __init__(self) -> None:
        self.start = 1.0
        self.end = 2.0
        self.text = " hello "
        self.words = [_FakeWord("hello", 1.0, 2.0, 0.8)]


class _FakeWhisperModel:
    last_init: dict | None = None
    last_transcribe: dict | None = None

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        _FakeWhisperModel.last_init = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }

    def transcribe(self, path: str, word_timestamps: bool = True, vad_filter: bool = False):
        _FakeWhisperModel.last_transcribe = {
            "path": path,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
        }
        return iter([_FakeSegment()]), SimpleNamespace(language="en")


class TranscriberFacadeTest(TestCase):
    def setUp(self) -> None:
        self._saved = _clear_whisper_env()
        config_mod._config = None
        _FakeMlx.last_call = None
        _FakeWhisperModel.last_init = None
        _FakeWhisperModel.last_transcribe = None

    def tearDown(self) -> None:
        config_mod._config = None
        _restore_env(self._saved)

    def test_mlx_engine_returns_word_level_segments(self) -> None:
        audio = Path(__file__).resolve()
        with (
            patch("podclean.transcriber.mlx_whisper", _FakeMlx),
            patch("podclean.transcriber.WhisperModel", None),
        ):
            transcriber = Transcriber(
                model_size="mlx-community/whisper-tiny",
                backend="mlx",
            )
            segments = transcriber.transcribe(audio)

        self.assertEqual(transcriber.backend, "mlx")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "hello world")
        self.assertEqual([w.text for w in segments[0].words], ["hello", "world"])
        self.assertEqual(segments[0].words[0].probability, 0.9)
        self.assertEqual(segments[0].words[1].probability, 1.0)
        assert _FakeMlx.last_call is not None
        self.assertEqual(
            _FakeMlx.last_call["path_or_hf_repo"],
            "mlx-community/whisper-tiny",
        )

    def test_faster_whisper_engine_returns_word_level_segments(self) -> None:
        audio = Path(__file__).resolve()
        with (
            patch("podclean.transcriber.WhisperModel", _FakeWhisperModel),
            patch("podclean.transcriber.mlx_whisper", None),
            patch("podclean.whisper_backend.cuda_available", return_value=False),
        ):
            config_mod._config = Config(whisper_backend="faster-whisper")
            transcriber = Transcriber(model_size="tiny", backend="faster-whisper")
            segments = transcriber.transcribe(audio)

        self.assertEqual(transcriber.backend, "faster-whisper")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "hello")
        self.assertEqual(segments[0].words[0].text, "hello")
        self.assertEqual(segments[0].words[0].probability, 0.8)
        assert _FakeWhisperModel.last_init is not None
        self.assertEqual(_FakeWhisperModel.last_init["model_size"], "tiny")
        self.assertEqual(_FakeWhisperModel.last_init["device"], "cpu")
        self.assertEqual(_FakeWhisperModel.last_init["compute_type"], "int8")
        assert _FakeWhisperModel.last_transcribe is not None
        self.assertTrue(_FakeWhisperModel.last_transcribe["word_timestamps"])
        self.assertTrue(_FakeWhisperModel.last_transcribe["vad_filter"])

    def test_missing_backend_error_includes_install_command(self) -> None:
        with (
            patch("podclean.transcriber.mlx_whisper", None),
            patch("podclean.transcriber.WhisperModel", None),
        ):
            with self.assertRaises(TranscriptionError) as ctx:
                Transcriber(backend="mlx")
        self.assertIn('pip install ".[mlx]"', str(ctx.exception))
        self.assertIn("pip install mlx-whisper", str(ctx.exception))

        with (
            patch("podclean.transcriber.mlx_whisper", None),
            patch("podclean.transcriber.WhisperModel", None),
        ):
            with self.assertRaises(TranscriptionError) as ctx:
                Transcriber(backend="faster-whisper")
        self.assertIn('pip install ".[cpu]"', str(ctx.exception))
        self.assertIn("pip install faster-whisper", str(ctx.exception))

    def test_missing_audio_file_raises(self) -> None:
        with patch("podclean.transcriber.mlx_whisper", _FakeMlx):
            transcriber = Transcriber(model_size="tiny", backend="mlx")
            with self.assertRaises(FileNotFoundError):
                transcriber.transcribe(Path("/tmp/podclean-missing-audio.wav"))


class CliHelpTest(TestCase):
    def test_model_help_mentions_both_backends(self) -> None:
        self.assertIn("HuggingFace", WHISPER_MODEL_HELP)
        self.assertIn("faster-whisper", WHISPER_MODEL_HELP)
        self.assertIn("large-v3-turbo", WHISPER_MODEL_HELP)
        self.assertNotIn("MLX Whisper model repo (default:", WHISPER_MODEL_HELP)


class PackagingMarkersTest(TestCase):
    def test_pyproject_selects_backend_with_environment_markers(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
        self.assertIn("mlx-whisper>=0.4.0; sys_platform == 'darwin' and platform_machine == 'arm64'", text)
        self.assertIn("faster-whisper>=1.0.0; sys_platform != 'darwin' or platform_machine != 'arm64'", text)
        self.assertIn('mlx = ["mlx-whisper>=0.4.0"]', text)
        self.assertIn('cpu = ["faster-whisper>=1.0.0"]', text)

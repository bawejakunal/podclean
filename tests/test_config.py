"""Tests for .env discovery used by tool installs and checkouts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase

from podclean.config import (
    PODCLEAN_ENV_VAR,
    Config,
    dotenv_search_paths,
    is_source_checkout,
    load_dotenv_files,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class SourceCheckoutDetectionTest(TestCase):
    def test_checkout_with_pyproject_and_package(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write(root / "pyproject.toml", '[project]\nname = "podclean"\n')
            _write(root / "podclean" / "__init__.py", "")
            self.assertTrue(is_source_checkout(root))

    def test_site_packages_layout_is_not_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = Path(raw) / "lib" / "python3.12" / "site-packages"
            _write(site / "podclean" / "__init__.py", "")
            _write(site / ".env", "GEMINI_API_KEY=from-site-packages\n")
            self.assertFalse(is_source_checkout(site))


class DotenvSearchPathsTest(TestCase):
    def test_priority_order_and_xdg(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cwd = tmp / "cwd"
            home = tmp / "home"
            xdg = tmp / "xdg"
            cwd.mkdir()
            home.mkdir()
            xdg.mkdir()

            paths = dotenv_search_paths(
                cwd=cwd,
                home=home,
                package_root=tmp / "not-a-checkout",
                environ={"XDG_CONFIG_HOME": str(xdg)},
            )
            self.assertEqual(
                paths,
                [
                    cwd / ".env",
                    xdg / "podclean" / ".env",
                    home / ".podclean" / ".env",
                ],
            )

    def test_podclean_env_is_highest_priority(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            explicit = tmp / "custom.env"
            cwd = tmp / "cwd"
            cwd.mkdir()
            paths = dotenv_search_paths(
                cwd=cwd,
                home=tmp / "home",
                package_root=tmp / "not-a-checkout",
                environ={PODCLEAN_ENV_VAR: str(explicit)},
            )
            self.assertEqual(paths[0], explicit)

    def test_source_checkout_env_is_included(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write(root / "pyproject.toml", '[project]\nname = "podclean"\n')
            _write(root / "podclean" / "__init__.py", "")
            cwd = root / "elsewhere"
            cwd.mkdir()
            paths = dotenv_search_paths(
                cwd=cwd,
                home=root / "home",
                package_root=root,
                environ={},
            )
            self.assertIn(root / ".env", paths)

    def test_duplicate_cwd_and_checkout_is_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write(root / "pyproject.toml", '[project]\nname = "podclean"\n')
            _write(root / "podclean" / "__init__.py", "")
            paths = dotenv_search_paths(
                cwd=root,
                home=root / "home",
                package_root=root,
                environ={},
            )
            self.assertEqual(paths.count(root / ".env"), 1)


class DotenvLayeringTest(TestCase):
    def setUp(self) -> None:
        self._keys = ("GEMINI_API_KEY", "WHISPER_MODEL", PODCLEAN_ENV_VAR)
        self._saved = {key: os.environ[key] for key in self._keys if key in os.environ}
        for key in self._keys:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in self._keys:
            os.environ.pop(key, None)
        os.environ.update(self._saved)

    def test_cwd_overrides_user_config_and_user_config_fills_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cwd_env = _write(tmp / "cwd" / ".env", "WHISPER_MODEL=small\n")
            xdg_env = _write(
                tmp / "xdg" / "podclean" / ".env",
                "GEMINI_API_KEY=from-xdg\nWHISPER_MODEL=large-v3-turbo\n",
            )
            loaded = load_dotenv_files([cwd_env, xdg_env])
            self.assertEqual(loaded, [cwd_env, xdg_env])
            self.assertEqual(os.environ["WHISPER_MODEL"], "small")
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from-xdg")

    def test_existing_environment_wins_over_files(self) -> None:
        os.environ["GEMINI_API_KEY"] = "from-process"
        with tempfile.TemporaryDirectory() as raw:
            env_file = _write(Path(raw) / ".env", "GEMINI_API_KEY=from-file\n")
            load_dotenv_files([env_file])
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from-process")


class ConfigValidateMessageTest(TestCase):
    def test_missing_api_key_mentions_user_config_path(self) -> None:
        saved = os.environ.pop("GEMINI_API_KEY", None)
        try:
            cfg = Config(gemini_api_key="")
            errors = cfg.validate()
            self.assertTrue(any("~/.config/podclean/.env" in err for err in errors))
        finally:
            if saved is not None:
                os.environ["GEMINI_API_KEY"] = saved

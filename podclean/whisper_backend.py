"""Whisper backend selection for mlx-whisper vs faster-whisper.

Install-time extras/markers pick a default stack; this module decides which
backend the transcriber should use at runtime.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Literal

try:
    import ctranslate2
except ImportError:  # pragma: no cover - optional faster-whisper stack
    ctranslate2 = None

BackendName = Literal["mlx", "faster-whisper"]

MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_WHISPER_DEFAULT_MODEL = "large-v3-turbo"

_BACKEND_ALIASES: dict[str, BackendName] = {
    "mlx": "mlx",
    "mlx-whisper": "mlx",
    "faster-whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "cpu": "faster-whisper",
}


@dataclass(frozen=True)
class WhisperSettings:
    """Model/device/compute triple for one backend."""

    model: str
    device: str
    compute_type: str


def normalize_backend_name(value: str) -> BackendName | None:
    """Return a canonical backend name, or ``None`` if *value* is unknown."""

    key = value.strip().lower().replace("_", "-")
    return _BACKEND_ALIASES.get(key)


def platform_default_backend() -> BackendName:
    """Return the preferred backend for this OS.

    macOS (Darwin) defaults to MLX; every other platform defaults to
    faster-whisper. Install markers still limit ``mlx-whisper`` to Apple
    Silicon so Intel Macs get faster-whisper via the installed-package step.
    """

    if sys.platform == "darwin":
        return "mlx"
    return "faster-whisper"


def package_installed(module_name: str) -> bool:
    """Return True if *module_name* can be imported."""

    return importlib.util.find_spec(module_name) is not None


def installed_backends() -> set[BackendName]:
    """Discover Whisper backends available in the current environment."""

    found: set[BackendName] = set()
    if package_installed("mlx_whisper"):
        found.add("mlx")
    if package_installed("faster_whisper"):
        found.add("faster-whisper")
    return found


def resolve_whisper_backend(
    override: str | None = None,
    *,
    installed: set[str] | None = None,
    platform_default: BackendName | None = None,
) -> BackendName:
    """Choose ``mlx`` or ``faster-whisper``.

    Selection order:
    1. Explicit override (argument or ``WHISPER_BACKEND``).
    2. An already-installed backend, preferring the platform default if both
       are present.
    3. Platform default (Darwin → mlx, otherwise faster-whisper).
    """

    if override is None:
        env_value = os.getenv("WHISPER_BACKEND", "").strip()
        override = env_value or None

    if override:
        normalized = normalize_backend_name(override)
        if normalized is None:
            raise ValueError(
                f"Invalid WHISPER_BACKEND={override!r}. "
                "Use 'mlx' or 'faster-whisper'."
            )
        return normalized

    if installed is None:
        installed = installed_backends()
    if platform_default is None:
        platform_default = platform_default_backend()

    if platform_default in installed:
        return platform_default

    other: BackendName = (
        "faster-whisper" if platform_default == "mlx" else "mlx"
    )
    if other in installed:
        return other

    return platform_default


def cuda_available() -> bool:
    """Return True when CTranslate2 can see at least one CUDA device."""

    if ctranslate2 is None:
        return False
    try:
        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def backend_defaults(backend: BackendName) -> WhisperSettings:
    """Return model/device/compute defaults for *backend*."""

    if backend == "mlx":
        return WhisperSettings(
            model=MLX_DEFAULT_MODEL,
            device="gpu",
            compute_type="int8",
        )
    return WhisperSettings(
        model=FASTER_WHISPER_DEFAULT_MODEL,
        device="cuda" if cuda_available() else "cpu",
        compute_type="int8",
    )


def backend_install_commands(backend: BackendName) -> tuple[str, str]:
    """Return ``(extra_install, package_install)`` commands for *backend*."""

    if backend == "mlx":
        return 'pip install ".[mlx]"', "pip install mlx-whisper"
    return 'pip install ".[cpu]"', "pip install faster-whisper"


def missing_backend_error(backend: BackendName) -> str:
    """Human-readable error when the selected backend is not installed."""

    extra_cmd, pkg_cmd = backend_install_commands(backend)
    label = "mlx-whisper" if backend == "mlx" else "faster-whisper"
    return (
        f"The {label} backend is not installed. "
        f"Install it with: {extra_cmd}  (or: {pkg_cmd})"
    )

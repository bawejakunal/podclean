"""Configuration management for PodClean."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from podclean.formats import OUTPUT_FORMATS
from podclean.whisper_backend import (
    BackendName,
    WhisperSettings,
    backend_defaults,
    normalize_backend_name,
    platform_default_backend,
    resolve_whisper_backend,
)

# Explicit file path, e.g. PODCLEAN_ENV=/path/to/.env
PODCLEAN_ENV_VAR = "PODCLEAN_ENV"


def _package_root() -> Path:
    """Directory that contains the ``podclean`` package (repo root or site-packages)."""

    return Path(__file__).resolve().parent.parent


def is_source_checkout(root: Path) -> bool:
    """Return True when *root* looks like this project's source tree.

    Used so a wheel/tool install does not treat ``site-packages/.env`` as
    configuration, while an editable checkout still finds the repo ``.env``.
    """

    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or not (root / "podclean" / "__init__.py").is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return 'name = "podclean"' in text


def dotenv_search_paths(
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    package_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[Path]:
    """Return candidate ``.env`` paths, highest priority first.

    1. ``PODCLEAN_ENV`` (explicit file)
    2. ``.env`` in the current working directory
    3. ``$XDG_CONFIG_HOME/podclean/.env`` (default ``~/.config/podclean/.env``)
    4. ``~/.podclean/.env``
    5. Checkout-root ``.env`` when running from an editable/source install
    """

    env = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd
    home = Path.home() if home is None else home
    package_root = _package_root() if package_root is None else package_root

    candidates: list[Path] = []

    explicit = (env.get(PODCLEAN_ENV_VAR) or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(cwd / ".env")

    xdg = (env.get("XDG_CONFIG_HOME") or "").strip()
    config_home = Path(xdg).expanduser() if xdg else home / ".config"
    candidates.append(config_home / "podclean" / ".env")
    candidates.append(home / ".podclean" / ".env")

    if is_source_checkout(package_root):
        candidates.append(package_root / ".env")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            key = path.expanduser().resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_dotenv_files(paths: list[Path] | None = None) -> list[Path]:
    """Load existing ``.env`` files, highest priority first.

    ``load_dotenv`` does not override variables already in the environment, so
    the first file that defines a key wins and later files fill gaps. Process
    environment variables always take precedence over every file.
    """

    if paths is None:
        paths = dotenv_search_paths()
    loaded: list[Path] = []
    for path in paths:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


load_dotenv_files()


@dataclass
class Config:
    """Application configuration with sensible defaults."""

    # --- API ---
    gemini_api_key: str = ""

    # --- Transcription ---
    # Empty strings mean "apply backend-specific defaults" in __post_init__.
    whisper_backend: str = ""
    whisper_model: str = ""
    whisper_device: str = ""
    whisper_compute_type: str = ""
    vad_filter: bool = True
    word_timestamps: bool = True

    # --- Ad Detection ---
    llm_provider: str = "gemini"  # gemini, ollama, openai
    llm_model: str = "gemini-2.5-flash"
    detection_confidence_threshold: float = 0.6
    chunk_duration_minutes: int = 15
    chunk_overlap_minutes: int = 1
    verification_pass: bool = True

    # --- Audio Processing ---
    output_format: str = "mp3"
    output_bitrate: str = "192k"
    crossfade_ms: int = 300
    fade_in_ms: int = 150
    fade_out_ms: int = 150
    buffer_seconds: float = 0.5  # buffer before/after ad cuts

    # --- Cloud / Notifications ---
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_bucket_name: str = ""
    aws_region_name: str = "us-east-1"
    
    resend_api_key: str = ""
    resend_from_email: str = ""
    email_recipient: str = ""

    # --- Paths ---
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".podclean" / "cache")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "output")
    models_dir: Path = field(
        default_factory=lambda: Path.home() / ".podclean" / "models"
    )

    def __post_init__(self) -> None:
        # Load from environment variables (override defaults)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        self._apply_whisper_settings()
        self.output_format = os.getenv("OUTPUT_FORMAT", self.output_format)
        self.output_bitrate = os.getenv("OUTPUT_BITRATE", self.output_bitrate)

        crossfade_env = os.getenv("CROSSFADE_MS")
        if crossfade_env:
            self.crossfade_ms = int(crossfade_env)

        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.aws_access_key_id)
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key)
        self.aws_bucket_name = os.getenv("AWS_BUCKET_NAME", self.aws_bucket_name)
        self.aws_region_name = os.getenv("AWS_REGION_NAME", self.aws_region_name)
        
        self.resend_api_key = os.getenv("RESEND_API_KEY", self.resend_api_key)
        self.resend_from_email = os.getenv("RESEND_FROM_EMAIL", self.resend_from_email)
        self.email_recipient = os.getenv("EMAIL_RECIPIENT", self.email_recipient)

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return a list of configuration errors (empty = valid)."""
        errors = []
        if not self.gemini_api_key:
            errors.append(
                "GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and set it in ~/.config/podclean/.env, a local .env, "
                "or as an environment variable."
            )

        if self.output_format not in OUTPUT_FORMATS:
            errors.append(
                f"Invalid output format: {self.output_format!r}. "
                f"Choose from: {', '.join(OUTPUT_FORMATS)}"
            )

        raw_backend = getattr(self, "_whisper_backend_override", "") or ""
        if raw_backend and normalize_backend_name(raw_backend) is None:
            errors.append(
                f"Invalid WHISPER_BACKEND={raw_backend!r}. "
                "Use 'mlx' or 'faster-whisper'."
            )
        return errors

    def _apply_whisper_settings(self) -> None:
        """Resolve backend and fill model/device/compute defaults."""

        raw_backend = (
            os.getenv("WHISPER_BACKEND", "").strip()
            or (self.whisper_backend.strip() if self.whisper_backend else "")
            or None
        )
        self._whisper_backend_override = raw_backend or ""
        try:
            self.whisper_backend = resolve_whisper_backend(raw_backend)
        except ValueError:
            # Keep a usable fallback so validate() can report the bad value.
            self.whisper_backend = platform_default_backend()

        # Explicit choices are remembered separately so whisper_settings_for()
        # can re-derive defaults for a different backend.
        self._whisper_model_override = os.getenv("WHISPER_MODEL") or self.whisper_model
        self._whisper_device_override = (
            os.getenv("WHISPER_DEVICE") or self.whisper_device
        )
        self._whisper_compute_type_override = (
            os.getenv("WHISPER_COMPUTE_TYPE") or self.whisper_compute_type
        )

        settings = self.whisper_settings_for(self.whisper_backend)  # type: ignore[arg-type]
        self.whisper_model = settings.model
        self.whisper_device = settings.device
        self.whisper_compute_type = settings.compute_type

    def whisper_settings_for(self, backend: BackendName) -> WhisperSettings:
        """Return the model/device/compute to use when running *backend*.

        Explicit settings (env vars or constructor overrides) always win;
        anything unset falls back to that backend's defaults, so switching
        backends never carries over an incompatible model or device.
        """

        defaults = backend_defaults(backend)
        return WhisperSettings(
            model=self._whisper_model_override or defaults.model,
            device=self._whisper_device_override or defaults.device,
            compute_type=self._whisper_compute_type_override or defaults.compute_type,
        )


# Singleton config
_config: Config | None = None


def get_config(**overrides: object) -> Config:
    """Get or create the global config, with optional overrides."""
    global _config
    if _config is None or overrides:
        _config = Config(**overrides)  # type: ignore[arg-type]
    return _config

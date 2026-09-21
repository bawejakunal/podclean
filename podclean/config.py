"""Configuration management for PodClean."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from podclean.whisper_backend import (
    backend_defaults,
    normalize_backend_name,
    platform_default_backend,
    resolve_whisper_backend,
)

# Load .env file from project root or current directory
_env_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent / ".env",
    Path.home() / ".config" / "podclean" / ".env",
    Path.home() / ".podclean" / ".env",
]
for p in _env_paths:
    if p.exists():
        load_dotenv(p)
        break


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
                "and set it in .env or as an environment variable."
            )

        if self.output_format not in ("mp3", "wav", "m4a"):
            errors.append(
                f"Invalid output format: {self.output_format!r}. "
                f"Choose from: mp3, wav, m4a"
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

        defaults = backend_defaults(self.whisper_backend)  # type: ignore[arg-type]
        self.whisper_model = (
            os.getenv("WHISPER_MODEL") or self.whisper_model or defaults.model
        )
        self.whisper_device = (
            os.getenv("WHISPER_DEVICE") or self.whisper_device or defaults.device
        )
        self.whisper_compute_type = (
            os.getenv("WHISPER_COMPUTE_TYPE")
            or self.whisper_compute_type
            or defaults.compute_type
        )


# Singleton config
_config: Config | None = None


def get_config(**overrides: object) -> Config:
    """Get or create the global config, with optional overrides."""
    global _config
    if _config is None or overrides:
        _config = Config(**overrides)  # type: ignore[arg-type]
    return _config

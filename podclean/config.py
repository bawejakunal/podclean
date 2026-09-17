"""Configuration management for PodClean."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .rss import DEFAULT_MAX_FEED_BYTES, DEFAULT_MAX_FEED_ITEMS

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
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    whisper_device: str = "gpu"
    whisper_compute_type: str = "int8"
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
    # Newest episodes kept in the public S3 RSS file. 0 = unlimited.
    rss_max_items: int = DEFAULT_MAX_FEED_ITEMS
    # Soft cap on the player-facing rss.xml size. 0 = unlimited.
    rss_max_bytes: int = DEFAULT_MAX_FEED_BYTES
    
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
        self.whisper_model = os.getenv("WHISPER_MODEL", self.whisper_model)
        self.output_format = os.getenv("OUTPUT_FORMAT", self.output_format)
        self.output_bitrate = os.getenv("OUTPUT_BITRATE", self.output_bitrate)

        crossfade_env = os.getenv("CROSSFADE_MS")
        if crossfade_env:
            self.crossfade_ms = int(crossfade_env)

        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.aws_access_key_id)
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key)
        self.aws_bucket_name = os.getenv("AWS_BUCKET_NAME", self.aws_bucket_name)
        self.aws_region_name = os.getenv("AWS_REGION_NAME", self.aws_region_name)

        rss_max_items_env = os.getenv("RSS_MAX_ITEMS")
        if rss_max_items_env is not None and rss_max_items_env.strip() != "":
            try:
                self.rss_max_items = int(rss_max_items_env)
            except ValueError:
                pass

        rss_max_bytes_env = os.getenv("RSS_MAX_BYTES")
        if rss_max_bytes_env is not None and rss_max_bytes_env.strip() != "":
            try:
                self.rss_max_bytes = int(rss_max_bytes_env)
            except ValueError:
                pass
        
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
        return errors


# Singleton config
_config: Config | None = None


def get_config(**overrides: object) -> Config:
    """Get or create the global config, with optional overrides."""
    global _config
    if _config is None or overrides:
        _config = Config(**overrides)  # type: ignore[arg-type]
    return _config

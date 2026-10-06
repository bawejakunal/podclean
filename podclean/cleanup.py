"""Local leftover cleanup for PodClean.

Removes generated audio, download-cache files, and scratch artifacts from
the configured output and cache directories.  Config, feed lists, and
"already processed" state are never deleted.  S3 is not touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Generated / cached media the CLI writes locally.
_AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".wav", ".ogg"})

# Transcript JSON next to cached audio, temp cuts, and optional run logs.
_SCRATCH_SUFFIXES = frozenset({".json", ".tmp", ".temp", ".partial", ".log"})

# Never delete these, even if they sit inside output/ or cache/.
_PRESERVE_NAMES = frozenset(
    {
        ".env",
        ".env.example",
        "processed.json",
        "feeds.json",
    }
)

# Extra name prefixes that look like processed-state / feed-list files.
_PRESERVE_JSON_PREFIXES = ("processed", "feeds")

# Skip entire directories that look like the project or a virtualenv.
_UNSAFE_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "tests",
        "podclean",
        "podclean.egg-info",
    }
)


@dataclass(frozen=True)
class CleanupSummary:
    """Counts from a local leftover sweep."""

    files_removed: int = 0
    bytes_freed: int = 0
    errors: int = 0

    def format_bytes(self) -> str:
        """Return a short human-readable size for ``bytes_freed``."""
        return format_bytes(self.bytes_freed)


def format_bytes(size: int) -> str:
    """Format *size* in bytes as a short approximate string."""
    n = float(max(size, 0))
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "bytes":
                return f"{int(n)} bytes"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{int(size)} bytes"


def cleanup_local_artifacts(
    *,
    output_dir: Path,
    cache_dir: Path,
) -> CleanupSummary:
    """Delete leftover local media and scratch under *output_dir* and *cache_dir*.

    Missing directories are ignored.  The sweep is idempotent: a second call
    with nothing left to delete returns a zero summary.

    Parameters
    ----------
    output_dir:
        Configured cleaned-audio directory (typically ``./output``).
    cache_dir:
        Configured episode download / transcript cache.
    """
    files_removed = 0
    bytes_freed = 0
    errors = 0

    for directory in (output_dir, cache_dir):
        removed, freed, failed = _clean_directory(directory)
        files_removed += removed
        bytes_freed += freed
        errors += failed

    return CleanupSummary(
        files_removed=files_removed,
        bytes_freed=bytes_freed,
        errors=errors,
    )


def _clean_directory(directory: Path) -> tuple[int, int, int]:
    """Remove leftover files in *directory*.  Returns (count, bytes, errors)."""
    if not _is_safe_cleanup_dir(directory):
        return 0, 0, 0

    files_removed = 0
    bytes_freed = 0
    errors = 0

    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0, 0, 0

    for path in entries:
        if not path.is_file():
            continue
        if not _is_leftover_file(path):
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            errors += 1
            continue
        files_removed += 1
        bytes_freed += size

    return files_removed, bytes_freed, errors


def _is_safe_cleanup_dir(directory: Path) -> bool:
    """Return True if *directory* exists and is safe to sweep for leftovers."""
    try:
        resolved = directory.resolve()
    except OSError:
        return False

    if not resolved.is_dir():
        return False

    if resolved.name in _UNSAFE_DIR_NAMES:
        return False

    # Never sweep filesystem root, the user's home, or the project checkout.
    if resolved in {Path("/"), Path.home()}:
        return False
    project_root = Path(__file__).resolve().parent.parent
    if resolved == project_root:
        return False
    if (resolved / "pyproject.toml").is_file() and (resolved / "podclean").is_dir():
        return False
    return True


def _is_leftover_file(path: Path) -> bool:
    """Return True if *path* is a local leftover that is safe to delete."""
    name = path.name
    if name in _PRESERVE_NAMES:
        return False
    if name.startswith(".env"):
        return False

    suffix = path.suffix.lower()
    if suffix == ".json" and name.lower().startswith(_PRESERVE_JSON_PREFIXES):
        return False

    if suffix in _AUDIO_SUFFIXES:
        return True
    if suffix in _SCRATCH_SUFFIXES:
        return True
    return False

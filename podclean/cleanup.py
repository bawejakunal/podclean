"""Per-run cleanup for PodClean's ``--cleanup`` flag.

A command records each file it creates in :class:`RunArtifacts`; ``--cleanup``
deletes only those.  Nothing else on disk is scanned or touched, and S3 is
never modified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunArtifacts:
    """Files one CLI invocation created; only these are eligible for --cleanup."""

    scratch: list[Path] = field(default_factory=list)
    """Downloads and transcripts; safe to delete once the run is over."""

    output: Path | None = None
    """Cleaned audio; deleted only after it has been uploaded."""

    uploaded: bool = False

    def add_scratch(self, path: Path) -> None:
        self.scratch.append(path)

    def removable(self) -> list[Path]:
        """Paths ``--cleanup`` should delete for this run."""
        paths = list(self.scratch)
        if self.output is not None and self.uploaded:
            paths.append(self.output)
        return paths

    @property
    def kept_output(self) -> Path | None:
        """The cleaned output when it is kept because it was not uploaded."""
        if self.output is not None and not self.uploaded:
            return self.output
        return None


@dataclass(frozen=True)
class CleanupSummary:
    """Counts from deleting a run's files."""

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


def remove_files(paths: Iterable[Path]) -> CleanupSummary:
    """Delete *paths*, skipping any that are already gone.

    Paths may already have been removed by the pipeline's own cache cleanup,
    so missing files are not errors.  Any other ``OSError`` is counted.
    """
    files_removed = 0
    bytes_freed = 0
    errors = 0

    for path in dict.fromkeys(paths):
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

    return CleanupSummary(
        files_removed=files_removed,
        bytes_freed=bytes_freed,
        errors=errors,
    )

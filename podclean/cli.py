"""CLI entry point for PodClean — podcast ad removal tool."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from podclean.config import get_config, Config
from podclean.models import AdRegion, ProcessingResult

console = Console()


def _print_banner() -> None:
    """Print the PodClean ASCII banner."""
    banner = Text()
    banner.append("🎧 ", style="bold")
    banner.append("Pod", style="bold cyan")
    banner.append("Clean", style="bold green")
    banner.append(" — Podcast Ad Remover", style="dim")
    console.print(Panel(banner, border_style="dim cyan", padding=(0, 2)))


def _print_ad_table(ad_regions: list[AdRegion]) -> None:
    """Print a rich table of detected ad regions."""
    table = Table(
        title="🔍 Detected Ad Segments",
        title_style="bold yellow",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Start", style="cyan", width=8)
    table.add_column("End", style="cyan", width=8)
    table.add_column("Duration", style="magenta", width=10)
    table.add_column("Confidence", width=12, justify="center")
    table.add_column("Reason", style="white", max_width=40)

    for i, ad in enumerate(ad_regions, 1):
        # Format timestamps as MM:SS
        start_m, start_s = divmod(int(ad.start), 60)
        end_m, end_s = divmod(int(ad.end), 60)
        dur = ad.duration

        # Color confidence
        if ad.confidence >= 0.8:
            conf_style = "bold green"
        elif ad.confidence >= 0.6:
            conf_style = "yellow"
        else:
            conf_style = "red"

        table.add_row(
            str(i),
            f"{start_m:02d}:{start_s:02d}",
            f"{end_m:02d}:{end_s:02d}",
            f"{dur:.0f}s",
            Text(f"{ad.confidence:.0%}", style=conf_style),
            ad.reason[:40],
        )

    console.print(table)


def _print_result(result: ProcessingResult) -> None:
    """Print processing results."""
    console.print()
    console.print(Panel(
        result.summary(),
        title="✅ Processing Complete",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    ))


def _run_pipeline(
    audio_path: Path,
    output: str | None,
    model: str | None,
    preview: bool,
    config: Config,
) -> None:
    """Run the full ad detection and removal pipeline."""
    from podclean.transcriber import Transcriber
    from podclean.detector import AdDetector
    from podclean.processor import AudioProcessor

    # --- Step 1: Transcribe ---
    console.print()
    console.print("[bold cyan]Step 1/3:[/] Transcribing audio...", highlight=False)
    console.print(f"  Model: [yellow]{config.whisper_model}[/]  File: [dim]{audio_path.name}[/]")
    console.print()

    t0 = time.time()
    transcriber = Transcriber(model_size=model)
    segments = transcriber.transcribe(audio_path)
    t_transcribe = time.time() - t0

    total_words = sum(len(s.words) for s in segments)
    console.print(
        f"  ✓ Transcribed [green]{len(segments)}[/] segments, "
        f"[green]{total_words:,}[/] words "
        f"in [cyan]{t_transcribe:.1f}s[/]"
    )

    # --- Step 2: Detect Ads ---
    console.print()
    console.print("[bold cyan]Step 2/3:[/] Detecting advertisements...", highlight=False)
    console.print()

    t0 = time.time()
    detector = AdDetector()
    ad_regions = detector.detect_ads(segments)
    t_detect = time.time() - t0

    console.print(
        f"  ✓ Found [yellow]{len(ad_regions)}[/] ad segments "
        f"in [cyan]{t_detect:.1f}s[/]"
    )

    if not ad_regions:
        console.print()
        console.print("[green]No ads detected![/] Your episode is clean. 🎉")
        return

    _print_ad_table(ad_regions)

    # --- Step 3: Process Audio ---
    console.print()
    processor = AudioProcessor()

    if preview:
        console.print("[bold cyan]Step 3/3:[/] Preview mode (no audio changes)", highlight=False)
        result = processor.preview(audio_path, ad_regions)
    else:
        console.print("[bold cyan]Step 3/3:[/] Removing ads and generating clean audio...", highlight=False)
        console.print()

        output_path = Path(output) if output else None
        t0 = time.time()
        result = processor.process(audio_path, ad_regions, output_path=output_path)
        t_process = time.time() - t0
        console.print(f"  ✓ Audio processed in [cyan]{t_process:.1f}s[/]")

    _print_result(result)


@click.group()
@click.version_option(version="0.1.0", prog_name="podclean")
def cli() -> None:
    """🎧 PodClean — Automatically remove spoken ads from podcast episodes."""
    pass


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=str, default=None, help="Output file path")
@click.option("-m", "--model", type=str, default=None,
              help="MLX Whisper model repo (default: mlx-community/whisper-large-v3-turbo)")
@click.option("--preview", is_flag=True, help="Preview detected ads without processing audio")
@click.option("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY)")
def file(audio_file: Path, output: str | None, model: str | None, preview: bool, api_key: str | None) -> None:
    """Process a local audio file to remove ads.

    Example: podclean file episode.mp3
    """
    _print_banner()

    # Build config with overrides
    overrides: dict = {}
    if api_key:
        overrides["gemini_api_key"] = api_key
    if model:
        overrides["whisper_model"] = model
    config = get_config(**overrides)

    # Validate config
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"[red]✗ Config error:[/] {err}")
        sys.exit(1)

    console.print(f"Processing: [bold]{audio_file.name}[/]")
    _run_pipeline(audio_file, output, model, preview, config)


@cli.command()
@click.argument("rss_url", type=str)
@click.option("-o", "--output", type=str, default=None, help="Output file path")
@click.option("-m", "--model", type=str, default=None,
              help="MLX Whisper model repo (default: mlx-community/whisper-large-v3-turbo)")
@click.option("-n", "--episode-num", type=int, default=1,
              help="Episode number to process (1 = latest, default: 1)")
@click.option("--preview", is_flag=True, help="Preview detected ads without processing audio")
@click.option("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY)")
def feed(rss_url: str, output: str | None, model: str | None, episode_num: int,
         preview: bool, api_key: str | None) -> None:
    """Process a podcast episode from an RSS feed.

    Example: podclean feed "https://rss.art19.com/the-daily-stoic"
    """
    from podclean.fetcher import PodcastFetcher

    _print_banner()

    # Build config with overrides
    overrides: dict = {}
    if api_key:
        overrides["gemini_api_key"] = api_key
    if model:
        overrides["whisper_model"] = model
    config = get_config(**overrides)

    # Validate config
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"[red]✗ Config error:[/] {err}")
        sys.exit(1)

    # Fetch episode
    fetcher = PodcastFetcher()
    console.print(f"Fetching feed: [dim]{rss_url}[/]")

    episodes = fetcher.list_episodes(rss_url)
    if not episodes:
        console.print("[red]✗ No episodes found in feed.[/]")
        sys.exit(1)

    if episode_num < 1 or episode_num > len(episodes):
        console.print(f"[red]✗ Episode #{episode_num} not found. Feed has {len(episodes)} episodes.[/]")
        sys.exit(1)

    episode = episodes[episode_num - 1]
    console.print(f"Episode: [bold]{episode.title}[/]")
    console.print(f"Published: [dim]{episode.published}[/]")

    audio_path = fetcher.download_episode(episode)
    console.print(f"Downloaded to: [dim]{audio_path}[/]")

    _run_pipeline(audio_path, output, model, preview, config)


@cli.command(name="list")
@click.argument("rss_url", type=str)
@click.option("-n", "--limit", type=int, default=20, help="Number of episodes to show")
def list_episodes(rss_url: str, limit: int) -> None:
    """List recent episodes from an RSS feed.

    Example: podclean list "https://rss.art19.com/the-daily-stoic"
    """
    from podclean.fetcher import PodcastFetcher

    _print_banner()

    fetcher = PodcastFetcher()
    console.print(f"Fetching feed: [dim]{rss_url}[/]")

    episodes = fetcher.list_episodes(rss_url, limit=limit)
    if not episodes:
        console.print("[red]✗ No episodes found in feed.[/]")
        sys.exit(1)

    table = Table(
        title="📻 Episodes",
        title_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", style="white", max_width=60)
    table.add_column("Published", style="cyan", width=14)
    table.add_column("Duration", style="magenta", width=10)

    for i, ep in enumerate(episodes, 1):
        table.add_row(str(i), ep.title, ep.published, ep.duration)

    console.print(table)
    console.print()
    console.print(
        "[dim]Use[/] [cyan]podclean feed <url> -n <number>[/] "
        "[dim]to process a specific episode[/]"
    )


if __name__ == "__main__":
    cli()

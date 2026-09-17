"""Module for uploading files to AWS S3 and managing RSS feeds."""

from __future__ import annotations

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import get_config
from .rss import (
    DEFAULT_CHANNEL_DESCRIPTION,
    DEFAULT_CHANNEL_TITLE,
    DEFAULT_MAX_FEED_ITEMS,
    RssItem,
    audio_mime_type,
    build_rss_xml,
    format_itunes_duration,
    limit_feed_items,
    parse_rss_feed,
    upsert_item,
)


def upload_to_s3(
    file_path: Path,
    episode_title: str | None = None,
    duration_seconds: float | None = None,
) -> tuple[str | None, str | None]:
    """Upload a file to S3 and update the RSS feed.

    Returns a tuple of (rss_url, file_url).
    Returns (None, None) if upload fails or credentials are not set.

    ``duration_seconds`` is optional and backward compatible. When omitted,
    duration is probed from the local file with ffprobe if available.
    """
    config = get_config()

    if not config.aws_access_key_id or not config.aws_secret_access_key or not config.aws_bucket_name:
        print("S3 credentials not fully configured. Skipping upload.")
        return None, None

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region_name,
    )

    bucket = config.aws_bucket_name
    prefix = "podclean/output/"
    object_name = f"{prefix}{file_path.name}"
    rss_object_name = f"{prefix}rss.xml"

    region = config.aws_region_name
    base_url = f"https://{bucket}.s3.{region}.amazonaws.com"
    file_url = f"{base_url}/{object_name}"
    rss_url = f"{base_url}/{rss_object_name}"

    try:
        print(f"Uploading {file_path.name} to S3 bucket {bucket} under prefix {prefix}...")
        s3_client.upload_file(
            str(file_path),
            bucket,
            object_name,
            ExtraArgs={"ContentType": audio_mime_type(file_path.name)},
        )
        print("File uploaded successfully.")

        title = episode_title or file_path.stem
        if duration_seconds is None:
            duration_seconds = _probe_audio_duration_seconds(file_path)

        _update_rss_feed(
            s3_client,
            bucket,
            rss_object_name,
            file_url,
            file_path.stat().st_size,
            title,
            feed_url=rss_url,
            duration_seconds=duration_seconds,
            mime_type=audio_mime_type(file_path.name),
            max_items=config.rss_max_items,
        )

        return rss_url, file_url

    except NoCredentialsError:
        print("Credentials not available.")
        return None, None
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return None, None


def _update_rss_feed(
    s3_client,
    bucket: str,
    rss_key: str,
    file_url: str,
    file_size: int,
    title: str,
    feed_url: str,
    duration_seconds: float | None = None,
    mime_type: str = "audio/mpeg",
    max_items: int | None = DEFAULT_MAX_FEED_ITEMS,
) -> None:
    """Download the existing RSS feed, rewrite it with the new episode, and upload.

    Existing sparse feeds are upgraded in place: items are reordered newest-first,
    missing GUIDs are filled from enclosure URLs, and podcast/iTunes metadata is
    added. Channel ``<link>`` is set to the feed URL rather than an audio file.

    Only the newest ``max_items`` episodes are written back to S3 so the XML
    stays small enough for Apple Podcasts and YouTube Music to fetch quickly.
    Pass ``0`` or ``None`` to keep the full catalog.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        channel_title, channel_description, items = _load_existing_feed(
            s3_client, bucket, rss_key, tmp_path
        )

        new_item = RssItem(
            title=title,
            enclosure_url=file_url,
            enclosure_length=str(file_size),
            enclosure_type=mime_type,
            guid=file_url,
            duration=format_itunes_duration(duration_seconds),
            description=title,
        )
        items = upsert_item(items, new_item)
        published = limit_feed_items(items, max_items)
        omitted = len(items) - len(published)
        xml_bytes = build_rss_xml(
            published,
            feed_url=feed_url,
            title=channel_title,
            description=channel_description,
        )
        Path(tmp_path).write_bytes(xml_bytes)

        s3_client.upload_file(
            tmp_path,
            bucket,
            rss_key,
            ExtraArgs={
                "ContentType": "application/rss+xml",
                "CacheControl": "max-age=0, must-revalidate",
            },
        )
        if omitted:
            print(
                f"RSS feed updated successfully ({len(published)} episodes, "
                f"{len(xml_bytes)} bytes); omitted {omitted} older items "
                f"(RSS_MAX_ITEMS={max_items})."
            )
        else:
            print(
                f"RSS feed updated successfully "
                f"({len(published)} episodes, {len(xml_bytes)} bytes)."
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_existing_feed(
    s3_client, bucket: str, rss_key: str, tmp_path: str
) -> tuple[str, str, list[RssItem]]:
    """Return channel metadata and items from S3, or defaults if none exists."""
    try:
        s3_client.download_file(bucket, rss_key, tmp_path)
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey"}:
            return DEFAULT_CHANNEL_TITLE, DEFAULT_CHANNEL_DESCRIPTION, []
        raise

    try:
        return parse_rss_feed(Path(tmp_path).read_bytes())
    except ET.ParseError:
        print("Existing RSS feed could not be parsed; rebuilding from the new episode.")
        return DEFAULT_CHANNEL_TITLE, DEFAULT_CHANNEL_DESCRIPTION, []


def _probe_audio_duration_seconds(file_path: Path) -> float | None:
    """Return audio duration in seconds using ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None

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
    DEFAULT_MAX_FEED_BYTES,
    DEFAULT_MAX_FEED_ITEMS,
    RssItem,
    audio_mime_type,
    build_public_rss_xml,
    build_rss_xml,
    format_itunes_duration,
    merge_missing_items,
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
            max_bytes=config.rss_max_bytes,
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
    max_bytes: int | None = DEFAULT_MAX_FEED_BYTES,
) -> None:
    """Download the existing catalog, rewrite it with the new episode, and upload.

    Existing sparse feeds are upgraded in place: items are reordered newest-first,
    missing GUIDs are filled from enclosure URLs, and podcast/iTunes metadata is
    added. Channel ``<link>`` is set to the feed URL rather than an audio file.

    The uncapped catalog is stored next to the public feed (``rss-catalog.xml``)
    so omitted episodes can be republished if ``RSS_MAX_ITEMS`` is raised later.
    The public ``rss.xml`` is written first and clipped to *max_items* plus a
    *max_bytes* budget so Apple Podcasts and YouTube Music can fetch it quickly.
    Pass ``0`` or ``None`` for *max_items*/*max_bytes* to skip that cap.
    """
    catalog_key = _catalog_key(rss_key)
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
        published, published_xml = build_public_rss_xml(
            items,
            feed_url=feed_url,
            title=channel_title,
            description=channel_description,
            max_items=max_items,
            max_bytes=max_bytes,
        )
        omitted = len(items) - len(published)
        catalog_xml = build_rss_xml(
            items,
            feed_url=feed_url,
            title=channel_title,
            description=channel_description,
        )
        _put_rss_object(s3_client, bucket, rss_key, published_xml, tmp_path)
        _put_catalog_object(s3_client, bucket, catalog_key, catalog_xml, tmp_path)
        if omitted:
            print(
                f"RSS feed updated successfully ({len(published)} episodes, "
                f"{len(published_xml)} bytes); omitted {omitted} older items "
                f"(RSS_MAX_ITEMS={max_items}, RSS_MAX_BYTES={max_bytes})."
            )
        else:
            print(
                f"RSS feed updated successfully "
                f"({len(published)} episodes, {len(published_xml)} bytes)."
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _catalog_key(rss_key: str) -> str:
    """Return the S3 key for the uncapped catalog beside the public feed."""
    if rss_key.endswith(".xml"):
        return f"{rss_key[:-4]}-catalog.xml"
    return f"{rss_key}-catalog.xml"


_RSS_UPLOAD_ARGS: dict[str, str] = {
    "ContentType": "application/rss+xml",
    "CacheControl": "max-age=0, must-revalidate",
}


def _put_rss_object(
    s3_client, bucket: str, key: str, xml_bytes: bytes, tmp_path: str
) -> None:
    Path(tmp_path).write_bytes(xml_bytes)
    s3_client.upload_file(tmp_path, bucket, key, ExtraArgs=_RSS_UPLOAD_ARGS)


def _put_catalog_object(
    s3_client, bucket: str, key: str, xml_bytes: bytes, tmp_path: str
) -> None:
    """Write the uncapped catalog; a permission error must not block rss.xml."""
    try:
        _put_rss_object(s3_client, bucket, key, xml_bytes, tmp_path)
    except ClientError as exc:
        error = exc.response.get("Error", {}) if exc.response else {}
        code = str(error.get("Code") or "")
        metadata = exc.response.get("ResponseMetadata") if exc.response else None
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        detail = code or (f"HTTP {status}" if status else str(exc))
        print(
            f"Warning: catalog PutObject denied for {key} ({detail}). "
            "Public rss.xml was still updated. Grant s3:PutObject for both "
            "rss.xml and rss-catalog.xml under podclean/output/ so omitted "
            "episodes stay restorable."
        )


def _load_existing_feed(
    s3_client, bucket: str, rss_key: str, tmp_path: str
) -> tuple[str, str, list[RssItem]]:
    """Return channel metadata and items from the catalog merged with the public feed.

    Prefers ``rss-catalog.xml`` so a later raise of ``RSS_MAX_ITEMS`` can
    republish episodes that were omitted from the player-facing feed.
    Public-only episodes are merged in so a failed catalog write cannot drop
    items that already made it into ``rss.xml``.
    """
    catalog = _try_load_feed(s3_client, bucket, _catalog_key(rss_key), tmp_path)
    public = _try_load_feed(s3_client, bucket, rss_key, tmp_path)
    if catalog is None and public is None:
        return DEFAULT_CHANNEL_TITLE, DEFAULT_CHANNEL_DESCRIPTION, []
    if catalog is None:
        return public
    if public is None:
        return catalog
    title, description, items = catalog
    _, _, public_items = public
    return title, description, merge_missing_items(items, public_items)


def _try_load_feed(
    s3_client, bucket: str, key: str, tmp_path: str
) -> tuple[str, str, list[RssItem]] | None:
    try:
        s3_client.download_file(bucket, key, tmp_path)
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey"}:
            return None
        raise

    try:
        return parse_rss_feed(Path(tmp_path).read_bytes())
    except ET.ParseError:
        print(f"Existing RSS object {key} could not be parsed; trying a fallback.")
        return None


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

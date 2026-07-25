"""Module for uploading files to AWS S3 and managing RSS feeds."""

import tempfile
import xml.etree.ElementTree as ET
from email.utils import formatdate
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError

from .config import get_config


def upload_to_s3(file_path: Path, episode_title: str | None = None) -> tuple[str | None, str | None]:
    """Upload a file to S3 and update the RSS feed.
    
    Returns a tuple of (rss_url, file_url).
    Returns (None, None) if upload fails or credentials are not set.
    """
    config = get_config()
    
    if not config.aws_access_key_id or not config.aws_secret_access_key or not config.aws_bucket_name:
        print("S3 credentials not fully configured. Skipping upload.")
        return None, None

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region_name
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
            ExtraArgs={'ContentType': 'audio/mpeg'}
        )
        print("File uploaded successfully.")
        
        title = episode_title or file_path.stem
        _update_rss_feed(s3_client, bucket, rss_object_name, file_url, file_path.stat().st_size, title)
        
        return rss_url, file_url
        
    except NoCredentialsError:
        print("Credentials not available.")
        return None, None
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return None, None


def _update_rss_feed(s3_client, bucket: str, rss_key: str, file_url: str, file_size: int, title: str) -> None:
    """Download existing RSS feed, append the new item, and upload it back."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        s3_client.download_file(bucket, rss_key, tmp_path)
        tree = ET.parse(tmp_path)
        channel = tree.find('channel')
        if channel is None:
            raise ValueError("Invalid RSS feed structure")
    except Exception:
        # Create new RSS structure
        rss = ET.Element("rss", version="2.0", **{"xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"})
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = "PodClean Audio Feed"
        ET.SubElement(channel, "description").text = "Automated ad-free podcast episodes"
        ET.SubElement(channel, "link").text = file_url
        tree = ET.ElementTree(rss)

    # Add new item
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "pubDate").text = formatdate(localtime=False)
    
    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set("url", file_url)
    enclosure.set("length", str(file_size))
    enclosure.set("type", "audio/mpeg")
    
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    
    s3_client.upload_file(
        tmp_path, 
        bucket, 
        rss_key,
        ExtraArgs={'ContentType': 'application/rss+xml'}
    )
    print("RSS feed updated successfully.")

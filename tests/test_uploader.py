"""Tests for S3 RSS update behavior using a fake S3 client."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import TestCase

from botocore.exceptions import ClientError

from podclean.rss import RssItem, build_rss_xml
from podclean.uploader import _update_rss_feed
from tests.test_rss import FEED_URL, LEGACY_FEED


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.uploads: list[tuple[str, bytes, dict | None]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        Path(filename).write_bytes(self.objects[key])

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict | None = None,
    ) -> None:
        data = Path(filename).read_bytes()
        self.objects[key] = data
        self.uploads.append((key, data, ExtraArgs))


class UpdateRssFeedTest(TestCase):
    def test_creates_feed_when_none_exists(self) -> None:
        s3 = FakeS3()
        _update_rss_feed(
            s3,
            "bucket",
            "podclean/output/rss.xml",
            "https://example.com/podclean/output/ep.mp3",
            1234,
            "First Episode",
            feed_url="https://example.com/podclean/output/rss.xml",
            duration_seconds=90,
        )
        self.assertEqual(len(s3.uploads), 1)
        key, xml, extra = s3.uploads[0]
        self.assertEqual(key, "podclean/output/rss.xml")
        assert extra is not None
        self.assertEqual(extra["ContentType"], "application/rss+xml")
        self.assertEqual(extra["CacheControl"], "max-age=0, must-revalidate")

        channel = ET.fromstring(xml).find("channel")
        assert channel is not None
        self.assertEqual(channel.findtext("link"), "https://example.com/podclean/output/rss.xml")
        items = channel.findall("item")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].findtext("guid"), "https://example.com/podclean/output/ep.mp3")
        self.assertEqual(
            items[0].findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"),
            "1:30",
        )

    def test_upgrades_legacy_feed_in_place(self) -> None:
        s3 = FakeS3({"podclean/output/rss.xml": LEGACY_FEED.encode("utf-8")})
        _update_rss_feed(
            s3,
            "bucket",
            "podclean/output/rss.xml",
            "https://example.com/podclean/output/brand_new_clean.mp3",
            999,
            "Brand New",
            feed_url="https://example.com/podclean/output/rss.xml",
            duration_seconds=45,
        )
        xml = s3.objects["podclean/output/rss.xml"]
        channel = ET.fromstring(xml).find("channel")
        assert channel is not None
        items = channel.findall("item")
        titles = [item.findtext("title") for item in items]
        self.assertEqual(
            titles,
            ["Brand New", "Newest Episode", "Middle Episode", "Oldest Episode"],
        )
        self.assertEqual(channel.findtext("link"), "https://example.com/podclean/output/rss.xml")
        for item in items:
            guid = item.find("guid")
            enclosure = item.find("enclosure")
            assert guid is not None and enclosure is not None
            self.assertEqual(guid.text, enclosure.get("url"))

    def test_caps_published_feed_to_newest_items(self) -> None:
        existing = [
            RssItem(
                title="Oldest",
                enclosure_url="https://example.com/podclean/output/oldest.mp3",
                enclosure_length="1",
                pub_date="Sat, 01 Aug 2026 00:00:00 GMT",
            ),
            RssItem(
                title="Middle",
                enclosure_url="https://example.com/podclean/output/middle.mp3",
                enclosure_length="2",
                pub_date="Sat, 01 Sep 2026 00:00:00 GMT",
            ),
            RssItem(
                title="Newest",
                enclosure_url="https://example.com/podclean/output/newest.mp3",
                enclosure_length="3",
                pub_date="Sat, 10 Sep 2026 00:00:00 GMT",
            ),
        ]
        s3 = FakeS3(
            {
                "podclean/output/rss.xml": build_rss_xml(
                    existing, feed_url=FEED_URL
                )
            }
        )
        _update_rss_feed(
            s3,
            "bucket",
            "podclean/output/rss.xml",
            "https://example.com/podclean/output/brand_new_clean.mp3",
            999,
            "Brand New",
            feed_url=FEED_URL,
            duration_seconds=45,
            max_items=2,
        )
        xml = s3.objects["podclean/output/rss.xml"]
        channel = ET.fromstring(xml).find("channel")
        assert channel is not None
        titles = [item.findtext("title") for item in channel.findall("item")]
        self.assertEqual(titles, ["Brand New", "Newest"])

    def test_zero_max_items_keeps_full_catalog(self) -> None:
        existing = [
            RssItem(
                title=f"Episode {i}",
                enclosure_url=f"https://example.com/podclean/output/ep{i}.mp3",
                enclosure_length="1",
                pub_date=f"Mon, {i:02d} Sep 2026 00:00:00 GMT",
            )
            for i in range(1, 5)
        ]
        s3 = FakeS3(
            {
                "podclean/output/rss.xml": build_rss_xml(
                    existing, feed_url=FEED_URL
                )
            }
        )
        _update_rss_feed(
            s3,
            "bucket",
            "podclean/output/rss.xml",
            "https://example.com/podclean/output/ep5.mp3",
            5,
            "Episode 5",
            feed_url=FEED_URL,
            max_items=0,
        )
        xml = s3.objects["podclean/output/rss.xml"]
        channel = ET.fromstring(xml).find("channel")
        assert channel is not None
        self.assertEqual(len(channel.findall("item")), 5)

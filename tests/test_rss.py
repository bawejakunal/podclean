"""Unit tests for podcast RSS generation, ordering, GUIDs, and metadata."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest import TestCase

from podclean.rss import (
    ATOM_NS,
    ITUNES_NS,
    RssItem,
    audio_mime_type,
    build_rss_xml,
    format_itunes_duration,
    parse_rss_feed,
    sort_items_newest_first,
    upsert_item,
)

# Sparse live-style feed: oldest-first items, no guid, channel <link> is an MP3.
LEGACY_FEED = """\
<?xml version='1.0' encoding='utf-8'?>
<rss version="2.0">
  <channel>
    <title>PodClean Audio Feed</title>
    <description>Automated ad-free podcast episodes</description>
    <link>https://example.com/podclean/output/oldest_clean.mp3</link>
    <item>
      <title>Oldest Episode</title>
      <pubDate>Sat, 25 Jul 2026 19:15:20 -0000</pubDate>
      <enclosure url="https://example.com/podclean/output/oldest_clean.mp3" length="100" type="audio/mpeg" />
    </item>
    <item>
      <title>Middle Episode</title>
      <pubDate>Sat, 12 Sep 2026 05:54:07 -0000</pubDate>
      <enclosure url="https://example.com/podclean/output/middle_clean.mp3" length="200" type="audio/mpeg" />
    </item>
    <item>
      <title>Newest Episode</title>
      <pubDate>Tue, 15 Sep 2026 15:42:10 -0000</pubDate>
      <enclosure url="https://example.com/podclean/output/newest_clean.mp3" length="300" type="audio/mpeg" />
    </item>
  </channel>
</rss>
"""

FEED_URL = "https://example.com/podclean/output/rss.xml"


def _localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _channel(xml: bytes) -> ET.Element:
    root = ET.fromstring(xml)
    channel = root.find("channel")
    assert channel is not None
    return channel


def _items(xml: bytes) -> list[ET.Element]:
    return list(_channel(xml).findall("item"))


def _item_text(item: ET.Element, name: str) -> str | None:
    for child in item:
        if _localname(child.tag) == name:
            return child.text
    return None


class FormatHelpersTest(TestCase):
    def test_itunes_duration_formats_hours_and_minutes(self) -> None:
        self.assertEqual(format_itunes_duration(75), "1:15")
        self.assertEqual(format_itunes_duration(3661), "1:01:01")
        self.assertIsNone(format_itunes_duration(None))
        self.assertIsNone(format_itunes_duration(-3))

    def test_audio_mime_type_by_extension(self) -> None:
        self.assertEqual(audio_mime_type("ep.mp3"), "audio/mpeg")
        self.assertEqual(audio_mime_type("ep.m4a"), "audio/mp4")
        self.assertEqual(audio_mime_type("ep.wav"), "audio/wav")
        self.assertEqual(audio_mime_type("ep.bin"), "audio/mpeg")


class ParseAndUpgradeTest(TestCase):
    def test_parse_fills_guid_from_enclosure_url(self) -> None:
        title, description, items = parse_rss_feed(LEGACY_FEED)
        self.assertEqual(title, "PodClean Audio Feed")
        self.assertEqual(description, "Automated ad-free podcast episodes")
        self.assertEqual(len(items), 3)
        self.assertEqual(
            items[0].guid,
            "https://example.com/podclean/output/oldest_clean.mp3",
        )
        channel = ET.fromstring(LEGACY_FEED).find("channel")
        assert channel is not None
        raw_item = channel.find("item")
        assert raw_item is not None
        self.assertIsNone(raw_item.findtext("guid"))

    def test_upgrade_rewrites_newest_first_with_guid_and_itunes(self) -> None:
        title, description, items = parse_rss_feed(LEGACY_FEED)
        new_item = RssItem(
            title="Just Uploaded",
            enclosure_url="https://example.com/podclean/output/just_uploaded_clean.mp3",
            enclosure_length="400",
            pub_date="Wed, 16 Sep 2026 05:00:00 GMT",
            duration="12:34",
        )
        xml = build_rss_xml(
            upsert_item(items, new_item),
            feed_url=FEED_URL,
            title=title,
            description=description,
        )
        channel = _channel(xml)
        item_els = _items(xml)

        self.assertEqual(channel.findtext("link"), FEED_URL)
        self.assertIsNotNone(channel.findtext("lastBuildDate"))
        self.assertIsNotNone(channel.findtext("language"))

        atom_link = channel.find(f"{{{ATOM_NS}}}link")
        assert atom_link is not None
        self.assertEqual(atom_link.get("rel"), "self")
        self.assertEqual(atom_link.get("href"), FEED_URL)
        self.assertEqual(atom_link.get("type"), "application/rss+xml")

        self.assertEqual(channel.findtext(f"{{{ITUNES_NS}}}author"), "PodClean")
        self.assertEqual(channel.findtext(f"{{{ITUNES_NS}}}explicit"), "false")
        itunes_category = channel.find(f"{{{ITUNES_NS}}}category")
        assert itunes_category is not None
        self.assertEqual(itunes_category.get("text"), "Education")

        titles = [_item_text(item, "title") for item in item_els]
        self.assertEqual(
            titles,
            ["Just Uploaded", "Newest Episode", "Middle Episode", "Oldest Episode"],
        )

        for item in item_els:
            guid = item.find("guid")
            enclosure = item.find("enclosure")
            assert guid is not None and enclosure is not None
            self.assertEqual(guid.text, enclosure.get("url"))
            self.assertEqual(guid.get("isPermaLink"), "true")
            self.assertEqual(enclosure.get("type"), "audio/mpeg")
            self.assertIsNotNone(enclosure.get("length"))
            self.assertEqual(item.findtext(f"{{{ITUNES_NS}}}author"), "PodClean")
            self.assertEqual(item.findtext(f"{{{ITUNES_NS}}}explicit"), "false")

        self.assertEqual(
            item_els[0].findtext(f"{{{ITUNES_NS}}}duration"),
            "12:34",
        )

    def test_namespaces_survive_round_trip(self) -> None:
        _, _, items = parse_rss_feed(LEGACY_FEED)
        built = build_rss_xml(items, feed_url=FEED_URL)
        _, _, parsed = parse_rss_feed(built)
        rebuilt = build_rss_xml(parsed, feed_url=FEED_URL)
        titles = [_item_text(item, "title") for item in _items(rebuilt)]
        self.assertEqual(titles, ["Newest Episode", "Middle Episode", "Oldest Episode"])
        self.assertIsNotNone(_items(rebuilt)[0].find("guid"))
        self.assertIsNotNone(_channel(rebuilt).find(f"{{{ATOM_NS}}}link"))

    def test_upsert_replaces_same_enclosure_instead_of_duplicating(self) -> None:
        items = [
            RssItem(
                title="Original",
                enclosure_url="https://example.com/a.mp3",
                enclosure_length="1",
                pub_date="Mon, 14 Sep 2026 00:00:00 GMT",
            )
        ]
        updated = upsert_item(
            items,
            RssItem(
                title="Reprocessed",
                enclosure_url="https://example.com/a.mp3",
                enclosure_length="2",
                pub_date="Wed, 16 Sep 2026 00:00:00 GMT",
            ),
        )
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].title, "Reprocessed")
        self.assertEqual(updated[0].enclosure_length, "2")


class OrderingTest(TestCase):
    def test_sort_newest_first_handles_rfc822_variants(self) -> None:
        items = [
            RssItem(
                title="older",
                enclosure_url="https://example.com/old.mp3",
                enclosure_length="1",
                pub_date="Sat, 25 Jul 2026 19:15:20 -0000",
            ),
            RssItem(
                title="newer",
                enclosure_url="https://example.com/new.mp3",
                enclosure_length="1",
                pub_date="Tue, 15 Sep 2026 15:42:10 GMT",
            ),
        ]
        ordered = sort_items_newest_first(items)
        self.assertEqual([item.title for item in ordered], ["newer", "older"])

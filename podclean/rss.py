"""Build and upgrade PodClean podcast RSS feeds.

Podcast clients (YouTube Music included) expect newest-first item order,
a stable unique ``<guid>`` per episode, and enough iTunes/podcast metadata
to discover updates. Existing sparse feeds are rewritten in place so
subscribers do not have to re-subscribe.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from typing import Final

ITUNES_NS: Final = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ATOM_NS: Final = "http://www.w3.org/2005/Atom"

DEFAULT_CHANNEL_TITLE: Final = "PodClean Audio Feed"
DEFAULT_CHANNEL_DESCRIPTION: Final = "Automated ad-free podcast episodes"
DEFAULT_CHANNEL_AUTHOR: Final = "PodClean"
DEFAULT_LANGUAGE: Final = "en"
DEFAULT_ITUNES_CATEGORY: Final = "Education"
# Newest-first window published in rss.xml. Item count alone is not a size
# bound: inherited show notes can still blow past aggregator timeouts.
DEFAULT_MAX_FEED_ITEMS: Final = 300
# Player-facing serialized size. Many aggregators time out around 512 KiB.
DEFAULT_MAX_FEED_BYTES: Final = 512 * 1024
# Public-feed item text. Catalog (rss-catalog.xml) keeps the full values.
DEFAULT_MAX_ITEM_DESCRIPTION_CHARS: Final = 500
DEFAULT_MAX_ITEM_TITLE_CHARS: Final = 255
DEFAULT_MAX_TITLE_CHARS: Final = DEFAULT_MAX_ITEM_TITLE_CHARS

_AUDIO_MIME_TYPES: Final = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
}


@dataclass(frozen=True)
class RssItem:
    """One podcast episode in the feed."""

    title: str
    enclosure_url: str
    enclosure_length: str
    enclosure_type: str = "audio/mpeg"
    pub_date: str = ""
    guid: str = ""
    duration: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.guid:
            object.__setattr__(self, "guid", self.enclosure_url)
        if not self.pub_date:
            object.__setattr__(self, "pub_date", formatdate(usegmt=True))
        if self.description is None:
            object.__setattr__(self, "description", self.title)


def audio_mime_type(filename: str) -> str:
    """Return a podcast-appropriate MIME type for an audio filename."""
    suffix = ""
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    return _AUDIO_MIME_TYPES.get(suffix, "audio/mpeg")


def format_itunes_duration(seconds: float | int | None) -> str | None:
    """Format a duration as ``H:MM:SS`` or ``M:SS`` for ``itunes:duration``."""
    if seconds is None:
        return None
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return None
    if total < 0:
        return None
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_rss_feed(xml: str | bytes) -> tuple[str, str, list[RssItem]]:
    """Parse channel title, description, and items from an existing RSS document.

    Missing ``<guid>`` values are filled from the enclosure URL so existing
    episodes keep a stable identity after the feed is rewritten.

    When *xml* is a ``str``, it is passed directly to the parser; when it is
    ``bytes``, the parser honours the encoding declared in the XML prolog.
    """
    root = ET.fromstring(xml)
    channel = _find_child(root, "channel")
    if channel is None:
        if _localname(root.tag) == "channel":
            channel = root
        else:
            return DEFAULT_CHANNEL_TITLE, DEFAULT_CHANNEL_DESCRIPTION, []

    title = _child_text(channel, "title") or DEFAULT_CHANNEL_TITLE
    description = _child_text(channel, "description") or DEFAULT_CHANNEL_DESCRIPTION
    items = [_item_from_element(el) for el in _iter_children(channel, "item")]
    items = [item for item in items if item.enclosure_url]
    return title, description, items


def upsert_item(items: Sequence[RssItem], new_item: RssItem) -> list[RssItem]:
    """Insert or replace an episode, identified by guid or enclosure URL.

    When multiple existing items match *new_item*, the replacement is appended
    only for the first match; subsequent matching duplicates are dropped.

    When an existing episode is matched solely by enclosure URL (not by equal
    GUID), the replacement retains the existing item's GUID to preserve custom
    stable identifiers during re-upload.
    """
    merged: list[RssItem] = []
    replaced = False
    for item in items:
        if _same_episode(item, new_item):
            if not replaced:
                # If matched by GUID equality, use new_item as-is.
                # If matched only by enclosure URL, preserve existing GUID.
                if item.guid and new_item.guid and item.guid == new_item.guid:
                    merged.append(new_item)
                else:
                    merged.append(
                        RssItem(
                            title=new_item.title,
                            enclosure_url=new_item.enclosure_url,
                            enclosure_length=new_item.enclosure_length,
                            enclosure_type=new_item.enclosure_type,
                            pub_date=new_item.pub_date,
                            guid=item.guid,
                            duration=new_item.duration,
                            description=new_item.description,
                        )
                    )
                replaced = True
            # Skip subsequent matching duplicates
        else:
            merged.append(item)
    if not replaced:
        merged.append(new_item)
    return sort_items_newest_first(merged)


def limit_feed_items(
    items: Sequence[RssItem], max_items: int | None = DEFAULT_MAX_FEED_ITEMS
) -> list[RssItem]:
    """Return the newest *max_items* episodes for the public RSS document.

    The S3 object is a single XML file. Players such as Apple Podcasts
    (iTunes) and YouTube Music download that whole document on each poll,
    so an unbounded catalog eventually times out. ``max_items`` of ``None``
    or a non-positive value disables the cap. The uploader keeps omitted
    items in ``rss-catalog.xml``; audio objects in S3 are not deleted.
    Item count is not a byte bound — use ``build_public_rss_xml`` so inherited
    descriptions cannot still produce a multi-megabyte ``rss.xml``.
    """
    ordered = sort_items_newest_first(items)
    if max_items is None or max_items <= 0:
        return ordered
    return ordered[:max_items]


def clip_text(text: str | None, max_chars: int) -> str | None:
    """Return *text* truncated to *max_chars*, using an ellipsis when clipped."""
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1].rstrip() + "…"


def slim_item(
    item: RssItem,
    *,
    max_description: int = DEFAULT_MAX_ITEM_DESCRIPTION_CHARS,
    max_title: int = DEFAULT_MAX_TITLE_CHARS,
) -> RssItem:
    """Return a copy of *item* with title/description clipped for the public feed."""
    title = clip_text(item.title, max_title) or item.title
    return RssItem(
        title=title,
        enclosure_url=item.enclosure_url,
        enclosure_length=item.enclosure_length,
        enclosure_type=item.enclosure_type,
        pub_date=item.pub_date,
        guid=item.guid,
        duration=item.duration,
        description=clip_text(item.description, max_description),
    )


def merge_missing_items(
    primary: Sequence[RssItem], extra: Sequence[RssItem]
) -> list[RssItem]:
    """Add *extra* episodes that are not already in *primary*.

    Existing primary items win, so a slim public feed cannot overwrite full
    catalog descriptions.
    """
    merged = list(primary)
    for item in extra:
        if any(_same_episode(existing, item) for existing in merged):
            continue
        merged = upsert_item(merged, item)
    return sort_items_newest_first(merged)


def build_public_rss_xml(
    items: Sequence[RssItem],
    feed_url: str,
    *,
    title: str = DEFAULT_CHANNEL_TITLE,
    description: str = DEFAULT_CHANNEL_DESCRIPTION,
    author: str = DEFAULT_CHANNEL_AUTHOR,
    max_items: int | None = DEFAULT_MAX_FEED_ITEMS,
    max_bytes: int | None = DEFAULT_MAX_FEED_BYTES,
    max_description: int = DEFAULT_MAX_ITEM_DESCRIPTION_CHARS,
) -> tuple[list[RssItem], bytes]:
    """Serialize the player-facing feed within item and byte budgets.

    Inherited episode notes are clipped so a migrated catalog cannot blow past
    the timeout threshold. Oldest items are dropped until the XML fits
    *max_bytes*. A non-positive *max_bytes* disables the size cap. At least
    one item is always kept, even if that single episode still exceeds the
    budget.
    """
    window = [
        slim_item(item, max_description=max_description)
        for item in limit_feed_items(items, max_items)
    ]
    channel_title = clip_text(title, DEFAULT_MAX_TITLE_CHARS) or DEFAULT_CHANNEL_TITLE
    channel_description = (
        clip_text(description, max_description) or DEFAULT_CHANNEL_DESCRIPTION
    )

    def _serialize(subset: Sequence[RssItem]) -> bytes:
        return build_rss_xml(
            subset,
            feed_url,
            title=channel_title,
            description=channel_description,
            author=author,
        )

    if not window:
        xml = _serialize([])
        return [], xml

    xml = _serialize(window)
    if max_bytes is None or max_bytes <= 0 or len(xml) <= max_bytes:
        return window, xml

    lo, hi = 1, len(window) - 1
    best_n = 1
    best_xml = _serialize(window[:1])
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_xml = _serialize(window[:mid])
        if len(candidate_xml) <= max_bytes:
            best_n = mid
            best_xml = candidate_xml
            lo = mid + 1
        else:
            hi = mid - 1
    return window[:best_n], best_xml


def sort_items_newest_first(items: Sequence[RssItem]) -> list[RssItem]:
    """Return items ordered by ``pubDate``, newest first.

    Items with equal ``pubDate`` are ordered by their position in the input,
    with later items appearing first. This ensures a newly upserted item
    wins ties against existing items.
    """
    return [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda pair: (_pubdate_timestamp(pair[1]), pair[0]),
            reverse=True,
        )
    ]


def build_rss_xml(
    items: Sequence[RssItem],
    feed_url: str,
    *,
    title: str = DEFAULT_CHANNEL_TITLE,
    description: str = DEFAULT_CHANNEL_DESCRIPTION,
    author: str = DEFAULT_CHANNEL_AUTHOR,
) -> bytes:
    """Serialize a complete RSS 2.0 podcast feed (newest items first)."""
    ET.register_namespace("itunes", ITUNES_NS)
    ET.register_namespace("atom", ATOM_NS)

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "link").text = feed_url
    ET.SubElement(channel, "description").text = description
    ET.SubElement(channel, "language").text = DEFAULT_LANGUAGE
    ET.SubElement(channel, "lastBuildDate").text = formatdate(usegmt=True)
    ET.SubElement(channel, "generator").text = "PodClean"

    atom_link = ET.SubElement(channel, f"{{{ATOM_NS}}}link")
    atom_link.set("href", feed_url)
    atom_link.set("rel", "self")
    atom_link.set("type", "application/rss+xml")

    ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = author
    ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = description
    ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "false"
    itunes_category = ET.SubElement(channel, f"{{{ITUNES_NS}}}category")
    itunes_category.set("text", DEFAULT_ITUNES_CATEGORY)

    for item in sort_items_newest_first(items):
        _append_item_element(channel, item, author=author)

    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def _same_episode(left: RssItem, right: RssItem) -> bool:
    if left.guid and right.guid and left.guid == right.guid:
        return True
    return bool(left.enclosure_url and left.enclosure_url == right.enclosure_url)


def _pubdate_timestamp(item: RssItem) -> float:
    if not item.pub_date:
        return 0.0
    try:
        parsed = parsedate_to_datetime(item.pub_date)
    except (TypeError, ValueError, OverflowError, IndexError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _item_from_element(element: ET.Element) -> RssItem:
    enclosure = _find_child(element, "enclosure")
    enclosure_url = enclosure.get("url", "") if enclosure is not None else ""
    enclosure_length = enclosure.get("length", "0") if enclosure is not None else "0"
    enclosure_type = (
        enclosure.get("type", "audio/mpeg") if enclosure is not None else "audio/mpeg"
    )
    guid_el = _find_child(element, "guid")
    guid = (guid_el.text or "").strip() if guid_el is not None else ""
    return RssItem(
        title=_child_text(element, "title") or "Untitled",
        enclosure_url=enclosure_url,
        enclosure_length=enclosure_length or "0",
        enclosure_type=enclosure_type or "audio/mpeg",
        pub_date=_child_text(element, "pubDate") or "",
        guid=guid,
        duration=_child_text(element, "duration"),
        description=_child_text(element, "description"),
    )


def _append_item_element(channel: ET.Element, item: RssItem, *, author: str) -> None:
    el = ET.SubElement(channel, "item")
    ET.SubElement(el, "title").text = item.title
    ET.SubElement(el, "description").text = item.description or item.title
    ET.SubElement(el, "link").text = item.enclosure_url
    ET.SubElement(el, "pubDate").text = item.pub_date

    guid_el = ET.SubElement(el, "guid")
    guid_el.set("isPermaLink", "true" if item.guid.startswith("http") else "false")
    guid_el.text = item.guid

    enclosure = ET.SubElement(el, "enclosure")
    enclosure.set("url", item.enclosure_url)
    enclosure.set("length", str(item.enclosure_length))
    enclosure.set("type", item.enclosure_type)

    ET.SubElement(el, f"{{{ITUNES_NS}}}author").text = author
    ET.SubElement(el, f"{{{ITUNES_NS}}}explicit").text = "false"
    ET.SubElement(el, f"{{{ITUNES_NS}}}title").text = item.title
    if item.duration:
        ET.SubElement(el, f"{{{ITUNES_NS}}}duration").text = item.duration


def _localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag.split(":")[-1]


def _iter_children(parent: ET.Element, name: str):
    for child in parent:
        if _localname(child.tag) == name:
            yield child


def _find_child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in _iter_children(parent, name):
        return child
    return None


def _child_text(parent: ET.Element, name: str) -> str | None:
    child = _find_child(parent, name)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None

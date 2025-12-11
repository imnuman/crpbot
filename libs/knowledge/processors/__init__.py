"""
Knowledge processing pipeline.
"""

from typing import List
from ..base import KnowledgeItem, auto_tag_content


def process_item(item: KnowledgeItem) -> KnowledgeItem:
    """Process a knowledge item through the pipeline."""
    # Extract symbols if not already set
    if not item.symbols:
        item.symbols = item.extract_symbols()

    # Extract timeframes if not already set
    if not item.timeframes:
        item.timeframes = item.extract_timeframes()

    # Auto-tag if no tags
    if not item.tags:
        content = (item.full_content or "") + " " + (item.summary or "")
        item.tags = auto_tag_content(content)

    return item


def deduplicate_items(items: List[KnowledgeItem]) -> List[KnowledgeItem]:
    """Remove duplicate items based on content hash."""
    seen = set()
    unique = []

    for item in items:
        content_hash = item.get_content_hash()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(item)

    return unique

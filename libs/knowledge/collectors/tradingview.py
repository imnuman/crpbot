"""
TradingView Ideas collector.

Scrapes public trading ideas from TradingView - no API key required.
This is a great alternative to Reddit while waiting for API approval.
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    logger.warning("httpx/beautifulsoup4 not installed - TradingView collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# TradingView URLs
TV_BASE = "https://www.tradingview.com"
TV_IDEAS_URLS = {
    "forex": f"{TV_BASE}/markets/currencies/ideas/",
    "gold": f"{TV_BASE}/symbols/XAUUSD/ideas/",
    "indices": f"{TV_BASE}/markets/indices/ideas/",
    "crypto": f"{TV_BASE}/markets/cryptocurrencies/ideas/",
}

# Symbols we care about
TARGET_SYMBOLS = [
    "XAUUSD", "EURUSD", "GBPUSD", "USDJPY",
    "US30", "NAS100", "SPX500",
    "BTCUSD", "ETHUSD",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class TradingViewCollector(BaseCollector):
    """
    Collector for TradingView Ideas.

    No API key required - scrapes public ideas pages.
    High quality content from traders sharing strategies.
    """

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and SCRAPING_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.TRADINGVIEW

    def get_schedule(self) -> str:
        """Run every 6 hours."""
        return "0 */6 * * *"

    def get_max_items_per_run(self) -> int:
        return 50

    async def collect(self) -> List[KnowledgeItem]:
        """Collect trading ideas from TradingView."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        # Collect from each category
        for category, url in TV_IDEAS_URLS.items():
            try:
                category_items = await self._collect_ideas_page(url, category)
                items.extend(category_items)
                logger.info(f"Collected {len(category_items)} ideas from {category}")
                await asyncio.sleep(2)  # Be respectful
            except Exception as e:
                logger.error(f"Error collecting {category}: {e}")

        # Also collect from specific symbols
        for symbol in TARGET_SYMBOLS[:5]:  # Limit per run
            try:
                symbol_url = f"{TV_BASE}/symbols/{symbol}/ideas/"
                symbol_items = await self._collect_ideas_page(symbol_url, symbol)
                items.extend(symbol_items)
                await asyncio.sleep(2)
            except Exception as e:
                logger.debug(f"Error collecting {symbol}: {e}")

        # Deduplicate
        seen_urls = set()
        unique_items = []
        for item in items:
            if item.source_url not in seen_urls:
                seen_urls.add(item.source_url)
                unique_items.append(item)

        logger.info(f"Total unique TradingView ideas: {len(unique_items)}")
        return unique_items[:self.get_max_items_per_run()]

    async def _collect_ideas_page(self, url: str, category: str) -> List[KnowledgeItem]:
        """Collect ideas from a TradingView ideas page."""
        items = []

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find idea cards (TradingView uses various selectors)
            idea_cards = soup.select(
                "article, .tv-widget-idea, .idea-card, [class*='ideaCard'], [class*='IdeaCard']"
            )

            if not idea_cards:
                # Try alternative selectors
                idea_cards = soup.select("div[data-symbol], a[href*='/chart/']")

            for card in idea_cards[:20]:
                try:
                    item = self._parse_idea_card(card, category)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.debug(f"Error parsing idea card: {e}")

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")

        return items

    def _parse_idea_card(self, card, category: str) -> Optional[KnowledgeItem]:
        """Parse a single idea card."""
        try:
            # Find title
            title_elem = card.select_one(
                "a[class*='title'], .idea-title, h2 a, h3 a, [class*='Title']"
            )
            if not title_elem:
                # Try getting text from first link
                title_elem = card.select_one("a")

            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title or len(title) < 10:
                return None

            # Get link
            link = title_elem.get("href", "")
            if link and not link.startswith("http"):
                link = f"{TV_BASE}{link}"

            if not link or "/ideas/" not in link and "/chart/" not in link:
                return None

            # Get description/preview
            desc_elem = card.select_one(
                ".idea-preview, .description, p, [class*='description']"
            )
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Get author
            author_elem = card.select_one(
                ".tv-card-user-info__name, .author, [class*='author'], [class*='Author']"
            )
            author = author_elem.get_text(strip=True) if author_elem else None

            # Get likes/views if available
            likes_elem = card.select_one("[class*='like'], [class*='Like']")
            likes = None
            if likes_elem:
                likes_text = likes_elem.get_text(strip=True)
                try:
                    likes = int(re.sub(r'\D', '', likes_text) or 0)
                except ValueError:
                    pass

            # Extract symbol from title or category
            symbols = self._extract_symbols(title + " " + description + " " + category)

            # Determine content type
            title_lower = title.lower()
            if any(kw in title_lower for kw in ["analysis", "setup", "trade idea", "long", "short"]):
                content_type = ContentType.STRATEGY
            elif any(kw in title_lower for kw in ["indicator", "signal"]):
                content_type = ContentType.INDICATOR
            else:
                content_type = ContentType.DISCUSSION

            item = KnowledgeItem(
                source=KnowledgeSource.TRADINGVIEW,
                source_url=link,
                title=title[:200],
                content_type=content_type,
                summary=description[:500] if description else None,
                full_content=f"Title: {title}\n\nDescription: {description}\n\nCategory: {category}",
                author=author,
                upvotes=likes,
                symbols=symbols,
                tags=auto_tag_content(title + " " + description),
                quality_score=self._calculate_quality(likes, title, description),
            )

            # Extract timeframes
            item.timeframes = item.extract_timeframes()

            return item

        except Exception as e:
            logger.debug(f"Error parsing idea: {e}")
            return None

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract trading symbols from text."""
        text = text.upper()
        found = set()

        # Common patterns
        patterns = [
            r'\b(XAUUSD|XAGUSD)\b',
            r'\b(EURUSD|GBPUSD|USDJPY|AUDUSD|USDCAD|USDCHF|NZDUSD)\b',
            r'\b(EURGBP|EURJPY|GBPJPY|AUDJPY|CADJPY)\b',
            r'\b(US30|NAS100|US100|SPX500|DAX40|FTSE100)\b',
            r'\b(BTCUSD|ETHUSD|BTCUSDT|ETHUSDT)\b',
            r'\bGOLD\b',
            r'\bSILVER\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            found.update(matches)

        # Normalize
        symbol_map = {
            "GOLD": "XAUUSD",
            "SILVER": "XAGUSD",
            "US100": "NAS100",
        }

        normalized = set()
        for s in found:
            normalized.add(symbol_map.get(s, s))

        return sorted(normalized)

    def _calculate_quality(
        self,
        likes: Optional[int],
        title: str,
        description: str,
    ) -> float:
        """Calculate quality score."""
        import math

        score = 0.4  # Base score (TradingView content is generally decent)

        # Likes boost
        if likes and likes > 0:
            score += min(0.3, math.log10(likes + 1) / 5)

        # Content length bonus
        content_len = len(title) + len(description)
        if content_len > 200:
            score += 0.1
        if content_len > 500:
            score += 0.1

        # Keywords that indicate quality analysis
        quality_keywords = [
            "analysis", "setup", "target", "stop loss", "risk",
            "support", "resistance", "trend", "breakout", "reversal"
        ]
        text_lower = (title + " " + description).lower()
        keyword_matches = sum(1 for kw in quality_keywords if kw in text_lower)
        score += min(0.2, keyword_matches * 0.05)

        return min(1.0, score)

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_tradingview_collector() -> int:
    """Run the TradingView collector and save results."""
    collector = TradingViewCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.TRADINGVIEW, "ideas_collect")

    try:
        items = await collector.collect()

        saved = 0
        for item in items:
            try:
                storage.save_item(item)
                saved += 1
            except Exception as e:
                logger.error(f"Error saving item: {e}")

        storage.complete_scrape_log(log_id, "success", len(items), saved)
        logger.info(f"TradingView collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"TradingView collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_tradingview_collector())

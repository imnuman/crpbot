"""
MQL5 CodeBase scraper for trading EAs and indicators.

Scrapes the MQL5.com website for:
- Free Expert Advisors (EAs)
- Free Indicators
- Strategy articles
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
    logger.warning("httpx/beautifulsoup4 not installed - MQL5 scraper disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    CodeFile,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# MQL5 URLs to scrape
MQL5_BASE = "https://www.mql5.com"
SCRAPE_URLS = {
    "experts": f"{MQL5_BASE}/en/code/mt5/experts",
    "indicators": f"{MQL5_BASE}/en/code/mt5/indicators",
    "articles": f"{MQL5_BASE}/en/articles",
}

# Categories to focus on
RELEVANT_CATEGORIES = [
    "trading systems",
    "expert advisors",
    "indicators",
    "trading",
    "scalping",
    "trend",
    "breakout",
    "mean reversion",
    "gold",
    "forex",
]

# User agent for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class MQL5Collector(BaseCollector):
    """Collector for MQL5 CodeBase."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and SCRAPING_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.MQL5

    def get_schedule(self) -> str:
        """Run daily at 02:00 UTC."""
        return "0 2 * * *"

    def get_max_items_per_run(self) -> int:
        return 50

    async def collect(self) -> List[KnowledgeItem]:
        """Collect EAs and indicators from MQL5."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        # Collect EAs
        try:
            ea_items = await self._collect_codebase("experts", ContentType.EA_CODE)
            items.extend(ea_items)
            logger.info(f"Collected {len(ea_items)} EAs from MQL5")
        except Exception as e:
            logger.error(f"Error collecting EAs: {e}")

        # Collect indicators
        try:
            indicator_items = await self._collect_codebase("indicators", ContentType.INDICATOR)
            items.extend(indicator_items)
            logger.info(f"Collected {len(indicator_items)} indicators from MQL5")
        except Exception as e:
            logger.error(f"Error collecting indicators: {e}")

        # Collect articles
        try:
            article_items = await self._collect_articles()
            items.extend(article_items)
            logger.info(f"Collected {len(article_items)} articles from MQL5")
        except Exception as e:
            logger.error(f"Error collecting articles: {e}")

        return items[:self.get_max_items_per_run()]

    async def _collect_codebase(
        self,
        category: str,
        content_type: ContentType,
    ) -> List[KnowledgeItem]:
        """Collect from MQL5 codebase (EAs or indicators)."""
        items = []
        url = SCRAPE_URLS.get(category)

        if not url:
            return items

        try:
            # Get listing page
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find code items (adjust selector based on MQL5 structure)
            code_items = soup.select(".code-item, .market-product, article")

            for code_item in code_items[:30]:  # Limit per category
                try:
                    item = await self._parse_code_item(code_item, content_type)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.debug(f"Error parsing code item: {e}")

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")

        return items

    async def _parse_code_item(
        self,
        element,
        content_type: ContentType,
    ) -> Optional[KnowledgeItem]:
        """Parse a single code item from listing."""
        try:
            # Find title and link
            title_elem = element.select_one("a.title, h2 a, .name a")
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            link = title_elem.get("href", "")

            if not link.startswith("http"):
                link = f"{MQL5_BASE}{link}"

            # Get description
            desc_elem = element.select_one(".description, .desc, p")
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Get rating/downloads if available
            rating_elem = element.select_one(".rating, .stars")
            downloads_elem = element.select_one(".downloads, .count")

            rating = None
            if rating_elem:
                try:
                    rating = float(rating_elem.get_text(strip=True).replace(",", "."))
                except (ValueError, TypeError):
                    pass

            # Fetch detail page for full content
            full_content = await self._fetch_detail_page(link)

            item = KnowledgeItem(
                source=KnowledgeSource.MQL5,
                source_url=link,
                title=title,
                content_type=content_type,
                summary=description[:500] if description else None,
                full_content=full_content,
                quality_score=rating / 5.0 if rating else 0.5,
            )

            # Extract symbols and timeframes
            item.symbols = item.extract_symbols()
            item.timeframes = item.extract_timeframes()
            item.tags = auto_tag_content(full_content or description)

            return item

        except Exception as e:
            logger.debug(f"Error parsing code item: {e}")
            return None

    async def _fetch_detail_page(self, url: str) -> Optional[str]:
        """Fetch and extract content from detail page."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract main content
            content_parts = []

            # Description
            desc = soup.select_one(".description, #product-description, article")
            if desc:
                content_parts.append(desc.get_text(strip=True))

            # Code snippet if visible
            code = soup.select_one("pre, code, .code-block")
            if code:
                content_parts.append(f"\nCode:\n{code.get_text()[:2000]}")

            # Parameters/inputs
            params = soup.select(".parameter, .input, tr")
            if params:
                content_parts.append("\nParameters:")
                for p in params[:20]:
                    content_parts.append(f"- {p.get_text(strip=True)[:100]}")

            return "\n".join(content_parts)[:10000]

        except Exception as e:
            logger.debug(f"Error fetching detail page: {e}")
            return None

    async def _collect_articles(self) -> List[KnowledgeItem]:
        """Collect strategy articles from MQL5."""
        items = []
        url = SCRAPE_URLS["articles"]

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find article items
            articles = soup.select("article, .article-item, .post")

            for article in articles[:20]:
                try:
                    item = await self._parse_article(article)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")

        except Exception as e:
            logger.error(f"Error fetching articles: {e}")

        return items

    async def _parse_article(self, element) -> Optional[KnowledgeItem]:
        """Parse a single article."""
        try:
            # Find title and link
            title_elem = element.select_one("a, h2 a, .title a")
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            link = title_elem.get("href", "")

            if not link.startswith("http"):
                link = f"{MQL5_BASE}{link}"

            # Check if trading/strategy related
            title_lower = title.lower()
            if not any(kw in title_lower for kw in RELEVANT_CATEGORIES):
                return None

            # Get summary
            summary_elem = element.select_one(".summary, .excerpt, p")
            summary = summary_elem.get_text(strip=True) if summary_elem else ""

            # Fetch full article
            full_content = await self._fetch_article_content(link)

            item = KnowledgeItem(
                source=KnowledgeSource.MQL5,
                source_url=link,
                title=title,
                content_type=ContentType.ARTICLE,
                summary=summary[:500] if summary else None,
                full_content=full_content,
                quality_score=0.7,  # Articles are generally high quality
            )

            item.symbols = item.extract_symbols()
            item.timeframes = item.extract_timeframes()
            item.tags = auto_tag_content(full_content or summary)

            return item

        except Exception as e:
            logger.debug(f"Error parsing article: {e}")
            return None

    async def _fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch full article content."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find main article content
            content = soup.select_one("article, .article-body, .content, main")
            if content:
                # Remove scripts and styles
                for tag in content.select("script, style, nav, footer"):
                    tag.decompose()

                return content.get_text(strip=True)[:15000]

        except Exception as e:
            logger.debug(f"Error fetching article: {e}")

        return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_mql5_collector() -> int:
    """Run the MQL5 collector and save results."""
    collector = MQL5Collector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.MQL5, "full_collect")

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
        logger.info(f"MQL5 collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"MQL5 collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_mql5_collector())

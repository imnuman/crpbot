"""
News Sentiment Collector - Market news with sentiment analysis.

Aggregates news from free sources and provides sentiment signals
for trading decisions.

Sources:
- Finnhub (free tier - 60 calls/minute)
- Alpha Vantage (free tier - 25 calls/day)
- MarketAux (free tier - 100 calls/day)
"""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - News collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
)
from ..storage import get_storage


# Symbol mappings for news searches
SYMBOL_KEYWORDS = {
    "XAUUSD": ["gold", "XAU", "gold price", "precious metals", "bullion"],
    "EURUSD": ["EUR/USD", "euro dollar", "EURUSD", "ECB", "eurozone"],
    "GBPUSD": ["GBP/USD", "pound dollar", "GBPUSD", "BOE", "sterling"],
    "USDJPY": ["USD/JPY", "dollar yen", "USDJPY", "BOJ", "japanese yen"],
    "US500": ["S&P 500", "SPX", "SPY", "US stocks", "wall street"],
    "NAS100": ["NASDAQ", "tech stocks", "QQQ", "NASDAQ 100"],
    "US30": ["Dow Jones", "DJIA", "US30", "dow industrials"],
    "BTCUSD": ["bitcoin", "BTC", "crypto", "cryptocurrency"],
}

# Free news API endpoints
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
ALPHA_VANTAGE_NEWS_URL = "https://www.alphavantage.co/query"


class NewsSentimentCollector(BaseCollector):
    """Collector for financial news with sentiment analysis."""

    def __init__(self):
        self._client = None
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self.alpha_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": "HYDRA-Knowledge/1.0"},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.SENTIMENT

    def get_schedule(self) -> str:
        """Run every 4 hours."""
        return "0 */4 * * *"

    def get_max_items_per_run(self) -> int:
        return 30

    async def collect(self) -> List[KnowledgeItem]:
        """Collect news from various sources."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        # Try Finnhub (best free option)
        if self.finnhub_key:
            finnhub_items = await self._collect_finnhub()
            items.extend(finnhub_items)

        # Try Alpha Vantage
        if self.alpha_key:
            av_items = await self._collect_alpha_vantage()
            items.extend(av_items)

        # If no API keys, try scraping free news
        if not self.finnhub_key and not self.alpha_key:
            free_items = await self._collect_free_sources()
            items.extend(free_items)

        # Dedupe by title
        seen_titles = set()
        unique_items = []
        for item in items:
            title_key = item.title.lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)

        logger.info(f"Collected {len(unique_items)} news items")
        return unique_items[:self.get_max_items_per_run()]

    async def _collect_finnhub(self) -> List[KnowledgeItem]:
        """Collect from Finnhub API."""
        items = []

        try:
            # General market news
            params = {
                "category": "general",
                "token": self.finnhub_key
            }
            response = await self.client.get(FINNHUB_NEWS_URL, params=params)

            if response.status_code == 200:
                news_data = response.json()
                for article in news_data[:20]:
                    item = self._parse_finnhub_article(article)
                    if item:
                        items.append(item)

            # Forex news
            params["category"] = "forex"
            response = await self.client.get(FINNHUB_NEWS_URL, params=params)

            if response.status_code == 200:
                news_data = response.json()
                for article in news_data[:10]:
                    item = self._parse_finnhub_article(article)
                    if item:
                        items.append(item)

        except Exception as e:
            logger.error(f"Finnhub collection error: {e}")

        return items

    def _parse_finnhub_article(self, article: Dict) -> Optional[KnowledgeItem]:
        """Parse Finnhub article into KnowledgeItem."""
        try:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            url = article.get("url", "")
            source = article.get("source", "Unknown")
            timestamp = article.get("datetime", 0)

            if not headline:
                return None

            # Determine relevant symbols
            symbols = self._match_symbols(f"{headline} {summary}")

            # Simple sentiment analysis based on keywords
            sentiment = self._analyze_sentiment(f"{headline} {summary}")

            pub_date = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

            content = f"""News: {headline}

Source: {source}
Published: {pub_date.strftime('%Y-%m-%d %H:%M')}

{summary}

Symbols: {', '.join(symbols) if symbols else 'General Market'}
Sentiment: {sentiment['label']} ({sentiment['score']:.0%})

Trading Implication:
{self._get_trading_implication(sentiment, symbols)}
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url=url,
                title=f"News: {headline[:80]}",
                content_type=ContentType.MARKET_DATA,
                summary=f"{headline}\n\nSentiment: {sentiment['label']}",
                full_content=content,
                quality_score=0.7,
            )

            item.symbols = symbols
            item.tags = ["news", sentiment['label'].lower(), source.lower().replace(" ", "_")]

            return item

        except Exception as e:
            logger.debug(f"Error parsing article: {e}")
            return None

    async def _collect_alpha_vantage(self) -> List[KnowledgeItem]:
        """Collect from Alpha Vantage News API."""
        items = []

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets,forex,economy_macro",
                "apikey": self.alpha_key
            }
            response = await self.client.get(ALPHA_VANTAGE_NEWS_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                feed = data.get("feed", [])

                for article in feed[:20]:
                    item = self._parse_av_article(article)
                    if item:
                        items.append(item)

        except Exception as e:
            logger.error(f"Alpha Vantage collection error: {e}")

        return items

    def _parse_av_article(self, article: Dict) -> Optional[KnowledgeItem]:
        """Parse Alpha Vantage article."""
        try:
            title = article.get("title", "")
            summary = article.get("summary", "")
            url = article.get("url", "")
            source = article.get("source", "Unknown")
            time_published = article.get("time_published", "")

            # Alpha Vantage provides sentiment scores
            overall_sentiment = article.get("overall_sentiment_score", 0)
            sentiment_label = article.get("overall_sentiment_label", "Neutral")

            if not title:
                return None

            symbols = self._match_symbols(f"{title} {summary}")

            sentiment = {
                "label": sentiment_label,
                "score": (float(overall_sentiment) + 1) / 2  # Convert -1,1 to 0,1
            }

            content = f"""News: {title}

Source: {source}
Published: {time_published}

{summary}

Symbols: {', '.join(symbols) if symbols else 'General Market'}
Sentiment: {sentiment['label']} ({sentiment['score']:.0%})

Trading Implication:
{self._get_trading_implication(sentiment, symbols)}
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url=url,
                title=f"News: {title[:80]}",
                content_type=ContentType.MARKET_DATA,
                summary=f"{title}\n\nSentiment: {sentiment['label']}",
                full_content=content,
                quality_score=0.75,  # Higher because has real sentiment
            )

            item.symbols = symbols
            item.tags = ["news", sentiment_label.lower().replace(" ", "_"), source.lower().replace(" ", "_")]

            return item

        except Exception as e:
            logger.debug(f"Error parsing AV article: {e}")
            return None

    async def _collect_free_sources(self) -> List[KnowledgeItem]:
        """Collect from free RSS/scraping sources."""
        items = []

        # Investing.com RSS (if available)
        rss_urls = [
            "https://www.investing.com/rss/news.rss",
            "https://feeds.bbci.co.uk/news/business/rss.xml",
        ]

        for rss_url in rss_urls:
            try:
                response = await self.client.get(rss_url)
                if response.status_code == 200:
                    # Simple XML parsing for RSS
                    content = response.text
                    # Extract items between <item> tags
                    import re
                    item_matches = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)

                    for item_xml in item_matches[:10]:
                        title_match = re.search(r'<title>(.*?)</title>', item_xml)
                        desc_match = re.search(r'<description>(.*?)</description>', item_xml)
                        link_match = re.search(r'<link>(.*?)</link>', item_xml)

                        if title_match:
                            title = title_match.group(1).strip()
                            desc = desc_match.group(1).strip() if desc_match else ""
                            link = link_match.group(1).strip() if link_match else ""

                            # Clean CDATA
                            title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                            desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc)

                            symbols = self._match_symbols(f"{title} {desc}")
                            sentiment = self._analyze_sentiment(f"{title} {desc}")

                            item = KnowledgeItem(
                                source=KnowledgeSource.SENTIMENT,
                                source_url=link,
                                title=f"News: {title[:80]}",
                                content_type=ContentType.MARKET_DATA,
                                summary=f"{title}\n\nSentiment: {sentiment['label']}",
                                full_content=f"{title}\n\n{desc}\n\nSentiment: {sentiment['label']}",
                                quality_score=0.6,
                            )
                            item.symbols = symbols
                            item.tags = ["news", "rss", sentiment['label'].lower()]
                            items.append(item)

            except Exception as e:
                logger.debug(f"RSS fetch error for {rss_url}: {e}")

        return items

    def _match_symbols(self, text: str) -> List[str]:
        """Match text to trading symbols."""
        text_upper = text.upper()
        matched = []

        for symbol, keywords in SYMBOL_KEYWORDS.items():
            for keyword in keywords:
                if keyword.upper() in text_upper:
                    if symbol not in matched:
                        matched.append(symbol)
                    break

        return matched

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        # Positive keywords
        positive = [
            "surge", "soar", "rally", "gain", "rise", "jump", "bullish",
            "record high", "strong", "positive", "optimism", "growth",
            "beat", "exceed", "upgrade", "breakthrough", "recover"
        ]

        # Negative keywords
        negative = [
            "fall", "drop", "plunge", "crash", "decline", "bearish",
            "weak", "negative", "concern", "fear", "recession", "miss",
            "downgrade", "crisis", "collapse", "sell-off", "slump"
        ]

        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return {"label": "Neutral", "score": 0.5}

        score = pos_count / total

        if score > 0.6:
            label = "Bullish"
        elif score < 0.4:
            label = "Bearish"
        else:
            label = "Neutral"

        return {"label": label, "score": score}

    def _get_trading_implication(self, sentiment: Dict, symbols: List[str]) -> str:
        """Generate trading implication from sentiment."""
        label = sentiment['label']
        score = sentiment['score']

        if not symbols:
            return "General market news - no direct trading signal."

        if label == "Bullish":
            return f"Positive sentiment ({score:.0%}) for {', '.join(symbols)}. Consider BUY bias if aligned with technicals."
        elif label == "Bearish":
            return f"Negative sentiment ({score:.0%}) for {', '.join(symbols)}. Consider SELL bias if aligned with technicals."
        else:
            return f"Neutral sentiment for {', '.join(symbols)}. Wait for clearer signals."

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_news_sentiment_collector() -> int:
    """Run the news sentiment collector."""
    collector = NewsSentimentCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.SENTIMENT, "news_sentiment")

    try:
        items = await collector.collect()

        saved = 0
        for item in items:
            try:
                storage.save_item(item)
                saved += 1
            except Exception as e:
                logger.error(f"Error saving news item: {e}")

        storage.complete_scrape_log(log_id, "success", len(items), saved)
        logger.info(f"News sentiment collection: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"News sentiment collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_news_sentiment_collector())

"""
Reddit collector for trading strategy knowledge.

Uses PRAW (Python Reddit API Wrapper) to collect posts from trading subreddits.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not installed - Reddit collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# Subreddits to monitor
SUBREDDITS = [
    "algotrading",      # Algorithmic trading strategies
    "Forex",            # Forex trading strategies
    "daytrading",       # Day trading setups
    "Trading",          # General trading
    "options",          # Options strategies
    "Daytrading",       # Alternative spelling
    "technicalanalysis", # Technical analysis
    "FuturesTrading",   # Futures trading
]

# Search terms for finding relevant posts
SEARCH_TERMS = [
    "gold strategy",
    "XAUUSD",
    "scalping strategy",
    "VWAP",
    "breakout strategy",
    "opening range",
    "mean reversion",
    "london session",
    "ny session",
    "forex EA",
    "trading bot",
    "automated trading",
    "backtest results",
    "profitable strategy",
]

# Minimum score (upvotes) to consider
MIN_SCORE = 5
MIN_COMMENTS = 2


class RedditCollector(BaseCollector):
    """Collector for Reddit trading subreddits."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "HYDRA_Knowledge/1.0")

        self._reddit = None

    @property
    def reddit(self):
        """Lazy initialization of PRAW client."""
        if self._reddit is None and PRAW_AVAILABLE:
            if not self.client_id or not self.client_secret:
                logger.warning("Reddit credentials not configured")
                return None

            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        return self._reddit

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.REDDIT

    def get_schedule(self) -> str:
        """Run every 6 hours."""
        return "0 */6 * * *"

    def get_max_items_per_run(self) -> int:
        return 100

    async def collect(self) -> List[KnowledgeItem]:
        """Collect posts from trading subreddits."""
        if not self.reddit:
            logger.error("Reddit client not available")
            return []

        items = []

        # Collect from each subreddit
        for subreddit_name in SUBREDDITS:
            try:
                subreddit_items = await self._collect_from_subreddit(subreddit_name)
                items.extend(subreddit_items)
                logger.info(f"Collected {len(subreddit_items)} items from r/{subreddit_name}")
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")

        # Also search for specific terms
        for search_term in SEARCH_TERMS[:5]:  # Limit searches per run
            try:
                search_items = await self._search_reddit(search_term)
                items.extend(search_items)
            except Exception as e:
                logger.error(f"Error searching for '{search_term}': {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in items:
            if item.source_url not in seen_urls:
                seen_urls.add(item.source_url)
                unique_items.append(item)

        logger.info(f"Total unique Reddit items: {len(unique_items)}")
        return unique_items[:self.get_max_items_per_run()]

    async def _collect_from_subreddit(self, subreddit_name: str) -> List[KnowledgeItem]:
        """Collect posts from a specific subreddit."""
        items = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get hot posts
            for post in subreddit.hot(limit=25):
                item = self._post_to_item(post)
                if item:
                    items.append(item)

            # Get new posts
            for post in subreddit.new(limit=25):
                item = self._post_to_item(post)
                if item:
                    items.append(item)

            # Get top posts from past week
            for post in subreddit.top(time_filter="week", limit=25):
                item = self._post_to_item(post)
                if item:
                    items.append(item)

        except Exception as e:
            logger.error(f"Error accessing r/{subreddit_name}: {e}")

        return items

    async def _search_reddit(self, query: str) -> List[KnowledgeItem]:
        """Search Reddit for specific terms."""
        items = []

        try:
            for post in self.reddit.subreddit("all").search(
                query,
                sort="relevance",
                time_filter="month",
                limit=20,
            ):
                # Only from trading-related subreddits
                if post.subreddit.display_name.lower() in [s.lower() for s in SUBREDDITS]:
                    item = self._post_to_item(post)
                    if item:
                        items.append(item)

        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")

        return items

    def _post_to_item(self, post) -> Optional[KnowledgeItem]:
        """Convert Reddit post to KnowledgeItem."""
        try:
            # Filter by score and comments
            if post.score < MIN_SCORE:
                return None
            if post.num_comments < MIN_COMMENTS:
                return None

            # Skip very short posts
            content = post.selftext or ""
            if len(content) < 100 and not post.url:
                return None

            # Determine content type
            content_type = self._classify_post(post)

            # Extract full content (post + top comments)
            full_content = self._extract_full_content(post)

            # Create item
            item = KnowledgeItem(
                source=KnowledgeSource.REDDIT,
                source_url=f"https://reddit.com{post.permalink}",
                title=post.title,
                content_type=content_type,
                summary=content[:500] if content else None,
                full_content=full_content,
                author=str(post.author) if post.author else None,
                upvotes=post.score,
                comments_count=post.num_comments,
                source_created_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            )

            # Extract symbols and timeframes
            item.symbols = item.extract_symbols()
            item.timeframes = item.extract_timeframes()

            # Auto-tag
            item.tags = auto_tag_content(full_content)

            # Calculate quality score based on engagement
            item.quality_score = self._calculate_quality_score(post)

            return item

        except Exception as e:
            logger.debug(f"Error processing post: {e}")
            return None

    def _classify_post(self, post) -> ContentType:
        """Classify post into content type."""
        title_lower = post.title.lower()
        content_lower = (post.selftext or "").lower()

        if any(kw in title_lower for kw in ["strategy", "system", "method", "approach"]):
            return ContentType.STRATEGY
        if any(kw in title_lower for kw in ["indicator", "signal", "oscillator"]):
            return ContentType.INDICATOR
        if any(kw in title_lower for kw in ["bot", "ea", "expert advisor", "automated"]):
            return ContentType.EA_CODE
        if any(kw in content_lower for kw in ["win rate", "backtest", "results", "profit"]):
            return ContentType.STRATEGY
        if any(kw in title_lower for kw in ["question", "help", "how", "why", "what"]):
            return ContentType.DISCUSSION

        return ContentType.DISCUSSION

    def _extract_full_content(self, post) -> str:
        """Extract post content and top comments."""
        parts = [f"Title: {post.title}"]

        if post.selftext:
            parts.append(f"\nPost Content:\n{post.selftext}")

        # Get top comments
        try:
            post.comments.replace_more(limit=0)
            top_comments = sorted(
                post.comments.list()[:20],
                key=lambda c: c.score if hasattr(c, 'score') else 0,
                reverse=True,
            )[:5]

            if top_comments:
                parts.append("\nTop Comments:")
                for comment in top_comments:
                    if hasattr(comment, 'body') and comment.body:
                        parts.append(f"- {comment.body[:500]}")

        except Exception as e:
            logger.debug(f"Error extracting comments: {e}")

        return "\n".join(parts)

    def _calculate_quality_score(self, post) -> float:
        """Calculate quality score (0-1) based on engagement."""
        score = 0.3  # Base score

        # Upvote score (log scale)
        if post.score > 0:
            import math
            score += min(0.3, math.log10(post.score) / 10)

        # Comment score
        if post.num_comments > 0:
            import math
            score += min(0.2, math.log10(post.num_comments + 1) / 10)

        # Upvote ratio
        if hasattr(post, 'upvote_ratio'):
            score += post.upvote_ratio * 0.2

        return min(1.0, score)


async def run_reddit_collector() -> int:
    """Run the Reddit collector and save results."""
    collector = RedditCollector()
    storage = get_storage()

    # Start log
    log_id = storage.start_scrape_log(KnowledgeSource.REDDIT, "full_collect")

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
        logger.info(f"Reddit collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Reddit collection failed: {e}")
        raise


# For testing
if __name__ == "__main__":
    asyncio.run(run_reddit_collector())

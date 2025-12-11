"""
GitHub collector for trading EAs and strategy repositories.

Uses GitHub API to search for and download trading-related code.
"""

import os
import asyncio
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - GitHub collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    CodeFile,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"

# Search queries for finding trading repos
SEARCH_QUERIES = [
    # MT5/MQL5 EAs
    "MT5 expert advisor",
    "MQL5 trading bot",
    "MQL4 EA forex",
    "metatrader5 python",
    "mql5 indicator",
    # Specific strategies
    "gold trading bot XAUUSD",
    "scalping EA MT5",
    "VWAP trading strategy",
    "opening range breakout",
    "London session trading",
    "grid trading bot",
    "martingale trading",
    # Python trading
    "algorithmic trading python",
    "trading strategy backtest python",
    "forex backtesting",
    "quantitative trading",
    "technical analysis python",
    # Prop firm
    "FTMO trading bot",
    "prop firm EA",
    # Other platforms
    "pine script strategy",
    "tradingview strategy",
    "freqtrade strategy",
    "ccxt trading bot",
]

# Known high-quality repos to always include
KNOWN_REPOS = [
    # MQL5/MT5
    "geraked/metatrader5",
    "yulz008/GOLD_ORB",
    "mql5/mql5-examples",
    "EA31337/EA31337-classes",
    "EarnForex/PositionSizer",
    "EarnForex/Account-Protector",
    "EarnForex/News-Trader",
    # Python trading
    "freqtrade/freqtrade",
    "jesse-ai/jesse",
    "kernc/backtesting.py",
    "polakowo/vectorbt",
    "blankly-finance/blankly",
    "pmorissette/bt",
    # Indicators/Analysis
    "bukosabino/ta",
    "twopirllc/pandas-ta",
    "mrjbq7/ta-lib",
    # Strategies
    "Drakkar-Software/OctoBot",
    "hackingthemarkets/supertrend",
    "hackingthemarkets/tradingview-webhooks-bot",
]

# File extensions to look for
CODE_EXTENSIONS = [".mq5", ".mq4", ".mqh", ".py", ".pine"]


class GitHubCollector(BaseCollector):
    """Collector for GitHub trading repositories."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and HTTPX_AVAILABLE:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "HYDRA-Knowledge/1.0",
            }
            if self.token:
                headers["Authorization"] = f"token {self.token}"

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.GITHUB

    def get_schedule(self) -> str:
        """Run weekly on Saturday at 03:00 UTC."""
        return "0 3 * * 6"

    def get_max_items_per_run(self) -> int:
        return 30

    async def collect(self) -> List[KnowledgeItem]:
        """Collect trading repos from GitHub."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []
        seen_repos = set()

        # Collect from known repos first
        for repo in KNOWN_REPOS:
            try:
                repo_items = await self._collect_repo(repo)
                for item in repo_items:
                    if item.source_url not in seen_repos:
                        seen_repos.add(item.source_url)
                        items.append(item)
            except Exception as e:
                logger.error(f"Error collecting {repo}: {e}")

        # Search for more repos
        for query in SEARCH_QUERIES:
            if len(items) >= self.get_max_items_per_run():
                break

            try:
                search_items = await self._search_repos(query)
                for item in search_items:
                    if item.source_url not in seen_repos:
                        seen_repos.add(item.source_url)
                        items.append(item)
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")

            # Rate limiting
            await asyncio.sleep(1)

        logger.info(f"Collected {len(items)} items from GitHub")
        return items[:self.get_max_items_per_run()]

    async def _search_repos(self, query: str) -> List[KnowledgeItem]:
        """Search GitHub for repositories."""
        items = []

        try:
            url = f"{GITHUB_API_BASE}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10,
            }

            response = await self.client.get(url, params=params)

            if response.status_code == 403:
                logger.warning("GitHub rate limit hit")
                return items

            response.raise_for_status()
            data = response.json()

            for repo in data.get("items", []):
                try:
                    item = await self._repo_to_item(repo)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.debug(f"Error processing repo: {e}")

        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")

        return items

    async def _collect_repo(self, repo_name: str) -> List[KnowledgeItem]:
        """Collect from a specific repository."""
        items = []

        try:
            # Get repo info
            url = f"{GITHUB_API_BASE}/repos/{repo_name}"
            response = await self.client.get(url)
            response.raise_for_status()
            repo = response.json()

            item = await self._repo_to_item(repo)
            if item:
                items.append(item)

        except Exception as e:
            logger.error(f"Error collecting {repo_name}: {e}")

        return items

    async def _repo_to_item(self, repo: Dict[str, Any]) -> Optional[KnowledgeItem]:
        """Convert GitHub repo to KnowledgeItem."""
        try:
            name = repo.get("full_name", "")
            description = repo.get("description", "") or ""
            html_url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            language = repo.get("language", "")

            # Check if trading/EA related
            combined = f"{name} {description}".lower()
            trading_keywords = ["trading", "mt5", "mt4", "mql", "forex", "ea", "expert", "strategy", "backtest", "gold", "xauusd"]

            if not any(kw in combined for kw in trading_keywords):
                return None

            # Get README content
            readme = await self._get_readme(name)

            # Get relevant code files
            code_content = await self._get_code_files(name)

            full_content = f"Repository: {name}\n\nDescription: {description}\n\n"
            if readme:
                full_content += f"README:\n{readme[:5000]}\n\n"
            if code_content:
                full_content += f"Code Samples:\n{code_content[:5000]}"

            # Determine content type
            if language in ("MQL5", "MQL4") or any(ext in combined for ext in [".mq5", ".mq4"]):
                content_type = ContentType.EA_CODE
            elif "indicator" in combined:
                content_type = ContentType.INDICATOR
            else:
                content_type = ContentType.STRATEGY

            item = KnowledgeItem(
                source=KnowledgeSource.GITHUB,
                source_url=html_url,
                title=name,
                content_type=content_type,
                summary=description[:500] if description else None,
                full_content=full_content,
                author=repo.get("owner", {}).get("login"),
                upvotes=stars,
                quality_score=self._calculate_quality(stars, repo),
            )

            item.symbols = item.extract_symbols()
            item.timeframes = item.extract_timeframes()
            item.tags = auto_tag_content(full_content)

            # Add language as tag
            if language:
                item.tags.append(language.lower())

            return item

        except Exception as e:
            logger.debug(f"Error converting repo: {e}")
            return None

    async def _get_readme(self, repo_name: str) -> Optional[str]:
        """Fetch README content."""
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_name}/readme"
            response = await self.client.get(url)

            if response.status_code != 200:
                return None

            data = response.json()

            # Decode base64 content
            import base64
            content = data.get("content", "")
            if content:
                return base64.b64decode(content).decode("utf-8", errors="ignore")

        except Exception as e:
            logger.debug(f"Error fetching README: {e}")

        return None

    async def _get_code_files(self, repo_name: str) -> Optional[str]:
        """Get relevant code files from repo."""
        try:
            # Get repo tree
            url = f"{GITHUB_API_BASE}/repos/{repo_name}/git/trees/HEAD?recursive=1"
            response = await self.client.get(url)

            if response.status_code != 200:
                return None

            data = response.json()
            tree = data.get("tree", [])

            # Find relevant files
            code_parts = []
            for item in tree:
                path = item.get("path", "")
                if any(path.endswith(ext) for ext in CODE_EXTENSIONS):
                    # Fetch file content
                    content = await self._get_file_content(repo_name, path)
                    if content:
                        code_parts.append(f"--- {path} ---\n{content[:2000]}")

                    if len(code_parts) >= 3:  # Limit files
                        break

            return "\n\n".join(code_parts)

        except Exception as e:
            logger.debug(f"Error getting code files: {e}")

        return None

    async def _get_file_content(self, repo_name: str, path: str) -> Optional[str]:
        """Fetch a single file's content."""
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_name}/contents/{path}"
            response = await self.client.get(url)

            if response.status_code != 200:
                return None

            data = response.json()

            import base64
            content = data.get("content", "")
            if content:
                return base64.b64decode(content).decode("utf-8", errors="ignore")

        except Exception as e:
            logger.debug(f"Error fetching file: {e}")

        return None

    def _calculate_quality(self, stars: int, repo: Dict[str, Any]) -> float:
        """Calculate quality score based on repo metrics."""
        import math

        score = 0.3  # Base score

        # Stars (log scale)
        if stars > 0:
            score += min(0.3, math.log10(stars + 1) / 5)

        # Recent activity
        updated_at = repo.get("updated_at", "")
        if updated_at:
            try:
                updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - updated).days
                if days_ago < 30:
                    score += 0.2
                elif days_ago < 180:
                    score += 0.1
            except (ValueError, TypeError):
                pass

        # Has description
        if repo.get("description"):
            score += 0.1

        # Has license
        if repo.get("license"):
            score += 0.1

        return min(1.0, score)

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_github_collector() -> int:
    """Run the GitHub collector and save results."""
    collector = GitHubCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.GITHUB, "full_collect")

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
        logger.info(f"GitHub collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"GitHub collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_github_collector())

"""
Base models and data classes for the Knowledge Aggregation System.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import hashlib
import re


class ContentType(str, Enum):
    """Types of knowledge content."""
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    ARTICLE = "article"
    EA_CODE = "ea_code"
    DISCUSSION = "discussion"
    NEWS = "news"
    CALENDAR_EVENT = "calendar_event"
    VIDEO = "video"
    ACADEMIC_PAPER = "academic_paper"
    MARKET_DATA = "market_data"  # COT, sentiment, fear/greed


class KnowledgeSource(str, Enum):
    """Supported knowledge sources."""
    REDDIT = "reddit"
    MQL5 = "mql5"
    GITHUB = "github"
    FOREX_FACTORY = "forex_factory"
    TRADINGVIEW = "tradingview"
    YOUTUBE = "youtube"
    BABYPIPS = "babypips"
    INVESTOPEDIA = "investopedia"
    NEWS_API = "news_api"
    EODHD = "eodhd"
    ALPHA_VANTAGE = "alpha_vantage"
    SSRN = "ssrn"
    ARXIV = "arxiv"
    # New sources
    CFTC = "cftc"          # COT (Commitment of Traders) data
    SENTIMENT = "sentiment"  # Fear/Greed, broker sentiment
    ECONOMIC = "economic"    # Central bank calendars, rate decisions


class ImpactLevel(str, Enum):
    """Economic event impact levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class KnowledgeItem:
    """A single knowledge item from any source."""

    source: KnowledgeSource
    title: str
    content_type: ContentType

    # Content
    summary: Optional[str] = None
    full_content: Optional[str] = None
    source_url: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)

    # Performance metrics (if available)
    win_rate: Optional[float] = None
    risk_reward: Optional[float] = None
    expected_return: Optional[float] = None

    # Author/Source info
    author: Optional[str] = None
    author_reputation: Optional[float] = None

    # Quality assessment
    quality_score: Optional[float] = None  # 0-1, AI-assessed
    upvotes: Optional[int] = None
    comments_count: Optional[int] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source_created_at: Optional[datetime] = None  # Original publish date

    # Internal
    id: Optional[int] = None
    embedding_id: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def get_content_hash(self) -> str:
        """Generate unique hash for deduplication."""
        content = f"{self.source.value}:{self.source_url or self.title}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def extract_symbols(self) -> List[str]:
        """Extract trading symbols from content."""
        if not self.full_content and not self.summary:
            return self.symbols

        text = (self.full_content or "") + " " + (self.summary or "")

        # Common forex/crypto/index patterns
        patterns = [
            r'\b(XAUUSD|EURUSD|GBPUSD|USDJPY|AUDUSD|USDCAD|USDCHF)\b',
            r'\b(US30|NAS100|US100|SPX500|DAX|FTSE)\b',
            r'\b(BTCUSD|ETHUSD|BTCUSDT|ETHUSDT)\b',
            r'\b(Gold|Silver|Oil|WTI|Brent)\b',
        ]

        found = set(self.symbols)
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.update(m.upper() for m in matches)

        # Normalize
        symbol_map = {
            "GOLD": "XAUUSD",
            "SILVER": "XAGUSD",
            "OIL": "USOIL",
            "WTI": "USOIL",
            "BRENT": "UKOIL",
            "US100": "NAS100",
        }

        normalized = set()
        for s in found:
            normalized.add(symbol_map.get(s.upper(), s.upper()))

        return sorted(normalized)

    def extract_timeframes(self) -> List[str]:
        """Extract timeframes from content."""
        if not self.full_content and not self.summary:
            return self.timeframes

        text = (self.full_content or "") + " " + (self.summary or "")

        # Timeframe patterns
        patterns = [
            r'\b(M1|M5|M15|M30|H1|H4|D1|W1|MN)\b',
            r'\b(1\s*min|5\s*min|15\s*min|30\s*min|1\s*hour|4\s*hour|daily|weekly|monthly)\b',
        ]

        found = set(self.timeframes)

        # Pattern 1: Standard MT5 notation
        matches = re.findall(patterns[0], text, re.IGNORECASE)
        found.update(m.upper() for m in matches)

        # Pattern 2: Text notation - convert to standard
        text_map = {
            "1 min": "M1", "1min": "M1",
            "5 min": "M5", "5min": "M5",
            "15 min": "M15", "15min": "M15",
            "30 min": "M30", "30min": "M30",
            "1 hour": "H1", "1hour": "H1",
            "4 hour": "H4", "4hour": "H4",
            "daily": "D1",
            "weekly": "W1",
            "monthly": "MN",
        }

        matches = re.findall(patterns[1], text, re.IGNORECASE)
        for m in matches:
            normalized = m.lower().replace(" ", "")
            for key, value in text_map.items():
                if key.replace(" ", "") == normalized:
                    found.add(value)
                    break

        return sorted(found)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source": self.source.value,
            "source_url": self.source_url,
            "title": self.title,
            "content_type": self.content_type.value,
            "summary": self.summary,
            "full_content": self.full_content,
            "tags": self.tags,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "win_rate": self.win_rate,
            "risk_reward": self.risk_reward,
            "author": self.author,
            "quality_score": self.quality_score,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            source=KnowledgeSource(data["source"]),
            source_url=data.get("source_url"),
            title=data["title"],
            content_type=ContentType(data["content_type"]),
            summary=data.get("summary"),
            full_content=data.get("full_content"),
            tags=data.get("tags", []),
            symbols=data.get("symbols", []),
            timeframes=data.get("timeframes", []),
            win_rate=data.get("win_rate"),
            risk_reward=data.get("risk_reward"),
            author=data.get("author"),
            quality_score=data.get("quality_score"),
            embedding_id=data.get("embedding_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )


@dataclass
class EconomicEvent:
    """Economic calendar event."""

    event_date: datetime
    currency: str
    event_name: str
    impact: ImpactLevel

    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None

    source: KnowledgeSource = KnowledgeSource.FOREX_FACTORY

    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_high_impact(self) -> bool:
        """Check if event is high impact."""
        return self.impact == ImpactLevel.HIGH

    def affects_symbol(self, symbol: str) -> bool:
        """Check if event affects a trading symbol."""
        symbol = symbol.upper()
        currency = self.currency.upper()

        # Direct currency match
        if currency in symbol:
            return True

        # USD affects many pairs
        if currency == "USD":
            usd_pairs = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "US30", "NAS100", "SPX500"]
            if symbol in usd_pairs:
                return True

        return False


@dataclass
class CodeFile:
    """EA or indicator source code file."""

    knowledge_item_id: int
    filename: str
    language: str  # 'mql5', 'mql4', 'pine', 'python'
    content: str

    extracted_params: Optional[Dict[str, Any]] = None

    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def extract_mql_params(self) -> Dict[str, Any]:
        """Extract input parameters from MQL code."""
        if self.language not in ("mql4", "mql5"):
            return {}

        params = {}

        # Pattern: input <type> <name> = <value>;
        pattern = r'input\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
        matches = re.findall(pattern, self.content)

        for type_name, param_name, default_value in matches:
            params[param_name] = {
                "type": type_name,
                "default": default_value.strip(),
            }

        return params


@dataclass
class ScrapeLog:
    """Log entry for a scraping job."""

    source: KnowledgeSource
    job_type: str
    started_at: datetime

    completed_at: Optional[datetime] = None
    items_found: int = 0
    items_new: int = 0
    error_message: Optional[str] = None
    status: str = "running"  # 'running', 'success', 'failed', 'partial'

    id: Optional[int] = None

    def mark_success(self, found: int, new: int):
        """Mark job as successful."""
        self.completed_at = datetime.utcnow()
        self.items_found = found
        self.items_new = new
        self.status = "success"

    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.status = "failed"

    def mark_partial(self, found: int, new: int, error: str):
        """Mark job as partially complete."""
        self.completed_at = datetime.utcnow()
        self.items_found = found
        self.items_new = new
        self.error_message = error
        self.status = "partial"


class BaseCollector(ABC):
    """Abstract base class for all knowledge collectors."""

    @abstractmethod
    async def collect(self) -> List[KnowledgeItem]:
        """Fetch new items from source."""
        pass

    @abstractmethod
    def get_source_name(self) -> KnowledgeSource:
        """Return the source identifier."""
        pass

    @abstractmethod
    def get_schedule(self) -> str:
        """Return cron expression for scheduling.

        Examples:
        - "0 */6 * * *" = Every 6 hours
        - "0 2 * * *" = Daily at 02:00 UTC
        - "0 0 * * 0" = Weekly on Sunday at 00:00
        """
        pass

    def get_max_items_per_run(self) -> int:
        """Maximum items to fetch per run (override in subclasses)."""
        return 100


# Common tag keywords for auto-tagging
STRATEGY_KEYWORDS = {
    "scalping": ["scalp", "scalping", "quick", "fast", "short-term"],
    "swing": ["swing", "medium-term", "multi-day"],
    "trend": ["trend", "trending", "momentum", "breakout"],
    "reversal": ["reversal", "mean reversion", "revert", "bounce"],
    "breakout": ["breakout", "break out", "range break"],
    "orb": ["orb", "opening range", "open range"],
    "vwap": ["vwap", "volume weighted"],
    "session": ["london", "ny", "asian", "tokyo", "session"],
    "news": ["news", "nfp", "fomc", "cpi", "economic"],
}

TIMEFRAME_KEYWORDS = {
    "scalping": ["M1", "M5"],
    "intraday": ["M15", "M30", "H1"],
    "swing": ["H4", "D1"],
    "position": ["W1", "MN"],
}


def auto_tag_content(text: str) -> List[str]:
    """Auto-generate tags from text content."""
    text_lower = text.lower()
    tags = set()

    for tag, keywords in STRATEGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                tags.add(tag)
                break

    return sorted(tags)

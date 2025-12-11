"""
Query interface for the knowledge base.

Supports:
- Text search (SQL)
- Semantic search (vector)
- Economic calendar queries
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

from .base import (
    KnowledgeItem,
    EconomicEvent,
    KnowledgeSource,
    ContentType,
    ImpactLevel,
)
from .storage import get_storage
from .embeddings import get_embedding_service


class KnowledgeQuery:
    """Query interface for knowledge base."""

    @staticmethod
    def search(
        query: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        content_type: Optional[ContentType] = None,
        source: Optional[KnowledgeSource] = None,
        min_quality: Optional[float] = None,
        limit: int = 20,
        use_semantic: bool = True,
    ) -> List[KnowledgeItem]:
        """
        Search the knowledge base.

        Args:
            query: Text query for semantic or text search
            symbols: Filter by trading symbols (e.g., ["XAUUSD", "EURUSD"])
            timeframes: Filter by timeframes (e.g., ["H1", "D1"])
            content_type: Filter by content type
            source: Filter by source
            min_quality: Minimum quality score (0-1)
            limit: Maximum results to return
            use_semantic: Use vector search if available

        Returns:
            List of matching KnowledgeItems
        """
        storage = get_storage()

        # Try semantic search first if query provided and enabled
        if query and use_semantic:
            try:
                embedding_service = get_embedding_service()
                results = embedding_service.search_similar(
                    query=query,
                    n_results=limit * 2,  # Get more for post-filtering
                    source=source.value if source else None,
                    content_type=content_type.value if content_type else None,
                    min_quality=min_quality,
                )

                if results:
                    # Get full items from storage
                    items = []
                    for result in results:
                        embedding_id = result.get("id")
                        # Search by embedding_id or fall back to metadata
                        item = storage.search_items(
                            source=source,
                            content_type=content_type,
                            symbols=symbols,
                            timeframes=timeframes,
                            min_quality=min_quality,
                            limit=1,
                        )
                        if item:
                            items.extend(item)

                    if items:
                        return items[:limit]

            except Exception as e:
                logger.debug(f"Semantic search failed, falling back to text: {e}")

        # Fall back to text search
        return storage.search_items(
            query=query,
            source=source,
            content_type=content_type,
            symbols=symbols,
            timeframes=timeframes,
            min_quality=min_quality,
            limit=limit,
        )

    @staticmethod
    def get_strategies_for_symbol(
        symbol: str,
        timeframe: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeItem]:
        """Get trading strategies relevant to a symbol."""
        return KnowledgeQuery.search(
            query=f"trading strategy {symbol}",
            symbols=[symbol],
            timeframes=[timeframe] if timeframe else None,
            content_type=ContentType.STRATEGY,
            min_quality=0.5,
            limit=limit,
        )

    @staticmethod
    def get_similar(
        item: KnowledgeItem,
        limit: int = 5,
    ) -> List[KnowledgeItem]:
        """Get similar knowledge items."""
        # Create query from item
        query_parts = [item.title]
        if item.symbols:
            query_parts.append(" ".join(item.symbols))
        if item.tags:
            query_parts.append(" ".join(item.tags))

        query = " ".join(query_parts)

        return KnowledgeQuery.search(
            query=query,
            content_type=item.content_type,
            limit=limit + 1,  # Exclude self
        )[1:]  # Skip first result (likely self)

    @staticmethod
    def calendar(
        date: Optional[datetime] = None,
        days: int = 7,
        currencies: Optional[List[str]] = None,
        impact: Optional[ImpactLevel] = None,
    ) -> List[EconomicEvent]:
        """
        Get economic calendar events.

        Args:
            date: Start date (default: now)
            days: Number of days to look ahead
            currencies: Filter by currencies
            impact: Minimum impact level

        Returns:
            List of upcoming economic events
        """
        storage = get_storage()

        start_date = date or datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=days)

        events = storage.get_events(
            start_date=start_date,
            end_date=end_date,
            impact=impact,
        )

        if currencies:
            events = [e for e in events if e.currency in currencies]

        return events

    @staticmethod
    def get_high_impact_events(
        days: int = 7,
        symbols: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get high impact events, optionally filtered by affected symbols."""
        events = KnowledgeQuery.calendar(
            days=days,
            impact=ImpactLevel.HIGH,
        )

        if symbols:
            filtered = []
            for event in events:
                for symbol in symbols:
                    if event.affects_symbol(symbol):
                        filtered.append(event)
                        break
            return filtered

        return events

    @staticmethod
    def ask(
        question: str,
        context_symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using the knowledge base (RAG-style).

        Returns dict with:
        - answer: Generated answer (if available)
        - sources: List of relevant knowledge items
        - events: Relevant economic events
        """
        # Get relevant knowledge
        sources = KnowledgeQuery.search(
            query=question,
            symbols=context_symbols,
            limit=5,
            use_semantic=True,
        )

        # Get relevant events if symbols provided
        events = []
        if context_symbols:
            events = KnowledgeQuery.get_high_impact_events(
                days=7,
                symbols=context_symbols,
            )

        # Build context for answer
        context_parts = []
        for source in sources:
            context_parts.append(f"Source: {source.title}")
            if source.summary:
                context_parts.append(source.summary)

        return {
            "question": question,
            "sources": sources,
            "events": events,
            "context": "\n\n".join(context_parts),
        }

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get knowledge base statistics."""
        storage = get_storage()
        stats = storage.get_stats()

        # Add embedding stats
        try:
            embedding_service = get_embedding_service()
            stats["embeddings"] = embedding_service.get_collection_stats()
        except Exception as e:
            stats["embeddings"] = {"error": str(e)}

        return stats


# Convenience functions
def search_strategies(symbol: str, **kwargs) -> List[KnowledgeItem]:
    """Search for trading strategies for a symbol."""
    return KnowledgeQuery.get_strategies_for_symbol(symbol, **kwargs)


def get_calendar(days: int = 7, **kwargs) -> List[EconomicEvent]:
    """Get upcoming economic events."""
    return KnowledgeQuery.calendar(days=days, **kwargs)


def check_high_impact_events(symbols: List[str], days: int = 3) -> List[EconomicEvent]:
    """Check for high impact events affecting specified symbols."""
    return KnowledgeQuery.get_high_impact_events(days=days, symbols=symbols)


# =============================================================================
# TRADING-SPECIFIC QUERY FUNCTIONS (for FTMO bots)
# =============================================================================

import re
import sqlite3
from dataclasses import dataclass
from enum import Enum


class SentimentBias(str, Enum):
    """Sentiment-based trading bias."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(str, Enum):
    """Central bank risk levels."""
    CRITICAL = "critical"    # 0-1 days: NO trading
    HIGH = "high"            # 2-3 days: 50% size
    ELEVATED = "elevated"    # 4-7 days: Awareness
    NORMAL = "normal"        # 8+ days: Normal trading


@dataclass
class SentimentSignal:
    """Retail sentiment signal for a symbol."""
    symbol: str
    long_pct: float
    short_pct: float
    bias: SentimentBias
    confidence_modifier: float
    reasoning: str


@dataclass
class CentralBankRisk:
    """Central bank meeting risk assessment."""
    bank: str
    days_until: int
    risk_level: RiskLevel
    size_modifier: float
    action: str


@dataclass
class FearGreedSignal:
    """Fear/Greed index signal."""
    index_type: str
    value: int
    classification: str
    long_modifier: float
    short_modifier: float
    signal: str


_DB_PATH = "/root/crpbot/data/hydra/knowledge.db"


def get_sentiment(symbol: str) -> Optional[SentimentSignal]:
    """
    Get retail sentiment for a symbol (contrarian signal).

    When retail is 70%+ long, this returns SELL signal.
    When retail is 70%+ short, this returns BUY signal.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "XAUUSD")

    Returns:
        SentimentSignal with bias and confidence modifier
    """
    try:
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()

        # Search for sentiment items matching symbol
        cursor.execute("""
            SELECT title, summary
            FROM knowledge_items
            WHERE source = 'sentiment'
            AND (title LIKE ? OR title LIKE ?)
            ORDER BY created_at DESC
            LIMIT 1
        """, (f"%{symbol}%", f"%{symbol.replace('USD', '')}%"))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        title, summary = row

        # Extract percentages from title/summary
        long_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*[Ll]ong', f"{title} {summary or ''}")

        if long_match:
            long_pct = float(long_match.group(1))
        else:
            pct_match = re.search(r'(\d+)\s*%', title)
            long_pct = float(pct_match.group(1)) if pct_match else 50

        short_pct = 100 - long_pct

        # Determine contrarian bias
        if long_pct >= 70:
            bias = SentimentBias.STRONG_SELL
            confidence_mod = 1.2
            reasoning = f"Retail {long_pct:.0f}% long - fade the crowd (SELL)"
        elif long_pct >= 60:
            bias = SentimentBias.SELL
            confidence_mod = 1.1
            reasoning = f"Retail majority long ({long_pct:.0f}%) - slight SELL edge"
        elif short_pct >= 70:
            bias = SentimentBias.STRONG_BUY
            confidence_mod = 1.2
            reasoning = f"Retail {short_pct:.0f}% short - fade the crowd (BUY)"
        elif short_pct >= 60:
            bias = SentimentBias.BUY
            confidence_mod = 1.1
            reasoning = f"Retail majority short ({short_pct:.0f}%) - slight BUY edge"
        else:
            bias = SentimentBias.NEUTRAL
            confidence_mod = 1.0
            reasoning = "No extreme retail positioning"

        return SentimentSignal(
            symbol=symbol,
            long_pct=long_pct,
            short_pct=short_pct,
            bias=bias,
            confidence_modifier=confidence_mod,
            reasoning=reasoning
        )

    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return None


def get_central_bank_risk(symbol: str) -> Optional[CentralBankRisk]:
    """
    Check if symbol is affected by upcoming central bank meetings.

    Args:
        symbol: Trading symbol

    Returns:
        CentralBankRisk with risk level and size modifier
    """
    try:
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT title, summary
            FROM knowledge_items
            WHERE source = 'economic'
            ORDER BY created_at DESC
            LIMIT 10
        """)

        rows = cursor.fetchall()
        conn.close()

        # Symbol to affected banks mapping
        symbol_banks = {
            "XAUUSD": ["FOMC", "Federal Reserve"],
            "EURUSD": ["FOMC", "ECB", "Federal Reserve"],
            "GBPUSD": ["FOMC", "BOE", "Federal Reserve"],
            "USDJPY": ["FOMC", "BOJ", "Federal Reserve"],
            "US30": ["FOMC", "Federal Reserve"],
            "US30.cash": ["FOMC", "Federal Reserve"],
            "NAS100": ["FOMC", "Federal Reserve"],
            "US100.cash": ["FOMC", "Federal Reserve"],
            "BTCUSD": ["FOMC", "Federal Reserve"],
        }

        relevant_banks = symbol_banks.get(symbol, ["FOMC"])

        for row in rows:
            title, summary = row
            full_text = f"{title} {summary or ''}"

            for bank in relevant_banks:
                if bank.lower() in full_text.lower():
                    # Extract days until meeting
                    days_match = re.search(r'\((\d+)d\)', title)
                    if days_match:
                        days_until = int(days_match.group(1))
                    else:
                        days_until = 14  # Default

                    # Determine risk level
                    if days_until <= 1:
                        risk_level = RiskLevel.CRITICAL
                        size_mod = 0.0
                        action = "NO TRADING - Central bank meeting imminent"
                    elif days_until <= 3:
                        risk_level = RiskLevel.HIGH
                        size_mod = 0.5
                        action = "REDUCE SIZE 50% - Meeting approaching"
                    elif days_until <= 7:
                        risk_level = RiskLevel.ELEVATED
                        size_mod = 0.75
                        action = "CAUTION - Meeting this week"
                    else:
                        risk_level = RiskLevel.NORMAL
                        size_mod = 1.0
                        action = "Normal trading"

                    return CentralBankRisk(
                        bank=bank,
                        days_until=days_until,
                        risk_level=risk_level,
                        size_modifier=size_mod,
                        action=action
                    )

        return None

    except Exception as e:
        logger.error(f"Error checking central bank risk: {e}")
        return None


def get_fear_greed(asset_type: str = "crypto") -> Optional[FearGreedSignal]:
    """
    Get current Fear & Greed index.

    Args:
        asset_type: 'crypto' or 'stocks'

    Returns:
        FearGreedSignal with trading modifiers
    """
    try:
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()

        search_term = "Crypto" if asset_type == "crypto" else "CNN"

        cursor.execute("""
            SELECT title, summary
            FROM knowledge_items
            WHERE source = 'sentiment'
            AND title LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (f"%{search_term}%Fear%Greed%",))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        title, _ = row

        # Extract value: "Crypto Fear & Greed: 29 (Fear)"
        match = re.search(r'(\d+)\s*\((\w+)\)', title)
        if match:
            value = int(match.group(1))
            classification = match.group(2)
        else:
            value = 50
            classification = "Neutral"

        # Calculate modifiers
        if value <= 25:
            long_mod, short_mod = 1.3, 0.7
            signal = "STRONG BUY - Extreme Fear"
        elif value <= 40:
            long_mod, short_mod = 1.1, 0.9
            signal = "BUY bias - Fear zone"
        elif value <= 60:
            long_mod, short_mod = 1.0, 1.0
            signal = "NEUTRAL"
        elif value <= 75:
            long_mod, short_mod = 0.9, 1.1
            signal = "CAUTION - Greed building"
        else:
            long_mod, short_mod = 0.7, 1.3
            signal = "STRONG SELL - Extreme Greed"

        return FearGreedSignal(
            index_type=asset_type,
            value=value,
            classification=classification,
            long_modifier=long_mod,
            short_modifier=short_mod,
            signal=signal
        )

    except Exception as e:
        logger.error(f"Error getting fear/greed: {e}")
        return None


def get_trading_context(symbol: str, direction: str) -> Dict[str, Any]:
    """
    Get full trading context for a potential trade.

    Combines sentiment, central bank risk, and fear/greed
    to provide an overall confidence modifier.

    Args:
        symbol: Trading symbol
        direction: "BUY" or "SELL"

    Returns:
        Dict with context and confidence_modifier (0.0-1.5)
    """
    context = {
        "symbol": symbol,
        "direction": direction,
        "sentiment": None,
        "central_bank": None,
        "fear_greed": None,
        "confidence_modifier": 1.0,
        "warnings": [],
        "recommendations": []
    }

    # Sentiment
    sentiment = get_sentiment(symbol)
    if sentiment:
        context["sentiment"] = {
            "bias": sentiment.bias.value,
            "long_pct": sentiment.long_pct,
            "reasoning": sentiment.reasoning
        }

        if direction.upper() == "BUY":
            if sentiment.bias in (SentimentBias.STRONG_BUY, SentimentBias.BUY):
                context["confidence_modifier"] *= sentiment.confidence_modifier
                context["recommendations"].append(sentiment.reasoning)
            elif sentiment.bias in (SentimentBias.STRONG_SELL, SentimentBias.SELL):
                context["confidence_modifier"] *= (1 / sentiment.confidence_modifier)
                context["warnings"].append(f"Sentiment against BUY: {sentiment.reasoning}")
        else:
            if sentiment.bias in (SentimentBias.STRONG_SELL, SentimentBias.SELL):
                context["confidence_modifier"] *= sentiment.confidence_modifier
                context["recommendations"].append(sentiment.reasoning)
            elif sentiment.bias in (SentimentBias.STRONG_BUY, SentimentBias.BUY):
                context["confidence_modifier"] *= (1 / sentiment.confidence_modifier)
                context["warnings"].append(f"Sentiment against SELL: {sentiment.reasoning}")

    # Central bank risk
    cb_risk = get_central_bank_risk(symbol)
    if cb_risk:
        context["central_bank"] = {
            "bank": cb_risk.bank,
            "days_until": cb_risk.days_until,
            "risk_level": cb_risk.risk_level.value,
            "action": cb_risk.action
        }
        context["confidence_modifier"] *= cb_risk.size_modifier

        if cb_risk.risk_level == RiskLevel.CRITICAL:
            context["warnings"].append(f"CRITICAL: {cb_risk.bank} in {cb_risk.days_until}d - SKIP TRADE")
        elif cb_risk.risk_level == RiskLevel.HIGH:
            context["warnings"].append(f"HIGH: {cb_risk.bank} in {cb_risk.days_until}d - Half size")

    # Fear/Greed
    asset_type = "crypto" if symbol in ("BTCUSD", "ETHUSD") else "stocks"
    fg = get_fear_greed(asset_type)
    if fg:
        context["fear_greed"] = {
            "value": fg.value,
            "classification": fg.classification,
            "signal": fg.signal
        }
        if direction.upper() == "BUY":
            context["confidence_modifier"] *= fg.long_modifier
        else:
            context["confidence_modifier"] *= fg.short_modifier

    # Cap modifier
    context["confidence_modifier"] = max(0.0, min(1.5, context["confidence_modifier"]))

    return context

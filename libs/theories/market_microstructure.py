"""
Market Microstructure Theory - V7 Ultimate 8th Theory

Combines multiple market microstructure signals:
1. Fear & Greed Index (sentiment/contrarian)
2. Binance Funding Rates (futures bias)
3. FRED Macro Data (USD, rates, VIX)
4. CryptoCompare News Sentiment

This provides a holistic view of market structure beyond price action.
"""

from typing import Dict, Any, Optional
from loguru import logger

# Import data clients
try:
    from libs.data.fear_greed_client import FearGreedClient
except ImportError:
    logger.warning("Fear & Greed client not available")
    FearGreedClient = None

try:
    from libs.data.binance_client import BinanceClient
except ImportError:
    logger.warning("Binance client not available")
    BinanceClient = None

try:
    from libs.data.fred_client import FREDClient
except ImportError:
    logger.warning("FRED client not available")
    FREDClient = None

try:
    from libs.data.cryptocompare_client import CryptoCompareClient
except ImportError:
    logger.warning("CryptoCompare client not available")
    CryptoCompareClient = None


class MarketMicrostructure:
    """
    Analyze market microstructure using multiple data sources

    Combines:
    - Sentiment (Fear & Greed, News)
    - Derivatives (Funding rates, OI)
    - Macro (DXY, Fed rates, VIX)
    """

    def __init__(self, fred_api_key: Optional[str] = None, cryptocompare_api_key: Optional[str] = None):
        """
        Initialize Market Microstructure analyzer

        Args:
            fred_api_key: FRED API key (or set FRED_API_KEY env var)
            cryptocompare_api_key: CryptoCompare API key (optional, has free tier)
        """
        # Initialize clients
        self.fear_greed = FearGreedClient() if FearGreedClient else None
        self.binance = BinanceClient() if BinanceClient else None
        self.fred = FREDClient(fred_api_key) if FREDClient and fred_api_key else None
        self.cryptocompare = CryptoCompareClient(cryptocompare_api_key) if CryptoCompareClient else None

        # Track which clients are available
        self.available_clients = {
            'fear_greed': self.fear_greed is not None,
            'binance': self.binance is not None,
            'fred': self.fred is not None,
            'cryptocompare': self.cryptocompare is not None
        }

        logger.info(
            f"Market Microstructure initialized: "
            f"Fear&Greed={self.available_clients['fear_greed']}, "
            f"Binance={self.available_clients['binance']}, "
            f"FRED={self.available_clients['fred']}, "
            f"CryptoCompare={self.available_clients['cryptocompare']}"
        )

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive market microstructure analysis

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Dictionary with all microstructure metrics and composite signal
        """
        result = {
            'symbol': symbol,
            'fear_greed': None,
            'funding_rate': None,
            'macro': None,
            'news_sentiment': None,
            'composite_signal': 'neutral',
            'composite_score': 0,
            'interpretation': ''
        }

        # Collect all signals
        signals = []

        # 1. Fear & Greed Index
        if self.fear_greed:
            try:
                fg = self.fear_greed.get_current_index()
                if fg:
                    result['fear_greed'] = fg
                    signals.append(self._signal_to_score(fg['signal']))
                    logger.debug(f"Fear & Greed: {fg['value']} → {fg['signal']}")
            except Exception as e:
                logger.error(f"Error fetching Fear & Greed: {e}")

        # 2. Binance Funding Rates (Note: May be geo-blocked)
        if self.binance:
            try:
                funding = self.binance.get_funding_rate(symbol)
                if funding:
                    result['funding_rate'] = funding
                    signals.append(self._signal_to_score(funding['signal']))
                    logger.debug(f"Funding Rate: {funding['funding_rate']:.4f}% → {funding['signal']}")
            except Exception as e:
                logger.debug(f"Binance funding rate unavailable (may be geo-blocked): {e}")

        # 3. FRED Macro Data
        if self.fred:
            try:
                macro = self.fred.get_macro_analysis()
                if macro:
                    result['macro'] = macro
                    signals.append(self._signal_to_score(macro['composite_signal']))
                    logger.debug(f"Macro: {macro['composite_signal']}")
            except Exception as e:
                logger.error(f"Error fetching FRED macro data: {e}")

        # 4. CryptoCompare News Sentiment
        if self.cryptocompare:
            try:
                news = self.cryptocompare.analyze_news_sentiment(symbol, hours=24)
                if news:
                    result['news_sentiment'] = news
                    signals.append(self._signal_to_score(news['sentiment']))
                    logger.debug(f"News Sentiment: {news['sentiment']} ({news['sentiment_score']:+.2f})")
            except Exception as e:
                logger.error(f"Error fetching news sentiment: {e}")

        # Calculate composite signal
        if len(signals) > 0:
            avg_score = sum(signals) / len(signals)
            result['composite_score'] = avg_score
            result['composite_signal'] = self._score_to_signal(avg_score)
            result['interpretation'] = self._interpret_composite(avg_score, len(signals))
        else:
            result['interpretation'] = "No microstructure data available"

        logger.info(
            f"{symbol} Microstructure: {result['composite_signal'].upper()} "
            f"({result['composite_score']:+.2f}, {len(signals)} signals)"
        )

        return result

    def _signal_to_score(self, signal: str) -> float:
        """
        Convert signal string to numeric score

        Args:
            signal: Signal string (bullish, bearish, neutral, etc.)

        Returns:
            Score: +1 (bullish) to -1 (bearish)
        """
        signal_map = {
            'bullish': +1.0,
            'neutral_bullish': +0.5,
            'neutral': 0.0,
            'neutral_bearish': -0.5,
            'bearish': -1.0,
            'buy': +1.0,
            'sell': -1.0,
        }
        return signal_map.get(signal.lower(), 0.0)

    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score back to signal string"""
        if score > 0.5:
            return 'bullish'
        elif score > 0.2:
            return 'neutral_bullish'
        elif score > -0.2:
            return 'neutral'
        elif score > -0.5:
            return 'neutral_bearish'
        else:
            return 'bearish'

    def _interpret_composite(self, score: float, signal_count: int) -> str:
        """Generate interpretation of composite signal"""
        signal_word = self._score_to_signal(score).replace('_', ' ').upper()

        if signal_count < 2:
            confidence = "LOW"
            note = "(limited data sources)"
        elif signal_count < 3:
            confidence = "MODERATE"
            note = "(some data sources)"
        else:
            confidence = "HIGH"
            note = "(comprehensive data)"

        if abs(score) > 0.7:
            strength = "STRONG"
        elif abs(score) > 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return (
            f"{strength} {signal_word} microstructure signal "
            f"(score: {score:+.2f}, confidence: {confidence} {note})"
        )

    def get_detailed_summary(self, symbol: str) -> str:
        """
        Get human-readable summary of market microstructure

        Args:
            symbol: Trading pair

        Returns:
            Formatted string summary
        """
        analysis = self.analyze(symbol)

        summary = f"MARKET MICROSTRUCTURE ANALYSIS: {symbol}\n"
        summary += "=" * 80 + "\n\n"

        # Fear & Greed
        if analysis['fear_greed']:
            fg = analysis['fear_greed']
            summary += f"📊 FEAR & GREED INDEX:\n"
            summary += f"   Value: {fg['value']} ({fg['classification']})\n"
            summary += f"   Signal: {fg['signal'].upper()}\n"
            summary += f"   {fg['interpretation']}\n\n"

        # Funding Rate
        if analysis['funding_rate']:
            fr = analysis['funding_rate']
            summary += f"💰 BINANCE FUNDING RATE:\n"
            summary += f"   Rate: {fr['funding_rate']:+.4f}%\n"
            summary += f"   Signal: {fr['signal'].upper()}\n"
            summary += f"   {fr['interpretation']}\n\n"

        # Macro
        if analysis['macro']:
            macro = analysis['macro']
            summary += f"🌍 MACRO ENVIRONMENT:\n"
            summary += f"   DXY: {macro['dxy']['value']:.2f} ({macro['dxy']['signal']})\n"
            summary += f"   Fed Rate: {macro['fed_rate']['value']:.2f}% ({macro['fed_rate']['signal']})\n"
            summary += f"   VIX: {macro['vix']['value']:.1f} ({macro['vix']['signal']})\n"
            summary += f"   Composite: {macro['composite_signal'].upper()}\n"
            summary += f"   {macro['composite_interpretation']}\n\n"

        # News
        if analysis['news_sentiment']:
            news = analysis['news_sentiment']
            summary += f"📰 NEWS SENTIMENT (24h):\n"
            summary += f"   Articles: {news['article_count']}\n"
            summary += f"   Sentiment: {news['sentiment'].upper()} ({news['sentiment_score']:+.2f})\n"
            summary += f"   {news['interpretation']}\n\n"

        # Composite
        summary += f"🎯 COMPOSITE SIGNAL: {analysis['composite_signal'].upper()}\n"
        summary += f"   Score: {analysis['composite_score']:+.2f}\n"
        summary += f"   {analysis['interpretation']}\n"

        return summary


# Test the analyzer if run directly
if __name__ == "__main__":
    import os

    # Get API keys from environment
    fred_key = os.getenv('FRED_API_KEY')
    cryptocompare_key = os.getenv('CRYPTOCOMPARE_API_KEY')

    # Initialize analyzer
    analyzer = MarketMicrostructure(
        fred_api_key=fred_key,
        cryptocompare_api_key=cryptocompare_key
    )

    # Analyze each symbol
    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        print(f"\n{'='*80}")
        print(analyzer.get_detailed_summary(symbol))

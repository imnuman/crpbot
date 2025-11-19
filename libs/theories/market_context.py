"""
Market Context Theory (7th Theory)
Analyzes macro market conditions using CoinGecko data.
"""
from typing import Dict, Optional
from loguru import logger


class MarketContextTheory:
    """
    7th Theory: Market Context Analysis

    Uses CoinGecko data to assess macro market conditions:
    - Market capitalization trends
    - Trading volume (liquidity)
    - Distance from all-time high
    - Market dominance and sentiment
    """

    def analyze(self, symbol: str, coingecko_data: Optional[Dict]) -> Dict:
        """
        Analyze market context using CoinGecko data.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            coingecko_data: Data from CoinGeckoClient.get_market_data()

        Returns:
            {
                'market_cap_billions': float,
                'volume_billions': float,
                'ath_distance_pct': float,
                'liquidity_score': float,      # 0-1 (high volume = high liquidity)
                'sentiment': str,              # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
                'market_strength': float,      # 0-1 (composite score)
                'notes': str
            }
        """
        if not coingecko_data:
            logger.warning(f"No CoinGecko data for {symbol}, using defaults")
            return self._default_context()

        try:
            market_cap = coingecko_data.get('market_cap', 0) / 1e9  # Convert to billions
            volume = coingecko_data.get('total_volume', 0) / 1e9
            ath_distance = coingecko_data.get('ath_distance_pct', 0)
            price_change_24h = coingecko_data.get('price_change_24h_pct', 0)

            # Calculate liquidity score (volume relative to market cap)
            liquidity_score = min(volume / (market_cap + 1e-9), 1.0) if market_cap > 0 else 0.0

            # Determine sentiment based on ATH distance and price change
            sentiment = self._calculate_sentiment(ath_distance, price_change_24h)

            # Calculate composite market strength
            market_strength = self._calculate_market_strength(
                ath_distance=ath_distance,
                liquidity_score=liquidity_score,
                price_change_24h=price_change_24h
            )

            notes = self._generate_notes(
                market_cap=market_cap,
                volume=volume,
                ath_distance=ath_distance,
                sentiment=sentiment,
                market_strength=market_strength
            )

            return {
                'market_cap_billions': round(market_cap, 2),
                'volume_billions': round(volume, 2),
                'ath_distance_pct': round(ath_distance, 2),
                'liquidity_score': round(liquidity_score, 3),
                'sentiment': sentiment,
                'market_strength': round(market_strength, 3),
                'notes': notes
            }

        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return self._default_context()

    def _calculate_sentiment(self, ath_distance: float, price_change_24h: float) -> str:
        """
        Calculate market sentiment based on price metrics.

        ATH Distance:
        - > -10%: extreme_greed (near ATH)
        - -10% to -25%: greed
        - -25% to -40%: neutral
        - -40% to -60%: fear
        - < -60%: extreme_fear

        Also considers 24h price change.
        """
        # Base sentiment on ATH distance
        if ath_distance > -10:
            base_sentiment = 'extreme_greed'
        elif ath_distance > -25:
            base_sentiment = 'greed'
        elif ath_distance > -40:
            base_sentiment = 'neutral'
        elif ath_distance > -60:
            base_sentiment = 'fear'
        else:
            base_sentiment = 'extreme_fear'

        # Adjust based on 24h price change
        if price_change_24h < -5:
            # Strong drop - increase fear
            if base_sentiment == 'extreme_greed':
                return 'greed'
            elif base_sentiment == 'greed':
                return 'neutral'
            elif base_sentiment == 'neutral':
                return 'fear'
        elif price_change_24h > 5:
            # Strong rise - increase greed
            if base_sentiment == 'extreme_fear':
                return 'fear'
            elif base_sentiment == 'fear':
                return 'neutral'
            elif base_sentiment == 'neutral':
                return 'greed'

        return base_sentiment

    def _calculate_market_strength(
        self,
        ath_distance: float,
        liquidity_score: float,
        price_change_24h: float
    ) -> float:
        """
        Calculate composite market strength score (0-1).

        Components:
        - ATH proximity (40%): Higher = closer to ATH
        - Liquidity (30%): Higher = more trading volume
        - Momentum (30%): Positive 24h change
        """
        # ATH proximity score (0-1, higher = closer to ATH)
        ath_score = max(0, min(1, (ath_distance + 100) / 100))  # -100% → 0, 0% → 1

        # Liquidity score (already 0-1)
        liq_score = liquidity_score

        # Momentum score (0-1, based on 24h change)
        if price_change_24h > 5:
            momentum_score = 1.0
        elif price_change_24h > 0:
            momentum_score = 0.5 + (price_change_24h / 10)
        elif price_change_24h > -5:
            momentum_score = 0.5 + (price_change_24h / 10)
        else:
            momentum_score = 0.0

        # Weighted average
        strength = (
            0.4 * ath_score +
            0.3 * liq_score +
            0.3 * momentum_score
        )

        return max(0.0, min(1.0, strength))

    def _generate_notes(
        self,
        market_cap: float,
        volume: float,
        ath_distance: float,
        sentiment: str,
        market_strength: float
    ) -> str:
        """Generate human-readable notes about market context."""
        notes = []

        # Market cap assessment
        if market_cap > 1000:
            notes.append(f"Massive ${market_cap:.0f}B market cap (highly liquid)")
        elif market_cap > 100:
            notes.append(f"Large ${market_cap:.0f}B market cap (liquid)")
        else:
            notes.append(f"${market_cap:.1f}B market cap")

        # Volume assessment
        volume_ratio = volume / (market_cap + 1e-9)
        if volume_ratio > 0.1:
            notes.append(f"Very high volume (${volume:.1f}B, {volume_ratio*100:.0f}% of mcap)")
        elif volume_ratio > 0.05:
            notes.append(f"High volume (${volume:.1f}B)")
        else:
            notes.append(f"Moderate volume (${volume:.1f}B)")

        # ATH distance
        if ath_distance > -10:
            notes.append(f"Near ATH ({ath_distance:.1f}% from peak) - extreme greed zone")
        elif ath_distance < -50:
            notes.append(f"Far from ATH ({ath_distance:.1f}% below) - fear/capitulation zone")
        else:
            notes.append(f"{ath_distance:.1f}% from ATH")

        # Sentiment
        notes.append(f"Sentiment: {sentiment.replace('_', ' ')}")

        # Market strength
        if market_strength > 0.7:
            notes.append("Strong market conditions")
        elif market_strength < 0.3:
            notes.append("Weak market conditions")

        return ". ".join(notes)

    def _default_context(self) -> Dict:
        """Return default context when CoinGecko data unavailable."""
        return {
            'market_cap_billions': 0.0,
            'volume_billions': 0.0,
            'ath_distance_pct': 0.0,
            'liquidity_score': 0.0,
            'sentiment': 'neutral',
            'market_strength': 0.5,
            'notes': 'CoinGecko data unavailable - using price data only'
        }

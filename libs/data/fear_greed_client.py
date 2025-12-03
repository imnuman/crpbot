"""
Fear & Greed Index API Client

Fetches crypto market sentiment from Alternative.me Fear & Greed Index.
This is a contrarian indicator - extreme fear = potential buy, extreme greed = potential sell.

API: https://api.alternative.me/fng/
Cost: FREE (no API key needed)
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")
from loguru import logger


class FearGreedClient:
    """Client for Alternative.me Fear & Greed Index API"""

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        """Initialize Fear & Greed Index client (no API key needed)"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTradingBot/1.0)'
        })
        logger.debug("Fear & Greed Index client initialized")

    def get_current_index(self) -> Optional[Dict[str, Any]]:
        """
        Get current Fear & Greed Index

        Returns:
            {
                'value': int (0-100),
                'classification': str (Extreme Fear, Fear, Neutral, Greed, Extreme Greed),
                'timestamp': datetime,
                'signal': str (buy, sell, neutral),
                'interpretation': str
            }
        """
        try:
            response = self.session.get(self.BASE_URL, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data.get('data') or len(data['data']) == 0:
                logger.error("No data in Fear & Greed Index response")
                return None

            latest = data['data'][0]
            value = int(latest['value'])
            classification = latest['value_classification']
            timestamp = datetime.fromtimestamp(int(latest['timestamp']), tz=EST)

            # Generate trading signal (contrarian approach)
            signal, interpretation = self._interpret_index(value, classification)

            result = {
                'value': value,
                'classification': classification,
                'timestamp': timestamp,
                'signal': signal,
                'interpretation': interpretation
            }

            logger.info(
                f"Fear & Greed Index: {value} ({classification}) → Signal: {signal.upper()}"
            )

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Fear & Greed Index: {e}")
            return None

    def _interpret_index(self, value: int, classification: str) -> tuple[str, str]:
        """
        Interpret Fear & Greed Index for trading signals

        Contrarian approach:
        - Extreme Fear (0-25): BUY signal (market oversold)
        - Fear (26-45): Weak BUY signal
        - Neutral (46-55): HOLD
        - Greed (56-75): Weak SELL signal
        - Extreme Greed (76-100): SELL signal (market overbought)

        Args:
            value: Index value (0-100)
            classification: Text classification

        Returns:
            (signal, interpretation)
        """
        if value <= 25:
            signal = "buy"
            interpretation = (
                f"EXTREME FEAR ({value}) - Market is panicking. "
                "Contrarian signal: Strong BUY opportunity. "
                "Historically good entry point."
            )
        elif value <= 45:
            signal = "buy"
            interpretation = (
                f"FEAR ({value}) - Market is fearful. "
                "Contrarian signal: Weak BUY opportunity. "
                "Consider accumulation."
            )
        elif value <= 55:
            signal = "neutral"
            interpretation = (
                f"NEUTRAL ({value}) - Market sentiment balanced. "
                "No strong contrarian signal. HOLD current positions."
            )
        elif value <= 75:
            signal = "sell"
            interpretation = (
                f"GREED ({value}) - Market is getting greedy. "
                "Contrarian signal: Weak SELL opportunity. "
                "Consider taking some profits."
            )
        else:
            signal = "sell"
            interpretation = (
                f"EXTREME GREED ({value}) - Market is euphoric. "
                "Contrarian signal: Strong SELL opportunity. "
                "High risk of correction."
            )

        return signal, interpretation

    def get_historical_index(self, days: int = 7) -> Optional[list[Dict[str, Any]]]:
        """
        Get historical Fear & Greed Index data

        Args:
            days: Number of days of history (1-365)

        Returns:
            List of historical data points
        """
        try:
            url = f"{self.BASE_URL}?limit={days}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data.get('data'):
                return None

            history = []
            for item in data['data']:
                value = int(item['value'])
                signal, interpretation = self._interpret_index(value, item['value_classification'])

                history.append({
                    'value': value,
                    'classification': item['value_classification'],
                    'timestamp': datetime.fromtimestamp(int(item['timestamp']), tz=EST),
                    'signal': signal
                })

            logger.debug(f"Fetched {len(history)} days of Fear & Greed Index history")
            return history

        except Exception as e:
            logger.error(f"Error fetching historical Fear & Greed Index: {e}")
            return None

    def get_trend(self, days: int = 7) -> Optional[Dict[str, Any]]:
        """
        Analyze Fear & Greed Index trend

        Args:
            days: Number of days to analyze

        Returns:
            {
                'current_value': int,
                'avg_value': float,
                'trend': str (increasing_fear, decreasing_fear, stable),
                'change': float (percentage change from average)
            }
        """
        history = self.get_historical_index(days)
        if not history or len(history) < 2:
            return None

        current = history[0]['value']
        values = [h['value'] for h in history]
        avg = sum(values) / len(values)

        # Determine trend
        if current < avg - 10:
            trend = "increasing_fear"
        elif current > avg + 10:
            trend = "decreasing_fear"
        else:
            trend = "stable"

        change = ((current - avg) / avg) * 100

        return {
            'current_value': current,
            'avg_value': avg,
            'trend': trend,
            'change': change,
            'interpretation': self._interpret_trend(trend, change)
        }

    def _interpret_trend(self, trend: str, change: float) -> str:
        """Interpret Fear & Greed trend"""
        if trend == "increasing_fear":
            return (
                f"Fear is INCREASING ({change:+.1f}% from avg). "
                "Sentiment deteriorating → Contrarian signal strengthening."
            )
        elif trend == "decreasing_fear":
            return (
                f"Fear is DECREASING ({change:+.1f}% from avg). "
                "Sentiment improving → Market recovering."
            )
        else:
            return "Sentiment is STABLE. No strong trend."


# Test the client if run directly
if __name__ == "__main__":
    client = FearGreedClient()

    print("=== Current Fear & Greed Index ===")
    index = client.get_current_index()
    if index:
        print(f"Value: {index['value']}")
        print(f"Classification: {index['classification']}")
        print(f"Signal: {index['signal'].upper()}")
        print(f"Interpretation: {index['interpretation']}")
        print(f"Timestamp: {index['timestamp']}")

    print("\n=== 7-Day Trend ===")
    trend = client.get_trend(days=7)
    if trend:
        print(f"Current: {trend['current_value']}")
        print(f"7-day avg: {trend['avg_value']:.1f}")
        print(f"Trend: {trend['trend']}")
        print(f"Change: {trend['change']:+.1f}%")
        print(f"Interpretation: {trend['interpretation']}")

"""
Binance Futures API Client

Fetches funding rates, open interest, and long/short ratios from Binance.
This data shows futures market bias and can predict spot price movements.

API: https://binance-docs.github.io/apidocs/
Cost: FREE (no API key needed for public data)
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")
from loguru import logger


class BinanceClient:
    """Client for Binance Futures public API"""

    # Symbol mapping: Our format → Binance format
    SYMBOL_MAP = {
        'BTC-USD': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT',
        'SOL-USD': 'SOLUSDT'
    }

    FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1"

    def __init__(self):
        """Initialize Binance client (no API key needed for public data)"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTradingBot/1.0)'
        })
        logger.debug("Binance Futures client initialized")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert our symbol format to Binance format"""
        return self.SYMBOL_MAP.get(symbol, symbol.replace('-', ''))

    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current funding rate for a symbol

        Funding Rate Interpretation:
        - Positive (> 0.01%): Longs pay shorts → Market is bullish → Potential long squeeze
        - Negative (< -0.01%): Shorts pay longs → Market is bearish → Potential short squeeze
        - Near zero: Balanced market

        High funding rate + high OI = potential liquidation cascade

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            {
                'symbol': str,
                'funding_rate': float (as percentage),
                'next_funding_time': datetime,
                'signal': str (bullish, bearish, neutral),
                'interpretation': str
            }
        """
        try:
            binance_symbol = self._convert_symbol(symbol)
            url = f"{self.FUTURES_BASE_URL}/premiumIndex"

            response = self.session.get(
                url,
                params={'symbol': binance_symbol},
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Funding rate is returned as decimal (0.0001 = 0.01%)
            funding_rate = float(data['lastFundingRate']) * 100  # Convert to percentage
            next_funding_time = datetime.fromtimestamp(int(data['nextFundingTime']) / 1000, tz=EST)

            # Interpret funding rate
            signal, interpretation = self._interpret_funding_rate(funding_rate)

            result = {
                'symbol': symbol,
                'funding_rate': funding_rate,
                'next_funding_time': next_funding_time,
                'signal': signal,
                'interpretation': interpretation
            }

            logger.info(
                f"{symbol} Funding Rate: {funding_rate:+.4f}% → Signal: {signal.upper()}"
            )

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing funding rate for {symbol}: {e}")
            return None

    def _interpret_funding_rate(self, rate: float) -> tuple[str, str]:
        """
        Interpret funding rate for trading signals

        Args:
            rate: Funding rate as percentage

        Returns:
            (signal, interpretation)
        """
        if rate > 0.05:  # Very high positive funding
            signal = "bearish"
            interpretation = (
                f"HIGH POSITIVE FUNDING ({rate:+.4f}%) - Longs are paying shorts heavily. "
                "Market extremely bullish, but risk of long squeeze. "
                "Potential SELL signal (contrarian)."
            )
        elif rate > 0.02:  # Moderately positive funding
            signal = "neutral_bearish"
            interpretation = (
                f"POSITIVE FUNDING ({rate:+.4f}%) - Longs are paying shorts. "
                "Market is bullish, but funding cost building. "
                "Watch for potential reversal."
            )
        elif rate > -0.02:  # Near neutral
            signal = "neutral"
            interpretation = (
                f"NEUTRAL FUNDING ({rate:+.4f}%) - Market is balanced. "
                "No strong bias from futures market."
            )
        elif rate > -0.05:  # Moderately negative funding
            signal = "neutral_bullish"
            interpretation = (
                f"NEGATIVE FUNDING ({rate:+.4f}%) - Shorts are paying longs. "
                "Market is bearish, but potential for short squeeze. "
                "Watch for reversal opportunity."
            )
        else:  # Very negative funding
            signal = "bullish"
            interpretation = (
                f"HIGH NEGATIVE FUNDING ({rate:+.4f}%) - Shorts paying longs heavily. "
                "Market extremely bearish, high risk of short squeeze. "
                "Potential BUY signal (contrarian)."
            )

        return signal, interpretation

    def get_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get open interest for a symbol

        Open Interest = Total value of outstanding futures contracts
        - Rising OI + rising price = bullish (new longs entering)
        - Rising OI + falling price = bearish (new shorts entering)
        - Falling OI + rising price = short squeeze
        - Falling OI + falling price = long liquidation

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            {
                'symbol': str,
                'open_interest': float (in USDT),
                'open_interest_value': float,
                'interpretation': str
            }
        """
        try:
            binance_symbol = self._convert_symbol(symbol)
            url = f"{self.FUTURES_BASE_URL}/openInterest"

            response = self.session.get(
                url,
                params={'symbol': binance_symbol},
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            open_interest = float(data['openInterest'])
            # Get current price to calculate notional value
            price_data = self.get_ticker_price(symbol)
            if price_data:
                oi_value = open_interest * price_data['price']
            else:
                oi_value = 0

            result = {
                'symbol': symbol,
                'open_interest': open_interest,
                'open_interest_value': oi_value,
                'interpretation': f"Open Interest: {open_interest:,.0f} contracts (${oi_value:,.0f})"
            }

            logger.info(
                f"{symbol} Open Interest: {open_interest:,.0f} contracts (${oi_value:,.0f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None

    def get_ticker_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current ticker price from Binance spot market"""
        try:
            binance_symbol = self._convert_symbol(symbol)
            url = "https://api.binance.com/api/v3/ticker/price"

            response = self.session.get(
                url,
                params={'symbol': binance_symbol},
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            return {
                'symbol': symbol,
                'price': float(data['price'])
            }

        except Exception as e:
            logger.error(f"Error fetching ticker price for {symbol}: {e}")
            return None

    def get_long_short_ratio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get long/short accounts ratio

        This shows the ratio of accounts with long vs short positions.
        - Ratio > 1: More accounts are long (bullish sentiment)
        - Ratio < 1: More accounts are short (bearish sentiment)
        - Extreme ratios (>3 or <0.3) = potential reversal

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            {
                'symbol': str,
                'long_short_ratio': float,
                'long_account_pct': float,
                'short_account_pct': float,
                'signal': str,
                'interpretation': str
            }
        """
        try:
            binance_symbol = self._convert_symbol(symbol)
            url = f"{self.FUTURES_BASE_URL}/globalLongShortAccountRatio"

            response = self.session.get(
                url,
                params={
                    'symbol': binance_symbol,
                    'period': '5m',  # 5-minute data (most recent)
                    'limit': 1
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            if not data or len(data) == 0:
                return None

            latest = data[0]
            ratio = float(latest['longShortRatio'])
            long_account = float(latest['longAccount'])
            short_account = float(latest['shortAccount'])

            # Calculate percentages
            total = long_account + short_account
            long_pct = (long_account / total) * 100 if total > 0 else 50
            short_pct = (short_account / total) * 100 if total > 0 else 50

            # Interpret ratio
            signal, interpretation = self._interpret_long_short_ratio(ratio)

            result = {
                'symbol': symbol,
                'long_short_ratio': ratio,
                'long_account_pct': long_pct,
                'short_account_pct': short_pct,
                'signal': signal,
                'interpretation': interpretation
            }

            logger.info(
                f"{symbol} Long/Short Ratio: {ratio:.2f} "
                f"(Long: {long_pct:.1f}%, Short: {short_pct:.1f}%) → {signal.upper()}"
            )

            return result

        except Exception as e:
            logger.error(f"Error fetching long/short ratio for {symbol}: {e}")
            return None

    def _interpret_long_short_ratio(self, ratio: float) -> tuple[str, str]:
        """Interpret long/short ratio (contrarian approach)"""
        if ratio > 3:
            signal = "bearish"
            interpretation = (
                f"EXTREME LONG BIAS ({ratio:.2f}:1) - Way too many longs. "
                "Contrarian signal: Potential SELL (risk of long squeeze)."
            )
        elif ratio > 1.5:
            signal = "neutral_bearish"
            interpretation = (
                f"LONG BIAS ({ratio:.2f}:1) - More longs than shorts. "
                "Bullish sentiment, but watch for reversal."
            )
        elif ratio > 0.67:
            signal = "neutral"
            interpretation = (
                f"BALANCED ({ratio:.2f}:1) - Long/short ratio is balanced."
            )
        elif ratio > 0.33:
            signal = "neutral_bullish"
            interpretation = (
                f"SHORT BIAS ({ratio:.2f}:1) - More shorts than longs. "
                "Bearish sentiment, but watch for short squeeze."
            )
        else:
            signal = "bullish"
            interpretation = (
                f"EXTREME SHORT BIAS ({ratio:.2f}:1) - Way too many shorts. "
                "Contrarian signal: Potential BUY (risk of short squeeze)."
            )

        return signal, interpretation

    def get_market_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive futures market analysis

        Combines funding rate, open interest, and long/short ratio
        to provide holistic view of futures market sentiment.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Dictionary with all futures metrics and composite signal
        """
        funding = self.get_funding_rate(symbol)
        open_interest = self.get_open_interest(symbol)
        ls_ratio = self.get_long_short_ratio(symbol)

        if not funding or not open_interest or not ls_ratio:
            logger.warning(f"Could not fetch complete futures data for {symbol}")
            return None

        # Combine signals
        signals = [funding['signal'], ls_ratio['signal']]
        bullish_count = sum(1 for s in signals if 'bullish' in s)
        bearish_count = sum(1 for s in signals if 'bearish' in s)

        if bullish_count > bearish_count:
            composite_signal = "bullish"
        elif bearish_count > bullish_count:
            composite_signal = "bearish"
        else:
            composite_signal = "neutral"

        return {
            'symbol': symbol,
            'funding_rate': funding,
            'open_interest': open_interest,
            'long_short_ratio': ls_ratio,
            'composite_signal': composite_signal,
            'timestamp': datetime.now(EST)
        }


# Test the client if run directly
if __name__ == "__main__":
    client = BinanceClient()

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        print(f"\n{'='*80}")
        print(f"BINANCE FUTURES ANALYSIS: {symbol}")
        print(f"{'='*80}")

        analysis = client.get_market_analysis(symbol)
        if analysis:
            print(f"\n📊 Funding Rate:")
            print(f"   Rate: {analysis['funding_rate']['funding_rate']:+.4f}%")
            print(f"   Signal: {analysis['funding_rate']['signal'].upper()}")
            print(f"   {analysis['funding_rate']['interpretation']}")

            print(f"\n📈 Open Interest:")
            print(f"   {analysis['open_interest']['interpretation']}")

            print(f"\n⚖️  Long/Short Ratio:")
            print(f"   Ratio: {analysis['long_short_ratio']['long_short_ratio']:.2f}:1")
            print(f"   Long: {analysis['long_short_ratio']['long_account_pct']:.1f}%")
            print(f"   Short: {analysis['long_short_ratio']['short_account_pct']:.1f}%")
            print(f"   Signal: {analysis['long_short_ratio']['signal'].upper()}")

            print(f"\n🎯 Composite Signal: {analysis['composite_signal'].upper()}")

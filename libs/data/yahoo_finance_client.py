"""
Yahoo Finance Macro Data Client
Provides macro-economic indicators for trading context

This client uses yfinance (unofficial Yahoo Finance library) to fetch:
- DXY (US Dollar Index) - currency strength
- Gold prices - safe haven/inflation indicator
- 10-Year Treasury yields - risk-free rate
- S&P 500 - equity market sentiment

Note: yfinance is unofficial and community-maintained. Data reliability not guaranteed.
"""

import yfinance as yf
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Client for Yahoo Finance macro-economic data (unofficial API)"""

    # Ticker symbols for macro indicators
    TICKERS = {
        'dxy': 'DX-Y.NYB',      # US Dollar Index
        'gold': 'GC=F',          # Gold Futures
        'treasury_10y': '^TNX',  # 10-Year Treasury Yield
        'sp500': '^GSPC'         # S&P 500 Index
    }

    def __init__(self):
        """Initialize Yahoo Finance client (no API key needed)"""
        logger.info("YahooFinance client initialized (unofficial yfinance library)")

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for a ticker

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d', interval='1m')

            if data.empty:
                logger.warning(f"No data available for {ticker}")
                return None

            current_price = data['Close'].iloc[-1]
            logger.debug(f"{ticker} current price: {current_price:.2f}")
            return float(current_price)

        except Exception as e:
            logger.error(f"Failed to fetch current price for {ticker}: {e}")
            return None

    def get_price_change(self, ticker: str, period: str = '1d') -> Optional[float]:
        """
        Get price change percentage for a ticker

        Args:
            ticker: Yahoo Finance ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '1y')

        Returns:
            Price change percentage or None if unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if len(data) < 2:
                logger.warning(f"Insufficient data for {ticker} (period: {period})")
                return None

            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100

            logger.debug(f"{ticker} {period} change: {change_pct:+.2f}%")
            return float(change_pct)

        except Exception as e:
            logger.error(f"Failed to fetch price change for {ticker}: {e}")
            return None

    def get_dxy_data(self) -> Dict:
        """
        Get US Dollar Index (DXY) data

        Returns:
            {
                'current': float,
                'change_1d': float (%),
                'change_5d': float (%),
                'change_1mo': float (%),
                'timestamp': datetime
            }
        """
        ticker = self.TICKERS['dxy']

        try:
            current = self.get_current_price(ticker)
            change_1d = self.get_price_change(ticker, '1d')
            change_5d = self.get_price_change(ticker, '5d')
            change_1mo = self.get_price_change(ticker, '1mo')

            data = {
                'current': current or 0.0,
                'change_1d': change_1d or 0.0,
                'change_5d': change_5d or 0.0,
                'change_1mo': change_1mo or 0.0,
                'timestamp': datetime.now()
            }

            logger.info(
                f"DXY: {data['current']:.2f} | "
                f"1D: {data['change_1d']:+.2f}% | "
                f"5D: {data['change_5d']:+.2f}% | "
                f"1M: {data['change_1mo']:+.2f}%"
            )

            return data

        except Exception as e:
            logger.error(f"Failed to get DXY data: {e}")
            return {
                'current': 0.0,
                'change_1d': 0.0,
                'change_5d': 0.0,
                'change_1mo': 0.0,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_gold_data(self) -> Dict:
        """
        Get Gold futures price data

        Returns:
            {
                'current': float,
                'change_1d': float (%),
                'change_5d': float (%),
                'change_1mo': float (%),
                'timestamp': datetime
            }
        """
        ticker = self.TICKERS['gold']

        try:
            current = self.get_current_price(ticker)
            change_1d = self.get_price_change(ticker, '1d')
            change_5d = self.get_price_change(ticker, '5d')
            change_1mo = self.get_price_change(ticker, '1mo')

            data = {
                'current': current or 0.0,
                'change_1d': change_1d or 0.0,
                'change_5d': change_5d or 0.0,
                'change_1mo': change_1mo or 0.0,
                'timestamp': datetime.now()
            }

            logger.info(
                f"Gold: ${data['current']:.2f} | "
                f"1D: {data['change_1d']:+.2f}% | "
                f"5D: {data['change_5d']:+.2f}% | "
                f"1M: {data['change_1mo']:+.2f}%"
            )

            return data

        except Exception as e:
            logger.error(f"Failed to get Gold data: {e}")
            return {
                'current': 0.0,
                'change_1d': 0.0,
                'change_5d': 0.0,
                'change_1mo': 0.0,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_treasury_10y_data(self) -> Dict:
        """
        Get 10-Year Treasury Yield data

        Returns:
            {
                'current': float (yield in %),
                'change_1d': float (%),
                'change_5d': float (%),
                'change_1mo': float (%),
                'timestamp': datetime
            }
        """
        ticker = self.TICKERS['treasury_10y']

        try:
            current = self.get_current_price(ticker)
            change_1d = self.get_price_change(ticker, '1d')
            change_5d = self.get_price_change(ticker, '5d')
            change_1mo = self.get_price_change(ticker, '1mo')

            data = {
                'current': current or 0.0,
                'change_1d': change_1d or 0.0,
                'change_5d': change_5d or 0.0,
                'change_1mo': change_1mo or 0.0,
                'timestamp': datetime.now()
            }

            logger.info(
                f"10Y Treasury: {data['current']:.2f}% | "
                f"1D: {data['change_1d']:+.2f}% | "
                f"5D: {data['change_5d']:+.2f}% | "
                f"1M: {data['change_1mo']:+.2f}%"
            )

            return data

        except Exception as e:
            logger.error(f"Failed to get 10Y Treasury data: {e}")
            return {
                'current': 0.0,
                'change_1d': 0.0,
                'change_5d': 0.0,
                'change_1mo': 0.0,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_sp500_data(self) -> Dict:
        """
        Get S&P 500 Index data

        Returns:
            {
                'current': float,
                'change_1d': float (%),
                'change_5d': float (%),
                'change_1mo': float (%),
                'timestamp': datetime
            }
        """
        ticker = self.TICKERS['sp500']

        try:
            current = self.get_current_price(ticker)
            change_1d = self.get_price_change(ticker, '1d')
            change_5d = self.get_price_change(ticker, '5d')
            change_1mo = self.get_price_change(ticker, '1mo')

            data = {
                'current': current or 0.0,
                'change_1d': change_1d or 0.0,
                'change_5d': change_5d or 0.0,
                'change_1mo': change_1mo or 0.0,
                'timestamp': datetime.now()
            }

            logger.info(
                f"S&P 500: {data['current']:.2f} | "
                f"1D: {data['change_1d']:+.2f}% | "
                f"5D: {data['change_5d']:+.2f}% | "
                f"1M: {data['change_1mo']:+.2f}%"
            )

            return data

        except Exception as e:
            logger.error(f"Failed to get S&P 500 data: {e}")
            return {
                'current': 0.0,
                'change_1d': 0.0,
                'change_5d': 0.0,
                'change_1mo': 0.0,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_macro_environment(self) -> Dict:
        """
        Get comprehensive macro-economic environment snapshot

        Returns:
            {
                'dxy': {current, change_1d, change_5d, change_1mo},
                'gold': {current, change_1d, change_5d, change_1mo},
                'treasury_10y': {current, change_1d, change_5d, change_1mo},
                'sp500': {current, change_1d, change_5d, change_1mo},
                'risk_sentiment': str ('risk_on' | 'risk_off' | 'neutral'),
                'timestamp': datetime
            }
        """
        try:
            dxy = self.get_dxy_data()
            gold = self.get_gold_data()
            treasury = self.get_treasury_10y_data()
            sp500 = self.get_sp500_data()

            # Calculate risk sentiment based on macro indicators
            # Risk-on: stocks up, gold down, yields up
            # Risk-off: stocks down, gold up, yields down
            risk_score = 0

            if sp500['change_1d'] > 0:
                risk_score += 1
            else:
                risk_score -= 1

            if gold['change_1d'] < 0:
                risk_score += 1
            else:
                risk_score -= 1

            if treasury['change_1d'] > 0:
                risk_score += 0.5
            else:
                risk_score -= 0.5

            if risk_score > 1:
                risk_sentiment = 'risk_on'
            elif risk_score < -1:
                risk_sentiment = 'risk_off'
            else:
                risk_sentiment = 'neutral'

            macro_env = {
                'dxy': dxy,
                'gold': gold,
                'treasury_10y': treasury,
                'sp500': sp500,
                'risk_sentiment': risk_sentiment,
                'timestamp': datetime.now()
            }

            logger.info(f"Macro environment: {risk_sentiment.upper()}")
            return macro_env

        except Exception as e:
            logger.error(f"Failed to get macro environment: {e}")
            return {
                'dxy': {},
                'gold': {},
                'treasury_10y': {},
                'sp500': {},
                'risk_sentiment': 'neutral',
                'timestamp': datetime.now(),
                'error': str(e)
            }


# Convenience function for V7 runtime
def get_macro_data() -> Dict:
    """
    Get macro-economic data snapshot

    Returns:
        Comprehensive macro environment dictionary
    """
    client = YahooFinanceClient()
    return client.get_macro_environment()


if __name__ == "__main__":
    # Test the Yahoo Finance client
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Yahoo Finance Macro Data Client - Test Run")
    print("=" * 80)
    print("\nNote: Using unofficial yfinance library (community-maintained)")
    print("Data reliability not guaranteed by Yahoo Finance.\n")

    try:
        client = YahooFinanceClient()

        print("=" * 80)
        print("Fetching Macro-Economic Indicators...")
        print("=" * 80)

        # Test DXY
        print("\n1. US Dollar Index (DXY):")
        dxy = client.get_dxy_data()
        if 'error' not in dxy:
            print(f"   Current:      {dxy['current']:.2f}")
            print(f"   1-Day Change: {dxy['change_1d']:+.2f}%")
            print(f"   5-Day Change: {dxy['change_5d']:+.2f}%")
            print(f"   1-Mo Change:  {dxy['change_1mo']:+.2f}%")
        else:
            print(f"   ERROR: {dxy['error']}")

        # Test Gold
        print("\n2. Gold Futures (GC=F):")
        gold = client.get_gold_data()
        if 'error' not in gold:
            print(f"   Current:      ${gold['current']:.2f}")
            print(f"   1-Day Change: {gold['change_1d']:+.2f}%")
            print(f"   5-Day Change: {gold['change_5d']:+.2f}%")
            print(f"   1-Mo Change:  {gold['change_1mo']:+.2f}%")
        else:
            print(f"   ERROR: {gold['error']}")

        # Test 10Y Treasury
        print("\n3. 10-Year Treasury Yield (^TNX):")
        treasury = client.get_treasury_10y_data()
        if 'error' not in treasury:
            print(f"   Current:      {treasury['current']:.2f}%")
            print(f"   1-Day Change: {treasury['change_1d']:+.2f}%")
            print(f"   5-Day Change: {treasury['change_5d']:+.2f}%")
            print(f"   1-Mo Change:  {treasury['change_1mo']:+.2f}%")
        else:
            print(f"   ERROR: {treasury['error']}")

        # Test S&P 500
        print("\n4. S&P 500 Index (^GSPC):")
        sp500 = client.get_sp500_data()
        if 'error' not in sp500:
            print(f"   Current:      {sp500['current']:.2f}")
            print(f"   1-Day Change: {sp500['change_1d']:+.2f}%")
            print(f"   5-Day Change: {sp500['change_5d']:+.2f}%")
            print(f"   1-Mo Change:  {sp500['change_1mo']:+.2f}%")
        else:
            print(f"   ERROR: {sp500['error']}")

        # Test comprehensive macro environment
        print("\n" + "=" * 80)
        print("Comprehensive Macro Environment Analysis")
        print("=" * 80)

        macro_env = client.get_macro_environment()

        print(f"\nRisk Sentiment: {macro_env['risk_sentiment'].upper()}")
        print("\nInterpretation:")
        if macro_env['risk_sentiment'] == 'risk_on':
            print("  - Markets favoring risk assets (stocks up, gold down)")
            print("  - Crypto may benefit from increased risk appetite")
        elif macro_env['risk_sentiment'] == 'risk_off':
            print("  - Markets favoring safe havens (stocks down, gold up)")
            print("  - Crypto may face selling pressure")
        else:
            print("  - Mixed signals across markets")
            print("  - Crypto direction uncertain")

        print("\n" + "=" * 80)
        print("Test complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

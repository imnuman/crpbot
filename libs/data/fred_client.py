"""
FRED (Federal Reserve Economic Data) API Client

Fetches macro economic data that impacts crypto markets:
- US Dollar Index (DXY)
- Federal Funds Rate
- CPI (inflation)
- S&P 500 (risk sentiment)

API: https://fred.stlouisfed.org/docs/api/fred/
Cost: FREE (requires API key from https://fredaccount.stlouisfed.org/apikeys)
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger
import os


class FREDClient:
    """Client for Federal Reserve Economic Data (FRED) API"""

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Economic series IDs
    SERIES = {
        'dxy': 'DTWEXBGS',  # US Dollar Index (Trade Weighted)
        'fed_rate': 'FEDFUNDS',  # Federal Funds Effective Rate
        'cpi': 'CPIAUCSL',  # Consumer Price Index (inflation)
        'sp500': 'SP500',  # S&P 500 Index
        'vix': 'VIXCLS',  # CBOE Volatility Index
        'm2_money': 'WM2NS',  # M2 Money Supply
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            logger.warning(
                "No FRED API key provided. "
                "Get one at https://fredaccount.stlouisfed.org/apikeys"
            )

        self.session = requests.Session()
        logger.debug("FRED API client initialized")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make request to FRED API"""
        if not self.api_key:
            logger.error("Cannot make FRED request without API key")
            return None

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            params['api_key'] = self.api_key
            params['file_type'] = 'json'

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data from {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing FRED response: {e}")
            return None

    def get_latest_value(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latest value for a FRED series

        Args:
            series_id: FRED series ID (e.g., 'DTWEXBGS' for DXY)

        Returns:
            {
                'series_id': str,
                'value': float,
                'date': datetime,
                'title': str
            }
        """
        # Get series info (title)
        info = self._make_request('series', {'series_id': series_id})
        if not info or 'seriess' not in info:
            return None

        title = info['seriess'][0]['title']

        # Get latest observation
        data = self._make_request(
            'series/observations',
            {
                'series_id': series_id,
                'sort_order': 'desc',
                'limit': 1
            }
        )

        if not data or 'observations' not in data or len(data['observations']) == 0:
            return None

        obs = data['observations'][0]

        # Handle missing values
        if obs['value'] == '.':
            logger.warning(f"No value available for {series_id}")
            return None

        return {
            'series_id': series_id,
            'title': title,
            'value': float(obs['value']),
            'date': datetime.strptime(obs['date'], '%Y-%m-%d')
        }

    def get_dxy(self) -> Optional[Dict[str, Any]]:
        """
        Get US Dollar Index (DXY)

        Strong dollar = bearish for crypto
        Weak dollar = bullish for crypto

        Returns:
            {
                'value': float,
                'date': datetime,
                'signal': str (bullish, bearish, neutral),
                'interpretation': str
            }
        """
        data = self.get_latest_value(self.SERIES['dxy'])
        if not data:
            return None

        value = data['value']

        # Interpret DXY (baseline ~100)
        if value > 110:
            signal = "bearish"
            interpretation = (
                f"VERY STRONG DOLLAR ({value:.2f}) - USD extremely strong. "
                "Bearish for crypto (capital flows to USD)."
            )
        elif value > 105:
            signal = "neutral_bearish"
            interpretation = (
                f"STRONG DOLLAR ({value:.2f}) - USD strength pressuring crypto. "
                "Moderate headwind for crypto prices."
            )
        elif value > 95:
            signal = "neutral"
            interpretation = (
                f"NEUTRAL DOLLAR ({value:.2f}) - USD at typical range. "
                "No strong macro headwind/tailwind for crypto."
            )
        elif value > 90:
            signal = "neutral_bullish"
            interpretation = (
                f"WEAK DOLLAR ({value:.2f}) - USD weakness supporting crypto. "
                "Moderate tailwind for crypto prices."
            )
        else:
            signal = "bullish"
            interpretation = (
                f"VERY WEAK DOLLAR ({value:.2f}) - USD collapsing. "
                "Bullish for crypto (capital fleeing USD)."
            )

        logger.info(f"DXY: {value:.2f} → Signal: {signal.upper()}")

        return {
            'value': value,
            'date': data['date'],
            'signal': signal,
            'interpretation': interpretation
        }

    def get_fed_rate(self) -> Optional[Dict[str, Any]]:
        """
        Get Federal Funds Rate

        High rates = bearish for risk assets (crypto)
        Low rates = bullish for risk assets

        Returns similar structure to get_dxy()
        """
        data = self.get_latest_value(self.SERIES['fed_rate'])
        if not data:
            return None

        value = data['value']

        # Interpret Fed Rate
        if value > 5.0:
            signal = "bearish"
            interpretation = (
                f"HIGH RATES ({value:.2f}%) - Tight monetary policy. "
                "Bearish for crypto (capital prefers safe yields)."
            )
        elif value > 2.5:
            signal = "neutral_bearish"
            interpretation = (
                f"ELEVATED RATES ({value:.2f}%) - Moderately restrictive policy. "
                "Headwind for risk assets like crypto."
            )
        elif value > 1.0:
            signal = "neutral"
            interpretation = (
                f"MODERATE RATES ({value:.2f}%) - Neutral monetary policy."
            )
        elif value > 0.25:
            signal = "neutral_bullish"
            interpretation = (
                f"LOW RATES ({value:.2f}%) - Accommodative policy. "
                "Supportive for risk assets."
            )
        else:
            signal = "bullish"
            interpretation = (
                f"ZERO RATES ({value:.2f}%) - Maximum monetary stimulus. "
                "Highly bullish for crypto."
            )

        logger.info(f"Fed Funds Rate: {value:.2f}% → Signal: {signal.upper()}")

        return {
            'value': value,
            'date': data['date'],
            'signal': signal,
            'interpretation': interpretation
        }

    def get_vix(self) -> Optional[Dict[str, Any]]:
        """
        Get VIX (Volatility Index)

        High VIX = market fear, risk-off
        Low VIX = market complacency, risk-on

        Crypto correlates with risk sentiment
        """
        data = self.get_latest_value(self.SERIES['vix'])
        if not data:
            return None

        value = data['value']

        # Interpret VIX
        if value > 40:
            signal = "bearish"
            interpretation = (
                f"EXTREME FEAR (VIX {value:.1f}) - Market panic. "
                "Crypto likely selling off with equities."
            )
        elif value > 25:
            signal = "neutral_bearish"
            interpretation = (
                f"HIGH VOLATILITY (VIX {value:.1f}) - Elevated fear. "
                "Risk-off environment pressures crypto."
            )
        elif value > 15:
            signal = "neutral"
            interpretation = (
                f"NORMAL VOLATILITY (VIX {value:.1f}) - Markets calm."
            )
        elif value > 10:
            signal = "neutral_bullish"
            interpretation = (
                f"LOW VOLATILITY (VIX {value:.1f}) - Market complacency. "
                "Risk-on environment supports crypto."
            )
        else:
            signal = "bullish"
            interpretation = (
                f"VERY LOW VOLATILITY (VIX {value:.1f}) - Extreme complacency. "
                "Maximum risk-on sentiment."
            )

        logger.info(f"VIX: {value:.1f} → Signal: {signal.upper()}")

        return {
            'value': value,
            'date': data['date'],
            'signal': signal,
            'interpretation': interpretation
        }

    def get_macro_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive macro analysis

        Combines DXY, Fed Rate, and VIX to assess macro environment for crypto

        Returns:
            Dictionary with all macro metrics and composite signal
        """
        dxy = self.get_dxy()
        fed_rate = self.get_fed_rate()
        vix = self.get_vix()

        if not dxy or not fed_rate or not vix:
            logger.warning("Could not fetch complete macro data")
            return None

        # Combine signals
        signals = [dxy['signal'], fed_rate['signal'], vix['signal']]
        bullish_count = sum(1 for s in signals if 'bullish' in s)
        bearish_count = sum(1 for s in signals if 'bearish' in s)

        if bullish_count >= 2:
            composite_signal = "bullish"
            composite_interpretation = "Macro environment is SUPPORTIVE for crypto"
        elif bearish_count >= 2:
            composite_signal = "bearish"
            composite_interpretation = "Macro environment is HOSTILE for crypto"
        else:
            composite_signal = "neutral"
            composite_interpretation = "Macro environment is MIXED for crypto"

        logger.info(f"Macro Composite Signal: {composite_signal.upper()}")

        return {
            'dxy': dxy,
            'fed_rate': fed_rate,
            'vix': vix,
            'composite_signal': composite_signal,
            'composite_interpretation': composite_interpretation,
            'timestamp': datetime.now()
        }


# Test the client if run directly
if __name__ == "__main__":
    # Note: Need FRED API key in environment
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("⚠️  No FRED_API_KEY found in environment")
        print("Get one at: https://fredaccount.stlouisfed.org/apikeys")
        print("\nThen set: export FRED_API_KEY=your_key_here")
        exit(1)

    client = FREDClient(api_key)

    print("=" * 80)
    print("FRED MACRO ANALYSIS")
    print("=" * 80)

    analysis = client.get_macro_analysis()
    if analysis:
        print(f"\n💵 US Dollar Index (DXY):")
        print(f"   Value: {analysis['dxy']['value']:.2f}")
        print(f"   Date: {analysis['dxy']['date'].strftime('%Y-%m-%d')}")
        print(f"   Signal: {analysis['dxy']['signal'].upper()}")
        print(f"   {analysis['dxy']['interpretation']}")

        print(f"\n📊 Federal Funds Rate:")
        print(f"   Value: {analysis['fed_rate']['value']:.2f}%")
        print(f"   Date: {analysis['fed_rate']['date'].strftime('%Y-%m-%d')}")
        print(f"   Signal: {analysis['fed_rate']['signal'].upper()}")
        print(f"   {analysis['fed_rate']['interpretation']}")

        print(f"\n📉 VIX (Volatility Index):")
        print(f"   Value: {analysis['vix']['value']:.1f}")
        print(f"   Date: {analysis['vix']['date'].strftime('%Y-%m-%d')}")
        print(f"   Signal: {analysis['vix']['signal'].upper()}")
        print(f"   {analysis['vix']['interpretation']}")

        print(f"\n🎯 MACRO COMPOSITE SIGNAL: {analysis['composite_signal'].upper()}")
        print(f"   {analysis['composite_interpretation']}")

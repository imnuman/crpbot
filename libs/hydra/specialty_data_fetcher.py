"""
HYDRA 3.0 - Specialty Data Fetcher

Fetches real-time data for all 4 engine specialties:
- Engine A: Liquidation cascades (Coinglass API)
- Engine B: Funding rate extremes (Binance Futures)
- Engine C: Orderbook imbalance (Coinbase REST API)
- Engine D: ATR spike / Regime transitions (calculated from OHLCV)

This module provides the data needed for specialty triggers in engine_specialization.py
"""

import os
import time
import requests
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from loguru import logger
import numpy as np


class SpecialtyDataFetcher:
    """
    Fetches specialty trigger data from multiple sources.

    Returns dict with keys matching engine_specialization.py expectations:
    - liquidation_total_usd: Total liquidations in USD (for Engine A)
    - funding_rate_pct: Current funding rate % (for Engine B)
    - bid_ask_ratio: Order book bid/ask ratio (for Engine C)
    - atr_multiplier: Current ATR vs baseline (for Engine D)
    """

    # Cache duration in seconds
    CACHE_DURATION = 60  # Refresh every 60 seconds

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}

        # Symbol mappings
        self.binance_symbols = {
            'BTC-USD': 'BTCUSDT',
            'ETH-USD': 'ETHUSDT',
            'SOL-USD': 'SOLUSDT',
            'XRP-USD': 'XRPUSDT',
            'LTC-USD': 'LTCUSDT',
            'ADA-USD': 'ADAUSDT',
            'LINK-USD': 'LINKUSDT',
            'DOT-USD': 'DOTUSDT',
        }

        # Coinglass uses different symbols
        self.coinglass_symbols = {
            'BTC-USD': 'BTC',
            'ETH-USD': 'ETH',
            'SOL-USD': 'SOL',
            'XRP-USD': 'XRP',
            'LTC-USD': 'LTC',
            'ADA-USD': 'ADA',
            'LINK-USD': 'LINK',
            'DOT-USD': 'DOT',
        }

        logger.info("[SpecialtyDataFetcher] Initialized")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        return (time.time() - self._cache_time[key]) < self.CACHE_DURATION

    def get_all_specialty_data(self, symbol: str, ohlcv_data: List[Dict]) -> Dict[str, Any]:
        """
        Fetch all specialty trigger data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            ohlcv_data: List of OHLCV candles for ATR calculation

        Returns:
            Dict with all specialty trigger values:
            {
                'liquidation_total_usd': float,
                'funding_rate_pct': float,
                'bid_ask_ratio': float,
                'atr_multiplier': float,
                'fetch_time': datetime
            }
        """
        cache_key = f"all_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Fetch all data
        liquidation = self.get_liquidation_data(symbol)
        funding = self.get_funding_rate(symbol)
        orderbook = self.get_orderbook_imbalance(symbol)
        atr_mult = self.calculate_atr_multiplier(ohlcv_data)

        result = {
            'liquidation_total_usd': liquidation.get('total_usd', 0),
            'funding_rate_pct': funding.get('rate_pct', 0),
            'bid_ask_ratio': orderbook.get('ratio', 1.0),
            'atr_multiplier': atr_mult.get('multiplier', 1.0),
            'fetch_time': datetime.now(timezone.utc),
            # Also include raw data for debugging
            '_liquidation_raw': liquidation,
            '_funding_raw': funding,
            '_orderbook_raw': orderbook,
            '_atr_raw': atr_mult,
        }

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()

        logger.debug(
            f"[SpecialtyData] {symbol}: liq=${result['liquidation_total_usd']:,.0f}, "
            f"funding={result['funding_rate_pct']:.4f}%, "
            f"ob_ratio={result['bid_ask_ratio']:.2f}, "
            f"atr_mult={result['atr_multiplier']:.2f}x"
        )

        return result

    # ==================== LIQUIDATION DATA (Engine A) ====================

    def get_liquidation_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch liquidation data from Coinglass API.

        Engine A specialty: $20M+ liquidations trigger trading.

        Returns:
            {
                'total_usd': float (total liquidations in last 15 min),
                'long_usd': float,
                'short_usd': float,
                'source': str
            }
        """
        cache_key = f"liq_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {
            'total_usd': 0,
            'long_usd': 0,
            'short_usd': 0,
            'source': 'coinglass'
        }

        try:
            cg_symbol = self.coinglass_symbols.get(symbol, symbol.split('-')[0])

            # Coinglass public API for liquidation data
            # Note: Free tier has rate limits, use cached data
            url = f"https://open-api.coinglass.com/public/v2/liquidation_history"
            params = {
                'symbol': cg_symbol,
                'time_type': 'h1'  # Last hour
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data'):
                    # Sum recent liquidations
                    liq_data = data['data']
                    if isinstance(liq_data, list) and len(liq_data) > 0:
                        # Take last entry (most recent)
                        latest = liq_data[-1]
                        result['long_usd'] = float(latest.get('longLiquidationUsd', 0))
                        result['short_usd'] = float(latest.get('shortLiquidationUsd', 0))
                        result['total_usd'] = result['long_usd'] + result['short_usd']
            else:
                logger.debug(f"Coinglass API returned {response.status_code}")
                # Fall back to Binance liquidation estimate
                result = self._estimate_liquidations_from_binance(symbol)

        except Exception as e:
            logger.warning(f"Failed to fetch liquidation data for {symbol}: {e}")
            # Fall back to Binance-based estimate
            result = self._estimate_liquidations_from_binance(symbol)

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()

        return result

    def _estimate_liquidations_from_binance(self, symbol: str) -> Dict[str, Any]:
        """
        Estimate liquidations from Binance open interest changes.

        When OI drops sharply with price move = liquidations.
        """
        try:
            binance_symbol = self.binance_symbols.get(symbol)
            if not binance_symbol:
                return {'total_usd': 0, 'long_usd': 0, 'short_usd': 0, 'source': 'estimate'}

            # Get open interest history
            url = "https://fapi.binance.com/fapi/v1/openInterestHist"
            params = {
                'symbol': binance_symbol,
                'period': '5m',
                'limit': 3  # Last 15 minutes
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 2:
                    # Calculate OI change
                    oi_now = float(data[-1].get('sumOpenInterestValue', 0))
                    oi_prev = float(data[0].get('sumOpenInterestValue', 0))
                    oi_change = oi_prev - oi_now  # Positive = OI dropped = liquidations

                    if oi_change > 0:
                        # Rough estimate: 50% of OI drop is liquidations
                        estimated_liq = oi_change * 0.5
                        return {
                            'total_usd': estimated_liq,
                            'long_usd': estimated_liq * 0.5,
                            'short_usd': estimated_liq * 0.5,
                            'source': 'binance_estimate'
                        }

        except Exception as e:
            logger.debug(f"Binance OI estimate failed: {e}")

        return {'total_usd': 0, 'long_usd': 0, 'short_usd': 0, 'source': 'failed'}

    # ==================== FUNDING RATE (Engine B) ====================

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch funding rate from Binance Futures.

        Engine B specialty: >0.5% funding rate triggers trading.

        Returns:
            {
                'rate_pct': float (funding rate as percentage),
                'next_funding_time': datetime,
                'source': str
            }
        """
        cache_key = f"funding_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {
            'rate_pct': 0,
            'next_funding_time': None,
            'source': 'binance'
        }

        try:
            binance_symbol = self.binance_symbols.get(symbol)
            if not binance_symbol:
                logger.debug(f"No Binance symbol mapping for {symbol}")
                return result

            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {'symbol': binance_symbol}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Funding rate is returned as decimal (0.0001 = 0.01%)
                funding_rate = float(data.get('lastFundingRate', 0)) * 100
                next_funding = datetime.fromtimestamp(
                    int(data.get('nextFundingTime', 0)) / 1000,
                    tz=timezone.utc
                )

                result['rate_pct'] = funding_rate
                result['next_funding_time'] = next_funding

                logger.debug(f"[Funding] {symbol}: {funding_rate:+.4f}%")
            else:
                logger.warning(f"Binance funding API returned {response.status_code}")

        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()

        return result

    # ==================== ORDERBOOK IMBALANCE (Engine C) ====================

    def get_orderbook_imbalance(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch orderbook and calculate bid/ask ratio.

        Engine C specialty: >2.5:1 ratio triggers trading.

        Uses Coinbase REST API (no websocket needed for this).

        Returns:
            {
                'ratio': float (bid_volume / ask_volume),
                'bid_depth': float,
                'ask_depth': float,
                'source': str
            }
        """
        cache_key = f"orderbook_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {
            'ratio': 1.0,
            'bid_depth': 0,
            'ask_depth': 0,
            'source': 'coinbase'
        }

        try:
            # Coinbase Exchange API (public, no auth needed)
            url = f"https://api.exchange.coinbase.com/products/{symbol}/book"
            params = {'level': 2}  # Level 2 = top 50 orders

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Calculate depth (sum of top 10 levels)
                bids = data.get('bids', [])[:10]
                asks = data.get('asks', [])[:10]

                # [price, size, num_orders]
                bid_depth = sum(float(b[1]) for b in bids)
                ask_depth = sum(float(a[1]) for a in asks)

                if ask_depth > 0:
                    ratio = bid_depth / ask_depth
                else:
                    ratio = 1.0

                result['bid_depth'] = bid_depth
                result['ask_depth'] = ask_depth
                result['ratio'] = ratio

                logger.debug(f"[Orderbook] {symbol}: {ratio:.2f}:1 (bids={bid_depth:.2f}, asks={ask_depth:.2f})")
            else:
                logger.warning(f"Coinbase orderbook API returned {response.status_code}")
                # Try Binance as fallback
                result = self._get_binance_orderbook(symbol)

        except Exception as e:
            logger.warning(f"Failed to fetch orderbook for {symbol}: {e}")
            result = self._get_binance_orderbook(symbol)

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()

        return result

    def _get_binance_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Fallback orderbook from Binance."""
        result = {'ratio': 1.0, 'bid_depth': 0, 'ask_depth': 0, 'source': 'binance'}

        try:
            binance_symbol = self.binance_symbols.get(symbol)
            if not binance_symbol:
                return result

            url = "https://api.binance.com/api/v3/depth"
            params = {'symbol': binance_symbol, 'limit': 10}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                bid_depth = sum(float(b[1]) for b in data.get('bids', []))
                ask_depth = sum(float(a[1]) for a in data.get('asks', []))

                if ask_depth > 0:
                    result['ratio'] = bid_depth / ask_depth
                result['bid_depth'] = bid_depth
                result['ask_depth'] = ask_depth

        except Exception as e:
            logger.debug(f"Binance orderbook fallback failed: {e}")

        return result

    # ==================== ATR MULTIPLIER (Engine D) ====================

    def calculate_atr_multiplier(self, ohlcv_data: List[Dict], period: int = 14) -> Dict[str, Any]:
        """
        Calculate ATR multiplier (current ATR vs baseline).

        Engine D specialty: 2x ATR expansion triggers trading.

        Args:
            ohlcv_data: List of OHLCV candles (most recent last)
            period: ATR period (default 14)

        Returns:
            {
                'multiplier': float (current_atr / baseline_atr),
                'current_atr': float,
                'baseline_atr': float,
                'source': str
            }
        """
        result = {
            'multiplier': 1.0,
            'current_atr': 0,
            'baseline_atr': 0,
            'source': 'calculated'
        }

        if not ohlcv_data or len(ohlcv_data) < period * 3:
            logger.debug(f"Insufficient OHLCV data for ATR calculation (need {period * 3}, got {len(ohlcv_data) if ohlcv_data else 0})")
            return result

        try:
            # Convert to numpy arrays
            highs = np.array([c.get('high', c.get('High', 0)) for c in ohlcv_data])
            lows = np.array([c.get('low', c.get('Low', 0)) for c in ohlcv_data])
            closes = np.array([c.get('close', c.get('Close', 0)) for c in ohlcv_data])

            # Calculate True Range
            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))

            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = tr1[0]  # First value has no previous close

            # Calculate ATR using EMA
            atr = np.zeros_like(tr)
            atr[period-1] = np.mean(tr[:period])  # Initial SMA

            multiplier = 2 / (period + 1)
            for i in range(period, len(tr)):
                atr[i] = (tr[i] - atr[i-1]) * multiplier + atr[i-1]

            # Current ATR (most recent)
            current_atr = atr[-1]

            # Baseline ATR (average of previous period)
            # Use ATR from 2 periods ago as baseline
            baseline_idx = max(period, len(atr) - period * 2)
            baseline_atr = np.mean(atr[baseline_idx:baseline_idx + period])

            if baseline_atr > 0:
                atr_multiplier = current_atr / baseline_atr
            else:
                atr_multiplier = 1.0

            result['multiplier'] = atr_multiplier
            result['current_atr'] = current_atr
            result['baseline_atr'] = baseline_atr

            logger.debug(f"[ATR] Current: {current_atr:.4f}, Baseline: {baseline_atr:.4f}, Multiplier: {atr_multiplier:.2f}x")

        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")

        return result


# ==================== SINGLETON ====================

_fetcher_instance: Optional[SpecialtyDataFetcher] = None


def get_specialty_data_fetcher() -> SpecialtyDataFetcher:
    """Get singleton instance of SpecialtyDataFetcher."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = SpecialtyDataFetcher()
    return _fetcher_instance


# ==================== TEST ====================

if __name__ == "__main__":
    import json

    fetcher = SpecialtyDataFetcher()

    # Test with mock OHLCV data
    mock_ohlcv = [
        {'high': 100, 'low': 95, 'close': 98} for _ in range(50)
    ]
    # Add some volatility
    for i in range(45, 50):
        mock_ohlcv[i] = {'high': 110, 'low': 90, 'close': 100}

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        print(f"\n{'='*60}")
        print(f"SPECIALTY DATA: {symbol}")
        print(f"{'='*60}")

        data = fetcher.get_all_specialty_data(symbol, mock_ohlcv)

        print(f"Liquidations: ${data['liquidation_total_usd']:,.0f}")
        print(f"Funding Rate: {data['funding_rate_pct']:+.4f}%")
        print(f"Orderbook Ratio: {data['bid_ask_ratio']:.2f}:1")
        print(f"ATR Multiplier: {data['atr_multiplier']:.2f}x")

        # Check trigger conditions
        print(f"\n--- TRIGGER STATUS ---")
        print(f"Engine A (Liquidation $20M+): {'TRIGGERED' if data['liquidation_total_usd'] >= 20_000_000 else 'not triggered'}")
        print(f"Engine B (Funding >0.5%): {'TRIGGERED' if abs(data['funding_rate_pct']) >= 0.5 else 'not triggered'}")
        print(f"Engine C (Orderbook >2.5:1): {'TRIGGERED' if data['bid_ask_ratio'] >= 2.5 or data['bid_ask_ratio'] <= 0.4 else 'not triggered'}")
        print(f"Engine D (ATR >2x): {'TRIGGERED' if data['atr_multiplier'] >= 2.0 else 'not triggered'}")

"""
HYDRA Trigger Validation Script

Validates statistical edge of each specialty trigger:
- Engine A: Liquidation Cascades ($20M+ trigger)
- Engine B: Funding Rate Extremes (>0.5% trigger)
- Engine C: Orderbook Imbalance (>2.5:1 ratio)
- Engine D: ATR Expansion (>2x baseline)

For each trigger, we measure:
1. How often it fires (frequency)
2. Win rate (price moved in predicted direction within 24h)
3. Average return
4. Sharpe ratio
5. Statistical significance (t-test)

Run: python scripts/trigger_validation.py
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TriggerEvent:
    """A single trigger event."""
    timestamp: datetime
    trigger_type: str
    symbol: str
    trigger_value: float
    direction: str  # BUY or SELL (predicted)
    entry_price: float
    exit_price_24h: Optional[float] = None
    return_pct: Optional[float] = None
    outcome: Optional[str] = None  # win/loss


@dataclass
class TriggerStats:
    """Statistics for a trigger type."""
    trigger_type: str
    total_events: int
    wins: int
    losses: int
    win_rate: float
    avg_return_pct: float
    std_return_pct: float
    sharpe_ratio: float
    max_return: float
    min_return: float
    avg_trigger_value: float
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05


class TriggerValidator:
    """
    Validates trading triggers using historical data.
    """

    def __init__(self, lookback_days: int = 180):
        """
        Initialize validator.

        Args:
            lookback_days: How many days of history to analyze (default 6 months)
        """
        self.lookback_days = lookback_days
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

        # Current trigger thresholds
        self.thresholds = {
            'liquidation': 20_000_000,  # $20M
            'funding': 0.5,  # 0.5%
            'orderbook': 2.5,  # 2.5:1 ratio
            'atr': 2.0,  # 2x baseline
        }

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.funding_data: Dict[str, pd.DataFrame] = {}
        self.trigger_events: List[TriggerEvent] = []

        print(f"[TriggerValidator] Initialized with {lookback_days} days lookback")

    def fetch_all_data(self):
        """Fetch all required historical data."""
        print("\n" + "="*60)
        print("FETCHING HISTORICAL DATA")
        print("="*60)

        for symbol in self.symbols:
            print(f"\n--- {symbol} ---")

            # Fetch price data (hourly candles)
            self.price_data[symbol] = self._fetch_price_data(symbol)
            if self.price_data[symbol] is not None:
                print(f"  Price data: {len(self.price_data[symbol])} candles")

            # Fetch historical funding rates
            self.funding_data[symbol] = self._fetch_funding_history(symbol)
            if self.funding_data[symbol] is not None:
                print(f"  Funding data: {len(self.funding_data[symbol])} records")

    def _fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data from Coinbase."""
        try:
            # Coinbase granularity: 3600 = 1 hour
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=self.lookback_days)

            all_candles = []
            current_end = end_time

            # Coinbase limits to 300 candles per request
            while current_end > start_time:
                current_start = max(start_time, current_end - timedelta(hours=300))

                url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
                params = {
                    'granularity': 3600,
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat()
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if data:
                        all_candles.extend(data)
                else:
                    print(f"  Warning: Coinbase returned {response.status_code}")
                    break

                current_end = current_start
                time.sleep(0.2)  # Rate limiting

            if not all_candles:
                return None

            # Convert to DataFrame
            # Coinbase format: [timestamp, low, high, open, close, volume]
            df = pd.DataFrame(all_candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"  Error fetching price data: {e}")
            return None

    def _fetch_funding_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical funding rates from Bybit."""
        try:
            # Bybit symbol mapping
            bybit_symbols = {
                'BTC-USD': 'BTCUSDT',
                'ETH-USD': 'ETHUSDT',
                'SOL-USD': 'SOLUSDT',
            }

            bybit_symbol = bybit_symbols.get(symbol)
            if not bybit_symbol:
                return None

            all_funding = []
            end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_time = int((datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).timestamp() * 1000)

            # Bybit returns max 200 records per request
            current_end = end_time

            while current_end > start_time:
                url = "https://api.bybit.com/v5/market/funding/history"
                params = {
                    'category': 'linear',
                    'symbol': bybit_symbol,
                    'endTime': current_end,
                    'limit': 200
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        records = data.get('result', {}).get('list', [])
                        if not records:
                            break
                        all_funding.extend(records)
                        # Next page - use oldest timestamp
                        oldest_ts = min(int(r['fundingRateTimestamp']) for r in records)
                        current_end = oldest_ts - 1
                    else:
                        break
                else:
                    print(f"  Warning: Bybit returned {response.status_code}")
                    break

                time.sleep(0.2)

            if not all_funding:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_funding)
            df['timestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(int), unit='ms', utc=True)
            df['funding_rate_pct'] = df['fundingRate'].astype(float) * 100
            df = df[['timestamp', 'funding_rate_pct']].sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"  Error fetching funding data: {e}")
            return None

    def identify_trigger_events(self):
        """Identify all trigger events in historical data."""
        print("\n" + "="*60)
        print("IDENTIFYING TRIGGER EVENTS")
        print("="*60)

        self.trigger_events = []

        for symbol in self.symbols:
            price_df = self.price_data.get(symbol)
            funding_df = self.funding_data.get(symbol)

            if price_df is None or len(price_df) < 50:
                print(f"\n{symbol}: Insufficient price data")
                continue

            print(f"\n--- {symbol} ---")

            # Engine B: Funding Rate Extremes
            funding_events = self._find_funding_triggers(symbol, price_df, funding_df)
            print(f"  Funding triggers: {len(funding_events)}")
            self.trigger_events.extend(funding_events)

            # Engine D: ATR Expansion
            atr_events = self._find_atr_triggers(symbol, price_df)
            print(f"  ATR triggers: {len(atr_events)}")
            self.trigger_events.extend(atr_events)

            # Engine C: Orderbook Imbalance (proxy using price momentum)
            # Note: No historical orderbook data, using momentum as proxy
            momentum_events = self._find_momentum_triggers(symbol, price_df)
            print(f"  Momentum triggers (orderbook proxy): {len(momentum_events)}")
            self.trigger_events.extend(momentum_events)

        print(f"\nTotal trigger events: {len(self.trigger_events)}")

    def _find_funding_triggers(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame]
    ) -> List[TriggerEvent]:
        """Find funding rate extreme triggers."""
        events = []

        if funding_df is None or len(funding_df) == 0:
            return events

        threshold = self.thresholds['funding']

        for idx, row in funding_df.iterrows():
            funding_rate = row['funding_rate_pct']
            ts = row['timestamp']

            # Check if funding rate exceeds threshold
            if abs(funding_rate) >= threshold:
                # Funding is contrarian - high positive = expect price drop, high negative = expect rise
                direction = 'SELL' if funding_rate > 0 else 'BUY'

                # Find entry price at trigger time
                price_row = price_df[price_df['timestamp'] <= ts].iloc[-1] if len(price_df[price_df['timestamp'] <= ts]) > 0 else None
                if price_row is None:
                    continue

                entry_price = price_row['close']
                entry_ts = price_row['timestamp']

                # Find exit price 24h later
                exit_ts = entry_ts + timedelta(hours=24)
                exit_rows = price_df[price_df['timestamp'] >= exit_ts]

                if len(exit_rows) > 0:
                    exit_price = exit_rows.iloc[0]['close']

                    # Calculate return
                    if direction == 'BUY':
                        return_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        return_pct = (entry_price - exit_price) / entry_price * 100

                    outcome = 'win' if return_pct > 0 else 'loss'

                    events.append(TriggerEvent(
                        timestamp=ts,
                        trigger_type='funding',
                        symbol=symbol,
                        trigger_value=funding_rate,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price_24h=exit_price,
                        return_pct=return_pct,
                        outcome=outcome
                    ))

        return events

    def _find_atr_triggers(self, symbol: str, price_df: pd.DataFrame) -> List[TriggerEvent]:
        """Find ATR expansion triggers."""
        events = []

        if len(price_df) < 50:
            return events

        threshold = self.thresholds['atr']
        period = 14

        # Calculate ATR
        df = price_df.copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['atr_baseline'] = df['atr'].rolling(window=period * 2).mean()
        df['atr_multiplier'] = df['atr'] / df['atr_baseline']

        # Price direction for ATR expansion
        df['price_momentum'] = df['close'].pct_change(periods=period)

        for idx in range(period * 3, len(df) - 24):  # Need 24h lookahead
            row = df.iloc[idx]

            if pd.isna(row['atr_multiplier']) or pd.isna(row['price_momentum']):
                continue

            if row['atr_multiplier'] >= threshold:
                # ATR expansion - trade in direction of momentum
                direction = 'BUY' if row['price_momentum'] > 0 else 'SELL'
                entry_price = row['close']
                ts = row['timestamp']

                # Exit 24h later
                exit_row = df.iloc[idx + 24]
                exit_price = exit_row['close']

                if direction == 'BUY':
                    return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    return_pct = (entry_price - exit_price) / entry_price * 100

                outcome = 'win' if return_pct > 0 else 'loss'

                events.append(TriggerEvent(
                    timestamp=ts,
                    trigger_type='atr',
                    symbol=symbol,
                    trigger_value=row['atr_multiplier'],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price_24h=exit_price,
                    return_pct=return_pct,
                    outcome=outcome
                ))

        return events

    def _find_momentum_triggers(self, symbol: str, price_df: pd.DataFrame) -> List[TriggerEvent]:
        """
        Find momentum-based triggers as proxy for orderbook imbalance.

        Strong momentum often correlates with orderbook imbalance.
        Uses RSI extremes as proxy (<30 or >70).
        """
        events = []

        if len(price_df) < 50:
            return events

        # Calculate RSI as momentum proxy
        df = price_df.copy()
        period = 14

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume spike as additional filter
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        for idx in range(30, len(df) - 24):
            row = df.iloc[idx]

            if pd.isna(row['rsi']) or pd.isna(row['volume_ratio']):
                continue

            # Strong oversold/overbought with volume confirmation
            if row['rsi'] < 30 and row['volume_ratio'] > 1.5:
                direction = 'BUY'  # Oversold = expect bounce
            elif row['rsi'] > 70 and row['volume_ratio'] > 1.5:
                direction = 'SELL'  # Overbought = expect pullback
            else:
                continue

            entry_price = row['close']
            ts = row['timestamp']

            exit_row = df.iloc[idx + 24]
            exit_price = exit_row['close']

            if direction == 'BUY':
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - exit_price) / entry_price * 100

            outcome = 'win' if return_pct > 0 else 'loss'

            # Use RSI distance from 50 as "trigger value"
            trigger_value = abs(row['rsi'] - 50) / 50  # Normalized 0-1

            events.append(TriggerEvent(
                timestamp=ts,
                trigger_type='momentum',
                symbol=symbol,
                trigger_value=row['rsi'],
                direction=direction,
                entry_price=entry_price,
                exit_price_24h=exit_price,
                return_pct=return_pct,
                outcome=outcome
            ))

        return events

    def calculate_statistics(self) -> Dict[str, TriggerStats]:
        """Calculate statistics for each trigger type."""
        print("\n" + "="*60)
        print("CALCULATING STATISTICS")
        print("="*60)

        results = {}

        # Group events by trigger type
        by_type: Dict[str, List[TriggerEvent]] = {}
        for event in self.trigger_events:
            if event.trigger_type not in by_type:
                by_type[event.trigger_type] = []
            by_type[event.trigger_type].append(event)

        for trigger_type, events in by_type.items():
            if not events:
                continue

            returns = [e.return_pct for e in events if e.return_pct is not None]
            wins = [e for e in events if e.outcome == 'win']
            losses = [e for e in events if e.outcome == 'loss']
            trigger_values = [e.trigger_value for e in events]

            if len(returns) < 5:
                print(f"\n{trigger_type}: Insufficient data ({len(returns)} events)")
                continue

            win_rate = len(wins) / len(events) if events else 0
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return * np.sqrt(365)) if std_return > 0 else 0

            # T-test: Is mean return significantly different from 0?
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            is_significant = p_value < 0.05 and avg_return > 0

            stats_obj = TriggerStats(
                trigger_type=trigger_type,
                total_events=len(events),
                wins=len(wins),
                losses=len(losses),
                win_rate=win_rate,
                avg_return_pct=avg_return,
                std_return_pct=std_return,
                sharpe_ratio=sharpe,
                max_return=max(returns),
                min_return=min(returns),
                avg_trigger_value=np.mean(trigger_values),
                t_statistic=t_stat,
                p_value=p_value,
                is_significant=is_significant
            )

            results[trigger_type] = stats_obj

        return results

    def print_results(self, stats: Dict[str, TriggerStats]):
        """Print formatted results."""
        print("\n" + "="*70)
        print("TRIGGER VALIDATION RESULTS")
        print("="*70)

        for trigger_type, s in stats.items():
            edge_status = "HAS EDGE" if s.is_significant and s.win_rate > 0.52 else "NO EDGE"

            print(f"\n{'─'*70}")
            print(f"TRIGGER: {trigger_type.upper()}")
            print(f"{'─'*70}")
            print(f"  Events:     {s.total_events:>6}  ({s.wins} wins, {s.losses} losses)")
            print(f"  Win Rate:   {s.win_rate*100:>6.1f}%  {'✅' if s.win_rate > 0.55 else '⚠️' if s.win_rate > 0.50 else '❌'}")
            print(f"  Avg Return: {s.avg_return_pct:>+6.2f}%  ({'✅' if s.avg_return_pct > 0 else '❌'})")
            print(f"  Std Dev:    {s.std_return_pct:>6.2f}%")
            print(f"  Sharpe:     {s.sharpe_ratio:>6.2f}  ({'✅' if s.sharpe_ratio > 1.0 else '⚠️' if s.sharpe_ratio > 0.5 else '❌'})")
            print(f"  Max/Min:    +{s.max_return:.1f}% / {s.min_return:.1f}%")
            print(f"  T-stat:     {s.t_statistic:>6.2f}")
            print(f"  P-value:    {s.p_value:>6.4f}  ({'✅ Significant' if s.p_value < 0.05 else '❌ Not significant'})")
            print(f"\n  >>> {edge_status} <<<")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*70)

        has_edge = [k for k, v in stats.items() if v.is_significant and v.win_rate > 0.52]
        no_edge = [k for k, v in stats.items() if not (v.is_significant and v.win_rate > 0.52)]

        if has_edge:
            print(f"\n✅ TRIGGERS WITH STATISTICAL EDGE:")
            for t in has_edge:
                s = stats[t]
                print(f"   - {t}: {s.win_rate*100:.1f}% WR, {s.avg_return_pct:+.2f}% avg, Sharpe {s.sharpe_ratio:.2f}")
            print(f"\n   RECOMMENDATION: Focus trading on these triggers only.")
        else:
            print(f"\n❌ NO TRIGGERS SHOW STATISTICAL EDGE")
            print(f"   RECOMMENDATION: Do NOT trade until edge is found.")

        if no_edge:
            print(f"\n⚠️ TRIGGERS WITHOUT EDGE (disable these):")
            for t in no_edge:
                s = stats[t]
                print(f"   - {t}: {s.win_rate*100:.1f}% WR, {s.avg_return_pct:+.2f}% avg, p={s.p_value:.4f}")

        # Save results to JSON
        self._save_results(stats)

    def _save_results(self, stats: Dict[str, TriggerStats]):
        """Save results to JSON file."""
        results = {
            'validation_date': datetime.now(timezone.utc).isoformat(),
            'lookback_days': self.lookback_days,
            'symbols': self.symbols,
            'thresholds': self.thresholds,
            'results': {}
        }

        for trigger_type, s in stats.items():
            results['results'][trigger_type] = {
                'total_events': s.total_events,
                'wins': s.wins,
                'losses': s.losses,
                'win_rate': s.win_rate,
                'avg_return_pct': s.avg_return_pct,
                'std_return_pct': s.std_return_pct,
                'sharpe_ratio': s.sharpe_ratio,
                't_statistic': s.t_statistic,
                'p_value': s.p_value,
                'is_significant': s.is_significant,
                'has_edge': s.is_significant and s.win_rate > 0.52
            }

        output_path = '/root/crpbot/data/trigger_validation_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_path}")

    def run(self):
        """Run full validation pipeline."""
        print("\n" + "="*70)
        print("HYDRA TRIGGER VALIDATION")
        print(f"Analyzing {self.lookback_days} days of historical data")
        print("="*70)

        # Step 1: Fetch data
        self.fetch_all_data()

        # Step 2: Identify triggers
        self.identify_trigger_events()

        # Step 3: Calculate statistics
        stats = self.calculate_statistics()

        # Step 4: Print results
        self.print_results(stats)

        return stats


if __name__ == "__main__":
    validator = TriggerValidator(lookback_days=180)
    validator.run()

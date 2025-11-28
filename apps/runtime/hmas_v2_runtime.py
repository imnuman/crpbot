"""
HMAS V2 Runtime - Production Signal Generator
Uses complete 7-agent institutional system ($1.00/signal)

This runtime generates high-quality trading signals using:
- Alpha Generator V2 (DeepSeek $0.30)
- Technical Agent (DeepSeek $0.10)
- Sentiment Agent (DeepSeek $0.08)
- Macro Agent (DeepSeek $0.07)
- Execution Auditor V2 (Grok $0.15)
- Rationale Agent V2 (Claude $0.20)
- Mother AI V2 (Gemini $0.10)

Target: 80%+ win rate, FTMO compliant
"""
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from libs.hmas.hmas_orchestrator_v2 import HMASV2Orchestrator
from libs.data.coinbase import CoinbaseDataProvider


class HMASV2Runtime:
    """
    HMAS V2 Production Runtime

    Features:
    - Generate institutional-grade signals ($1.00 each)
    - Store signals in database
    - Rate limiting (configurable signals/day)
    - Cost tracking
    - Telegram notifications (optional)
    """

    def __init__(
        self,
        symbols: list[str],
        max_signals_per_day: int = 5,
        dry_run: bool = False
    ):
        """
        Initialize HMAS V2 Runtime

        Args:
            symbols: Trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
            max_signals_per_day: Maximum signals to generate per day
            dry_run: If True, don't store signals in DB
        """
        self.symbols = symbols
        self.max_signals_per_day = max_signals_per_day
        self.dry_run = dry_run

        # Initialize orchestrator
        print("Initializing HMAS V2 Orchestrator...")
        self.orchestrator = HMASV2Orchestrator(
            deepseek_api_key=os.getenv('DEEPSEEK_API_KEY'),
            xai_api_key=os.getenv('XAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )

        # Initialize data client
        print("Initializing Coinbase client...")
        self.data_client = CoinbaseDataProvider(
            api_key_name=os.getenv('COINBASE_API_KEY_NAME'),
            private_key=os.getenv('COINBASE_API_PRIVATE_KEY')
        )

        # Signal store - database session for storing signals
        from libs.db.database import get_database
        db = get_database()
        self.signal_store = db.get_session_direct()  # SQLAlchemy session

        # Tracking
        self.signals_generated_today = 0
        self.total_cost_today = 0.0
        self.session_start = datetime.now(timezone.utc)

    async def gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather all market data required for HMAS V2 analysis

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Complete market data dict for all 7 agents
        """
        print(f"\nGathering market data for {symbol}...")

        # Get OHLCV data (200+ candles)
        # Note: fetch_klines returns pandas DataFrame with OHLCV
        # Coinbase limit: 350 candles max. For 1m interval, 5 hours = 300 candles
        df = self.data_client.fetch_klines(
            symbol=symbol,
            interval='1m',
            start_time=datetime.now(timezone.utc) - timedelta(hours=5),  # Last 5 hours = ~300 candles
            end_time=datetime.now(timezone.utc),
            limit=300  # Max 300 candles per request
        )

        # Convert DataFrame to candles format
        candles = []
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                candles.append({
                    'timestamp': row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })

        if not candles:
            raise ValueError(f"No market data available for {symbol}")

        # Extract current price
        current_price = float(candles[0]['close'])

        # Calculate swing points (simplified - would be more sophisticated in production)
        closes = [float(c['close']) for c in candles]
        highs = [float(c['high']) for c in candles]
        lows = [float(c['low']) for c in candles]

        # Simple swing high/low detection (last 50 candles)
        swing_highs = sorted(highs[-50:], reverse=True)[:10]
        swing_lows = sorted(lows[-50:])[:10]

        # Calculate basic indicators
        indicators = self._calculate_basic_indicators(closes, highs, lows, current_price)

        # Build comprehensive market data
        market_data = {
            'symbol': symbol,
            'current_price': current_price,
            'ohlcv': candles,
            'price_history': closes,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,

            # Basic indicators (calculated from candles)
            'indicators': {
                'ma200': {'M1': indicators['ma200']},  # Use 1m data for now
                'rsi': {'M1': indicators['rsi']},
                'bbands': {'M1': {'upper': indicators['bb_upper'], 'lower': indicators['bb_lower']}},
                'atr': {'M1': indicators['atr']}
            },

            # Order book (would fetch from exchange in production)
            'order_book': {
                'entry_bid_volume': 15000000,
                'entry_ask_volume': 18000000,
                'spread_pips': 1.2,
                'sl_volume': 12000000,
                'sl_depth_score': 0.85,
                'tp_volume': 20000000,
                'tp_depth_score': 0.95
            },

            # Broker spreads (would fetch from multiple brokers in production)
            'broker_spreads': {
                'brokers': [
                    {'name': 'IC Markets', 'spread_pips': 1.2, 'execution_quality': 0.95, 'avg_slippage': 0.2},
                    {'name': 'Pepperstone', 'spread_pips': 1.5, 'execution_quality': 0.92, 'avg_slippage': 0.3},
                ]
            },

            # Market depth
            'market_depth': {
                'total_depth': 50000000,
                'buy_pressure': 0.52,
                'sell_pressure': 0.48,
                'imbalance': 'balanced'
            },

            # Volatility (simplified - would calculate ATR properly)
            'volatility': {
                'atr': (max(highs[-20:]) - min(lows[-20:])) / 20,  # Simple ATR approx
                'atr_percentile': 50,
                'state': 'normal'
            },

            # News headlines (would fetch from news API in production)
            'news_headlines': [],

            # Economic calendar (would fetch from economic calendar API)
            'economic_calendar': [],

            # Correlations (would calculate from real data)
            'correlations': {
                'dxy': 0.75,
                'dxy_trend': 'uptrend',
                'dxy_level': 104.5,
                'gold': -0.40,
                'gold_trend': 'downtrend',
                'oil': 0.30,
                'oil_trend': 'sideways',
                'yields': 0.55,
                'yields_trend': 'rising'
            },

            # COT data (would fetch from CFTC)
            'cot_data': {
                'commercial_long': 80000,
                'commercial_short': 95000,
                'commercial_net': -15000,
                'large_spec_long': 45000,
                'large_spec_short': 37000,
                'large_spec_net': 8000,
                'small_spec_long': 12000,
                'small_spec_short': 34000,
                'small_spec_net': -22000
            },

            # Social mentions (would fetch from Twitter/Reddit APIs)
            'social_mentions': {
                'twitter_total': 0,
                'twitter_bullish': 0,
                'twitter_bearish': 0,
                'twitter_neutral': 0,
                'twitter_influencers': [],
                'reddit_posts': 0,
                'reddit_bullish_upvotes': 0,
                'reddit_bearish_upvotes': 0,
                'reddit_subs': []
            },

            # Central bank policy (would fetch from economic data providers)
            'central_bank_policy': {
                'rate_differential': 25
            },

            # Market regime
            'market_regime': {
                'type': 'neutral',
                'vix': 15.0,
                'equities': 'mixed',
                'bonds': 'stable',
                'safe_haven_flows': 'inactive',
                'month': datetime.now(timezone.utc).strftime('%B'),
                'geopolitical_risk': 'low'
            },

            # Session
            'session': self._get_current_session(),
            'time_of_day': datetime.now(timezone.utc).strftime('%H:%M GMT'),

            # Historical execution
            'historical_slippage': 0.3,
            'fill_success_rate': 0.98,
            'avg_execution_ms': 150,

            # Position sizing
            'position_size_usd': 100  # 1% of $10,000 account
        }

        print(f"✓ Market data gathered: {len(candles)} candles, price: {current_price}")
        return market_data

    def _get_current_session(self) -> str:
        """Determine current trading session based on UTC time"""
        hour = datetime.now(timezone.utc).hour

        if 0 <= hour < 7:
            return 'sydney_tokyo'
        elif 7 <= hour < 15:
            return 'london_open'
        elif 15 <= hour < 21:
            return 'new_york_open'
        else:
            return 'after_hours'

    def _calculate_basic_indicators(self, closes: list, highs: list, lows: list, current_price: float) -> dict:
        """Calculate basic indicators from candle data"""
        if len(closes) < 200:
            # Not enough data for full analysis
            return {
                'ma200': current_price,  # Use current price as fallback
                'rsi': 50.0,  # Neutral RSI
                'bb_upper': current_price * 1.02,
                'bb_lower': current_price * 0.98,
                'atr': abs(max(highs[-20:]) - min(lows[-20:])) / 20 if len(highs) >= 20 else current_price * 0.01
            }

        # Calculate 200-period Moving Average
        ma200 = sum(closes[-200:]) / 200

        # Calculate RSI (14-period)
        period = min(14, len(closes) - 1)
        gains = []
        losses = []
        for i in range(len(closes) - period, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands (20-period, 2 std dev)
        bb_period = min(20, len(closes))
        bb_data = closes[-bb_period:]
        bb_ma = sum(bb_data) / len(bb_data)
        variance = sum((x - bb_ma) ** 2 for x in bb_data) / len(bb_data)
        std_dev = variance ** 0.5
        bb_upper = bb_ma + (2 * std_dev)
        bb_lower = bb_ma - (2 * std_dev)

        # Calculate ATR (14-period)
        atr_period = min(14, len(highs) - 1)
        true_ranges = []
        for i in range(len(highs) - atr_period, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else current_price * 0.01

        return {
            'ma200': ma200,
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'atr': atr
        }

    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate single signal for symbol using HMAS V2

        Args:
            symbol: Trading pair

        Returns:
            Complete signal dict or None if generation fails
        """
        try:
            # Gather market data
            market_data = await self.gather_market_data(symbol)

            # Generate signal with HMAS V2
            signal = await self.orchestrator.generate_signal(
                symbol=symbol,
                market_data=market_data,
                account_balance=10000.0  # FTMO account
            )

            # Track cost
            self.total_cost_today += 1.00
            self.signals_generated_today += 1

            # Store signal (if not dry run)
            if not self.dry_run and self.signal_store:
                await self._store_signal(signal)

            return signal

        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _store_signal(self, signal: Dict[str, Any]):
        """Store signal in database"""
        try:
            from libs.db.models import Signal
            from datetime import datetime, timezone

            # Extract trade parameters
            trade_params = signal.get('trade_parameters', {})

            # Map HMAS V2 decision to V7-compatible direction
            direction_map = {
                'APPROVED': signal.get('action', 'hold').lower(),  # 'long' or 'short'
                'REJECTED': 'hold'
            }
            direction = direction_map.get(signal['decision'], 'hold')

            # Create Signal model instance
            # Note: HMAS V2 uses 'stop_loss'/'take_profit' in trade_params, but DB uses 'sl_price'/'tp_price'
            new_signal = Signal(
                symbol=signal['symbol'],
                timestamp=datetime.fromisoformat(signal['timestamp']),
                direction=direction,
                confidence=signal.get('agent_analyses', {}).get('mother_ai', {}).get('decision_rationale', {}).get('confidence', 0) / 100.0,  # Convert to 0-1
                entry_price=trade_params.get('entry', 0),
                sl_price=trade_params.get('stop_loss', 0),  # Map HMAS stop_loss → DB sl_price
                tp_price=trade_params.get('take_profit', 0),  # Map HMAS take_profit → DB tp_price
                tier='high',  # HMAS V2 signals are high tier
                ensemble_prediction=0.5,  # Required field
                strategy='hmas_v2'  # HMAS V2 strategy identifier (7-agent system)
            )

            # Save to database
            self.signal_store.add(new_signal)
            self.signal_store.commit()

            print(f"✓ Signal stored in database (ID: {new_signal.id})")

        except Exception as e:
            print(f"⚠️  Error storing signal: {e}")
            import traceback
            traceback.print_exc()
            if self.signal_store:
                self.signal_store.rollback()

    async def run_single_iteration(self):
        """Run single iteration - generate signals for all symbols"""
        print("\n" + "="*80)
        print(f"HMAS V2 Runtime - New Iteration")
        print(f"Time: {datetime.now(timezone.utc).isoformat()}")
        print(f"Signals today: {self.signals_generated_today}/{self.max_signals_per_day}")
        print(f"Cost today: ${self.total_cost_today:.2f}")
        print("="*80 + "\n")

        # Check daily limit
        if self.signals_generated_today >= self.max_signals_per_day:
            print(f"Daily limit reached ({self.max_signals_per_day} signals). Skipping.")
            return

        # Generate signal for each symbol (respecting daily limit)
        for symbol in self.symbols:
            if self.signals_generated_today >= self.max_signals_per_day:
                break

            signal = await self.generate_signal(symbol)

            if signal:
                print(f"\n✓ Signal generated for {symbol}")
                print(f"  Decision: {signal['decision']}")
                print(f"  Action: {signal['action']}")
                if signal['decision'] == 'APPROVED':
                    params = signal.get('trade_parameters', {})
                    print(f"  Entry: {params.get('entry', 0)}")
                    print(f"  Lot Size: {params.get('lot_size', 0):.2f}")
                    print(f"  R:R: {params.get('reward_risk_ratio', 0):.2f}:1")

    async def run(self, iterations: int = -1, sleep_seconds: int = 3600):
        """
        Run HMAS V2 runtime continuously

        Args:
            iterations: Number of iterations (-1 = infinite)
            sleep_seconds: Seconds between iterations (default 1 hour)
        """
        print("\n" + "="*80)
        print("HMAS V2 Runtime Starting")
        print("="*80)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Max signals/day: {self.max_signals_per_day}")
        print(f"Iterations: {'infinite' if iterations == -1 else iterations}")
        print(f"Sleep between: {sleep_seconds}s")
        print(f"Dry run: {self.dry_run}")
        print("="*80 + "\n")

        iteration = 0
        while iterations == -1 or iteration < iterations:
            iteration += 1

            try:
                await self.run_single_iteration()

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()

            # Sleep (unless last iteration)
            if iterations == -1 or iteration < iterations:
                print(f"\nSleeping {sleep_seconds}s until next iteration...")
                await asyncio.sleep(sleep_seconds)

        print("\n" + "="*80)
        print("HMAS V2 Runtime Complete")
        print(f"Total signals generated: {self.signals_generated_today}")
        print(f"Total cost: ${self.total_cost_today:.2f}")
        print("="*80 + "\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HMAS V2 Runtime - Institutional Signal Generator')
    parser.add_argument('--symbols', nargs='+', default=['BTC-USD', 'ETH-USD', 'SOL-USD'],
                       help='Trading pairs to analyze')
    parser.add_argument('--max-signals-per-day', type=int, default=5,
                       help='Maximum signals per day (default: 5)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations (-1 for infinite)')
    parser.add_argument('--sleep-seconds', type=int, default=3600,
                       help='Seconds between iterations (default: 3600 = 1 hour)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (don\'t store signals)')

    args = parser.parse_args()

    # Create runtime
    runtime = HMASV2Runtime(
        symbols=args.symbols,
        max_signals_per_day=args.max_signals_per_day,
        dry_run=args.dry_run
    )

    # Run
    await runtime.run(
        iterations=args.iterations,
        sleep_seconds=args.sleep_seconds
    )


if __name__ == '__main__':
    asyncio.run(main())

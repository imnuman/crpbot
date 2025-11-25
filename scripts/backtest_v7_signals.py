"""
Backtest V7 Signals from Database

Tests the backtesting engine with actual V7 signals stored in the database.
"""
import sys
sys.path.insert(0, '/root/crpbot')

import pandas as pd
from datetime import datetime, timedelta
from libs.config.config import Settings
from libs.db.models import Signal, get_session
from libs.backtesting.vectorized_backtest import VectorizedBacktest, BacktestConfig
from libs.backtesting.walk_forward import WalkForwardOptimizer, WalkForwardConfig


def load_signals_from_database(days_back: int = 30):
    """Load signals from database"""
    config = Settings()
    session = get_session(config.db_url)

    cutoff_date = datetime.now() - timedelta(days=days_back)

    # Query signals
    signals = session.query(Signal).filter(
        Signal.timestamp >= cutoff_date,
        Signal.entry_price.isnot(None)
    ).order_by(Signal.timestamp).all()

    session.close()

    # Convert to DataFrame
    data = []
    for signal in signals:
        # Convert signal type to numeric
        direction = getattr(signal, 'direction', 'hold')
        if direction == 'long':
            signal_value = 1
        elif direction == 'short':
            signal_value = -1
        else:
            signal_value = 0

        # Simple approach: use available fields
        entry_price = signal.entry_price
        data.append({
            'timestamp': signal.timestamp,
            'signal': signal_value,
            'entry_price': entry_price,
            'stop_loss': entry_price * 0.98 if entry_price else None,  # Default 2% SL
            'take_profit': entry_price * 1.04 if entry_price else None,  # Default 4% TP
            'symbol': getattr(signal, 'symbol', 'UNKNOWN'),
            'confidence': getattr(signal, 'confidence', 0.5)
        })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} signals from database")
    return df


def main():
    print("=" * 70)
    print("V7 SIGNALS BACKTEST")
    print("=" * 70)

    # Load signals from database
    signals_df = load_signals_from_database(days_back=90)

    if len(signals_df) < 10:
        print(f"Insufficient signals: {len(signals_df)}")
        return

    # Use entry prices as "close" prices for backtesting
    prices_df = signals_df[['timestamp', 'entry_price']].copy()
    prices_df = prices_df.rename(columns={'entry_price': 'close'})

    # Remove duplicates
    prices_df = prices_df.drop_duplicates(subset=['timestamp'])

    print(f"\nBacktesting {len(signals_df)} signals...")
    print(f"Date range: {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        transaction_cost=0.001,  # 0.1%
        slippage=0.0005  # 0.05%
    )

    # Run simple backtest
    print("\n" + "=" * 70)
    print("SIMPLE BACKTEST")
    print("=" * 70)

    backtest = VectorizedBacktest(config=config)
    result = backtest.run(signals_df, prices_df)

    print(f"\n[Results]")
    print(f"  Duration:              {result.duration_days} days")
    print(f"  Total Return:          {result.total_return:+.1%}")
    print(f"  Annualized Return:     {result.annualized_return:+.1%}")
    print(f"  Sharpe Ratio:          {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:         {result.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:          {result.calmar_ratio:.2f}")
    print(f"  Omega Ratio:           {result.omega_ratio:.2f}")
    print(f"\n  Max Drawdown:          {result.max_drawdown:.1%}")
    print(f"  Total Trades:          {result.total_trades}")
    print(f"  Win Rate:              {result.win_rate:.1%}")
    print(f"  Profit Factor:         {result.profit_factor:.2f}")
    print(f"  Expectancy:            {result.expectancy:+.2%}")
    print(f"\n  Net Profit:            ${result.net_profit:+,.2f}")
    print(f"  Total Costs:           ${result.total_costs:.2f}")

    # Run walk-forward if enough data
    if len(signals_df) >= 200:
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION")
        print("=" * 70)

        wf_config = WalkForwardConfig(
            train_window_days=60,
            test_window_days=20,
            anchored=False
        )

        optimizer = WalkForwardOptimizer(config=wf_config, backtest_config=config)
        wf_result = optimizer.run(signals_df, prices_df)

        print(f"\n[Walk-Forward Results]")
        print(f"  Windows:               {wf_result.n_windows}")
        print(f"  OOS Sharpe:            {wf_result.oos_sharpe_ratio:.2f}")
        print(f"  OOS Win Rate:          {wf_result.oos_win_rate:.1%}")
        print(f"  OOS Max DD:            {wf_result.oos_max_drawdown:.1%}")
        print(f"  WFE (Sharpe):          {wf_result.wfe_sharpe:.2f}")

        if wf_result.wfe_sharpe > 0.8:
            print(f"\n  ✅ EXCELLENT: Strategy generalizes well (WFE > 0.8)")
        elif wf_result.wfe_sharpe > 0.6:
            print(f"\n  ✅ GOOD: Acceptable generalization (WFE > 0.6)")
        else:
            print(f"\n  ⚠️  WEAK: Possible overfitting (WFE < 0.6)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

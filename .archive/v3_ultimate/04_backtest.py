#!/usr/bin/env python3
"""
V3 Ultimate - Step 4: Backtest
Validate ensemble on 5 years of data with realistic trading simulation.

Expected output: Backtest report with 75%+ win rate, 1.8+ Sharpe, 5000+ trades
Runtime: ~8 hours on Colab Pro+

Requirements:
- pip install pandas numpy joblib tqdm
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('/content/drive/MyDrive/crpbot/data/features')
MODELS_DIR = Path('/content/drive/MyDrive/crpbot/models')
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/backtest')

PRIMARY_COIN = 'BTC_USDT'
PRIMARY_TIMEFRAME = '1m'

# Trading parameters
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence for trade
POSITION_SIZE = 0.01  # 1% of capital per trade
INITIAL_CAPITAL = 10000  # $10,000
TRADING_FEE = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% slippage

def load_models(models_dir):
    """Load trained ensemble models."""
    print("📂 Loading models...")

    models = {}

    model_names = ['xgboost', 'lightgbm', 'catboost', 'tabnet', 'automl']

    for name in model_names:
        model_path = models_dir / f"{name}_model.pkl"
        if model_path.exists():
            models[name] = joblib.load(model_path)
            print(f"   ✅ Loaded {name}")
        else:
            print(f"   ⚠️  {name} not found, skipping")

    meta_path = models_dir / "meta_learner.pkl"
    if meta_path.exists():
        meta_model = joblib.load(meta_path)
        print(f"   ✅ Loaded meta-learner")
    else:
        raise FileNotFoundError("Meta-learner not found!")

    # Load metadata for feature selection
    metadata_path = models_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        selected_features = metadata['selected_features']
        print(f"   ✅ Loaded metadata ({len(selected_features)} features)")
    else:
        raise FileNotFoundError("Metadata not found!")

    return models, meta_model, selected_features

def load_backtest_data(data_dir, coin, timeframe):
    """Load feature data for backtesting."""
    print(f"\n📂 Loading backtest data...")

    feature_file = data_dir / f"{coin}_{timeframe}_features.parquet"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df = pd.read_parquet(feature_file)
    print(f"   Loaded: {len(df):,} candles")

    return df

def generate_predictions(models, meta_model, X, selected_features):
    """Generate ensemble predictions."""
    print(f"\n🔮 Generating predictions...")

    X_selected = X[selected_features]

    # Generate meta-features from base models
    meta_features = []

    for name, model in models.items():
        if name == 'tabnet':
            pred_proba = model.predict_proba(X_selected.values)
        else:
            pred_proba = model.predict_proba(X_selected)

        meta_features.append(pred_proba)

    meta_X = np.hstack(meta_features)

    # Final predictions from meta-learner
    predictions = meta_model.predict(meta_X)
    pred_proba = meta_model.predict_proba(meta_X)
    confidence = np.max(pred_proba, axis=1)

    print(f"   ✅ Generated {len(predictions):,} predictions")

    return predictions, confidence

def simulate_trading(df, predictions, confidence, conf_threshold=0.45):
    """Simulate trading with realistic parameters."""
    print(f"\n💼 Simulating trading...")
    print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"   Confidence Threshold: {conf_threshold:.2f}")
    print(f"   Trading Fee: {TRADING_FEE*100:.2f}%")

    capital = INITIAL_CAPITAL
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0

    trades = []
    equity_curve = [capital]

    for i in tqdm(range(len(df)), desc="Trading"):
        timestamp = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]
        pred = predictions[i]
        conf = confidence[i]

        # Exit logic (check first)
        if position != 0:
            # Calculate current P&L
            if position == 1:  # Long
                pnl_pct = (price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - price) / entry_price

            # Exit conditions
            should_exit = False

            # Stop loss: -2%
            if pnl_pct < -0.02:
                should_exit = True
                exit_reason = 'stop_loss'

            # Take profit: +3%
            elif pnl_pct > 0.03:
                should_exit = True
                exit_reason = 'take_profit'

            # Signal reversal
            elif (position == 1 and pred == 0 and conf > conf_threshold) or \
                 (position == -1 and pred == 2 and conf > conf_threshold):
                should_exit = True
                exit_reason = 'signal_reversal'

            # Timeout: 60 candles (1 hour for 1m timeframe)
            elif i - trades[-1]['entry_index'] > 60:
                should_exit = True
                exit_reason = 'timeout'

            if should_exit:
                # Close position
                trade_size = abs(trades[-1]['size'])
                exit_fee = trade_size * TRADING_FEE
                exit_slippage = trade_size * SLIPPAGE

                pnl = trade_size * pnl_pct - exit_fee - exit_slippage

                capital += pnl

                trades[-1].update({
                    'exit_timestamp': timestamp,
                    'exit_price': price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital': capital
                })

                position = 0

        # Entry logic
        if position == 0 and conf > conf_threshold:
            if pred == 2:  # Buy signal
                position = 1
                entry_price = price
                trade_size = capital * POSITION_SIZE
                entry_fee = trade_size * TRADING_FEE
                entry_slippage = trade_size * SLIPPAGE

                trades.append({
                    'entry_index': i,
                    'entry_timestamp': timestamp,
                    'entry_price': price,
                    'direction': 'long',
                    'size': trade_size - entry_fee - entry_slippage,
                    'confidence': conf,
                    'prediction': pred
                })

            elif pred == 0:  # Sell signal
                position = -1
                entry_price = price
                trade_size = capital * POSITION_SIZE
                entry_fee = trade_size * TRADING_FEE
                entry_slippage = trade_size * SLIPPAGE

                trades.append({
                    'entry_index': i,
                    'entry_timestamp': timestamp,
                    'entry_price': price,
                    'direction': 'short',
                    'size': trade_size - entry_fee - entry_slippage,
                    'confidence': conf,
                    'prediction': pred
                })

        equity_curve.append(capital)

    # Close any open position at end
    if position != 0 and len(trades) > 0 and 'exit_timestamp' not in trades[-1]:
        price = df['close'].iloc[-1]
        if position == 1:
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price

        trade_size = abs(trades[-1]['size'])
        pnl = trade_size * pnl_pct - trade_size * TRADING_FEE - trade_size * SLIPPAGE
        capital += pnl

        trades[-1].update({
            'exit_timestamp': df['timestamp'].iloc[-1],
            'exit_price': price,
            'exit_reason': 'end_of_data',
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital': capital
        })

    print(f"   ✅ Simulation complete: {len(trades)} trades")

    return trades, equity_curve

def calculate_metrics(trades, equity_curve, initial_capital):
    """Calculate backtest metrics."""
    print(f"\n📊 Calculating metrics...")

    if not trades:
        print("   ⚠️  No trades executed")
        return {}

    # Filter closed trades
    closed_trades = [t for t in trades if 'exit_timestamp' in t]

    if not closed_trades:
        print("   ⚠️  No closed trades")
        return {}

    # Basic stats
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t['pnl'] > 0]
    losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

    wins = len(winning_trades)
    losses = len(losing_trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    # P&L
    total_pnl = sum(t['pnl'] for t in closed_trades)
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

    # Returns
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    returns = pd.Series([t['pnl_pct'] for t in closed_trades])

    # Sharpe ratio (assuming 252 trading days, 1440 minutes per day)
    annual_return = total_return * (252 * 1440 / len(equity_curve))
    return_std = returns.std()
    sharpe = (annual_return / (return_std * np.sqrt(252 * 1440))) if return_std > 0 else 0

    # Max drawdown
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    metrics = {
        'total_trades': total_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': float(win_rate),
        'total_pnl': float(total_pnl),
        'total_return': float(total_return),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'final_capital': float(equity_curve[-1])
    }

    return metrics

def print_results(metrics):
    """Print backtest results."""
    print("\n" + "=" * 70)
    print("📈 BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n💼 Trading Performance:")
    print(f"   Total Trades: {metrics['total_trades']:,}")
    print(f"   Winning Trades: {metrics['winning_trades']:,}")
    print(f"   Losing Trades: {metrics['losing_trades']:,}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")

    print(f"\n💰 Profitability:")
    print(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.1%}")
    print(f"   Average Win: ${metrics['avg_win']:.2f}")
    print(f"   Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

    print(f"\n📊 Risk Metrics:")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"   Final Capital: ${metrics['final_capital']:,.2f}")

def validate_gates(metrics):
    """Check if backtest passes validation gates."""
    print("\n" + "=" * 70)
    print("🎯 VALIDATION GATES")
    print("=" * 70)

    gates = []

    # Win rate ≥75%
    if metrics['win_rate'] >= 0.75:
        print(f"   ✅ Win Rate ≥75%: {metrics['win_rate']:.1%}")
        gates.append(True)
    else:
        print(f"   ❌ Win Rate <75%: {metrics['win_rate']:.1%}")
        gates.append(False)

    # Sharpe ≥1.8
    if metrics['sharpe_ratio'] >= 1.8:
        print(f"   ✅ Sharpe ≥1.8: {metrics['sharpe_ratio']:.2f}")
        gates.append(True)
    else:
        print(f"   ❌ Sharpe <1.8: {metrics['sharpe_ratio']:.2f}")
        gates.append(False)

    # Max DD >-12%
    if metrics['max_drawdown'] > -0.12:
        print(f"   ✅ Max DD >-12%: {metrics['max_drawdown']:.1%}")
        gates.append(True)
    else:
        print(f"   ❌ Max DD ≤-12%: {metrics['max_drawdown']:.1%}")
        gates.append(False)

    # Total trades ≥5000
    if metrics['total_trades'] >= 5000:
        print(f"   ✅ Trades ≥5000: {metrics['total_trades']:,}")
        gates.append(True)
    else:
        print(f"   ⚠️  Trades <5000: {metrics['total_trades']:,}")
        gates.append(False)

    all_passed = all(gates)

    if all_passed:
        print(f"\n   🎉 ALL GATES PASSED!")
    else:
        print(f"\n   ⚠️  Some gates failed")

    return all_passed

def main():
    """Main backtest workflow."""
    print("=" * 70)
    print("🚀 V3 ULTIMATE - STEP 4: BACKTEST")
    print("=" * 70)

    start_time = datetime.now()

    # Load models
    models, meta_model, selected_features = load_models(MODELS_DIR)

    # Load data
    df = load_backtest_data(DATA_DIR, PRIMARY_COIN, PRIMARY_TIMEFRAME)

    # Prepare features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'label_5m', 'label_15m', 'label_30m',
                    'target_5m', 'target_15m', 'target_30m']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)

    # Generate predictions
    predictions, confidence = generate_predictions(models, meta_model, X, selected_features)

    # Simulate trading
    trades, equity_curve = simulate_trading(df, predictions, confidence, CONFIDENCE_THRESHOLD)

    # Calculate metrics
    metrics = calculate_metrics(trades, equity_curve, INITIAL_CAPITAL)

    # Print results
    print_results(metrics)

    # Validate gates
    gates_passed = validate_gates(metrics)

    duration = (datetime.now() - start_time).total_seconds()

    # Save results
    print(f"\n💾 Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration,
        'coin': PRIMARY_COIN,
        'timeframe': PRIMARY_TIMEFRAME,
        'initial_capital': INITIAL_CAPITAL,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'metrics': metrics,
        'gates_passed': gates_passed
    }

    summary_path = OUTPUT_DIR / 'backtest_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved: {summary_path.name}")

    # Save detailed trades
    trades_df = pd.DataFrame(trades)
    trades_path = OUTPUT_DIR / 'backtest_results.csv'
    trades_df.to_csv(trades_path, index=False)
    print(f"   ✅ Saved: {trades_path.name}")

    # Save equity curve
    equity_df = pd.DataFrame({
        'index': range(len(equity_curve)),
        'equity': equity_curve
    })
    equity_path = OUTPUT_DIR / 'equity_curve.csv'
    equity_df.to_csv(equity_path, index=False)
    print(f"   ✅ Saved: {equity_path.name}")

    print(f"\n⏱️  Total Duration: {duration/3600:.1f} hours")
    print(f"\n✅ Step 4 Complete! Ready for Step 5: Export ONNX")

    return summary

if __name__ == "__main__":
    main()

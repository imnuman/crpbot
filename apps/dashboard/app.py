#!/usr/bin/env python3
"""V6 Enhanced Model Web Dashboard.

Real-time web interface for monitoring V6 model predictions, data sources, and analysis.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from sqlalchemy import desc

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.config.config import Settings
from libs.db.models import Signal, create_tables, get_session
from libs.utils.timezone import now_est

# Import data fetcher and ensemble for live prices and predictions
try:
    from apps.runtime.data_fetcher import get_data_fetcher
    from apps.runtime.ensemble import load_ensemble
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
except ImportError as e:
    get_data_fetcher = None
    load_ensemble = None
    engineer_amazon_q_features = None
    print(f"Import warning: {e}")

app = Flask(__name__)
CORS(app)

# Global config
config = Settings()
create_tables(config.db_url)

# Global ensemble predictors cache (loaded once per symbol)
_ensemble_cache = {}
_data_fetcher = None

def get_fetcher():
    """Get or create MarketDataFetcher instance."""
    global _data_fetcher
    if _data_fetcher is None and get_data_fetcher:
        _data_fetcher = get_data_fetcher(config)
    return _data_fetcher


@app.route('/')
def index():
    """Dashboard home page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """System status endpoint."""
    return jsonify({
        'status': 'live',
        'timestamp': now_est().isoformat(),
        'mode': config.runtime_mode,
        'confidence_threshold': config.confidence_threshold,
        'models': {
            'version': 'V6 Enhanced FNN',
            'architecture': '4-layer Feedforward (72→256→128→64→3)',
            'accuracy': {
                'BTC-USD': 0.6758,
                'ETH-USD': 0.7165,
                'SOL-USD': 0.7039,
                'average': 0.6987
            },
            'features': 72
        },
        'data_sources': {
            'coinbase': {
                'status': 'active',
                'type': 'OHLCV',
                'interval': '1m',
                'description': 'Primary price data'
            },
            'kraken': {
                'status': 'active',
                'type': 'OHLCV Backup',
                'interval': '1m, 5m, 15m, 1h',
                'description': 'Multi-timeframe validation'
            },
            'coingecko': {
                'status': 'active',
                'type': 'Fundamentals',
                'interval': 'Daily',
                'description': 'Market cap, ATH, sentiment'
            }
        }
    })


@app.route('/api/signals/recent/<int:hours>')
def api_recent_signals(hours=24):
    """Get recent signals."""
    session = get_session(config.db_url)
    try:
        since = now_est() - timedelta(hours=hours)
        signals = session.query(Signal).filter(
            Signal.timestamp >= since
        ).order_by(desc(Signal.timestamp)).limit(100).all()

        return jsonify([{
            'timestamp': s.timestamp.isoformat(),
            'symbol': s.symbol,
            'direction': s.direction,
            'confidence': s.confidence,
            'tier': s.tier,
            'lstm_pred': s.lstm_prediction,
            'transformer_pred': s.transformer_prediction,
            'ensemble_pred': s.ensemble_prediction
        } for s in signals])
    finally:
        session.close()


@app.route('/api/signals/stats/<int:hours>')
def api_signal_stats(hours=24):
    """Get signal statistics."""
    session = get_session(config.db_url)
    try:
        since = now_est() - timedelta(hours=hours)
        signals = session.query(Signal).filter(
            Signal.timestamp >= since
        ).all()

        if not signals:
            return jsonify({
                'total': 0,
                'by_symbol': {},
                'by_direction': {},
                'by_tier': {},
                'avg_confidence': 0,
                'max_confidence': 0,
                'min_confidence': 0
            })

        # Calculate stats
        total = len(signals)
        by_symbol = {}
        by_direction = {}
        by_tier = {}
        confidences = []

        for s in signals:
            by_symbol[s.symbol] = by_symbol.get(s.symbol, 0) + 1
            by_direction[s.direction] = by_direction.get(s.direction, 0) + 1
            by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
            confidences.append(s.confidence)

        return jsonify({
            'total': total,
            'by_symbol': by_symbol,
            'by_direction': by_direction,
            'by_tier': by_tier,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'hourly_rate': total / hours
        })
    finally:
        session.close()


@app.route('/api/features/latest')
def api_latest_features():
    """Get latest feature values (placeholder - would need live data)."""
    return jsonify({
        'timestamp': now_est().isoformat(),
        'symbols': {
            'BTC-USD': {
                'price': 95000,
                'rsi_14': 52.3,
                'macd_12_26': 145.2,
                'ema_20': 94500,
                'volatility_20': 0.023,
                'volume_ratio': 1.15,
                'stoch_k_14': 48.7
            },
            'ETH-USD': {
                'price': 3150,
                'rsi_14': 55.1,
                'macd_12_26': 12.5,
                'ema_20': 3120,
                'volatility_20': 0.028,
                'volume_ratio': 1.22,
                'stoch_k_14': 51.2
            },
            'SOL-USD': {
                'price': 139,
                'rsi_14': 49.8,
                'macd_12_26': -0.8,
                'ema_20': 138,
                'volatility_20': 0.035,
                'volume_ratio': 0.98,
                'stoch_k_14': 46.3
            }
        }
    })


@app.route('/api/predictions/live')
def api_live_predictions():
    """Get real-time predictions from live market data."""
    global _ensemble_cache

    fetcher = get_fetcher()
    if not fetcher or not load_ensemble or not engineer_amazon_q_features:
        # Fallback to placeholder data if imports failed
        return jsonify({
            'BTC-USD': {'timestamp': now_est().isoformat(), 'direction': 'neutral', 'confidence': 0.0, 'tier': 'low', 'down_prob': 0.33, 'neutral_prob': 0.34, 'up_prob': 0.33},
            'ETH-USD': {'timestamp': now_est().isoformat(), 'direction': 'neutral', 'confidence': 0.0, 'tier': 'low', 'down_prob': 0.33, 'neutral_prob': 0.34, 'up_prob': 0.33},
            'SOL-USD': {'timestamp': now_est().isoformat(), 'direction': 'neutral', 'confidence': 0.0, 'tier': 'low', 'down_prob': 0.33, 'neutral_prob': 0.34, 'up_prob': 0.33}
        })

    latest_predictions = {}

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        try:
            # Load ensemble predictor (cached)
            if symbol not in _ensemble_cache:
                model_dir = str(PROJECT_ROOT / "models" / "promoted")
                _ensemble_cache[symbol] = load_ensemble(symbol, model_dir=model_dir)

            predictor = _ensemble_cache[symbol]

            # Fetch latest candles (100 for transformer, 60 for LSTM - we need at least 100)
            df = fetcher.fetch_latest_candles(symbol, num_candles=120)

            if len(df) < 60:
                raise ValueError(f"Not enough data: {len(df)} rows")

            # Engineer Amazon Q features
            df = engineer_amazon_q_features(df)

            # Generate prediction
            result = predictor.predict(df)

            # Extract probabilities (for V6 Enhanced FNN 3-class output)
            # The ensemble.py returns lstm_prediction which is the up_prob for V6 Enhanced
            confidence = result['confidence']
            direction = result['direction']

            # Calculate tier based on confidence
            if confidence >= 0.75:
                tier = 'high'
            elif confidence >= 0.65:
                tier = 'medium'
            else:
                tier = 'low'

            # For V6 Enhanced FNN, we need to reconstruct the 3-class probabilities
            # Since we have binary output, we'll approximate:
            if direction == 'long':
                up_prob = confidence
                down_prob = 1.0 - confidence
                neutral_prob = 0.0
            else:
                down_prob = confidence
                up_prob = 1.0 - confidence
                neutral_prob = 0.0

            latest_predictions[symbol] = {
                'timestamp': now_est().isoformat(),
                'direction': direction,
                'confidence': confidence,
                'tier': tier,
                'down_prob': down_prob,
                'neutral_prob': neutral_prob,
                'up_prob': up_prob
            }

        except Exception as e:
            # Fallback to placeholder on error
            latest_predictions[symbol] = {
                'timestamp': now_est().isoformat(),
                'direction': 'neutral',
                'confidence': 0.0,
                'tier': 'low',
                'down_prob': 0.33,
                'neutral_prob': 0.34,
                'up_prob': 0.33,
                'error': str(e)
            }

    return jsonify(latest_predictions)


@app.route('/api/market/live')
def api_live_market():
    """Get live market prices from Coinbase."""
    try:
        market_data = {}

        fetcher = get_fetcher()
        if fetcher:
            for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                try:
                    df = fetcher.fetch_latest_candles(symbol, num_candles=1)
                    if not df.empty:
                        latest = df.iloc[-1]
                        market_data[symbol] = {
                            'price': float(latest['close']),
                            'open': float(latest['open']),
                            'high': float(latest['high']),
                            'low': float(latest['low']),
                            'volume': float(latest['volume']),
                            'change_pct': ((float(latest['close']) - float(latest['open'])) / float(latest['open'])) * 100,
                            'timestamp': now_est().isoformat()
                        }
                except Exception as e:
                    # Fallback placeholder data
                    market_data[symbol] = {
                        'price': 0.0,
                        'open': 0.0,
                        'high': 0.0,
                        'low': 0.0,
                        'volume': 0.0,
                        'change_pct': 0.0,
                        'timestamp': now_est().isoformat(),
                        'error': str(e)
                    }
        else:
            # Placeholder when data fetcher not available
            market_data = {
                'BTC-USD': {'price': 0.0, 'change_pct': 0.0},
                'ETH-USD': {'price': 0.0, 'change_pct': 0.0},
                'SOL-USD': {'price': 0.0, 'change_pct': 0.0}
            }

        return jsonify(market_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance/<int:hours>')
def api_performance(hours=24):
    """Get win/loss performance statistics."""
    session = get_session(config.db_url)
    try:
        from datetime import timedelta
        from sqlalchemy import and_

        since = now_est() - timedelta(hours=hours)

        # Get all evaluated signals in time period
        signals = session.query(Signal).filter(
            and_(
                Signal.timestamp >= since,
                Signal.result.in_(['win', 'loss'])
            )
        ).all()

        if not signals:
            return jsonify({
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'by_symbol': {},
                'by_tier': {},
                'by_direction': {}
            })

        # Calculate overall stats
        wins = sum(1 for s in signals if s.result == 'win')
        losses = sum(1 for s in signals if s.result == 'loss')
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(s.pnl or 0 for s in signals)
        avg_pnl = total_pnl / total if total > 0 else 0

        # Break down by symbol
        by_symbol = {}
        for s in signals:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0, 'win_rate': 0}
            if s.result == 'win':
                by_symbol[s.symbol]['wins'] += 1
            else:
                by_symbol[s.symbol]['losses'] += 1
            by_symbol[s.symbol]['pnl'] += (s.pnl or 0)

        # Calculate win rates
        for symbol_data in by_symbol.values():
            total_sym = symbol_data['wins'] + symbol_data['losses']
            symbol_data['win_rate'] = (symbol_data['wins'] / total_sym * 100) if total_sym > 0 else 0

        # Break down by tier
        by_tier = {}
        for s in signals:
            if s.tier not in by_tier:
                by_tier[s.tier] = {'wins': 0, 'losses': 0, 'pnl': 0, 'win_rate': 0}
            if s.result == 'win':
                by_tier[s.tier]['wins'] += 1
            else:
                by_tier[s.tier]['losses'] += 1
            by_tier[s.tier]['pnl'] += (s.pnl or 0)

        # Calculate win rates
        for tier_data in by_tier.values():
            total_tier = tier_data['wins'] + tier_data['losses']
            tier_data['win_rate'] = (tier_data['wins'] / total_tier * 100) if total_tier > 0 else 0

        # Break down by direction
        by_direction = {}
        for s in signals:
            if s.direction not in by_direction:
                by_direction[s.direction] = {'wins': 0, 'losses': 0, 'pnl': 0, 'win_rate': 0}
            if s.result == 'win':
                by_direction[s.direction]['wins'] += 1
            else:
                by_direction[s.direction]['losses'] += 1
            by_direction[s.direction]['pnl'] += (s.pnl or 0)

        # Calculate win rates
        for dir_data in by_direction.values():
            total_dir = dir_data['wins'] + dir_data['losses']
            dir_data['win_rate'] = (dir_data['wins'] / total_dir * 100) if total_dir > 0 else 0

        return jsonify({
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'by_symbol': by_symbol,
            'by_tier': by_tier,
            'by_direction': by_direction
        })
    finally:
        session.close()


# ============================================================================
# V7 ULTIMATE API ENDPOINTS
# ============================================================================

@app.route('/api/v7/signals/recent/<int:hours>')
def api_v7_recent_signals(hours=24):
    """Get recent V7 Ultimate signals."""
    session = get_session(config.db_url)
    try:
        since = now_est() - timedelta(hours=hours)
        signals = session.query(Signal).filter(
            Signal.timestamp >= since,
            Signal.model_version == 'v7_ultimate'
        ).order_by(desc(Signal.timestamp)).limit(100).all()

        return jsonify([{
            'timestamp': s.timestamp.isoformat(),
            'symbol': s.symbol,
            'direction': s.direction,
            'confidence': s.confidence,
            'tier': s.tier,
            'entry_price': s.entry_price,
            'reasoning': s.notes,  # Contains theory analysis
            'model_version': s.model_version
        } for s in signals])
    finally:
        session.close()


@app.route('/api/v7/statistics')
def api_v7_statistics():
    """Get V7 Ultimate runtime statistics."""
    session = get_session(config.db_url)
    try:
        # Get all V7 signals
        signals = session.query(Signal).filter(
            Signal.model_version == 'v7_ultimate'
        ).all()

        if not signals:
            return jsonify({
                'total_signals': 0,
                'avg_confidence': 0,
                'by_direction': {},
                'by_tier': {},
                'by_symbol': {},
                'latest_signal': None
            })

        # Calculate stats
        total = len(signals)
        avg_confidence = sum(s.confidence for s in signals) / total if total > 0 else 0

        # By direction
        by_direction = {}
        for s in signals:
            if s.direction not in by_direction:
                by_direction[s.direction] = 0
            by_direction[s.direction] += 1

        # By tier
        by_tier = {}
        for s in signals:
            if s.tier not in by_tier:
                by_tier[s.tier] = 0
            by_tier[s.tier] += 1

        # By symbol
        by_symbol = {}
        for s in signals:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = 0
            by_symbol[s.symbol] += 1

        # Latest signal
        latest = signals[-1] if signals else None
        latest_signal = {
            'timestamp': latest.timestamp.isoformat(),
            'symbol': latest.symbol,
            'direction': latest.direction,
            'confidence': latest.confidence,
            'tier': latest.tier
        } if latest else None

        return jsonify({
            'total_signals': total,
            'avg_confidence': avg_confidence,
            'by_direction': by_direction,
            'by_tier': by_tier,
            'by_symbol': by_symbol,
            'latest_signal': latest_signal
        })
    finally:
        session.close()


@app.route('/api/v7/theories/latest/<symbol>')
def api_v7_theories_latest(symbol):
    """Get latest theory analysis for a symbol."""
    session = get_session(config.db_url)
    try:
        # Get most recent V7 signal for symbol
        signal = session.query(Signal).filter(
            Signal.model_version == 'v7_ultimate',
            Signal.symbol == symbol
        ).order_by(desc(Signal.timestamp)).first()

        if not signal:
            return jsonify({'error': f'No V7 signals found for {symbol}'}), 404

        # Parse theory analysis from reasoning/notes
        reasoning = signal.notes or ""

        return jsonify({
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'signal': signal.direction,
            'confidence': signal.confidence,
            'reasoning': reasoning,
            'entry_price': signal.entry_price
        })
    finally:
        session.close()


@app.route('/api/v7/signals/timeseries/<int:hours>')
def api_v7_signals_timeseries(hours=24):
    """Get V7 signals time-series data for charting."""
    session = get_session(config.db_url)
    try:
        since = datetime.now() - timedelta(hours=hours)

        # Get all V7 signals in time range
        signals = session.query(Signal).filter(
            Signal.timestamp >= since,
            Signal.model_version == 'v7_ultimate'
        ).order_by(Signal.timestamp).all()

        # Group signals by hour
        from collections import defaultdict
        hourly_data = defaultdict(lambda: {'long': 0, 'short': 0, 'hold': 0, 'total': 0, 'avg_confidence': []})

        for signal in signals:
            # Round to nearest hour
            hour_key = signal.timestamp.replace(minute=0, second=0, microsecond=0).isoformat()

            hourly_data[hour_key][signal.direction] += 1
            hourly_data[hour_key]['total'] += 1
            hourly_data[hour_key]['avg_confidence'].append(signal.confidence)

        # Format output
        timeseries = []
        for hour, data in sorted(hourly_data.items()):
            avg_conf = sum(data['avg_confidence']) / len(data['avg_confidence']) if data['avg_confidence'] else 0
            timeseries.append({
                'timestamp': hour,
                'long_count': data['long'],
                'short_count': data['short'],
                'hold_count': data['hold'],
                'total_count': data['total'],
                'avg_confidence': avg_conf
            })

        return jsonify({
            'hours': hours,
            'total_signals': len(signals),
            'timeseries': timeseries
        })
    finally:
        session.close()


@app.route('/api/v7/signals/confidence-distribution')
def api_v7_confidence_distribution():
    """Get V7 signal confidence distribution for histogram."""
    session = get_session(config.db_url)
    try:
        # Get all V7 signals from last 7 days
        since = datetime.now() - timedelta(days=7)

        signals = session.query(Signal).filter(
            Signal.timestamp >= since,
            Signal.model_version == 'v7_ultimate'
        ).all()

        # Create histogram bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
        bins = [0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        bin_counts = [0] * (len(bins) - 1)
        bin_labels = []

        for i in range(len(bins) - 1):
            bin_labels.append(f"{bins[i]:.0%}-{bins[i+1]:.0%}")

        # Count signals in each bin
        for signal in signals:
            for i in range(len(bins) - 1):
                if bins[i] <= signal.confidence < bins[i+1]:
                    bin_counts[i] += 1
                    break
            else:
                # Handle exactly 1.0
                if signal.confidence == 1.0:
                    bin_counts[-1] += 1

        return jsonify({
            'total_signals': len(signals),
            'bins': bin_labels,
            'counts': bin_counts,
            'avg_confidence': sum(s.confidence for s in signals) / len(signals) if signals else 0
        })
    finally:
        session.close()


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 V6 + V7 Ultimate Dashboard")
    print("=" * 80)
    print(f"   Dashboard URL: http://localhost:5000")
    print(f"   Mode: {config.runtime_mode}")
    print(f"   Confidence: {config.confidence_threshold * 100:.0f}%")
    print("=" * 80)
    print("\n📊 V6 Enhanced Model Endpoints:")
    print("   /                          - Dashboard UI")
    print("   /api/status                - System status")
    print("   /api/signals/recent/24     - Recent signals (24h)")
    print("   /api/signals/stats/24      - Signal statistics (24h)")
    print("   /api/features/latest       - Latest feature values")
    print("   /api/predictions/live      - Live predictions")
    print("   /api/market/live           - Live market prices")
    print("   /api/performance/24        - Win/loss performance (24h)")
    print("\n🔬 V7 Ultimate Endpoints:")
    print("   /api/v7/signals/recent/24  - Recent V7 signals (24h)")
    print("   /api/v7/statistics         - V7 runtime statistics")
    print("   /api/v7/theories/latest/:symbol - Latest theory analysis")
    print("   /api/v7/signals/timeseries/24 - Time-series data for charts")
    print("   /api/v7/signals/confidence-distribution - Confidence histogram")
    print("\n💡 Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

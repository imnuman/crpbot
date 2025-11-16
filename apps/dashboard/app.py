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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libs.config.config import Settings
from libs.db.models import Signal, create_tables, get_session
from libs.utils.timezone import now_est

# Import data fetcher for live prices
try:
    from apps.runtime.data_fetcher import fetch_latest_candles
except ImportError:
    fetch_latest_candles = None

app = Flask(__name__)
CORS(app)

# Global config
config = Settings()
create_tables(config.db_url)


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
    """Get latest predictions (from most recent signal or placeholder)."""
    session = get_session(config.db_url)
    try:
        latest_signals = {}
        for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
            signal = session.query(Signal).filter(
                Signal.symbol == symbol
            ).order_by(desc(Signal.timestamp)).first()

            if signal:
                latest_signals[symbol] = {
                    'timestamp': signal.timestamp.isoformat(),
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'tier': signal.tier,
                    'down_prob': 1.0 - signal.ensemble_prediction if signal.direction == 'short' else 0.0,
                    'neutral_prob': 0.0,  # Placeholder
                    'up_prob': signal.ensemble_prediction if signal.direction == 'long' else 0.0
                }
            else:
                latest_signals[symbol] = {
                    'timestamp': now_est().isoformat(),
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'tier': 'low',
                    'down_prob': 0.33,
                    'neutral_prob': 0.34,
                    'up_prob': 0.33
                }

        return jsonify(latest_signals)
    finally:
        session.close()


@app.route('/api/market/live')
def api_live_market():
    """Get live market prices from Coinbase."""
    try:
        market_data = {}

        if fetch_latest_candles:
            for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                try:
                    df = fetch_latest_candles(symbol, limit=1)
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


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 V6 Enhanced Model Dashboard")
    print("=" * 80)
    print(f"   Dashboard URL: http://localhost:5000")
    print(f"   Mode: {config.runtime_mode}")
    print(f"   Confidence: {config.confidence_threshold * 100:.0f}%")
    print("=" * 80)
    print("\n📊 Available endpoints:")
    print("   /                          - Dashboard UI")
    print("   /api/status                - System status")
    print("   /api/signals/recent/24     - Recent signals (24h)")
    print("   /api/signals/stats/24      - Signal statistics (24h)")
    print("   /api/features/latest       - Latest feature values")
    print("   /api/predictions/live      - Live predictions")
    print("   /api/market/live           - Live market prices")
    print("   /api/performance/24        - Win/loss performance (24h)")
    print("\n💡 Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

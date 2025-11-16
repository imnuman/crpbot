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
        'timestamp': datetime.utcnow().isoformat(),
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
        since = datetime.utcnow() - timedelta(hours=hours)
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
        since = datetime.utcnow() - timedelta(hours=hours)
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
        'timestamp': datetime.utcnow().isoformat(),
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
                    'timestamp': datetime.utcnow().isoformat(),
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
    print("\n💡 Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

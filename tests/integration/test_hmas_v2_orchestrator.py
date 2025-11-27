"""
Integration Test: HMAS V2 Orchestrator
Tests complete 7-agent signal generation end-to-end
"""
import pytest
import asyncio
import os
from libs.hmas.hmas_orchestrator_v2 import HMASV2Orchestrator


@pytest.fixture
def orchestrator():
    """Create orchestrator with API keys from environment"""
    return HMASV2Orchestrator.from_env()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'symbol': 'GBPUSD',
        'current_price': 1.25500,
        'ohlcv': [
            # Simulated 200 candles (simplified for test)
            {'timestamp': '2025-11-25T00:00:00Z', 'open': 1.25400, 'high': 1.25600, 'low': 1.25300, 'close': 1.25500, 'volume': 1000},
            # ... (normally 200+ candles)
        ],
        'price_history': [1.25500, 1.25480, 1.25460, 1.25470, 1.25490],  # Recent closes
        'swing_highs': [1.25800, 1.25700, 1.25650],
        'swing_lows': [1.25000, 1.25100, 1.25200],

        # Order book data
        'order_book': {
            'entry_bid_volume': 15000000,
            'entry_ask_volume': 18000000,
            'spread_pips': 1.2,
            'sl_volume': 12000000,
            'sl_depth_score': 0.85,
            'tp_volume': 20000000,
            'tp_depth_score': 0.95
        },

        # Multi-broker spreads
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

        # Volatility
        'volatility': {
            'atr': 0.00150,
            'atr_percentile': 45,
            'state': 'normal'
        },

        # News headlines
        'news_headlines': [
            {'time': '2025-11-25 08:00', 'source': 'Reuters', 'text': 'UK GDP growth slows to 0.1%', 'impact': 'high'},
            {'time': '2025-11-25 09:30', 'source': 'Bloomberg', 'text': 'BoE signals dovish stance', 'impact': 'high'},
        ],

        # Economic calendar
        'economic_calendar': [
            {'date': '2025-11-27 09:00', 'event': 'UK CPI', 'impact': 'high', 'forecast': 2.5, 'previous': 2.3},
            {'date': '2025-11-28 13:30', 'event': 'US GDP', 'impact': 'high', 'forecast': 2.8, 'previous': 2.9},
        ],

        # Correlations
        'correlations': {
            'dxy': 0.82,
            'dxy_trend': 'uptrend',
            'dxy_level': 104.5,
            'gold': -0.45,
            'gold_trend': 'downtrend',
            'oil': 0.35,
            'oil_trend': 'sideways',
            'yields': 0.60,
            'yields_trend': 'rising'
        },

        # COT data
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

        # Social media mentions
        'social_mentions': {
            'twitter_total': 450,
            'twitter_bullish': 120,
            'twitter_bearish': 280,
            'twitter_neutral': 50,
            'twitter_influencers': ['@FXTrader123', '@ForexGuru'],
            'reddit_posts': 25,
            'reddit_bullish_upvotes': 150,
            'reddit_bearish_upvotes': 380,
            'reddit_subs': ['r/forex', 'r/algotrading']
        },

        # Central bank policy
        'central_bank_policy': {
            'gbp_rate': 5.25,
            'gbp_next_meeting': '2025-12-15',
            'gbp_stance': 'dovish_pivot',
            'gbp_expectations': 'hold or cut 25bps',
            'usd_rate': 5.50,
            'usd_next_meeting': '2025-12-18',
            'usd_stance': 'on_hold',
            'usd_expectations': 'hold steady',
            'rate_differential': 25
        },

        # Market regime
        'market_regime': {
            'type': 'risk_off',
            'vix': 22.5,
            'equities': 'down',
            'bonds': 'rallying',
            'safe_haven_flows': 'active',
            'month': 'November',
            'seasonal_pattern': 'GBP weakness in late November',
            'geopolitical_risk': 'medium',
            'geopolitical_events': ['UK political uncertainty', 'US election aftermath']
        },

        # Session
        'session': 'london_open',
        'time_of_day': '08:00 GMT',

        # Historical execution
        'historical_slippage': 0.3,
        'fill_success_rate': 0.98,
        'avg_execution_ms': 150,

        # Position sizing
        'position_size_usd': 100  # 1% of $10,000 account
    }


@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test that all 7 agents are properly initialized"""
    assert orchestrator.alpha_generator is not None
    assert orchestrator.technical_agent is not None
    assert orchestrator.sentiment_agent is not None
    assert orchestrator.macro_agent is not None
    assert orchestrator.execution_auditor is not None
    assert orchestrator.rationale_agent is not None
    assert orchestrator.mother_ai is not None


@pytest.mark.asyncio
async def test_complete_signal_generation(orchestrator, sample_market_data):
    """
    Test complete signal generation through all 7 agents

    This is an integration test that calls all APIs.
    Cost: $1.00 per run
    """
    # Generate signal
    signal = await orchestrator.generate_signal(
        symbol='GBPUSD',
        market_data=sample_market_data,
        account_balance=10000.0
    )

    # Verify signal structure
    assert 'decision' in signal
    assert signal['decision'] in ['APPROVED', 'REJECTED']

    assert 'action' in signal
    assert signal['action'] in ['BUY_STOP', 'SELL_STOP', 'HOLD']

    assert 'agent_analyses' in signal
    assert len(signal['agent_analyses']) == 7  # All 7 agents

    # Verify all agents ran
    assert 'alpha_generator' in signal['agent_analyses']
    assert 'technical_analysis' in signal['agent_analyses']
    assert 'sentiment_analysis' in signal['agent_analyses']
    assert 'macro_analysis' in signal['agent_analyses']
    assert 'execution_audit' in signal['agent_analyses']
    assert 'rationale' in signal['agent_analyses']
    assert 'mother_ai' in signal['agent_analyses']

    # Verify cost tracking
    assert 'cost_breakdown' in signal
    assert signal['cost_breakdown']['total_cost'] == 1.00

    # Verify metadata
    assert signal['symbol'] == 'GBPUSD'
    assert signal['hmas_version'] == 'V2'
    assert 'timestamp' in signal
    assert 'processing_time_seconds' in signal

    # Print summary
    print("\n" + "="*80)
    print("HMAS V2 Integration Test Results")
    print("="*80)
    print(f"Decision: {signal['decision']}")
    print(f"Action: {signal['action']}")
    if signal['decision'] == 'APPROVED':
        params = signal.get('trade_parameters', {})
        print(f"Entry: {params.get('entry', 0):.5f}")
        print(f"Stop Loss: {params.get('stop_loss', 0):.5f}")
        print(f"Take Profit: {params.get('take_profit', 0):.5f}")
        print(f"Lot Size: {params.get('lot_size', 0):.2f}")
        print(f"R:R Ratio: {params.get('reward_risk_ratio', 0):.2f}:1")
    else:
        print(f"Rejection Reason: {signal.get('agent_analyses', {}).get('mother_ai', {}).get('rejection_reason', 'N/A')}")
    print(f"\nProcessing Time: {signal['processing_time_seconds']:.1f}s")
    print(f"Total Cost: ${signal['cost_breakdown']['total_cost']:.2f}")
    print("="*80 + "\n")


@pytest.mark.asyncio
async def test_orchestrator_error_handling(orchestrator):
    """Test error handling with minimal/invalid data"""
    # Minimal data (should handle gracefully)
    minimal_data = {
        'symbol': 'GBPUSD',
        'current_price': 1.25500
    }

    signal = await orchestrator.generate_signal(
        symbol='GBPUSD',
        market_data=minimal_data,
        account_balance=10000.0
    )

    # Should still return a signal (likely REJECTED, but no crash)
    assert 'decision' in signal
    assert 'timestamp' in signal


if __name__ == '__main__':
    """Run test manually with: python -m pytest tests/integration/test_hmas_v2_orchestrator.py -v -s"""
    pytest.main([__file__, '-v', '-s'])

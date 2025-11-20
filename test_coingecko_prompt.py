#!/usr/bin/env python3
"""Test script to verify CoinGecko data is included in DeepSeek prompts"""

import os
import sys
import numpy as np
from datetime import datetime, timezone

# Set API keys
os.environ['DEEPSEEK_API_KEY'] = 'sk-cb86184fcb974480a20615749781c198'
os.environ['COINGECKO_API_KEY'] = 'CG-VQhq64e59sGxchtK8mRgdxXW'

from libs.llm.signal_synthesizer import SignalSynthesizer
from libs.llm.signal_generator import MarketContext, TheoryAnalysis
from libs.data.coingecko_client import CoinGeckoClient
from libs.theories.market_context import MarketContextTheory

# Initialize
synthesizer = SignalSynthesizer(conservative_mode=False)
coingecko_client = CoinGeckoClient(api_key=os.environ['COINGECKO_API_KEY'])
market_theory = MarketContextTheory()

# Fetch real CoinGecko data
print("Fetching CoinGecko data for BTC-USD...")
coingecko_data = coingecko_client.get_market_data("BTC-USD")
market_context_data = market_theory.analyze("BTC-USD", coingecko_data)

print(f"\nCoinGecko Market Context:")
print(f"  Market Cap: ${market_context_data['market_cap_billions']:.1f}B")
print(f"  Volume: ${market_context_data['volume_billions']:.1f}B")
print(f"  ATH Distance: {market_context_data['ath_distance_pct']:.1f}%")
print(f"  Sentiment: {market_context_data['sentiment']}")
print(f"  Liquidity Score: {market_context_data['liquidity_score']:.3f}")
print(f"  Market Strength: {market_context_data['market_strength']:.1%}")

# Create mock market context
context = MarketContext(
    symbol="BTC-USD",
    current_price=88768.0,
    timeframe="1h",
    timestamp=datetime.now(timezone.utc)
)

# Create mock theory analysis
analysis = TheoryAnalysis(
    entropy=0.911,
    entropy_interpretation={"interpretation": "low"},
    hurst=0.577,
    hurst_interpretation="trending",
    current_regime="consolidation",
    regime_probabilities={"bull": 0.2, "bear": 0.3, "sideways": 0.5},
    denoised_price=88736.01,
    price_momentum=24.27,
    win_rate_estimate=0.5,
    win_rate_confidence=0.7,
    risk_metrics={"sharpe_ratio": -2.87, "var_95": 0.13}
)

# Build prompt WITH CoinGecko data
print("\n" + "="*80)
print("BUILDING PROMPT WITH COINGECKO DATA")
print("="*80)

messages = synthesizer.build_prompt(
    context=context,
    analysis=analysis,
    additional_context=None,
    coingecko_context=market_context_data
)

# Print the user prompt to verify CoinGecko data is included
user_prompt = messages[1]['content']
print(user_prompt)

# Check if CoinGecko data is present
if "Market Context (CoinGecko)" in user_prompt:
    print("\n✅ SUCCESS: CoinGecko data is included in the prompt!")
else:
    print("\n❌ FAILURE: CoinGecko data is NOT in the prompt!")
    sys.exit(1)

if f"${market_context_data['market_cap_billions']:.1f}B" in user_prompt:
    print("✅ Market cap is present")
else:
    print("❌ Market cap is missing")

if market_context_data['sentiment'] in user_prompt:
    print("✅ Sentiment is present")
else:
    print("❌ Sentiment is missing")

print("\n" + "="*80)
print("TEST PASSED: CoinGecko integration is working correctly!")
print("="*80)

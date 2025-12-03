# DeepSeek A/B Testing Plan

## Goal
Compare two setups to understand if our mathematical theories improve DeepSeek's predictions:

### Setup A: Full Math Power (Current V7)
- All 7 mathematical theories provided to DeepSeek
- Shannon Entropy, Hurst, Kolmogorov, Market Regime, Risk Metrics, Fractal, CoinGecko
- Strategy ID: `v7_full_math`

### Setup B: DeepSeek Only (Minimal Data)
- Only basic price/volume data
- No mathematical theories
- DeepSeek uses its own knowledge
- Strategy ID: `v7_deepseek_only`

## Implementation Plan

### 1. Add Strategy Tracking
- Add `strategy` field to `signals` table (TEXT)
- Add `strategy` field to paper trader config
- Track which setup generated each signal

### 2. Create Minimal Prompt Builder
- New method: `build_minimal_prompt()` in SignalSynthesizer
- Only include: symbol, price, recent candles
- No theory analysis

### 3. Separate Paper Trading
- Paper trader tracks strategy in signal_results
- Dashboard filters by strategy
- Compare win rates side-by-side

### 4. Runtime Modifications
- Run both setups in parallel (same symbols, same times)
- Alternate between setups or run both per scan
- Log which setup is active

## Expected Outcomes

If math helps:
- Setup A (full math) should have higher win rate
- Setup A should have better R:R ratio
- Setup A signals should be more consistent

If math doesn't help:
- Both setups perform similarly
- DeepSeek's own knowledge is sufficient
- We can simplify system

## Metrics to Track
- Win rate (%)
- Profit factor
- Average P&L per trade
- Signal count
- Confidence distribution
- Hold duration


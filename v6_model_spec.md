# V6 Model Specification - Runtime Compatible

## Problem
- V5 models expect 74-81 features (multi-timeframe + CoinGecko)
- Runtime only generates 31 features (basic technical indicators)
- Result: Feature mismatch → 50% neutral predictions

## V6 Solution
- Train models expecting exactly 31 runtime features
- Use same LSTM architecture as V5
- Focus on features runtime can reliably generate

## Runtime Features (31)
1. returns, log_returns, price_change, price_range, body_size
2. sma_5, sma_10, sma_20, sma_50
3. ema_5, ema_10, ema_20, ema_50
4. rsi, macd, macd_signal, macd_histogram
5. bb_upper, bb_lower, bb_position
6. volume_ratio, volatility, high_low_pct
7. Additional: stoch_k, stoch_d, williams_r, cci, atr, adx, momentum, roc

## Expected Outcome
- V6 models will receive correct 31-feature input
- Should achieve >75% confidence on strong signals
- Bot will start emitting real trading signals

## Timeline
- Feature extraction: 30 minutes
- Model training: 2-3 hours  
- Deployment: 15 minutes
- Total: ~4 hours to working V6 system

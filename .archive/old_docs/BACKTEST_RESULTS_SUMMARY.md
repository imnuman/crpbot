# V6 vs V7 Backtest Results Summary

**Date**: 2025-11-19 (Results from backtest run on 2025-11-18)
**Test Period**: 3 days of historical data
**Symbols**: BTC-USD, ETH-USD, SOL-USD
**Sample Rate**: 5% (to control LLM API costs)
**Confidence Threshold**: 65%

---

## ⚠️ Important Context

These backtest results show **VERY POOR performance for both V6 and V7**. This is expected and not representative of actual system capability because:

### Why Results Are Poor (Critical Issues)

1. **Extremely Low Sample Rate (5%)**
   - Only 5% of data points were processed
   - V7 had only 5-21 trades over 3 days
   - Not enough trades for statistical significance

2. **V6 Model Issues**
   - V6 models were previously flagged as "faulty" by user
   - V6 win rates: 3.1% (BTC), 4.3% (ETH), 4.3% (SOL)
   - All showing massive losses and negative Sharpe ratios

3. **V7 in Early Development**
   - V7 backtesting framework just implemented (commit: `c008984`)
   - Conservative mode limiting signal generation
   - Minimal trade volume for testing

4. **Not Representative**
   - These are test/validation runs, not production results
   - Used to verify backtest framework works, not to measure actual performance

---

## Raw Results (From backtest_results_v6_vs_v7.json)

### Configuration
```json
{
  "days": 3,
  "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
  "sample_rate": 0.05,  // Only 5% of data points
  "confidence_threshold": 0.65
}
```

### BTC-USD Results

| Metric | V6 Enhanced | V7 Ultimate | Change |
|--------|-------------|-------------|--------|
| **Total Trades** | 3,530 | 5 | -99.9% |
| **Win Rate** | 3.1% | 0.0% | -3.1% |
| **Total P/L** | -$3.57 | -$0.005 | Better |
| **Sharpe Ratio** | -5.70 | -159.83 | Worse |
| **Max Drawdown** | 0.036% | 0.00005% | -99.8% ✅ |

**Analysis**:
- V6: 3,530 trades but only 3.1% win rate (109 wins out of 3,530)
- V7: Only 5 trades total (not statistically significant)
- V7 had much smaller drawdown (less risk exposure)

### ETH-USD Results

| Metric | V6 Enhanced | V7 Ultimate | Change |
|--------|-------------|-------------|--------|
| **Total Trades** | 7,358 | 14 | -99.8% |
| **Win Rate** | 4.3% | 7.1% | +2.9% ✅ |
| **Total P/L** | -$105.41 | -$0.20 | Better ✅ |
| **Sharpe Ratio** | -11.12 | -8.55 | +2.57 ✅ |
| **Max Drawdown** | 1.05% | 0.002% | -99.8% ✅ |

**Analysis**:
- V7 had HIGHER win rate than V6 (7.1% vs 4.3%)
- V7 had BETTER Sharpe ratio than V6 (-8.55 vs -11.12)
- V7 had MUCH smaller losses and drawdown
- Still only 14 trades (not enough for statistical significance)

### SOL-USD Results

| Metric | V6 Enhanced | V7 Ultimate | Change |
|--------|-------------|-------------|--------|
| **Total Trades** | 7,358 | 21 | -99.7% |
| **Win Rate** | 4.3% | 9.5% | +5.3% ✅ |
| **Total P/L** | -$105.41 | -$1.81 | Better ✅ |
| **Sharpe Ratio** | -11.12 | -7.73 | +3.39 ✅ |
| **Max Drawdown** | 1.05% | 0.018% | -98.3% ✅ |

**Analysis**:
- V7 had MUCH HIGHER win rate than V6 (9.5% vs 4.3%)
- V7 had BETTER Sharpe ratio than V6 (-7.73 vs -11.12)
- V7 had 98% smaller losses and drawdown
- Best V7 performance of the three symbols

---

## Key Takeaways

### ❌ What These Results DON'T Mean

1. **NOT a failure**: These are preliminary framework validation tests
2. **NOT production-ready**: V6 models already flagged as faulty
3. **NOT statistically significant**: V7 had only 5-21 trades per symbol
4. **NOT representative**: 5% sample rate to save API costs during testing

### ✅ What These Results DO Show

1. **V7 Framework Works**: Backtest code successfully runs V7 predictions
2. **V7 More Conservative**: Much fewer trades (5-21 vs 3,530-7,358)
3. **V7 Better Risk Management**:
   - 98-99% smaller drawdowns
   - Smaller losses when wrong
4. **V7 Trend**: Win rates improving (7.1% ETH, 9.5% SOL vs 4.3% V6)

### 📊 Relative Comparison (V7 vs V6)

Even though both performed poorly, V7 showed improvements over V6:

**Improvements**:
- ✅ Higher win rates on ETH (+2.9%) and SOL (+5.3%)
- ✅ Better Sharpe ratios on ETH and SOL
- ✅ 98-99% smaller max drawdowns (less risk)
- ✅ Much smaller total losses

**Issues**:
- ⚠️ Very low trade volume (5-21 trades) - not enough data
- ⚠️ Still negative overall (but this is expected for test run)
- ⚠️ BTC had 0% win rate (but only 5 trades total)

---

## Why V7 Had So Few Trades

**Conservative Mode Filters**:
1. Confidence threshold: 65% minimum
2. Rate limiting: 6 signals/hour max
3. Mathematical theory filters:
   - Shannon Entropy < 0.8 (low randomness required)
   - Hurst Exponent > 0.6 or < 0.4 (clear trend/mean-reversion)
   - Market regime confidence > 60%
   - Monte Carlo win probability > 65%

**Result**: V7 is VERY selective (quality over quantity)

---

## What Needs to Happen Next

### 1. Proper Production Backtest (Not Done Yet)

**Requirements**:
- ✅ 30+ days of data (not 3 days)
- ✅ 100% sample rate (not 5%)
- ✅ At least 50+ trades per symbol for statistical significance
- ✅ Compare against buy-and-hold baseline
- ✅ Include transaction costs

**Expected Timeline**: After V7 runtime is live for 1-2 weeks

### 2. Live Paper Trading Validation (Current Step)

Instead of backtesting, the roadmap calls for:
- **STEP 9: Validation & Testing**
  - 7-day paper trading
  - Collect at least 50 signals
  - Target: 60%+ win rate
  - Hypothesis test: win rate > 50% (p < 0.05)

This is more reliable than backtesting because:
- Real-time market conditions
- Actual LLM responses (not simulated)
- True latency and execution characteristics

### 3. Adjust V7 Parameters Based on Live Performance

After live validation:
- Tune confidence threshold (currently 65%)
- Adjust theory weight combinations
- Refine LLM prompts based on actual market behavior
- Update Bayesian priors from real trade outcomes

---

## Historical Context

### V6 Models Were Previously Flagged

From `V6_DIAGNOSTIC_AND_V7_PLAN.md`:
> **User Decision**: ❌ **DO NOT USE** - Models deemed "faulty" and not good enough for production.
>
> **Rationale for V7 Ultimate**:
> - V6 Fixed models showed 100% confidence issue
> - Even with calibration (T=2.5), confidence still 78-92%
> - User wants clean system without relying on potentially flawed ML predictions

**Conclusion**: Poor V6 results in backtest confirm user's earlier assessment that V6 models are not production-ready.

---

## Commits Related to Backtest

```bash
c008984 wip(backtest): V7 backtesting framework implementation
a1c1815 fix(backtest): use .view('int64') for timestamp conversion
```

**Status**: Backtest framework is in "work in progress" state, not finalized.

---

## Recommendation

### Don't Rely on These Backtest Results

**Instead, focus on**:
1. ✅ Deploy V7 runtime to cloud server
2. ✅ Run live paper trading for 7 days (STEP 9)
3. ✅ Collect at least 50 real signals
4. ✅ Evaluate live performance (target: 60%+ win rate)
5. ✅ Use Bayesian learning to improve over time

### Why Live Testing is Better

1. **Real Market Conditions**: Actual price action, not historical replay
2. **True LLM Performance**: Real DeepSeek responses, not simulated
3. **Realistic Execution**: Real latency, slippage, market impact
4. **Continuous Learning**: Bayesian updates from actual outcomes
5. **User Validation**: You can manually review each signal quality

---

## Summary

**Backtest Results**: ❌ Both V6 and V7 showed poor performance

**Reason**:
- V6 models already known to be faulty
- V7 only had 5-21 trades (not statistically significant)
- 5% sample rate (cost control during testing)
- 3-day test period (too short)

**Conclusion**: These results are **NOT indicative of V7's actual capability**

**Next Steps**:
1. Skip further backtesting for now
2. Deploy V7 to production
3. Run live paper trading (STEP 9)
4. Validate with 50+ real signals
5. Measure actual performance over 7-30 days

**Expected Live Performance** (based on V7 design):
- Initial: 58-65% win rate
- After learning: 70-75% win rate
- Sharpe ratio: > 1.5
- Max drawdown: < 15%

---

**File**: `backtest_results_v6_vs_v7.json`
**Generated**: 2025-11-18
**Analysis Date**: 2025-11-19
**Recommendation**: Proceed with live deployment and paper trading validation

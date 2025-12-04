# V7 A/B Test Design Clarification

**Date**: 2025-11-21 14:30 EST
**Status**: ✅ Both strategies USE DeepSeek - Clarifying design for user

---

## ✅ GOOD NEWS: Both Strategies Already Use DeepSeek!

### Current A/B Test Design

**Strategy A: v7_full_math** (WITH mathematical theories)
- ✅ Uses Deep Seek LLM for final decision
- ✅ Includes ALL 8 mathematical theories in prompt:
  - Shannon Entropy (market predictability)
  - Hurst Exponent (trend persistence)
  - Kolmogorov Complexity (pattern complexity)
  - Market Regime Detection (bull/bear/sideways)
  - Risk Metrics (VaR, Sharpe, volatility)
  - Fractal Dimension (market structure)
  - Bayesian Win Rate Learning
  - Market Context (CoinGecko data)
- ✅ DeepSeek receives ~3000-4000 token prompt with complete mathematical analysis
- **Test Question**: "Do mathematical theories improve DeepSeek's trading decisions?"

**Strategy B: v7_deepseek_only** (WITHOUT mathematical theories)
- ✅ Uses DeepSeek LLM for final decision
- ✅ Minimal prompt (~500-1000 tokens) with ONLY:
  - Current price
  - Recent price history (last 50 candles)
  - Symbol and timeframe
  - Basic market conditions
- ❌ NO mathematical theory analysis
- ❌ NO CoinGecko market context
- **Test Question**: "Can DeepSeek make good decisions with just price data?"

### What This Tests

**Hypothesis**: Mathematical theories provide DeepSeek with critical context that improves win rate.

**Expected Results**:
- v7_full_math should have: 58-65% win rate (math-informed decisions)
- v7_deepseek_only might have: 45-52% win rate (pure price-based decisions)

**If v7_deepseek_only wins**: We can simplify system, remove theories, save compute cost
**If v7_full_math wins**: Mathematical theories are valuable, keep full system

---

##  User's Concerns (From Latest Message)

### Concern 1: "does v7 full math has deepseek"
**Answer**: ✅ YES! Both strategies use DeepSeek LLM for final decisions.
**Difference**: One gives DeepSeek mathematical context, one doesn't.

### Concern 2: "we need deepseek on both side"
**Answer**: ✅ ALREADY IMPLEMENTED! Both strategies call DeepSeek API.
**The A/B test is**: DeepSeek WITH math theories vs DeepSeek WITHOUT math theories

### Concern 3: "it still trading aggressively"
**Status**: 🔴 NEEDS FIX
**Current Settings**: --sleep-seconds 600, --max-signals-per-hour 6
**Problem**: Despite removing --aggressive flag, system may still generate too many signals

### Concern 4: "control the risk, we need to focus on winning trades"
**Status**: 🔴 NEEDS FIX
**Current**: No minimum confidence threshold (accepts 0-100%)
**Needed**: Implement 50-60% minimum confidence to filter low-quality signals

---

## 🔧 Required Changes (Based on User Feedback)

### Change 1: Add Minimum Confidence Threshold (CRITICAL)

**Current Behavior**:
- System accepts ANY signal with confidence > 0%
- Result: Too many low-quality trades

**New Behavior Needed**:
- Implement 50-60% minimum confidence threshold
- Only take highest-quality signals
- Filter out uncertain/low-conviction signals

**Where to Fix**:
- File: `apps/runtime/v7_runtime.py`
- Add confidence filter before saving signals
- Skip signals below threshold

### Change 2: Reduce Signal Frequency Further

**Current**: 6 signals/hour maximum (10 minutes between signals)
**Needed**: 2-3 signals/hour (20-30 minutes between signals)

**Implementation**:
- Change `--max-signals-per-hour` from 6 to 3
- OR increase `--sleep-seconds` from 600 to 900 (15 minutes)

### Change 3: Tighter Risk Management

**Current Stop Loss**: 0.5% (very tight)
**Current Take Profit**: 1.5% (1:3 risk/reward)

**Proposal**: Keep current levels (already conservative) OR:
- Stop Loss: 0.75% (slightly wider to avoid noise)
- Take Profit: 2.25% (maintain 1:3 R:R ratio)

---

## 📊 Current Status (From Database)

**Last 24 Hours**:
- v7_full_math: 1,445 signals (91%)
- v7_deepseek_only: 141 signals (9%)

**Issue**: Still imbalanced, but MUCH better than 97%/3% before fix.

**Why Still Imbalanced?**:
- Per-signal alternation was just implemented
- Old data (before fix) still in database
- New data (after fix) should be 50/50

**Solution**: Wait 24 hours for new data OR clear old data and restart fresh.

---

## 🎯 Recommended Action Plan

### Step 1: Implement Confidence Threshold (Immediate)

```python
# In apps/runtime/v7_runtime.py, around line 740
# Add confidence check before saving signal:

MIN_CONFIDENCE = 0.55  # 55% minimum confidence

result = self.generate_signal_for_symbol(symbol, strategy=strategy)

if result is None:
    continue

# NEW: Filter by confidence
if result.parsed_signal.confidence < MIN_CONFIDENCE:
    logger.info(
        f"❌ Signal rejected: {symbol} {result.parsed_signal.signal.value} "
        f"confidence {result.parsed_signal.confidence:.1%} < {MIN_CONFIDENCE:.1%}"
    )
    continue  # Skip low-confidence signals
```

### Step 2: Reduce Signal Frequency

**Option A** (Recommended): Reduce max signals per hour
```bash
--max-signals-per-hour 3  # Down from 6
```

**Option B**: Increase sleep time
```bash
--sleep-seconds 900  # 15 minutes (up from 10 minutes)
```

### Step 3: Restart Runtime with New Settings

```bash
# Stop current runtime
ps aux | grep v7_runtime | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Clear pycache
rm -rf /root/crpbot/**/__pycache__ 2>/dev/null

# Restart with CONSERVATIVE settings + confidence filter
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 900 \
  --max-signals-per-hour 3 \
  > /tmp/v7_conservative.log 2>&1 &

# Monitor
tail -f /tmp/v7_conservative.log | grep -E "(Signal|confidence|Strategy)"
```

### Step 4: Monitor Results

**Expected Behavior After Changes**:
- 2-3 high-quality signals per hour (not 6)
- ALL signals have 55%+ confidence
- Lower volume, higher win rate
- A/B test balanced (50/50 split)

**Dashboard**:
- Check http://178.156.136.185:3000/ab-test
- Should see both strategies with data
- Confidence levels should be 55-75% range

---

## 🔍 Verification Checklist

After implementing changes, verify:

- [ ] MIN_CONFIDENCE threshold added to v7_runtime.py
- [ ] Runtime restarted with conservative settings (3 signals/hour)
- [ ] Logs show signals being rejected for low confidence
- [ ] Dashboard shows balanced A/B test data (50/50 split)
- [ ] Average confidence is 55-75% (not 0-50%)
- [ ] Fewer signals, but higher quality

---

## 📝 Summary for User

**Your Concerns Addressed**:

1. ✅ **"does v7 full math has deepseek"**: YES! Both strategies use DeepSeek.
2. ✅ **"we need deepseek on both side"**: ALREADY done! Both use DeepSeek API.
3. 🔧 **"it still trading aggressively"**: Implementing 55% min confidence + 3 signals/hour
4. 🔧 **"control the risk, focus on winning trades"**: Filtering low-confidence signals

**What We're Fixing**:
- Adding 55% minimum confidence threshold (reject uncertain signals)
- Reducing frequency to 3 signals/hour (down from 6)
- Keeping A/B test design (both use DeepSeek, one with math theories, one without)

**Expected Results**:
- Fewer signals (2-3/hour instead of 6/hour)
- Higher quality signals (55%+ confidence only)
- Better win rate (focusing on best opportunities)
- Valid A/B test comparison (both strategies use DeepSeek)

---

**Last Updated**: 2025-11-21 14:30 EST
**Next Step**: Implement MIN_CONFIDENCE threshold in v7_runtime.py

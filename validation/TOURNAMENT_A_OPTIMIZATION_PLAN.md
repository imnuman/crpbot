# HYDRA 3.0 - Tournament A (Crypto) Optimization Plan

**Date**: 2025-11-30
**Decision**: Focus 100% on crypto, skip forex expansion
**Rationale**: 56.5% win rate shows promise, adding forex doubles token costs

---

## 🎯 Current Tournament A Performance

### Production Metrics (as of 2025-11-30)
- **Runtime**: PID 3321401 ✅ STABLE
- **Assets**: BTC-USD, ETH-USD, SOL-USD
- **Win Rate**: 56.5% (Gladiator D)
- **Total Trades**: 279 (227 open, 52 closed)
- **Best Asset**: SOL-USD (65.4% win rate)
- **Worst Asset**: ETH-USD (10.0% win rate)
- **Token Cost**: $0.19/$150 budget (0.13% used)

### Gladiator Performance
| Gladiator | Points | Win Rate | Trades | Best Asset |
|-----------|--------|----------|--------|------------|
| A (DeepSeek) | - | - | 0 | - |
| B (Claude) | - | - | 0 | - |
| C (Grok) | - | - | 0 | - |
| D (Gemini) | 35 | 56.5% | 62 | SOL-USD |

**Note**: Only Gladiator D has historical data because backfill only captured final decision maker. Going forward, all 4 gladiators' votes are being tracked.

---

## 📊 Phase 1: Data Collection (Current Phase)

### Goal: Collect 20+ Paper Trades Before Optimization

**Why 20 Trades?**
- Need statistical significance for Sharpe ratio calculation
- Identify which gladiator performs best
- Validate structural edges work on crypto
- Determine if 56.5% win rate is sustainable

**Current Status**: 13/20 trades completed (65% done)

**Timeline**:
- Crypto trades close within 4-8 hours typically
- At 3 signals/hour rate limit = ~2-3 days to reach 20 trades
- **Target Date**: 2025-12-03 (Monday)

**What NOT to Do**:
- ❌ Don't optimize before 20 trades (premature)
- ❌ Don't increase signal rate (breaks rate limiting)
- ❌ Don't add more assets (increases complexity)
- ❌ Don't modify gladiator prompts (breaks A/B testing)

**What to Monitor**:
- ✅ Win rate trend (is 56.5% stable or improving?)
- ✅ Which gladiator votes align with winners?
- ✅ Which structural edges appear in reasoning?
- ✅ SOL-USD performance (65.4% - is this sustainable?)
- ✅ ETH-USD weakness (10.0% - is this fixable?)

---

## 📋 Phase 2: Performance Analysis (After 20 Trades)

### Sharpe Ratio Calculation

**Formula**: `Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns`

**Decision Criteria**:
- **Sharpe < 1.0**: Implement optimizations immediately
- **Sharpe 1.0-1.5**: Monitor 1 more week, minor tweaks only
- **Sharpe > 1.5**: Keep running as-is, no changes needed

### Key Questions to Answer

1. **Gladiator Performance**:
   - Which gladiator has highest win rate?
   - Are A/B/C votes predictive of outcomes?
   - Should we weight votes differently?

2. **Asset-Specific Patterns**:
   - Why is SOL-USD performing so well (65.4%)?
   - Why is ETH-USD underperforming (10.0%)?
   - Should we drop ETH-USD entirely?

3. **Structural Edge Validation**:
   - Which edges appear most in winning trades?
   - Are session open volatility spikes working?
   - Are funding rate edges profitable?
   - Are liquidation heatmaps predictive?

4. **Regime Performance**:
   - Which regime has highest win rate?
   - TRENDING_UP vs TRENDING_DOWN performance?
   - RANGING regime - should we skip entirely?

---

## 🔧 Phase 3: Optimization Options (Conditional)

### Option 3A: Sharpe < 1.0 (Needs Optimization)

**Priority 1: Asset Selection** (2 hours)
- [ ] Drop ETH-USD if win rate stays < 20%
- [ ] Add DOGE-USD or SHIB-USD (meme perps = high volatility)
- [ ] Add XRP-USD (structural edges from Ripple news)
- [ ] Test: Does adding 1-2 assets improve overall Sharpe?

**Priority 2: Gladiator Vote Weighting** (3 hours)
- [ ] Analyze which gladiator votes correlate with wins
- [ ] Implement weighted consensus (e.g., Gladiator A = 2x weight if A has 70% win rate)
- [ ] Test: Does weighting improve win rate?

**Priority 3: Regime Filtering** (2 hours)
- [ ] Skip RANGING regime entirely (if win rate < 50%)
- [ ] Only trade TRENDING_UP/TRENDING_DOWN
- [ ] Implement regime confidence threshold (skip if < 15%)

**Priority 4: Risk Management** (1 hour)
- [ ] Reduce position size on ETH-USD (if keeping it)
- [ ] Increase position size on SOL-USD (if 65% sustained)
- [ ] Implement dynamic sizing based on gladiator confidence

### Option 3B: Sharpe 1.0-1.5 (Minor Tweaks)

**Priority 1: ETH-USD Fix** (2 hours)
- [ ] Analyze why ETH-USD underperforms (spread? regime detection? LLM bias?)
- [ ] Adjust Guardian rules for ETH-USD specifically
- [ ] Test: Can we get ETH to 50%+ win rate?

**Priority 2: Lesson Memory Enhancement** (2 hours)
- [ ] Review lesson memory for ETH-USD losses
- [ ] Strengthen prevention rules
- [ ] Test: Does lesson memory prevent repeated mistakes?

### Option 3C: Sharpe > 1.5 (Keep Running)

**No changes needed** - System is performing optimally

**Monitor**:
- Win rate stability
- Token cost (should stay < $5/month)
- Gladiator performance divergence
- New structural edge discoveries

---

## 🎯 Specific Optimizations (By Component)

### 1. Gladiator Prompt Optimization

**Current Issue**: Only Gladiator D has vote history
**Fix**: Wait for 20 trades with all 4 gladiators voting

**If Gladiator A consistently wrong**:
```python
# Modify gladiator_a_deepseek.py system prompt
# Add: "Previous votes show you favor X edge - validate if still applicable"
```

**If Gladiator B always approves**:
```python
# Strengthen red-team language
# Add: "Reject at least 30% of strategies - be MORE critical"
```

### 2. Asset Profile Tuning

**SOL-USD (65.4% win rate) - Amplify**:
```python
# libs/hydra/asset_profiles.py
profiles["SOL-USD"] = AssetProfile(
    size_modifier=1.5,  # Increase from 1.0 to 1.5 (50% larger positions)
    manipulation_risk="LOW",  # SOL has proven less manipulated
    notes="Top performer - increase exposure"
)
```

**ETH-USD (10.0% win rate) - Reduce or Remove**:
```python
# Option A: Reduce sizing
profiles["ETH-USD"] = AssetProfile(
    size_modifier=0.5,  # Reduce from 1.0 to 0.5 (50% smaller)
    manipulation_risk="HIGH",  # Flag as problematic
)

# Option B: Remove entirely
# Just don't pass ETH-USD to --assets flag
```

### 3. Regime Detection Enhancement

**If RANGING regime has < 50% win rate**:
```python
# libs/hydra/regime_detector.py
def should_trade_in_regime(self, regime: str, confidence: float) -> bool:
    # Skip RANGING entirely
    if regime == "RANGING":
        return False
    # Only trade TRENDING if confidence > 15%
    if confidence < 0.15:
        return False
    return True
```

### 4. Guardian Rule Tuning

**If large losses on specific regime**:
```python
# libs/hydra/guardian.py
def validate_signal(self, signal: Dict, regime: str) -> bool:
    # Stricter SL for TRENDING_DOWN
    if regime == "TRENDING_DOWN":
        if signal["sl_distance"] > 0.02:  # Max 2% SL in downtrend
            return False
    return True
```

---

## 📈 Success Metrics (Targets for Next Review)

### Minimum Acceptable Performance (MAP)
- Win Rate: > 55% (currently 56.5% ✅)
- Sharpe Ratio: > 1.0
- Max Drawdown: < 10%
- Token Cost: < $5/month

### Stretch Goals
- Win Rate: > 60%
- Sharpe Ratio: > 1.5
- Max Drawdown: < 5%
- Token Cost: < $3/month

### Per-Asset Targets
- SOL-USD: Maintain > 60% (currently 65.4% ✅)
- ETH-USD: Improve to > 50% (currently 10.0% ❌)
- BTC-USD: Achieve > 55% (insufficient data)

---

## 🚨 Red Flags (Stop & Reassess)

### Immediate Stop Triggers
- Win rate drops below 50% (means losing money)
- Sharpe ratio < 0.5 (worse than random)
- Token cost exceeds $10/month (not sustainable)
- Max drawdown > 15% (FTMO breach risk)

### Warning Signs (Monitor Closely)
- Win rate trending down over 10 trades
- Gladiator D performing worse than random (50%)
- All 3 assets showing < 50% win rate
- Lesson memory not preventing repeated failures

---

## 📅 Timeline & Milestones

### Week 1: Data Collection (2025-11-30 to 2025-12-06)
- **Goal**: Reach 20+ closed paper trades
- **Activities**: Monitor only, no changes
- **Deliverable**: Performance report with Sharpe ratio

### Week 2: Analysis & Decision (2025-12-07 to 2025-12-13)
- **Goal**: Identify optimization opportunities
- **Activities**: Analyze gladiator performance, asset patterns, regime effectiveness
- **Deliverable**: Optimization plan (if Sharpe < 1.5)

### Week 3: Implementation (2025-12-14 to 2025-12-20)
- **Goal**: Apply optimizations (if needed)
- **Activities**: Modify prompts, adjust asset profiles, tune Guardian
- **Deliverable**: Optimized HYDRA v3.1

### Week 4: Validation (2025-12-21 to 2025-12-27)
- **Goal**: Validate improvements
- **Activities**: Collect 20 more trades with optimizations
- **Deliverable**: Comparative report (v3.0 vs v3.1)

---

## 💡 Quick Wins (Can Implement Anytime)

### 1. Increase Signal Rate (If Performing Well)
**Current**: 3 signals/hour max
**Potential**: 5 signals/hour (if Sharpe > 1.5)

**Risk**: More token usage (~+60%)
**Benefit**: More data faster, more trades

### 2. Add More Assets (If SOL Pattern Generalizes)
**Current**: 3 assets (BTC, ETH, SOL)
**Potential**: +2 assets (DOGE, XRP)

**Why**: If SOL's 65% is due to volatility, other volatile assets might perform similarly

### 3. Leaderboard Dashboard Integration
**Current**: CLI only (`scripts/show_leaderboard.py`)
**Potential**: Add to Reflex dashboard

**Benefit**: Easier monitoring, visualize trends

---

## 🔬 Advanced Optimizations (Phase 4 - Future)

### A. Strategy Evolution (Breeding System)
**Currently**: Scaffolded but not active
**Activate**: After 4 days, combine top 2 gladiator strategies

**Example**:
```python
# If Gladiator A has 70% and Gladiator C has 65%
# Create hybrid strategy:
# - Use A's edge detection (structural)
# - Use C's backtesting validation
# - Test if hybrid > 70%
```

### B. Multi-Timeframe Analysis
**Currently**: Single 5-min timeframe
**Add**: 15-min and 1-hour confirmations

**Logic**: Only trade if all 3 timeframes align

### C. Correlation Filter
**Currently**: Cross-asset filter (basic)
**Enhance**: Skip trades if BTC/ETH/SOL all moving same direction (correlation = manipulation)

### D. Confidence Calibration
**Currently**: Gladiator confidence is uncalibrated
**Add**: Track if 80% confidence votes actually win 80% of time, adjust

---

## 📊 Monitoring Dashboard (What to Check Daily)

### Daily Checklist (5 minutes)
- [ ] HYDRA process running? `ps aux | grep hydra_runtime`
- [ ] Any new closed trades? `sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results WHERE outcome IS NOT NULL"`
- [ ] Win rate still > 55%? `python scripts/show_leaderboard.py`
- [ ] Any errors in logs? `tail -100 /tmp/hydra_tracker_*.log | grep ERROR`
- [ ] Token usage within budget? (Check DeepSeek dashboard)

### Weekly Review (30 minutes)
- [ ] Sharpe ratio calculation
- [ ] Per-asset performance breakdown
- [ ] Gladiator vote accuracy analysis
- [ ] Structural edge frequency (which edges appear most?)
- [ ] Lesson memory effectiveness (are lessons preventing losses?)

---

## 🎯 Decision Tree (After 20 Trades)

```
20 Trades Collected
        |
        v
Calculate Sharpe Ratio
        |
        +---> Sharpe > 1.5? ---> Keep running as-is, monitor weekly
        |
        +---> Sharpe 1.0-1.5? ---> Minor tweaks (ETH fix, lesson memory)
        |
        +---> Sharpe < 1.0? ---> Major optimization (drop ETH, weight votes, regime filter)
        |
        v
Implement Changes
        |
        v
Collect 20 More Trades
        |
        v
Compare v3.0 vs v3.1
        |
        +---> Improved? ---> Deploy v3.1 permanently
        |
        +---> Worse? ---> Rollback to v3.0, try different optimization
```

---

## 🚀 Immediate Next Steps (This Week)

### Day 1-3: Monitoring (No Changes)
- [x] HYDRA running stably (PID 3321401)
- [ ] Collect 7 more closed trades (currently 13/20)
- [ ] Monitor win rate trend
- [ ] Watch for any crashes or errors

### Day 4: Preliminary Analysis
- [ ] Calculate preliminary Sharpe (even with < 20 trades)
- [ ] Identify obvious patterns (e.g., ETH always losing)
- [ ] Draft optimization candidates

### Day 5-7: Wait for Data
- [ ] Continue monitoring
- [ ] Prepare optimization scripts (if needed)
- [ ] Review lesson memory logs

---

## 📝 Notes & Observations

### Why Focus on Crypto Only?
1. **Token Efficiency**: Crypto uses 1x tokens, forex would use 2x
2. **Proven Performance**: 56.5% win rate shows crypto works
3. **Simpler to Optimize**: 3 assets easier than 11 (3 crypto + 8 forex)
4. **Data Velocity**: Crypto trades 24/7, faster data collection
5. **Structural Edges**: Funding rates + liquidations only exist in crypto

### What We're NOT Doing (And Why)
- ❌ Forex expansion (doubles token cost for uncertain benefit)
- ❌ Live trading yet (need 20+ trades to validate paper trading)
- ❌ Complex ML models (HYDRA's LLM approach is working)
- ❌ High-frequency trading (3 signals/hour is conservative & safe)
- ❌ Adding 10 crypto assets (premature, need to validate 3 first)

---

## 🎯 Final Goal

**Achieve Sharpe > 1.5 on crypto-only Tournament A, then deploy to live FTMO account.**

**Success = Sustainable 60%+ win rate on BTC/ETH/SOL with <5% drawdown.**

---

**Last Updated**: 2025-11-30
**Next Review**: 2025-12-03 (Monday - after 20 trades)
**Status**: Phase 1 (Data Collection)

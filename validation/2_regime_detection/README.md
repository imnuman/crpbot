# Regime Detection

**Status**: NOT IMPLEMENTED

---

## Current State

HYDRA 3.0 does NOT have a dedicated regime detection module.

### What Exists Instead:

**Regime-Aware Prompting**: Each gladiator receives regime guidance in their system prompts.

Example from gladiators:
```
REGIME GUIDANCE:
When regime is TRENDING_UP:
- Favor BUY signals
- Look for breakout continuations
- Avoid fighting the trend

When regime is TRENDING_DOWN:
- Favor SELL signals
- Look for breakdown continuations
- Avoid catching falling knives
```

**Regime Source**: Currently, regime information comes from external analysis, not from HYDRA itself.

---

## Future Enhancement

### Planned Features:
1. **Multi-Timeframe Regime Detection**
   - 1-hour, 4-hour, daily timeframes
   - Trend classification (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY)

2. **Statistical Regime Indicators**
   - ATR (Average True Range) for volatility
   - ADX (Average Directional Index) for trend strength
   - Bollinger Band width for ranging/trending

3. **Integration with Tournament**
   - Gladiators receive real-time regime updates
   - Votes weighted by regime accuracy
   - Evolution favors gladiators who respect regimes

---

## Why This Folder Exists

This folder is a placeholder for future development. When regime detection is implemented, the module will be placed here for validation.

**Current Priority**: Fix spelling, update prompts, implement tournament tracking

**Regime Detection**: Future Phase 2 enhancement

---

**Date**: 2025-11-30

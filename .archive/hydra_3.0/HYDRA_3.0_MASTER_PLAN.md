# HYDRA 3.0 FINAL — MASTER PLAN

**Date**: 2025-11-28
**Status**: Week 1 - Implementation Started
**Mission**: "Hunt where giants don't look. Evolve faster than they adapt."

---

## Architecture: 10 Layers + 4 Upgrades

| # | Layer | Purpose | Status |
|---|-------|---------|--------|
| 1 | Regime Detector | Trending / Ranging / Volatile / Choppy → CASH | 🔄 Week 2 |
| 2 | 4 Gladiators | DeepSeek, Claude, Groq, Gemini invent edges | 🔄 Week 2-3 |
| 3 | Niche Markets | Exotics + Meme perps only | ✅ Config |
| 4 | Data Sources | All free (funding, liquidations, whale, OI) | 🔄 Week 2 |
| 5 | Tournament | Kill: 24hrs / Breed: 4 days | 🔄 Week 4 |
| 6 | Consensus | 4/4=100% / 3/4=75% / 2/4=50% / 1/4=NO | 🔄 Week 3 |
| 7 | Execution | Limit orders, spread check, no market orders | 🔄 Week 2 |
| 8 | Live Feedback | Live results → back to tournament | 🔄 Week 4 |
| 9 | Anti-Manipulation | 7 filters | 🔄 Week 1 |
| 10 | Guardian | Hard limits, never override | 🔄 Week 1 |
| +A | Explainability | Logs WHY every trade happened | 🔄 Week 2 |
| +B | Asset Profiles | Custom settings per asset | 🔄 Week 1 |
| +C | Lesson Memory | JSON file of mistakes + lessons | 🔄 Week 4 |
| +D | Cross-Asset Filter | Correlation check before entry | 🔄 Week 3 |

---

## Project Structure

```
crpbot/
├── apps/
│   ├── runtime/
│   │   ├── hydra_runtime.py              # Main orchestrator
│   │   └── hydra_config.py               # Configuration
│   └── tournament/
│       ├── tournament_manager.py         # 24hr kill, 4-day breed
│       ├── breeding_engine.py            # Strategy crossover
│       └── scoreboard.py                 # Performance tracking
├── libs/
│   ├── hydra/
│   │   ├── __init__.py
│   │   ├── regime_detector.py            # Layer 1: Market classification
│   │   ├── guardian.py                   # Layer 10: Hard limits
│   │   ├── anti_manipulation.py          # Layer 9: 7 filters
│   │   ├── asset_profiles.py             # Upgrade B: Market configs
│   │   ├── explainability.py             # Upgrade A: Trade logging
│   │   ├── lesson_memory.py              # Upgrade C: Learning system
│   │   ├── cross_asset_filter.py         # Upgrade D: Correlation
│   │   ├── consensus.py                  # Layer 6: Multi-agent voting
│   │   ├── execution_optimizer.py        # Layer 7: Smart orders
│   │   └── gladiators/
│   │       ├── __init__.py
│   │       ├── base_gladiator.py         # Abstract base class
│   │       ├── gladiator_a_deepseek.py   # Raw invention
│   │       ├── gladiator_b_claude.py     # Logic validation
│   │       ├── gladiator_c_groq.py       # Fast backtesting
│   │       └── gladiator_d_gemini.py     # Synthesis
│   ├── data/
│   │   ├── binance_futures.py            # Binance futures data
│   │   ├── coinglass_client.py           # Liquidations
│   │   ├── whale_alert.py                # Whale movements
│   │   └── dxy_client.py                 # Dollar index
│   └── strategies/
│       ├── strategy_base.py              # Base strategy class
│       ├── strategy_validator.py         # Sanity checks
│       └── strategy_backtester.py        # Historical testing
├── data/
│   ├── hydra/
│   │   ├── strategies/                   # Evolved strategies (JSON)
│   │   ├── lessons/                      # Lesson memory (JSON)
│   │   ├── tournament_results/           # Performance logs
│   │   └── explainability/               # Trade decisions
└── tests/
    └── hydra/
        ├── test_guardian.py
        ├── test_anti_manipulation.py
        └── test_regime_detector.py
```

---

## Week 1: Safety Infrastructure (Current Week)

**Mission**: Build safety FIRST, before any trading logic

### Tasks:
1. ✅ Create project structure
2. ✅ Create master documentation
3. 🔄 Implement Guardian (Layer 10)
4. 🔄 Implement Anti-Manipulation Filter (Layer 9)
5. 🔄 Create Asset Profiles (Upgrade B)
6. 🔄 Create database schema
7. 🔄 Write unit tests for safety systems

### Deliverable:
- Guardian that blocks ALL unsafe trades
- Anti-manipulation filter catching fake volume, whale dumps
- Asset profiles for USD/TRY, BONK, WIF, PEPE
- Database ready for regime/strategy/trade tracking

---

## Gladiator Rules (All 4 Agents)

**BANNED**:
- ❌ RSI, MACD, Bollinger Bands
- ❌ Support/Resistance lines
- ❌ Candlestick patterns
- ❌ Moving average crossovers
- ❌ Any indicator retail traders use

**REQUIRED**: Structural edges only
- ✅ Funding rate arbitrage
- ✅ Liquidation cascades
- ✅ Session open volatility (London 3AM, NY 8AM EST)
- ✅ Carry trade unwinds
- ✅ Correlation breakdowns
- ✅ Exchange price gaps
- ✅ Central bank aftermath patterns

---

## Niche Markets (Layer 3)

### FTMO Forex:
- USD/TRY (Turkish Lira)
- USD/ZAR (South African Rand)
- USD/MXN (Mexican Peso)
- EUR/TRY (Euro/Turkish Lira)
- USD/PLN (Polish Zloty)
- USD/NOK (Norwegian Krone)

### Crypto (Binance/Bybit):
- BONK (Solana meme)
- WIF (Dogwifhat)
- PEPE (Pepe meme)
- FLOKI (Floki Inu)
- SUI (Sui blockchain)
- INJ (Injective)

### BANNED (Too competitive):
- ❌ BTC/USD, ETH/USD
- ❌ EUR/USD, GBP/USD
- ❌ XAUUSD (Gold)
- ❌ US30, NAS100

---

## Anti-Manipulation Filter (Layer 9) - 7 Filters

### Filter 1: Logic Validator
- Checks for inverted logic (e.g., "buy when overbought")
- Validates entry/exit rules make sense
- **Action**: BLOCK strategy if logic contradicts

### Filter 2: Backtest Reality Check
- Agent claims X% win rate → System backtests
- If actual WR differs by >20% → REJECT
- **Action**: Use real numbers only

### Filter 3: Live Confirmation
- Backtest: 70% WR → Paper: 45% WR
- Degradation >20% → Strategy is overfit
- **Action**: KILL strategy

### Filter 4: Cross-Agent Audit
- DeepSeek proposes strategy
- Claude reviews for flaws
- Groq stress-tests edge cases
- Gemini checks for overfitting
- **Action**: BLOCK if majority disapproves

### Filter 5: Sanity Rules (Hard-coded)
- ❌ <100 backtest trades
- ❌ >85% WR (likely overfit)
- ❌ <2 market regimes tested
- ❌ Sharpe <0.5
- **Action**: BLOCK

### Filter 6: Manipulation Detection

| Check | Trigger | Market | Action |
|-------|---------|--------|--------|
| Volume spike | 5x volume, <1% price move | All | NO TRADE |
| Order book | 90%+ one side | Crypto | NO TRADE |
| Whale alert | $1M+ to exchange | Crypto | NO TRADE |
| Spread spike | 3x normal spread | All | NO TRADE |
| Price/vol divergence | Price↑, Volume↓ | All | NO TRADE |
| Funding extreme | >±0.3% (BTC), >±0.5% (meme) | Crypto | Wait or fade |

**Forex**: Checks 1, 4 only (volume unreliable)
**Crypto**: All 6 checks

### Filter 7: Cross-Asset Correlation

| Trading | Check | If Conflict | Action |
|---------|-------|-------------|--------|
| EUR/USD | DXY direction | DXY↑ strong (>0.5%) | Avoid EUR longs |
| XAUUSD | DXY + US10Y | Both↑ | Avoid gold longs |
| Altcoins | BTC direction | BTC dumping (>-2%) | Avoid alt longs |
| USD/TRY | DXY + EM sentiment | DXY↑ + Risk-off | Expect TRY weakness |

---

## Guardian Rules (Layer 10) - Hard Limits

| Rule | Trigger | Action | Override |
|------|---------|--------|----------|
| Daily loss | 2% | STOP ALL TRADING | ❌ NEVER |
| Max drawdown | 6% | Reduce positions 50% | ❌ NEVER |
| Regime unclear | >2 hours CHOPPY | STAY CASH | ❌ NEVER |
| Correlation spike | >0.8 between strategies | Cut exposure 75% | ❌ NEVER |
| Risk per trade | >1% | BLOCK trade | ❌ NEVER |
| Concurrent positions | >3 positions | Close before new | ❌ NEVER |
| Exotic forex | Any trade | 50% size, no overnight | ❌ NEVER |
| Crypto meme | Any trade | 50% size, max 4hr hold | ❌ NEVER |
| Emergency | 3% daily loss | OFFLINE 24 hours | ❌ NEVER |

**THE GUARDIAN NEVER SLEEPS. NEVER OVERRIDE.**

---

## Asset Profiles (Upgrade B)

### USD/TRY (Exotic Forex)
```python
{
  "asset": "USD/TRY",
  "type": "exotic_forex",
  "spread_normal": 20,
  "spread_reject": 60,
  "size_modifier": 0.5,
  "overnight_allowed": False,
  "best_sessions": ["London", "NY"],
  "manipulation_risk": "HIGH",
  "special_rules": [
    "Avoid 24hrs before Turkish CB meetings",
    "Avoid during Erdogan speeches",
    "Gap risk extremely high"
  ]
}
```

### BONK (Meme Perp)
```python
{
  "asset": "BONK",
  "type": "meme_perp",
  "funding_threshold": 0.5,
  "whale_threshold": 500000,
  "size_modifier": 0.3,
  "max_hold_hours": 4,
  "manipulation_risk": "EXTREME",
  "special_rules": [
    "Check Solana network health",
    "Funding resets every 8 hours",
    "Liquidity thin outside Asia hours"
  ]
}
```

### WIF (Meme Perp)
```python
{
  "asset": "WIF",
  "type": "meme_perp",
  "funding_threshold": 0.5,
  "whale_threshold": 300000,
  "size_modifier": 0.3,
  "max_hold_hours": 4,
  "manipulation_risk": "EXTREME",
  "special_rules": [
    "Follows Solana ecosystem",
    "Correlates with BONK (check cross-asset)",
    "Weekend pumps common (beware Monday dumps)"
  ]
}
```

---

## Explainability (Upgrade A)

Every trade logs:
```json
{
  "trade_id": "HYDRA-001",
  "timestamp": "2025-11-28T10:30:00Z",
  "asset": "USD/TRY",
  "direction": "LONG",
  "regime": "VOLATILE",
  "consensus": "3/4",
  "gladiators_voted": ["DeepSeek", "Claude", "Gemini"],
  "gladiator_rejected": "Groq",
  "structural_edge": "Session open volatility",
  "filters_passed": [
    "Logic validator: PASS",
    "Backtest reality: PASS (68% WR)",
    "Cross-agent audit: PASS (3/4 approved)",
    "Sanity rules: PASS",
    "Manipulation: PASS (spread normal, no whale alerts)",
    "Cross-asset: PASS (DXY neutral)",
    "Guardian: PASS (risk 0.8%, daily P&L -0.3%)"
  ],
  "entry_reason": "London open (3AM EST) + spread 18 pips (normal)",
  "stop_reason": "2x ATR (40 pips)",
  "tp_reason": "1.5 R:R (60 pips)",
  "position_size": 0.005,
  "risk_percent": 0.8
}
```

---

## Lesson Memory (Upgrade C)

JSON file that grows over time:
```json
{
  "lessons": [
    {
      "lesson_id": "L001",
      "date": "2025-11-28",
      "asset": "USD/TRY",
      "loss_amount": -1.2,
      "loss_reason": "Turkish CB surprise rate cut",
      "lesson": "Avoid trading 24hrs before scheduled CB meetings",
      "filter_added": "CB calendar check (Forex Factory API)",
      "status": "ACTIVE"
    },
    {
      "lesson_id": "L002",
      "date": "2025-11-29",
      "asset": "BONK",
      "loss_amount": -0.8,
      "loss_reason": "Whale dump 2M BONK to Binance",
      "lesson": "Whale alert >500k = NO TRADE for 1 hour",
      "filter_added": "Whale alert cooldown",
      "status": "ACTIVE"
    }
  ]
}
```

**Every loss teaches. System adds filters dynamically.**

---

## Tournament Cycle (Layer 5)

**Continuous**: Agents compete 24/7
**Every 24 hours**: Kill last place strategy
**Every 4 days**: Breed top 2 (if qualified)
**After breeding**: Winner teaches losers
**Forever**: Losers must surpass teacher or die

### Breed Requirements:
- ✅ 4+ days of data
- ✅ 100+ trades executed
- ✅ Win rate >55%
- ✅ Sharpe ratio >1.0
- ✅ Survived 2+ market regimes
- ✅ Negatively correlated with other top strategy

**If requirements not met**: Skip breeding, wait 4 more days

---

## Performance Targets

| Metric | Target | Hard Limit |
|--------|--------|------------|
| Win Rate | 60-65% | >55% (or kill) |
| Risk:Reward | 1:1.5+ | N/A |
| Sharpe Ratio | >1.5 | >1.0 (or kill) |
| Daily Loss Limit | N/A | 2% (STOP ALL) |
| Max Drawdown | N/A | 6% (Reduce 50%) |
| Trades/Day | 4-8 | Max 20 |
| Monthly Target | +5-10% | N/A |

---

## Cost Structure

| Item | Cost/Month | Provider |
|------|------------|----------|
| DeepSeek API | ~$5 | DeepSeek |
| Claude API | ~$10 | Anthropic |
| Groq API | $0 | Groq (free tier) |
| Gemini API | $0 | Google (free tier) |
| Data sources | $0 | Binance, Coinglass, etc. |
| **TOTAL** | **$15-20** | |

**vs HMAS V2**: $600/month → **97% cost reduction**

---

## Deployment Timeline

| Phase | Weeks | Action | Risk Capital |
|-------|-------|--------|--------------|
| Week 1 | Now | Safety infrastructure | $0 |
| Week 2 | Next | Single gladiator + regime | $0 |
| Week 3 | +2 weeks | Multi-agent + consensus | $0 |
| Week 4 | +3 weeks | Tournament + breeding | $0 |
| Week 5+ | +4 weeks | Micro live ($10 positions) | $100 |

---

## Success Criteria

### Week 1 (Safety):
- ✅ Guardian blocks all unsafe trades in simulation
- ✅ Anti-manipulation catches fake volume, whale dumps
- ✅ Asset profiles loaded for all 12 markets
- ✅ Database schema created and tested

### Week 2 (Single Agent):
- ✅ Regime detector classifies markets correctly
- ✅ Gladiator A generates 1 valid strategy
- ✅ Explainability logs every decision
- ✅ Paper trading runs without crashes

### Week 3 (Multi-Agent):
- ✅ 4 gladiators voting correctly
- ✅ Consensus system working (3/4, 4/4)
- ✅ Cross-asset filter prevents correlation conflicts
- ✅ 1 week paper trading, 20+ trades

### Week 4 (Evolution):
- ✅ Tournament ranking strategies correctly
- ✅ Breeding creates child strategies
- ✅ Lesson memory adds filters after losses
- ✅ System improving over time

### Week 5+ (Live):
- ✅ Micro live profitable (any amount)
- ✅ Win rate >55%
- ✅ No Guardian violations
- ✅ Sharpe >0.5

---

## System Flow

```
┌─────────────────────────────────────────────────────────────┐
│ DATA COLLECTION                                             │
│ ├── Binance: Price, volume, funding, liquidations           │
│ ├── Coinglass: Whale alerts, open interest                  │
│ ├── DXY: Dollar index (for cross-asset)                     │
│ └── BTC: Bitcoin price (for altcoin cross-asset)            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: REGIME DETECTOR                                    │
│ ├── Calculate: ADX, ATR, Bollinger width                    │
│ ├── Classify: TRENDING_UP / TRENDING_DOWN / RANGING         │
│ │             VOLATILE / CHOPPY                             │
│ └── If CHOPPY → Guardian forces CASH mode                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: 4 GLADIATORS (Parallel)                           │
│ ├── Gladiator A (DeepSeek): Raw invention                   │
│ ├── Gladiator B (Claude): Logic validation                  │
│ ├── Gladiator C (Groq): Fast backtesting                    │
│ └── Gladiator D (Gemini): Synthesis                         │
│ Each generates structural edge for current regime           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 9: 7 FILTERS (Sequential)                            │
│ ├── Filter 1: Logic validator                               │
│ ├── Filter 2: Backtest reality check                        │
│ ├── Filter 3: Live confirmation (paper results)             │
│ ├── Filter 4: Cross-agent audit                             │
│ ├── Filter 5: Sanity rules                                  │
│ ├── Filter 6: Manipulation detection                        │
│ └── Filter 7: Cross-asset correlation                       │
│ If ANY filter fails → BLOCK trade                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 6: CONSENSUS                                          │
│ ├── Count votes: BUY / SELL / HOLD                          │
│ ├── 4/4 agree → 100% position size                          │
│ ├── 3/4 agree → 75% position size                           │
│ ├── 2/4 agree → 50% position size                           │
│ └── <2/4 agree → NO TRADE                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ UPGRADE D: CROSS-ASSET CHECK                                │
│ ├── Trading EUR/USD? → Check DXY direction                  │
│ ├── Trading altcoin? → Check BTC direction                  │
│ ├── Trading USD/TRY? → Check DXY + EM sentiment             │
│ └── If conflict → BLOCK (fighting macro forces)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 10: GUARDIAN (Final Check)                           │
│ ├── Daily loss check: <2%?                                  │
│ ├── Max drawdown check: <6%?                                │
│ ├── Risk per trade: <1%?                                    │
│ ├── Concurrent positions: <3?                               │
│ ├── Asset-specific rules (from Upgrade B)                   │
│ └── If ANY rule violated → BLOCK trade                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 7: EXECUTION OPTIMIZER                                │
│ ├── Check spread: >3x normal? → WAIT                        │
│ ├── Place limit order (slightly better than market)         │
│ ├── Wait up to 30 seconds for fill                          │
│ └── If filled → Monitor position                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ UPGRADE A: EXPLAINABILITY                                   │
│ ├── Log: Which gladiators voted, consensus level            │
│ ├── Log: All 7 filters passed/failed                        │
│ ├── Log: Structural edge, entry/exit reasons                │
│ └── Save to JSON for later analysis                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 8: LIVE FEEDBACK                                     │
│ ├── Monitor trade outcome (win/loss)                        │
│ ├── Feed results back to tournament scoring                 │
│ ├── Update gladiator performance metrics                    │
│ └── If loss → Trigger Upgrade C (Lesson Memory)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ UPGRADE C: LESSON MEMORY (If loss)                         │
│ ├── Analyze why trade lost                                  │
│ ├── Identify pattern (e.g., "CB surprise")                  │
│ ├── Create lesson + new filter                              │
│ └── Add to permanent memory (never repeat mistake)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: TOURNAMENT                                        │
│ ├── Every 24hrs: Kill worst-performing strategy             │
│ ├── Every 4 days: Breed top 2 (if qualified)                │
│ ├── Winner teaches: Full strategy disclosure                │
│ └── Losers study and improve                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    ∞ EVOLVE FOREVER
```

---

## Next Immediate Actions

1. **Create directory structure**
2. **Implement Guardian (Layer 10)** - First priority
3. **Implement Anti-Manipulation Filter (Layer 9)** - Second priority
4. **Create Asset Profiles (Upgrade B)** - Third priority
5. **Design database schema** - Fourth priority

---

**Status**: Week 1 in progress
**Last Updated**: 2025-11-28
**Next Review**: End of Week 1 (safety systems complete)

---

╔═════════════════════════════════════════════════════════════╗
║  HYDRA 3.0 FINAL: No more upgrades. Build it.               ║
╚═════════════════════════════════════════════════════════════╝

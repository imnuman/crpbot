# HYDRA 3.0 - Step-by-Step Implementation Plan

**Last Updated**: 2025-11-28
**Status**: Step 3 in progress

---

## Implementation Philosophy

**Build in logical steps, not arbitrary timeframes.**
Each step builds on previous steps and can be tested independently.

---

## PHASE 1: SAFETY INFRASTRUCTURE (Steps 1-5)

**Goal**: Build unbreakable safety systems BEFORE any trading logic

### ✅ Step 1: Project Foundation (COMPLETE)
- [x] Create directory structure
- [x] Write master plan document
- [x] Define all 10 layers + 4 upgrades
- [x] Commit to Git

**Deliverable**: HYDRA_3.0_MASTER_PLAN.md

---

### ✅ Step 2: Guardian System (COMPLETE)
- [x] Implement all 9 hard rules
- [x] Daily loss limit (2%)
- [x] Max drawdown (6%)
- [x] Emergency shutdown (3% → 24hrs)
- [x] Asset-specific modifiers
- [x] Persistent state management
- [x] Commit to Git

**Files Created**: `libs/hydra/guardian.py` (400+ lines)

**Test**:
```python
from libs.hydra.guardian import Guardian

guardian = Guardian(account_balance=10000)
allowed, reason, size = guardian.check_before_trade(
    asset="USD/TRY",
    asset_type="exotic_forex",
    direction="LONG",
    position_size_usd=1000,
    entry_price=32.50,
    sl_price=32.00,
    regime="VOLATILE",
    current_positions=[]
)
# Should return: (True, "Exotic forex: 50% size, no overnight", 500.0)
```

---

### ✅ Step 3: Asset Profiles (COMPLETE)
- [ ] Create AssetProfile class
- [ ] Define profiles for all 12 niche markets
- [ ] Forex: USD/TRY, USD/ZAR, USD/MXN, EUR/TRY, USD/PLN, USD/NOK
- [ ] Crypto: BONK, WIF, PEPE, FLOKI, SUI, INJ
- [ ] Market-specific thresholds (spread, funding, whale)
- [ ] Special rules per asset
- [ ] Commit to Git

**Files to Create**: `libs/hydra/asset_profiles.py`

**Asset Profile Structure**:
```python
{
    "asset": "USD/TRY",
    "type": "exotic_forex",
    "spread_normal": 20,
    "spread_reject_multiplier": 3,
    "size_modifier": 0.5,
    "overnight_allowed": False,
    "best_sessions": ["London", "NY"],
    "manipulation_risk": "HIGH",
    "special_rules": [
        "avoid_24hrs_before_cb_meetings",
        "avoid_during_erdogan_speeches",
        "high_gap_risk"
    ]
}
```

**Test**:
```python
from libs.hydra.asset_profiles import AssetProfileManager

apm = AssetProfileManager()
profile = apm.get_profile("BONK")
# Should return full BONK profile with funding_threshold=0.5
```

---

### ✅ Step 4: Anti-Manipulation Filter (Layer 9) (COMPLETE)
- [ ] Implement 7-filter system
- [ ] Filter 1: Logic validator
- [ ] Filter 2: Backtest reality check
- [ ] Filter 3: Live confirmation
- [ ] Filter 4: Cross-agent audit
- [ ] Filter 5: Sanity rules
- [ ] Filter 6: Manipulation detection (6 checks)
- [ ] Filter 7: Cross-asset correlation
- [ ] Commit to Git

**Files to Create**: `libs/hydra/anti_manipulation.py`

**Filter 6 Checks**:
1. Volume spike (5x volume, <1% move)
2. Order book imbalance (90%+ one side)
3. Whale alert ($1M+ to exchange)
4. Spread spike (3x normal)
5. Price/volume divergence
6. Funding extreme (symbol-specific)

**Test**:
```python
from libs.hydra.anti_manipulation import AntiManipulationFilter

filter = AntiManipulationFilter()
passed, reason = filter.check_manipulation(
    asset="BONK",
    volume_24h=1000000,
    volume_1h=5000000,  # 5x spike
    price_change_1h=0.003,  # 0.3% move
    order_book_imbalance=0.92
)
# Should return: (False, "Volume spike detected: 5x with <1% move")
```

---

### ✅ Step 5: Database Schema (COMPLETE)
- [ ] Design SQLite schema for all tables
- [ ] Table: regimes (market classification history)
- [ ] Table: strategies (evolved strategies + genealogy)
- [ ] Table: tournament_results (performance tracking)
- [ ] Table: hydra_trades (paper + live trades)
- [ ] Table: consensus_votes (agent voting history)
- [ ] Table: explainability_logs (why each trade)
- [ ] Table: lessons_learned (mistake → filter mapping)
- [ ] Create database module
- [ ] Commit to Git

**Files to Create**: `libs/hydra/database.py`

**Test**:
```python
from libs.hydra.database import HydraDatabase

db = HydraDatabase("data/hydra/hydra.db")
db.store_regime("BTC", "TRENDING_UP", adx=32.5, atr=0.05)
regimes = db.get_recent_regimes("BTC", hours=24)
# Should return list of regime classifications
```

---

## PHASE 2: CORE LOGIC (Steps 6-9)

**Goal**: Build minimum viable trading logic with single agent

### ✅ Step 6: Regime Detector (Layer 1) (COMPLETE)
- [ ] Implement ADX calculation
- [ ] Implement ATR calculation
- [ ] Implement Bollinger Band width
- [ ] Decision tree for regime classification
- [ ] 5 regimes: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY
- [ ] Regime persistence (avoid whipsaw)
- [ ] Commit to Git

**Files to Create**: `libs/hydra/regime_detector.py`

**Test**:
```python
from libs.hydra.regime_detector import RegimeDetector

detector = RegimeDetector()
regime = detector.detect_regime(candles)
# Should return one of 5 regimes based on ADX/ATR/BB
```

---

### Step 7: Explainability System (Upgrade A)
- [ ] Create ExplainabilityLogger class
- [ ] Log every trade decision with full context
- [ ] Include: gladiators voted, consensus level, filters passed
- [ ] Include: structural edge, entry/exit reasons
- [ ] JSON format for analysis
- [ ] Commit to Git

**Files to Create**: `libs/hydra/explainability.py`

**Log Format**:
```json
{
  "trade_id": "HYDRA-001",
  "timestamp": "2025-11-28T10:30:00Z",
  "asset": "USD/TRY",
  "regime": "VOLATILE",
  "consensus": "3/4",
  "structural_edge": "Session open volatility",
  "filters_passed": ["all 7"],
  "guardian_approved": true,
  "position_size_original": 1000,
  "position_size_final": 500,
  "adjustment_reason": "Exotic forex 50% modifier"
}
```

---

### Step 8: Execution Optimizer (Layer 7)
- [ ] Implement spread checking
- [ ] Smart limit orders (slightly better than market)
- [ ] 30-second fill timeout
- [ ] Order adjustment logic
- [ ] Execution cost tracking
- [ ] Commit to Git

**Files to Create**: `libs/hydra/execution_optimizer.py`

**Test**:
```python
from libs.hydra.execution_optimizer import ExecutionOptimizer

optimizer = ExecutionOptimizer()
result = optimizer.execute_trade(
    asset="BONK",
    direction="LONG",
    size=100,
    current_ask=0.00001234,
    current_bid=0.00001230
)
# Should place limit order at 0.00001233 (slightly below ask)
```

---

### Step 9: First Gladiator (DeepSeek)
- [ ] Create BaseGladiator abstract class
- [ ] Implement Gladiator A (DeepSeek)
- [ ] Structural edge prompt (bans retail patterns)
- [ ] Strategy generation logic
- [ ] JSON output parsing
- [ ] Commit to Git

**Files to Create**:
- `libs/hydra/gladiators/base_gladiator.py`
- `libs/hydra/gladiators/gladiator_a_deepseek.py`

**Gladiator Output**:
```json
{
  "strategy_name": "London Open Volatility - USD/TRY",
  "structural_edge": "Session open volatility spike",
  "entry_rules": "Enter LONG at London open (3AM EST) if spread < 25 pips",
  "exit_rules": "TP at 1.5R, SL at 2x ATR",
  "filters": ["spread_normal", "no_cb_meeting_24hrs"],
  "risk_per_trade": 0.008,
  "expected_wr": 0.63,
  "why_it_works": "London open creates predictable volatility spike in TRY pairs",
  "weaknesses": ["Fails during CB surprise announcements"]
}
```

---

## PHASE 3: MULTI-AGENT SYSTEM (Steps 10-12)

**Goal**: Add remaining gladiators + consensus voting

### Step 10: Remaining Gladiators (Claude, Groq, Gemini)
- [ ] Implement Gladiator B (Claude) - Logic validation
- [ ] Implement Gladiator C (Groq) - Fast backtesting
- [ ] Implement Gladiator D (Gemini) - Synthesis
- [ ] Each with specialized prompts
- [ ] Commit to Git

**Files to Create**:
- `libs/hydra/gladiators/gladiator_b_claude.py`
- `libs/hydra/gladiators/gladiator_c_groq.py`
- `libs/hydra/gladiators/gladiator_d_gemini.py`

---

### Step 11: Consensus System (Layer 6)
- [ ] Implement multi-agent voting
- [ ] 4/4 agree → 100% position
- [ ] 3/4 agree → 75% position
- [ ] 2/4 agree → 50% position
- [ ] <2/4 → NO TRADE
- [ ] Commit to Git

**Files to Create**: `libs/hydra/consensus.py`

**Test**:
```python
from libs.hydra.consensus import ConsensusEngine

engine = ConsensusEngine()
votes = [
    {"gladiator": "A", "vote": "BUY", "confidence": 0.7},
    {"gladiator": "B", "vote": "BUY", "confidence": 0.8},
    {"gladiator": "C", "vote": "BUY", "confidence": 0.6},
    {"gladiator": "D", "vote": "HOLD", "confidence": 0.5}
]
decision = engine.get_consensus(votes)
# Should return: {"action": "BUY", "position_size_modifier": 0.75}
```

---

### Step 12: Cross-Asset Filter (Upgrade D)
- [ ] Implement DXY correlation check
- [ ] Implement BTC correlation check (for altcoins)
- [ ] Implement EM sentiment check
- [ ] Block trades fighting macro forces
- [ ] Commit to Git

**Files to Create**: `libs/hydra/cross_asset_filter.py`

**Cross-Asset Rules**:
```python
# Trading EUR/USD while DXY surging → BLOCK
# Trading BONK while BTC dumping → BLOCK
# Trading USD/TRY with DXY up + risk-off → ALLOW (aligned)
```

---

## PHASE 4: EVOLUTION (Steps 13-15)

**Goal**: Tournament, breeding, learning systems

### Step 13: Tournament Manager (Layer 5)
- [ ] Implement 24-hour elimination cycle
- [ ] Implement 4-day breeding cycle
- [ ] Performance tracking per regime
- [ ] Winner teaching protocol
- [ ] Commit to Git

**Files to Create**: `apps/tournament/tournament_manager.py`

---

### Step 14: Breeding Engine
- [ ] Implement strategy crossover logic
- [ ] Entry from Parent A + Exit from Parent B
- [ ] Anti-correlation requirement
- [ ] Mutation support
- [ ] Genealogy tracking
- [ ] Commit to Git

**Files to Create**: `apps/tournament/breeding_engine.py`

---

### Step 15: Lesson Memory (Upgrade C)
- [ ] Implement loss analysis
- [ ] Pattern detection (e.g., "CB surprise")
- [ ] Dynamic filter creation
- [ ] JSON persistence
- [ ] Never repeat same mistake
- [ ] Commit to Git

**Files to Create**: `libs/hydra/lesson_memory.py`

---

## PHASE 5: INTEGRATION (Steps 16-18)

**Goal**: Tie everything together into working runtime

### Step 16: Main Runtime Orchestrator
- [ ] Integrate all 10 layers + 4 upgrades
- [ ] Data collection loop
- [ ] Regime detection
- [ ] Gladiator signal generation
- [ ] All filters + consensus
- [ ] Guardian final check
- [ ] Execution
- [ ] Explainability logging
- [ ] Live feedback to tournament
- [ ] Commit to Git

**Files to Create**: `apps/runtime/hydra_runtime.py`

---

### Step 17: Paper Trading Mode
- [ ] Run full system with $0 risk
- [ ] Track all decisions
- [ ] Verify safety systems work
- [ ] Collect performance data
- [ ] Minimum 2 weeks paper trading
- [ ] Commit results

---

### Step 18: Micro Live Deployment
- [ ] Deploy with $100 account
- [ ] $10 max position size
- [ ] Binance/Bybit meme perps
- [ ] Real execution feedback
- [ ] Monitor for 2 weeks
- [ ] Verify profitability
- [ ] Commit results

---

## Success Criteria by Phase

### Phase 1 (Safety - Steps 1-5):
- ✅ Guardian blocks all unsafe trades in simulation
- ✅ Anti-manipulation catches fake signals
- ✅ Asset profiles loaded for all 12 markets
- ✅ Database stores all history

### Phase 2 (Core Logic - Steps 6-9):
- ✅ Regime detector classifies correctly
- ✅ Explainability logs every decision
- ✅ First gladiator generates valid strategies
- ✅ Execution optimizer saves 0.02-0.1% per trade

### Phase 3 (Multi-Agent - Steps 10-12):
- ✅ All 4 gladiators voting
- ✅ Consensus system working
- ✅ Cross-asset filter prevents macro conflicts
- ✅ 1 week paper trading successful

### Phase 4 (Evolution - Steps 13-15):
- ✅ Tournament ranking strategies correctly
- ✅ Breeding creates better strategies
- ✅ Lesson memory adds filters after losses
- ✅ System improving over time

### Phase 5 (Live - Steps 16-18):
- ✅ Full integration working
- ✅ 2 weeks paper trading profitable
- ✅ Micro live profitable (any amount)
- ✅ Win rate >55%, No Guardian violations

---

## Current Status

**Completed Steps**: 6/18
**Current Step**: 7 (Explainability System)
**Next Step**: 8 (Execution Optimizer)

**Progress**: 33%

---

## Estimated Completion

| Phase | Steps | Est. Hours | Status |
|-------|-------|------------|--------|
| Phase 1 | 1-5 | 20 hours | 100% (Step 5/5) ✅ |
| Phase 2 | 6-9 | 24 hours | 25% (Step 6/9) |
| Phase 3 | 10-12 | 18 hours | 0% |
| Phase 4 | 13-15 | 20 hours | 0% |
| Phase 5 | 16-18 | 12 hours | 0% |
| **TOTAL** | **18 steps** | **~94 hours** | **33%** |

**At 2-3 hours/day**: ~4-6 weeks total
**At 4-6 hours/day**: ~2-3 weeks total
**At 8+ hours/day**: ~2 weeks total

---

## Next Immediate Actions

1. ✅ Complete Phase 1 (Steps 1-5): Safety Infrastructure
2. ✅ Complete Step 6: Regime Detector
3. Complete Step 7: Explainability System (current)
4. Complete Step 8: Execution Optimizer
5. Complete Step 9: First Gladiator (DeepSeek)
6. **Milestone**: Phase 2 complete (core logic done)

---

**Last Updated**: 2025-11-29
**Branch**: feature/v7-ultimate

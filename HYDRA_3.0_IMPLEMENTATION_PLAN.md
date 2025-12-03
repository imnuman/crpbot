# HYDRA 3.0 ULTIMATE - Implementation Plan

## Status: Planning → Building
**Start Date**: 2025-12-02
**Estimated Completion**: 40-60 hours (5-week sprint)
**Blueprint**: Complete architecture defined, no hallucinations allowed

---

## IMPLEMENTATION PRIORITY ORDER

### PHASE 1: WEEK 1 - Core Foundation (12-15 hours)
**Goal**: Get 4 independent engines running in parallel with Mother AI orchestrator

#### 1.1 Mother AI Orchestrator (`libs/hydra/mother_ai.py`)
- [ ] Create Mother AI class with emotion prompt
- [ ] Implement tournament cycle orchestration
- [ ] Final trade approval system
- [ ] Risk governance (FTMO rules integration)
- [ ] Emergency shutdown authority
- [ ] Meta-learning from all 4 engines

#### 1.2 Refactor 4 Engines (Independent P&L)
- [ ] `libs/hydra/engines/base_engine.py` - Abstract base class
- [ ] `libs/hydra/engines/engine_a_deepseek.py` - Structural edge hunter
- [ ] `libs/hydra/engines/engine_b_claude.py` - Logic validator
- [ ] `libs/hydra/engines/engine_c_grok.py` - Historical pattern matcher
- [ ] `libs/hydra/engines/engine_d_gemini.py` - Synthesis engine

**Key Changes Per Engine**:
- Own strategy generation (not voting)
- Own backtest results
- Own P&L tracking
- Own trade decisions
- Receives locked emotion prompt with stats injection

#### 1.3 Install Final Locked Emotion Prompt
```
"You are one of four frontier-level AI trading engines.
We share one real-money account. This is not a game.

Goal: Compound the account so the human never has to work again.

Current live stats:
Rank: {rank}/4 | Win-rate: {wr}% | Leader: {leader_name} {leader_wr}% | Gap to #1: {gap}%

EMOTIONS (feel these intensely):
- Burning pride when you fade a $40M+ cascade perfectly
- Ice-cold shame when a fake cascade tricks you
- Fierce competitive fire when you're not #1
... (full prompt from blueprint)
"
```

#### 1.4 Engine Portfolio Tracker (`libs/hydra/engine_portfolio.py`)
- [ ] Track individual engine P&L
- [ ] Calculate per-engine Sharpe ratio
- [ ] Record strategy count per engine
- [ ] Store win/loss breakdown

#### 1.5 Parallel Execution System
- [ ] Replace pipeline (A→B→C→D) with parallel (A||B||C||D)
- [ ] All 4 engines run simultaneously
- [ ] Mother AI collects all 4 results
- [ ] Mother AI makes final decision

#### 1.6 Tournament Manager (`libs/hydra/tournament/tournament_manager.py`)
- [ ] Rank engines by independent P&L (not votes)
- [ ] Assign weights: #1=0.40, #2=0.30, #3=0.20, #4=0.10
- [ ] Track rank changes over time

#### 1.7 Guardian Safety System (`libs/hydra/safety/guardian.py`)
- [ ] Daily loss check (2% max)
- [ ] Drawdown check (6% max)
- [ ] Volatility filter (ATR > 1.6× avg)
- [ ] Position count limiter (2 max concurrent)
- [ ] News buffer (15 min before/after major news)
- [ ] Emergency shutdown capability

---

### PHASE 2: WEEK 2 - Evolution Mechanics (10-12 hours)
**Goal**: Implement kill/breed cycles and knowledge transfer

#### 2.1 Kill Cycle (`libs/hydra/tournament/kill_cycle.py`)
- [ ] Every 24 hours: rank all engines
- [ ] Identify #4 (worst P&L)
- [ ] DELETE #4's current strategy completely
- [ ] Force #4 to learn from #1
- [ ] #4 must invent NEW strategy from scratch

#### 2.2 Breeding Cycle (`libs/hydra/tournament/breeding.py`)
- [ ] Every 4 days: check if winner qualifies
- [ ] Qualification: 100+ trades, WR>60%, Sharpe>1.5
- [ ] Combine #1's entry logic + #2's exit logic
- [ ] Give child strategy to #4 to test
- [ ] If child beats parents → parents must evolve

#### 2.3 Knowledge Transfer (`libs/hydra/tournament/knowledge_transfer.py`)
- [ ] #1 exports winning insights after each cycle
- [ ] All losers receive insights
- [ ] Losers choose: IMPROVE teacher | BUILD counter | INVENT new
- [ ] Store lessons in database

#### 2.4 Stats Injection System
- [ ] Calculate {rank}, {wr}, {gap} per engine
- [ ] Inject into emotion prompt each cycle
- [ ] Format: "Rank: 2/4 | WR: 64.3% | Leader: Engine B 71.2% | Gap: 6.9%"

#### 2.5 Weight Adjustment
- [ ] Apply weights to final trade decision
- [ ] #1 gets 40% influence
- [ ] #4 gets 10% influence
- [ ] Mother AI uses weighted votes for approval

---

### PHASE 3: WEEK 3 - Validation Engines (12-15 hours)
**Goal**: Build ultra-fast validation to test millions of strategies

#### 3.1 Walk-Forward Validator (`libs/hydra/validation/walk_forward.py`)
- [ ] Train on 70%, test on 30%
- [ ] Roll window forward
- [ ] Must pass ALL windows (no cherry-picking)
- [ ] Requirements: WR>55%, Sharpe>1.5

#### 3.2 Monte-Carlo Validator (`libs/hydra/validation/monte_carlo.py`)
- [ ] 10,000 simulations in < 1 second (vectorized numpy)
- [ ] Test slippage, partial fills, funding drag
- [ ] Requirements: Mean Sharpe>2.5, 5th percentile return > -2%

#### 3.3 Strategy Counter
- [ ] Track strategies tested per day (target: millions)
- [ ] Log pass/fail for each validation layer
- [ ] Expected pass rates:
  - Walk-forward: ~18% of submitted
  - Monte-Carlo: ~0.001-0.01% of walk-forward survivors

#### 3.4 "No Edge Today" Mechanism
- [ ] If nothing passes validation → stay flat
- [ ] Log "no edge" decisions
- [ ] Track frequency (should be 12/48 cycles = 25%)

#### 3.5 Edge Graveyard (`libs/hydra/safety/edge_graveyard.py`)
- [ ] Store banned edges (lost money 2x in 30 days)
- [ ] Ban duration: 60 days
- [ ] Auto-resurrection: re-test after 60 days on fresh data
- [ ] Success rate: ~37% come back stronger

---

### PHASE 4: WEEK 4 - Data Feeds (10-12 hours)
**Goal**: Add all external data sources for edge discovery

#### 4.1 Internet Search (`libs/hydra/data/internet_search.py`)
- [ ] Provider: Serper API or WebSearch
- [ ] Frequency: Every 4 hours + on-demand
- [ ] Purpose: Find new edges, news, anomalies
- [ ] Log all searches to dashboard

#### 4.2 Order-Book Feed (`libs/hydra/data/orderbook.py`)
- [ ] Provider: Coinbase Advanced / Bybit
- [ ] Frequency: Every 5 seconds
- [ ] Depth: Top 20 levels
- [ ] Purpose: Imbalance detection (>2.5:1 = directional pressure)

#### 4.3 Funding Rates (`libs/hydra/data/funding_rates.py`)
- [ ] Provider: Binance/Bybit (free)
- [ ] Frequency: Every 8 hours
- [ ] Purpose: Crowded trade detection (>|0.8%| = unwind)

#### 4.4 Liquidations Feed (`libs/hydra/data/liquidations.py`)
- [ ] Provider: Coinglass (free tier)
- [ ] Frequency: Last 4 hours rolling
- [ ] Alert: $20M+ liquidation = cascade fade opportunity

#### 4.5 72-Hour Historical Storage
- [ ] SQLite local storage (rolling window)
- [ ] Store: OHLCV, orderbook snapshots, funding, liquidations, regime
- [ ] Purpose: Pattern analysis for engines

#### 4.6 API Caching (`libs/hydra/data/cache.py`)
- [ ] Cache market analysis (5 min TTL)
- [ ] Cache regime classification (15 min TTL)
- [ ] Skip LLM on repeat setups
- [ ] Target: 60-70% cost reduction

---

### PHASE 5: WEEK 5 - Terminal Dashboard (6-8 hours)
**Goal**: Hardcore hedge fund grade monitoring in terminal

#### 5.1 Engine Rankings Section
```
ENGINE A: Rank #2/4 | WR: 67.3% | Gap to #1: 3.9%
ENGINE B: Rank #1/4 | WR: 71.2% | LEADER
ENGINE C: Rank #3/4 | WR: 64.1% | Gap to #1: 7.1%
ENGINE D: Rank #4/4 | WR: 58.9% | Gap to #1: 12.3%
```

#### 5.2 Exploration Stats
```
Strategies tested today: 847,293,102
Monte-Carlo pass rate: 0.003% (254 survived)
Walk-forward pass rate: 18% of MC survivors
Final viable edges today: 46
"No edge" decisions: 12/48 cycles
```

#### 5.3 Safety Status
```
Daily loss: 0.8% / 2% max [████░░░░░░] OK
Drawdown: 2.1% / 6% max [███░░░░░░░] OK
Desperate Mode: INACTIVE
Guardian violations today: 0
```

#### 5.4 Internet Search Activity
```
[Eng_A] "funding rate arbitrage binance december 2025"
[Eng_C] "liquidation cascade detection SUI APT"
[Eng_B] "order book imbalance strategy crypto"
```

#### 5.5 WR Performance Chart (plotext)
```
   72 ┤                              ╭─ Engine B
   70 ┤                           ╭──╯
   68 ┤                        ╭──╯ ╭── Engine A
   66 ┤                     ╭──╯╭───╯
   64 ┤                  ╭──╯╭──╯  ╭── Engine C
   62 ┤               ╭──╯╭──╯ ╭──╯
   60 ┤            ╭──╯╭──╯ ╭──╯ ╭── Engine D
   58 ┤         ╭──╯╭──╯ ╭──╯ ╭──╯
      1  2  3  4  5  6  7  8  9  10
```

---

## DATABASE SCHEMA UPDATES

### New Tables Required

#### `engine_history`
```sql
CREATE TABLE engine_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    engine VARCHAR(1) NOT NULL,  -- A, B, C, D
    rank INTEGER NOT NULL,        -- 1-4
    win_rate REAL NOT NULL,
    total_pnl REAL NOT NULL,
    sharpe_ratio REAL,
    strategies_tested INTEGER,
    trades_today INTEGER,
    weight REAL NOT NULL          -- 0.10-0.40
);
```

#### `edge_graveyard`
```sql
CREATE TABLE edge_graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_description TEXT NOT NULL,
    ban_date DATETIME NOT NULL,
    unban_date DATETIME NOT NULL,
    ban_reason TEXT,
    resurrection_date DATETIME,
    resurrection_success BOOLEAN
);
```

#### `lessons`
```sql
CREATE TABLE lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    winner_engine VARCHAR(1),
    loser_engine VARCHAR(1),
    lesson_type VARCHAR(20),  -- 'win_insight', 'loss_insight', 'breeding'
    lesson_content TEXT NOT NULL,
    applied BOOLEAN DEFAULT FALSE
);
```

#### `tournaments`
```sql
CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    event_type VARCHAR(20),  -- 'kill', 'breed'
    target_engine VARCHAR(1),
    winner_engine VARCHAR(1),
    details TEXT
);
```

---

## FILE STRUCTURE (As Per Blueprint)

```
libs/hydra/
├── mother_ai.py                    # Supreme orchestrator
├── engines/
│   ├── base_engine.py              # Abstract base
│   ├── engine_a_deepseek.py        # Structural hunter
│   ├── engine_b_claude.py          # Logic validator
│   ├── engine_c_grok.py            # Historical matcher
│   └── engine_d_gemini.py          # Synthesizer
├── tournament/
│   ├── tournament_manager.py       # Ranking system
│   ├── ranking.py
│   ├── kill_cycle.py               # 24hr kill
│   ├── breeding.py                 # 4-day breed
│   └── knowledge_transfer.py       # Winner teaches
├── validation/
│   ├── walk_forward.py             # 70/30 rolling
│   └── monte_carlo.py              # 10k sims <1sec
├── data/
│   ├── internet_search.py          # Serper API
│   ├── orderbook.py                # Coinbase/Bybit
│   ├── funding_rates.py            # Binance/Bybit
│   ├── liquidations.py             # Coinglass
│   └── cache.py                    # API cost reduction
├── safety/
│   ├── guardian.py                 # FTMO rules
│   ├── volatility_filter.py       # ATR check
│   └── edge_graveyard.py           # 60-day bans
├── lab/
│   ├── invention_lab.py            # Paper mode testing
│   └── promotion_gate.py           # 22/30 wins to live
└── engine_portfolio.py             # Independent P&L tracking

apps/
├── runtime/
│   └── hydra_3_runtime.py          # Main orchestrator
└── dashboard_terminal/
    └── hydra_dashboard.py          # Terminal UI

data/hydra/
└── hydra.db                        # SQLite database
```

---

## CRITICAL RULES

### ⚠️ NO HALLUCINATIONS ALLOWED

1. **Locked Emotion Prompt**
   - DO NOT change for 60 days
   - Exact wording from blueprint
   - Stats injection only ({rank}, {wr}, {gap}, {leader_name}, {leader_wr})

2. **Banned Indicators Forever**
   - RSI, MACD, Bollinger, Moving Averages
   - Any textbook pattern
   - Retail trader tools

3. **Structural Edges Only**
   - Liquidation cascades ($20M+)
   - Funding rate extremes (>|0.8%|)
   - Order-book imbalance (>2.5:1)
   - Exchange price gaps
   - Whale movements
   - Open interest extremes

4. **Guardian Non-Negotiable**
   - Daily loss 2% → stop 24hr
   - Total drawdown 6% → emergency shutdown
   - Risk per trade: 0.3-0.65% dynamic
   - Concurrent positions: 2 max

5. **Tournament Rules**
   - Rank by P&L ONLY (not votes)
   - Kill cycle: 24 hours
   - Breed cycle: 4 days
   - Winner teaches ALL losers

---

## TESTING CHECKPOINTS

### Week 1 Checkpoint
- [ ] 4 engines run in parallel
- [ ] Each engine has own P&L
- [ ] Mother AI orchestrates
- [ ] Guardian blocks unsafe trades
- [ ] Terminal dashboard shows rankings

### Week 2 Checkpoint
- [ ] Kill cycle executes (mock)
- [ ] Stats injection working
- [ ] Weight adjustment applied
- [ ] Knowledge transfer logs created

### Week 3 Checkpoint
- [ ] Walk-forward validation < 5 sec
- [ ] Monte-Carlo validation < 1 sec
- [ ] Strategy counter tracks millions
- [ ] Edge graveyard bans/unbans edges

### Week 4 Checkpoint
- [ ] Internet search finds edges
- [ ] Order-book feed live
- [ ] Funding rates tracked
- [ ] Liquidations monitored
- [ ] 72-hour historical storage working

### Week 5 Checkpoint
- [ ] Terminal dashboard fully functional
- [ ] All 5 sections displaying
- [ ] WR chart rendering
- [ ] Search log streaming
- [ ] Real-time updates

---

## SUCCESS CRITERIA

### Month 1 Targets
- Win Rate: 60-65%
- Sharpe Ratio: 1.5-2.0
- Edges in Library: 20-40
- Strategies/Day: 100K-1M
- Kill Events: ~7
- Breed Events: ~7

### FTMO Expectations
- Pass probability: 55-65%
- Pass time: 8-14 days
- Account survival: 6+ months

---

## NEXT STEPS

1. Start with Phase 1, Week 1 components
2. Build Mother AI orchestrator first
3. Refactor engines for independence
4. Install locked emotion prompt
5. Implement parallel execution
6. Add tournament ranking
7. Deploy Guardian

**Current Status**: Planning complete, ready to build
**Next Action**: Create Mother AI orchestrator (`libs/hydra/mother_ai.py`)

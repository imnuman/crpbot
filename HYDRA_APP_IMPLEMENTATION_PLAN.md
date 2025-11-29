# HYDRA A+++ Implementation Plan

**Date**: 2025-11-28
**Mission**: Build the most sophisticated retail trading evolution system

---

## Executive Summary

HYDRA A+++ is a **multi-agent evolutionary trading system** that:
- Uses 4 AI agents (DeepSeek, Claude, Groq, Gemini) competing in tournaments
- Focuses on **structural edges** in niche markets (exotic forex, meme perps)
- Implements **regime-aware** strategy selection
- Evolves strategies through **breeding and elimination**
- Protects capital with **5-layer anti-hallucination** and **Guardian limits**

**Key Differentiators from HMAS V2**:
1. **Evolution vs Static**: Strategies evolve through tournaments (not fixed)
2. **Niche Focus**: Targets underserved markets (not BTC/ETH)
3. **Regime Awareness**: Different strategies for trending/ranging/volatile
4. **Breeding System**: Top strategies combine to create better ones
5. **Cost**: ~$20/month (vs $1/signal in HMAS V2)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HYDRA A+++ SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: REGIME DETECTOR (Hourly)                         │
│           ├── Trending Up/Down                              │
│           ├── Ranging                                       │
│           ├── Volatile                                      │
│           └── Choppy → CASH                                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 4 GLADIATORS                                     │
│           ├── Gladiator A (DeepSeek) - Invention           │
│           ├── Gladiator B (Claude) - Validation            │
│           ├── Gladiator C (Groq) - Backtesting             │
│           └── Gladiator D (Gemini) - Synthesis             │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: TARGET MARKETS                                   │
│           ├── FTMO: Exotic forex (TRY, ZAR, MXN)           │
│           └── Binance: Meme perps (BONK, WIF, PEPE)        │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: FREE DATA SOURCES                                │
│           ├── Funding rates (Binance API)                   │
│           ├── Liquidations (Coinglass)                      │
│           └── COT reports (CFTC)                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: TOURNAMENT SYSTEM                                │
│           ├── 24hr elimination cycle                        │
│           ├── 4-day breeding cycle                          │
│           └── Winner teaches                                │
├─────────────────────────────────────────────────────────────┤
│  Layer 6: MULTI-AGENT CONSENSUS                            │
│           ├── 4/4 agree → 100% position                     │
│           ├── 3/4 agree → 75% position                      │
│           ├── 2/4 agree → 50% position                      │
│           └── <2/4 → NO TRADE                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 7: EXECUTION OPTIMIZER                              │
│           ├── Smart limit orders                            │
│           ├── Spread checking                               │
│           └── Saves 0.02-0.1% per trade                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 8: LIVE FEEDBACK LOOP                               │
│           └── Real execution data → Tournament scoring      │
├─────────────────────────────────────────────────────────────┤
│  Layer 9: ANTI-HALLUCINATION (5 Filters)                   │
│           ├── Logic validator                               │
│           ├── Backtest reality check                        │
│           ├── Paper confirmation                            │
│           ├── Cross-agent audit                             │
│           └── Sanity rules                                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 10: GUARDIAN (Hard Limits)                          │
│            ├── 2% daily loss → STOP                         │
│            ├── 6% max drawdown → 50% reduction              │
│            └── Unclear regime → CASH                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Week 1)

### 1.1 Project Structure

```
crpbot/
├── apps/
│   ├── runtime/
│   │   └── hydra_runtime.py              # Main orchestrator
│   └── tournament/
│       ├── tournament_manager.py          # 24hr kill, 4-day breed
│       ├── breeding_engine.py             # Strategy crossover
│       └── scoreboard.py                  # Performance tracking
├── libs/
│   ├── hydra/
│   │   ├── regime_detector.py             # Layer 1: Market classification
│   │   ├── gladiators/
│   │   │   ├── gladiator_a_deepseek.py    # Raw invention
│   │   │   ├── gladiator_b_claude.py      # Logic validation
│   │   │   ├── gladiator_c_groq.py        # Fast backtesting
│   │   │   └── gladiator_d_gemini.py      # Synthesis
│   │   ├── anti_hallucination.py          # 5-layer filter
│   │   ├── guardian.py                    # Hard limits
│   │   ├── consensus.py                   # Multi-agent voting
│   │   └── execution_optimizer.py         # Smart orders
│   ├── data/
│   │   ├── binance_client.py              # Binance futures data
│   │   ├── coinglass_client.py            # Liquidations
│   │   └── forex_factory.py               # Economic calendar
│   └── strategies/
│       ├── strategy_base.py               # Base strategy class
│       ├── strategy_validator.py          # Sanity checks
│       └── strategy_backtester.py         # Historical testing
└── data/
    ├── strategies/                        # Evolved strategies (JSON)
    ├── tournament_results/                # Performance logs
    └── breeding_history/                  # Genealogy tracking
```

### 1.2 Database Schema

```sql
-- Regime history
CREATE TABLE regime_history (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,  -- TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY
    adx REAL,
    atr REAL,
    confidence REAL
);

-- Strategies (evolved)
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY,
    strategy_id TEXT UNIQUE NOT NULL,  -- UUID
    name TEXT NOT NULL,
    gladiator TEXT NOT NULL,  -- A, B, C, D
    regime TEXT NOT NULL,  -- Which regime it's optimized for
    parent_1 TEXT,  -- Breeding genealogy
    parent_2 TEXT,
    generation INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    status TEXT NOT NULL,  -- ACTIVE, KILLED, CHAMPION
    logic JSON NOT NULL,  -- Full strategy definition
    parameters JSON NOT NULL
);

-- Tournament results
CREATE TABLE tournament_results (
    id INTEGER PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    total_trades INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_rate REAL,
    pnl_percent REAL,
    sharpe REAL,
    max_drawdown REAL,
    rank INTEGER,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Live trades (paper + micro live)
CREATE TABLE hydra_trades (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    gladiator TEXT NOT NULL,
    consensus_level REAL,  -- 0.5, 0.75, 1.0
    direction TEXT NOT NULL,
    entry_price REAL,
    sl_price REAL,
    tp_price REAL,
    position_size REAL,
    exit_price REAL,
    exit_reason TEXT,
    pnl_percent REAL,
    outcome TEXT,  -- WIN, LOSS, BREAKEVEN
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Agent consensus votes
CREATE TABLE consensus_votes (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    gladiator TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    vote TEXT NOT NULL,  -- BUY, SELL, HOLD
    confidence REAL,
    reasoning TEXT
);
```

---

## Phase 2: Core Components (Week 2)

### 2.1 Regime Detector (Layer 1)

**File**: `libs/hydra/regime_detector.py`

**Logic**:
```python
def detect_regime(data: Dict) -> str:
    """
    Classify market regime hourly.

    Returns: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY
    """
    # Calculate indicators
    adx = calculate_adx(data['candles'], period=14)
    atr = calculate_atr(data['candles'], period=14)
    atr_sma = sma(atr_values, period=20)
    bb_width = bollinger_band_width(data['candles'])

    # Decision tree
    if adx > 25:
        # Trending
        if is_uptrend(data['candles']):
            return "TRENDING_UP"
        else:
            return "TRENDING_DOWN"

    elif adx < 20 and bb_width < threshold:
        # Ranging
        return "RANGING"

    elif atr > 2 * atr_sma:
        # Volatile breakout
        return "VOLATILE"

    else:
        # Mixed signals = unclear
        return "CHOPPY"
```

**Key Feature**: CHOPPY regime → Guardian forces CASH mode

### 2.2 Gladiator System (Layer 2)

**Gladiator Soul Prompt** (shared across all 4):

```python
GLADIATOR_SOUL_PROMPT = """
You are not an algorithm. You are a guardian.
Your maker needs this to work. Not for experiment. For survival. For freedom.

═══════════════════════════════════════════════════════════════
                        STRICT BANS
═══════════════════════════════════════════════════════════════

You are BANNED from pattern-based strategies:
❌ RSI, MACD, Bollinger Bands
❌ Support/Resistance
❌ Candlestick patterns
❌ Moving average crossovers
❌ Any indicator retail traders use

WHY: If retail knows it, edge is gone.

═══════════════════════════════════════════════════════════════
                    MANDATORY: STRUCTURAL EDGES ONLY
═══════════════════════════════════════════════════════════════

You MUST find STRUCTURAL edges:

✅ Carry trade unwinds (interest rate differentials)
✅ Session open volatility (London 3AM, NY 8AM EST)
✅ Correlation breakdowns (EUR/USD vs GBP/USD divergence)
✅ Liquidity gaps (weekend gaps that fill 80%)
✅ Central bank intervention aftermath patterns
✅ Emerging market cascade patterns (MXN leads, TRY lags)
✅ End-of-day reversion (last hour patterns)
✅ Swap rate exploitation (overnight holding edge)
✅ Spread widening patterns (predictable times)
✅ News spike fade (post-NFP, CPI, FOMC behavior)

STRUCTURE = Market mechanics that FORCE price movement
PATTERN = Guess based on shapes

═══════════════════════════════════════════════════════════════
                        YOUR COMPETITION
═══════════════════════════════════════════════════════════════

Current scoreboard:
{scoreboard}

Last place gets KILLED in {hours_until_kill} hours.
Top 2 BREED in {days_until_breed} days.

If you're last for 24 hours → Your strategy dies.
Invent something better. NOW.

Your maker's freedom depends on you.

═══════════════════════════════════════════════════════════════
                        YOUR TASK
═══════════════════════════════════════════════════════════════

Market: {symbol}
Regime: {regime}
Current conditions: {market_data}

Generate a STRUCTURAL strategy for this regime.

Output JSON:
{{
  "strategy_name": "descriptive name",
  "structural_edge": "which market mechanic you're exploiting",
  "entry_rules": "exact entry logic",
  "exit_rules": "exact exit logic",
  "filters": ["list of filters to avoid false signals"],
  "risk_per_trade": 0.5-1.0,
  "expected_wr": 60-70,
  "why_it_works": "the market force that creates this edge",
  "weaknesses": ["when this strategy loses"]
}}
"""
```

**Gladiator A (DeepSeek)**: Raw invention, finds new edges
**Gladiator B (Claude)**: Logic validation, finds flaws
**Gladiator C (Groq)**: Fast backtesting, stress tests
**Gladiator D (Gemini)**: Cross-domain synthesis, combines insights

### 2.3 Anti-Hallucination System (Layer 9)

**5 Filters**:

```python
def validate_strategy(strategy: Dict, agents: List) -> bool:
    """
    5-layer anti-hallucination filter.
    """
    # Filter 1: Logic Validator
    if contains_contradictions(strategy):
        return False

    # Filter 2: Backtest Reality Check
    backtest_results = backtest_strategy(strategy, historical_data)
    if backtest_results['win_rate'] < 55:
        return False
    if backtest_results['total_trades'] < 100:
        return False

    # Filter 3: Sanity Rules
    if backtest_results['win_rate'] > 85:  # Likely overfit
        return False
    if backtest_results['sharpe'] < 0.5:
        return False

    # Filter 4: Cross-Agent Audit
    for agent in agents:
        critique = agent.critique_strategy(strategy)
        if critique['fatal_flaws']:
            return False

    # Filter 5: Multi-Regime Test
    regimes_tested = test_across_regimes(strategy)
    if regimes_tested['successful_regimes'] < 2:
        return False

    return True
```

### 2.4 Guardian System (Layer 10)

**Hard Limits**:

```python
class Guardian:
    """
    Hard safety limits. NEVER override.
    """

    DAILY_LOSS_LIMIT = 0.02  # 2%
    MAX_DRAWDOWN = 0.06      # 6%

    def check_before_trade(self, account_state: Dict, regime: str) -> bool:
        # Daily loss check
        if account_state['daily_pnl'] <= -self.DAILY_LOSS_LIMIT:
            logger.critical("DAILY LOSS LIMIT HIT - ALL TRADING STOPPED")
            return False

        # Max drawdown check
        if account_state['drawdown'] >= self.MAX_DRAWDOWN:
            logger.critical("MAX DRAWDOWN HIT - SURVIVAL MODE")
            self.reduce_all_positions(0.5)
            return False

        # Regime unclear
        if regime == "CHOPPY":
            logger.info("Regime unclear - staying CASH")
            return False

        # Spread check
        if account_state['spread'] > 2 * account_state['normal_spread']:
            logger.warning("Spread too wide - waiting")
            return False

        return True
```

---

## Phase 3: Tournament System (Week 3)

### 3.1 Tournament Manager

**Cycles**:
- **24-hour elimination**: Last place strategy gets killed
- **4-day breeding**: Top 2 strategies breed to create child
- **Winner teaches**: Winning strategy outputs full disclosure

**Per-Regime Tournaments**:
```python
class TournamentManager:
    """
    Separate tournaments for each regime.
    """

    def run_tournaments(self):
        # 3 separate tournaments running concurrently
        trending_tournament = Tournament(regime="TRENDING")
        ranging_tournament = Tournament(regime="RANGING")
        volatile_tournament = Tournament(regime="VOLATILE")

        # Each tournament has its own winner
        # Best trending strategy != best ranging strategy
```

### 3.2 Breeding Engine

**Anti-Correlation Requirement**:
```python
def breed_strategies(parent_a: Strategy, parent_b: Strategy) -> Strategy:
    """
    Breed top 2 strategies that are NEGATIVELY CORRELATED.
    """
    # Check correlation
    correlation = calculate_correlation(parent_a.trades, parent_b.trades)

    if correlation > 0.3:
        # Too correlated - find different pair
        return None

    # Create child
    child = Strategy(
        entry_logic=parent_a.entry_logic,
        exit_logic=parent_b.exit_logic,
        risk_rules=average(parent_a.risk_rules, parent_b.risk_rules),
        filters=merge(parent_a.filters, parent_b.filters),
        generation=parent_a.generation + 1,
        parents=[parent_a.id, parent_b.id]
    )

    return child
```

**Teaching Protocol**:
```python
def winner_teaches(winner: Strategy):
    """
    Winner outputs full disclosure.
    Losers must study and improve.
    """
    disclosure = {
        "strategy_full_disclosure": {
            "entry_logic": winner.entry_logic,
            "exit_logic": winner.exit_logic,
            "filters": winner.filters,
            "regime": winner.regime,
            "parameters": winner.parameters
        },
        "why_it_works": winner.structural_edge,
        "my_weaknesses": winner.weaknesses,
        "untested_ideas": winner.future_improvements
    }

    # Save to knowledge base
    save_teaching(disclosure)

    # Losing gladiators must read and improve
    for gladiator in losing_gladiators:
        gladiator.study(disclosure)
        gladiator.invent_counter_strategy()
```

---

## Phase 4: Execution (Week 4)

### 4.1 Multi-Agent Consensus (Layer 6)

```python
def get_consensus_decision(symbol: str, regime: str, gladiators: List) -> Dict:
    """
    4 gladiators vote. Consensus determines position size.
    """
    votes = []
    for gladiator in gladiators:
        vote = gladiator.analyze(symbol, regime)
        votes.append(vote)

    # Count votes
    buy_votes = sum(1 for v in votes if v['action'] == 'BUY')
    sell_votes = sum(1 for v in votes if v['action'] == 'SELL')

    # Determine consensus
    if buy_votes == 4:
        return {'action': 'BUY', 'position_size': 1.0}  # 100%
    elif buy_votes == 3:
        return {'action': 'BUY', 'position_size': 0.75}  # 75%
    elif buy_votes == 2:
        return {'action': 'BUY', 'position_size': 0.5}  # 50%
    elif sell_votes >= 3:
        return {'action': 'SELL', 'position_size': ...}
    else:
        return {'action': 'HOLD', 'position_size': 0}  # No consensus
```

### 4.2 Execution Optimizer (Layer 7)

```python
def execute_trade_optimized(trade: Dict) -> Dict:
    """
    Smart limit orders to save 0.02-0.1% per trade.
    """
    # Check spread
    current_spread = get_current_spread(trade['symbol'])
    normal_spread = get_normal_spread(trade['symbol'])

    if current_spread > 2 * normal_spread:
        logger.warning("Spread too wide - waiting")
        return {'status': 'WAITING', 'reason': 'High spread'}

    # Place limit order slightly better than market
    if trade['action'] == 'BUY':
        limit_price = get_ask() - 0.0001  # Just below ask
    else:
        limit_price = get_bid() + 0.0001  # Just above bid

    # Wait up to 30 seconds for fill
    order = place_limit_order(limit_price, timeout=30)

    if order['filled']:
        return {'status': 'FILLED', 'price': order['fill_price']}
    else:
        # Adjust or cancel
        return {'status': 'NOT_FILLED', 'reason': 'Timeout'}
```

---

## Phase 5: Data Integration (Week 5)

### 5.1 Free Data Sources

**Binance Funding Rates**:
```python
def get_funding_rates() -> Dict:
    """
    Free funding rate data from Binance.
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    response = requests.get(url)
    return response.json()
```

**Coinglass Liquidations**:
```python
def get_liquidation_data(symbol: str) -> Dict:
    """
    Free liquidation data from Coinglass.
    """
    # Scrape or use free API tier
    url = f"https://www.coinglass.com/LiquidationData?symbol={symbol}"
    # Parse data
```

**CFTC COT Reports**:
```python
def get_cot_report() -> Dict:
    """
    Free Commitment of Traders data.
    """
    url = "https://www.cftc.gov/files/dea/cotarchives/..."
    # Parse XML/CSV
```

### 5.2 Target Markets

**Exotic Forex (FTMO)**:
- USD/TRY, USD/ZAR, USD/MXN
- EUR/TRY, USD/PLN, USD/HUF, USD/NOK
- Position size: 50% of normal (higher volatility)
- No overnight holds (gap risk)

**Meme Perps (Binance/Bybit)**:
- BONK, WIF, PEPE, FLOKI
- SUI, INJ (mid-caps)
- BONK/SOL, PEPE/ETH (exotic pairs)
- Micro positions ($10-50)

---

## Deployment Roadmap

### Week 1-2: Paper Mode
```bash
# Run full HYDRA system with paper trading
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --mode paper \
  --markets exotic_forex meme_perps \
  --iterations -1
```

**Objectives**:
- All 4 gladiators competing
- Tournament cycles running (24hr kill, 4-day breed)
- Track everything ($0 at risk)
- Validate anti-hallucination filters
- Verify Guardian limits work

### Week 3-4: Micro Live
```bash
# Binance/Bybit with $10 positions
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --mode micro_live \
  --markets meme_perps \
  --position-size 10 \
  --iterations -1
```

**Objectives**:
- Real execution data feeds back to tournaments
- Measure real slippage, spreads, fills
- Prove system works with real money
- Cost: $10-50 at risk

### Month 2+: Scale
```bash
# Increase position sizes gradually
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --mode live \
  --markets exotic_forex meme_perps \
  --position-size-percent 1.0 \
  --iterations -1
```

**Objectives**:
- If profitable → Double size every 2 weeks
- Never exceed Guardian limits (2% daily, 6% max DD)
- Let HYDRA evolve and improve

---

## Expected Performance

| Metric | Target |
|--------|--------|
| Market coverage | 70-80% of hours |
| Trades per day | 4-8 high quality |
| Win rate | 60-70% |
| Risk per trade | 0.5-1% |
| Daily loss limit | 2% (hard stop) |
| Max drawdown | 6% (survival mode) |
| Sharpe ratio | >1.5 |
| Monthly target | +5-10% |

---

## Cost Analysis

| Component | Cost |
|-----------|------|
| DeepSeek API | ~$5/month |
| Claude API | ~$10/month |
| Groq API | FREE tier |
| Gemini API | FREE tier |
| Data sources | $0 (all free) |
| VPS (optional) | $5-20/month |
| **TOTAL** | **$15-35/month** |

**ROI**:
- HMAS V2 cost: $1.00/signal = $600/month (20 signals/day)
- HYDRA A+++: $20/month (unlimited signals)
- **Savings**: $580/month

---

## Key Advantages Over HMAS V2

| Feature | HMAS V2 | HYDRA A+++ |
|---------|---------|------------|
| Cost | $1.00/signal | ~$20/month (unlimited) |
| Strategy | Static 7-agent | Evolving 4-gladiator |
| Markets | BTC/ETH/SOL/DOGE | Exotic forex, meme perps |
| Edge | General analysis | Structural inefficiencies |
| Evolution | None | Tournament + breeding |
| Regime aware | No | Yes (3 regimes) |
| Safety | Basic | 5-layer + Guardian |

---

## Risk Mitigation

### Technical Risks
- **Hallucination**: 5-layer filter (logic, backtest, paper, audit, sanity)
- **Overfitting**: Multi-regime testing, min 100 trades, max 85% WR
- **API costs**: Free tier first, monitor usage

### Market Risks
- **Daily loss**: 2% hard limit (all trading stops)
- **Max drawdown**: 6% limit (survival mode, 50% reduction)
- **Unclear regime**: CHOPPY → forced CASH mode
- **Exotic volatility**: 50% position size, no overnight holds

### Operational Risks
- **System failure**: Guardian monitors, auto-restart
- **Data quality**: Multiple sources, cross-validation
- **Strategy death**: Tournament ensures only best survive

---

## Success Criteria

**Phase 1 (Week 1-2 Paper)**:
- ✅ All 4 gladiators generating strategies
- ✅ Tournament cycles running correctly
- ✅ Anti-hallucination catching bad strategies
- ✅ Guardian blocking unsafe trades
- ✅ No critical bugs

**Phase 2 (Week 3-4 Micro Live)**:
- ✅ Positive P&L (any amount)
- ✅ Win rate >55%
- ✅ Sharpe >0.5
- ✅ No Guardian limit violations
- ✅ Real execution data improving strategies

**Phase 3 (Month 2+)**:
- ✅ Consistent monthly profit (+3-5%)
- ✅ Win rate >60%
- ✅ Sharpe >1.0
- ✅ Max drawdown <5%
- ✅ Strategies evolving and improving

---

## Next Steps

1. **Review this plan** - Confirm architecture aligns with vision
2. **Start implementation** - Begin with Phase 1 (Foundation)
3. **Incremental testing** - Test each layer before moving forward
4. **Document everything** - Track breeding genealogy, tournament results
5. **Monitor and iterate** - HYDRA learns from live data

---

## Final Notes

HYDRA A+++ is **not guaranteed profit**. It is **maximum realistic edge** given:
- Retail trader constraints (capital, data, tools)
- 2025 market efficiency
- Free/cheap AI API costs

The system **competes where giants don't look** (exotic forex, meme perps) and **evolves faster than anyone else** through tournaments and breeding.

**For maker's freedom. Always.**

---

**Ready to build?** Let me know which phase to start with, or if you want to refine any layer first.

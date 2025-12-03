# HYDRA 3.0 - ARCHITECTURE REBUILD PLAN
**Created**: 2025-12-01
**Status**: 🔴 CRITICAL - Core competition system requires complete rebuild

---

## 🚨 PROBLEM STATEMENT

**Current Implementation** (WRONG):
```
Pipeline Architecture:
A generates strategy →
B validates A's strategy →
C backtests B's validated strategy →
D synthesizes all into final strategy →
4 Gladiators vote on D's strategy →
1 Trade executed

Result: All 4 gladiators work on SAME strategy
        No independent competition
        No individual P&L tracking
        Tournament ranks strategies, not gladiators
```

**Required Implementation** (PER BLUEPRINT):
```
Independent Trader Architecture:
A generates → A backtests → A trades → Track A's P&L
B generates → B backtests → B trades → Track B's P&L
C generates → C backtests → C trades → Track C's P&L
D generates → D backtests → D trades → Track D's P&L

Result: 4 independent traders competing
        Each has own P&L
        Tournament ranks GLADIATORS by performance
        Winner teaches losers
        Worst performer must evolve or die
```

---

##  🔴 CRITICAL FIX #1: GLADIATOR ARCHITECTURE

### Current State (BROKEN)
- Gladiator A: "Structural Edge Generator" only
- Gladiator B: "Logic Validator" (validates A's work)
- Gladiator C: "Fast Backtester" (backtests A+B's work)
- Gladiator D: "Synthesizer" (combines A+B+C into final strategy)

###Required State
Each gladiator must be a COMPLETE INDEPENDENT TRADER:

```python
class BaseGladiator:
    """Each gladiator is a complete trading system"""

    def generate_strategy(self, asset, regime, market_data) -> Dict:
        """Generate OWN unique strategy"""
        pass

    def backtest_strategy(self, strategy, historical_data) -> Dict:
        """Backtest OWN strategy"""
        pass

    def decide_trade(self, asset, market_data) -> Dict:
        """Make OWN trade decision (BUY/SELL/HOLD)"""
        pass

    def execute_trade(self, decision) -> str:
        """Execute trade in own paper portfolio"""
        pass

    def get_performance(self) -> Dict:
        """Return own P&L, win rate, Sharpe ratio"""
        return {
            "total_pnl": self.portfolio.total_pnl,
            "win_rate": self.portfolio.win_rate,
            "sharpe_ratio": self.portfolio.sharpe_ratio,
            "trades_count": len(self.portfolio.trades)
        }

    def learn_from_winner(self, winner_insights) -> None:
        """Receive and incorporate winner's strategies"""
        self.learned_edges.append(winner_insights)
```

### Files to Modify

**1. `/root/crpbot/libs/hydra/gladiators/base_gladiator.py`**
- Add: `backtest_strategy()` method
- Add: `decide_trade()` method
- Add: `execute_trade()` method
- Add: `get_performance()` method
- Add: `learn_from_winner()` method
- Add: `portfolio` attribute (GladiatorPortfolio instance)

**2. `/root/crpbot/libs/hydra/gladiators/gladiator_a_deepseek.py`**
- Keep: `generate_strategy()` (structural edge focus)
- Add: Own backtesting logic
- Add: Own trade decision logic
- Remove: Nothing (it's already a generator)

**3. `/root/crpbot/libs/hydra/gladiators/gladiator_b_claude.py`**
- Change: From "validator" to "logic-based trader"
- Add: `generate_strategy()` - focus on logical consistency
- Add: Own backtesting logic
- Add: Own trade decision logic
- Keep: Critical thinking / skepticism in strategy generation

**4. `/root/crpbot/libs/hydra/gladiators/gladiator_c_grok.py`**
- Change: From "backtester" to "pattern-based trader"
- Add: `generate_strategy()` - focus on historical patterns
- Keep: Pattern recognition strength
- Add: Own trade decision logic

**5. `/root/crpbot/libs/hydra/gladiators/gladiator_d_gemini.py`**
- Change: From "synthesizer" to "holistic trader"
- Add: `generate_strategy()` - focus on multi-factor synthesis
- Add: Own backtesting logic
- Add: Own trade decision logic

**6. `/root/crpbot/apps/runtime/hydra_runtime.py`**
- **COMPLETE REWRITE** of `_generate_signal()` method:

```python
def _generate_signal(self, asset: str) -> Optional[Dict]:
    """
    NEW FLOW: Each gladiator trades independently
    """
    # Get market data
    market_data = self._get_market_data(asset)
    regime = self.regime_detector.detect_regime(asset, market_data)

    # Each gladiator generates their own strategy
    strategy_a = self.gladiator_a.generate_strategy(asset, "crypto", self.profiles[asset], regime, 0.8, market_data)
    strategy_b = self.gladiator_b.generate_strategy(asset, "crypto", self.profiles[asset], regime, 0.8, market_data)
    strategy_c = self.gladiator_c.generate_strategy(asset, "crypto", self.profiles[asset], regime, 0.8, market_data)
    strategy_d = self.gladiator_d.generate_strategy(asset, "crypto", self.profiles[asset], regime, 0.8, market_data)

    # Each gladiator backtests their own strategy
    backtest_a = self.gladiator_a.backtest_strategy(strategy_a, market_data)
    backtest_b = self.gladiator_b.backtest_strategy(strategy_b, market_data)
    backtest_c = self.gladiator_c.backtest_strategy(strategy_c, market_data)
    backtest_d = self.gladiator_d.backtest_strategy(strategy_d, market_data)

    # Each gladiator decides their own trade
    decision_a = self.gladiator_a.decide_trade(asset, "crypto", regime, strategy_a, signal, market_data)
    decision_b = self.gladiator_b.decide_trade(asset, "crypto", regime, strategy_b, signal, market_data)
    decision_c = self.gladiator_c.decide_trade(asset, "crypto", regime, strategy_c, signal, market_data)
    decision_d = self.gladiator_d.decide_trade(asset, "crypto", regime, strategy_d, signal, market_data)

    # Execute all 4 trades independently (in paper trading)
    self.gladiator_a.execute_trade(decision_a)
    self.gladiator_b.execute_trade(decision_b)
    self.gladiator_c.execute_trade(decision_c)
    self.gladiator_d.execute_trade(decision_d)

    # Track each gladiator's performance
    self.tournament_tracker.record_trade("A", decision_a, strategy_a)
    self.tournament_tracker.record_trade("B", decision_b, strategy_b)
    self.tournament_tracker.record_trade("C", decision_c, strategy_c)
    self.tournament_tracker.record_trade("D", decision_d, strategy_d)

    # OPTIONAL: Also generate consensus trade (secondary)
    if self.consensus_enabled:
        consensus = self.consensus.get_consensus([decision_a, decision_b, decision_c, decision_d])
        return consensus

    return None  # No single "signal" anymore - 4 independent trades
```

**Effort**: 8-12 hours

---

## 🔴 CRITICAL FIX #2: INDEPENDENT P&L TRACKING

### Current State (BROKEN)
- `paper_trades.jsonl` tracks all trades together
- No separation by gladiator
- Win rate/P&L is combined, not per-gladiator

### Required State
Each gladiator has separate portfolio:

**New File**: `/root/crpbot/libs/hydra/gladiator_portfolio.py`

```python
class GladiatorPortfolio:
    """Independent portfolio for each gladiator"""

    def __init__(self, gladiator_name: str, starting_capital: float = 10000):
        self.gladiator_name = gladiator_name
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.trades: List[Dict] = []
        self.open_positions: List[Dict] = []

    def execute_trade(self, trade: Dict) -> str:
        """Execute trade in this gladiator's portfolio"""
        trade_id = f"{self.gladiator_name}_{trade['asset']}_{int(time.time())}"

        position = {
            "trade_id": trade_id,
            "gladiator": self.gladiator_name,
            "asset": trade["asset"],
            "direction": trade["direction"],
            "entry_price": trade["entry_price"],
            "size_usd": self._calculate_position_size(trade),
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "status": "OPEN",
            "entry_timestamp": datetime.utcnow().isoformat()
        }

        self.open_positions.append(position)
        self.trades.append(position)
        return trade_id

    def update_positions(self, current_prices: Dict) -> None:
        """Check for SL/TP hits, close positions"""
        for position in self.open_positions[:]:
            current_price = current_prices.get(position["asset"])
            if not current_price:
                continue

            # Check SL/TP
            if self._check_exit(position, current_price):
                self._close_position(position, current_price)

    def get_performance(self) -> Dict:
        """Calculate this gladiator's performance"""
        closed_trades = [t for t in self.trades if t.get("status") == "CLOSED"]

        if not closed_trades:
            return {"total_pnl": 0, "win_rate": 0, "sharpe_ratio": 0, "trades_count": 0}

        wins = [t for t in closed_trades if t.get("outcome") == "win"]
        total_pnl = sum(t.get("pnl_usd", 0) for t in closed_trades)
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0

        # Calculate Sharpe ratio
        returns = [t.get("pnl_percent", 0) for t in closed_trades]
        sharpe_ratio = self._calculate_sharpe(returns)

        return {
            "gladiator": self.gladiator_name,
            "total_pnl": total_pnl,
            "total_pnl_percent": (total_pnl / self.starting_capital) * 100,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "trades_count": len(closed_trades),
            "open_positions": len(self.open_positions)
        }

    def save_to_disk(self, filepath: str) -> None:
        """Save this gladiator's trades to disk"""
        with open(filepath, 'w') as f:
            for trade in self.trades:
                f.write(json.dumps(trade) + "\n")
```

### Files to Modify

**1. Modify `/root/crpbot/libs/hydra/paper_trader.py`**
- Remove: Combined portfolio logic
- Add: Support for multiple portfolios (one per gladiator)
- Change: File storage to separate by gladiator

```python
# New structure:
data/hydra/portfolios/
  ├── gladiator_a_trades.jsonl
  ├── gladiator_b_trades.jsonl
  ├── gladiator_c_trades.jsonl
  └── gladiator_d_trades.jsonl
```

**Effort**: 4-6 hours

---

## 🔴 CRITICAL FIX #3: TOURNAMENT RANKING BY P&L

### Current State (BROKEN)
- Ranks strategies (all from same pipeline)
- Uses vote accuracy (meaningless when all vote on same thing)

### Required State
Rank gladiators by their independent trading performance:

**Modify**: `/root/crpbot/libs/hydra/tournament_manager.py`

```python
def rank_gladiators(self) -> List[Dict]:
    """Rank gladiators by independent performance"""

    # Get each gladiator's performance
    performances = [
        self.gladiator_a.get_performance(),
        self.gladiator_b.get_performance(),
        self.gladiator_c.get_performance(),
        self.gladiator_d.get_performance()
    ]

    # Rank by composite score: Sharpe (60%) + Win Rate (40%)
    for perf in performances:
        perf["composite_score"] = (
            perf["sharpe_ratio"] * 0.6 +
            perf["win_rate"] * 0.4
        )

    # Sort by composite score
    ranked = sorted(performances, key=lambda x: x["composite_score"], reverse=True)

    logger.info(f"Tournament Rankings:")
    for i, perf in enumerate(ranked, 1):
        logger.info(
            f"  #{i}: Gladiator {perf['gladiator']} - "
            f"Sharpe: {perf['sharpe_ratio']:.2f}, "
            f"WR: {perf['win_rate']:.1%}, "
            f"P&L: {perf['total_pnl_percent']:.2f}%"
        )

    return ranked
```

**Effort**: 2-3 hours

---

## 🔴 CRITICAL FIX #4: WINNER TEACHES LOSERS

### Current State (BROKEN)
- Not implemented

### Required State
After each tournament cycle, winner shares insights with losers:

**Add to `/root/crpbot/libs/hydra/gladiators/base_gladiator.py`**:

```python
def get_winning_strategies(self) -> List[Dict]:
    """Extract insights from this gladiator's best strategies"""
    # Get top 3 winning trades
    wins = [t for t in self.portfolio.trades if t.get("outcome") == "win"]
    top_wins = sorted(wins, key=lambda x: x.get("pnl_percent", 0), reverse=True)[:3]

    insights = []
    for trade in top_wins:
        insights.append({
            "structural_edge": trade.get("strategy", {}).get("structural_edge"),
            "entry_rules": trade.get("strategy", {}).get("entry_rules"),
            "filters": trade.get("strategy", {}).get("filters"),
            "regime": trade.get("regime"),
            "why_it_worked": f"Won {trade.get('pnl_percent', 0):.1%} in {trade.get('asset')}"
        })

    return insights

def learn_from_winner(self, winner_insights: List[Dict]) -> None:
    """Incorporate winner's insights into own strategy generation"""
    logger.info(f"Gladiator {self.name} learning from winner...")

    # Add to learned edges
    for insight in winner_insights:
        self.learned_edges.append({
            "source": "tournament_winner",
            "timestamp": datetime.utcnow().isoformat(),
            "edge": insight["structural_edge"],
            "entry_rules": insight["entry_rules"],
            "filters": insight["filters"],
            "regime": insight["regime"]
        })

    logger.success(f"Gladiator {self.name} learned {len(winner_insights)} new edges")
```

**Add to `/root/crpbot/libs/hydra/tournament_manager.py`**:

```python
def run_teaching_cycle(self) -> None:
    """Winner teaches losers after tournament"""
    rankings = self.rank_gladiators()

    winner = rankings[0]
    losers = rankings[1:]

    logger.info(f"🏆 Gladiator {winner['gladiator']} won tournament!")
    logger.info(f"📚 Teaching losers...")

    # Get winner's insights
    winner_gladiator = self._get_gladiator(winner['gladiator'])
    winner_insights = winner_gladiator.get_winning_strategies()

    # Teach each loser
    for loser_perf in losers:
        loser_gladiator = self._get_gladiator(loser_perf['gladiator'])
        loser_gladiator.learn_from_winner(winner_insights)

        logger.info(
            f"  Gladiator {loser_perf['gladiator']} received {len(winner_insights)} insights"
        )
```

**Effort**: 3-4 hours

---

## 🔴 CRITICAL FIX #5: 24-HOUR ELIMINATION CYCLE

### Current State (BROKEN)
- Kills strategies (but they're all from same pipeline)
- Doesn't force gladiator to evolve

### Required State
Every 24 hours, worst gladiator must kill their worst strategy and invent new one:

**Modify**: `/root/crpbot/libs/hydra/tournament_manager.py`

```python
def run_elimination_cycle(self) -> None:
    """24-hour elimination: worst gladiator reinvents"""
    rankings = self.rank_gladiators()
    loser = rankings[-1]
    winner = rankings[0]

    logger.warning(f"⚔️  24-HOUR ELIMINATION")
    logger.warning(f"  Loser: Gladiator {loser['gladiator']} (Sharpe: {loser['sharpe_ratio']:.2f})")

    # Get loser gladiator
    loser_gladiator = self._get_gladiator(loser['gladiator'])

    # Kill their worst strategy
    worst_strategy = loser_gladiator.get_worst_strategy()
    loser_gladiator.kill_strategy(worst_strategy["strategy_id"])

    logger.info(f"  ❌ Killed strategy: {worst_strategy.get('strategy_name')}")

    # Force them to learn from winner
    winner_gladiator = self._get_gladiator(winner['gladiator'])
    winner_insights = winner_gladiator.get_winning_strategies()
    loser_gladiator.learn_from_winner(winner_insights)

    # Force them to generate NEW strategy (must be different)
    new_strategy = loser_gladiator.generate_new_strategy_after_death()

    logger.success(f"  ✨ New strategy created: {new_strategy.get('strategy_name')}")
    logger.info(f"  📈 Loser must now outperform winner or die again in 24hrs")
```

**Effort**: 2-3 hours

---

## 🔴 CRITICAL FIX #6: 4-DAY BREEDING CYCLE

### Current State (BROKEN)
- Breeding tries to combine strategies from same pipeline
- Meaningless genetic mixing

### Required State
Every 4 days, top 2 gladiators' best strategies breed:

**Modify**: `/root/crpbot/libs/hydra/breeding_engine.py`

```python
def breed_top_strategies(self) -> Dict:
    """Combine top 2 gladiators' strategies into offspring"""
    rankings = self.tournament_manager.rank_gladiators()

    parent1 = rankings[0]  # Best gladiator
    parent2 = rankings[1]  # Second best

    logger.info(f"🧬 BREEDING CYCLE")
    logger.info(f"  Parent 1: Gladiator {parent1['gladiator']} (Sharpe: {parent1['sharpe_ratio']:.2f})")
    logger.info(f"  Parent 2: Gladiator {parent2['gladiator']} (Sharpe: {parent2['sharpe_ratio']:.2f})")

    # Get best strategies from each parent
    p1_gladiator = self.tournament_manager._get_gladiator(parent1['gladiator'])
    p2_gladiator = self.tournament_manager._get_gladiator(parent2['gladiator'])

    p1_best = p1_gladiator.get_best_strategy()
    p2_best = p2_gladiator.get_best_strategy()

    # Create offspring by combining best elements
    offspring = {
        "strategy_id": f"BRED_{parent1['gladiator']}_{parent2['gladiator']}_{int(time.time())}",
        "strategy_name": f"Hybrid: {p1_best['strategy_name']} x {p2_best['strategy_name']}",
        "structural_edge": p1_best["structural_edge"],  # From best parent
        "entry_rules": p1_best["entry_rules"],          # From best parent
        "exit_rules": p2_best["exit_rules"],            # From second parent
        "filters": list(set(p1_best["filters"] + p2_best["filters"])),  # Combined
        "risk_per_trade": min(p1_best["risk_per_trade"], p2_best["risk_per_trade"]),  # Conservative
        "parents": [parent1['gladiator'], parent2['gladiator']],
        "bred_timestamp": datetime.utcnow().isoformat()
    }

    logger.success(f"  ✨ Offspring created: {offspring['strategy_name']}")

    # Give offspring to worst gladiator to test
    loser = rankings[-1]
    loser_gladiator = self.tournament_manager._get_gladiator(loser['gladiator'])
    loser_gladiator.receive_bred_strategy(offspring)

    logger.info(f"  📦 Offspring given to Gladiator {loser['gladiator']} to test")

    return offspring
```

**Effort**: 3-4 hours

---

## 🟡 MEDIUM FIX #7: DATA FEEDS (Currently Stubbed)

### Issues
1. `_get_dxy_data()` returns None
2. `news_events` always empty
3. `session` detection returns "Unknown"

### Fixes

**1. DXY Data** (USD Index):
```python
def _get_dxy_data(self) -> Optional[Dict]:
    """Get USD Index (DXY) data from TradingView or Alpha Vantage"""
    try:
        # Option 1: Alpha Vantage (free tier)
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=EUR&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()

        # Calculate DXY proxy from USD/EUR
        usd_eur_rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        dxy_proxy = 100 / usd_eur_rate  # Simplified DXY calculation

        return {"dxy": dxy_proxy, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"DXY data fetch failed: {e}")
        return None
```

**2. News Events** (Economic Calendar):
```python
def _get_news_events(self, asset: str) -> List[Dict]:
    """Get upcoming economic news from Forex Factory or Trading Economics"""
    try:
        # Use ForexFactory RSS or Trading Economics API
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        response = requests.get(url, timeout=10)
        events = response.json()

        # Filter high-impact events in next 24 hours
        upcoming = []
        now = datetime.utcnow()
        for event in events:
            event_time = datetime.fromisoformat(event["date"])
            if event_time > now and event_time < now + timedelta(hours=24):
                if event["impact"] == "High":
                    upcoming.append({
                        "title": event["title"],
                        "time": event_time.isoformat(),
                        "impact": event["impact"],
                        "currency": event["currency"]
                    })

        return upcoming
    except Exception as e:
        logger.error(f"News events fetch failed: {e}")
        return []
```

**3. Session Detection**:
```python
def _get_session(self) -> str:
    """Detect trading session based on UTC hour"""
    now_utc = datetime.utcnow()
    hour = now_utc.hour

    # Session times (UTC)
    if 0 <= hour < 7:
        return "ASIA"  # Tokyo: 00:00-09:00 UTC
    elif 7 <= hour < 12:
        return "LONDON"  # London: 08:00-16:00 UTC
    elif 12 <= hour < 21:
        return "NEW_YORK"  # NY: 13:00-22:00 UTC
    else:
        return "ASIA"  # Late NY / Early Asia overlap
```

**Effort**: 4-6 hours

---

## 🔴 CRITICAL FIX #7: MOTHER AI (L1 Supervisor)

### Current State (MISSING ENTIRELY)
- `hydra_runtime.py` directly orchestrates gladiators
- No meta-learning layer
- No emotional pressure mechanism
- No final approval gate before trades

### Required State (PER BLUEPRINT)

**Mother AI** sits ABOVE all 4 gladiators as supervisor/risk governance layer:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MOTHER AI (L1)                             │
│              Supervisor / Risk Governance                        │
│                    Model: Gemini Pro or Claude Opus             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FUNCTIONS:                                                     │
│  1. Orchestration - Manages all 4 gladiators                   │
│  2. Final Approval - Last check before any trade               │
│  3. Meta-Learning - Learns HOW gladiators solve problems       │
│  4. Goal Decomposition - Breaks targets into tasks             │
│  5. Delegation - Assigns tasks to right gladiator              │
│  6. Risk Governance - Non-negotiable FTMO rules                │
│  7. Emotional Pressure - If WR < 70% after 2 cycles,           │
│     triggers "DESPERATE MODE" forcing exploration              │
│                                                                 │
│                         ↓                                       │
│    ┌─────────┬─────────┬─────────┬─────────┐                  │
│    │ Glad A  │ Glad B  │ Glad C  │ Glad D  │                  │
│    │DeepSeek │ Claude  │  Groq   │ Gemini  │                  │
│    └─────────┴─────────┴─────────┴─────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

**New File**: `/root/crpbot/libs/hydra/mother_ai.py`

```python
class MotherAI:
    """
    L1 Supervisor - Sits above all gladiators

    Role: Meta-learning, orchestration, final approval, emotional pressure
    Model: Gemini Pro (cheap, fast) or Claude Opus (expensive, brilliant)
    """

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.model = model  # "gemini-pro" or "claude-opus"
        self.api_key = api_key
        self.tournament_cycles = 0
        self.desperate_mode = False
        self.meta_learnings: List[Dict] = []

    def orchestrate_cycle(self, assets: List[str]) -> Dict:
        """
        Main orchestration loop - runs all 4 gladiators
        """
        logger.info(f"🧠 Mother AI orchestrating cycle {self.tournament_cycles + 1}")

        results = {
            "cycle_number": self.tournament_cycles,
            "trades_approved": [],
            "trades_rejected": [],
            "gladiator_performance": {},
            "meta_insights": []
        }

        for asset in assets:
            # Delegate to all 4 gladiators
            decisions = self._delegate_to_gladiators(asset)

            # Final approval for each decision
            for gladiator_name, decision in decisions.items():
                approved = self._final_approval(decision)

                if approved:
                    results["trades_approved"].append(decision)
                else:
                    results["trades_rejected"].append(decision)

        # Meta-learning: Learn from this cycle
        self._meta_learn(results)

        # Check if we need to trigger desperate mode
        self._check_emotional_pressure()

        self.tournament_cycles += 1
        return results

    def _final_approval(self, decision: Dict) -> bool:
        """
        Final approval gate before any trade executes

        Mother AI reviews:
        - FTMO rules compliance
        - Risk/reward ratio
        - Correlation with open positions
        - Overall portfolio heat
        - Market conditions (news, volatility)
        """
        prompt = f"""You are Mother AI, the final approval layer.

Review this trade decision from Gladiator {decision['gladiator']}:

TRADE:
- Asset: {decision['asset']}
- Direction: {decision['direction']}
- Entry: {decision['entry_price']}
- Stop Loss: {decision['stop_loss']}
- Take Profit: {decision['take_profit']}
- Risk: {decision['risk_percent']}%
- Confidence: {decision['confidence']:.1%}

REASONING:
{decision['reasoning']}

CURRENT PORTFOLIO:
- Open positions: {self.get_open_positions_count()}
- Total heat: {self.get_portfolio_heat()}%
- Daily P&L: {self.get_daily_pnl()}%

FTMO RULES:
- Daily loss limit: 4.5% (remaining: {self.get_daily_loss_remaining()}%)
- Total loss limit: 9% (remaining: {self.get_total_loss_remaining()}%)

Should this trade be APPROVED or REJECTED?

Consider:
1. FTMO rule compliance
2. Risk/reward quality
3. Portfolio correlation
4. Market conditions
5. Gladiator's track record

Output JSON:
{{
  "decision": "APPROVE|REJECT",
  "reasoning": "Why approved/rejected",
  "concerns": ["Any red flags"],
  "confidence": 0.85
}}"""

        response = self._call_llm(prompt)
        result = json.loads(response)

        if result["decision"] == "APPROVE":
            logger.success(f"✅ Mother AI APPROVED {decision['asset']} {decision['direction']}")
            return True
        else:
            logger.warning(f"❌ Mother AI REJECTED {decision['asset']} {decision['direction']}")
            logger.warning(f"   Reason: {result['reasoning']}")
            return False

    def _meta_learn(self, cycle_results: Dict) -> None:
        """
        Meta-learning: Learn HOW gladiators solve problems

        Mother AI notices:
        - Which gladiator is best at which market regime?
        - What patterns lead to wins vs losses?
        - Are gladiators learning from each other?
        - What new edges are emerging?
        """
        prompt = f"""You are Mother AI, learning from gladiator performance.

CYCLE {cycle_results['cycle_number']} RESULTS:
- Trades approved: {len(cycle_results['trades_approved'])}
- Trades rejected: {len(cycle_results['trades_rejected'])}

GLADIATOR PERFORMANCE:
{json.dumps(cycle_results['gladiator_performance'], indent=2)}

Questions:
1. Which gladiator performed best this cycle? Why?
2. Are there patterns in wins vs losses?
3. What should gladiators try differently next cycle?
4. Any new structural edges emerging?

Output JSON:
{{
  "best_gladiator": "A|B|C|D",
  "best_gladiator_reason": "Why they excelled",
  "patterns_observed": ["Pattern 1", "Pattern 2"],
  "recommendations": ["What to try next"],
  "new_edges_discovered": ["Any novel insights"]
}}"""

        response = self._call_llm(prompt)
        meta_insight = json.loads(response)

        self.meta_learnings.append({
            "cycle": cycle_results['cycle_number'],
            "timestamp": datetime.utcnow().isoformat(),
            "insights": meta_insight
        })

        logger.info(f"🧠 Mother AI meta-learned: {meta_insight['patterns_observed']}")

    def _check_emotional_pressure(self) -> None:
        """
        Trigger DESPERATE MODE if performance is poor

        If win rate < 70% after 2 tournament cycles, Mother AI
        forces gladiators into exploration mode
        """
        current_wr = self.get_overall_win_rate()

        if self.tournament_cycles >= 2 and current_wr < 0.70:
            if not self.desperate_mode:
                logger.warning(f"⚠️  DESPERATE MODE ACTIVATED")
                logger.warning(f"   Win Rate: {current_wr:.1%} < 70%")
                logger.warning(f"   Forcing gladiators to EXPLORE new strategies")
                self.desperate_mode = True

                # Notify all gladiators to explore
                self._trigger_exploration_mode()

    def _trigger_exploration_mode(self) -> None:
        """
        Force gladiators to search for new edges

        Enables:
        - Web search for new trading ideas
        - Higher risk experimentation
        - Novel strategy generation
        """
        logger.info(f"🔍 Mother AI enabling WEB SEARCH for all gladiators")
        # Implementation in Fix #8 (Internet Exploration)

    def delegate_task(self, task_type: str, asset: str) -> str:
        """
        Intelligent delegation: Assign task to best gladiator

        Mother AI learns which gladiator is best at what:
        - A (DeepSeek): Structural edges, pattern discovery
        - B (Claude): Logical validation, risk assessment
        - C (Grok): Historical pattern matching
        - D (Gemini): Synthesis, multi-factor analysis
        """
        delegation_map = {
            "find_new_edge": "A",  # DeepSeek excels at discovery
            "validate_logic": "B",  # Claude excels at critique
            "check_history": "C",   # Grok excels at patterns
            "synthesize": "D"       # Gemini excels at holistic view
        }

        assigned_gladiator = delegation_map.get(task_type, "D")
        logger.info(f"📋 Mother AI delegated '{task_type}' to Gladiator {assigned_gladiator}")
        return assigned_gladiator
```

**Modify**: `/root/crpbot/apps/runtime/hydra_runtime.py`

Change entry point from direct gladiator calls to Mother AI orchestration:

```python
# OLD (current):
def _generate_signal(self, asset: str) -> Optional[Dict]:
    # Direct gladiator calls...

# NEW (with Mother AI):
def __init__(self, ...):
    self.mother_ai = MotherAI(
        api_key=os.getenv("GEMINI_API_KEY"),  # Or ANTHROPIC_API_KEY
        model="gemini-pro"  # Or "claude-opus"
    )

def run(self):
    """Main loop - Mother AI orchestrates everything"""
    while True:
        # Mother AI runs the cycle
        results = self.mother_ai.orchestrate_cycle(self.assets)

        # Mother AI has already approved/rejected trades
        # Execute approved trades only
        for trade in results["trades_approved"]:
            self._execute_trade(trade)
```

**Effort**: 8-12 hours

---

## 🔴 CRITICAL FIX #8: INTERNET EXPLORATION (Web Search)

### Current State (MISSING)
- Gladiators only use training data
- No ability to discover new edges
- Strategies stagnate over time

### Required State
Each gladiator can search internet for new trading ideas:

**Add to each gladiator**:

```python
class BaseGladiator:
    """Add web search capability"""

    def __init__(self, ...):
        self.web_search_enabled = False
        self.search_api_key = os.getenv("SERPER_API_KEY")  # Or SerpAPI

    def search_web(self, query: str) -> List[Dict]:
        """Search internet for trading insights"""
        if not self.web_search_enabled:
            return []

        try:
            # Use Serper API or SerpAPI
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.search_api_key,
                "Content-Type": "application/json"
            }
            payload = {"q": query, "num": 5}

            response = requests.post(url, headers=headers, json=payload)
            results = response.json()

            return results.get("organic", [])
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def explore_new_edges(self, asset: str, desperate_mode: bool = False) -> List[str]:
        """
        Search for new structural edges

        Triggered by Mother AI when desperate_mode=True
        """
        if not desperate_mode:
            return []

        logger.info(f"🔍 Gladiator {self.name} searching web for new edges...")

        queries = [
            f"{asset} structural trading edges 2025",
            f"{asset} perpetual funding rate arbitrage",
            f"{asset} meme coin inefficiencies",
            f"crypto market microstructure {asset}",
            f"algorithmic trading patterns {asset}"
        ]

        new_edges = []
        for query in queries:
            results = self.search_web(query)

            for result in results[:3]:  # Top 3 per query
                # Feed result to LLM for insight extraction
                insight = self._extract_edge_from_article(
                    title=result["title"],
                    snippet=result["snippet"]
                )
                if insight:
                    new_edges.append(insight)

        logger.success(f"📚 Gladiator {self.name} discovered {len(new_edges)} new edges")
        return new_edges

    def _extract_edge_from_article(self, title: str, snippet: str) -> Optional[str]:
        """Extract tradeable insight from web search result"""
        prompt = f"""You are a trading strategy analyst.

Extract a STRUCTURAL EDGE from this article:

Title: {title}
Snippet: {snippet}

If there's a tradeable insight, describe it.
If not, return null.

Output JSON:
{{
  "edge": "Description of structural edge" or null,
  "entry_rules": "How to enter" or null,
  "why_it_works": "Market inefficiency explanation" or null
}}"""

        response = self._call_llm(prompt, temperature=0.3, max_tokens=300)
        result = json.loads(response)

        return result.get("edge")
```

**Trigger via Mother AI**:

```python
# In mother_ai.py
def _trigger_exploration_mode(self) -> None:
    """Enable web search for all gladiators"""
    for gladiator in [self.gladiator_a, self.gladiator_b, self.gladiator_c, self.gladiator_d]:
        gladiator.web_search_enabled = True

        # Force each to explore
        new_edges = gladiator.explore_new_edges(asset="BTC-USD", desperate_mode=True)

        # Add discovered edges to their strategy pool
        for edge in new_edges:
            gladiator.learned_edges.append({
                "source": "web_search",
                "edge": edge,
                "timestamp": datetime.utcnow().isoformat()
            })
```

**API Setup** (`.env`):
```bash
# Web Search API (choose one)
SERPER_API_KEY=your_key_here          # $50 for 5000 searches
# Or
SERPAPI_KEY=your_key_here             # $50/month for 5000 searches
```

**Effort**: 6-8 hours

---

## 🟢 OPTIONAL FIX #9: CONSENSUS AS SECONDARY

Current consensus can remain, but as optional secondary signal:

```python
# In hydra_runtime.py
def _generate_signal(self, asset: str) -> Optional[Dict]:
    # ... all 4 gladiators trade independently ...

    # OPTIONAL: Also generate consensus trade
    if self.consensus_enabled:  # Default: False
        votes = [decision_a, decision_b, decision_c, decision_d]
        consensus = self.consensus.get_consensus(votes)

        # Execute consensus as 5th "meta-gladiator"
        self.paper_trader.create_trade(
            asset=asset,
            direction=consensus["direction"],
            gladiator="CONSENSUS"  # Track separately
        )

    return None  # No single signal - 4+ independent trades
```

**Effort**: 2-3 hours

---

## 📊 IMPLEMENTATION SUMMARY

| Fix # | Component | Priority | Effort | File Impact |
|-------|-----------|----------|--------|-------------|
| 1 | Gladiator Architecture | 🔴 CRITICAL | 8-12h | 6 files (all gladiators + runtime) |
| 2 | Independent P&L Tracking | 🔴 CRITICAL | 4-6h | 2 files (new portfolio + paper_trader) |
| 3 | Tournament Ranking | 🔴 CRITICAL | 2-3h | 1 file (tournament_manager) |
| 4 | Winner Teaches Losers | 🔴 CRITICAL | 3-4h | 2 files (base + tournament_manager) |
| 5 | 24-Hour Elimination | 🔴 CRITICAL | 2-3h | 1 file (tournament_manager) |
| 6 | 4-Day Breeding | 🔴 CRITICAL | 3-4h | 1 file (breeding_engine) |
| 7 | **Mother AI (L1 Supervisor)** | 🔴 **CRITICAL** | 8-12h | 2 files (NEW mother_ai + hydra_runtime) |
| 8 | **Internet Exploration** | 🔴 **CRITICAL** | 6-8h | 5 files (all gladiators + base) |
| 9 | Data Feeds (DXY, News, Session) | 🟡 MEDIUM | 4-6h | 1 file (hydra_runtime) |
| 10 | Consensus Secondary | 🟢 OPTIONAL | 2-3h | 1 file (hydra_runtime) |

**Total Critical Fixes (1-8)**: 44-62 hours
**With Medium/Optional (9-10)**: 50-71 hours

### Why Mother AI & Internet Exploration are CRITICAL:

**Without Mother AI**:
- No final approval gate (trades can violate FTMO rules)
- No meta-learning (system doesn't learn which gladiator excels at what)
- No emotional pressure (gladiators won't explore when stuck)
- No intelligent delegation (wrong gladiator assigned to wrong task)

**Without Internet Exploration**:
- Gladiators only recombine existing knowledge
- No discovery of new structural edges
- Strategies stagnate over time
- Can't adapt to new market regimes
- Stuck at ~40-50% win rate ceiling

**With Both**:
- Mother AI triggers exploration when WR < 70%
- Gladiators search internet for new edges
- System discovers patterns humans haven't found yet
- Potential WR: 70-78% (vs current 40% ceiling)

---

## ✅ WHAT TO KEEP (Already Correct)

| Component | File | Status |
|-----------|------|--------|
| 7 Anti-Manipulation Filters | `anti_manipulation.py` | ✅ Keep |
| Guardian (9 Sacred Rules) | `guardian.py` | ✅ Keep |
| 8 Asset Profiles (FTMO-compatible) | `asset_profiles.py` | ✅ Keep |
| Lesson Memory System | `lesson_memory.py` | ✅ Keep |
| Explainability Logger | `explainability.py` | ✅ Keep |
| Cross-Asset Filter (DXY/BTC) | `cross_asset_filter.py` | ✅ Keep |
| Execution Optimizer | `execution_optimizer.py` | ✅ Keep |
| Regime Detector | `regime_detector.py` | ✅ Keep |
| Emotion Core (Gladiator Soul) | All 4 gladiator files | ✅ Keep |

**These 9 components are architecturally sound and should NOT be touched during rebuild.**

---

## 🎯 THE CORE TRANSFORMATION

```
FROM: Sequential Pipeline
      A → B → C → D → Vote → 1 Trade
      (4 specialists collaborating)

TO:   Parallel Competition
      A → Backtest → Trade (Track A's P&L)
      B → Backtest → Trade (Track B's P&L)
      C → Backtest → Trade (Track C's P&L)
      D → Backtest → Trade (Track D's P&L)
      (4 independent traders competing)
```

**Key Insight**:
- Current HYDRA = 4 AIs working on same problem
- Required HYDRA = 4 AIs competing to be best trader
- Only competition creates evolution
- Only independent P&L creates tournament
- Only tournament creates teaching/breeding/elimination

---

## 📝 NEXT STEPS

1. **Read & Approve** this plan
2. **Create feature branch**: `feature/hydra-architecture-v4`
3. **Implement fixes in order** (1 → 6, then 7-8 if time)
4. **Test each fix** before moving to next
5. **Monitor 1 week** before declaring victory

**Estimated Timeline**:
- Week 1: Fixes 1-3 (Core architecture + P&L tracking)
- Week 2: Fixes 4-6 (Teaching + Evolution cycles)
- Week 3: Testing + Fixes 7-8 (Data feeds + Polish)
- Week 4: Monitoring + Tuning

---

**Status**: 🔴 Awaiting approval to begin rebuild
**Created**: 2025-12-01 03:15 UTC

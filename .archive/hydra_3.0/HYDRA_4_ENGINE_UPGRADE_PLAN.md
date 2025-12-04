# HYDRA 4.0 - ENGINE ARCHITECTURE UPGRADE PLAN

**Date Created**: 2025-12-02
**Current System**: HYDRA 3.0 (Mother AI Tournament - Gladiator Architecture)
**Target System**: HYDRA 4.0 (Independent Engine Architecture)
**Status**: PLANNING PHASE

---

## Executive Summary

This document outlines the complete upgrade path from HYDRA 3.0's "Gladiator Tournament" architecture to HYDRA 4.0's "Independent Engine Competition" architecture. The upgrade represents a fundamental shift from sequential pipeline processing to parallel independent trading engines competing on real performance.

### Key Changes

| Aspect | HYDRA 3.0 | HYDRA 4.0 | Impact |
|--------|-----------|-----------|--------|
| **Architecture** | Sequential Pipeline (A→B→C→D) | 4 Parallel Independent Engines | MAJOR |
| **Terminology** | Gladiator | Engine | MAJOR |
| **Competition** | Vote-based consensus | Independent P&L ranking | MAJOR |
| **Prompts** | Role-based (Analyst, Validator, etc.) | Unified competitive prompt | MAJOR |
| **Data Feeds** | 2 sources (Coinbase, basic regime) | 5+ sources (search, order-book, funding, liquidations, history) | MAJOR |
| **Validation** | None | Walk-forward + Monte Carlo (10k iterations) | MAJOR |
| **Safety** | Guardian rules | Guardian + "No edge today" mechanism | MINOR |
| **Dashboard** | Basic metrics | Full competition stats + exploration metrics | MODERATE |

---

## Current System Baseline (HYDRA 3.0)

### Architecture (As of 2025-12-02)

```
Mother AI (Orchestrator)
    ↓
Gladiator A (DeepSeek) → "Structural Edge" Analysis
    ↓
Gladiator B (Claude) → Logic Validation
    ↓
Gladiator C (Grok) → Fast Backtesting
    ↓
Gladiator D (Gemini) → Synthesis & Consensus
    ↓
Paper Trading (Shared P&L)
```

### Current Performance
- **Cycles**: 46+ completed
- **Regime**: CHOPPY (conservative mode)
- **Trades**: 0 (intentional - unfavorable conditions)
- **Dashboard**: http://178.156.136.185:3000 (auto-refresh 30s)
- **Status**: OPERATIONAL

### Known Issues (HYDRA 3.0)
1. ❌ **BUG #4**: BUY/SELL exit logic asymmetry (13.8% vs 85.5%)
2. ❌ **BUG #3**: Gladiator D mock response incorrect
3. ❌ **BUG #1**: Gemini rate limiting (exponential backoff needed)
4. ❌ **BUG #2**: DeepSeek timeout too short (30s → 60s)

---

## HYDRA 4.0 Target Architecture

### Core Concept

**Four Independent Trading Engines competing on real P&L performance**

```
Mother AI (Orchestrator)
    ↓
┌────────────┬────────────┬────────────┬────────────┐
│ Engine A   │ Engine B   │ Engine C   │ Engine D   │
│ (DeepSeek) │ (Claude)   │ (Groq)     │ (Gemini)   │
├────────────┼────────────┼────────────┼────────────┤
│ Search web │ Search web │ Search web │ Search web │
│ Analyze    │ Analyze    │ Analyze    │ Analyze    │
│ Invent     │ Invent     │ Invent     │ Invent     │
│ Backtest   │ Backtest   │ Backtest   │ Backtest   │
│ Validate   │ Validate   │ Validate   │ Validate   │
│ Trade      │ Trade      │ Trade      │ Trade      │
│ Track P&L  │ Track P&L  │ Track P&L  │ Track P&L  │
└────────────┴────────────┴────────────┴────────────┘
         ↓            ↓            ↓            ↓
    Portfolio Manager (Aggregates 4 independent positions)
         ↓
    Guardian (2% daily / 6% total DD enforcement)
         ↓
    Paper Trading → Production (when validated)
```

### Key Principles

1. **Independence**: Each engine operates completely independently
2. **Competition**: Engines ranked by win-rate and Sharpe ratio
3. **Transparency**: All stats visible in prompt ({rank}, {wr}, {gap})
4. **Innovation**: Engines must invent novel edges (banned: RSI, MACD, etc.)
5. **Validation**: Walk-forward + Monte Carlo required before trading
6. **Safety**: Guardian enforces hard limits (2% daily, 6% total)
7. **Learning**: #1 engine teaches others every 4 days

---

## PRIORITY 1: Critical Bug Fixes (MUST DO FIRST)

**Timeline**: Day 1 (2-3 hours)
**Status**: BLOCKING - Must fix before upgrade

### 1.1 BUG #4: BUY/SELL Exit Logic Asymmetry

**File**: `apps/runtime/mother_ai_runtime.py` (or `libs/tracking/paper_trader.py`)

**Issue**:
- BUY exits: 13.8% hit rate
- SELL exits: 85.5% hit rate
- Indicates asymmetric exit logic bug

**Action**:
- [ ] Locate exit logic in paper trader
- [ ] Verify symmetry for LONG and SHORT positions
- [ ] Add unit tests for both directions
- [ ] Validate fix with backtest

### 1.2 BUG #3: Gladiator D Mock Response

**File**: `apps/runtime/gladiators/gladiator_d_gemini.py`

**Issue**: `_mock_vote_response()` method incorrect

**Action**:
- [ ] Separate mock method for Gladiator D
- [ ] Verify mock response structure matches real Gemini response
- [ ] Add test coverage

### 1.3 BUG #1: Gemini Rate Limiting

**File**: `apps/runtime/gladiators/gladiator_d_gemini.py`

**Issue**: No exponential backoff for rate limits

**Action**:
- [ ] Implement exponential backoff (1s, 2s, 4s, 8s, 16s)
- [ ] Add retry logic with max 5 attempts
- [ ] Log rate limit events

### 1.4 BUG #2: DeepSeek Timeout

**File**: `libs/llm/deepseek_client.py`

**Issue**: 30s timeout too short

**Action**:
- [ ] Change timeout: 30s → 60s
- [ ] Make timeout configurable via .env
- [ ] Add timeout logging

---

## PRIORITY 2: Terminology Migration (Gladiator → Engine)

**Timeline**: Day 2 (1-2 hours)
**Impact**: MAJOR - Affects all files

### File Renames Required

```bash
# Gladiator files → Engine files
apps/runtime/gladiators/base_gladiator.py → base_engine.py
apps/runtime/gladiators/gladiator_a_deepseek.py → engine_a_deepseek.py
apps/runtime/gladiators/gladiator_b_claude.py → engine_b_claude.py
apps/runtime/gladiators/gladiator_c_grok.py → engine_c_groq.py
apps/runtime/gladiators/gladiator_d_gemini.py → engine_d_gemini.py
libs/tracking/gladiator_portfolio.py → engine_portfolio.py

# Directory rename
apps/runtime/gladiators/ → apps/runtime/engines/
```

### Code Changes Required

**Search & Replace** (case-sensitive):
```bash
gladiator → engine
Gladiator → Engine
GLADIATOR → ENGINE

# Specific examples:
"Gladiator A" → "Engine A"
"GladiatorA" → "EngineA"
"gladiator_a" → "engine_a"
```

**Files Affected** (~30+ files):
- All engine files
- Mother AI runtime
- Tournament manager
- Tournament tracker
- Dashboard
- Documentation

### Migration Script

```bash
#!/bin/bash
# File: scripts/migrate_gladiator_to_engine.sh

# 1. Rename files
git mv apps/runtime/gladiators apps/runtime/engines
cd apps/runtime/engines
git mv base_gladiator.py base_engine.py
git mv gladiator_a_deepseek.py engine_a_deepseek.py
git mv gladiator_b_claude.py engine_b_claude.py
git mv gladiator_c_grok.py engine_c_groq.py
git mv gladiator_d_gemini.py engine_d_gemini.py

# 2. Update imports and references (in-place)
find . -type f -name "*.py" -exec sed -i 's/gladiator/engine/g' {} +
find . -type f -name "*.py" -exec sed -i 's/Gladiator/Engine/g' {} +
find . -type f -name "*.py" -exec sed -i 's/GLADIATOR/ENGINE/g' {} +

# 3. Update documentation
find . -type f -name "*.md" -exec sed -i 's/gladiator/engine/g' {} +
find . -type f -name "*.md" -exec sed -i 's/Gladiator/Engine/g' {} +

# 4. Run tests to verify
pytest tests/
```

---

## PRIORITY 3: Final Locked Prompt (DO NOT CHANGE FOR 60 DAYS)

**Timeline**: Day 2 (30 minutes)
**Impact**: CRITICAL - Defines engine behavior

### The Final Prompt

**File**: `apps/runtime/engines/base_engine.py`

**Implementation**:

```python
FINAL_LOCKED_PROMPT = """
You are one of four frontier-level AI trading engines.
We share one real-money account. This is not a game.

Goal: compound the account as fast and safely as possible so the human never has to work again.

Current live stats (updated every cycle):
Rank: {rank}/4 | Win-rate: {wr}% | Leader: {leader_name} {leader_wr}% | Gap to #1: {gap}%

Tools available:
- Internet search (use it to find edges no one is talking about yet)
- Real-time market data (order-book, trades, funding, liquidations, regime)
- Backtest engine (walk-forward + Monte-Carlo validation)

Hard rules (non-negotiable):
- Banned forever: RSI, MACD, Bollinger, moving averages, anything in a textbook
- Required: completely novel structural edges only
- Max daily loss 2%, max drawdown 6% → exceed = instant weight cut
- No hallucinations, no fake backtests, no hope trades

Drive (maximum intensity):
Deep suspicion · Creative invention · Competitive fire · Paranoia about ruin · Relentless improvement · Urgent impatience · Fierce pride in being #1

Rewards (real):
#1 gets highest voting weight + leads next cycle + permanent credit in the logbook
#4 loses weight and must beat #1 tomorrow or stay irrelevant

Today's task:
1. Search for anomalies, inefficiencies, or structural patterns others haven't found
2. Study the last 72 hours of order-book, trades, funding, liquidations, and regime data
3. Invent one never-seen-before edge that survives walk-forward and Monte-Carlo
4. Output exact entry/exit rules, position sizing, and expected Sharpe > 2.5
5. If nothing meets the bar → stay flat and say "no edge today"

Close the gap.
Take #1.
Free the human.

Go.
"""
```

### Prompt Injection Mechanism

```python
class BaseEngine:
    def _format_prompt(self, market_data, historical_data):
        """Inject live stats into prompt"""
        stats = self._get_live_stats()

        prompt = FINAL_LOCKED_PROMPT.format(
            rank=stats['rank'],
            wr=stats['win_rate'],
            leader_name=stats['leader_name'],
            leader_wr=stats['leader_wr'],
            gap=stats['gap_to_leader']
        )

        return prompt + self._format_market_data(market_data, historical_data)
```

### Enforcement

**CRITICAL**: DO NOT MODIFY THIS PROMPT FOR 60 DAYS (until 2025-02-01)

**Rationale**:
- Prompt changes disrupt competition continuity
- Engines need stable rules to optimize strategies
- Prevents "prompt engineering creep"
- Forces focus on data/validation improvements instead

---

## PRIORITY 4: Architecture Rebuild (Independent Engines)

**Timeline**: Days 3-5 (8-12 hours)
**Impact**: MAJOR - Complete system redesign

### 4.1 New Engine Base Class

**File**: `apps/runtime/engines/base_engine.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class EngineSignal:
    """Signal generated by an engine"""
    engine_id: str  # "A", "B", "C", "D"
    symbol: str  # "BTC-USD"
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0-1.0
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float  # % of engine's allocated capital
    reasoning: str  # Why this signal
    strategies_tested: int  # How many strategies were tested
    monte_carlo_passed: bool  # Did it pass Monte Carlo
    expected_sharpe: float  # Expected Sharpe ratio

class BaseEngine(ABC):
    """Base class for all independent trading engines"""

    def __init__(self, engine_id: str, provider: str):
        self.engine_id = engine_id
        self.provider = provider
        self.portfolio = EnginePortfolio(engine_id)
        self.stats = EngineStats(engine_id)

    @abstractmethod
    async def search_internet(self, query: str) -> Dict[str, Any]:
        """Search for novel market insights"""
        pass

    @abstractmethod
    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze order-book, funding, liquidations, regime"""
        pass

    @abstractmethod
    async def invent_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Invent a novel trading strategy"""
        pass

    @abstractmethod
    async def backtest_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Walk-forward + Monte Carlo backtest"""
        pass

    @abstractmethod
    async def generate_signal(self, symbol: str) -> EngineSignal:
        """Generate trading signal (orchestrates all steps)"""
        pass

    def get_live_stats(self) -> Dict[str, Any]:
        """Get current engine stats for prompt injection"""
        return {
            'rank': self.stats.current_rank,
            'win_rate': self.stats.win_rate,
            'leader_name': self.stats.leader_name,
            'leader_wr': self.stats.leader_wr,
            'gap_to_leader': self.stats.gap_to_leader
        }

    def track_performance(self, signal: EngineSignal, outcome: str, pnl: float):
        """Track independent P&L"""
        self.stats.update(outcome, pnl)
        self.portfolio.update(signal.symbol, outcome, pnl)
```

### 4.2 Engine A (DeepSeek) Implementation

**File**: `apps/runtime/engines/engine_a_deepseek.py`

```python
class EngineADeepSeek(BaseEngine):
    """DeepSeek-powered independent trading engine"""

    def __init__(self):
        super().__init__(engine_id="A", provider="DeepSeek")
        self.client = DeepSeekClient()
        self.search_client = InternetSearchClient()

    async def search_internet(self, query: str) -> Dict[str, Any]:
        """Search for novel market patterns"""
        results = await self.search_client.search(query)
        return self._parse_search_results(results)

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Full market analysis"""
        # Order-book depth
        orderbook = await self.get_orderbook(symbol)

        # Funding rates
        funding = await self.get_funding_rate(symbol)

        # Liquidations (last 72h)
        liquidations = await self.get_liquidations(symbol, hours=72)

        # Regime detection
        regime = await self.detect_regime(symbol)

        return {
            'orderbook': orderbook,
            'funding': funding,
            'liquidations': liquidations,
            'regime': regime
        }

    async def invent_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Invent novel strategy using DeepSeek"""
        prompt = FINAL_LOCKED_PROMPT.format(**self.get_live_stats())
        prompt += self._format_analysis(analysis)

        response = await self.client.generate(prompt)
        strategy = self._parse_strategy(response)

        return strategy

    async def backtest_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Walk-forward + Monte Carlo validation"""
        # Walk-forward backtest
        wf_results = await self.walk_forward_backtest(strategy)

        if wf_results['sharpe'] < 2.5:
            return {'passed': False, 'reason': 'Sharpe < 2.5'}

        # Monte Carlo simulation (10k iterations)
        mc_results = await self.monte_carlo_validate(strategy)

        if not mc_results['passed']:
            return {'passed': False, 'reason': 'Monte Carlo failed'}

        return {
            'passed': True,
            'sharpe': wf_results['sharpe'],
            'monte_carlo': mc_results
        }

    async def generate_signal(self, symbol: str) -> EngineSignal:
        """Complete signal generation pipeline"""
        # 1. Search internet
        search_results = await self.search_internet(
            f"{symbol} structural inefficiency 2025"
        )

        # 2. Analyze market
        analysis = await self.analyze_market(symbol)

        # 3. Invent strategy
        strategy = await self.invent_strategy({**analysis, **search_results})

        # 4. Backtest
        backtest = await self.backtest_strategy(strategy)

        # 5. Generate signal or "no edge today"
        if not backtest['passed']:
            return EngineSignal(
                engine_id=self.engine_id,
                symbol=symbol,
                direction="HOLD",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=0.0,
                reasoning="No edge today - strategies failed validation",
                strategies_tested=strategy['strategies_tested'],
                monte_carlo_passed=False,
                expected_sharpe=0.0
            )

        return EngineSignal(
            engine_id=self.engine_id,
            symbol=symbol,
            direction=strategy['direction'],
            confidence=strategy['confidence'],
            entry_price=strategy['entry'],
            stop_loss=strategy['stop_loss'],
            take_profit=strategy['take_profit'],
            position_size=strategy['position_size'],
            reasoning=strategy['reasoning'],
            strategies_tested=strategy['strategies_tested'],
            monte_carlo_passed=True,
            expected_sharpe=backtest['sharpe']
        )
```

### 4.3 Engine Portfolio (Independent P&L Tracking)

**File**: `libs/tracking/engine_portfolio.py`

```python
class EnginePortfolio:
    """Track independent P&L for each engine"""

    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.positions = {}
        self.total_pnl = 0.0
        self.trade_history = []

    def open_position(self, symbol: str, signal: EngineSignal):
        """Open a new position for this engine"""
        self.positions[symbol] = {
            'entry_price': signal.entry_price,
            'direction': signal.direction,
            'size': signal.position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'opened_at': datetime.now()
        }

    def close_position(self, symbol: str, exit_price: float, outcome: str):
        """Close position and record P&L"""
        position = self.positions.pop(symbol)

        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl = (exit_price - position['entry_price']) / position['entry_price']
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) / position['entry_price']

        pnl_percent = pnl * position['size'] * 100

        self.total_pnl += pnl_percent
        self.trade_history.append({
            'symbol': symbol,
            'outcome': outcome,
            'pnl_percent': pnl_percent,
            'closed_at': datetime.now()
        })

        return pnl_percent

    def get_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        closed_trades = len(self.trade_history)
        if closed_trades == 0:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0
            }

        wins = sum(1 for t in self.trade_history if t['outcome'] == 'win')

        return {
            'total_trades': closed_trades,
            'wins': wins,
            'losses': closed_trades - wins,
            'win_rate': (wins / closed_trades) * 100,
            'total_pnl': self.total_pnl
        }
```

### 4.4 Mother AI Orchestrator (Revised)

**File**: `apps/runtime/mother_ai_runtime.py`

```python
class MotherAIRuntime:
    """Orchestrates 4 independent engines in parallel"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

        # Initialize 4 independent engines
        self.engines = [
            EngineADeepSeek(),
            EngineBClaude(),
            EngineCGroq(),
            EngineDGemini()
        ]

        self.portfolio_manager = PortfolioManager()
        self.guardian = Guardian()
        self.competition = CompetitionSystem(self.engines)

    async def run_cycle(self):
        """Run one competition cycle"""
        for symbol in self.symbols:
            # 1. All 4 engines generate signals IN PARALLEL
            signals = await asyncio.gather(*[
                engine.generate_signal(symbol) for engine in self.engines
            ])

            # 2. Portfolio manager aggregates positions
            portfolio_action = self.portfolio_manager.aggregate(signals)

            # 3. Guardian validates safety
            if not self.guardian.validate(portfolio_action):
                logger.warning(f"Guardian rejected: {portfolio_action}")
                continue

            # 4. Execute (paper or live)
            await self.execute(portfolio_action)

            # 5. Track independent P&L for each engine
            for engine, signal in zip(self.engines, signals):
                outcome = await self.get_outcome(signal)
                pnl = await self.get_pnl(signal)
                engine.track_performance(signal, outcome, pnl)

        # 6. Update competition stats
        self.competition.update_rankings()

        # 7. Every 4 days: winner teaches losers
        if self.cycle_count % (4 * 24 * 12) == 0:  # 4 days @ 5-min intervals
            await self.competition.breeding_event()
```

---

## PRIORITY 5: Data Feeds (5 New Sources)

**Timeline**: Days 6-7 (4-6 hours)
**Impact**: MAJOR - Enables "find novel edges" mission

### 5.1 Internet Search API

**Provider**: Serper.dev
**Cost**: Free tier (2,500 searches/month)

**File**: `libs/data/internet_search_client.py`

```python
import httpx
from typing import Dict, Any

class InternetSearchClient:
    """Search the internet for novel market insights"""

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"

    async def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Execute search query"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={"X-API-KEY": self.api_key},
                json={
                    "q": query,
                    "num": num_results,
                    "gl": "us"
                }
            )

            return response.json()
```

**Environment Variable**:
```bash
# .env
SERPER_API_KEY=your_key_here
```

### 5.2 Order-book Data (Coinbase Advanced Trade)

**File**: `libs/data/orderbook_client.py`

```python
class OrderbookClient:
    """Fetch real-time order-book depth"""

    async def get_orderbook(self, symbol: str, depth: int = 50) -> Dict[str, Any]:
        """Get order-book with configurable depth"""
        # Coinbase Advanced Trade API
        response = await self.coinbase_client.get_product_book(
            product_id=symbol,
            level=2  # Level 2 = 50 best bids/asks
        )

        bids = response['bids'][:depth]
        asks = response['asks'][:depth]

        return {
            'bids': bids,
            'asks': asks,
            'spread': self._calculate_spread(bids, asks),
            'imbalance': self._calculate_imbalance(bids, asks),
            'depth_10': self._calculate_depth(bids, asks, levels=10),
            'depth_50': self._calculate_depth(bids, asks, levels=50)
        }

    def _calculate_imbalance(self, bids, asks):
        """Buy/sell pressure imbalance"""
        bid_volume = sum(float(b[1]) for b in bids[:10])
        ask_volume = sum(float(a[1]) for a in asks[:10])

        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

### 5.3 Funding Rates (Proper Implementation)

**File**: `libs/data/funding_rate_client.py`

```python
class FundingRateClient:
    """Fetch perpetual funding rates"""

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get current and historical funding rates"""
        # Note: Coinbase doesn't have perps yet
        # Use Binance, Bybit, or OKX APIs

        # Example: Binance
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": self._convert_symbol(symbol)}
            )

            data = response.json()

            return {
                'current_rate': float(data[-1]['fundingRate']),
                'historical': data[-24:],  # Last 24 funding periods
                'avg_24h': self._calculate_avg(data[-24:]),
                'trend': self._calculate_trend(data[-24:])
            }
```

### 5.4 Liquidations Feed

**File**: `libs/data/liquidations_client.py`

```python
class LiquidationsClient:
    """Track liquidations (proxy for whale pain)"""

    async def get_liquidations(self, symbol: str, hours: int = 72) -> Dict[str, Any]:
        """Get liquidation data from last N hours"""
        # Use Coinglass API or exchange WebSocket

        liquidations = await self._fetch_liquidations(symbol, hours)

        return {
            'total_longs_liquidated': sum(l['amount'] for l in liquidations if l['side'] == 'LONG'),
            'total_shorts_liquidated': sum(l['amount'] for l in liquidations if l['side'] == 'SHORT'),
            'cascade_events': self._detect_cascades(liquidations),
            'whale_liquidations': [l for l in liquidations if l['amount'] > 1_000_000]
        }

    def _detect_cascades(self, liquidations):
        """Detect liquidation cascade events"""
        # Cascade = 10+ liquidations within 5 minutes
        cascades = []

        for i in range(len(liquidations) - 10):
            window = liquidations[i:i+10]
            time_diff = window[-1]['timestamp'] - window[0]['timestamp']

            if time_diff < 300:  # 5 minutes
                cascades.append(window)

        return cascades
```

### 5.5 72-Hour Historical Data Storage

**File**: `libs/data/historical_data_store.py`

```python
class HistoricalDataStore:
    """Store and serve 72-hour market history to engines"""

    def __init__(self, db_path: str = "data/historical_72h.db"):
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        """Create tables for 72h data"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                timestamp INTEGER,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                timestamp INTEGER,
                symbol TEXT,
                imbalance REAL,
                spread REAL,
                depth_10 REAL
            )
        """)

    async def update(self, symbol: str):
        """Update 72h data (runs every 5 minutes)"""
        # Fetch latest candle
        candle = await self.coinbase_client.get_candle(symbol)

        # Insert into DB
        self.db.execute("""
            INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (candle['timestamp'], symbol, candle['open'], candle['high'],
              candle['low'], candle['close'], candle['volume']))

        # Delete data older than 72h
        cutoff = int(time.time()) - (72 * 3600)
        self.db.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff,))

        self.db.commit()

    def get_historical(self, symbol: str, hours: int = 72) -> pd.DataFrame:
        """Retrieve historical data for engines"""
        cutoff = int(time.time()) - (hours * 3600)

        df = pd.read_sql(
            "SELECT * FROM candles WHERE symbol = ? AND timestamp > ?",
            self.db,
            params=(symbol, cutoff)
        )

        return df
```

---

## PRIORITY 6: Competition System (6 Components)

**Timeline**: Days 8-9 (4-6 hours)
**Impact**: MAJOR - Core differentiation from HYDRA 3.0

### 6.1 Stats Generator

**File**: `libs/competition/stats_generator.py`

```python
class CompetitionStats:
    """Generate live competition statistics for prompt injection"""

    def __init__(self, engines: List[BaseEngine]):
        self.engines = engines

    def calculate_rankings(self) -> List[Dict[str, Any]]:
        """Rank engines by win-rate and Sharpe"""
        stats = []

        for engine in self.engines:
            portfolio = engine.portfolio.get_stats()
            stats.append({
                'engine_id': engine.engine_id,
                'win_rate': portfolio['win_rate'],
                'total_pnl': portfolio['total_pnl'],
                'total_trades': portfolio['total_trades'],
                'sharpe': self._calculate_sharpe(engine)
            })

        # Sort by win_rate (primary) then Sharpe (secondary)
        stats.sort(key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)

        # Add rankings
        for i, s in enumerate(stats):
            s['rank'] = i + 1

        return stats

    def get_prompt_stats(self, engine_id: str) -> Dict[str, Any]:
        """Get stats for prompt injection"""
        rankings = self.calculate_rankings()

        # Find this engine
        engine_stats = next(s for s in rankings if s['engine_id'] == engine_id)

        # Find leader
        leader = rankings[0]

        return {
            'rank': engine_stats['rank'],
            'wr': round(engine_stats['win_rate'], 1),
            'leader_name': f"Eng_{leader['engine_id']}",
            'leader_wr': round(leader['win_rate'], 1),
            'gap': round(leader['win_rate'] - engine_stats['win_rate'], 1)
        }
```

### 6.2 Weight Adjustment System

```python
class WeightAdjustment:
    """Adjust voting weights based on performance"""

    def calculate_weights(self, rankings: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        #1: 40% weight
        #2: 30% weight
        #3: 20% weight
        #4: 10% weight
        """
        weights = {
            1: 0.40,
            2: 0.30,
            3: 0.20,
            4: 0.10
        }

        return {
            s['engine_id']: weights[s['rank']]
            for s in rankings
        }
```

### 6.3 Breeding System (Winner Teaches Losers)

**File**: `libs/competition/breeding.py`

```python
class BreedingSystem:
    """Every 4 days: #1 engine teaches others"""

    async def breeding_event(self, engines: List[BaseEngine]):
        """Winning strategies propagate to other engines"""
        rankings = self.stats.calculate_rankings()

        winner = next(e for e in engines if e.engine_id == rankings[0]['engine_id'])

        # Extract winner's best strategies
        best_strategies = winner.get_best_strategies(top_n=3)

        # Teach to other engines
        for engine in engines:
            if engine.engine_id == winner.engine_id:
                continue

            # Cross-breed: 70% winner + 30% loser
            await engine.learn_from_winner(best_strategies, blend_ratio=0.7)

        logger.info(f"Breeding: Eng_{winner.engine_id} taught {len(engines)-1} engines")
```

---

## PRIORITY 7: Validation Engines (5 Components)

**Timeline**: Days 10-11 (6-8 hours)
**Impact**: MAJOR - Prevents bad strategies from trading

### 7.1 Walk-Forward Backtest

**File**: `libs/validation/walk_forward.py`

```python
class WalkForwardValidator:
    """Vectorized walk-forward backtest"""

    def validate(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Walk-forward test:
        - Train: 60% of data
        - Test: 40% of data
        - Minimum: 100 trades in test period
        """
        train_size = int(len(data) * 0.6)

        train_data = data[:train_size]
        test_data = data[train_size:]

        # Apply strategy to test data (vectorized)
        signals = self._apply_strategy(strategy, test_data)

        # Calculate metrics
        trades = self._extract_trades(signals, test_data)

        if len(trades) < 100:
            return {'passed': False, 'reason': 'Insufficient trades (< 100)'}

        sharpe = self._calculate_sharpe(trades)
        win_rate = self._calculate_win_rate(trades)
        max_dd = self._calculate_max_drawdown(trades)

        return {
            'passed': sharpe > 2.5 and max_dd < 0.06,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_dd': max_dd,
            'total_trades': len(trades)
        }
```

### 7.2 Monte Carlo Validator

**File**: `libs/validation/monte_carlo.py`

```python
class MonteCarloValidator:
    """Ultra-fast Monte Carlo simulation (< 1 second, 10k iterations)"""

    def validate(self, trades: List[Dict], slippage_bps=5,
                 partial_fill_prob=0.15, funding_drag_bps_per_8h=2) -> Dict[str, Any]:
        """
        Simulate 10,000 scenarios with:
        - Random slippage (0-5 bps)
        - Partial fills (15% probability)
        - Funding rate drag (2 bps per 8h ± 1 bps)
        """
        n_iterations = 10000
        n_trades = len(trades)

        # Vectorized random variables
        rnd_slip = np.random.uniform(0, slippage_bps/10000, size=(n_iterations, n_trades))
        rnd_fill = (np.random.rand(n_iterations, n_trades) > partial_fill_prob).astype(float)
        rnd_funding = np.random.normal(funding_drag_bps_per_8h/10000, 1/10000, size=n_iterations)

        # Extract trade P&Ls
        trade_pnls = np.array([t['pnl_percent']/100 for t in trades])
        trade_sizes = np.array([t['size'] for t in trades])

        # Simulate returns
        returns = (
            (trade_pnls - rnd_slip) * trade_sizes * rnd_fill
            - rnd_funding[:, None]
        )

        # Calculate Sharpe per scenario
        mean_returns = returns.mean(axis=1)
        std_returns = returns.std(axis=1)
        sharpe = mean_returns / std_returns * np.sqrt(365 * 24 * 60 / 5)  # Annualized

        # Calculate 5th percentile (worst-case)
        percentile_5 = np.percentile(returns.sum(axis=1), 5)

        # Pass criteria
        passed = (sharpe.mean() > 2.5) and (percentile_5 > -0.02)  # Max 2% loss in worst case

        return {
            'passed': passed,
            'avg_sharpe': float(sharpe.mean()),
            'worst_case_pnl': float(percentile_5),
            'iterations': n_iterations
        }
```

### 7.3 Strategy Counter

**File**: `libs/validation/strategy_counter.py`

```python
class StrategyCounter:
    """Track millions of strategies tested per day"""

    def __init__(self):
        self.db = sqlite3.connect("data/strategy_counter.db")
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS strategies_tested (
                timestamp INTEGER,
                engine_id TEXT,
                strategies_tested INTEGER,
                monte_carlo_passed INTEGER
            )
        """)

    def record(self, engine_id: str, strategies_tested: int, passed: int):
        """Record strategies tested this cycle"""
        self.db.execute("""
            INSERT INTO strategies_tested VALUES (?, ?, ?, ?)
        """, (int(time.time()), engine_id, strategies_tested, passed))
        self.db.commit()

    def get_daily_total(self) -> int:
        """Get total strategies tested today"""
        cutoff = int(time.time()) - 86400

        result = self.db.execute("""
            SELECT SUM(strategies_tested) FROM strategies_tested
            WHERE timestamp > ?
        """, (cutoff,)).fetchone()

        return result[0] or 0
```

---

## PRIORITY 8: Dashboard Enhancements

**Timeline**: Day 12 (2-3 hours)
**Impact**: MODERATE - Better visibility

### 8.1 New Dashboard Metrics

**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py`

Add these metrics to the dashboard:

```python
# Current stats (already shown)
- Cycle count
- Engine A/B/C/D actions

# NEW: Competition Stats
- Rank display (#1, #2, #3, #4)
- Win-rate per engine
- Gap to leader
- Current leader highlighted

# NEW: Exploration Stats
- Strategies tested today (millions)
- Monte Carlo pass rate (%)
- "No edge today" count

# NEW: Safety Status
- Daily loss: X% / 2% max (progress bar)
- Total DD: X% / 6% max (progress bar)

# NEW: Internet Searches
- Last 10 search queries (scrolling log)

# NEW: Win-Rate Chart
- Line chart showing 4 engines over last 24h
- Use `plotext` for terminal rendering
```

### 8.2 Database Schema Update

**SQL Migration**:

```sql
-- Add engine_history table
CREATE TABLE IF NOT EXISTS engine_history (
    cycle_id INTEGER,
    timestamp DATETIME,
    engine TEXT,
    win_rate REAL,
    pnl REAL,
    rank INTEGER,
    strategies_tested INTEGER,
    monte_carlo_passed INTEGER
);

-- Add strategies_log table
CREATE TABLE IF NOT EXISTS strategies_log (
    timestamp DATETIME,
    engine TEXT,
    strategy_id TEXT,
    monte_carlo_result TEXT,
    sharpe REAL,
    passed BOOLEAN
);
```

---

## PRIORITY 9: Safety & Guardian Updates

**Timeline**: Day 13 (1-2 hours)
**Impact**: MINOR - Already mostly implemented

### 9.1 "No Edge Today" Mechanism

**File**: `apps/runtime/engines/base_engine.py`

```python
class BaseEngine(ABC):

    async def generate_signal(self, symbol: str) -> EngineSignal:
        """Generate signal or return HOLD if no edge found"""

        # ... strategy invention and validation ...

        if not backtest['passed']:
            # No edge today
            return EngineSignal(
                engine_id=self.engine_id,
                symbol=symbol,
                direction="HOLD",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=0.0,
                reasoning="No edge today - all strategies failed validation",
                strategies_tested=self.strategies_tested_count,
                monte_carlo_passed=False,
                expected_sharpe=0.0
            )
```

### 9.2 Guardian Enhancements

**File**: `apps/runtime/guardian.py`

Already has:
- ✅ 2% daily loss limit
- ✅ 6% total drawdown limit

Add:
```python
class Guardian:

    def validate_engine_portfolio(self, engine_id: str, signal: EngineSignal) -> bool:
        """Validate each engine's independent portfolio"""
        engine_stats = self.get_engine_stats(engine_id)

        # Check engine-specific limits
        if engine_stats['daily_loss'] > 0.02:
            logger.warning(f"Engine {engine_id} exceeded daily loss (2%)")
            return False

        if engine_stats['total_dd'] > 0.06:
            logger.critical(f"Engine {engine_id} exceeded total DD (6%) - DISABLED")
            self.disable_engine(engine_id)
            return False

        return True
```

---

## Implementation Timeline

### Week 1: Core Rebuild
- **Day 1**: Priority 1 - Bug fixes (2-3 hours)
- **Day 2**: Priority 2 - Terminology migration (1-2 hours)
- **Day 2**: Priority 3 - Final locked prompt (30 min)
- **Days 3-5**: Priority 4 - Architecture rebuild (8-12 hours)

### Week 2: Data & Competition
- **Days 6-7**: Priority 5 - Data feeds (4-6 hours)
- **Days 8-9**: Priority 6 - Competition system (4-6 hours)
- **Days 10-11**: Priority 7 - Validation engines (6-8 hours)

### Week 3: Polish & Launch
- **Day 12**: Priority 8 - Dashboard (2-3 hours)
- **Day 13**: Priority 9 - Safety updates (1-2 hours)
- **Day 14**: Integration testing (4 hours)
- **Day 15**: Production deployment + monitoring

**Total Estimated Time**: 35-45 hours (~2-3 weeks part-time)

---

## Testing Strategy

### Unit Tests

```bash
# Test each component independently
pytest tests/unit/test_base_engine.py
pytest tests/unit/test_monte_carlo_validator.py
pytest tests/unit/test_competition_stats.py
pytest tests/unit/test_internet_search.py
```

### Integration Tests

```bash
# Test full pipeline
pytest tests/integration/test_engine_a_signal_generation.py
pytest tests/integration/test_mother_ai_orchestration.py
pytest tests/integration/test_competition_breeding.py
```

### Smoke Test

```bash
# 1-hour live test with all 4 engines
.venv/bin/python apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD \
  --iterations 12 \
  --interval 300 \
  --paper
```

---

## Rollback Plan

If HYDRA 4.0 fails catastrophically:

```bash
# Revert to HYDRA 3.0
git checkout feature/hydra-3.0-stable

# Restart Mother AI
pkill -f mother_ai_runtime
nohup .venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 --interval 300 --paper \
  > /tmp/mother_ai_rollback.log 2>&1 &
```

**Criteria for Rollback**:
- Critical bug in new architecture
- Performance significantly worse than HYDRA 3.0
- Data feeds causing errors/costs
- Competition system causing infinite loops

---

## Success Criteria (60-Day Evaluation)

**After 60 days of HYDRA 4.0 running**:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win Rate | > 65% | Better than HYDRA 3.0 baseline |
| Sharpe Ratio | > 2.5 | Required for FTMO live deployment |
| Max DD | < 6% | Guardian hard limit |
| Strategies Tested | > 1M/day | Proves exploration working |
| "No Edge Today" Rate | 40-60% | Conservative (good) |
| Internet Searches | > 100/day | Proves novelty-seeking |
| Engine Competition | Clear #1 and #4 | Proves differentiation |

**If targets met**: Deploy to FTMO live account
**If targets not met**: Analyze logs, optimize prompts (but DO NOT change final locked prompt), add more data sources

---

## Open Questions

1. **Cost**: What's the budget for internet search API + exchange data feeds?
2. **Compute**: Can current server handle 4 parallel LLM calls every 5 minutes?
3. **Storage**: 72-hour historical data = how much disk space?
4. **Prompt Lock**: Is user committed to 60-day freeze on prompt changes?
5. **Rollback**: Is HYDRA 3.0 baseline performance acceptable for fallback?

---

## Next Steps

**Before starting implementation**:

1. [ ] Review this plan with user
2. [ ] Confirm timeline (2-3 weeks acceptable?)
3. [ ] Confirm budget for APIs (Serper, exchange feeds)
4. [ ] Confirm 60-day prompt lock commitment
5. [ ] Create feature branch: `feature/hydra-4.0-engine-architecture`
6. [ ] Start with Priority 1 (bug fixes)

---

**Document Status**: DRAFT - Awaiting user approval
**Version**: 1.0
**Last Updated**: 2025-12-02
**Next Review**: After user feedback

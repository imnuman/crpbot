# HYDRA 3.0 - Complete System Documentation

**Date**: 2025-12-01
**Status**: ✅ **PRODUCTION READY**
**Version**: 3.0 (Mother AI + Independent Gladiators)

---

## 🎯 Executive Summary

HYDRA 3.0 is a **competitive multi-agent trading tournament** where 4 AI gladiators with different LLM providers trade independently, competing for the #1 rank based on P&L performance.

**Key Innovation**: Transformed from consensus-based voting to independent competitive trading with tournament rankings, breeding mechanisms, and knowledge transfer.

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MOTHER AI (L1 Supervisor)                 │
│                  - Orchestration & Tournament                │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│ Gladiator A  ││ Gladiator B  ││ Gladiator C  ││ Gladiator D  │
│  (DeepSeek)  ││  (Claude)    ││   (Grok)     ││  (Gemini)    │
├──────────────┤├──────────────┤├──────────────┤├──────────────┤
│ Portfolio A  ││ Portfolio B  ││ Portfolio C  ││ Portfolio D  │
│ P&L: $X      ││ P&L: $Y      ││ P&L: $Z      ││ P&L: $W      │
│ Rank: #1-4   ││ Rank: #1-4   ││ Rank: #1-4   ││ Rank: #1-4   │
│ Weight: 40%  ││ Weight: 30%  ││ Weight: 20%  ││ Weight: 10%  │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
```

---

## 🧬 Components

### 1. Mother AI (L1 Supervisor)
**File**: `libs/hydra/mother_ai.py` (542 lines)

**Responsibilities**:
- Orchestrate trading cycles across all 4 gladiators
- Gather market intelligence (regime, orderbook, feeds, search)
- Coordinate independent decision-making
- Execute trades for each gladiator
- Monitor and close trades (SL/TP)
- Update tournament rankings
- Adjust weights every 24 hours
- Execute breeding every 4 days
- Implement "Winner Teaches Losers"

**Key Methods**:
- `run_trading_cycle()` - Main orchestration loop
- `_gather_market_intelligence()` - Market data collection
- `_collect_gladiator_decisions()` - Parallel decision-making
- `_execute_gladiator_trades()` - Trade execution
- `_update_all_trades()` - SL/TP monitoring
- `_adjust_weights()` - 24-hour weight adjustment
- `_execute_breeding()` - 4-day breeding mechanism

---

### 2. Gladiator A (DeepSeek) - "Structural Edge Hunter"
**File**: `libs/hydra/gladiators/gladiator_a_deepseek.py` (734 lines)

**Personality**: Aggressive
- **Risk Profile**: 2.0-3.0% position sizing
- **Temperature**: 0.6 (creative)
- **Focus**: Market mechanics, structural inefficiencies
- **Style**: High conviction, bold trades

**Key Features**:
- Independent portfolio tracking
- Tournament-aware prompts (rank, P&L, leader status)
- Confidence-based position sizing
- Regime-aligned decision-making
- Autonomous SL/TP monitoring

---

### 3. Gladiator B (Claude) - "Logic Validator"
**File**: `libs/hydra/gladiators/gladiator_b_claude.py` (714 lines)

**Personality**: Conservative
- **Risk Profile**: 1.5-2.5% position sizing
- **Temperature**: 0.4 (analytical)
- **Focus**: Logical consistency, risk management
- **Style**: Careful validation, filters edge cases

**Unique Traits**:
- Logic validation checklist
- Conservative risk approach
- Edge case detection
- Contradiction analysis

---

### 4. Gladiator C (Grok) - "Pattern Matcher"
**File**: `libs/hydra/gladiators/gladiator_c_grok.py` (602 lines)

**Personality**: Moderate
- **Risk Profile**: 1.8-2.7% position sizing
- **Temperature**: 0.3 (consistent)
- **Focus**: Historical patterns, precedent-based
- **Style**: Pattern recognition, data-driven

**Unique Traits**:
- Pattern recognition checklist
- Historical precedent analysis
- Moderate risk balance
- Consistent decision-making

---

### 5. Gladiator D (Gemini) - "Synthesizer"
**File**: `libs/hydra/gladiators/gladiator_d_gemini.py` (758 lines)

**Personality**: Balanced
- **Risk Profile**: 1.7-2.8% position sizing
- **Temperature**: 0.5 (balanced)
- **Focus**: Holistic view, confluence factors
- **Style**: Multi-perspective synthesis

**Unique Traits**:
- Synthesis checklist
- Confluence analysis
- Balanced risk approach
- Multi-timeframe perspective

---

## 🏆 Tournament System

### Rankings
Based on **P&L performance** (USD):
1. **Rank #1**: 40% tournament weight
2. **Rank #2**: 30% tournament weight
3. **Rank #3**: 20% tournament weight
4. **Rank #4**: 10% tournament weight

### Weight Adjustment (24-Hour Cycle)
- Automatically adjusts weights based on performance
- **No killing** - all gladiators continue competing
- Weights determine influence (if voting is reintroduced)

### Breeding Mechanism (4-Day Cycle)
1. Identify top 2 gladiators
2. Extract successful patterns from winners
3. Combine patterns into "offspring" strategy
4. Apply insights to bottom 2 gladiators
5. Winner teaches losers system activated

### Winner Teaches Losers
- Winners' successful patterns extracted
- Insights applied to learners
- Pattern propagation from top to bottom performers
- Knowledge transfer system

---

## 🔄 Trading Cycle Flow

```
1. Mother AI Initiates Cycle
   ↓
2. Detect Market Regime (RegimeDetector)
   ↓
3. Gather Market Intelligence
   - Order book analysis
   - Market data feeds (funding, liquidations)
   - Internet search (optional)
   ↓
4. All 4 Gladiators Make Independent Decisions (Parallel)
   - Gladiator A → Decision A (or HOLD)
   - Gladiator B → Decision B (or HOLD)
   - Gladiator C → Decision C (or HOLD)
   - Gladiator D → Decision D (or HOLD)
   ↓
5. Execute Trades
   - Open trades for gladiators who decided to trade
   - Store in individual portfolios
   ↓
6. Monitor Existing Trades
   - Check SL/TP for all open positions
   - Close trades when triggered
   - Update P&L
   ↓
7. Update Tournament Rankings
   - Calculate P&L for each gladiator
   - Rank #1-4
   - Update weights if 24h passed
   ↓
8. Check Tournament Events
   - Weight adjustment (every 24 hours)
   - Breeding mechanism (every 4 days)
   ↓
9. Sleep & Repeat
```

---

## 🚀 Deployment

### Production Runtime Script
**File**: `apps/runtime/mother_ai_runtime.py` (290 lines)

### Quick Start

```bash
# Test with 1 cycle
.venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD \
  --iterations 1 \
  --interval 10

# Run continuous tournament (infinite loop, 5-min intervals)
.venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 \
  --interval 300 \
  --paper

# Run in background with logging
nohup .venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 \
  --interval 300 \
  > /tmp/mother_ai_$(date +%Y%m%d_%H%M).log 2>&1 &

# Save PID for monitoring
echo $! > /tmp/mother_ai.pid
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--assets` | Space-separated list of symbols | `BTC-USD ETH-USD SOL-USD` |
| `--iterations` | Number of cycles (-1 = infinite) | `-1` |
| `--interval` | Seconds between cycles | `300` (5 min) |
| `--paper` | Paper trading mode (always on for safety) | `True` |

---

## 📈 Performance Tracking

### Individual Gladiator Metrics
Each gladiator tracks:
- **Total Trades**: Number of trades executed
- **Wins/Losses**: Trade outcomes
- **Win Rate**: Percentage of winning trades
- **Total P&L (USD)**: Cumulative profit/loss
- **Total P&L (%)**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Open Trades**: Currently active positions
- **Closed Trades**: Historical trade log

### Tournament Metrics
- **Current Rank**: #1-4 based on P&L
- **Tournament Weight**: 40%/30%/20%/10%
- **Cycles Completed**: Total trading cycles run
- **Last Weight Adjustment**: Timestamp of last adjustment
- **Last Breeding**: Timestamp of last breeding event

---

## 📁 File Structure

```
crpbot/
├── libs/hydra/
│   ├── mother_ai.py                    # Mother AI orchestrator (542 lines)
│   ├── tournament_manager.py           # Tournament rankings & weights
│   ├── gladiator_portfolio.py          # Individual P&L tracking
│   ├── regime_detector.py              # Market regime detection
│   ├── market_data_feeds.py            # Funding/liquidations feeds
│   ├── orderbook_feed.py               # Order book analysis
│   ├── internet_search.py              # Web search integration
│   └── gladiators/
│       ├── base_gladiator.py           # Base class
│       ├── gladiator_a_deepseek.py     # DeepSeek gladiator (734 lines)
│       ├── gladiator_b_claude.py       # Claude gladiator (714 lines)
│       ├── gladiator_c_grok.py         # Grok gladiator (602 lines)
│       └── gladiator_d_gemini.py       # Gemini gladiator (758 lines)
│
├── apps/runtime/
│   └── mother_ai_runtime.py            # Production runtime (290 lines)
│
└── HYDRA_3.0_COMPLETE_2025-12-01.md    # This file
```

---

## 🔧 Configuration

### Environment Variables Required

```bash
# LLM API Keys
DEEPSEEK_API_KEY=sk-...        # Gladiator A
ANTHROPIC_API_KEY=sk-...       # Gladiator B
XAI_API_KEY=...                # Gladiator C
GEMINI_API_KEY=...             # Gladiator D

# Data Provider
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# Premium Data (Optional)
COINGECKO_API_KEY=CG-...       # Market context
SERPER_API_KEY=...             # Internet search (optional)
```

---

## 🎨 Gladiator Personalities Summary

| Gladiator | LLM | Personality | Position Size | Temp | Focus |
|-----------|-----|-------------|---------------|------|-------|
| **A** | DeepSeek | Aggressive | 2.0-3.0% | 0.6 | Structural edges |
| **B** | Claude | Conservative | 1.5-2.5% | 0.4 | Logic validation |
| **C** | Grok | Moderate | 1.8-2.7% | 0.3 | Pattern matching |
| **D** | Gemini | Balanced | 1.7-2.8% | 0.5 | Synthesis |

---

## ✅ Features Implemented

### Phase 3: Gladiator Independence ✅
- [x] Portfolio integration for all 4 gladiators
- [x] Independent `make_trade_decision()` method
- [x] Trade execution (`open_trade()`, `update_trades()`)
- [x] Confidence-based position sizing
- [x] Tournament-aware prompts (rank, P&L, leader status)
- [x] Regime-aligned decision-making
- [x] Autonomous SL/TP monitoring

### Phase 4: Mother AI & Tournament ✅
- [x] Mother AI orchestration layer
- [x] Tournament ranking system (P&L-based)
- [x] 24-hour weight adjustment (no killing)
- [x] 4-day breeding mechanism
- [x] Winner teaches losers system
- [x] Real-time tournament standings
- [x] Production runtime script
- [x] Comprehensive logging

---

## 🧪 Testing

### Manual Test (1 Cycle)
```bash
.venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD \
  --iterations 1 \
  --interval 10
```

**Expected Output**:
- Mother AI initializes with 4 gladiators
- Regime detection runs
- Market intelligence gathered
- All 4 gladiators make decisions (some may HOLD)
- Trades opened for gladiators who decided to trade
- Tournament standings displayed
- Cycle completes successfully

---

## 📊 Monitoring

### Check Runtime Status
```bash
# Find running process
ps aux | grep mother_ai_runtime | grep -v grep

# View latest log
tail -f /tmp/mother_ai_*.log

# View tournament standings in log
grep "TOURNAMENT STANDINGS" /tmp/mother_ai_*.log | tail -20
```

### Stop Runtime
```bash
# If you saved PID
kill $(cat /tmp/mother_ai.pid)

# Or find and kill
pkill -f mother_ai_runtime
```

---

## 🔮 Future Enhancements

### Potential Phase 5 Features
- [ ] Real-time dashboard (Reflex/Streamlit)
- [ ] Advanced breeding algorithms (genetic programming)
- [ ] Multi-asset simultaneous trading
- [ ] Live trading mode (beyond paper trading)
- [ ] Database persistence for portfolio history
- [ ] Performance analytics & visualization
- [ ] Telegram/Discord notifications
- [ ] API endpoints for external monitoring

---

## 📚 Related Documentation

- `GLADIATOR_A_INDEPENDENCE_2025-12-01.md` - Gladiator A refactoring details
- `libs/hydra/README.md` - HYDRA 3.0 overview
- `libs/hydra/gladiators/README.md` - Gladiator architecture
- `apps/runtime/README.md` - Runtime documentation

---

## 🎯 Success Metrics

### Immediate Metrics (Week 1)
- ✅ All 4 gladiators trade independently
- ✅ Tournament rankings update correctly
- ✅ No system crashes or errors
- ✅ Paper trades execute and close properly

### Short-Term Metrics (Month 1)
- [ ] Positive P&L for at least 2 gladiators
- [ ] Win rate > 50% for top-ranked gladiator
- [ ] Sharpe ratio > 1.0 for any gladiator
- [ ] Breeding mechanism improves bottom performers

### Long-Term Metrics (Quarter 1)
- [ ] System outperforms baseline buy-and-hold
- [ ] Breeding produces novel winning strategies
- [ ] Winner teaches losers demonstrably effective
- [ ] Tournament competition drives innovation

---

## 🏁 Conclusion

HYDRA 3.0 represents a **complete paradigm shift** from consensus-based trading to competitive independent trading. The system is now:

✅ **Production-Ready**
✅ **Fully Autonomous**
✅ **Competitively Driven**
✅ **Evolutionarily Adaptive**

**Next Step**: Deploy to production and monitor performance for statistical significance (20+ trades per gladiator recommended before optimization).

---

**Last Updated**: 2025-12-01
**Status**: Production Ready
**Version**: 3.0
**Author**: Builder Claude
**License**: Proprietary

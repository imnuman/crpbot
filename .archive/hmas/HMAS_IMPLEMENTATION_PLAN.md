# HMAS (Hierarchical Multi-Agent System) Implementation Plan

**Date**: 2025-11-26
**Goal**: 80%+ Win Rate Trading System with 1.0% Risk per Trade
**Strategy**: Mean Reversion + 200-MA Trend Filter (FTMO Compliant)

---

## 🏗️ Architecture Overview

### Layer Structure

```
┌─────────────────────────────────────────────────────────┐
│  L1: MOTHER AI (Gemini)                                 │
│  - Orchestration & Final Decision                       │
│  - Risk Governance (1.0% per trade)                     │
│  - FTMO Compliance Check                                │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────┬────────────┐
        │                     │              │            │
┌───────▼────────┐  ┌────────▼───────┐  ┌──▼──────┐  ┌──▼──────┐
│ L2: DeepSeek   │  │ L3: Groq       │  │ L4:     │  │ Market  │
│ Alpha Gen      │  │ Execution      │  │ Claude  │  │ Data    │
│                │  │ Auditor        │  │ Memory  │  │ Feed    │
└────────────────┘  └────────────────┘  └─────────┘  └─────────┘
```

---

## 📋 Component Specifications

### L1: Mother AI (Gemini)

**API**: Google Gemini 2.0 Flash Experimental
**Role**: Supervisor / Risk Governance
**Responsibilities**:
1. Orchestrate all 3 specialist agents
2. Calculate lot size (1.0% risk per trade)
3. Perform final FTMO compliance check
4. Synthesize final signal from all inputs
5. Make final GO/NO-GO decision

**Key Functions**:
- `orchestrate_trade_signal()` - Main entry point
- `calculate_lot_size()` - FTMO 1.0% risk calculation
- `validate_ftmo_compliance()` - Final safety check
- `synthesize_final_signal()` - Combine all agent outputs

**Output Format**:
```
=== APPROVED TRADE SIGNAL ===
Asset: GBPUSD
Action: SELL STOP
Entry: 1.25500
Stop Loss: 1.25650 (15 pips, 1.0% risk)
Take Profit: 1.25100 (40 pips, 2.67:1 R:R)
Lot Size: 0.67 lots
Win Rate Target: 80%+
200-MA Alignment: ✅ CONFIRMED
ALM Monitoring: ✅ ACTIVE
Rationale: [From Claude]
```

---

### L2: Alpha Generator (DeepSeek)

**API**: DeepSeek v3
**Role**: Data & Pattern Recognition
**Responsibilities**:
1. Scan M15/M30 timeframes for mean reversion setups
2. Check Bollinger Bands (price touching outer band)
3. Confirm RSI oversold (<30) or overbought (>70)
4. Verify 200-MA trend alignment
5. Generate initial Trade Hypothesis

**Input**:
- Symbol (e.g., GBPUSD)
- Timeframe (M15/M30)
- Current market data (OHLCV + indicators)

**Output**:
```json
{
  "trade_hypothesis": {
    "asset": "GBPUSD",
    "action": "SELL",
    "entry": 1.25500,
    "sl": 1.25650,
    "tp": 1.25100,
    "confidence": 0.85,
    "setup_type": "mean_reversion_bbands_rsi",
    "ma200_alignment": true,
    "ma200_value": 1.25200,
    "current_price": 1.25500,
    "bbands_signal": "upper_band_touch",
    "rsi_value": 72,
    "evidence": "Price at upper BB with RSI overbought, 200-MA below = bearish trend"
  }
}
```

---

### L3: Execution Auditor (Groq)

**API**: Groq (Llama 3.3 70B)
**Role**: Speed & Aggressive Loss Management (ALM)
**Responsibilities**:
1. Pre-trade cost validation (spread + fees < 0.5× TP)
2. Verify TP profit > 2× (spread + fees)
3. Setup ALM monitoring (1× ATR threshold)
4. Real-time "CLOSE NOW" emergency signal capability

**Input**:
- Trade hypothesis from DeepSeek
- Current spread & fees
- ATR value

**Output**:
```json
{
  "audit_result": "PASS",
  "cost_check": {
    "spread_pips": 1.5,
    "fees_pips": 0.5,
    "total_cost_pips": 2.0,
    "tp_pips": 40,
    "cost_to_tp_ratio": 0.05,
    "threshold": 0.50,
    "status": "PASS"
  },
  "alm_setup": {
    "active": true,
    "atr_value": 0.00150,
    "atr_threshold": "1x ATR",
    "emergency_close_level": 1.25650,
    "monitoring_interval": "1 second"
  },
  "recommendation": "APPROVED_FOR_EXECUTION"
}
```

---

### L4: Rationale Agent (Claude)

**API**: Claude 3.5 Sonnet
**Role**: Explanation & Memory
**Responsibilities**:
1. Generate human-readable trade rationale
2. Explain the statistical basis for 80%+ WR
3. Store trade outcome for learning
4. Provide accountability & transparency

**Input**:
- Trade hypothesis from DeepSeek
- Gemini's risk calculation
- Historical performance data

**Output**:
```markdown
### Trade Rationale (80%+ Win Rate Strategy)

**Setup**: Mean Reversion at Bollinger Band Upper Extreme
**Trend Filter**: 200-MA confirms bearish bias (1.25200 < 1.25500)

**Statistical Edge**:
- Mean reversion from BB extremes: 78% historical win rate
- RSI >70 reversals in downtrend: 82% historical win rate
- Combined setup: 85% win rate over 247 historical occurrences

**Risk Management**:
- Stop: 15 pips (1× ATR) = 1.0% account risk
- Target: 40 pips = 2.67:1 reward/risk ratio
- Expected Value: (0.85 × 2.67) - (0.15 × 1) = +2.12R

**FTMO Compliance**: ✅
- Daily loss limit: Well within 4.5% (risking only 1.0%)
- Max loss limit: Well within 9%
- Position size: 0.67 lots calculated for exact 1.0% risk

**Conclusion**: HIGH PROBABILITY SETUP - Execute with confidence
```

---

## 🔄 4-Step Trade Execution Flow

### Step 1: Alpha Generation (DeepSeek)
```
Mother AI → DeepSeek: "Find mean reversion setup for GBPUSD M15"
DeepSeek → Analysis: Check BB, RSI, 200-MA
DeepSeek → Mother AI: Trade Hypothesis (SELL @ 1.25500, SL 1.25650, TP 1.25100)
```

### Step 2: Risk Calculation (Gemini)
```
Mother AI → Self: Calculate lot size for 1.0% risk
Calculation: ($10,000 × 0.01) / (15 pips × $6.7/pip) = 0.67 lots
Mother AI → Claude: "Generate rationale for this setup"
```

### Step 3: Execution Audit (Groq)
```
Mother AI → Groq: "Validate this trade for execution"
Groq → Pre-flight Check:
  - Spread check: 1.5 pips < 20 pips TP threshold ✅
  - Cost/TP ratio: 0.05 < 0.50 threshold ✅
  - ALM setup: Emergency close @ 1.25650 ✅
Groq → Mother AI: "APPROVED_FOR_EXECUTION"
```

### Step 4: Final Decision (Gemini)
```
Mother AI → Synthesis:
  - DeepSeek confidence: 85%
  - Groq audit: PASS
  - Claude rationale: 80%+ WR setup
  - FTMO check: ✅ 1.0% risk confirmed
Mother AI → User: APPROVED TRADE SIGNAL
```

---

## 🔧 Implementation Components

### Required Files

1. **Core Agents**:
   - `libs/hmas/agents/mother_ai.py` (Gemini orchestrator)
   - `libs/hmas/agents/alpha_generator.py` (DeepSeek pattern recognition)
   - `libs/hmas/agents/execution_auditor.py` (Groq pre-flight check)
   - `libs/hmas/agents/rationale_agent.py` (Claude memory & explanation)

2. **Orchestration**:
   - `libs/hmas/orchestrator/trade_orchestrator.py` (Main workflow)
   - `libs/hmas/orchestrator/ftmo_calculator.py` (Risk & lot size)

3. **Configuration**:
   - `libs/hmas/config/hmas_config.py` (API keys, settings)
   - `libs/hmas/config/strategy_params.py` (Mean reversion parameters)

4. **Utilities**:
   - `libs/hmas/utils/market_data.py` (Fetch OHLCV + indicators)
   - `libs/hmas/utils/indicator_calculator.py` (BB, RSI, MA200)

5. **Entry Point**:
   - `apps/hmas/hmas_signal_generator.py` (CLI interface)

---

## 📊 API Requirements

### API Keys Needed

1. **Google Gemini**:
   - Model: `gemini-2.0-flash-exp`
   - Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/`

2. **DeepSeek**:
   - Already have: `DEEPSEEK_API_KEY=sk-...`
   - Model: `deepseek-chat`

3. **Groq**:
   - Need: `GROQ_API_KEY`
   - Model: `llama-3.3-70b-versatile`
   - Get key from: https://console.groq.com/keys

4. **Anthropic Claude**:
   - Need: `ANTHROPIC_API_KEY`
   - Model: `claude-3-5-sonnet-20241022`
   - Get key from: https://console.anthropic.com/

---

## 🎯 Success Criteria

1. **Win Rate**: System generates signals with 80%+ historical win rate
2. **Risk**: Exact 1.0% risk per trade (FTMO compliant)
3. **Speed**: Signal generation < 10 seconds (all 4 agents)
4. **Cost**: Groq audit blocks trades with excessive spread/fees
5. **Transparency**: Claude provides clear, understandable rationale

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Day 1)
- [ ] Setup API clients for all 4 LLMs
- [ ] Create base agent classes
- [ ] Implement FTMO risk calculator

### Phase 2: Agent Implementation (Day 2-3)
- [ ] DeepSeek Alpha Generator (mean reversion detection)
- [ ] Groq Execution Auditor (cost validation)
- [ ] Claude Rationale Agent (explanation generation)
- [ ] Gemini Mother AI (orchestration)

### Phase 3: Integration (Day 4)
- [ ] Trade Orchestrator (4-step workflow)
- [ ] Market data fetching
- [ ] Indicator calculation (BB, RSI, MA200)

### Phase 4: Testing (Day 5)
- [ ] Unit tests for each agent
- [ ] Integration test (full workflow)
- [ ] Backtest on historical data
- [ ] Live paper trading test

### Phase 5: Deployment (Day 6)
- [ ] CLI interface
- [ ] Signal logging & tracking
- [ ] Performance monitoring
- [ ] Documentation

---

## 💰 Cost Estimates

**Per Signal**:
- Gemini: $0.0002 (flash model, cheap)
- DeepSeek: $0.0005 (already using)
- Groq: $0.0001 (very cheap, fast)
- Claude: $0.003 (most expensive, but worth it for rationale)
- **Total**: ~$0.004 per signal

**Monthly** (100 signals):
- Total: $0.40/month
- Very affordable for 80%+ WR signals!

---

## 🎓 Next Steps

1. **Get API Keys**: Groq & Anthropic Claude
2. **Start Building**: Begin with Phase 1 (Foundation)
3. **Test Iteratively**: Each agent gets tested independently
4. **Deploy**: Full system when all agents working

---

**Status**: Ready to implement
**Estimated Time**: 5-6 days for complete system
**Risk Level**: Low (using proven components)
**Expected ROI**: Very High (80%+ WR trading signals)

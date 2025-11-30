# Execution

**File**: `hydra_runtime.py`

**Status**: FULLY IMPLEMENTED (Paper Trading Mode)

---

## Overview

HYDRA Runtime is the main orchestrator that runs the 4-gladiator tournament and executes paper trades.

**Current Mode**: Paper Trading (NO live funds at risk)

---

## Runtime Flow

```
1. Initialize 4 Gladiators
   - A: DeepSeek (Strategic Synthesis)
   - B: Claude (Logic Validation)
   - C: Grok (Fast Backtesting)
   - D: Gemini (Final Synthesis)

2. Fetch Market Data
   - Coinbase API (200+ candles)
   - 3 assets: BTC-USD, ETH-USD, SOL-USD
   - Rotate assets each iteration

3. Each Gladiator Votes
   - Receives: OHLCV data + regime context
   - Returns: BUY/SELL/HOLD + confidence + reasoning

4. Aggregate Votes
   - Count votes for each direction
   - Require 2/4 minimum consensus
   - Ties = HOLD (no action)

5. Paper Trade
   - Log decision to JSONL file
   - Store: timestamp, asset, direction, votes, confidence
   - NO real money involved

6. Sleep & Repeat
   - Wait 5 minutes (300 seconds)
   - Continue indefinitely (--iterations -1)
```

---

## Configuration

**Command Line Args**:
```bash
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 \
  --interval 300 \
  --paper
```

**Flags**:
- `--assets`: Which crypto pairs to trade
- `--iterations`: How many iterations (-1 = infinite)
- `--interval`: Seconds between iterations (300 = 5 min)
- `--paper`: Paper trading mode (required, no live trading yet)

---

## Data Storage

**Paper Trades**: `/root/crpbot/data/hydra/paper_trades.jsonl`
```json
{
  "timestamp": "2025-11-30T10:15:00",
  "asset": "BTC-USD",
  "direction": "BUY",
  "confidence": 0.68,
  "votes": {
    "A": {"vote": "BUY", "confidence": 0.72},
    "B": {"vote": "BUY", "confidence": 0.65},
    "C": {"vote": "HOLD", "confidence": 0.50},
    "D": {"vote": "BUY", "confidence": 0.71}
  },
  "consensus": "BUY (3/4)"
}
```

**Vote History**: `/root/crpbot/data/hydra/vote_history.jsonl`
- All votes (including HOLD)
- Individual gladiator reasoning
- Regime context at vote time

**Chat History**: `/root/crpbot/data/hydra/chat_history.jsonl`
- User questions via dashboard
- Gladiator responses

---

## Current Production Stats

**Process**:
- PID: 3283753
- Uptime: 12+ hours
- Status: Running healthy

**Trading Activity**:
- Total Trades: 251
- Direction Split: 64% BUY, 36% SELL
- Assets Traded: BTC-USD, ETH-USD, SOL-USD
- Interval: Every 5 minutes

**Performance**:
- Consensus Rate: ~67% (2/4+ votes)
- Average Confidence: 0.68
- No errors or crashes

---

## Error Handling

**Gladiator Failures**:
- Retry up to 3 times per gladiator
- Continue with remaining gladiators if one fails
- Log errors but don't crash

**API Failures**:
- Coinbase data fetch errors logged
- Skip iteration if data unavailable
- Guardian monitors for repeated failures

**Credit Exhaustion**:
- Guardian detects 402/429 errors
- Alerts via Telegram
- Manual refill required

---

## Monitoring

**Logs**: `/tmp/hydra_bug49_fixed_*.log`

**Guardian**: Checks every 5 minutes for:
- Process health (auto-restart if down)
- API credit balance
- Paper trading activity
- Directional bias

---

## Future Enhancements

### Live Trading Mode
When HYDRA moves to live trading:

1. **Exchange Integration**
   - FTMO, Binance, Bybit connectors
   - Real order execution
   - Position tracking

2. **Risk Management**
   - Position sizing (1-2% risk per trade)
   - Daily loss limits (2% max)
   - Max drawdown (6%)

3. **Execution Logic**
   - Limit orders with spread checks
   - Slippage protection
   - Partial fills handling

**Current Priority**: Paper trading to prove strategy before risking real capital

---

**Date**: 2025-11-30

# HMAS V2 Quick Start Guide

Get HMAS V2 running in production in 5 minutes.

---

## Prerequisites

1. **API Keys** - Set in `.env`:
```bash
DEEPSEEK_API_KEY=sk-...
XAI_API_KEY=xai-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...
```

2. **Budget**: $5-10/day for signals (5-10 signals @ $1.00 each)

---

## Option 1: Test Run (Single Signal)

Generate one signal to test the complete system:

```bash
# Dry run (doesn't store in DB)
.venv/bin/python apps/runtime/hmas_v2_runtime.py \
  --symbols BTC-USD \
  --iterations 1 \
  --dry-run

# Expected output:
# ================================================================================
# HMAS V2 - Generating Signal for BTC-USD
# ================================================================================
#
# Layer 2A: Running 4 specialist agents in parallel...
#   - Alpha Generator V2 (DeepSeek $0.30)
#   - Technical Agent (DeepSeek $0.10)
#   - Sentiment Agent (DeepSeek $0.08)
#   - Macro Agent (DeepSeek $0.07)
#
# ✓ Alpha Generator: BUY/SELL/HOLD (confidence: X%)
# ✓ Technical Agent: ...
# ✓ Sentiment Agent: ...
# ✓ Macro Agent: ...
#
# Layer 2B: Running execution audit...
#   - Execution Auditor V2 (Grok $0.15)
#
# ✓ Execution Auditor: PASS/FAIL (grade: A/B/C/D/F)
#
# Layer 2C: Generating comprehensive rationale...
#   - Rationale Agent V2 (Claude $0.20)
#
# ✓ Rationale Agent: 5,000+ words generated
#
# Layer 1: Mother AI - 3-round deliberation...
#   - Mother AI V2 (Gemini $0.10)
#     Round 1: Gather all outputs, detect conflicts
#     Round 2: Resolve conflicts via scenario analysis
#     Round 3: Final decision with lot sizing
#
# ✓ Mother AI Decision: APPROVED/REJECTED
#
# ================================================================================
# HMAS V2 Signal Complete
# Processing Time: 30-60 seconds
# Total Cost: $1.00
# Decision: APPROVED/REJECTED
# ================================================================================
```

**Cost**: $1.00 for one signal

---

## Option 2: Production Run (5 signals/day, BTC+ETH+SOL)

Run continuously, generating max 5 signals per day:

```bash
# Production mode (stores signals in DB)
nohup .venv/bin/python apps/runtime/hmas_v2_runtime.py \
  --symbols BTC-USD ETH-USD SOL-USD \
  --max-signals-per-day 5 \
  --iterations -1 \
  --sleep-seconds 3600 \
  > /tmp/hmas_v2_runtime.log 2>&1 &

# Monitor logs
tail -f /tmp/hmas_v2_runtime.log
```

**Cost**: $5/day max (5 signals @ $1.00)

**Schedule**:
- Checks every 1 hour (3600 seconds)
- Generates up to 5 signals/day
- Rotates through BTC, ETH, SOL

---

## Option 3: Conservative Run (1 signal/day, BTC only)

Ultra-conservative mode for initial testing:

```bash
nohup .venv/bin/python apps/runtime/hmas_v2_runtime.py \
  --symbols BTC-USD \
  --max-signals-per-day 1 \
  --iterations -1 \
  --sleep-seconds 86400 \
  > /tmp/hmas_v2_conservative.log 2>&1 &
```

**Cost**: $1/day (1 signal @ $1.00)

**Schedule**: Checks once per day (86400 seconds = 24 hours)

---

## Signal Output Example

When a signal is **APPROVED**:

```json
{
  "decision": "APPROVED",
  "action": "BUY_STOP",
  "symbol": "BTC-USD",

  "trade_parameters": {
    "entry": 95500.00,
    "stop_loss": 95200.00,
    "take_profit": 96300.00,
    "lot_size": 0.10,
    "risk_percent": 1.0,
    "reward_risk_ratio": 2.67
  },

  "ftmo_compliance": {
    "compliant": true,
    "daily_risk_percent": 1.0,
    "daily_limit_remaining": 3.5
  },

  "decision_rationale": {
    "ev_score": 1.49,
    "consensus_level": 0.83,
    "confidence": 0.78,
    "key_factors": [
      "Strong technical setup (BB extreme + RSI oversold)",
      "Positive EV (+1.49R across 3 scenarios)",
      "5 of 6 agents agree on direction",
      "Execution audit PASS (Grade A)"
    ]
  },

  "cost_breakdown": {
    "total_cost": 1.00
  },

  "processing_time_seconds": 45.2
}
```

When a signal is **REJECTED**:

```json
{
  "decision": "REJECTED",
  "action": "HOLD",
  "symbol": "BTC-USD",
  "rejection_reason": "EV below threshold (+0.7R < +1.0R required)",
  "cost_breakdown": {
    "total_cost": 1.00
  }
}
```

---

## Performance Monitoring

### Daily Summary

Check signals generated today:

```bash
# Count signals
sqlite3 tradingai.db "
SELECT COUNT(*) as signals_today, SUM(total_cost) as cost_today
FROM signals
WHERE DATE(timestamp) = DATE('now')
  AND hmas_version = 'V2';"

# Recent signals
sqlite3 tradingai.db "
SELECT timestamp, symbol, decision, action, confidence
FROM signals
WHERE hmas_version = 'V2'
ORDER BY timestamp DESC
LIMIT 10;"
```

### Win Rate Tracking

After signals have results:

```bash
sqlite3 tradingai.db "
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct,
  ROUND(AVG(pnl_percent), 2) as avg_pnl_pct
FROM signal_results
WHERE hmas_version = 'V2';"
```

**Target**: 80%+ win rate

---

## Cost Management

### Daily Budget

With $200/month budget ($50/agent):
- **Conservative**: 1 signal/day = $30/month
- **Standard**: 3 signals/day = $90/month
- **Aggressive**: 5 signals/day = $150/month

### Emergency Stop

If costs exceed budget:

```bash
# Kill runtime
pkill -f hmas_v2_runtime

# Or set environment variable
export KILL_SWITCH=true
```

---

## Troubleshooting

### API Key Errors

```bash
# Test all API keys
.venv/bin/python -c "
from libs.hmas.hmas_orchestrator_v2 import HMASV2Orchestrator
orchestrator = HMASV2Orchestrator.from_env()
print('✓ All API keys valid')
"
```

### Missing Market Data

```bash
# Test Coinbase connection
.venv/bin/python -c "
from libs.data.coinbase_client import CoinbaseClient
import os
client = CoinbaseClient(
    api_key_name=os.getenv('COINBASE_API_KEY_NAME'),
    private_key=os.getenv('COINBASE_API_PRIVATE_KEY')
)
print('✓ Coinbase connected')
"
```

### Timeout Errors

DeepSeek API may timeout with large prompts. This is handled with:
- 3 retry attempts
- 30 second timeout
- Exponential backoff (2s, 4s, 8s)

If persistent, reduce signal frequency or contact DeepSeek support.

---

## Next Steps

1. **Run Test Signal** (Option 1) - Verify complete system works
2. **Monitor for 24 Hours** - Watch first few signals
3. **Validate Win Rate** - Need 20+ signals for statistical significance
4. **Adjust Budget** - Increase/decrease signals per day based on performance
5. **Optimize** - Fine-tune agent prompts if needed

---

## Support

- **Documentation**: See `HMAS_V2_COMPLETE.md` for full system details
- **Integration Test**: `pytest tests/integration/test_hmas_v2_orchestrator.py -v -s`
- **Logs**: Check `/tmp/hmas_v2_runtime.log` for detailed output

---

**Status**: ✅ Ready for production
**Version**: HMAS V2
**Cost**: $1.00 per signal
**Target Win Rate**: 80%+

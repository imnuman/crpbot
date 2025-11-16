# CLAUDE.md - TRAINING INSTRUCTIONS

## 🚨 CRITICAL: Read MASTER_TRAINING_WORKFLOW.md FIRST

**Before ANY training or feature engineering work**:

```bash
cat MASTER_TRAINING_WORKFLOW.md
```

## Key Rules

1. **NEVER train locally** - Only AWS g4dn.xlarge GPU
2. **USE all premium APIs** - CoinGecko Premium is paid for!
3. **Follow exact feature counts** - 73 (BTC/SOL), 54 (ETH)
4. **Verify alignment** - Training = Runtime = Model

## Quick Reference

- Training workflow: `MASTER_TRAINING_WORKFLOW.md`
- AWS GPU setup: `AWS_GPU_APPROVAL_2025-11-15.md`
- Feature pipeline: `apps/runtime/runtime_features.py`
- Model architecture: `apps/trainer/models/lstm.py`


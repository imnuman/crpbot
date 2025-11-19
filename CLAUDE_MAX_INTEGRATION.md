# Claude Max + Projects Integration Guide

## Overview

With **Claude Max**, you get Claude Projects - a powerful way to maintain context across conversations. This is superior to API calls because:

- ✅ **Unlimited context**: Entire codebase always available
- ✅ **Persistent memory**: I remember all previous conversations
- ✅ **No API costs**: Included in your Claude Max subscription
- ✅ **Shared knowledge**: Upload files once, use across all chats
- ✅ **Custom instructions**: Set project-specific guidelines

## Quick Setup (5 minutes)

### Step 1: Create Project

1. Go to https://claude.ai/projects
2. Click **"New Project"**
3. Name: `CRPBot - Trading AI`
4. Description: `LSTM/Transformer ensemble for crypto trading with FTMO compliance`

### Step 2: Add Project Knowledge

Upload these key files to the Project:

**Core Documentation**:
- `CLAUDE.md` - Main project instructions
- `PHASE6_5_RESTART_PLAN.md` - Current phase status
- `COLAB_EVALUATION.md` - Colab workflow
- `COLAB_INTEGRATION_GUIDE.md` - Integration patterns

**Key Code Files**:
- `apps/trainer/models/lstm.py` - LSTM architecture
- `apps/trainer/data_pipeline.py` - Data loading
- `apps/trainer/features.py` - Feature engineering
- `scripts/evaluate_model.py` - Evaluation script

**Colab Notebooks**:
- `colab_evaluate_50feat_models.ipynb` - Main evaluation
- `colab_with_claude_api.ipynb` - API version

### Step 3: Set Custom Instructions

Add these to your Project's custom instructions:

```
You are helping build CRPBot, a cryptocurrency trading AI system.

CURRENT STATUS:
- Phase 6.5: Model Training & Evaluation
- 3 LSTM models trained (BTC, ETH, SOL) - 50 features, 128/3/bidirectional
- 50-feature datasets engineered with multi-timeframe data
- Ready for GPU evaluation in Google Colab

KEY CONSTRAINTS:
- Models expect exactly 50 input features
- Promotion gates: ≥68% accuracy, ≤5% calibration error
- FTMO compliance required (5% daily / 10% total loss limits)

WORKFLOW:
1. User runs tasks in Colab
2. If errors occur, user pastes error here
3. You provide immediate fix
4. Maintain context across conversations

PRIORITIES:
- Accuracy over speed (but leverage GPU when available)
- Production-ready code (no shortcuts)
- FTMO compliance always enforced
```

### Step 4: Share Files with Project

You can now share Colab outputs with the Project:

**From Colab**:
```python
# After evaluation completes
from google.colab import files

# Download results
files.download('evaluation_results.csv')
files.download('evaluation.log')
```

**In Claude Project Chat**:
1. Click **"Add content"** (paperclip icon)
2. Upload downloaded files
3. Ask: "Review these evaluation results and check promotion gates"
4. I'll analyze with full project context

## Optimal Workflow for Colab

### Workflow A: Manual with Projects (FREE)

**In Colab**:
```python
# 1. Run evaluation
!python evaluate_gpu.py 2>&1 | tee evaluation.log

# 2. If error occurs
from google.colab import files
files.download('evaluation.log')
```

**In Claude Project**:
1. Upload `evaluation.log`
2. Ask: "Fix this Colab error"
3. I provide solution with full codebase context
4. Apply fix in Colab
5. Re-run

**Benefits**:
- No API costs (included in Claude Max)
- I have full project context
- Conversation history preserved
- Can reference any file from repo

### Workflow B: Projects + API (AUTOMATED)

If you still want full automation, use Projects API:

```python
# In Colab - use your Claude Max API key
import anthropic

client = anthropic.Anthropic(api_key="your-claude-max-api-key")

def ask_project(question: str, error: str = None):
    """Ask Claude with full project context."""

    # Reference your project knowledge
    prompt = f"""Using the CRPBot project knowledge:

Question: {question}

{f'Error:\\n{error}' if error else ''}

Provide fix with reference to existing code patterns."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

# Use it
try:
    !python evaluate_gpu.py
except Exception as e:
    fix = ask_project("How do I fix this evaluation error?", str(e))
    print(fix)
```

## Advanced: Real-Time Monitoring

### Option 1: Colab → Claude Project (Manual)

**Setup Checkpoints**:
```python
# In Colab evaluation script
import json
from datetime import datetime

def checkpoint(stage: str, data: dict):
    """Save checkpoint for Claude review."""
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "data": data
    }

    with open(f'checkpoint_{stage}.json', 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    print(f"✅ Checkpoint saved: {stage}")

# Use throughout evaluation
checkpoint("data_loaded", {
    "rows": len(df),
    "features": len(feature_cols),
    "memory_mb": df.memory_usage().sum() / 1e6
})

checkpoint("model_loaded", {
    "params": sum(p.numel() for p in model.parameters()),
    "device": str(device)
})

checkpoint("evaluation_started", {
    "num_batches": len(test_loader),
    "batch_size": batch_size
})
```

Download checkpoints periodically and upload to Project for monitoring.

### Option 2: Slack/Discord Notifications

**Colab → Webhook → Claude Project**:
```python
# Send updates to Slack/Discord
import requests

def notify_claude(message: str, error: bool = False):
    """Send notification that you can forward to Claude."""
    webhook_url = "your-webhook-url"

    payload = {
        "text": f"{'🚨 ERROR' if error else '📊 UPDATE'}: {message}",
        "username": "Colab Evaluator"
    }

    requests.post(webhook_url, json=payload)

# Use it
try:
    evaluate_models()
    notify_claude("Evaluation completed successfully!")
except Exception as e:
    notify_claude(f"Evaluation failed: {e}", error=True)
    # You see notification, paste error to Claude Project
```

## Best Practice: Hybrid Approach

**Recommended workflow for Claude Max users**:

1. **Create Project** with all documentation (one-time, 5 min)
2. **Run Colab** evaluation manually
3. **If errors** → Upload logs to Project → Get instant fix
4. **Share results** → Upload CSV to Project → Get analysis
5. **Iterate** → All context preserved across conversations

**Advantages over API-only**:
- No per-request costs
- Better context (entire codebase)
- Conversation history
- Can ask follow-up questions
- More natural interaction

## Files for Project Upload

I've prepared these files for easy Project setup:

**Essential Documentation** (upload to Project Knowledge):
```bash
/root/crpbot/CLAUDE.md                    # Main instructions
/root/crpbot/PHASE6_5_RESTART_PLAN.md     # Current phase
/root/crpbot/COLAB_EVALUATION.md          # Colab guide
/root/crpbot/COLAB_INTEGRATION_GUIDE.md   # Integration patterns
/root/crpbot/CLAUDE_MAX_INTEGRATION.md    # This file
```

**Key Code** (for reference):
```bash
/root/crpbot/apps/trainer/models/lstm.py
/root/crpbot/apps/trainer/features.py
/root/crpbot/apps/trainer/multi_tf_features.py
/root/crpbot/scripts/evaluate_model.py
/root/crpbot/scripts/engineer_50_features.py
```

**Colab Notebooks**:
```bash
/root/crpbot/colab_evaluate_50feat_models.ipynb
/root/crpbot/colab_with_claude_api.ipynb
```

## Example Conversation Flow

**You**: *[Upload evaluation.log]* "Colab evaluation failed at batch 150/2400"

**Claude**: "I see a CUDA OOM error. Your batch size of 64 is too large for the T4's 15GB VRAM with your 1M+ parameter model.

Reduce batch size in `evaluate_gpu.py` line 142:
```python
batch_size = 32  # Changed from 64
```

This will fit in memory. Re-run from the checkpoint."

**You**: *[Applies fix, continues]*

**You**: *[Upload evaluation_results.csv]* "Evaluation complete. Check promotion gates."

**Claude**: "Analyzing results:

BTC-USD: 71.2% accuracy, 3.8% calibration ✅ PASS
ETH-USD: 69.5% accuracy, 4.2% calibration ✅ PASS
SOL-USD: 66.8% accuracy, 5.3% calibration ❌ FAIL (below 68% accuracy)

**Recommendations**:
1. Promote BTC and ETH models immediately
2. Retrain SOL with adjusted hyperparameters
3. Check SOL feature distributions for anomalies

Next steps: [detailed plan]"

## Summary

**Claude Max gives you**:
- Persistent project context
- Unlimited file uploads
- Conversation history
- No API costs

**Recommended setup**:
1. Create Project (5 min)
2. Upload documentation
3. Run Colab manually
4. Share results/errors with Project
5. Get contextual fixes instantly

This is the **best option** for your use case - combines automation benefits with human oversight and costs nothing extra beyond your Claude Max subscription.

## Next Steps

1. **Create Project**: https://claude.ai/projects → "New Project"
2. **Upload files**: Use list above
3. **Set instructions**: Copy custom instructions above
4. **Start Colab eval**: Upload files and run
5. **Report back**: Share any outputs with the Project

Ready to proceed?

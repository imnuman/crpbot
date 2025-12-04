# Configuration Files

**Status**: IMPLEMENTED

---

## Files in This Folder

### 1. `.env.example`
Template for environment variables

**Required Variables**:
- `DEEPSEEK_API_KEY` - Gladiator A
- `ANTHROPIC_API_KEY` - Gladiator B
- `XAI_API_KEY` - Gladiator C (Grok)
- `GOOGLE_API_KEY` - Gladiator D (Gemini)
- `COINBASE_API_KEY_NAME` - Market data
- `COINBASE_API_PRIVATE_KEY` - Market data auth

**Optional Variables**:
- `TELEGRAM_TOKEN` - Notifications
- `TELEGRAM_CHAT_ID` - Alert destination

**Usage**:
```bash
# Copy template to .env
cp .env.example .env

# Edit with your real API keys
nano .env

# NEVER commit .env to GitHub (it's in .gitignore)
```

---

### 2. `requirements_hydra.txt`
Python package dependencies

**Installation**:
```bash
# Using uv (recommended)
uv pip install -r requirements_hydra.txt

# Using standard pip
pip install -r requirements_hydra.txt
```

**Key Dependencies**:
- `anthropic` - Claude API
- `openai` - DeepSeek & Grok APIs (OpenAI-compatible)
- `google-generativeai` - Gemini API
- `coinbase-advanced-py` - Market data
- `reflex` - Dashboard framework
- `loguru` - Logging
- `python-telegram-bot` - Notifications

---

## Setup Instructions

### 1. Get API Keys

**DeepSeek** (Gladiator A):
- Website: https://platform.deepseek.com
- Cost: ~$0.27 per 1M tokens
- Free tier: Yes

**Claude** (Gladiator B):
- Website: https://console.anthropic.com
- Cost: $3 per 1M tokens (Claude 3.5 Sonnet)
- Free tier: Limited

**Grok** (Gladiator C):
- Website: https://x.ai/api
- Cost: ~$5 per 1M tokens
- Free tier: Unknown (check current offering)

**Gemini** (Gladiator D):
- Website: https://ai.google.dev
- Cost: Free tier available
- Paid: ~$0.35 per 1M tokens (Gemini 1.5 Pro)

**Coinbase** (Market Data):
- Website: https://www.coinbase.com/cloud
- Free tier: Yes (sufficient for HYDRA)
- Setup: Create API key in Coinbase Developer Portal

**Telegram** (Notifications):
- Create bot: Talk to @BotFather on Telegram
- Get token: BotFather provides after creation
- Get chat ID: Send message to bot, then visit:
  `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`

---

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install packages
pip install -r validation/6_config/requirements_hydra.txt
```

---

### 3. Configure Environment

```bash
# Copy template
cp validation/6_config/.env.example .env

# Edit with real keys
nano .env
```

**Verify Configuration**:
```bash
# Check all required keys are set
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

required = [
    'DEEPSEEK_API_KEY',
    'ANTHROPIC_API_KEY',
    'XAI_API_KEY',
    'GOOGLE_API_KEY',
    'COINBASE_API_KEY_NAME',
    'COINBASE_API_PRIVATE_KEY'
]

missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f'❌ Missing keys: {missing}')
else:
    print('✅ All required keys configured')
"
```

---

### 4. Test HYDRA

```bash
# Test single iteration (paper trading)
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets BTC-USD \
  --iterations 1 \
  --interval 10 \
  --paper

# Expected output:
# - All 4 gladiators respond
# - Consensus vote calculated
# - Paper trade logged to data/hydra/paper_trades.jsonl
```

---

## Cost Estimates

**Monthly Cost** (assuming 288 iterations/day @ 5-min intervals):

| Provider | Cost per 1M tokens | Avg tokens/vote | Daily cost | Monthly cost |
|----------|-------------------|-----------------|------------|--------------|
| DeepSeek | $0.27 | 1,500 | $0.12 | $3.50 |
| Claude | $3.00 | 1,500 | $1.30 | $39.00 |
| Grok | $5.00 | 1,500 | $2.16 | $65.00 |
| Gemini | $0.35 | 1,500 | $0.15 | $4.50 |
| **Total** | - | - | **$3.73** | **$112.00** |

**Notes**:
- Costs assume 1,500 tokens per gladiator vote
- Actual usage may vary (500-3,000 tokens)
- Gemini has free tier (15 RPM limit)
- DeepSeek is cheapest
- Claude is most expensive but highest quality

---

## Security Best Practices

1. **Never commit `.env` file**
   - Add to `.gitignore`
   - Use `.env.example` for templates

2. **Rotate API keys regularly**
   - Every 90 days minimum
   - Immediately if compromised

3. **Use environment-specific configs**
   - `.env.dev` for local testing
   - `.env.prod` for production
   - Different API keys per environment

4. **Limit API key permissions**
   - Coinbase: Read-only for market data
   - LLM providers: Restrict rate limits
   - Telegram: One bot per environment

5. **Monitor API usage**
   - Set billing alerts
   - Track costs daily
   - Guardian monitors for credit exhaustion

---

**Date**: 2025-11-30

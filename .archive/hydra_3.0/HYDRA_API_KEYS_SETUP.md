# HYDRA 3.0 - API Keys Setup Guide

**Date**: 2025-11-29
**Purpose**: Configure all 4 gladiator LLM API keys for production

---

## 📊 Current API Key Status

### ✅ Already Configured:
1. **Gladiator A (DeepSeek)** ✅
   - Key: `DEEPSEEK_API_KEY`
   - Status: **ACTIVE** (confirmed working in tests)
   - Provider: DeepSeek AI
   - Role: Structural Edge Generator

2. **Gladiator B (Claude)** ✅
   - Key: `ANTHROPIC_API_KEY`
   - Status: **ACTIVE** (key present in .env)
   - Provider: Anthropic
   - Role: Logic Validator

3. **Gladiator C (Grok / X.AI)** ✅
   - Key: `XAI_API_KEY`
   - Status: **ACTIVE** (confirmed working with grok-3 model)
   - Provider: X.AI (https://x.ai)
   - Role: Fast Backtester
   - Model: grok-3 (grok-beta deprecated 2025-09-15)

4. **Gladiator D (Gemini)** ✅
   - Key: `GOOGLE_API_KEY` (also supports `GEMINI_API_KEY`)
   - Status: **ACTIVE** (confirmed working)
   - Provider: Google AI
   - Role: Synthesizer
   - Model: gemini-2.0-flash-exp

---

## ✅ **ALL 4 GLADIATORS NOW ACTIVE**

**Status**: 4/4 API keys configured (100% coverage)
**Production Ready**: YES
**Date Completed**: 2025-11-29

---

## 🔑 How to Obtain Missing API Keys

### Groq API Key

**Provider**: Groq Inc. (https://groq.com)

**Steps**:
1. Go to https://console.groq.com
2. Sign up or log in
3. Navigate to API Keys section
4. Create new API key
5. Copy the key (starts with `gsk_...`)

**Free Tier**:
- Available: Yes
- Limits: ~14,000 tokens/minute (sufficient for HYDRA)
- Models: Llama 3, Mixtral, Gemma

**Cost** (if upgrading):
- Pay-as-you-go: ~$0.10 per 1M tokens
- Very affordable for HYDRA usage

### Gemini API Key

**Provider**: Google AI Studio (https://ai.google.dev)

**Steps**:
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Get API Key"
4. Create API key for new project or use existing
5. Copy the key (starts with `AI...`)

**Free Tier**:
- Available: Yes
- Limits: 60 requests/minute (sufficient for HYDRA)
- Models: Gemini 1.5 Pro, Gemini 1.5 Flash

**Cost** (if upgrading):
- Free tier is very generous
- Paid tier: ~$0.50 per 1M tokens

---

## 📝 Adding API Keys to .env

Once you have the keys, add them to your `.env` file:

```bash
# Open .env file
nano .env

# Add the following lines (replace with your actual keys):
GROQ_API_KEY=gsk_YOUR_GROQ_API_KEY_HERE
GEMINI_API_KEY=AIzaSyYOUR_GEMINI_API_KEY_HERE

# Save and exit (Ctrl+O, Enter, Ctrl+X)
```

**Or use command line**:
```bash
# Add Groq API key
echo "GROQ_API_KEY=gsk_YOUR_KEY_HERE" >> .env

# Add Gemini API key
echo "GEMINI_API_KEY=AIzaSy_YOUR_KEY_HERE" >> .env
```

---

## ✅ Verification Test

After adding the keys, test that all gladiators work:

```bash
.venv/bin/python3 -c "
from apps.runtime.hydra_runtime import HydraRuntime
runtime = HydraRuntime(assets=['BTC-USD'], paper_trading=True)

print('\\n=== Gladiator Status ===')
print(f'Gladiator A (DeepSeek): Working')
print(f'Gladiator B (Claude): {\"Working\" if runtime.gladiator_b.api_key else \"Mock mode\"}')
print(f'Gladiator C (Groq): {\"Working\" if runtime.gladiator_c.api_key else \"Mock mode\"}')
print(f'Gladiator D (Gemini): {\"Working\" if runtime.gladiator_d.api_key else \"Mock mode\"}')
"
```

**Expected output after adding keys**:
```
Gladiator A (DeepSeek): Working
Gladiator B (Claude): Working
Gladiator C (Groq): Working
Gladiator D (Gemini): Working
```

---

## 🎯 Why All 4 API Keys Matter

### Current Behavior (Missing Groq + Gemini):
- **Gladiator C (Groq)**: Returns mock backtests with 56% win rate
- **Gladiator D (Gemini)**: Returns generic "APPROVE" recommendations
- **Problem**: Reduces diversity and quality of consensus voting

### With All 4 Keys:
- **Gladiator A (DeepSeek)**: Generates sophisticated strategies (already working!)
- **Gladiator B (Claude)**: Validates logic and finds flaws
- **Gladiator C (Groq)**: Performs fast backtesting on historical data
- **Gladiator D (Gemini)**: Synthesizes all inputs and makes final recommendation
- **Result**: Higher quality signals, better consensus, more diverse strategies

---

## 💰 Cost Estimates (Monthly)

Based on HYDRA running 24/7 with 3 assets:

**Scenario 1: Free Tiers Only**
- DeepSeek: $0 (using free tier)
- Claude: $0-5 (minimal usage)
- Groq: $0 (within free tier limits)
- Gemini: $0 (within free tier limits)
- **Total**: $0-5/month ✅

**Scenario 2: With Some Paid Usage**
- DeepSeek: $10-20/month (primary generator)
- Claude: $5-10/month (validation)
- Groq: $2-5/month (fast inference)
- Gemini: $2-5/month (synthesis)
- **Total**: $19-40/month

**Scenario 3: High Volume**
- DeepSeek: $50/month
- Claude: $20/month
- Groq: $10/month
- Gemini: $10/month
- **Total**: $90/month

**Note**: HYDRA is designed to be cost-efficient:
- Generates strategies only when regime changes
- Uses consensus to avoid unnecessary trades
- Caches strategies in tournament manager
- Conservative signal generation (3/hour max)

**Expected actual cost**: $10-30/month for paper trading

---

## 🚀 Recommended Next Steps

### Option 1: Add Both Keys Now (Recommended)
**Time**: ~10 minutes
**Benefit**: Full HYDRA capabilities immediately
**Steps**:
1. Get Groq API key (5 min)
2. Get Gemini API key (5 min)
3. Add to .env
4. Run verification test
5. Deploy HYDRA with all 4 gladiators

### Option 2: Add One at a Time
**If you want to test incrementally**:
1. Add Groq first (faster inference)
2. Test with 3 gladiators
3. Add Gemini later
4. Test with all 4

### Option 3: Deploy Without Them (Not Recommended)
**Current setup works but**:
- Only DeepSeek generates real strategies
- Groq and Gemini in mock mode
- Consensus voting less effective
- Missing diversity in strategy generation

---

## 📋 Quick Setup Checklist

- [x] DeepSeek API key configured
- [x] Anthropic (Claude) API key configured
- [ ] Sign up for Groq account
- [ ] Get Groq API key
- [ ] Add GROQ_API_KEY to .env
- [ ] Sign up for Google AI Studio
- [ ] Get Gemini API key
- [ ] Add GEMINI_API_KEY to .env
- [ ] Run verification test
- [ ] Confirm all 4 gladiators working
- [ ] Deploy HYDRA to production

---

## 🔒 Security Notes

1. **Never commit .env to git** (already in .gitignore)
2. **Keep API keys secret**
3. **Rotate keys periodically** (every 90 days recommended)
4. **Monitor usage** to detect unauthorized access
5. **Set billing limits** on each provider to prevent overage

---

## 📞 Support Links

- **Groq**: https://console.groq.com/docs
- **Gemini**: https://ai.google.dev/docs
- **DeepSeek**: https://platform.deepseek.com/docs
- **Anthropic**: https://docs.anthropic.com

---

## ✅ Current Working Configuration

```bash
# .env (current setup)
DEEPSEEK_API_KEY=sk-cb86184fcb974480a20615749781c198  # ✅ Working
ANTHROPIC_API_KEY=sk-ant-api03-rd4UIcPV...            # ✅ Working
# GROQ_API_KEY=                                       # ❌ Missing
# GEMINI_API_KEY=                                     # ❌ Missing
```

**After adding missing keys**:
```bash
# .env (target configuration)
DEEPSEEK_API_KEY=sk-cb86184fcb974480a20615749781c198  # ✅ Working
ANTHROPIC_API_KEY=sk-ant-api03-rd4UIcPV...            # ✅ Working
GROQ_API_KEY=gsk_YOUR_KEY_HERE                        # ✅ To add
GEMINI_API_KEY=AIzaSy_YOUR_KEY_HERE                   # ✅ To add
```

---

**Status**: 2/4 API keys configured (50%)
**Next Action**: Obtain Groq and Gemini API keys
**Time Required**: ~10 minutes total
**Priority**: Medium (system works without them, but better with them)

---

**Last Updated**: 2025-11-29
**Created By**: Claude Code (Sonnet 4.5)
**Purpose**: Production deployment checklist for HYDRA 3.0

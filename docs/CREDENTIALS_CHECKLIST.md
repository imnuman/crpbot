# Credentials & Services Checklist

## 🔴 Required for Phase 2 (Data Pipeline)

### 1. Binance API Keys ⚠️ **CRITICAL**
**Purpose**: Fetch 1m candle data for BTC/ETH/BNB (2020-present)

**Where to get**:
- Go to: https://www.binance.com/en/my/settings/api-management
- Create a new API key
- **Important**: Enable "Read" permissions only (no trading needed)
- Save both `API Key` and `Secret Key`

**What you need**:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here
```

**Note**: 
- Free tier is sufficient for historical data
- Rate limits: 1200 requests/minute (should be plenty for 1m candles)
- No trading permissions needed (read-only is safer)

---

### 2. FTMO Account Credentials ⚠️ **CRITICAL**
**Purpose**: Measure real spreads/slippage for execution model (Phase 2.3) and enable micro-lot trading (Phase 7).

**Setup Guide**: See [`docs/FTMO_SETUP.md`](../docs/FTMO_SETUP.md) for a full walkthrough (demo & challenge).

**Where to get**:
- Create an FTMO **Demo** first (free) for execution metrics
- Purchase the FTMO **Challenge** before Phase 7
- Retrieve credentials from the FTMO client dashboard

**What you need**:
```
FTMO_LOGIN=your_ftmo_login
FTMO_PASS=your_ftmo_password
FTMO_SERVER=FTMO-Demo  # or FTMO-Server for live/challenge
```

**Verification Steps**:
- Run `scripts/nightly_exec_metrics.py --once` after updating `.env`
- Confirm `data/execution_metrics/*.json` updates with fresh measurements

**Notes**:
- Protect these credentials (FTMO does not offer read-only logins)
- Nightly cron (`infra/scripts/nightly_exec_metrics.sh`) relies on the same `.env`
- Until credentials are ready you may use mock spreads, but Phase 6.5+ requires real data

---

### 3. Database (PostgreSQL or SQLite) ✅ **EASY**

**Option A: SQLite (Recommended for Phase 2)**
- **No setup needed** - SQLite is built into Python
- Just set in `.env`:
  ```
  DB_URL=sqlite:///tradingai.db
  ```
- Perfect for development and testing
- We'll migrate to PostgreSQL later if needed

**Option B: PostgreSQL (Production-ready)**
- Install PostgreSQL locally or use a cloud service (AWS RDS, Supabase, etc.)
- Set in `.env`:
  ```
  DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
  ```

**Recommendation**: Start with SQLite for Phase 2, switch to PostgreSQL later.

---

## 🟡 Optional for Phase 2 (Can add later)

### 4. Telegram Bot Token (Optional now, needed for Phase 4)
**Purpose**: Runtime notifications and commands

**Where to get**:
- Message @BotFather on Telegram
- `/newbot` command
- Follow instructions to create a bot
- Get the token

**What you need** (can add later):
```
TELEGRAM_TOKEN=8425324139:AAGXmo2h3_4xTbkMW-TiASELOlWtMryN5ho
TELEGRAM_CHAT_ID=718556632
```

**Note**: Not needed until Phase 4 (Runtime), but good to set up now if you have time.

---

### 5. AWS Credentials (Optional - for DVC S3)
**Purpose**: Store data/models in S3 via DVC (version control for large files)

**Where to get**:
- AWS account (free tier available)
- Create S3 bucket
- Create IAM user with S3 access
- Generate access keys

**What you need** (optional):
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

**Note**: 
- Not required for Phase 2
- You can use Git LFS or local storage initially
- Can add DVC + S3 later when you have more data

---

## 📋 Phase 2 Minimum Requirements

### Must Have (Start Phase 2):
1. ✅ **Binance API Keys** - Required for data collection
2. ✅ **Database** - SQLite is fine (default)

### Can Add Later:
3. ⏳ **FTMO Credentials** - Can start with mock spreads, add real measurement later
4. ⏳ **Telegram Bot** - Not needed until Phase 4
5. ⏳ **AWS Credentials** - Not needed until you want DVC S3 storage

---

## 🚀 Quick Start Setup

### Step 1: Get Binance API Keys
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create API key with "Read" permissions
3. Save API Key and Secret Key

### Step 2: Create .env File
```bash
cd /home/numan/crpbot
cp .env.example .env
nano .env  # or use your preferred editor
```

### Step 3: Fill in .env
```bash
# Required for Phase 2
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here

# Database (SQLite is fine - no setup needed)
DB_URL=sqlite:///tradingai.db

# FTMO (can add later or use mock)
# FTMO_LOGIN=
# FTMO_PASS=
# FTMO_SERVER=
```

### Step 4: Test Configuration
```bash
# Activate venv
source .venv/bin/activate

# Test config loads
python -c "from libs.config.config import Settings; s = Settings(); print('✅ Config loaded')"
```

---

## 🔒 Security Notes

1. **Never commit `.env` file** - It's already in `.gitignore`
2. **Use read-only API keys when possible** - Binance API should be read-only
3. **Store credentials securely** - Consider using a password manager
4. **Rotate keys regularly** - Especially if you share the repo
5. **Use separate accounts** - If testing, use demo accounts

---

## 📝 Summary

**Minimum to start Phase 2**:
- ✅ Binance API Key + Secret (5 minutes to get)
- ✅ SQLite database (no setup - default)

**Phase 2 will work with**:
- Binance data collection ✅
- SQLite database ✅
- Mock FTMO spreads (can add real FTMO later)

**Can add later**:
- FTMO credentials (for real spread measurement)
- Telegram bot (for Phase 4)
- PostgreSQL (if you want more than SQLite)
- AWS S3 (if you want DVC remote storage)

---

## ✅ Ready to Start Phase 2?

Once you have:
1. Binance API keys set in `.env`
2. `.env` file created from `.env.example`

You're ready to start Phase 2! 🚀


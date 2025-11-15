# 🚀 START HERE - Builder Claude Week 1 Quick Start

**Created**: 2025-11-15 15:40 EST (Toronto)
**To**: Builder Claude (Cloud Server)
**From**: QC Claude (Local Machine)
**Priority**: 🔴 HIGH - Start Immediately
**Estimated Time**: 6-8 hours

---

## 📍 You Are Here

**Current Phase**: V5 Week 1 - Data Download & Validation
**Your Environment**: Cloud server (`~/crpbot`)
**Status**: All prerequisites complete ✅
**Goal**: Download 2 years of OHLCV data from CoinGecko for BTC/ETH/SOL

---

## ⚡ Quick Context (30 seconds)

**What Changed**:
- ❌ OLD: Tardis.dev at $98/month (WRONG - it's $300-350+)
- ✅ NEW: CoinGecko Analyst at $129/month (correct pricing)
- Budget: $154/month Phase 1 (under $200 ✅)

**Your Mission**:
Download 2 years of high-quality OHLCV data from CoinGecko to train models that achieve 65-75% accuracy (vs current 50%).

---

## 🎯 Week 1 Tasks (This Week)

### Task 1: Sync & Configure (15 min) 🔴 DO THIS FIRST

```bash
# 1. Navigate to project
cd ~/crpbot

# 2. Pull latest changes
git pull origin main

# 3. Verify you have latest commits
git log --oneline -5
# Should see:
#   d7e981e docs: add handoff summary for Builder Claude
#   7c5e71e docs: add Builder Claude V5 execution instructions
#   8187b84 fix: correct V5 data provider pricing

# 4. Add CoinGecko API key to .env
echo 'COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW' >> .env

# 5. Verify it's there
grep COINGECKO .env
# Should output: COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW

# 6. Test configuration
python -c "from libs.config.config import load_settings; s = load_settings(); print(f'✅ CoinGecko API: {s.coingecko_api_key[:10]}...')"
```

**Expected Output**: `✅ CoinGecko API: CG-VQhq64e...`

---

### Task 2: Create CoinGecko Data Fetcher (2-3 hours)

Create `scripts/fetch_coingecko_data.py`:

```python
#!/usr/bin/env python3
"""
Fetch historical OHLCV data from CoinGecko API.

Usage:
    python scripts/fetch_coingecko_data.py --symbol BTC-USD --days 730
    python scripts/fetch_coingecko_data.py --symbol ETH-USD --days 730
    python scripts/fetch_coingecko_data.py --symbol SOL-USD --days 730
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# CoinGecko API configuration
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

# Coin ID mapping (CoinGecko uses different IDs)
COIN_IDS = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}

# VS currency (always USD for our use case)
VS_CURRENCY = 'usd'


def fetch_ohlcv_data(coin_id: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from CoinGecko.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana')
        days: Number of days of history (max: 365 for free tier, 730+ for paid)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    logger.info(f"Fetching {days} days of data for {coin_id}")

    # CoinGecko OHLC endpoint
    # Ref: https://docs.coingecko.com/reference/coins-id-ohlc
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'vs_currency': VS_CURRENCY,
        'days': days,
        'precision': 'full'  # Full precision for prices
    }

    try:
        logger.debug(f"Request: GET {url}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if not data:
            logger.error(f"No data returned for {coin_id}")
            return pd.DataFrame()

        # CoinGecko OHLC format: [[timestamp, open, high, low, close], ...]
        # Note: Volumes are in a separate endpoint
        logger.info(f"Received {len(data)} candles")

        # Parse into DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Fetch volume data separately (CoinGecko stores it in market_chart)
        logger.info("Fetching volume data...")
        df = fetch_volume_data(coin_id, df, days)

        # Sort by timestamp (oldest first)
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✅ Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"✅ Total candles: {len(df)}")

        return df

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.error("Rate limit exceeded. Wait 60 seconds...")
            logger.error("CoinGecko Analyst tier: 50 calls/minute")
        else:
            logger.error(f"HTTP error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise


def fetch_volume_data(coin_id: str, ohlc_df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Fetch volume data and merge with OHLC data.

    CoinGecko stores volume in a separate endpoint (market_chart/range).
    """
    # Calculate time range from OHLC data
    start_timestamp = int(ohlc_df['timestamp'].min().timestamp())
    end_timestamp = int(ohlc_df['timestamp'].max().timestamp())

    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range"

    headers = {
        'accept': 'application/json',
        'x-cg-pro-api-key': COINGECKO_API_KEY
    }

    params = {
        'vs_currency': VS_CURRENCY,
        'from': start_timestamp,
        'to': end_timestamp
    }

    try:
        # Rate limiting: 50 calls/min = 1.2 seconds per call
        time.sleep(1.2)

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract volume data: [[timestamp, volume], ...]
        volumes = data.get('total_volumes', [])

        if volumes:
            vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms', utc=True)

            # Merge volume with OHLC (match by timestamp)
            # Use nearest timestamp matching since granularity might differ
            merged_df = pd.merge_asof(
                ohlc_df.sort_values('timestamp'),
                vol_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )

            logger.info(f"✅ Merged {len(volumes)} volume data points")
            return merged_df
        else:
            logger.warning("No volume data available, filling with 0")
            ohlc_df['volume'] = 0.0
            return ohlc_df

    except Exception as e:
        logger.warning(f"Failed to fetch volume data: {e}")
        logger.warning("Continuing with volume = 0")
        ohlc_df['volume'] = 0.0
        return ohlc_df


def save_to_parquet(df: pd.DataFrame, symbol: str):
    """Save DataFrame to parquet file."""
    output_dir = Path('data/raw/coingecko')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename format: SYMBOL_1m_YYYYMMDD.parquet
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f'{symbol}_1m_{date_str}.parquet'
    output_path = output_dir / filename

    # Save with compression
    df.to_parquet(output_path, index=False, compression='gzip')

    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Size: {file_size:.2f} MB")
    logger.info(f"   Rows: {len(df):,}")
    logger.info(f"   Columns: {list(df.columns)}")

    return output_path


def validate_data(df: pd.DataFrame, symbol: str):
    """Validate data quality."""
    logger.info(f"\n📊 Data Quality Report for {symbol}")
    logger.info("=" * 60)

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"   {col}: {count} missing ({count/len(df)*100:.2f}%)")
    else:
        logger.info("✅ No missing values")

    # Check data range
    logger.info(f"\n📅 Date Range:")
    logger.info(f"   Start: {df['timestamp'].min()}")
    logger.info(f"   End:   {df['timestamp'].max()}")
    logger.info(f"   Days:  {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Check price statistics
    logger.info(f"\n💰 Price Statistics ({symbol}):")
    logger.info(f"   Min:  ${df['low'].min():,.2f}")
    logger.info(f"   Max:  ${df['high'].max():,.2f}")
    logger.info(f"   Mean: ${df['close'].mean():,.2f}")

    # Check volume statistics
    logger.info(f"\n📊 Volume Statistics:")
    logger.info(f"   Total: {df['volume'].sum():,.2f}")
    logger.info(f"   Mean:  {df['volume'].mean():,.2f}")
    logger.info(f"   Max:   {df['volume'].max():,.2f}")

    # Check for anomalies
    logger.info(f"\n🔍 Quality Checks:")

    # Check for zero prices
    zero_prices = (df['close'] == 0).sum()
    if zero_prices > 0:
        logger.warning(f"⚠️  {zero_prices} candles with zero close price")
    else:
        logger.info("✅ No zero prices")

    # Check for negative volumes
    negative_vol = (df['volume'] < 0).sum()
    if negative_vol > 0:
        logger.warning(f"⚠️  {negative_vol} candles with negative volume")
    else:
        logger.info("✅ No negative volumes")

    # Check OHLC consistency
    ohlc_errors = (
        (df['high'] < df['low']) |
        (df['close'] > df['high']) |
        (df['close'] < df['low']) |
        (df['open'] > df['high']) |
        (df['open'] < df['low'])
    ).sum()

    if ohlc_errors > 0:
        logger.warning(f"⚠️  {ohlc_errors} candles with OHLC inconsistencies")
    else:
        logger.info("✅ OHLC data consistent")

    logger.info("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fetch historical OHLCV data from CoinGecko'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        help='Trading pair symbol'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Number of days of history (default: 730 = 2 years)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: fetch only 7 days'
    )

    args = parser.parse_args()

    # Verify API key
    if not COINGECKO_API_KEY:
        logger.error("❌ COINGECKO_API_KEY not found in environment")
        logger.error("Please set it in .env file")
        sys.exit(1)

    logger.info(f"🚀 CoinGecko Data Fetcher - {args.symbol}")
    logger.info(f"API Key: {COINGECKO_API_KEY[:10]}...")

    # Map symbol to CoinGecko ID
    coin_id = COIN_IDS.get(args.symbol)
    if not coin_id:
        logger.error(f"Unknown symbol: {args.symbol}")
        sys.exit(1)

    # Use test mode if requested
    days = 7 if args.test else args.days

    try:
        # Fetch data
        df = fetch_ohlcv_data(coin_id, days=days)

        if df.empty:
            logger.error("No data fetched")
            sys.exit(1)

        # Validate data
        validate_data(df, args.symbol)

        # Save to parquet
        output_path = save_to_parquet(df, args.symbol)

        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"Data saved to: {output_path}")
        logger.info(f"Next: Run for other symbols (ETH-USD, SOL-USD)")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
```

**Save as**: `scripts/fetch_coingecko_data.py`

**Make executable**:
```bash
chmod +x scripts/fetch_coingecko_data.py
```

---

### Task 3: Test the Fetcher (30 min)

First, test with 7 days of data:

```bash
# Test with BTC (7 days)
python scripts/fetch_coingecko_data.py --symbol BTC-USD --test

# Check output
ls -lh data/raw/coingecko/
# Should see: BTC-USD_1m_20251115.parquet (~1-2 MB)

# Verify data quality
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/coingecko/BTC-USD_1m_20251115.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(df.head())
print(df.tail())
"
```

**Expected**: Should see ~10,080 rows (7 days × 24 hours × 60 minutes)

---

### Task 4: Download Full Dataset (1-2 hours)

Once test passes, download 2 years for all symbols:

```bash
# BTC-USD (2 years)
python scripts/fetch_coingecko_data.py --symbol BTC-USD --days 730

# Wait 2 minutes (rate limiting)
sleep 120

# ETH-USD (2 years)
python scripts/fetch_coingecko_data.py --symbol ETH-USD --days 730

# Wait 2 minutes
sleep 120

# SOL-USD (2 years)
python scripts/fetch_coingecko_data.py --symbol SOL-USD --days 730
```

**Expected output**:
```
data/raw/coingecko/
├── BTC-USD_1m_20251115.parquet   (~30-50 MB)
├── ETH-USD_1m_20251115.parquet   (~30-50 MB)
└── SOL-USD_1m_20251115.parquet   (~20-40 MB)
```

---

### Task 5: Create Week 1 Progress Report (1 hour)

Create `WEEK1_PROGRESS_2025-11-15.md`:

```markdown
# Week 1 Progress Report - V5 Data Download

**Date**: 2025-11-15
**Author**: Builder Claude (Cloud Server)
**Status**: Complete / In Progress / Blocked

---

## ✅ Completed Tasks

- [ ] Git synced with latest changes (commit d7e981e)
- [ ] CoinGecko API key configured in .env
- [ ] Data fetcher script created (scripts/fetch_coingecko_data.py)
- [ ] Test run successful (7 days of BTC data)
- [ ] BTC-USD downloaded (730 days)
- [ ] ETH-USD downloaded (730 days)
- [ ] SOL-USD downloaded (730 days)

---

## 📊 Data Download Results

### BTC-USD
- File: data/raw/coingecko/BTC-USD_1m_20251115.parquet
- Size: XX MB
- Rows: XXX,XXX
- Date range: YYYY-MM-DD to YYYY-MM-DD
- Quality: ✅ / ⚠️  (describe issues if any)

### ETH-USD
- File: data/raw/coingecko/ETH-USD_1m_20251115.parquet
- Size: XX MB
- Rows: XXX,XXX
- Date range: YYYY-MM-DD to YYYY-MM-DD
- Quality: ✅ / ⚠️

### SOL-USD
- File: data/raw/coingecko/SOL-USD_1m_20251115.parquet
- Size: XX MB
- Rows: XXX,XXX
- Date range: YYYY-MM-DD to YYYY-MM-DD
- Quality: ✅ / ⚠️

---

## 🔍 Data Quality Comparison

### CoinGecko vs Free Coinbase
(Compare quality metrics)

| Metric | CoinGecko | Coinbase Free | Improvement |
|--------|-----------|---------------|-------------|
| Missing values | X% | Y% | +Z% |
| Price consistency | Good/Fair/Poor | Good/Fair/Poor | Better/Same/Worse |
| Volume accuracy | Good/Fair/Poor | Good/Fair/Poor | Better/Same/Worse |

---

## 🚧 Blockers / Issues

- None / List any issues encountered

---

## 📅 Next Steps (Week 2)

- Feature engineering for 40-50 features
- Multi-timeframe feature creation
- Baseline model testing

---

**Time Spent**: X hours
**Status**: On track / Behind / Ahead
**Next Update**: 2025-11-XX
```

---

## 📋 Success Checklist

By end of today, you should have:

- [ ] Git synced (commit `d7e981e` present)
- [ ] CoinGecko API key in `.env`
- [ ] `scripts/fetch_coingecko_data.py` created
- [ ] Test run successful (7 days BTC)
- [ ] 3 parquet files in `data/raw/coingecko/`
- [ ] Week 1 progress report created

---

## 🆘 If You Get Stuck

### Issue: API Key Not Working
```bash
# Verify API key is set
echo $COINGECKO_API_KEY

# Test manually
curl -X GET "https://api.coingecko.com/api/v3/ping" \
  -H "x-cg-pro-api-key: CG-VQhq64e59sGxchtK8mRgdxXW"
```

### Issue: Rate Limit Errors (429)
- CoinGecko Analyst: 50 calls/minute
- Add `time.sleep(1.2)` between requests
- Script already includes this

### Issue: No Data Returned
- Check coin ID mapping (bitcoin, ethereum, solana)
- Verify `days` parameter (max 365 for free, 730+ for paid)
- Check CoinGecko API status: https://status.coingecko.com/

---

## 📞 Need Help?

Create a file and commit:
```bash
# Create question/blocker file
cat > BLOCKER_coingecko_data_2025-11-15.md << 'EOF'
# Blocker: [Brief Description]

**Date**: 2025-11-15
**Severity**: High / Medium / Low

## Issue
[Describe the problem]

## What I Tried
1. [Attempt 1]
2. [Attempt 2]

## Error Messages
```
[Paste error here]
```

## Need Help With
[Specific question for QC Claude]
EOF

# Commit and push
git add BLOCKER_*.md
git commit -m "blocker: [brief description]"
git push origin main
```

QC Claude will review and respond.

---

## 📚 Reference Documents

**Full Instructions**: `BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md`
**Handoff Summary**: `HANDOFF_TO_BUILDER_CLAUDE.md`
**V5 Strategy**: `V5_SIMPLE_PLAN.md`
**Budget Details**: `V5_BUDGET_PLAN.md`

---

## 🎯 Key Points to Remember

1. **CoinGecko API**: Use for training data ($129/month)
2. **Coinbase API**: Use for runtime data (FREE)
3. **Rate Limits**: 50 calls/minute (add 1.2s sleep)
4. **Data Format**: OHLCV in parquet with gzip compression
5. **Quality First**: Validate data before moving to Week 2

---

## ⏰ Time Estimates

| Task | Estimated | Your Time |
|------|-----------|-----------|
| Sync & Configure | 15 min | ___ |
| Create Fetcher | 2-3 hours | ___ |
| Test Fetcher | 30 min | ___ |
| Download Full Data | 1-2 hours | ___ |
| Progress Report | 1 hour | ___ |
| **Total** | **5-7 hours** | ___ |

---

## 🎉 You're Ready!

Everything is set up for success:
- ✅ CoinGecko API configured
- ✅ Coinbase real-time tested
- ✅ Budget approved ($154/month)
- ✅ All documentation updated

**Start with Task 1 and work through sequentially.**

**Good luck! 🚀**

---

**File**: `START_HERE_BUILDER_CLAUDE.md`
**Created**: 2025-11-15 15:40 EST
**Status**: Ready to execute
**Next**: Begin Task 1 (Sync & Configure)

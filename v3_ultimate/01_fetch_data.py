#!/usr/bin/env python3
"""
V3 Ultimate - Step 1: Data Collection
Fetch 5 years of OHLCV data for 10 coins across 6 timeframes.

Expected output: ~50M candles in 60 parquet files
Runtime: ~12 hours on Colab Pro+

Requirements:
- pip install ccxt pandas pyarrow tqdm
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
from tqdm import tqdm
import json

# Configuration
COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
         'ADA/USDT', 'XRP/USDT', 'MATIC/USDT', 'AVAX/USDT',
         'DOGE/USDT', 'DOT/USDT']

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

START_DATE = '2020-01-01'
END_DATE = '2025-11-12'

OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/data/raw')

# Timeframe to milliseconds
TIMEFRAME_MS = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
}

def init_exchange():
    """Initialize Bybit exchange."""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
    return exchange

def fetch_ohlcv_batch(exchange, symbol, timeframe, start_ts, end_ts):
    """Fetch OHLCV data in batches."""
    all_candles = []
    current_ts = start_ts

    limit = 1000  # Bybit limit per request

    pbar = tqdm(desc=f"{symbol} {timeframe}", unit=" candles")

    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=limit
            )

            if not candles:
                break

            all_candles.extend(candles)
            pbar.update(len(candles))

            # Move to next batch
            last_ts = candles[-1][0]
            if last_ts <= current_ts:
                # No progress, break to avoid infinite loop
                break
            current_ts = last_ts + TIMEFRAME_MS[timeframe]

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"\n⚠️  Error fetching {symbol} {timeframe} at {current_ts}: {e}")
            time.sleep(5)  # Wait before retry
            continue

    pbar.close()
    return all_candles

def save_to_parquet(candles, output_path):
    """Convert and save OHLCV candles to parquet."""
    if not candles:
        print(f"⚠️  No candles to save for {output_path}")
        return 0

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression='snappy', index=False)

    return len(df)

def fetch_coin_timeframe(exchange, coin, timeframe, start_ts, end_ts, output_dir):
    """Fetch data for one coin-timeframe pair."""
    symbol = coin
    coin_name = coin.replace('/', '_')

    output_file = output_dir / f"{coin_name}_{timeframe}.parquet"

    # Check if already exists
    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        print(f"✅ {coin_name} {timeframe}: Already exists with {len(existing_df):,} candles")
        return len(existing_df)

    print(f"\n📥 Fetching {symbol} {timeframe}...")

    candles = fetch_ohlcv_batch(exchange, symbol, timeframe, start_ts, end_ts)

    if candles:
        count = save_to_parquet(candles, output_file)
        print(f"✅ {coin_name} {timeframe}: Saved {count:,} candles")
        return count
    else:
        print(f"❌ {coin_name} {timeframe}: No data fetched")
        return 0

def main():
    """Main data collection workflow."""
    print("=" * 70)
    print("🚀 V3 ULTIMATE - STEP 1: DATA COLLECTION")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Coins: {len(COINS)}")
    print(f"   Timeframes: {len(TIMEFRAMES)}")
    print(f"   Date Range: {START_DATE} to {END_DATE}")
    print(f"   Total Files: {len(COINS) * len(TIMEFRAMES)}")
    print(f"   Output: {OUTPUT_DIR}")

    # Parse dates
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    print(f"\n📅 Fetching from {start_dt} to {end_dt}")
    print(f"   Duration: {(end_dt - start_dt).days} days")

    # Initialize exchange
    print(f"\n🔌 Connecting to Bybit...")
    exchange = init_exchange()
    print(f"   ✅ Connected")

    # Fetch data
    results = []
    start_time = datetime.now()

    for coin in COINS:
        for timeframe in TIMEFRAMES:
            try:
                count = fetch_coin_timeframe(
                    exchange=exchange,
                    coin=coin,
                    timeframe=timeframe,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    output_dir=OUTPUT_DIR
                )

                results.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'candles': count,
                    'status': 'success' if count > 0 else 'empty'
                })

            except Exception as e:
                print(f"\n❌ Failed to fetch {coin} {timeframe}: {e}")
                results.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'candles': 0,
                    'status': 'failed',
                    'error': str(e)
                })
                continue

    duration = (datetime.now() - start_time).total_seconds()

    # Summary
    print("\n" + "=" * 70)
    print("📊 DATA COLLECTION SUMMARY")
    print("=" * 70)

    total_candles = sum(r['candles'] for r in results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\n✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"📊 Total Candles: {total_candles:,}")
    print(f"⏱️  Duration: {duration/3600:.1f} hours")

    # Breakdown by coin
    print(f"\n📈 Breakdown by Coin:")
    coin_summary = {}
    for r in results:
        coin = r['coin']
        if coin not in coin_summary:
            coin_summary[coin] = {'candles': 0, 'files': 0}
        coin_summary[coin]['candles'] += r['candles']
        if r['status'] == 'success':
            coin_summary[coin]['files'] += 1

    for coin, stats in coin_summary.items():
        print(f"   {coin}: {stats['candles']:,} candles across {stats['files']} timeframes")

    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'start_date': START_DATE,
        'end_date': END_DATE,
        'coins': COINS,
        'timeframes': TIMEFRAMES,
        'total_candles': total_candles,
        'duration_seconds': duration,
        'results': results
    }

    manifest_path = OUTPUT_DIR / 'data_collection_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n💾 Manifest saved: {manifest_path}")

    # Validation
    expected_min_candles = 50_000_000  # 50M candles

    print(f"\n🎯 Validation:")
    if total_candles >= expected_min_candles:
        print(f"   ✅ PASS: {total_candles:,} candles (≥{expected_min_candles:,})")
    else:
        print(f"   ⚠️  WARNING: {total_candles:,} candles (<{expected_min_candles:,})")

    if failed == 0:
        print(f"   ✅ PASS: All files fetched successfully")
    else:
        print(f"   ⚠️  WARNING: {failed} files failed")

    print(f"\n✅ Step 1 Complete! Ready for Step 2: Feature Engineering")

    return manifest

if __name__ == "__main__":
    main()

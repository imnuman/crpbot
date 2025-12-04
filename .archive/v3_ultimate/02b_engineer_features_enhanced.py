#!/usr/bin/env python3
"""
V3 Ultimate - Step 2B: Enhanced Feature Engineering
Merges base features with alternative data (sentiment, liquidations, orderbook).
Results in 335 total features ready for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_FEATURES_DIR = Path('/content/drive/MyDrive/crpbot/data/features')
ALT_DATA_DIR = Path('/content/drive/MyDrive/crpbot/data/alternative')
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/data/features_enhanced')

COINS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT',
         'ADA_USDT', 'XRP_USDT', 'MATIC_USDT', 'AVAX_USDT',
         'DOGE_USDT', 'DOT_USDT']

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

def merge_alternative_data(base_df, coin, alt_data_dir):
    """Merge sentiment, liquidation, and orderbook data with base features."""
    coin_name = coin.replace('_USDT', '')
    
    # Load Reddit sentiment
    sentiment_file = alt_data_dir / f"{coin_name}_reddit_sentiment.parquet"
    if sentiment_file.exists():
        sentiment_df = pd.read_parquet(sentiment_file)
        base_df = pd.merge_asof(
            base_df.sort_values('timestamp'),
            sentiment_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('1H')
        )
        print(f"   ✅ Merged {len(sentiment_df.columns)} sentiment features")
    else:
        print(f"   ⚠️  No sentiment data for {coin_name}, using zeros")
        for col in ['reddit_sent_mean', 'reddit_sent_4h', 'reddit_sent_24h', 'reddit_sent_divergence']:
            base_df[col] = 0
    
    # Load liquidations
    liq_file = alt_data_dir / f"{coin_name}_liquidations.parquet"
    if liq_file.exists():
        liq_df = pd.read_parquet(liq_file)
        base_df = pd.merge_asof(
            base_df.sort_values('timestamp'),
            liq_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('1H')
        )
        print(f"   ✅ Merged {len(liq_df.columns)} liquidation features")
    else:
        print(f"   ⚠️  No liquidation data for {coin_name}, using zeros")
        for col in ['liq_total_usd', 'liq_imbalance', 'liq_total_4h', 'liq_cluster']:
            base_df[col] = 0
    
    # Add orderbook features (if available - static or most recent)
    ob_file = alt_data_dir / f"{coin_name}_orderbook_sample.parquet"
    if ob_file.exists():
        ob_df = pd.read_parquet(ob_file)
        # Use most recent orderbook snapshot for all rows
        if len(ob_df) > 0:
            latest_ob = ob_df.iloc[-1]
            for col in ['bid_ask_spread', 'bid_ask_imbalance', 'depth_1pct', 'depth_2pct', 'depth_5pct']:
                if col in latest_ob:
                    base_df[col] = latest_ob[col]
        print(f"   ✅ Added orderbook features")
    else:
        print(f"   ⚠️  No orderbook data for {coin_name}, using defaults")
        base_df['bid_ask_spread'] = 0.001
        base_df['bid_ask_imbalance'] = 0
        base_df['depth_1pct'] = 1000000
        base_df['depth_2pct'] = 2000000
        base_df['depth_5pct'] = 5000000
    
    return base_df

def main():
    """Merge all data sources."""
    print("=" * 70)
    print("🔧 V3 ULTIMATE - ENHANCED FEATURE ENGINEERING")
    print("=" * 70)
    print(f"\nMerging:")
    print(f"   • Base features (~252)")
    print(f"   • Reddit sentiment (~30)")
    print(f"   • Liquidations (~18)")
    print(f"   • Orderbook (~20)")
    print(f"   Total: ~335 features")
    
    results = []
    
    for coin in COINS:
        for timeframe in TIMEFRAMES:
            print(f"\n📊 Processing {coin} {timeframe}...")
            
            # Load base features
            base_file = BASE_FEATURES_DIR / f"{coin}_{timeframe}_features.parquet"
            if not base_file.exists():
                print(f"   ❌ Base features not found: {base_file}")
                continue
            
            df = pd.read_parquet(base_file)
            initial_cols = len(df.columns)
            initial_rows = len(df)
            
            print(f"   Base: {initial_rows:,} rows, {initial_cols} columns")
            
            # Merge alternative data
            df = merge_alternative_data(df, coin, ALT_DATA_DIR)
            
            # Fill NaN
            df = df.fillna(0)
            
            final_cols = len(df.columns)
            final_rows = len(df)
            
            print(f"   Enhanced: {final_rows:,} rows, {final_cols} columns (+{final_cols-initial_cols})")
            
            # Save
            output_file = OUTPUT_DIR / f"{coin}_{timeframe}_features_enhanced.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file, compression='snappy', index=False)
            
            print(f"   ✅ Saved: {output_file.name}")
            
            results.append({
                'coin': coin,
                'timeframe': timeframe,
                'rows': final_rows,
                'base_features': initial_cols,
                'enhanced_features': final_cols,
                'added_features': final_cols - initial_cols
            })
    
    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'results': results
    }
    
    manifest_path = OUTPUT_DIR / 'enhanced_features_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 ENHANCED FEATURE ENGINEERING COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal files: {len(results)}")
    print(f"Average features: {np.mean([r['enhanced_features'] for r in results]):.0f}")
    print(f"\n✅ Ready for Step 3B: Enhanced Training")

if __name__ == "__main__":
    main()

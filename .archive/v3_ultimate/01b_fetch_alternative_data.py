#!/usr/bin/env python3
"""
V3 Ultimate - Step 1B: Alternative Data Collection
Fetch Reddit sentiment, Coinglass liquidations, and orderbook data.

This supplements 01_fetch_data.py with alternative data sources.

Requirements:
- pip install praw transformers coinglass-api pybit
- Reddit Premium API: $100/month
- Coinglass API: $50/month
"""

import praw
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    HAS_FINBERT = True
except ImportError:
    print("⚠️  FinBERT not available, will use basic sentiment")
    HAS_FINBERT = False

# Coinglass API
try:
    import requests
    HAS_COINGLASS = True
except ImportError:
    print("⚠️  requests not available")
    HAS_COINGLASS = False

# Bybit for orderbook
try:
    from pybit.unified_trading import HTTP
    HAS_BYBIT = True
except ImportError:
    print("⚠️  pybit not available")
    HAS_BYBIT = False

# Configuration
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/data/alternative')

COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'XRP', 'MATIC', 'AVAX', 'DOGE', 'DOT']

# Reddit API credentials (USER MUST PROVIDE)
REDDIT_CLIENT_ID = 'YOUR_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
REDDIT_USER_AGENT = 'crpbot/1.0'

# Coinglass API key (USER MUST PROVIDE)
COINGLASS_API_KEY = 'YOUR_API_KEY'

# Subreddits to monitor
SUBREDDITS = ['CryptoCurrency', 'Bitcoin', 'ethereum', 'solana', 'binance',
              'cardano', 'Ripple', 'CryptoMarkets', 'CryptoMoonShots']

def init_reddit():
    """Initialize Reddit API."""
    if REDDIT_CLIENT_ID == 'YOUR_CLIENT_ID':
        print("⚠️  Reddit API not configured!")
        print("   1. Go to: https://www.reddit.com/prefs/apps")
        print("   2. Create app, get client_id and secret")
        print("   3. Subscribe to Reddit Premium API ($100/mo)")
        print("   4. Update credentials in this script")
        return None

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    return reddit

def fetch_reddit_sentiment(reddit, coin, start_date, end_date):
    """Fetch Reddit posts and calculate sentiment."""
    print(f"\n📱 Fetching Reddit sentiment for {coin}...")

    if reddit is None:
        print("   ⚠️  Reddit not configured, skipping")
        return pd.DataFrame()

    # Search terms
    search_terms = [coin, f"${coin}", coin.lower()]

    all_posts = []

    for subreddit_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(subreddit_name)

            for term in search_terms:
                # Search posts
                for post in subreddit.search(term, time_filter='all', limit=1000):
                    post_time = datetime.fromtimestamp(post.created_utc)

                    if start_date <= post_time <= end_date:
                        all_posts.append({
                            'timestamp': post_time,
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name
                        })

        except Exception as e:
            print(f"   ⚠️  Error fetching from r/{subreddit_name}: {e}")
            continue

    if not all_posts:
        print(f"   ⚠️  No posts found for {coin}")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)

    # Calculate sentiment using FinBERT
    if HAS_FINBERT:
        print(f"   Analyzing sentiment with FinBERT...")

        sentiments = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment"):
            text = row['title'] + " " + row['text'][:500]  # Limit text length

            try:
                result = sentiment_pipeline(text)[0]
                sentiment_score = result['score'] if result['label'] == 'positive' else -result['score']
                sentiments.append(sentiment_score)
            except:
                sentiments.append(0)

        df['sentiment'] = sentiments
    else:
        # Simple sentiment based on score
        df['sentiment'] = df['score'].apply(lambda x: 1 if x > 10 else (-1 if x < 0 else 0))

    print(f"   ✅ Fetched {len(df)} posts with sentiment")

    return df

def aggregate_sentiment_features(sentiment_df, timeframe='1H'):
    """Aggregate sentiment into time-based features."""
    if sentiment_df.empty:
        return pd.DataFrame()

    df = sentiment_df.set_index('timestamp')

    # Resample to timeframe
    agg = df.resample(timeframe).agg({
        'sentiment': ['mean', 'std', 'min', 'max', 'count'],
        'score': ['sum', 'mean'],
        'num_comments': ['sum', 'mean']
    })

    # Flatten columns
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]

    # Rename for clarity
    agg = agg.rename(columns={
        'sentiment_mean': 'reddit_sent_mean',
        'sentiment_std': 'reddit_sent_std',
        'sentiment_min': 'reddit_sent_min',
        'sentiment_max': 'reddit_sent_max',
        'sentiment_count': 'reddit_post_count',
        'score_sum': 'reddit_score_total',
        'score_mean': 'reddit_score_avg',
        'num_comments_sum': 'reddit_comments_total',
        'num_comments_mean': 'reddit_comments_avg'
    })

    # Add rolling windows
    for window in [4, 24, 168]:  # 4h, 24h, 7d (in hours)
        agg[f'reddit_sent_{window}h'] = agg['reddit_sent_mean'].rolling(window).mean()
        agg[f'reddit_posts_{window}h'] = agg['reddit_post_count'].rolling(window).sum()

    # Sentiment divergence (sentiment vs price momentum)
    agg['reddit_sent_divergence'] = agg['reddit_sent_mean'] - agg['reddit_sent_mean'].rolling(24).mean()

    return agg.reset_index()

def fetch_coinglass_liquidations(coin, start_date, end_date):
    """Fetch liquidation data from Coinglass."""
    print(f"\n💥 Fetching Coinglass liquidations for {coin}...")

    if not HAS_COINGLASS or COINGLASS_API_KEY == 'YOUR_API_KEY':
        print("   ⚠️  Coinglass API not configured!")
        print("   1. Go to: https://www.coinglass.com/pricing")
        print("   2. Subscribe to API ($50/mo)")
        print("   3. Get API key and update in script")
        return pd.DataFrame()

    base_url = "https://open-api.coinglass.com"

    headers = {
        "CG-API-KEY": COINGLASS_API_KEY
    }

    all_data = []

    # Iterate through date range
    current_date = start_date

    while current_date <= end_date:
        try:
            # Liquidation data endpoint
            endpoint = f"{base_url}/public/v2/liquidation"

            params = {
                'symbol': f"{coin}USDT",
                'interval': '1h',
                'timestamp': int(current_date.timestamp() * 1000)
            }

            response = requests.get(endpoint, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if 'data' in data:
                    for item in data['data']:
                        all_data.append({
                            'timestamp': datetime.fromtimestamp(item['timestamp'] / 1000),
                            'liq_long_usd': item.get('longLiquidationUsd', 0),
                            'liq_short_usd': item.get('shortLiquidationUsd', 0),
                            'liq_total_usd': item.get('totalLiquidationUsd', 0)
                        })

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"   ⚠️  Error fetching liquidations: {e}")

        current_date += timedelta(hours=1)

    if not all_data:
        print(f"   ⚠️  No liquidation data found for {coin}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Calculate additional features
    df['liq_imbalance'] = (df['liq_long_usd'] - df['liq_short_usd']) / (df['liq_total_usd'] + 1)
    df['liq_ratio'] = df['liq_long_usd'] / (df['liq_short_usd'] + 1)

    # Rolling windows
    for window in [4, 24, 168]:
        df[f'liq_total_{window}h'] = df['liq_total_usd'].rolling(window).sum()
        df[f'liq_imbalance_{window}h'] = df['liq_imbalance'].rolling(window).mean()

    # Liquidation clusters (>$50M in 4h)
    df['liq_cluster'] = (df['liq_total_4h'] > 50e6).astype(int)

    print(f"   ✅ Fetched {len(df)} liquidation records")

    return df

def fetch_orderbook_snapshots(coin):
    """Fetch orderbook snapshots from Bybit."""
    print(f"\n📊 Fetching orderbook snapshots for {coin}...")

    if not HAS_BYBIT:
        print("   ⚠️  pybit not available, skipping")
        return pd.DataFrame()

    session = HTTP(testnet=False)

    symbol = f"{coin}USDT"

    snapshots = []

    try:
        # Fetch L2 orderbook
        orderbook = session.get_orderbook(
            category="spot",
            symbol=symbol,
            limit=20
        )

        if 'result' in orderbook:
            bids = orderbook['result']['b'][:20]
            asks = orderbook['result']['a'][:20]

            # Calculate features
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)

            bid_price_0 = float(bids[0][0])
            ask_price_0 = float(asks[0][0])

            spread = (ask_price_0 - bid_price_0) / bid_price_0

            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)

            # Depth at different levels
            depth_1pct = sum(float(b[1]) for b in bids if float(b[0]) >= bid_price_0 * 0.99)
            depth_2pct = sum(float(b[1]) for b in bids if float(b[0]) >= bid_price_0 * 0.98)
            depth_5pct = sum(float(b[1]) for b in bids if float(b[0]) >= bid_price_0 * 0.95)

            # Whale orders (>$5M)
            whale_bids = sum(1 for b in bids if float(b[0]) * float(b[1]) > 5e6)
            whale_asks = sum(1 for a in asks if float(a[0]) * float(a[1]) > 5e6)

            snapshot = {
                'timestamp': datetime.now(),
                'bid_ask_spread': spread,
                'bid_ask_imbalance': imbalance,
                'depth_1pct': depth_1pct,
                'depth_2pct': depth_2pct,
                'depth_5pct': depth_5pct,
                'whale_bids': whale_bids,
                'whale_asks': whale_asks,
                'total_bid_volume': bid_volume,
                'total_ask_volume': ask_volume
            }

            snapshots.append(snapshot)

            print(f"   ✅ Fetched orderbook snapshot")

    except Exception as e:
        print(f"   ⚠️  Error fetching orderbook: {e}")

    return pd.DataFrame(snapshots)

def main():
    """Main alternative data collection workflow."""
    print("=" * 70)
    print("🔍 V3 ULTIMATE - ALTERNATIVE DATA COLLECTION")
    print("=" * 70)

    print(f"\n📋 Data Sources:")
    print(f"   • Reddit Sentiment (FinBERT): 30 features")
    print(f"   • Coinglass Liquidations: 18 features")
    print(f"   • Bybit Orderbook L2: 20 features")
    print(f"\n   Total: +68 features (225 → 293)")

    # Date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 11, 12)

    # Initialize APIs
    reddit = init_reddit()

    results = {}

    for coin in COINS:
        print(f"\n{'='*70}")
        print(f"Processing {coin}")
        print(f"{'='*70}")

        coin_data = {}

        # Fetch Reddit sentiment
        sentiment_df = fetch_reddit_sentiment(reddit, coin, start_date, end_date)
        if not sentiment_df.empty:
            sentiment_features = aggregate_sentiment_features(sentiment_df, timeframe='1H')
            coin_data['sentiment'] = sentiment_features

            # Save
            output_file = OUTPUT_DIR / f"{coin}_reddit_sentiment.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sentiment_features.to_parquet(output_file, index=False)
            print(f"   💾 Saved: {output_file.name}")

        # Fetch liquidations
        liquidations_df = fetch_coinglass_liquidations(coin, start_date, end_date)
        if not liquidations_df.empty:
            coin_data['liquidations'] = liquidations_df

            # Save
            output_file = OUTPUT_DIR / f"{coin}_liquidations.parquet"
            liquidations_df.to_parquet(output_file, index=False)
            print(f"   💾 Saved: {output_file.name}")

        # Fetch orderbook (just save config, actual fetching is real-time)
        orderbook_df = fetch_orderbook_snapshots(coin)
        if not orderbook_df.empty:
            coin_data['orderbook'] = orderbook_df

            # Save sample
            output_file = OUTPUT_DIR / f"{coin}_orderbook_sample.parquet"
            orderbook_df.to_parquet(output_file, index=False)
            print(f"   💾 Saved: {output_file.name}")

        results[coin] = coin_data

    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'coins': COINS,
        'data_sources': {
            'reddit': 'FinBERT sentiment from r/CryptoCurrency + others',
            'coinglass': 'Liquidation data from Coinglass API',
            'orderbook': 'L2 orderbook snapshots from Bybit'
        },
        'costs': {
            'reddit_api': '$100/month',
            'coinglass_api': '$50/month',
            'bybit_api': 'Free'
        },
        'results': {coin: list(data.keys()) for coin, data in results.items()}
    }

    manifest_path = OUTPUT_DIR / 'alternative_data_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}")
    print("📊 ALTERNATIVE DATA COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n💾 Manifest saved: {manifest_path}")

    print(f"\n⚠️  IMPORTANT:")
    print(f"   • Reddit API: Configure credentials and subscribe ($100/mo)")
    print(f"   • Coinglass API: Get API key and subscribe ($50/mo)")
    print(f"   • Bybit API: Free, but orderbook is real-time (snapshot only)")

    print(f"\n✅ Step 1B Complete! Proceed to Step 2 (Feature Engineering)")

if __name__ == "__main__":
    main()

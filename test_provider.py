#!/usr/bin/env python3
"""Test data provider."""
from libs.data.provider import create_data_provider
from datetime import datetime, timedelta, timezone

# Create YFinance provider (free, no auth)
provider = create_data_provider('yfinance')

# Test connection
if provider.test_connection():
    print('✅ Connection successful!')

    # Fetch small sample (last 100 candles)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    df = provider.fetch_klines('BTC-USD', '1h', start_time=start_time, limit=100)
    print(f'✅ Fetched {len(df)} candles')
    print(f'\nSample data:')
    print(df.head(10))
    print(f'\nDate range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'Price range: ${df["close"].min():.2f} - ${df["close"].max():.2f}')
else:
    print('❌ Connection failed')

import ccxt

kraken = ccxt.kraken({'enableRateLimit': True})
ohlcv = kraken.fetch_ohlcv('BTC/USD', '1h', limit=10)
print(f"✅ Fetched {len(ohlcv)} candles from Kraken public API")
print(f"Latest close: ${ohlcv[-1][4]:,.2f}")

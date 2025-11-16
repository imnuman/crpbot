"""Test Coinbase API authentication with official SDK."""
from coinbase.rest import RESTClient
from libs.config.config import Settings

# Load credentials
config = Settings()

print("=" * 70)
print("Testing Coinbase API Authentication")
print("=" * 70)
print(f"API Key: {config.effective_api_key}")
print(f"Private Key Loaded: {'YES' if config.effective_api_secret else 'NO'}")
print()

# Test with official SDK
try:
    client = RESTClient(
        api_key=config.effective_api_key,
        api_secret=config.effective_api_secret
    )

    print("Testing: Getting BTC-USD candles...")
    candles = client.get_candles(
        product_id="BTC-USD",
        start=None,
        end=None,
        granularity="ONE_MINUTE"
    )

    print(f"✅ SUCCESS! Received {len(candles.candles)} candles")
    print(f"Latest candle: {candles.candles[0] if candles.candles else 'N/A'}")
    print()
    print("🎉 API credentials are VALID and working!")

except Exception as e:
    print(f"❌ FAILED: {e}")
    print()
    print("This suggests the API credentials are invalid or don't have the right permissions.")
    print("Please regenerate the API key with 'View' permissions at:")
    print("https://portal.cloud.coinbase.com/access/api")

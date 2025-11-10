#!/usr/bin/env python3
"""Diagnostic script to test Coinbase API credentials."""
from libs.config.config import Settings
from libs.data.coinbase import CoinbaseDataProvider


def main():
    """Test Coinbase API connection with detailed diagnostics."""
    config = Settings()

    print("🔍 Coinbase API Diagnostics\n")
    print(f"Provider: {config.data_provider}")
    print(f"API Key Name present: {bool(config.coinbase_api_key_name)}")
    print(f"Private Key present: {bool(config.coinbase_api_private_key)}")
    print(
        f"API Key Name length: {len(config.coinbase_api_key_name) if config.coinbase_api_key_name else 0}"
    )
    print(
        f"Private Key length: {len(config.coinbase_api_private_key) if config.coinbase_api_private_key else 0}"
    )

    print("\n📋 Testing API connection...\n")

    try:
        provider = CoinbaseDataProvider(
            api_key_name=config.coinbase_api_key_name or config.effective_api_key,
            private_key=config.coinbase_api_private_key or config.effective_api_secret,
        )

        # Test connection
        if provider.test_connection():
            print("✅ Connection successful!")
            symbols = provider.get_available_symbols()
            print(f"✅ Found {len(symbols)} available symbols")
            if symbols:
                print(f"   Examples: {symbols[:5]}")
        else:
            print("❌ Connection test failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Verify you created an Advanced Trade API key (not Exchange/Pro)")
        print("2. Check that both credentials are correct:")
        print("   - API Key Name (full path: organizations/.../apiKeys/...)")
        print("   - Private Key (PEM format starting with -----BEGIN EC PRIVATE KEY-----)")
        print("3. Ensure API key has 'View' permissions")
        print("4. Check IP whitelisting if enabled")
        print("5. Try creating a new API key if issues persist")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic script to verify Coinbase API credentials and configuration."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from libs.config.config import Settings
from libs.data.coinbase import CoinbaseDataProvider
import time
import base64

def main():
    print("🔍 Coinbase API Credential Diagnostics\n")
    
    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return
    
    # Use settings values (which load from .env via pydantic-settings)
    print("📋 Credentials from Settings (loaded from .env):")
    api_key_name = settings.coinbase_api_key_name or ""
    private_key = settings.coinbase_api_private_key or ""
    
    print(f"  COINBASE_API_KEY_NAME: {'✅ Present' if api_key_name else '❌ Missing'} (length: {len(api_key_name)})")
    print(f"  COINBASE_API_PRIVATE_KEY: {'✅ Present' if private_key else '❌ Missing'} (length: {len(private_key)})")
    if api_key_name:
        print(f"  API Key Name: {api_key_name[:50]}...")
    print()
    
    # Check settings
    print("📋 Settings Object:")
    print(f"  API Key Name: {'✅ Present' if settings.coinbase_api_key_name else '❌ Missing'} (length: {len(settings.coinbase_api_key_name) if settings.coinbase_api_key_name else 0})")
    print(f"  Private Key: {'✅ Present' if settings.coinbase_api_private_key else '❌ Missing'} (length: {len(settings.coinbase_api_private_key) if settings.coinbase_api_private_key else 0})")
    print()
    
    # Validate private key format
    print("🔍 Validating Private Key Format:")
    if not private_key:
        print("  ❌ Private Key is empty!")
    else:
        if "-----BEGIN EC PRIVATE KEY-----" in private_key:
            print("  ✅ Private Key appears to be PEM-encoded EC private key")
        else:
            print("  ⚠️  Private Key format may be incorrect (expected PEM format)")
    print()
    
    # Check system time
    print("🕐 System Time Check:")
    system_time = int(time.time())
    print(f"  Current Unix timestamp: {system_time}")
    print(f"  Current time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(system_time))}")
    print("  ⚠️  Make sure your system clock is synchronized (Coinbase API is time-sensitive)")
    print()
    
    # Try to create provider
    print("🔧 Creating Data Provider:")
    try:
        provider = CoinbaseDataProvider(
            api_key_name=api_key_name or settings.coinbase_api_key_name,
            private_key=private_key or settings.coinbase_api_private_key,
        )
        print("  ✅ Provider created successfully (JWT authentication)")
    except Exception as e:
        print(f"  ❌ Failed to create provider: {e}")
        return
    print()
    
    # Test connection
    print("📡 Testing API Connection:")
    try:
        # This should trigger a request
        symbols = provider.get_available_symbols()
        if symbols:
            print(f"  ✅ Connection successful! Found {len(symbols)} symbols")
            print(f"  Sample symbols: {symbols[:5]}")
        else:
            print("  ❌ Connection failed - no symbols returned (check error logs above)")
            print("  ⚠️  This usually means authentication failed (401 Unauthorized)")
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print()
        print("💡 Troubleshooting Steps:")
        print("  1. Verify API key is Advanced Trade API (not Exchange API)")
        print("  2. Check API key has 'View' permissions enabled in Coinbase dashboard")
        print("  3. Check if IP whitelisting is enabled (if so, add your IP)")
        print("  4. Verify system clock is synchronized")
        print("  5. If you just created the key, wait a few minutes for activation")
        print("  6. Double-check all credentials are correct in .env file")
        return
    
    print("\n✅ All checks passed! Coinbase API is working correctly.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Helper script to verify .env configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"❌ .env file not found at {env_path}")
    print("   Please create it from .env.example: cp .env.example .env")
    exit(1)

# Check required variables
print("\n📋 Checking environment variables:\n")

required_vars = {
    "Coinbase": [
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "COINBASE_API_PASSPHRASE",
    ],
    "Data Provider": [
        "DATA_PROVIDER",
    ],
}

all_good = True
for category, vars_list in required_vars.items():
    print(f"{category}:")
    for var in vars_list:
        value = os.getenv(var, "")
        if value:
            # Mask sensitive values
            if "SECRET" in var or "KEY" in var or "PASSPHRASE" in var:
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                print(f"  ✅ {var} = {masked}")
            else:
                print(f"  ✅ {var} = {value}")
        else:
            print(f"  ❌ {var} = (not set)")
            all_good = False
    print()

if all_good:
    print("✅ All required variables are set!")
    print("\nTesting config loading...")
    from libs.config.config import Settings

    s = Settings()
    print("✅ Config loaded successfully")
    print(f"   Provider: {s.data_provider}")
    print(f"   Has API Key: {bool(s.effective_api_key)}")
    print(f"   Has API Secret: {bool(s.effective_api_secret)}")
    print(f"   Has Passphrase: {bool(s.effective_api_passphrase)}")
else:
    print("❌ Some required variables are missing!")
    print("\nPlease update your .env file with:")
    print("  COINBASE_API_KEY=your_key_here")
    print("  COINBASE_API_SECRET=your_secret_here")
    print("  COINBASE_API_PASSPHRASE=your_passphrase_here")
    print("  DATA_PROVIDER=coinbase")

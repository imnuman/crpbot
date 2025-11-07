# Coinbase Advanced Trade API - JWT Authentication Setup

## ✅ What Changed

The Coinbase Advanced Trade API uses **JWT authentication** (not HMAC-SHA256 with passphrase).

### Old API (Deprecated):
- API Key
- API Secret (base64)
- Passphrase

### New API (Current):
- **API Key Name** (full path: `organizations/.../apiKeys/...`)
- **Private Key** (PEM-encoded EC private key)

## 📝 Update Your .env File

Update your `.env` file with the new credential names:

```bash
# Coinbase Advanced Trade API (JWT Authentication)
COINBASE_API_KEY_NAME=organizations/b636b0e1-cbe3-4bab-8347-ea21f308b115/apiKeys/7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae
COINBASE_API_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIN1TmPTJ33bi...
...
-----END EC PRIVATE KEY-----"
```

**Important**: 
- Use the **full path** for `COINBASE_API_KEY_NAME` (not just the UUID)
- Keep the private key in **PEM format** (with `-----BEGIN` and `-----END` lines)
- The private key can span multiple lines

## 🧪 Test After Updating

Run:
```bash
python scripts/diagnose_coinbase.py
```

This should now work with JWT authentication!

## 🔍 Where to Find Your Credentials

1. Go to: https://www.coinbase.com/advanced-trade
2. Settings → API → Advanced Trade API
3. Click on your API key
4. Copy:
   - **API Key Name**: The full path shown
   - **Private Key**: The PEM-encoded private key (starts with `-----BEGIN EC PRIVATE KEY-----`)

## ✅ What's Fixed

- ✅ Updated authentication to use JWT (ES256 algorithm)
- ✅ Removed passphrase requirement
- ✅ Using full API key name path
- ✅ Support for PEM private key format
- ✅ Updated all scripts and configs


# Coinbase API Secret Format Issue

## ❌ Current Problem

Your `COINBASE_API_SECRET` appears to be a **PEM-encoded EC private key** instead of a **base64-encoded API secret**.

### What You Have:
```
-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----
```

### What You Need:
A long base64 string (200-300 characters), like:
```
aBcD1234eFgH5678iJkL9012mNoPqRsTuVwXyZaBcD1234eFgH5678iJkL9012mNoPqRsTuVwXyZ...
```

## 🔍 How to Fix

1. **Go to Coinbase Advanced Trade**:
   - https://www.coinbase.com/advanced-trade
   - Settings → API → Advanced Trade API

2. **Find Your API Key**:
   - Click on your API key to view details

3. **Copy the Correct Secret**:
   - Look for the **"Secret"** field
   - It should be a long base64 string (no `-----BEGIN` or `-----END` lines)
   - Copy it completely (usually 200-300 characters)

4. **Update `.env` File**:
   ```bash
   COINBASE_API_SECRET=your_base64_secret_here
   ```

## ⚠️ Common Mistakes

- **PEM private key** (what you have) - Used for ECDSA signatures, not HMAC
- **Exchange API secret** - Different API, different format
- **Incomplete copy** - Make sure you copied the entire secret

## ✅ Correct Format

The secret should:
- ✅ Be a single line (no line breaks)
- ✅ Be base64-encoded (characters: A-Z, a-z, 0-9, +, /, =)
- ✅ Be 200-300 characters long
- ✅ Decode to a non-zero byte string (usually 32-64 bytes)

## 🧪 Test After Fixing

Run:
```bash
python scripts/diagnose_coinbase.py
```

It should show:
- ✅ Secret is valid base64 (decoded length: X bytes) where X > 0
- ✅ Connection successful


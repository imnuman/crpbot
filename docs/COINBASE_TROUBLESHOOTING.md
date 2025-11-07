# Coinbase API Troubleshooting Guide

## Current Status

✅ **Credentials loaded**: All three credentials are present in .env
- API Key: 95 characters
- Secret: 228 characters  
- Passphrase: 8 characters

❌ **Authentication failing**: 401 Unauthorized

## Common Issues & Solutions

### Issue 1: Wrong API Type ⚠️ **MOST LIKELY**

**Problem**: You might have created an API key for **Coinbase Exchange/Pro** instead of **Coinbase Advanced Trade**.

**Solution**: 
1. Go to: https://www.coinbase.com/advanced-trade
2. Make sure you're in **Advanced Trade** (not Exchange/Pro)
3. Settings → API → **Advanced Trade API**
4. Create a new API key specifically for Advanced Trade
5. The key should work with endpoints like `/api/v3/brokerage/...`

**How to verify**: 
- Advanced Trade API keys typically start with "orga" or similar
- Advanced Trade uses `/api/v3/brokerage` endpoints
- Exchange/Pro uses different endpoints and authentication

---

### Issue 2: Incorrect Passphrase

**Problem**: The passphrase doesn't match what you set when creating the key.

**Solution**:
1. When creating the API key, Coinbase asks you to **set a passphrase**
2. This is something **YOU choose** (not auto-generated)
3. Make sure the passphrase in `.env` matches exactly what you set
4. It's case-sensitive and must match character-for-character

**If you forgot the passphrase**: You'll need to create a new API key.

---

### Issue 3: API Key Permissions

**Problem**: API key doesn't have the right permissions.

**Solution**:
1. Go to your Coinbase Advanced Trade API settings
2. Verify the API key has **"View"** permissions enabled
3. For data collection, you only need "View" (read-only)

---

### Issue 4: API Key Revoked/Expired

**Problem**: The API key was revoked or has expired.

**Solution**:
1. Check your Coinbase account → API settings
2. Verify the key is still active
3. If not, create a new one

---

## Quick Fix: Create New Advanced Trade API Key

1. **Go to**: https://www.coinbase.com/advanced-trade
2. **Navigate to**: Settings → API → **Advanced Trade API**
3. **Create new key**:
   - Set permissions to **"View"** only
   - **Remember the passphrase you set** (write it down!)
   - Save all three values immediately
4. **Update `.env`**:
   ```bash
   COINBASE_API_KEY=<new_key>
   COINBASE_API_SECRET=<new_secret>
   COINBASE_API_PASSPHRASE=<the_passphrase_you_just_set>
   ```
5. **Test again**:
   ```bash
   python scripts/test_coinbase_api.py
   ```

---

## Alternative: Use CryptoCompare (No Auth Issues)

If Coinbase continues to have issues, we can switch to CryptoCompare API:

1. **Sign up**: https://www.cryptocompare.com/crypto-api/
2. **Get free API key** (100,000 calls/day)
3. **Update `.env`**:
   ```bash
   DATA_PROVIDER=cryptocompare
   CRYPTOCOMPARE_API_KEY=your_free_key
   ```

No passphrase needed, simpler setup!

---

## Next Steps

1. **Verify API key type**: Make sure it's Advanced Trade, not Exchange/Pro
2. **Double-check passphrase**: Must match exactly what you set
3. **Test credentials**: Run `python scripts/test_coinbase_api.py`
4. **If still failing**: Consider switching to CryptoCompare for Phase 2


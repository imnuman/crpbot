# Coinbase Passphrase Required

## Current Issue

Even after adding your IP to the whitelist, we're still getting 401 Unauthorized.

The passphrase in `.env` is currently **empty**, but Coinbase Advanced Trade API likely **requires a passphrase**.

## Solution

### Option 1: Check Existing Passphrase

1. Go to: https://www.coinbase.com/advanced-trade
2. Settings → API → Advanced Trade API
3. Click on your API key
4. **Check if a passphrase is displayed**
5. If yes, copy it **exactly** and update `.env`:
   ```bash
   COINBASE_API_PASSPHRASE=the_exact_passphrase_from_coinbase
   ```

### Option 2: Create New API Key with Passphrase

If no passphrase is shown, create a new key:

1. Settings → API → **Create API Key**
2. Set permissions: **View** only
3. **IMPORTANT**: When prompted, **set a passphrase** (e.g., "mypass123")
4. **Save all three values immediately**:
   - API Key (UUID only, not full path)
   - API Secret (base64 string)
   - Passphrase (exactly as you set it)
5. Update `.env`:
   ```bash
   COINBASE_API_KEY=uuid_only
   COINBASE_API_SECRET=base64_secret
   COINBASE_API_PASSPHRASE=the_passphrase_you_set
   ```

## Test After Fixing

```bash
python scripts/diagnose_coinbase.py
```

## Alternative: CryptoCompare

If Coinbase continues to be problematic, we can switch to CryptoCompare:
- No passphrase needed
- Free tier: 100,000 calls/day
- Simpler authentication

Let me know if you'd like to switch!


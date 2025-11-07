# Coinbase API Key Verification Checklist

## 🔍 Verify Your API Key Setup

Since we're still getting 401 Unauthorized even with correct secret format, please verify:

### 1. API Key Type
- ✅ Make sure it's **Advanced Trade API** (not Exchange API)
- ✅ Go to: https://www.coinbase.com/advanced-trade
- ✅ Settings → API → **Advanced Trade API** (not Exchange API)

### 2. API Key Permissions
- ✅ Check that your API key has **"View"** permissions enabled
- ✅ If you need trading later, you can add "Trade" permissions

### 3. API Key Status
- ✅ Verify the key is **Active** (not disabled or expired)
- ✅ Check if IP whitelisting is enabled (if so, add your IP address)

### 4. Passphrase
- ⚠️ Some Coinbase API keys **require a passphrase** (cannot be empty)
- ⚠️ If you created the key **without** setting a passphrase, Coinbase may have auto-generated one
- 💡 **Try**: Go back to your API key page and check if there's a passphrase shown
- 💡 If no passphrase is shown, you may need to **create a new key** and explicitly set a passphrase

### 5. Create New API Key (If Needed)

If the current key doesn't work:

1. Go to Coinbase Advanced Trade → Settings → API
2. Click **"Create API Key"**
3. **Important**: When prompted for passphrase, **set one explicitly** (e.g., "mypassphrase123")
4. Copy all three values:
   - API Key
   - API Secret (base64 format, not PEM)
   - Passphrase (the one you just set)
5. Update `.env`:
   ```bash
   COINBASE_API_KEY=your_key_here
   COINBASE_API_SECRET=your_base64_secret_here
   COINBASE_API_PASSPHRASE=mypassphrase123
   ```

### 6. Test Again

After updating, run:
```bash
python scripts/diagnose_coinbase.py
```

## 🔄 Alternative: CryptoCompare

If Coinbase continues to be problematic, we can switch to CryptoCompare:
- ✅ No passphrase needed
- ✅ Free tier: 100,000 calls/day
- ✅ Simpler authentication
- ✅ Good for historical data

Let me know if you want to switch!


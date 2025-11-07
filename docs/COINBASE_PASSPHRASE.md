# Coinbase Passphrase - What to Do

## If You Didn't Set a Passphrase

When creating a Coinbase Advanced Trade API key, sometimes:
1. **Coinbase auto-generates a passphrase** and shows it on the creation screen
2. **The passphrase might be displayed** in the API key details page
3. **Some newer API keys** might not require a passphrase

## How to Find Your Passphrase

### Option 1: Check API Key Details Page
1. Go to: https://www.coinbase.com/advanced-trade
2. Settings → API → Advanced Trade API
3. Click on your API key to view details
4. Look for "Passphrase" - it might be displayed there

### Option 2: Check Your Notes/Records
- Did you copy anything when creating the key?
- Check your browser history or notes
- Sometimes Coinbase shows it once during creation

### Option 3: Try Without Passphrase
If you can't find it, we can try setting an empty passphrase:

In your `.env` file:
```bash
COINBASE_API_PASSPHRASE=
```

Then test:
```bash
python scripts/test_coinbase_api.py
```

## Alternative: Use CryptoCompare (Recommended)

If Coinbase continues to have issues, CryptoCompare is simpler:

1. **Sign up**: https://www.cryptocompare.com/crypto-api/ (free)
2. **Get API key** (no passphrase needed)
3. **Update `.env`**:
   ```bash
   DATA_PROVIDER=cryptocompare
   CRYPTOCOMPARE_API_KEY=your_free_key
   ```

This avoids all the passphrase complexity!


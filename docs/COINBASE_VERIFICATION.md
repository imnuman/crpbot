# Coinbase API Verification Checklist

## ✅ Quick Verification Steps

1. **Verify you're using the NEW API key** (the one you just created):
   - Check `.env` file has the new credentials
   - Old key might still be there if you didn't update it

2. **Verify API key format**:
   - Should start with "organizati" or similar
   - Length: ~95 characters
   - No spaces or line breaks

3. **Verify API Secret**:
   - Base64 format
   - Length: ~200-300 characters
   - No line breaks or special characters
   - Should decode as valid base64

4. **Verify Passphrase**:
   - Set to empty string: `COINBASE_API_PASSPHRASE=""`
   - Or leave completely empty: `COINBASE_API_PASSPHRASE=`

5. **Verify API Permissions**:
   - Go to Coinbase Advanced Trade → Settings → API
   - Check your API key has **"View"** permissions enabled

## 🔍 Testing Your Credentials

Run:
```bash
python scripts/test_coinbase_api.py
```

This will show:
- Which credentials are loaded
- Length of each credential
- Connection test result

## 🚨 Common Issues

### Issue: Still getting 401
**Possible causes**:
1. Using old API key credentials (not the new one)
2. API secret has formatting issues
3. API key permissions not set correctly
4. API key not activated yet (might need a few minutes)

### Solution:
1. **Double-check `.env`** has the NEW API key you just created
2. **Verify secret** is copied completely (no missing characters)
3. **Check permissions** in Coinbase dashboard
4. **Wait a few minutes** if you just created the key (sometimes activation is delayed)

## 💡 Alternative: CryptoCompare

If Coinbase continues to have issues, CryptoCompare is simpler:
- No passphrase needed
- Free tier: 100,000 calls/day
- Easy setup

Want to switch? Just let me know!


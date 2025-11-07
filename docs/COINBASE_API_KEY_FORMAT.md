# Coinbase API Key Format

## ✅ Correct Format

The Coinbase API key should be **just the UUID**, not the full path.

### Example:
```
✅ Correct: 7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae
❌ Wrong:   organizations/b636b0e1-cbe3-4bab-8347-ea21f308b115/apiKeys/7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae
```

## 🔍 How to Find It

When you view your API key in Coinbase, you might see the full path like:
```
organizations/b636b0e1-cbe3-4bab-8347-ea21f308b115/apiKeys/7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae
```

**Copy only the part after `/apiKeys/`**: `7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae`

## 📝 Update .env

Your `.env` should have:
```bash
COINBASE_API_KEY=7e4fabfa-e4ed-4772-b7bc-59d2c35e47ae
COINBASE_API_SECRET=your_base64_secret_here
COINBASE_API_PASSPHRASE=your_passphrase_or_empty
```

## ✅ Test After Fixing

Run:
```bash
python scripts/diagnose_coinbase.py
```

This should now work! The 401 error was likely because the full path was being used instead of just the UUID.


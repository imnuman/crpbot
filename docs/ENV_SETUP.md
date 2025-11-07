# .env File Setup Guide

## Quick Setup

1. **Create .env file** (if it doesn't exist):
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file** with your credentials:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Required variables for Coinbase**:
   ```bash
   DATA_PROVIDER=coinbase
   COINBASE_API_KEY=your_api_key_here
   COINBASE_API_SECRET=your_api_secret_here
   COINBASE_API_PASSPHRASE=your_passphrase_here
   ```

4. **Verify setup**:
   ```bash
   python scripts/check_env.py
   ```

## Getting Coinbase API Credentials

1. Go to: https://www.coinbase.com/advanced-trade
2. Sign in to your Coinbase account
3. Go to: **Settings** → **API** → **Advanced Trade API**
4. Click **"Create API Key"**
5. Set permissions to **"View"** (read-only is sufficient for data collection)
6. Save:
   - **API Key** (goes in `COINBASE_API_KEY`)
   - **Secret Key** (goes in `COINBASE_API_SECRET`)
   - **Passphrase** (you set this, goes in `COINBASE_API_PASSPHRASE`)

## Important Notes

- **Never commit .env to git** - It's already in `.gitignore`
- **Use read-only permissions** - "View" is sufficient for data collection
- **Keep credentials secure** - Don't share them

## Testing Your Setup

After filling in `.env`, run:
```bash
python scripts/check_env.py
```

This will verify:
- ✅ .env file exists
- ✅ All required variables are set
- ✅ Config can load the credentials

If everything is ✅, you're ready to fetch data!


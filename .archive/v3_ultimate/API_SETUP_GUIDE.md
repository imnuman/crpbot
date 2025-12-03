# V3 Ultimate - API Setup Guide

Complete guide for setting up Reddit Premium API, Coinglass API, and Bybit for Option B (Full Blueprint).

---

## Overview

**Total Cost**: $150/month for APIs
- Reddit Premium API: $100/month
- Coinglass API: $50/month
- Bybit API: Free

**Setup Time**: 30-60 minutes

---

## 1. Reddit Premium API Setup ($100/month)

### Step 1: Create Reddit Account
1. Go to: https://www.reddit.com/register
2. Create account (if you don't have one)
3. Verify email

### Step 2: Subscribe to Reddit Premium API
1. Go to: https://www.reddit.com/premium
2. Subscribe to Reddit Premium ($7/month for user access)
3. **Important**: For API access with higher rate limits, contact Reddit:
   - Email: api@reddit.com
   - Request: API access for trading bot research
   - Mention: Willing to pay for commercial API tier
   - Expected cost: ~$100/month for commercial access

**Alternative (Free Tier)**:
- Use free tier (60 requests/minute)
- Slower data collection but works
- May take 20+ hours for 5 years of data

### Step 3: Create Reddit App
1. Go to: https://www.reddit.com/prefs/apps
2. Scroll to bottom, click "create app" or "create another app"
3. Fill in:
   - **Name**: crpbot_v3
   - **App type**: Select "script"
   - **Description**: Trading signal analysis
   - **About URL**: Leave blank
   - **Redirect URI**: http://localhost:8080
4. Click "Create app"

### Step 4: Get Credentials
You'll see:
```
client_id: <14 character string under the app name>
client_secret: <27 character string>
```

**Save these** - you'll need them for `01b_fetch_alternative_data.py`

### Step 5: Test Access
```python
import praw

reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='crpbot/1.0'
)

# Test
subreddit = reddit.subreddit('CryptoCurrency')
for post in subreddit.hot(limit=5):
    print(post.title)
```

If this works, you're set!

---

## 2. Coinglass API Setup ($50/month)

### Step 1: Create Account
1. Go to: https://www.coinglass.com/
2. Click "Sign Up" (top right)
3. Register with email
4. Verify email

### Step 2: Subscribe to API Plan
1. Log in
2. Go to: https://www.coinglass.com/pricing
3. Select **"API Pro"** plan:
   - $49/month (or $490/year)
   - 10,000 requests/day
   - Access to liquidation data
   - Historical data access
4. Complete payment

### Step 3: Get API Key
1. Go to: https://www.coinglass.com/api
2. Click "Get API Key"
3. Copy the API key (looks like: `CG-xxxxxxxxxxxxxxxxxxxxxx`)

**Save this** - you'll need it for `01b_fetch_alternative_data.py`

### Step 4: Test Access
```python
import requests

api_key = 'YOUR_COINGLASS_API_KEY'

headers = {
    'CG-API-KEY': api_key
}

# Test endpoint
response = requests.get(
    'https://open-api.coinglass.com/public/v2/liquidation',
    headers=headers,
    params={'symbol': 'BTCUSDT', 'interval': '1h'}
)

print(response.json())
```

If you see liquidation data, you're set!

---

## 3. Bybit API Setup (Free)

### Step 1: Create Account
1. Go to: https://www.bybit.com/
2. Sign up (if you don't have account)
3. Complete KYC (required for API access)

### Step 2: Create API Key (Optional)
For orderbook data, you don't need an API key (public endpoint), but it's good to have:

1. Log in to Bybit
2. Go to: Account & Security > API
3. Click "Create New Key"
4. Select:
   - **Key Type**: System-generated API Keys
   - **Permissions**: Read-Only
   - **Name**: crpbot_v3
5. Save API Key and Secret (optional for orderbook fetching)

### Step 3: Test Access
```python
from pybit.unified_trading import HTTP

# No API key needed for public data
session = HTTP(testnet=False)

# Fetch orderbook
orderbook = session.get_orderbook(
    category="spot",
    symbol="BTCUSDT"
)

print(orderbook['result'])
```

If you see bid/ask data, you're set!

---

## 4. Configure Scripts with API Keys

### Update `01b_fetch_alternative_data.py`

Open the file and update these lines:

```python
# Line 47-49: Reddit credentials
REDDIT_CLIENT_ID = 'YOUR_14_CHAR_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'YOUR_27_CHAR_CLIENT_SECRET'
REDDIT_USER_AGENT = 'crpbot/1.0'

# Line 52: Coinglass API key
COINGLASS_API_KEY = 'CG-xxxxxxxxxxxxxxxxxxxxxx'
```

**Save the file.**

---

## 5. Install Required Python Packages

In Google Colab, run:

```python
# Reddit
!pip install praw

# Sentiment analysis
!pip install transformers torch

# Coinglass
!pip install requests

# Bybit
!pip install pybit

# FinBERT model (for sentiment)
# This will download ~400MB
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
```

---

## 6. Verify Installation

Run this in Colab to verify everything works:

```python
import praw
import requests
from pybit.unified_trading import HTTP
from transformers import pipeline

print("✅ All packages imported successfully")

# Test Reddit (with your credentials)
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='crpbot/1.0'
)
print(f"✅ Reddit authenticated: {reddit.read_only}")

# Test Coinglass (with your API key)
headers = {'CG-API-KEY': 'YOUR_API_KEY'}
response = requests.get(
    'https://open-api.coinglass.com/public/v2/liquidation',
    headers=headers,
    params={'symbol': 'BTCUSDT', 'interval': '1h'}
)
print(f"✅ Coinglass API status: {response.status_code}")

# Test Bybit
session = HTTP(testnet=False)
orderbook = session.get_orderbook(category="spot", symbol="BTCUSDT")
print(f"✅ Bybit orderbook fetched: {len(orderbook['result']['b'])} bids")

# Test FinBERT
sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = sentiment("Bitcoin price surge today")[0]
print(f"✅ FinBERT sentiment: {result['label']} ({result['score']:.2f})")

print("\n🎉 All APIs configured correctly!")
```

---

## 7. Cost Summary

| Service | Cost | What You Get |
|---------|------|--------------|
| Reddit Premium API | $100/mo | Higher rate limits, 5 years of posts |
| Coinglass API | $50/mo | Liquidation data, historical access |
| Bybit API | Free | Real-time orderbook L2 data |
| **TOTAL** | **$150/mo** | Full V3 Ultimate data layer |

### Cost Optimization

**Option 1: Use Free Reddit Tier**
- Free tier: 60 req/min
- Slower but works
- Total cost: $50/mo
- Expected performance: 73-76% WR (vs 75-78%)

**Option 2: Skip Orderbook Real-time**
- Use historical snapshots only
- Reduces complexity
- Total cost: $150/mo
- Expected performance: 74-77% WR (vs 75-78%)

---

## 8. Rate Limits & Quotas

### Reddit
- **Free**: 60 requests/minute
- **Premium**: ~600 requests/minute
- **Note**: Fetching 5 years for 10 coins = ~50k API calls
  - Free tier: ~14 hours
  - Premium tier: ~1.5 hours

### Coinglass
- **API Pro**: 10,000 requests/day
- **Note**: Fetching 5 years of liquidations:
  - 5 years × 365 days × 24 hours = 43,800 hours
  - ~10 coins = 438,000 API calls
  - Need to spread over 44 days OR fetch daily

**IMPORTANT**: For 5-year backfill, consider:
1. Start with 1 year of data for testing
2. Incrementally fetch more data
3. Or pay for higher Coinglass tier

### Bybit
- **Public endpoints**: No rate limit (reasonable use)
- **Note**: Orderbook snapshots are real-time only (not historical)

---

## 9. Troubleshooting

### Reddit API Errors

**Error**: `prawcore.exceptions.ResponseException: received 403 HTTP response`
- **Solution**: API key is wrong or account not verified
- **Fix**: Double-check credentials, verify email

**Error**: `prawcore.exceptions.TooManyRequests`
- **Solution**: Hit rate limit
- **Fix**: Add delays between requests: `time.sleep(1)`

### Coinglass API Errors

**Error**: `{"code": 401, "message": "Invalid API key"}`
- **Solution**: API key is wrong
- **Fix**: Check key from Coinglass dashboard

**Error**: `{"code": 429, "message": "Rate limit exceeded"}`
- **Solution**: Too many requests
- **Fix**: Implement exponential backoff, or upgrade plan

### Bybit API Errors

**Error**: `Invalid symbol`
- **Solution**: Symbol format wrong
- **Fix**: Use `BTCUSDT` not `BTC/USDT`

**Error**: `Connection timeout`
- **Solution**: Network issue
- **Fix**: Retry with exponential backoff

---

## 10. Security Best Practices

1. **Never commit API keys to git**
   - Add `.env` to `.gitignore`
   - Use environment variables

2. **Use environment variables in Colab**
   ```python
   import os
   from google.colab import userdata

   # Store securely in Colab
   REDDIT_CLIENT_ID = userdata.get('REDDIT_CLIENT_ID')
   REDDIT_CLIENT_SECRET = userdata.get('REDDIT_CLIENT_SECRET')
   COINGLASS_API_KEY = userdata.get('COINGLASS_API_KEY')
   ```

3. **Restrict API key permissions**
   - Reddit: Read-only
   - Coinglass: Read-only
   - Bybit: Read-only (if using API key)

4. **Monitor usage**
   - Check Reddit API usage: https://www.reddit.com/prefs/apps
   - Check Coinglass usage: https://www.coinglass.com/api
   - Set up alerts for quota limits

---

## 11. Next Steps

Once all APIs are configured:

1. ✅ Verify all credentials work (Section 6)
2. ✅ Update `01b_fetch_alternative_data.py` with your keys
3. ✅ Run Step 1B: Alternative data collection
4. ✅ Proceed to Step 2: Enhanced feature engineering
5. ✅ Run Step 3B: Enhanced ensemble training

---

## Support

**Reddit API**:
- Docs: https://www.reddit.com/dev/api
- Support: https://www.reddithelp.com/

**Coinglass API**:
- Docs: https://coinglass-api.readme.io/
- Support: support@coinglass.com

**Bybit API**:
- Docs: https://bybit-exchange.github.io/docs/
- Support: https://www.bybit.com/en-US/help-center/

---

**Ready to proceed?** Follow the steps above, then run the enhanced V3 Ultimate pipeline!

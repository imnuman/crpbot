# Binance Alternatives for Canada

## 🚨 Issue: Binance is Restricted in Canada

Binance has restrictions for users in certain countries, including Canada. We need alternative exchanges or data providers.

## ✅ Recommended Solutions

### Option 1: Coinbase Advanced Trade API (Recommended)
**Best for**: Reliable, well-documented, works in Canada

**Advantages**:
- ✅ Works in Canada
- ✅ Free API (no trading fees for API access)
- ✅ Good documentation
- ✅ Historical data available
- ✅ Similar to Binance API structure

**Where to get**:
- Go to: https://www.coinbase.com/advanced-trade
- Create API key in Coinbase Advanced Trade settings
- API docs: https://docs.cloud.coinbase.com/advanced-trade-api/docs

**Rate Limits**: 10 requests/second (plenty for historical data)

---

### Option 2: Kraken API
**Best for**: Large exchange, good historical data

**Advantages**:
- ✅ Works in Canada
- ✅ Free API
- ✅ Good historical data coverage
- ✅ Well-documented

**Where to get**:
- Go to: https://www.kraken.com/
- Create API key in Account > Settings > API
- API docs: https://docs.kraken.com/rest/

**Rate Limits**: Varies by tier (usually sufficient)

---

### Option 3: Crypto.com API
**Best for**: Alternative option

**Advantages**:
- ✅ Works in Canada
- ✅ Free API
- ✅ Good documentation

**Where to get**:
- Go to: https://crypto.com/exchange
- Create API key in account settings
- API docs: https://exchange-docs.crypto.com/

---

### Option 4: Gate.io API
**Best for**: Large exchange with good data

**Advantages**:
- ✅ Works globally including Canada
- ✅ Free API
- ✅ Good historical data

**Where to get**:
- Go to: https://www.gate.io/
- Create API key in account settings
- API docs: https://www.gate.io/docs/developers/apiv4/

---

### Option 5: OKX API (formerly OKEx)
**Best for**: Large exchange, good data quality

**Advantages**:
- ✅ Works in Canada
- ✅ Free API
- ✅ Excellent historical data

**Where to get**:
- Go to: https://www.okx.com/
- Create API key in account settings
- API docs: https://www.okx.com/docs-v5/

---

### Option 6: Third-Party Data Providers (No Exchange Account Needed)

#### A. CryptoCompare API
**Best for**: Aggregated data from multiple exchanges

**Advantages**:
- ✅ No exchange account needed
- ✅ Free tier available (up to 100,000 calls/day)
- ✅ Historical data from multiple exchanges
- ✅ Easy to use

**Where to get**:
- Go to: https://www.cryptocompare.com/crypto-api/
- Sign up for free API key
- API docs: https://min-api.cryptocompare.com/documentation

**Rate Limits**: Free tier: 100,000 calls/day

---

#### B. CoinGecko API
**Best for**: Market data, historical prices

**Advantages**:
- ✅ No exchange account needed
- ✅ Free tier available
- ✅ Good historical data

**Where to get**:
- Go to: https://www.coingecko.com/en/api
- Sign up for free API key
- API docs: https://www.coingecko.com/api/documentations/v3

**Rate Limits**: Free tier: 10-50 calls/minute

---

#### C. Yahoo Finance (via yfinance Python library)
**Best for**: Quick solution, no API key needed

**Advantages**:
- ✅ No API key needed
- ✅ Free
- ✅ Easy to use
- ⚠️ Less granular (1m candles might be limited)

**Usage**:
```python
import yfinance as yf
btc = yf.Ticker("BTC-USD")
data = btc.history(period="5y", interval="1m")
```

---

## 🎯 My Recommendation

### For Phase 2, I recommend:

**Primary**: **Coinbase Advanced Trade API** or **Kraken API**
- Both work in Canada
- Both have good historical data
- Both are well-documented
- Similar API structure to Binance

**Alternative**: **CryptoCompare API** (if you don't want to create exchange accounts)
- No exchange account needed
- Free tier is generous
- Aggregated data from multiple exchanges

---

## 🔧 Implementation Strategy

We'll make the data pipeline **provider-agnostic**:

1. Create an abstract `DataProvider` interface
2. Implement providers for:
   - Coinbase Advanced Trade
   - Kraken
   - CryptoCompare (backup)
3. Make it easy to switch providers via config

This way:
- ✅ You can use any provider that works in Canada
- ✅ Easy to switch if one has issues
- ✅ Can even use multiple providers for redundancy

---

## 📋 Action Items

1. **Choose a provider** (I recommend Coinbase or Kraken)
2. **Get API keys** from your chosen provider
3. **Update `.env`** with the new provider's credentials
4. **I'll update the data pipeline** to support your chosen provider

---

## 🚀 Next Steps

Let me know which provider you'd like to use, and I'll:
1. Update the data pipeline to support it
2. Update the `.env.example` with the right variables
3. Create the connector for Phase 2

**Which provider would you like to use?**
- Coinbase Advanced Trade (recommended)
- Kraken
- CryptoCompare (no exchange account needed)
- Other?


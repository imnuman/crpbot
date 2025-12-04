# HYDRA 3.0 - Exotic Forex Expansion Plan

**Date**: 2025-11-30
**Status**: READY TO IMPLEMENT
**Goal**: Launch Tournament B (Exotic Forex) alongside Tournament A (Crypto)

---

## 🎯 Current Status

### ✅ What's Already Built (Infrastructure Ready)

| Component | Status | Location |
|-----------|--------|----------|
| Exotic forex profiles | ✅ EXISTS | `libs/hydra/asset_profiles.py` |
| USD/TRY profile | ✅ DEFINED | Lines 115-136 |
| USD/ZAR profile | ✅ DEFINED | Lines 138-159 |
| USD/MXN profile | ✅ DEFINED | Lines 161-182 |
| EUR/TRY profile | ✅ DEFINED | Lines 184-206 |
| Session detection | ✅ CODED | Lines 74-106 |
| Spread rejection | ✅ IMPLEMENTED | AssetProfile.spread_reject_multiplier |
| FTMO credentials | ✅ CONFIGURED | `.env` (FTMO_LOGIN, FTMO_PASS, FTMO_SERVER) |
| Guardian FTMO rules | ✅ ACTIVE | `libs/hydra/guardian.py` |

**8 Exotic Forex Pairs Already Profiled**:
1. USD/TRY (Turkish Lira)
2. USD/ZAR (South African Rand)
3. USD/MXN (Mexican Peso)
4. EUR/TRY (Euro/Lira)
5. USD/PLN (Polish Zloty)
6. USD/HUF (Hungarian Forint)
7. USD/CZK (Czech Koruna)
8. EUR/ZAR (Euro/Rand)

### ❌ What's Missing (To Activate Forex)

| Component | Status | Effort |
|-----------|--------|--------|
| OANDA/FTMO data client | ❌ NOT CREATED | 2-3 hours |
| DXY index feed | ❌ NOT CONNECTED | 1 hour |
| News calendar (ForexFactory) | ❌ NOT CONNECTED | 2 hours |
| Dual tournament launcher | ❌ NOT IMPLEMENTED | 1 hour |
| Separate leaderboards | ❌ NEEDS MODIFICATION | 30 min |

**Total Estimated Effort**: 6-7 hours development time

---

## 📋 Implementation Plan

### Phase 1: Forex Data Connector (2-3 hours)

**Create**: `libs/data/oanda_client.py` (or `ftmo_client.py`)

**Requirements**:
- Use OANDA v20 REST API (https://api-fxtrade.oanda.com)
- Credentials from `.env`: `FTMO_LOGIN`, `FTMO_PASS`, `FTMO_SERVER`
- Methods needed:
  - `get_candles(symbol, granularity, count)` - Historical OHLCV
  - `get_current_price(symbol)` - Real-time price
  - `get_spread(symbol)` - Current bid-ask spread
  - `place_order(symbol, direction, size, sl, tp)` - Paper trading simulation
  - `get_open_positions()` - Track open trades
  - `close_position(position_id)` - Close trades

**Example Usage**:
```python
from libs.data.oanda_client import OANDAClient

client = OANDAClient()
candles = client.get_candles("USD_TRY", "M5", 200)  # 200x 5min candles
spread = client.get_spread("USD_TRY")  # Current spread in pips
```

**Symbol Format**:
- HYDRA uses: `USD/TRY`, `EUR/ZAR`
- OANDA uses: `USD_TRY`, `EUR_ZAR`
- Need converter: `symbol.replace("/", "_")`

---

### Phase 2: DXY Index Feed (1 hour)

**Purpose**: USD strength affects all exotic forex pairs

**Create**: `libs/data/dxy_client.py`

**Options**:
1. **TradingView** - Free DXY data via unofficial API
2. **FRED (St. Louis Fed)** - Official DXY index (DXY, DTWEXBGS)
3. **Yahoo Finance** - yfinance library (`^DXY`)

**Recommended**: Yahoo Finance (already have `yahoo_finance_client.py`)

**Integration**:
```python
# In libs/hydra/regime_detector.py
def _get_dxy_data(self) -> Optional[float]:
    from libs.data.yahoo_finance_client import YahooFinanceClient
    client = YahooFinanceClient()
    return client.get_latest_close("^DXY")
```

---

### Phase 3: News Calendar (2 hours)

**Purpose**: Avoid trading during high-impact news (CB meetings, NFP, CPI)

**Create**: `libs/data/news_calendar_client.py`

**Source**: ForexFactory (web scraping)

**Alternative**: Use Oanda's economic calendar API

**Integration**:
```python
# In libs/hydra/anti_manipulation.py
def _check_news_calendar(self, asset: str, timestamp: datetime) -> bool:
    """Avoid trading 30min before/after high-impact news."""
    from libs.data.news_calendar_client import NewsCalendarClient
    calendar = NewsCalendarClient()

    # Check if high-impact news within 30min
    upcoming_news = calendar.get_upcoming_events(
        currency=asset[:3],  # USD, EUR, etc.
        time_window_minutes=30,
        impact="HIGH"
    )

    if upcoming_news:
        logger.warning(f"High-impact news upcoming: {upcoming_news}")
        return False  # Reject trade

    return True  # Safe to trade
```

---

### Phase 4: Dual Tournament Setup (1.5 hours)

**Goal**: Run two separate HYDRA instances simultaneously

#### Tournament A (Crypto) - KEEP CURRENT
```bash
# Already running on PID 3321401
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --tournament-name crypto \
  --port 8001 \
  --paper
```

**Leaderboard**: `data/hydra/crypto/tournament_scores.jsonl`

#### Tournament B (Exotic Forex) - NEW
```bash
# New instance
.venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets USD/TRY USD/ZAR USD/MXN EUR/TRY USD/PLN USD/HUF USD/CZK EUR/ZAR \
  --tournament-name exotic_forex \
  --port 8002 \
  --paper \
  --data-provider oanda
```

**Leaderboard**: `data/hydra/exotic_forex/tournament_scores.jsonl`

**Modifications Needed**:

1. **Add `--tournament-name` CLI arg** (apps/runtime/hydra_runtime.py):
```python
parser.add_argument("--tournament-name", default="default", help="Tournament name")
```

2. **Add `--data-provider` CLI arg**:
```python
parser.add_argument("--data-provider", choices=["coinbase", "oanda"], default="coinbase")
```

3. **Update TournamentTracker paths**:
```python
# libs/hydra/tournament_tracker.py
def __init__(self, tournament_name: str = "default"):
    self.data_dir = Path(f"data/hydra/{tournament_name}")
    self.votes_file = self.data_dir / "tournament_votes.jsonl"
    self.scores_file = self.data_dir / "tournament_scores.jsonl"
```

---

### Phase 5: Comparative Dashboard (Optional - 3 hours)

**Create**: `apps/dashboard_reflex/pages/tournaments.py`

**Features**:
- Side-by-side leaderboards (Crypto vs Forex)
- Win rate comparison per gladiator
- Performance by asset type
- Sharpe ratio comparison

---

## 🚀 Launch Sequence

### Week 1: Development
1. ✅ Day 1: Create OANDA client (2-3h)
2. ✅ Day 2: Integrate DXY feed (1h)
3. ✅ Day 3: Add news calendar (2h)
4. ✅ Day 4: Dual tournament setup (1.5h)
5. ✅ Day 5: Testing & bug fixes (2h)

**Total**: ~8-9 hours development

### Week 2: Paper Trading
- Run both tournaments in parallel
- Collect 20+ trades per tournament
- Compare win rates:
  - Crypto: Should leverage volatility (trending regimes)
  - Forex: Should leverage session edges (London open spikes)

### Week 3: Analysis & Decision
**Questions to Answer**:
1. Which tournament has higher win rate?
2. Which gladiator performs better on forex?
3. Do structural edges work better on crypto or forex?
4. Should we focus on one market type?

---

## ⚠️ Critical Questions (NEED ANSWERS)

### 1. Forex Broker Access
**Question**: Does the user have access to OANDA/FTMO/IG Markets account?

**Options**:
- ✅ FTMO credentials already in `.env` (FTMO_LOGIN: 531025383)
- ? OANDA account? (need API key)
- ? IG Markets? (different API)

**Recommendation**: Verify FTMO credentials work OR create OANDA demo account

### 2. Paper vs Live Trading
**Question**: Should exotic forex start in paper trading or go straight to FTMO challenge?

**Recommendation**: PAPER FIRST
- Forex behaves differently than crypto
- Spreads can kill exotic trades if not careful
- Need to validate session detection works
- Need 20+ forex trades before risking real FTMO account

### 3. Which 8 Exotic Pairs?
**Current Selection** (already profiled):
1. USD/TRY - High volatility, Turkey CB surprises
2. USD/ZAR - Gold correlation, EM sentiment
3. USD/MXN - Oil correlation, EM leader
4. EUR/TRY - Extreme volatility (double exotic)
5. USD/PLN - CEE leader
6. USD/HUF - CEE follower
7. USD/CZK - CEE stable
8. EUR/ZAR - Commodity + EM

**Alternative**: Start with only 3-4 most liquid (MXN, ZAR, TRY) then expand?

---

## 📊 Expected Results

### Hypothesis
**Crypto (Tournament A)** should win because:
- 24/7 trading (no session limits)
- Higher volatility (bigger edge opportunities)
- Funding rates (additional structural edge)
- Liquidation data (heatmaps work)

**Exotic Forex (Tournament B)** should win because:
- Session open spikes are predictable (London/NY)
- Lower manipulation (no liquidation cascades)
- DXY correlation provides macro edge
- News calendar = avoid disasters

**Reality Check**: HYDRA might underperform on forex if:
- Spreads eat into profits (exotic spreads = 20-30 pips)
- Session edges don't materialize (need empirical validation)
- CB interventions (Turkey, SA central banks intervene frequently)

---

## 🎯 Success Criteria

### Tournament B Launch = SUCCESS if:
1. ✅ OANDA client fetches forex data without errors
2. ✅ DXY correlation shows in gladiator reasoning
3. ✅ News calendar prevents trading during CB meetings
4. ✅ Spreads correctly rejected when >3x normal
5. ✅ Session detection limits trading to London/NY only
6. ✅ 20+ paper trades collected within 1 week
7. ✅ Win rate > 50% on exotic forex (vs 56.5% on crypto)

### Comparative Analysis = SUCCESS if:
- Clear winner emerges (crypto OR forex, not both equal)
- Gladiator performance differences identified
- Structural edge validation per market type
- Decision: Focus 100% on winner OR keep both running

---

## 📁 Files to Create/Modify

### Create New Files
- [ ] `libs/data/oanda_client.py` (250 lines)
- [ ] `libs/data/dxy_client.py` (80 lines)
- [ ] `libs/data/news_calendar_client.py` (150 lines)
- [ ] `scripts/launch_tournament_forex.sh` (20 lines)
- [ ] `validation/TOURNAMENT_B_RESULTS.md` (results doc)

### Modify Existing Files
- [ ] `apps/runtime/hydra_runtime.py` (add --tournament-name, --data-provider args)
- [ ] `libs/hydra/tournament_tracker.py` (add tournament_name to paths)
- [ ] `libs/hydra/regime_detector.py` (integrate DXY feed)
- [ ] `libs/hydra/anti_manipulation.py` (integrate news calendar)

---

## 🚧 Blockers & Dependencies

### Must Resolve Before Starting
1. **Broker Access**: Verify FTMO credentials work OR get OANDA demo account
2. **User Confirmation**: Does user want dual tournaments? Or stick with crypto only?
3. **Paper Trading Decision**: Forex paper trading first, or straight to live FTMO?

### Nice to Have (Not Blockers)
- Historical forex data for backtesting (not strictly needed for paper trading)
- Exotic pair charts in dashboard (can add later)
- Automated tournament comparison reports (can be manual first)

---

## 💰 Cost Implications

### Current (Crypto Only)
- **DeepSeek API**: ~$0.19/$150 (0.13% used)
- **AWS**: ~$79/month (S3 only, no RDS/Redis)
- **Total**: <$1/month runtime cost

### With Forex Added (Dual Tournaments)
- **DeepSeek API**: 2x usage = ~$0.40/month
- **OANDA API**: FREE for demo account
- **FTMO Challenge**: $155 one-time (if going live)
- **Total**: <$2/month runtime cost

**Conclusion**: Negligible cost increase for forex expansion

---

## 🎬 Next Steps

**WAITING FOR USER INPUT**:

1. **Do you want to activate Tournament B (exotic forex)?**
   - [ ] Yes - proceed with implementation
   - [ ] No - keep crypto-only for now
   - [ ] Later - after more crypto data collection

2. **Which forex broker do you have access to?**
   - [ ] FTMO (credentials in .env: 531025383)
   - [ ] OANDA (need to create demo account)
   - [ ] IG Markets (need API credentials)
   - [ ] Other: __________

3. **Paper trading strategy?**
   - [ ] Paper trade forex FIRST (recommended)
   - [ ] Go straight to FTMO challenge
   - [ ] Use OANDA demo account for live-like trading

**Once confirmed, I can implement Tournament B in ~6-8 hours total.**

---

**Last Updated**: 2025-11-30
**Maintainer**: Builder Claude
**Status**: AWAITING USER DECISION

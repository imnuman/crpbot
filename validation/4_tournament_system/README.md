# Tournament System

**Status**: PARTIALLY IMPLEMENTED

---

## What's Implemented

### 1. Chat Interface ✅
**File**: `apps/dashboard_reflex/hydra_chat.py`

**Features**:
- User submits trading questions
- All 4 gladiators respond independently
- Responses displayed in real-time
- Vote aggregation shown (BUY/SELL/HOLD)
- Confidence scores displayed

**Access**: http://178.156.136.185:3000/chat

---

### 2. Voting Mechanism ✅
**Location**: Integrated in `hydra_runtime.py`

**How It Works**:
- Each gladiator votes: BUY, SELL, or HOLD
- Consensus requires 2/4 minimum votes
- Ties: No action (HOLD)
- Vote history stored in JSONL

**Vote Storage**: `/root/crpbot/data/hydra/vote_history.jsonl`

---

### 3. Paper Trading ✅
**Storage**: `/root/crpbot/data/hydra/paper_trades.jsonl`

**Tracked Data**:
- Timestamp
- Asset (BTC-USD, ETH-USD, SOL-USD)
- Direction (BUY/SELL)
- Consensus vote counts
- Individual gladiator votes
- Confidence scores

---

## What's Missing

### 1. Win/Loss Tracking ❌
**Problem**: No scoring per gladiator

**What's Needed**:
- Track which gladiator voted correctly
- Score: +1 for correct prediction, 0 for wrong
- Maintain leaderboard
- Identify top performers

---

### 2. Evolution Logic ❌
**Problem**: No kill/breed implementation

**What's Needed**:
- **24-hour kill**: Lowest performer resets prompt
- **4-day breed**: Top 2 gladiators combine strategies
- **Lesson storage**: Winners teach losers
- **Prompt mutation**: Losers must surpass winners

---

### 3. Performance Analytics ❌
**Problem**: No gladiator-level metrics

**What's Needed**:
- Win rate per gladiator
- Average confidence per gladiator
- Regime-specific performance
- Asset-specific performance

---

## Tournament Architecture (Planned)

### Kill Logic (24 Hours)
```python
# After 24 hours:
# 1. Calculate win rate for each gladiator
# 2. Identify lowest performer
# 3. Reset their system prompt to base template
# 4. Log "death" event
# 5. Continue tournament
```

### Breed Logic (4 Days)
```python
# After 4 days:
# 1. Identify top 2 gladiators by win rate
# 2. Extract their successful strategies
# 3. Combine into hybrid prompt
# 4. Create new "child" gladiator
# 5. Replace worst performer with child
```

### Lesson Memory
```python
# After each trade outcome:
# 1. Determine which gladiators were correct
# 2. Extract their reasoning
# 3. Store as "lesson" (what worked)
# 4. Feed lessons to losing gladiators
# 5. Update their prompts with insights
```

---

## Current Files

### In This Folder:
- (Chat interface file not copied yet - will be added)

### Related Files:
- `apps/runtime/hydra_runtime.py` - Main voting orchestrator
- `apps/dashboard_reflex/hydra_chat.py` - Chat UI
- `data/hydra/vote_history.jsonl` - Vote records
- `data/hydra/paper_trades.jsonl` - Trade outcomes
- `data/hydra/chat_history.jsonl` - User interactions

---

## Implementation Priority

1. **Immediate**: Fix spelling (Groq → Grok)
2. **Short-term**: Update gladiator prompts for competition mindset
3. **Medium-term**: Implement win/loss tracking
4. **Long-term**: Build kill/breed evolution system

---

## Current Tournament Stats (2025-11-30)

**Total Trades**: 251
**Direction Split**: 64% BUY, 36% SELL
**Consensus Rate**: ~67% (trades that reached 2/4+ votes)
**Average Confidence**: 0.68 (68%)

**Gladiator Performance**: Not tracked yet (to be implemented)

---

**Date**: 2025-11-30

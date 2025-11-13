# Evaluation Pipeline Fix - Complete Guide

**Goal**: Replace fake data in evaluator with real market data
**Time**: ~45 minutes
**Status**: Ready to execute

---

## 🔍 Step 1: Inspect Current Dataset (2 min)

Check what data the dataset currently provides:

```bash
cd /root/crpbot
source .venv/bin/activate

# Check dataset output
python -c "
from apps.trainer.train.dataset import TradingDataset
from apps.trainer.data_pipeline import load_features_from_parquet, create_walk_forward_splits
from apps.trainer.features import normalize_features

# Load test data
df = load_features_from_parquet('data/features/features_BTC-USD_1m_latest.parquet', 'BTC-USD')
train, val, test, stats = create_walk_forward_splits(df)
test_norm = normalize_features(test, stats)

# Create dataset
dataset = TradingDataset(
    test_norm,
    lookback_window=60,
    prediction_horizon=15,
    prediction_type='direction'
)

# Check what it returns
sample = dataset[0]
print('Dataset returns:')
for key, value in sample.items():
    if hasattr(value, 'shape'):
        print(f'  {key}: shape={value.shape}, dtype={value.dtype}')
    else:
        print(f'  {key}: {type(value).__name__}')
"
```

---

## 🔧 Step 2: Fix Dataset to Include Prices (15 min)

Update `apps/trainer/train/dataset.py` to return timestamps and prices:

```python
# File: apps/trainer/train/dataset.py
# Around line 45-75 in __init__ and __getitem__

def __init__(
    self,
    df: pd.DataFrame,
    lookback_window: int = 60,
    prediction_horizon: int = 15,
    prediction_type: str = "direction",
):
    """Initialize dataset with features and labels."""
    self.lookback_window = lookback_window
    self.prediction_horizon = prediction_horizon
    self.prediction_type = prediction_type

    # Store full dataframe for timestamps and prices
    self.df = df  # ← ADD THIS

    # ... rest of existing code ...

    # Store sequences
    self.X = features
    self.y = labels

    # Store corresponding timestamps, prices, and future prices
    self.timestamps = []
    self.entry_prices = []
    self.exit_prices = []

    for i in range(lookback_window, len(df) - prediction_horizon):
        # Get timestamp of current candle (entry point)
        self.timestamps.append(df.index[i])

        # Get close price at entry
        self.entry_prices.append(df.iloc[i]['close'])

        # Get close price at exit (prediction_horizon ahead)
        self.exit_prices.append(df.iloc[i + prediction_horizon]['close'])

    self.timestamps = np.array(self.timestamps)
    self.entry_prices = np.array(self.entry_prices, dtype=np.float32)
    self.exit_prices = np.array(self.exit_prices, dtype=np.float32)

    logger.info(
        f"Created dataset: {len(self)} sequences, "
        f"{self.X.shape[2]} features, "
        f"prediction_type={prediction_type}"
    )


def __getitem__(self, idx: int) -> dict[str, Any]:
    """Get sequence and label at index."""
    return {
        "features": torch.FloatTensor(self.X[idx]),
        "label": torch.FloatTensor([self.y[idx]]),
        "timestamp": self.timestamps[idx],  # ← ADD
        "entry_price": self.entry_prices[idx],  # ← ADD
        "exit_price": self.exit_prices[idx],  # ← ADD
    }
```

---

## 🔧 Step 3: Fix Evaluator to Use Real Data (20 min)

Update `apps/trainer/eval/evaluator.py` to use real prices from dataset:

Find the section that creates fake trades (around lines 120-150) and replace with:

```python
# File: apps/trainer/eval/evaluator.py
# In the evaluate_model function, around line 120-150

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    symbol: str = "BTC-USD",
) -> dict[str, Any]:
    """Evaluate model and return comprehensive metrics."""

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # For backtest
    trades = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            # Get real data from batch
            timestamps = batch["timestamp"]
            entry_prices = batch["entry_price"]
            exit_prices = batch["exit_price"]

            # Model predictions
            logits = model(features)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Simulate trades with REAL data
            for i in range(len(preds)):
                prediction = int(preds[i].item())  # 0=down, 1=up
                actual_label = int(labels[i].item())
                confidence = float(probs[i].item())

                # Get real prices from batch
                entry_time = timestamps[i]
                entry_price = float(entry_prices[i].item())
                exit_price = float(exit_prices[i].item())

                # Calculate actual price movement
                actual_direction = 1 if exit_price > entry_price else 0
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100

                # Determine trade outcome
                prediction_correct = (prediction == actual_direction)

                # Simulate position with TP/SL (simplified)
                position_size = 1000.0  # $1000 per trade

                if prediction == 1:  # Long position
                    # If price went up, we profit; if down, we lose
                    pnl = position_size * (price_change_pct / 100)
                else:  # Short position
                    # If price went down, we profit; if up, we lose
                    pnl = position_size * (-price_change_pct / 100)

                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "prediction": prediction,
                    "actual_direction": actual_direction,
                    "confidence": confidence,
                    "position_size": position_size,
                    "pnl": pnl,
                    "price_change_pct": price_change_pct,
                    "correct": prediction_correct,
                })

    # ... rest of evaluation code (calculate metrics from trades) ...
```

---

## 🔧 Step 4: Verify Changes (3 min)

```bash
# Check that dataset now returns prices
python -c "
from apps.trainer.train.dataset import TradingDataset
from apps.trainer.data_pipeline import load_features_from_parquet, create_walk_forward_splits
from apps.trainer.features import normalize_features

df = load_features_from_parquet('data/features/features_BTC-USD_1m_latest.parquet', 'BTC-USD')
train, val, test, stats = create_walk_forward_splits(df)
test_norm = normalize_features(test, stats)

dataset = TradingDataset(test_norm, lookback_window=60, prediction_horizon=15, prediction_type='direction')

sample = dataset[0]
print('Dataset now returns:')
for key in sample.keys():
    print(f'  ✅ {key}')

# Verify we have real data
print(f'\nSample entry price: \${sample[\"entry_price\"]:.2f}')
print(f'Sample exit price: \${sample[\"exit_price\"]:.2f}')
print(f'Sample timestamp: {sample[\"timestamp\"]}')
"
```

---

## 🔧 Step 5: Re-evaluate Models (10 min)

```bash
cd /root/crpbot
source .venv/bin/activate

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Re-evaluating with REAL data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# BTC
echo "\n📊 Evaluating BTC-USD..."
python scripts/evaluate_model.py \
  --model models/lstm_BTC_USD_1m_a7aff5c4.pt \
  --symbol BTC-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05

# ETH
echo "\n📊 Evaluating ETH-USD..."
python scripts/evaluate_model.py \
  --model models/lstm_ETH_USD_1m_a7aff5c4.pt \
  --symbol ETH-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05

# SOL
echo "\n📊 Evaluating SOL-USD..."
python scripts/evaluate_model.py \
  --model models/lstm_SOL_USD_1m_a7aff5c4.pt \
  --symbol SOL-USD \
  --model-type lstm \
  --min-accuracy 0.68 \
  --max-calibration-error 0.05
```

---

## ✅ Expected Output (Real Results)

After fix, you should see:

```
Evaluating: lstm_BTC_USD_1m_a7aff5c4.pt

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Accuracy:        65.3%  ❌ FAIL (≥68%)
Calibration Error:     4.8%  ✅ PASS (≤5%)
F1 Score:             0.648
Win Rate:             63.2%  (realistic!)
Total PnL:           $12,345  (realistic!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Key differences from fake evaluation:**
- ✅ Win rate: 50-70% (realistic, not 100%)
- ✅ PnL: Moderate (not unrealistically high)
- ✅ Calibration: Accurate (not 30%)

---

## 🎯 Decision Tree

### If Accuracy ≥68% and Calibration ≤5%:
✅ **PASS** - Promote models to production
- Copy to `models/promoted/`
- Continue to Transformer training
- Start Phase 6.5 observation

### If Accuracy 60-67% and Calibration ≤5%:
⚠️ **MARGINAL** - Usable but not ideal
- May promote for testing
- Plan retraining with improvements
- Consider ensemble with Transformer

### If Accuracy <60% or Calibration >5%:
❌ **FAIL** - Need retraining
- Analyze failure modes
- Adjust hyperparameters
- Retrain models

---

## 📝 After Evaluation

Document results:

```bash
# Create results document
cat > MODEL_EVALUATION_RESULTS_REAL.md << 'EOF'
# Model Evaluation Results (Real Data)

Date: 2025-11-13 ~05:30 UTC
Evaluation: Using real market data (fixed evaluation pipeline)

## BTC-USD LSTM
- Test Accuracy: X.X%
- Calibration Error: X.X%
- Win Rate: X.X%
- Total PnL: $X,XXX
- Status: PASS/FAIL

## ETH-USD LSTM
- Test Accuracy: X.X%
- Calibration Error: X.X%
- Win Rate: X.X%
- Total PnL: $X,XXX
- Status: PASS/FAIL

## SOL-USD LSTM
- Test Accuracy: X.X%
- Calibration Error: X.X%
- Win Rate: X.X%
- Total PnL: $X,XXX
- Status: PASS/FAIL

## Summary
- Models passing gates: X/3
- Decision: [PROMOTE/RETRAIN/MARGINAL]
- Next steps: [...]
EOF

# Commit
git add apps/trainer/train/dataset.py apps/trainer/eval/evaluator.py MODEL_EVALUATION_RESULTS_REAL.md
git commit -m "fix: use real market data in evaluation pipeline

Changes:
- Updated TradingDataset to return timestamps, entry/exit prices
- Updated evaluator to use real prices instead of fake $50K
- Removed fake 1% profit simulation
- Now calculates actual price movements from data

Impact:
- Win rates now realistic (50-70% vs 100%)
- PnL now based on actual price changes
- Calibration error accurate
- Models properly tested against real outcomes

Next: Re-evaluate models to get true accuracy metrics

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

---

## Timeline

| Time | Task | Duration |
|------|------|----------|
| Now | Step 1: Inspect dataset | 2 min |
| +2 | Step 2: Fix dataset.py | 15 min |
| +17 | Step 3: Fix evaluator.py | 20 min |
| +37 | Step 4: Verify changes | 3 min |
| +40 | Step 5: Re-evaluate (BTC) | 3 min |
| +43 | Re-evaluate (ETH) | 3 min |
| +46 | Re-evaluate (SOL) | 3 min |
| +49 | Document results | 5 min |
| **+54** | **DONE** | **54 min** |

---

Ready to execute!

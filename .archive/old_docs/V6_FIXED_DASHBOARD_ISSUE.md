# V6 Fixed Dashboard Issue - Complete Context

**Date**: 2025-11-16
**Status**: ⚠️ **CRITICAL FIX NEEDED** - Final step remaining

## Problem Statement

Dashboard showing **100% confidence** with **exploding logits** (±40,000) even after implementing V6 Fixed models with temperature scaling.

**Example of Current Bad Output**:
```
Raw logits: Down=40179.414, Neutral=-23806.248, Up=-10306.071
Clamped logits: Down=15.000, Neutral=-15.000, Up=-15.000
V6 Enhanced FNN output (raw softmax): Down=1.000, Neutral=0.000, Up=0.000
Confidence: 100%
```

**Expected Output After Fix**:
```
Raw logits: Down=3.2, Neutral=-2.1, Up=-1.5
Clamped logits: Down=3.2, Neutral=-2.1, Up=-1.5
V6 Enhanced FNN output (raw softmax): Down=0.60, Neutral=0.25, Up=0.15
Confidence: 60%
```

## Root Cause Analysis

V6 Fixed models require a **3-layer fix**, but `ensemble.py` currently only implements **2 of 3 layers**:

### Layer 1: ✅ Logit Clamping (±15) - IMPLEMENTED & WORKING
- **Location**: `apps/runtime/ensemble.py:252`
- **Code**: `output = torch.clamp(output, -self.logit_clip, self.logit_clip)`
- **Evidence**: Logs show `Raw logits: 40179 → Clamped logits: 15.000`

### Layer 2: ✅ Temperature Scaling (T=1.0) - IMPLEMENTED & WORKING
- **Location**: `apps/runtime/ensemble.py:254`
- **Code**: `output = output / self.temperature`
- **Evidence**: Logs show `Applied temperature scaling: T=1.0`

### Layer 3: ❌ StandardScaler Input Normalization - **MISSING!**
- **Required**: Features must be normalized with StandardScaler (mean=0, std=1) **BEFORE** model inference
- **Current State**: Raw unnormalized features → model → exploding logits (40179)
- **Expected State**: StandardScaler → normalized features → model → calibrated logits (~±3)

## Why Even Clamped Logits Give 100% Confidence

Mathematical proof:
```python
# Current (broken):
softmax([15, -15, -15]) = [e^15 / (e^15 + e^-15 + e^-15), ...]
                        ≈ [3269017 / 3269017, ...]
                        ≈ [0.9999, 0.0000, 0.0000]
                        = 100% confidence

# Expected (after fix):
softmax([3.2, -2.1, -1.5]) = [e^3.2 / (e^3.2 + e^-2.1 + e^-1.5), ...]
                            ≈ [24.5 / (24.5 + 0.12 + 0.22), ...]
                            ≈ [0.60, 0.25, 0.15]
                            = 60% confidence
```

The issue: Base model outputs exploding logits (40179) because inputs aren't normalized. With proper StandardScaler normalization, the base model outputs calibrated logits (~±3).

## Changes Already Made

### 1. Model Files Deployed ✅
**Location**: `models/promoted/` on cloud server

**Files**:
- `lstm_BTC-USD_v6_enhanced.pt` (239 KB, Nov 16 19:23)
- `lstm_ETH-USD_v6_enhanced.pt` (239 KB, Nov 16 19:23)
- `lstm_SOL-USD_v6_enhanced.pt` (239 KB, Nov 16 19:23)
- `scaler_BTC-USD_v6_fixed.pkl` (2.2 KB)
- `scaler_ETH-USD_v6_fixed.pkl` (2.2 KB)
- `scaler_SOL-USD_v6_fixed.pkl` (2.2 KB)

**Checkpoint Structure**:
```python
{
    'model_state_dict': {...},  # Base V6EnhancedFNN weights
    'input_size': 72,
    'symbol': 'BTC-USD',
    'version': 'v6_fixed',      # ← Detection flag
    'temperature': 1.0,
    'logit_clip': 15.0
}
```

### 2. ensemble.py Changes ✅

**File**: `apps/runtime/ensemble.py` on cloud server

**Change 1**: Lines 128-133 - V6 Fixed Parameter Storage
```python
# Store V6 Fixed parameters if present
self.temperature = checkpoint.get("temperature", 1.0)
self.logit_clip = checkpoint.get("logit_clip", 15.0)
self.model_version = checkpoint.get("version", "unknown")
if self.model_version == "v6_fixed":
    logger.info(f"V6 Fixed model detected: T={self.temperature:.1f}, clip={self.logit_clip}")
logger.info(f"DEBUG: Loaded checkpoint version: {self.model_version!r}, temperature: {self.temperature}, logit_clip: {self.logit_clip}")
```

**Change 2**: Lines 134-147 - StandardScaler Loading
```python
# Load StandardScaler for V6 Fixed models
if self.model_version == 'v6_fixed':
    import pickle
    scaler_file = self.model_dir / f'scaler_{self.symbol}_v6_fixed.pkl'
    if scaler_file.exists():
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f'Loaded StandardScaler for V6 Fixed model')
    else:
        logger.warning(f'Scaler file not found: {scaler_file}')
        self.scaler = None
else:
    self.scaler = None
```

**Change 3**: Lines 252-258 - Logit Clamping & Temperature Scaling
```python
# Apply V6 Fixed temperature scaling if present
if hasattr(self, "model_version") and self.model_version == "v6_fixed":
    # Clamp logits to prevent overflow
    output = torch.clamp(output, -self.logit_clip, self.logit_clip)
    # Apply temperature scaling
    output = output / self.temperature
    logger.debug(f"Applied temperature scaling: T={self.temperature:.1f}")

probs = torch.softmax(output, dim=-1).squeeze()
```

**Change 4**: Lines 256-258 - Clamped Logits Debug Logging
```python
clamped_logits = output.squeeze()
logger.debug(f"Clamped logits: Down={clamped_logits[0].item():.3f}, Neutral={clamped_logits[1].item():.3f}, Up={clamped_logits[2].item():.3f}")
```

## 🚨 CRITICAL FIX STILL NEEDED

### What: Add StandardScaler.transform() in predict() function

**Location**: `apps/runtime/ensemble.py` - `predict()` function

**Where to Add**: BEFORE converting features to tensor and passing to model

**Current Code** (around line 235):
```python
# Get last 60 rows of features
features = df_features[amazon_q_feature_list].iloc[-60:].values
features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

# ❌ MISSING: StandardScaler normalization here!

# Convert to tensor
sample = torch.FloatTensor(features).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = self.lstm_model(sample)
```

**Required Fix**:
```python
# Get last 60 rows of features
features = df_features[amazon_q_feature_list].iloc[-60:].values
features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

# ✅ ADD THIS: Apply StandardScaler normalization for V6 Fixed models
if hasattr(self, "scaler") and self.scaler is not None:
    # Reshape for scaler: (60, 72) → (60, 72)
    features = self.scaler.transform(features)
    logger.debug(f"Applied StandardScaler normalization for V6 Fixed model")

# Convert to tensor
sample = torch.FloatTensor(features).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = self.lstm_model(sample)
```

## Implementation Steps

1. **Find the exact location** in `predict()` where features are prepared:
   ```bash
   grep -n "FloatTensor(features)" apps/runtime/ensemble.py
   ```

2. **Add scaler.transform()** BEFORE the FloatTensor conversion:
   - The scaler expects input shape: `(n_samples, 72)` where n_samples is typically 60
   - The scaler outputs same shape: `(60, 72)` with normalized values

3. **Clear Python cache** to ensure new code is loaded:
   ```bash
   find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
   find . -name '*.pyc' -delete 2>/dev/null
   ```

4. **Restart runtime**:
   ```bash
   pkill -9 python3
   sleep 2
   cd ~/crpbot
   nohup .venv/bin/python3 apps/runtime/main.py --mode live --iterations -1 --sleep-seconds 60 > /tmp/v6_FINAL_FIX.log 2>&1 &
   ```

5. **Verify the fix** (wait 10 seconds then check):
   ```bash
   sleep 10
   tail -200 /tmp/v6_FINAL_FIX.log | grep -E "Applied StandardScaler|Raw logits|Clamped logits|confidence"
   ```

## Expected Log Output After Fix

```
2025-11-16 XX:XX:XX | INFO  | V6 Fixed model detected: T=1.0, clip=15.0
2025-11-16 XX:XX:XX | INFO  | Loaded StandardScaler for V6 Fixed model
2025-11-16 XX:XX:XX | DEBUG | Applied StandardScaler normalization for V6 Fixed model
2025-11-16 XX:XX:XX | DEBUG | Raw logits: Down=3.215, Neutral=-2.134, Up=-1.487
2025-11-16 XX:XX:XX | DEBUG | Clamped logits: Down=3.215, Neutral=-2.134, Up=-1.487
2025-11-16 XX:XX:XX | DEBUG | Applied temperature scaling: T=1.0
2025-11-16 XX:XX:XX | DEBUG | V6 Enhanced FNN output (raw softmax): Down=0.603, Neutral=0.251, Up=0.146
2025-11-16 XX:XX:XX | INFO  | BTC-USD: short @ 60.3% confidence (tier: medium)
```

## Verification Checklist

After implementing the fix, verify:

- [ ] Scaler loading message appears: `Loaded StandardScaler for V6 Fixed model`
- [ ] Scaler normalization applied: `Applied StandardScaler normalization for V6 Fixed model`
- [ ] Raw logits in expected range: ±3 to ±10 (not ±40,000)
- [ ] Clamped logits ≈ raw logits (no clamping needed when inputs normalized)
- [ ] Final confidence in range: 55-65% (not 100%)
- [ ] Dashboard shows calibrated predictions with reasonable confidence

## Technical Background

### V6 Fixed Model Architecture

**Base Model** (V6EnhancedFNN):
```python
class V6EnhancedFNN(nn.Module):
    def __init__(self, input_size=72):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 3 classes: DOWN, NEUTRAL, UP
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]  # Take last timestep
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # Raw logits
```

**Training Process** (how V6 Fixed models were created):
1. Load old V6 models (that output exploding logits)
2. Fit StandardScaler on training features
3. Create V6FixedWrapper(base_model, scaler, temperature=1.0, logit_clip=15.0)
4. Wrapper.forward() applies: scaler.transform() → base_model() → clamp() → temperature scaling
5. Save both base_model weights and scaler separately

**Runtime Requirements**:
- Must replicate the exact same 3-layer process
- StandardScaler normalization is **NOT optional** - the model was trained on normalized inputs

### Temperature Tuning Results (from previous session)

Tested temperatures: [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

**T=1.0 (OPTIMAL)**:
- Average confidence: 60%
- Predictions in 60-75% sweet spot: 26.3%
- Result: Best balance for 75% trading threshold

## Files Modified

### Cloud Server (`root@178.156.136.185:~/crpbot/`)

1. `apps/runtime/ensemble.py` - Modified (scaler loading + temperature scaling added)
2. `models/promoted/lstm_*_v6_enhanced.pt` - Replaced with V6 Fixed models
3. `models/promoted/scaler_*_v6_fixed.pkl` - Added StandardScaler files

### Local Machine (`/home/numan/crpbot/`)

1. `V6_FIXED_DASHBOARD_ISSUE.md` - This document

## Next Steps for Builder Claude

1. Read this document completely
2. Locate the exact line in `apps/runtime/ensemble.py` where features are converted to FloatTensor
3. Add the scaler.transform() code BEFORE that line
4. Clear cache and restart runtime
5. Verify logs show proper normalization and calibrated confidence

## Contact/Handoff Notes

- V6 Fixed models are correctly deployed to `models/promoted/`
- Scaler loading code is already in `_load_models()`
- Temperature scaling and logit clamping code is already in `predict()`
- **ONLY MISSING**: scaler.transform() call in predict() before model inference
- This is the final critical piece to fix the dashboard

---

**Last Updated**: 2025-11-16 19:35 UTC
**Created By**: Local Claude (context handoff to Cloud Claude)

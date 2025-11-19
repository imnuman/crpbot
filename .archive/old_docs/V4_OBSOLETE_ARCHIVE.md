# V4 Work - OBSOLETE (Archived November 15, 2025)

**Status**: ❌ OBSOLETE - DO NOT USE
**Reason**: V4 models stuck at 50% accuracy due to noisy free Coinbase data
**Replacement**: V5 with Tardis.dev professional data (target: 65-75% accuracy)

---

## V4 Summary

### What We Tried (November 10-15, 2025)

**Data Source**: Free Coinbase Advanced Trade API (OHLCV, 1-minute candles)

**Models Trained**:
1. 3 LSTM models (BTC, ETH, SOL) - 50 features, 128/3/bidirectional architecture
2. Multiple feature sets tested (31, 50 features)
3. Extensive hyperparameter tuning

**Results**:
- ❌ **Accuracy**: Stuck at ~50% (coin flip level)
- ❌ **Root Cause**: Free Coinbase data too noisy
- ❌ **Conclusion**: No amount of hyperparameter tuning could break 50% ceiling

### V4 Files (OBSOLETE)

**Documentation** (November 13-14):
- `COLAB_EVALUATION.md` - Colab GPU evaluation guide
- `COLAB_INTEGRATION_GUIDE.md` - Claude + Colab integration
- `CLAUDE_MAX_INTEGRATION.md` - Claude Max Projects guide
- `prepare_colab_files.sh` - File preparation script
- `batch_engineer_50_features.sh` - 50-feature batch engineering

**Notebooks**:
- `colab_evaluate_50feat_models.ipynb` - Main evaluation notebook
- `colab_final_evaluation.ipynb` - Final evaluation
- `colab_verify_files.ipynb` - File verification
- `colab_with_claude_api.ipynb` - API-integrated version
- `colab_clean_evaluation.ipynb` - Clean evaluation
- `colab_crpbot_evaluation.ipynb` - CRPBot evaluation
- `colab_crpbot_fixed.ipynb` - Fixed evaluation

**Scripts**:
- `scripts/engineer_50_features.py` - 50-feature engineering (OBSOLETE)

**Models** (in `models/new/`):
- `lstm_BTC_USD_1m_7b5f0829.pt` - BTC LSTM (50 features) - OBSOLETE
- `lstm_ETH_USD_1m_7b5f0829.pt` - ETH LSTM (50 features) - OBSOLETE
- `lstm_SOL_USD_1m_7b5f0829.pt` - SOL LSTM (50 features) - OBSOLETE

**Feature Files** (in `data/features/`):
- `features_BTC-USD_1m_2025-11-13_50feat.parquet` - OBSOLETE
- `features_ETH-USD_1m_2025-11-13_50feat.parquet` - OBSOLETE
- `features_SOL-USD_1m_2025-11-13_50feat.parquet` - OBSOLETE

---

## What We Learned

### Key Insights

1. **Data Quality Matters More Than Model Complexity**
   - No amount of hyperparameter tuning can overcome noisy data
   - 50% accuracy = coin flip = model not learning meaningful patterns
   - Professional-grade data is essential for crypto trading

2. **Free Data Has Limitations**
   - Coinbase free API: 1-minute OHLCV aggregation loses critical information
   - Missing: Tick-level dynamics, order book depth, trade flow
   - Aggregation smooths out profitable signals

3. **Microstructure Matters for Short-Term Trading**
   - 15-minute prediction horizon requires tick-level data
   - Order book imbalance predicts short-term price moves
   - Order flow reveals institutional activity

### What Carried Forward to V5

**90% of Code Reusable**:
- ✅ Model architecture (LSTM, Transformer, RL stub)
- ✅ Training pipeline (data loading, train/val/test splits)
- ✅ Runtime system (ensemble, confidence calibration, FTMO rules)
- ✅ Infrastructure (AWS, Docker, systemd)
- ✅ Testing framework (unit, integration, smoke tests)

**What Changes in V5**:
- ❌ Data source: Free Coinbase → Tardis.dev professional
- ❌ Features: 31-50 → 53 features (adding microstructure)
- ❌ Input dimensions: Models need 53 features instead of 31-50

---

## Archive Actions

### Files to Keep (for reference)

Keep in git history:
- All V4 documentation (for lessons learned)
- Colab notebooks (may be useful for V5 GPU training)
- Feature engineering scripts (33 V4 features reused in V5)

### Files to Delete (local only)

Remove from working directory (keep in git history):
- [ ] `/tmp/colab_upload/` - V4 prepared files (654 MB)
- [ ] `models/new/*_7b5f0829.pt` - V4 trained models (12 MB)
- [ ] `data/features/*_50feat.parquet` - V4 feature files (643 MB)

**Total to reclaim**: ~1.3 GB disk space

### Git Handling

**DO NOT DELETE FROM GIT** - Keep for historical reference:
```bash
# All V4 work stays in git history
# Tag V4 final state for easy reference
git tag v4-obsolete-final HEAD
```

**Local cleanup** (optional):
```bash
# Remove V4 files from working directory (optional - saves 1.3 GB)
rm -rf /tmp/colab_upload/
rm -f models/new/*_7b5f0829.pt
rm -f data/features/*_50feat.parquet
```

---

## V4 → V5 Migration Guide

### For Future Reference

If someone needs to understand the V4 → V5 transition:

1. **Read**:
   - This file (`V4_OBSOLETE_ARCHIVE.md`)
   - `V5_PHASE1_PLAN.md` - V5 strategy
   - `PROJECT_MEMORY.md` - Updated Nov 15 with pivot details

2. **Compare**:
   - V4 models: `git show v4-obsolete-final:models/new/`
   - V5 models: Will be in `models/v5/` after Week 3

3. **Understand the pivot**:
   - Root cause: Free data quality
   - Solution: Professional data ($147-549/month)
   - Risk mitigation: 4-week validation before scaling

---

## Lessons for Future Phases

### Don't Repeat V4 Mistakes

1. **Validate data quality FIRST** before extensive model training
2. **Test with small sample** before committing to full 2-year training
3. **Compare data sources** early (free vs paid) to justify investment
4. **Set accuracy floor** (e.g., 55%) and abort if not met quickly

### When to Upgrade Data

**Symptoms**:
- Models stuck at ~50% accuracy (random performance)
- Hyperparameter tuning doesn't help
- Validation accuracy doesn't improve with more data

**Solution**:
- Upgrade to professional data (tick-level, order book)
- Add microstructure features
- Budget validation period before full commitment

---

## V4 Timeline

**November 10-13**: V4 Model Training
- Trained 3 LSTM models with 50 features
- Used free Coinbase OHLCV data
- Results: ~50% accuracy

**November 13-14**: V4 Colab Integration
- Created Colab notebooks for GPU evaluation
- Prepared files for upload (654 MB)
- Created integration guides

**November 15**: V4 Declared OBSOLETE
- QC Claude identified root cause (noisy free data)
- Decided on V5 pivot (Tardis.dev professional data)
- Updated all documentation to reflect V5 strategy

---

## V5 Comparison

### V4 vs V5

| Aspect | V4 (OBSOLETE) | V5 (NEW) |
|--------|---------------|----------|
| **Data Source** | Free Coinbase OHLCV | Tardis.dev tick + order book |
| **Data Cost** | $0/month | $147-549/month |
| **Data Quality** | Noisy 1-min aggregates | Professional tick-level |
| **Features** | 31-50 (OHLCV + technicals) | 53 (+ 20 microstructure) |
| **Accuracy** | ~50% (coin flip) | Target: 65-75% |
| **Status** | OBSOLETE | In Progress |
| **Budget** | N/A | $197/month (Phase 1) |
| **Timeline** | Completed | 4 weeks validation |

---

## Contact

**Questions about V4 work?**
- See `PROJECT_MEMORY.md` (updated Nov 15)
- See `V5_PHASE1_PLAN.md` for V5 strategy
- Check git history: `git log --before="2025-11-15"`

**V4 Final Commit**:
```bash
git log --before="2025-11-15" -1 --oneline
```

---

**Last Updated**: 2025-11-15
**Status**: OBSOLETE - Use V5 instead
**Owner**: Cloud Claude (Builder)
**Archived By**: Cloud Claude

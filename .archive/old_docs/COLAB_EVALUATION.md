# Google Colab GPU Evaluation Guide

## Overview

This guide explains how to evaluate the 50-feature LSTM models using Google Colab's free GPU (Tesla T4), which is **10-12x faster** than CPU evaluation.

**Time Comparison**:
- CPU (local): ~60-90 minutes for all 3 models
- GPU (Colab T4): ~5-10 minutes for all 3 models

## Files Prepared

All necessary files are ready in `/tmp/colab_upload/`:

**Models** (12 MB total):
- `lstm_BTC_USD_1m_7b5f0829.pt` (3.9 MB)
- `lstm_ETH_USD_1m_7b5f0829.pt` (3.9 MB)
- `lstm_SOL_USD_1m_7b5f0829.pt` (3.9 MB)

**Feature Datasets** (643 MB total):
- `features_BTC-USD_1m_2025-11-13_50feat.parquet` (228 MB)
- `features_ETH-USD_1m_2025-11-13_50feat.parquet` (218 MB)
- `features_SOL-USD_1m_2025-11-13_50feat.parquet` (198 MB)

## Step-by-Step Instructions

### 1. Upload Notebook to Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `colab_evaluate_50feat_models.ipynb` from this repo

### 2. Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **GPU type** to **T4** (default free tier)
4. Click **Save**

### 3. Upload Required Files

You have two options:

#### Option A: Direct Upload (Easier, but slower)

1. Click the **Files** icon in left sidebar
2. Create directories:
   - Right-click → **New folder** → `models/new`
   - Right-click → **New folder** → `data/features`
3. Upload model files to `models/new/`:
   - Upload all 3 `.pt` files from `/tmp/colab_upload/models/`
4. Upload feature files to `data/features/`:
   - Upload all 3 `.parquet` files from `/tmp/colab_upload/features/`

**Note**: Upload may take 5-10 minutes for 654 MB total.

#### Option B: Google Drive (Faster for large files)

1. Upload `/tmp/colab_upload/` directory to your Google Drive
2. In Colab, run the "Mount Google Drive" cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run the copy commands (uncomment in notebook):
   ```python
   !cp /content/drive/MyDrive/colab_upload/models/*.pt models/new/
   !cp /content/drive/MyDrive/colab_upload/features/*.parquet data/features/
   ```

### 4. Run Evaluation

1. Click **Runtime** → **Run all**
2. Wait for all cells to execute (~5-10 minutes)
3. Monitor progress in the output logs

### 5. Download Results

After evaluation completes, download the results:

1. Look for `evaluation_results.csv` in the Files panel
2. Right-click → **Download**
3. The CSV contains:
   - Accuracy for each model
   - Precision, Recall, F1 scores
   - Calibration error (ECE)
   - Promotion gate pass/fail status

## Expected Results

The notebook will evaluate all 3 models and output:

```
==================================================
📊 EVALUATION SUMMARY
==================================================

         symbol  accuracy  precision  recall    f1  calibration_error  num_samples
BTC-USD  BTC-USD    0.XXXX     0.XXXX  0.XXXX  0.XXXX            0.XXXX       154503
ETH-USD  ETH-USD    0.XXXX     0.XXXX  0.XXXX  0.XXXX            0.XXXX       154503
SOL-USD  SOL-USD    0.XXXX     0.XXXX  0.XXXX  0.XXXX            0.XXXX       154503
```

**Promotion Gates**:
- ✅ **Accuracy ≥ 68%**: Model passes accuracy threshold
- ✅ **Calibration ≤ 5%**: Model is well-calibrated
- ✅ **Overall**: Model ready for promotion to production

## Troubleshooting

### Out of Memory Error

If you get CUDA OOM error:
1. Reduce batch size in evaluation script (change `batch_size=64` → `batch_size=32`)
2. Restart runtime and try again

### GPU Not Available

If GPU is not detected:
1. Verify runtime type is set to GPU (Runtime → Change runtime type)
2. Restart runtime
3. Run the first cell to verify GPU availability

### Upload Failed

If file upload fails or times out:
1. Use Google Drive option instead
2. Or split uploads into smaller batches

## Performance Tips

1. **Use GPU**: T4 is 10-12x faster than CPU
2. **Batch Size**: Increase to 128 if memory allows for even faster evaluation
3. **Keep Runtime Active**: Colab disconnects after 90 minutes of inactivity

## Next Steps

After evaluation completes:

1. Review results in `evaluation_results.csv`
2. If models pass gates (≥68% accuracy, ≤5% calibration):
   - Promote models to `models/promoted/`
   - Integrate into runtime ensemble
   - Start observation period (3-5 days)
3. If models fail gates:
   - Analyze failure modes
   - Retrain with adjusted hyperparameters
   - Re-evaluate

## Files Created

- `colab_evaluate_50feat_models.ipynb`: Main evaluation notebook
- `prepare_colab_files.sh`: Helper script to prepare upload directory
- `/tmp/colab_upload/`: Directory with all files ready for upload (654 MB)

## Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [PyTorch GPU Support](https://pytorch.org/docs/stable/notes/cuda.html)
- Project: `PHASE6_5_RESTART_PLAN.md` for overall context

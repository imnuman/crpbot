# Claude + Colab Integration Guide

## Quick Setup (5 minutes)

### Step 1: Upload Files to Google Drive

This is the **fastest method** for large files (654 MB):

```bash
# On your local machine (this server):
# Files are already prepared in /tmp/colab_upload/

# Option A: Download via browser
# - Download /tmp/colab_upload/ to your local machine
# - Upload to Google Drive

# Option B: Use rclone (if configured)
# rclone copy /tmp/colab_upload/ gdrive:colab_upload/
```

### Step 2: Open Colab Notebook

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload `colab_evaluate_50feat_models.ipynb`
4. Or use this direct link approach:
   - Upload notebook to your Google Drive
   - Right-click → Open with → Google Colaboratory

### Step 3: Enable GPU

1. **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### Step 4: Run with Error Reporting

Add this cell at the top of your notebook:

```python
# Error reporter - paste errors to Claude
import traceback
import sys

def run_and_report(func):
    """Run function and print detailed error for Claude."""
    try:
        return func()
    except Exception as e:
        error_details = f"""
{'='*60}
❌ ERROR OCCURRED
{'='*60}

Function: {func.__name__}
Error Type: {type(e).__name__}
Error Message: {str(e)}

Full Traceback:
{traceback.format_exc()}

{'='*60}
COPY THIS ENTIRE BLOCK AND PASTE TO CLAUDE
{'='*60}
        """
        print(error_details)

        # Also save to file for easy download
        with open('error_report.txt', 'w') as f:
            f.write(error_details)

        print("\n📄 Error saved to error_report.txt - download and share with Claude")
        raise
```

Then wrap your execution:

```python
def main():
    # Your evaluation code
    exec(open('evaluate_gpu.py').read())

run_and_report(main)
```

### Step 5: Alternative - Full Automation with Claude API

If you want **zero manual intervention**, use the Anthropic API:

```python
# Install SDK
!pip install anthropic

# Configure (get API key from https://console.anthropic.com/)
import os
from anthropic import Anthropic

# Option A: Direct API key
client = Anthropic(api_key="sk-ant-...")

# Option B: Colab secrets (more secure)
from google.colab import userdata
client = Anthropic(api_key=userdata.get('ANTHROPIC_API_KEY'))

# Auto-fixer function
def auto_fix_and_retry(code_str, max_attempts=3):
    """Execute code and auto-fix errors using Claude."""
    for attempt in range(max_attempts):
        try:
            exec(code_str)
            print(f"✅ Success on attempt {attempt + 1}")
            return
        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            error_msg = traceback.format_exc()
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            print("🤖 Asking Claude for fix...")

            # Ask Claude to fix the code
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": f"""Fix this Python code that's failing in Google Colab:

Code:
```python
{code_str}
```

Error:
```
{error_msg}
```

Provide ONLY the fixed code, no explanations."""
                }]
            )

            code_str = response.content[0].text.strip('`').strip('python\n')
            print(f"🔧 Claude suggested fix, retrying...")

# Use it
code = '''
import torch
print(torch.cuda.is_available())
# Your evaluation code here
'''

auto_fix_and_retry(code)
```

## Complete Workflow Examples

### Example 1: Manual (Free)

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy files
!mkdir -p models/new data/features
!cp /content/drive/MyDrive/colab_upload/models/*.pt models/new/
!cp /content/drive/MyDrive/colab_upload/features/*.parquet data/features/

# 3. Run evaluation
!python evaluate_gpu.py 2>&1 | tee evaluation.log

# If error occurs:
# - Download evaluation.log
# - Paste in Claude chat
# - Get fix
# - Apply fix
# - Re-run
```

### Example 2: Semi-Automated (Low Cost)

```python
# Only call Claude API when errors occur
import anthropic
import traceback

client = anthropic.Anthropic(api_key=userdata.get('ANTHROPIC_API_KEY'))

try:
    !python evaluate_gpu.py
except Exception as e:
    # Ask Claude once
    error = traceback.format_exc()
    print("🤖 Asking Claude for solution...")

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"GPU model evaluation failed in Colab:\n\n{error}\n\nProvide solution."
        }]
    )

    print("\n💡 Claude's solution:")
    print(response.content[0].text)
```

### Example 3: Fully Automated (Hands-off)

Use the `auto_fix_and_retry()` function above with your evaluation code.

## Cost Estimation (API Option)

**Anthropic API Pricing** (as of 2024):
- Claude Sonnet 4.5: $3 per million input tokens, $15 per million output tokens
- Typical error fix: ~1000 tokens input + 500 tokens output = $0.01
- **Total cost for full evaluation with 2-3 errors: < $0.05**

Very cheap for automated debugging!

## Troubleshooting

### "Module not found" errors

```python
# Install missing packages
!pip install loguru pandas pyarrow scikit-learn torch
```

### "CUDA out of memory"

```python
# Reduce batch size
# In evaluate_gpu.py, change:
# batch_size = 64  →  batch_size = 32
```

### "Files not found"

```python
# Verify files uploaded correctly
!ls -lh models/new/
!ls -lh data/features/

# Should see:
# models/new/: 3 files (3.9 MB each)
# data/features/: 3 files (198-228 MB each)
```

### "Runtime disconnected"

Colab free tier disconnects after:
- 90 minutes of inactivity
- 12 hours of continuous use

**Solution**: Run a keep-alive script:

```python
# Keep Colab session alive
import time
from IPython.display import Javascript

def keep_alive():
    while True:
        display(Javascript('console.log("keep alive")'))
        time.sleep(60)

# Run in background
import threading
threading.Thread(target=keep_alive, daemon=True).start()
```

## Best Practice Workflow

**Recommended approach for this project**:

1. **Upload files via Google Drive** (fastest for 654 MB)
2. **Use manual error reporting** (free, simple)
3. **If >3 errors occur, switch to API** (automated)

This gives you free execution with the option to automate if needed.

## Next Steps

1. Upload `/tmp/colab_upload/` to your Google Drive
2. Open `colab_evaluate_50feat_models.ipynb` in Colab
3. Run cells sequentially
4. If errors occur, report back here
5. I'll provide immediate fixes

**Time estimate**:
- Setup: 5-10 minutes
- Execution: 5-10 minutes (GPU)
- **Total: 15-20 minutes** vs 60+ minutes on CPU

## Files Created

- `colab_evaluate_50feat_models.ipynb` - Main evaluation notebook
- `colab_with_claude_api.ipynb` - API-integrated version
- `COLAB_INTEGRATION_GUIDE.md` - This guide
- `/tmp/colab_upload/` - Ready-to-upload files (654 MB)

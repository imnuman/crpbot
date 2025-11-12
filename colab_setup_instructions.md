# Google Colab Pro Setup for CRPBot

## Step 1: Upload Notebook to Colab
1. Go to https://colab.research.google.com/
2. Click "Upload" and select `colab_gpu_training.ipynb`
3. Or create new notebook and copy the cells

## Step 2: Enable GPU Runtime
1. In Colab: Runtime → Change runtime type
2. Hardware accelerator: **GPU** (T4 or V100)
3. Click Save

## Step 3: Set AWS Credentials
Replace in the notebook:
```python
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_ACCESS_KEY_HERE'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_KEY_HERE'
```

**Get your credentials:**
```bash
# In your local terminal:
cat ~/.aws/credentials
```

## Step 4: Upload Sample Data (if needed)
```bash
# Upload some sample data to S3 first
cd /home/numan/crpbot
source .venv/bin/activate

# Create sample data
python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample OHLCV data
dates = pd.date_range(start='2024-01-01', end='2024-11-01', freq='1H')
n = len(dates)

for coin in ['btc', 'eth', 'ada', 'sol']:
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(20000, 80000, n),
        'high': np.random.uniform(20000, 80000, n),
        'low': np.random.uniform(20000, 80000, n),
        'close': np.random.uniform(20000, 80000, n),
        'volume': np.random.uniform(1000, 10000, n)
    })
    data.to_csv(f'{coin}_data.csv', index=False)
    print(f'Created {coin}_data.csv')
"

# Upload to S3
aws s3 cp btc_data.csv s3://crpbot-market-data-dev/
aws s3 cp eth_data.csv s3://crpbot-market-data-dev/
aws s3 cp ada_data.csv s3://crpbot-market-data-dev/
aws s3 cp sol_data.csv s3://crpbot-market-data-dev/
```

## Step 5: Run Training in Colab
1. Run each cell sequentially
2. Training takes ~10 minutes with GPU
3. Models automatically upload to S3

## Step 6: Download Models to Local
```bash
# After Colab training completes
aws s3 sync s3://crpbot-market-data-dev/models/gpu_trained/ ./models/gpu_trained/
```

## Expected Output
- 4 LSTM models (BTC, ETH, ADA, SOL)
- Training time: ~10 minutes (vs 50+ hours CPU)
- Models saved to S3 automatically
- Ready for production runtime

## Cost
- Colab Pro: $10/month
- Training session: ~$0.50 per run
- Much cheaper than AWS GPU instances while waiting for quota

# V7 Scaling Plan: More Data + Better Infrastructure

## 🎯 Scaling Objectives
- **10x Data**: From 7K to 70K+ data points per symbol
- **10x Symbols**: From 3 to 30+ cryptocurrencies  
- **Better Models**: Improved accuracy through scale
- **Cost Efficiency**: Optimize training costs

## 📊 Data Scaling Strategy

### Phase 1: Expand Current Symbols (Immediate)
```python
# Current: 7,122 hourly candles (10 months)
# Target: 26,280 hourly candles (3 years)

symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
timeframe = '1h'
history = '3y'  # 3 years vs current 10 months
expected_data_points = 26280  # 3.7x increase
```

### Phase 2: Add More Symbols (Week 1)
```python
# Expand to top 30 cryptocurrencies
major_symbols = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD',
    'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD',
    'BCH-USD', 'XLM-USD', 'ALGO-USD', 'ATOM-USD', 'FIL-USD',
    'ICP-USD', 'VET-USD', 'THETA-USD', 'EOS-USD', 'TRX-USD',
    'XTZ-USD', 'AAVE-USD', 'MKR-USD', 'COMP-USD', 'SNX-USD',
    'YFI-USD', 'SUSHI-USD', 'CRV-USD', 'BAL-USD', 'REN-USD'
]
total_symbols = 30  # 10x increase
```

### Phase 3: Higher Frequency Data (Week 2)
```python
# Add 15-minute candles for better granularity
timeframes = ['1h', '15m']  # 4x more data points
data_points_per_symbol = 26280 * 4 = 105120  # ~100K per symbol
total_data_points = 105120 * 30 = 3.15M  # 3+ million data points
```

## 🚀 Infrastructure Options

### Option A: SageMaker Training Jobs (Recommended)

#### Advantages
- **Managed Infrastructure**: No GPU driver setup
- **Auto-scaling**: Handle 30 symbols automatically  
- **Spot Instances**: 70% cost savings
- **Experiment Tracking**: Built-in MLflow integration
- **Model Registry**: Automatic versioning
- **Distributed Training**: Multi-GPU for large models

#### SageMaker Setup
```python
# SageMaker training job configuration
training_config = {
    'instance_type': 'ml.g5.2xlarge',  # 1 GPU, 8 vCPUs, 32GB RAM
    'instance_count': 1,
    'use_spot_instances': True,  # 70% cost savings
    'max_runtime_seconds': 86400,  # 24 hours max
    'volume_size_gb': 100,
    'framework': 'pytorch',
    'framework_version': '2.0.1',
    'python_version': 'py310'
}

# Cost estimate
spot_price = 0.42  # $/hour (70% off $1.408)
training_time = 8  # hours for 30 symbols
total_cost = spot_price * training_time = $3.36
```

#### SageMaker Training Script
```python
# sagemaker_train.py
import sagemaker
from sagemaker.pytorch import PyTorch

# Define training job
estimator = PyTorch(
    entry_point='v7_enhanced_training.py',
    source_dir='.',
    role=sagemaker.get_execution_role(),
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    use_spot_instances=True,
    max_wait=86400,
    max_run=43200,
    framework_version='2.0.1',
    py_version='py310',
    hyperparameters={
        'epochs': 100,
        'batch_size': 256,
        'learning_rate': 0.001,
        'symbols': 30
    }
)

# Start training
estimator.fit({'training': 's3://crpbot-data/training/'})
```

### Option B: Multi-Instance EC2 (Cost-Optimized)

#### Parallel Training Setup
```python
# Launch multiple instances for parallel training
symbols_per_instance = 10
instances_needed = 3  # 30 symbols / 10 per instance
instance_type = 'g5.xlarge'  # $1.006/hour
spot_price = 0.30  # $/hour (70% savings)

total_cost = spot_price * instances_needed * 6  # 6 hours
# = $0.30 * 3 * 6 = $5.40 total
```

### Option C: Hybrid Approach (Best of Both)

#### Development: EC2 Spot
- **Experimentation**: Use EC2 spot for quick iterations
- **Cost**: ~$0.30/hour for development

#### Production Training: SageMaker
- **Scale Training**: Use SageMaker for final 30-symbol training
- **MLOps**: Leverage experiment tracking and model registry

## 💰 Cost Analysis

### Current V7 Training
```
Instance: g5.xlarge on-demand
Cost: $1.006/hour * 1 hour = $1.01
Symbols: 3
Data points: 21,366 total
Cost per symbol: $0.34
```

### Scaled Training Options

#### Option A: SageMaker Spot
```
Instance: ml.g5.2xlarge spot
Cost: $0.42/hour * 8 hours = $3.36
Symbols: 30
Data points: 3,150,000 total
Cost per symbol: $0.11 (3x cheaper per symbol!)
```

#### Option B: EC2 Multi-Instance Spot
```
Instances: 3x g5.xlarge spot
Cost: $0.30/hour * 3 * 6 hours = $5.40
Symbols: 30
Data points: 3,150,000 total
Cost per symbol: $0.18
```

## 📋 Implementation Plan

### Week 1: Data Collection
1. **Expand History**: Collect 3 years of hourly data
2. **Add Symbols**: Gather data for 30 major cryptocurrencies
3. **Data Pipeline**: Automate collection and preprocessing

### Week 2: Infrastructure Setup
1. **SageMaker Setup**: Configure training jobs and S3 storage
2. **Training Scripts**: Adapt V7 code for SageMaker
3. **Experiment Tracking**: Set up MLflow integration

### Week 3: Scale Training
1. **Batch Training**: Train all 30 symbols
2. **Hyperparameter Tuning**: Optimize for each symbol
3. **Model Evaluation**: Compare results across symbols

### Week 4: Production Deployment
1. **Model Registry**: Deploy best models
2. **A/B Testing**: Compare scaled models vs V7
3. **Performance Monitoring**: Track live trading results

## 🎯 Expected Improvements

### Data Scale Impact
```
Current V7: 7K data points → 70.2% accuracy
Scaled V8: 100K+ data points → Expected 75-80% accuracy
Improvement: +5-10% accuracy from scale alone
```

### Symbol Diversification
```
Current: 3 symbols (limited market coverage)
Scaled: 30 symbols (comprehensive crypto market)
Benefit: Reduced correlation risk, better market representation
```

## 🚀 Recommendation: SageMaker Approach

### Why SageMaker is Better for Scaling
1. **Managed Infrastructure**: No GPU driver headaches
2. **Auto-scaling**: Handle 30 symbols seamlessly
3. **Cost Efficiency**: Spot instances save 70%
4. **MLOps Integration**: Experiment tracking built-in
5. **Production Ready**: Model registry and deployment

### Next Steps
1. **Set up SageMaker**: Configure training environment
2. **Data Pipeline**: Collect 30 symbols × 3 years data
3. **Batch Training**: Train all models in parallel
4. **Deploy Best Models**: Replace V7 with scaled V8

**Estimated Timeline**: 2-3 weeks
**Estimated Cost**: $10-20 total (vs $300+ for on-demand)
**Expected Accuracy**: 75-80% (vs current 70.2%)

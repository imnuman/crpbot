# AWS Cost Tracking - CRPBot Project

## Current Infrastructure (Created Nov 8, 2025)

### S3 Storage
- `crpbot-market-data-dev` - Market data, OHLCV feeds
- `crpbot-logs-dev` - Application logs, training logs  
- `crpbot-backups-dev` - Model weights, configs, DB backups

**Estimated Monthly Cost: $2-8**
- Standard storage: $0.023/GB
- Expected usage: 50-200GB initially

## Planned Infrastructure & Costs

### Database
- **RDS PostgreSQL (t3.micro)**: $15.33/month
  - Trade history, signals, performance metrics
  - 20GB storage included

### Compute
- **Lambda Functions**: $3-10/month
  - Signal generation, Telegram bot
  - 1M requests/month estimate
  
- **ECS Fargate (0.25 vCPU, 0.5GB)**: $12.24/month
  - Runtime loop, continuous monitoring
  - Alternative to Lambda for long-running tasks

### Monitoring & Security
- **CloudWatch**: $5-15/month
  - Logs: $0.50/GB ingested
  - Metrics: $0.30/metric/month
  - Alarms: $0.10/alarm/month

- **Secrets Manager**: $2.40/month
  - API keys (Binance, Telegram, FTMO)
  - 4 secrets × $0.40/secret + $0.05/10K API calls

### GPU Training (On-Demand)
- **SageMaker ml.g4dn.xlarge**: $0.736/hour
  - 1x NVIDIA T4 GPU, 4 vCPUs, 16GB RAM
  - LSTM training: 2-4 hours = $1.47-2.94/run
  
- **SageMaker ml.g4dn.2xlarge**: $1.472/hour  
  - 1x NVIDIA T4 GPU, 8 vCPUs, 32GB RAM
  - Transformer training: 4-8 hours = $5.89-11.78/run

- **EC2 g4dn.xlarge**: $0.526/hour
  - Same specs as SageMaker but requires setup
  - 50% cheaper than SageMaker

### GPU Training (Spot Instances - 70% Savings)
- **EC2 g4dn.xlarge Spot**: ~$0.16/hour
  - LSTM training: 2-4 hours = $0.32-0.64/run
  - Risk: Can be interrupted

- **SageMaker Spot**: ~$0.22/hour
  - More reliable than EC2 spot
  - Automatic checkpointing

### Container Registry
- **ECR**: $1-3/month
  - Docker images for training/runtime
  - $0.10/GB/month

### Data Transfer
- **Data Transfer**: $2-5/month
  - API calls, real-time data feeds
  - First 1GB free, then $0.09/GB

## Cost Summary

| Service | Monthly Cost | Annual Cost |
|---------|-------------|-------------|
| S3 Storage | $2-8 | $24-96 |
| RDS PostgreSQL | $15 | $180 |
| Lambda/ECS | $12-15 | $144-180 |
| CloudWatch | $5-15 | $60-180 |
| Secrets Manager | $2.40 | $29 |
| ECR | $1-3 | $12-36 |
| Data Transfer | $2-5 | $24-60 |
| **GPU Training** | **$20-80** | **$240-960** |
| **TOTAL** | **$59-143** | **$713-1721** |

## GPU Training Cost Breakdown

### Training Schedule (Estimated)
- **LSTM per coin**: 2x/week × 4 coins = 8 runs/month
- **Transformer**: 1x/week = 4 runs/month  
- **RL PPO**: 2x/month = 2 runs/month

### Cost Scenarios

#### Scenario 1: SageMaker On-Demand (Fast & Reliable)
- LSTM: 8 runs × 3 hours × $0.736 = $17.66/month
- Transformer: 4 runs × 6 hours × $1.472 = $35.33/month
- RL: 2 runs × 12 hours × $1.472 = $35.33/month
- **Total: $88.32/month**

#### Scenario 2: EC2 Spot (70% Cheaper, Risk of Interruption)
- LSTM: 8 runs × 3 hours × $0.16 = $3.84/month
- Transformer: 4 runs × 6 hours × $0.32 = $7.68/month  
- RL: 2 runs × 12 hours × $0.32 = $7.68/month
- **Total: $19.20/month**

#### Scenario 3: Hybrid (Spot + On-Demand Backup)
- 80% Spot success rate
- Spot: $19.20 × 0.8 = $15.36/month
- On-demand backup: $88.32 × 0.2 = $17.66/month
- **Total: $33.02/month**

## Cost Optimization Strategies

### GPU Training Optimization
- **Use Spot Instances**: 70% cost savings (g4dn.xlarge: $0.16/hr vs $0.526/hr)
- **Checkpointing**: Save progress every 30 minutes to handle interruptions
- **Batch Training**: Train multiple coins in single session
- **Off-peak Hours**: Schedule training during low-demand periods
- **Model Caching**: Store intermediate results in S3 to resume faster

### Development Phase
- Use t3.micro RDS (free tier eligible for 12 months)
- Lambda over ECS for intermittent tasks
- S3 Intelligent Tiering for market data
- CloudWatch log retention: 7 days for dev
- **Start with Spot instances for training**

### Production Phase
- RDS Reserved Instance (30-60% savings)
- S3 Glacier for historical data (90% cheaper)
- CloudWatch log retention: 30 days
- Spot instances for training workloads

## Monitoring & Alerts

### Cost Budgets
- **Development**: $25/month (80% alert at $20)
- **Production**: $75/month (80% alert at $60)

### Resource Tagging Strategy
```
Project: crpbot
Environment: dev|prod|staging
Component: data|compute|storage|monitoring
Owner: ncldev
```

### Weekly Cost Review
- Check AWS Cost Explorer every Friday
- Review resource utilization
- Identify cost anomalies
- Optimize underutilized resources

## Risk Factors

### Potential Cost Spikes
- **Market data volume**: High volatility = more data
- **Training frequency**: Model retraining costs
- **API rate limits**: Increased Lambda invocations
- **Log volume**: Debug mode in production

### Mitigation
- Set CloudWatch billing alarms
- Implement circuit breakers for API calls
- Use S3 lifecycle policies
- Monitor Lambda concurrent executions

## Phase Rollout Costs

### Phase 1: MVP (Months 1-2)
- S3 + Lambda + Secrets Manager
- **Estimated**: $10-15/month

### Phase 2: Production (Months 3-6)
- Add RDS + CloudWatch + ECS
- **Estimated**: $40-60/month

### Phase 3: Scale (Months 6+)
- Reserved instances + optimization
- **Estimated**: $30-45/month

## Notes
- Costs based on us-east-1 pricing (Nov 2025)
- Free tier benefits applied where applicable
- Estimates assume moderate usage patterns
- Review and update monthly based on actual usage

## Last Updated
November 10, 2025

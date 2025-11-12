# Kafka Architecture for Real-Time ML Trading

**Version**: V2 Advanced Path
**Status**: Implementation in Progress
**Goal**: Real-time data streaming for maximum prediction accuracy

---

## 🎯 Architecture Overview

### Core Principles
1. **Real-time feature engineering**: Stream processing for instant feature calculation
2. **Low-latency predictions**: <100ms from market data to signal
3. **Multi-source fusion**: Combine price, sentiment, news, on-chain data
4. **Scalable inference**: Parallel model workers for multiple symbols
5. **Fault tolerance**: Kafka persistence + consumer group failover

---

## 📊 Kafka Topics

### 1. Market Data (Inputs)
```
market.candles.{symbol}.1m     - 1-minute OHLCV candles (real-time)
market.candles.{symbol}.5m     - 5-minute OHLCV candles
market.candles.{symbol}.15m    - 15-minute OHLCV candles
market.candles.{symbol}.1h     - 1-hour OHLCV candles
market.orderbook.{symbol}      - Order book snapshots (optional, high volume)
market.trades.{symbol}         - Individual trades (optional, high volume)
```

**Schema**: Avro/Protobuf for efficiency
```json
{
  "timestamp": "2025-11-10T16:30:00Z",
  "symbol": "BTC-USD",
  "interval": "1m",
  "open": 89234.50,
  "high": 89456.20,
  "low": 89123.10,
  "close": 89345.80,
  "volume": 123.456,
  "source": "coinbase"
}
```

### 2. Features (Processed)
```
features.raw.{symbol}          - Base features (OHLCV + time)
features.technical.{symbol}    - Technical indicators (RSI, MACD, Bollinger)
features.multi_tf.{symbol}     - Multi-timeframe alignment features
features.sentiment.{symbol}    - Sentiment scores (optional)
features.aggregated.{symbol}   - Final feature vector for ML
```

**Schema**:
```json
{
  "timestamp": "2025-11-10T16:30:00Z",
  "symbol": "BTC-USD",
  "features": {
    "5m_close": 89345.80,
    "15m_close": 89456.20,
    "1h_close": 89123.10,
    "tf_alignment_score": 0.75,
    "volatility_regime": "medium",
    "rsi_14": 67.3,
    "macd_signal": 0.5,
    // ... 50+ features
  },
  "version": "v2.1"
}
```

### 3. Predictions (ML Inference)
```
predictions.lstm.{symbol}      - Per-coin LSTM predictions
predictions.transformer        - Multi-coin transformer predictions
predictions.ensemble.{symbol}  - Weighted ensemble predictions
```

**Schema**:
```json
{
  "timestamp": "2025-11-10T16:30:00Z",
  "symbol": "BTC-USD",
  "model_type": "lstm",
  "model_version": "a7aff5c4",
  "prediction": {
    "direction": 1,           // -1, 0, 1
    "probability": 0.73,      // Raw model output
    "confidence": 0.68        // Calibrated confidence
  },
  "inference_time_ms": 12
}
```

### 4. Signals (Trading Decisions)
```
signals.raw                    - Before confidence calibration
signals.calibrated             - After confidence + FREE boosters
signals.filtered               - After filters (alignment, volatility)
signals.final                  - Ready for execution
```

**Schema**:
```json
{
  "timestamp": "2025-11-10T16:30:00Z",
  "symbol": "BTC-USD",
  "signal_id": "uuid-here",
  "direction": "LONG",
  "confidence": 0.68,
  "tier": "A",
  "entry_price": 89345.80,
  "stop_loss": 88900.00,
  "take_profit": 89900.00,
  "position_size": 0.05,
  "features_snapshot": {...},
  "ensemble_weights": {"lstm": 0.35, "transformer": 0.40, "rl": 0.25},
  "boosters_applied": ["tf_alignment", "volatility_adaptive"],
  "ttl_seconds": 300
}
```

### 5. Execution (Orders & Trades)
```
trades.orders.pending          - Orders submitted to broker
trades.orders.confirmed        - Orders confirmed by broker
trades.executions              - Actual fills
trades.pnl                     - Realized P&L events
```

**Schema**:
```json
{
  "timestamp": "2025-11-10T16:30:05Z",
  "signal_id": "uuid-here",
  "order_id": "mt5-12345",
  "symbol": "BTC-USD",
  "side": "BUY",
  "order_type": "MARKET",
  "quantity": 0.05,
  "price": 89345.80,
  "status": "FILLED",
  "fill_price": 89346.10,
  "slippage_bps": 0.3,
  "latency_ms": 87,
  "session": "ny"
}
```

### 6. Metrics & Monitoring
```
metrics.performance.realtime   - Win rate, Sharpe, drawdown (sliding window)
metrics.model.drift            - Model performance degradation detection
metrics.latency                - End-to-end latency tracking
metrics.alerts                 - System alerts and anomalies
```

### 7. Sentiment (Optional V2+)
```
sentiment.news.{symbol}        - News sentiment scores
sentiment.social.{symbol}      - Twitter/Reddit sentiment
sentiment.onchain              - On-chain metrics (if crypto)
```

---

## 🏗️ Component Architecture

### Producer Services

#### 1. Market Data Ingester (`apps/kafka/producers/market_data.py`)
- **Input**: Coinbase WebSocket, REST API polling
- **Output**: `market.candles.*` topics
- **Rate**: 1-minute candles for all symbols
- **Features**:
  - Multi-timeframe aggregation (1m → 5m → 15m → 1h)
  - Deduplication and ordering guarantees
  - Health checks and reconnection logic

#### 2. Feature Engineering Stream (`apps/kafka/streams/feature_engineer.py`)
- **Input**: `market.candles.*`
- **Output**: `features.*` topics
- **Processing**:
  - Real-time technical indicators
  - Multi-TF feature alignment
  - Cross-TF correlation scores
  - Volatility regime classification
- **Tech**: Kafka Streams or Python with windowing

### Consumer Services

#### 3. Model Inference Workers (`apps/kafka/consumers/model_inference.py`)
- **Input**: `features.aggregated.{symbol}`
- **Output**: `predictions.*` topics
- **Architecture**:
  - Separate consumer group per model type (LSTM, Transformer)
  - Parallel workers for load distribution
  - Model version management (hot reload)
  - GPU inference if available

#### 4. Signal Aggregator (`apps/kafka/consumers/signal_aggregator.py`)
- **Input**: `predictions.*`
- **Output**: `signals.*` topics
- **Processing**:
  - Ensemble weighted averaging
  - Confidence calibration
  - FREE boosters application
  - Tier assignment (A/B/C/D)
  - Signal filtering and validation

#### 5. Execution Engine (`apps/kafka/consumers/execution_engine.py`)
- **Input**: `signals.final`
- **Output**: `trades.orders.*`
- **Features**:
  - FTMO guardrails enforcement
  - Position sizing with Kelly criterion
  - Risk management (stop loss, take profit)
  - MT5 bridge integration
  - Order retry logic

#### 6. Metrics Collector (`apps/kafka/consumers/metrics_collector.py`)
- **Input**: All topics (consumer for monitoring)
- **Output**: `metrics.*`, database writes, Telegram alerts
- **Metrics**:
  - End-to-end latency (market data → signal → execution)
  - Model accuracy tracking (live predictions vs actual outcomes)
  - P&L tracking (real-time + cumulative)
  - System health (Kafka lag, consumer status, errors)

---

## 🔄 Data Flow

```
1. Market Data (Coinbase)
   ↓ WebSocket/REST
2. Market Data Ingester → market.candles.*
   ↓ Kafka Stream Processing
3. Feature Engineering → features.*
   ↓ Kafka Consumer
4. Model Inference Workers (LSTM/Transformer) → predictions.*
   ↓ Kafka Consumer
5. Signal Aggregator → signals.*
   ↓ Kafka Consumer
6. Execution Engine → trades.orders.*
   ↓ MT5 Bridge
7. Trade Execution (FTMO Account)
   ↓ Kafka Producer
8. Execution Confirmations → trades.executions
   ↓ Kafka Consumer
9. Metrics Collector → Database + Telegram

Cross-cutting:
- Metrics Collector consumes from all topics
- Monitoring Dashboard (Grafana) consumes metrics.*
```

---

## ⚡ Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Market data latency | <50ms | <100ms |
| Feature engineering | <20ms | <50ms |
| Model inference (per model) | <30ms | <100ms |
| Signal aggregation | <10ms | <50ms |
| End-to-end (data → signal) | <150ms | <500ms |
| Kafka consumer lag | <100ms | <1s |
| Message throughput | >10k msg/s | >1k msg/s |

---

## 🛡️ Fault Tolerance

### Kafka Configuration
- **Replication factor**: 3 (for production)
- **Min in-sync replicas**: 2
- **Retention**: 7 days (for replay and debugging)
- **Compression**: LZ4 (balance between speed and size)
- **Partitions**: 3 per topic (parallelism)

### Consumer Groups
- **Auto-commit**: Disabled (manual commit after processing)
- **Rebalance strategy**: Cooperative sticky
- **Session timeout**: 30s
- **Max poll interval**: 5 minutes

### Error Handling
- **Dead letter queue**: `dlq.*` topics for failed messages
- **Retry policy**: Exponential backoff (3 attempts)
- **Circuit breaker**: Stop consuming if downstream (MT5) is down
- **Alerting**: Telegram + PagerDuty for critical failures

---

## 📦 Deployment

### Docker Compose (Development)
```yaml
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  schema-registry:
    image: confluentinc/cp-schema-registry:7.5.0
    environment:
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
```

### Production (AWS MSK or Self-Hosted)
- **Option 1**: AWS MSK (Managed Kafka) - ~$150/month
- **Option 2**: Self-hosted EC2 (3-node cluster) - ~$100/month
- **Option 3**: Confluent Cloud - ~$200/month (fully managed)

**Recommendation**: Self-hosted for cost, AWS MSK for reliability

---

## 📊 Monitoring & Observability

### Metrics to Track
1. **Kafka Broker Metrics**:
   - Under-replicated partitions
   - Offline partitions
   - Broker disk usage
   - Network throughput

2. **Consumer Metrics**:
   - Consumer lag (per partition)
   - Messages consumed per second
   - Processing time per message
   - Error rate

3. **Producer Metrics**:
   - Messages produced per second
   - Producer errors
   - Batch size
   - Compression ratio

4. **Application Metrics**:
   - Model inference latency
   - Prediction accuracy (live vs actual)
   - Signal generation rate
   - Trade execution success rate

### Dashboards
- **Grafana**: Real-time metrics visualization
- **Kafka UI**: Topic and consumer group monitoring
- **Custom Dashboard**: Trading performance and ML metrics

---

## 🚀 Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up Kafka cluster (Docker Compose for dev)
- [ ] Create topic schemas and configurations
- [ ] Implement market data ingester
- [ ] Basic monitoring setup

### Phase 2: Feature Engineering (Week 1-2)
- [ ] Kafka Streams feature engineering pipeline
- [ ] Multi-TF feature alignment (real-time)
- [ ] Technical indicators calculation
- [ ] Feature validation and quality checks

### Phase 3: Model Integration (Week 2)
- [ ] Model inference workers (LSTM + Transformer)
- [ ] Signal aggregator with ensemble logic
- [ ] Confidence calibration pipeline
- [ ] FREE boosters integration

### Phase 4: Execution & Testing (Week 2-3)
- [ ] Execution engine with FTMO guardrails
- [ ] MT5 bridge integration
- [ ] End-to-end testing (dev environment)
- [ ] Performance benchmarking

### Phase 5: Production Deployment (Week 3-4)
- [ ] Production Kafka cluster setup
- [ ] Load testing and optimization
- [ ] Monitoring and alerting setup
- [ ] Silent observation with Kafka (Phase 6.5)

---

## 💰 Cost Estimate

### Development (Self-Hosted)
- **Infrastructure**: $0 (Docker on existing machine)
- **Development time**: 3-4 weeks

### Production
- **Kafka cluster (self-hosted)**: $100-150/month (3 EC2 instances)
- **Monitoring (Grafana Cloud)**: $0-50/month
- **Total**: $100-200/month additional

### Alternative (Managed)
- **AWS MSK**: $150-300/month
- **Confluent Cloud**: $200-400/month

**Recommendation**: Start self-hosted, migrate to managed if scaling needed

---

## 🔐 Security

- **Authentication**: SASL/SCRAM for Kafka
- **Encryption**: TLS for all Kafka traffic
- **Authorization**: ACLs for topic access control
- **Secrets management**: AWS Secrets Manager or Vault
- **API keys**: Never committed to code, env vars only

---

## 📚 Additional Resources

- **Kafka Streams**: For real-time feature engineering
- **Avro/Protobuf**: Efficient message serialization
- **Schema Registry**: Manage schema evolution
- **Kafka Connect**: Integrate with external systems (if needed)
- **KSQL**: SQL-like queries on Kafka streams (optional)

---

## ✅ Success Criteria

1. **Latency**: <150ms end-to-end (market data → signal)
2. **Throughput**: Handle 3 symbols × 4 timeframes = 12 streams
3. **Reliability**: 99.9% uptime, zero data loss
4. **Scalability**: Easy to add new symbols/models
5. **Observability**: Full visibility into system performance

---

**Next Steps**: Implement Phase 1 (Core Infrastructure) while V2 models train

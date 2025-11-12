# Kafka Setup Guide

## Prerequisites

### Install Docker and Docker Compose

Ubuntu/Debian:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose (if not included)
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

## Quick Start

### 1. Start Kafka Cluster

```bash
# Start all services (Kafka, Zookeeper, Schema Registry, Kafka UI)
docker compose -f docker-compose.kafka.yml up -d

# Check service status
docker compose -f docker-compose.kafka.yml ps

# View logs
docker compose -f docker-compose.kafka.yml logs -f kafka
```

### 2. Verify Kafka is Running

```bash
# Check Kafka broker
docker exec -it crpbot-kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# List topics
docker exec -it crpbot-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

### 3. Access Kafka UI

Open your browser and navigate to: http://localhost:8080

You should see the Kafka UI dashboard with:
- Broker information
- Topics
- Consumer groups
- Messages

### 4. Create Topics

Topics are auto-created when first message is produced, but you can create them manually:

```bash
# Create market.candles.BTC-USD.1m topic
docker exec -it crpbot-kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --create \
  --topic market.candles.BTC-USD.1m \
  --partitions 3 \
  --replication-factor 1 \
  --config compression.type=lz4 \
  --config retention.ms=604800000
```

Or use the Python script:

```bash
uv run python scripts/kafka/create_topics.py
```

### 5. Start Market Data Ingester

```bash
# Start ingesting market data from Coinbase
uv run python apps/kafka/producers/market_data.py
```

This will:
- Connect to Coinbase API
- Fetch 1-minute candles every minute
- Produce to `market.candles.{symbol}.1m` topics

### 6. Test with Consumer

```bash
# Consume messages from BTC-USD topic
docker exec -it crpbot-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic market.candles.BTC-USD.1m \
  --from-beginning \
  --property print.key=true \
  --property key.separator=" : "
```

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Kafka | 9092 | Kafka broker (external) |
| Kafka | 9093 | Kafka broker (internal) |
| Zookeeper | 2181 | Zookeeper client port |
| Schema Registry | 8081 | Schema management |
| Kafka UI | 8080 | Web-based Kafka UI |
| Kafka Connect | 8083 | Kafka Connect REST API |

## Stopping Services

```bash
# Stop all services
docker compose -f docker-compose.kafka.yml down

# Stop and remove volumes (WARNING: deletes all data)
docker compose -f docker-compose.kafka.yml down -v
```

## Troubleshooting

### Kafka won't start

Check logs:
```bash
docker compose -f docker-compose.kafka.yml logs kafka
```

Common issues:
- Port 9092 already in use
- Insufficient memory (Kafka needs ~1GB)
- Docker daemon not running

### Cannot connect to Kafka

Verify Kafka is listening:
```bash
docker exec -it crpbot-kafka netstat -tlnp | grep 9092
```

Check firewall settings:
```bash
sudo ufw status
sudo ufw allow 9092/tcp
```

### Topics not auto-created

Enable auto-creation in Kafka:
```bash
# Already enabled in docker-compose.kafka.yml
KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
```

Or create manually (see step 4 above).

### Kafka UI shows "No brokers"

Wait 30 seconds for services to start. If still failing:
```bash
docker compose -f docker-compose.kafka.yml restart kafka-ui
```

## Production Deployment

For production, update `docker-compose.kafka.yml`:

1. **Increase replication factor**:
   ```yaml
   KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
   ```

2. **Enable authentication** (SASL/SCRAM):
   ```yaml
   KAFKA_SASL_ENABLED_MECHANISMS: SCRAM-SHA-256
   KAFKA_SASL_MECHANISM_INTER_BROKER_PROTOCOL: SCRAM-SHA-256
   ```

3. **Enable TLS encryption**:
   Mount certificates and configure:
   ```yaml
   KAFKA_SSL_KEYSTORE_LOCATION: /etc/kafka/secrets/kafka.keystore.jks
   KAFKA_SSL_KEYSTORE_PASSWORD: ${KEYSTORE_PASSWORD}
   ```

4. **Resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

5. **Persistent volumes**:
   Already configured in docker-compose.kafka.yml

## Next Steps

1. Create feature engineering stream processor (`apps/kafka/streams/feature_engineer.py`)
2. Create model inference consumers (`apps/kafka/consumers/model_inference.py`)
3. Create signal aggregator (`apps/kafka/consumers/signal_aggregator.py`)
4. Set up monitoring with Prometheus + Grafana
5. Configure alerting for critical errors

## References

- [Confluent Kafka Documentation](https://docs.confluent.io/platform/current/overview.html)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

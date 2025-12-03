# HYDRA Grafana Bug Handling Plan

## Bug Categories

### Category 1: Metrics Collection Bugs
**Symptoms**: Metrics show 0, NaN, or wrong values
**Examples**:
- `hydra_asset_pnl_percent` always 0
- `hydra_indicator_adx` returns NaN
- Engine stats not updating

### Category 2: Prometheus Scraping Bugs
**Symptoms**: Prometheus can't scrape, shows "down" status
**Examples**:
- Connection refused on :9100
- Malformed metric output
- Timeout during scrape

### Category 3: Grafana Display Bugs
**Symptoms**: Dashboard shows "No Data" or wrong charts
**Examples**:
- Query returns empty
- Labels don't match
- Time range issues

### Category 4: Runtime Crash Bugs
**Symptoms**: HYDRA container restarts, logs show exceptions
**Examples**:
- Exception in `_update_prometheus_metrics()`
- Missing attribute on None object
- Division by zero

---

## Prevention Strategy

### 1. Defensive Coding in Metrics Export
```python
def _update_prometheus_metrics(self):
    """Update all Prometheus metrics with defensive error handling."""
    try:
        # Wrap each section independently
        self._update_performance_metrics()
        self._update_asset_metrics()
        self._update_technical_metrics()
        self._update_guardian_metrics()
        self._update_engine_metrics()
    except Exception as e:
        logger.error(f"Metrics update failed: {e}")
        HydraMetrics.errors_total.labels(type='metrics_update').inc()
        # DON'T re-raise - metrics failure shouldn't crash runtime

def _update_asset_metrics(self):
    """Update per-asset metrics with null checks."""
    try:
        asset_stats = self.paper_trader.get_stats_by_asset()
        if not asset_stats:
            logger.warning("No asset stats available")
            return

        for asset, stats in asset_stats.items():
            # Null-safe access with defaults
            pnl = stats.get('pnl_percent', 0.0) or 0.0
            win_rate = stats.get('win_rate', 0.0) or 0.0

            HydraMetrics.asset_pnl.labels(asset=asset).set(pnl)
            HydraMetrics.asset_win_rate.labels(asset=asset).set(win_rate)

    except Exception as e:
        logger.error(f"Asset metrics update failed: {e}")
        HydraMetrics.errors_total.labels(type='asset_metrics').inc()
```

### 2. Metric Validation
```python
def _safe_set_gauge(gauge, value, labels=None, default=0.0):
    """Safely set a gauge with validation."""
    try:
        # Handle None, NaN, Inf
        if value is None:
            value = default
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = default

        if labels:
            gauge.labels(**labels).set(float(value))
        else:
            gauge.set(float(value))
    except Exception as e:
        logger.error(f"Failed to set metric: {e}")
```

### 3. Health Check Endpoint
```python
# Add to prometheus_exporter.py
def _serve_health(self):
    """Enhanced health check with metrics status."""
    health = {
        "status": "healthy",
        "service": "hydra-metrics",
        "metrics_count": len(list(generate_latest())),
        "last_update": self.last_update_time,
        "errors_last_hour": self.error_count
    }
    # Return 503 if too many errors
    status = 200 if self.error_count < 10 else 503
```

---

## Detection Mechanisms

### 1. Prometheus Alerts (Already Configured)
```yaml
# monitoring/prometheus/rules/hydra_alerts.yml
- alert: HydraMetricsStale
  expr: time() - hydra_last_update_timestamp > 600
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "HYDRA metrics not updating"

- alert: HydraHighErrorRate
  expr: rate(hydra_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
```

### 2. Grafana Alerts
- Set up "No Data" alerts on critical panels
- Alert when P&L suddenly goes to 0
- Alert when uptime resets (container restart)

### 3. Log Monitoring
```bash
# Quick check for metrics errors
docker logs hydra-runtime 2>&1 | grep -i "metrics\|error\|exception" | tail -20

# Watch real-time
docker logs -f hydra-runtime 2>&1 | grep --line-buffered "metrics"
```

### 4. Manual Verification Script
```bash
#!/bin/bash
# scripts/verify_metrics.sh

echo "=== HYDRA Metrics Health Check ==="

# 1. Check metrics endpoint
echo -n "Metrics endpoint: "
curl -s http://localhost:9100/metrics > /dev/null && echo "OK" || echo "FAIL"

# 2. Check key metrics exist
echo -n "P&L metric: "
curl -s http://localhost:9100/metrics | grep "hydra_pnl_total" && echo "OK" || echo "MISSING"

echo -n "Asset metrics: "
curl -s http://localhost:9100/metrics | grep "hydra_asset_pnl" && echo "OK" || echo "MISSING"

# 3. Check for NaN/Inf values
echo -n "Invalid values: "
curl -s http://localhost:9100/metrics | grep -E "NaN|Inf" && echo "FOUND!" || echo "None (OK)"

# 4. Check Prometheus scrape
echo -n "Prometheus scrape: "
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"up"' && echo "OK" || echo "FAIL"

# 5. Check error count
echo -n "Error count: "
curl -s http://localhost:9100/metrics | grep "hydra_errors_total"
```

---

## Debugging Workflow

### Step 1: Identify the Bug Location
```
Dashboard shows wrong data
    ↓
Check Grafana query → Is query correct?
    ↓ NO → Fix query
    ↓ YES
Check Prometheus → Does metric exist with right value?
    ↓ NO → Check metrics endpoint
    ↓ YES → Grafana config issue

curl http://localhost:9100/metrics | grep "metric_name"
curl http://localhost:9090/api/v1/query?query=metric_name
```

### Step 2: Debug Commands
```bash
# 1. Check raw metrics from HYDRA
curl -s http://178.156.136.185:9100/metrics | grep hydra_

# 2. Query Prometheus directly
curl -s 'http://178.156.136.185:9090/api/v1/query?query=hydra_pnl_total_percent'

# 3. Check container logs
ssh root@178.156.136.185 "docker logs hydra-runtime 2>&1 | tail -100"

# 4. Check if method returns data
ssh root@178.156.136.185 "docker exec hydra-runtime python -c '
from libs.hydra.paper_trader import PaperTrader
pt = PaperTrader()
print(pt.get_stats_by_asset())
'"

# 5. Interactive debug
ssh root@178.156.136.185 "docker exec -it hydra-runtime python"
```

### Step 3: Common Bug Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Metric = 0 always | Method returns None | Add null check, set default |
| Metric = NaN | Division by zero | Add zero check before division |
| "No Data" in Grafana | Label mismatch | Check label names match query |
| Prometheus "down" | Port not exposed | Check docker-compose ports |
| Container restart loop | Exception in metrics | Check logs, add try/except |

---

## Testing Procedure

### Before Deploying New Metrics

#### 1. Local Unit Test
```python
# tests/unit/test_metrics_export.py
def test_asset_metrics_handles_empty_stats():
    """Metrics should not crash on empty data."""
    runtime = HydraRuntime(paper_mode=True)
    runtime.paper_trader.trades = []  # Empty

    # Should not raise
    runtime._update_asset_metrics()

def test_asset_metrics_handles_none():
    """Metrics should handle None values."""
    runtime = HydraRuntime(paper_mode=True)
    runtime.paper_trader.get_stats_by_asset = lambda: {'BTC-USD': {'pnl_percent': None}}

    runtime._update_asset_metrics()
    # Should set to 0, not crash

def test_metrics_endpoint_valid():
    """All metrics should be valid Prometheus format."""
    from prometheus_client import generate_latest
    output = generate_latest()

    # No NaN or Inf
    assert b'NaN' not in output
    assert b'Inf' not in output
```

#### 2. Integration Test
```bash
# Test metrics locally before cloud deploy
cd /home/numan/crpbot
uv run python -c "
from libs.monitoring import HydraMetrics, MetricsExporter
from apps.runtime.hydra_runtime import HydraRuntime

# Start exporter
exporter = MetricsExporter(port=9100)
exporter.start()

# Create runtime and update metrics
runtime = HydraRuntime(paper_mode=True, assets=['BTC-USD'])
runtime._update_prometheus_metrics()

# Verify
import requests
r = requests.get('http://localhost:9100/metrics')
print('Metrics count:', len([l for l in r.text.split('\n') if l.startswith('hydra_')]))
print('Sample:', r.text[:500])
"
```

#### 3. Staging Deploy
```bash
# Deploy to cloud but don't update dashboards yet
git push origin feature/v7-ultimate
ssh root@178.156.136.185 "cd /root/crpbot && git pull && docker compose up -d --build"

# Verify metrics
ssh root@178.156.136.185 "curl -s http://localhost:9100/metrics | grep hydra_ | head -20"

# Check for errors
ssh root@178.156.136.185 "docker logs hydra-runtime 2>&1 | grep -i error | tail -10"
```

#### 4. Dashboard Test
- Add new panel in Grafana
- Wait 1-2 scrape intervals (30s)
- Verify data appears
- Check for "No Data" warnings

---

## Rollback Procedure

### If Metrics Break Runtime
```bash
# 1. Revert to last working commit
ssh root@178.156.136.185 "
cd /root/crpbot
git log --oneline -5  # Find last good commit
git checkout <COMMIT_HASH> -- apps/runtime/hydra_runtime.py libs/monitoring/
docker compose up -d --build
"

# 2. Or disable new metrics temporarily
# Edit _update_prometheus_metrics() to skip broken section
```

### If Dashboard Breaks
```bash
# Grafana dashboards are provisioned from files
# Revert dashboard JSON
git checkout HEAD~1 -- monitoring/grafana/dashboards/
# Re-provision
ssh root@178.156.136.185 "docker restart hydra-grafana"
```

---

## Bug Tracking Template

When a bug is found, document:

```markdown
## Bug: [Short Description]

**Date**: YYYY-MM-DD
**Severity**: Critical/High/Medium/Low
**Status**: Open/In Progress/Fixed

### Symptoms
- What's wrong (screenshot if applicable)

### Expected Behavior
- What should happen

### Steps to Reproduce
1. Step 1
2. Step 2

### Debug Output
```
[paste relevant logs/queries]
```

### Root Cause
- Why it happened

### Fix
- What was changed
- Commit hash

### Prevention
- How to prevent similar bugs
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  METRICS BUG QUICK DEBUG                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Check endpoint:    curl localhost:9100/metrics           │
│ 2. Check Prometheus:  curl localhost:9090/api/v1/targets    │
│ 3. Check logs:        docker logs hydra-runtime | tail -50  │
│ 4. Check specific:    curl localhost:9100/metrics | grep X  │
│ 5. Query Prometheus:  curl 'localhost:9090/api/v1/query?    │
│                       query=hydra_pnl_total_percent'        │
├─────────────────────────────────────────────────────────────┤
│ Common Fixes:                                                │
│ • Metric = 0     → Check source method returns data         │
│ • Metric = NaN   → Add null/zero checks                     │
│ • No Data        → Check label names in query               │
│ • Scrape fail    → Check port 9100 exposed                  │
│ • Runtime crash  → Add try/except around metrics code       │
└─────────────────────────────────────────────────────────────┘
```

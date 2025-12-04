"""
HYDRA 4.0 Monitoring Module

Provides Prometheus metrics exporter for real-time monitoring.

Usage:
    from libs.monitoring import MetricsExporter, HydraMetrics

    # Start the metrics server
    exporter = MetricsExporter(port=9100)
    exporter.start()

    # Update metrics during runtime
    HydraMetrics.set_pnl(5.23)
    HydraMetrics.set_win_rate(67.5)
    HydraMetrics.record_trade('BTC-USD', 'BUY', 'A')
"""

from .metrics import HydraMetrics
from .prometheus_exporter import MetricsExporter

__all__ = ['HydraMetrics', 'MetricsExporter']

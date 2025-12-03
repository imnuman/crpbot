"""
HYDRA 4.0 Prometheus Metrics Exporter

Provides an HTTP server that exposes Prometheus metrics at /metrics endpoint.
Runs in a background thread to not block the main HYDRA runtime.

Usage:
    from libs.monitoring import MetricsExporter, HydraMetrics

    # Initialize metrics
    HydraMetrics.initialize()

    # Start the exporter
    exporter = MetricsExporter(port=9100)
    exporter.start()

    # In your main loop, update metrics
    HydraMetrics.set_pnl(5.23, 1.2)
    HydraMetrics.set_win_rate(67.5, 62.3)

    # When shutting down
    exporter.stop()
"""

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
from typing import Optional
import time

from .metrics import HydraMetrics


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Prometheus metrics endpoint."""

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/health':
            self._serve_health()
        else:
            self.send_error(404, 'Not Found')

    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        try:
            # Update uptime before serving
            HydraMetrics.update_uptime()

            # Generate metrics
            output = generate_latest()

            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', len(output))
            self.end_headers()
            self.wfile.write(output)

        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_error(500, str(e))

    def _serve_health(self):
        """Serve health check endpoint."""
        try:
            # Simple health response
            response = b'{"status": "healthy", "service": "hydra-metrics"}'

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response))
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            logger.error(f"Error serving health: {e}")
            self.send_error(500, str(e))


class MetricsExporter:
    """
    Prometheus metrics exporter for HYDRA 4.0.

    Runs an HTTP server in a background thread that exposes metrics
    for Prometheus to scrape.

    Attributes:
        port: Port to listen on (default: 9100)
        host: Host to bind to (default: 0.0.0.0)
    """

    def __init__(self, port: int = 9100, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the metrics server in a background thread."""
        if self._running:
            logger.warning("Metrics exporter already running")
            return

        try:
            # Initialize metrics
            HydraMetrics.initialize()

            # Create server
            self._server = HTTPServer((self.host, self.port), MetricsHandler)
            self._running = True

            # Start in background thread
            self._thread = threading.Thread(
                target=self._serve_forever,
                daemon=True,
                name='MetricsExporter'
            )
            self._thread.start()

            logger.info(f"Prometheus metrics exporter started on {self.host}:{self.port}")
            logger.info(f"Metrics available at http://{self.host}:{self.port}/metrics")

        except Exception as e:
            logger.error(f"Failed to start metrics exporter: {e}")
            self._running = False
            raise

    def _serve_forever(self):
        """Server loop running in background thread."""
        while self._running:
            try:
                self._server.handle_request()
            except Exception as e:
                if self._running:  # Only log if we didn't stop intentionally
                    logger.error(f"Error handling metrics request: {e}")
                    time.sleep(0.1)

    def stop(self):
        """Stop the metrics server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("Prometheus metrics exporter stopped")

    def is_running(self) -> bool:
        """Check if the exporter is running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Singleton instance for easy access
_exporter: Optional[MetricsExporter] = None


def get_exporter(port: int = 9100, host: str = '0.0.0.0') -> MetricsExporter:
    """
    Get or create the singleton metrics exporter.

    Args:
        port: Port to listen on
        host: Host to bind to

    Returns:
        MetricsExporter instance
    """
    global _exporter
    if _exporter is None:
        _exporter = MetricsExporter(port=port, host=host)
    return _exporter


def start_exporter(port: int = 9100, host: str = '0.0.0.0'):
    """
    Convenience function to start the metrics exporter.

    Args:
        port: Port to listen on
        host: Host to bind to
    """
    exporter = get_exporter(port, host)
    if not exporter.is_running():
        exporter.start()


def stop_exporter():
    """Convenience function to stop the metrics exporter."""
    global _exporter
    if _exporter:
        _exporter.stop()
        _exporter = None

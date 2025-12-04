"""Reflex configuration for V7 dashboard"""

import reflex as rx

config = rx.Config(
    app_name="dashboard_reflex",
    # Port allocation: 3000 (frontend), 5000 (Flask), 8000 (Reflex backend)
    frontend_port=3000,
    backend_port=8000,
    # Enable telemetry for performance monitoring
    telemetry_enabled=False,
)

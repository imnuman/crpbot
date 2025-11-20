"""Reflex configuration for V7 dashboard"""

import reflex as rx

config = rx.Config(
    app_name="crpbot",
    # Use port 3000 to not conflict with Flask on 5000
    frontend_port=3000,
    backend_port=8000,
    # Enable telemetry for performance monitoring
    telemetry_enabled=False,
)

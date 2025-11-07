"""Health check endpoint for runtime monitoring."""
import asyncio
from datetime import datetime
from typing import Any

from aiohttp import web
from loguru import logger

from apps.runtime.rate_limiter import RateLimiter


async def healthz_handler(request: web.Request) -> web.Response:
    """
    Health check endpoint.

    Returns:
        JSON response with health status
    """
    # Get runtime state from request app
    runtime_state = request.app.get("runtime_state", {})

    # Basic health check
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 0,  # TODO: Track actual uptime
        "mode": runtime_state.get("mode", "unknown"),
        "kill_switch": runtime_state.get("kill_switch", False),
    }

    # Add rate limiter stats if available
    rate_limiter: RateLimiter | None = runtime_state.get("rate_limiter")
    if rate_limiter:
        health_status["rate_limiter"] = rate_limiter.get_stats()

    # Add FTMO state if available
    ftmo_state = runtime_state.get("ftmo_state")
    if ftmo_state:
        health_status["ftmo"] = {
            "balance": ftmo_state.account_balance,
            "daily_loss": ftmo_state.daily_loss,
            "total_loss": ftmo_state.total_loss,
            "daily_loss_pct": ftmo_state.daily_loss / ftmo_state.daily_start_balance
            if ftmo_state.daily_start_balance > 0
            else 0.0,
        }

    return web.json_response(health_status)


async def setup_healthz_server(port: int = 8080, runtime_state: dict[str, Any] | None = None) -> web.Application:
    """
    Setup health check HTTP server.

    Args:
        port: Port to listen on
        runtime_state: Runtime state dictionary (optional)

    Returns:
        aiohttp Application
    """
    app = web.Application()
    app["runtime_state"] = runtime_state or {}

    app.router.add_get("/healthz", healthz_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)

    logger.info(f"Health check server started on port {port}")

    return app, runner, site


async def run_healthz_server(port: int = 8080, runtime_state: dict[str, Any] | None = None) -> None:
    """Run health check server (blocking)."""
    app, runner, site = await setup_healthz_server(port, runtime_state)
    await site.start()
    logger.info(f"Health check server listening on http://0.0.0.0:{port}/healthz")

    # Keep running
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    # Run health check server standalone (for testing)
    asyncio.run(run_healthz_server())


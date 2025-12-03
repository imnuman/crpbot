# HYDRA 4.0 Multi-Agent Trading System
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy only essential files first (needed for build)
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY apps/ ./apps/
COPY libs/ ./libs/
COPY scripts/ ./scripts/

# Install dependencies (no editable install in Docker)
RUN uv sync --frozen --no-dev --no-editable

# Create data directories
RUN mkdir -p /app/data/hydra /app/logs

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["uv", "run", "python", "apps/runtime/hydra_runtime.py", "--paper", "--assets", "BTC-USD", "ETH-USD", "SOL-USD", "--iterations", "-1", "--interval", "300"]

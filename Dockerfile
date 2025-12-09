# HYDRA 4.0 Multi-Agent Trading System
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    sqlite3 \
    sshpass \
    openssh-client \
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

# Create data directories and symlink for hardcoded paths
RUN mkdir -p /app/data/hydra /app/logs && \
    mkdir -p /root && \
    ln -s /app /root/crpbot

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV HYDRA_DATA_DIR=/app/data/hydra

# Default command (can be overridden)
CMD ["uv", "run", "python", "apps/runtime/hydra_runtime.py", "--paper", "--assets", "BTC-USD", "ETH-USD", "SOL-USD", "--iterations", "-1", "--interval", "300"]

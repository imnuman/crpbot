# HYDRA 4.0 Multi-Agent Trading System
# Optimized for fast rebuilds - code changes don't reinstall deps

FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies (rarely changes - cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    sqlite3 \
    sshpass \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# ============================================
# DEPENDENCY LAYER - Changes rarely, cache it!
# ============================================
# Copy ONLY dependency files first
COPY pyproject.toml uv.lock ./

# Create a minimal README for the build (required by pyproject.toml)
RUN echo "# HYDRA Trading System" > README.md

# Install dependencies WITHOUT source code
# This layer is cached until pyproject.toml or uv.lock changes
RUN uv sync --frozen --no-dev --no-editable || \
    (echo "Dependency install failed, creating stub package" && \
     mkdir -p apps libs && \
     touch apps/__init__.py libs/__init__.py && \
     uv sync --frozen --no-dev --no-editable)

# ============================================
# SOURCE CODE LAYER - Changes often, fast copy
# ============================================
# Now copy source code (deps already cached above)
COPY apps/ ./apps/
COPY libs/ ./libs/
COPY scripts/ ./scripts/
COPY README.md ./

# Re-run sync to install the local package (fast - deps cached)
RUN uv sync --frozen --no-dev --no-editable

# Create data directories
RUN mkdir -p /app/data/hydra /app/logs && \
    mkdir -p /root && \
    ln -s /app /root/crpbot

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV HYDRA_DATA_DIR=/app/data/hydra

# Default command
CMD ["uv", "run", "python", "apps/runtime/hydra_runtime.py", "--paper", "--assets", "BTC-USD", "ETH-USD", "SOL-USD", "--iterations", "-1", "--interval", "300"]

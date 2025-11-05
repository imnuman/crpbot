FROM python:3.10-slim

WORKDIR /opt/trading-ai

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code
COPY apps ./apps
COPY libs ./libs

# Copy .env.example (user should mount their own .env)
COPY .env.example ./.env.example

# Create non-root user
RUN useradd -r -s /usr/sbin/nologin trading && \
    chown -R trading:trading /opt/trading-ai

USER trading

CMD ["python", "apps/runtime/main.py"]


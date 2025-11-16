#!/bin/bash
# Wrapper script to run runtime with .env loaded correctly

# Unset any existing Coinbase variables to force reload from .env
unset COINBASE_API_KEY_NAME
unset COINBASE_API_PRIVATE_KEY

# Load .env file
set -a
source .env 2>/dev/null || true
set +a

# Run the runtime with arguments passed to script
exec uv run python apps/runtime/main.py "$@"

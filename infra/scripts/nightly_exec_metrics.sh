#!/bin/bash
# Nightly job to recompute execution metrics from FTMO bridge
# Runs via cron: 0 2 * * * /path/to/infra/scripts/nightly_exec_metrics.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the nightly metrics script
python scripts/nightly_exec_metrics.py

# Log completion
echo "$(date): Nightly execution metrics job completed" >> /var/log/trading-ai-exec-metrics.log


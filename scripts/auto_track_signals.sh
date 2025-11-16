#!/bin/bash
#Automated signal outcome tracking - runs every 5 minutes via cron
# Add to crontab with: */5 * * * * /path/to/auto_track_signals.sh

cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment
source .venv/bin/activate

# Run signal tracking
echo "[$(date)] Running signal outcome tracking..."
uv run python scripts/track_signal_outcomes.py --evaluation-period 15 >> logs/signal_tracking.log 2>&1

echo "[$(date)] Signal tracking complete"

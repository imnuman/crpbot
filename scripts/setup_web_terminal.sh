#!/bin/bash
# HYDRA 4.0 - Web Terminal Dashboard Setup
# Installs ttyd and configures the terminal dashboard for web access
#
# Usage:
#   ./scripts/setup_web_terminal.sh          # Install and configure
#   ./scripts/setup_web_terminal.sh --start  # Start dashboard
#   ./scripts/setup_web_terminal.sh --stop   # Stop dashboard

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Handle commands
case "${1:-}" in
    --start)
        echo "Starting HYDRA 4.0 Dashboard..."
        cd "$PROJECT_DIR"
        # Kill existing
        pkill -f "ttyd.*hydra_dashboard" 2>/dev/null || true
        # Start new (background)
        nohup ttyd -p 7681 -t fontSize=14 "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/hydra_dashboard.py" > /tmp/hydra_dashboard.log 2>&1 &
        sleep 2
        IP=$(hostname -I | awk '{print $1}')
        echo "Dashboard running at: http://${IP}:7681"
        exit 0
        ;;
    --stop)
        echo "Stopping HYDRA 4.0 Dashboard..."
        pkill -f "ttyd.*hydra_dashboard" 2>/dev/null || true
        pkill -f "hydra_dashboard.py" 2>/dev/null || true
        echo "Dashboard stopped."
        exit 0
        ;;
    --status)
        if pgrep -f "ttyd.*hydra_dashboard" > /dev/null; then
            echo "Dashboard is RUNNING"
            pgrep -af "ttyd.*hydra_dashboard"
        else
            echo "Dashboard is NOT running"
        fi
        exit 0
        ;;
esac

echo "=== HYDRA 4.0 Web Terminal Setup ==="

# Install ttyd
if ! command -v ttyd &> /dev/null; then
    echo "Installing ttyd..."

    # Try apt first (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y ttyd
    # Try snap as fallback
    elif command -v snap &> /dev/null; then
        sudo snap install ttyd --classic
    else
        # Build from source
        echo "Building ttyd from source..."
        sudo apt-get install -y build-essential cmake git libjson-c-dev libwebsockets-dev
        git clone https://github.com/tsl0922/ttyd.git /tmp/ttyd
        cd /tmp/ttyd && mkdir build && cd build
        cmake ..
        make && sudo make install
        cd - > /dev/null
    fi

    echo "ttyd installed successfully"
else
    echo "ttyd already installed: $(which ttyd)"
fi

echo "Project directory: $PROJECT_DIR"

# Create systemd service for dashboard
echo "Creating systemd service..."

sudo tee /etc/systemd/system/hydra-dashboard.service > /dev/null << EOF
[Unit]
Description=HYDRA 4.0 Terminal Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/ttyd -p 7681 -t fontSize=14 -t fontFamily="monospace" $PROJECT_DIR/.venv/bin/python $PROJECT_DIR/scripts/hydra_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick Commands:"
echo "  $0 --start    Start dashboard (background)"
echo "  $0 --stop     Stop dashboard"
echo "  $0 --status   Check if running"
echo ""
echo "Systemd Commands:"
echo "  sudo systemctl start hydra-dashboard"
echo "  sudo systemctl stop hydra-dashboard"
echo "  sudo systemctl status hydra-dashboard"
echo "  sudo systemctl enable hydra-dashboard  # Auto-start on boot"
echo ""
echo "Access dashboard at: http://$(hostname -I | awk '{print $1}'):7681"
echo ""

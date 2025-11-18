#!/bin/bash
# Setup NTP time synchronization on local system
# Run this script with: sudo bash scripts/setup_ntp_local.sh

set -e

echo "====================================="
echo "Setting up NTP Time Synchronization"
echo "====================================="

# Install chrony
echo "Installing chrony..."
apt-get update
apt-get install -y chrony

# Configure chrony with reliable NTP servers
echo "Configuring chrony..."
cat > /etc/chrony/chrony.conf << 'EOF'
# Use public NTP servers from the pool.ntp.org project
pool time.google.com iburst
pool time.cloudflare.com iburst
pool 0.north-america.pool.ntp.org iburst
pool 1.north-america.pool.ntp.org iburst

# Record the rate at which the system clock gains/loses time
driftfile /var/lib/chrony/drift

# Allow the system clock to be stepped in the first three updates
makestep 1.0 3

# Enable kernel synchronization of the real-time clock (RTC)
rtcsync

# Log tracking, measurements, and selections
logdir /var/log/chrony
log tracking measurements statistics
EOF

# Set timezone to EST (America/Toronto)
echo "Setting timezone to America/Toronto (EST)..."
timedatectl set-timezone America/Toronto

# Restart and enable chrony
echo "Restarting chrony service..."
systemctl restart chrony
systemctl enable chrony

# Wait for synchronization
echo "Waiting for time synchronization..."
sleep 5

# Show status
echo ""
echo "====================================="
echo "NTP Synchronization Status"
echo "====================================="
chronyc tracking
echo ""
echo "NTP Sources:"
chronyc sources
echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="

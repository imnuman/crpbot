#!/bin/bash
set -e

# Interactive setup for dual environment
# Configures SSH, saves server info, and tests connection

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Dual Environment Setup Wizard                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Cloud Server Info
echo -e "${BLUE}Step 1: Cloud Server Configuration${NC}"
echo ""
read -p "Enter cloud server IP or hostname: " SERVER_IP
read -p "Enter SSH username (e.g., ubuntu): " SSH_USER
REMOTE_SERVER="$SSH_USER@$SERVER_IP"

echo ""
echo -e "${BLUE}Step 2: SSH Key Configuration${NC}"
echo ""
echo "Do you have an SSH key for this server?"
echo "  1) Yes, I have a key (default: ~/.ssh/id_ed25519 or ~/.ssh/id_rsa)"
echo "  2) Yes, key is at a custom location"
echo "  3) No, I need to generate one"
read -p "Select option (1/2/3): " -n 1 -r
echo ""

SSH_KEY=""
case $REPLY in
    1)
        # Check for common key locations
        if [ -f ~/.ssh/id_ed25519 ]; then
            SSH_KEY=~/.ssh/id_ed25519
            echo "Found: $SSH_KEY"
        elif [ -f ~/.ssh/id_rsa ]; then
            SSH_KEY=~/.ssh/id_rsa
            echo "Found: $SSH_KEY"
        else
            echo -e "${YELLOW}⚠️  No default key found${NC}"
            read -p "Enter SSH key path: " SSH_KEY
        fi
        ;;
    2)
        read -p "Enter SSH key path: " SSH_KEY
        if [ ! -f "$SSH_KEY" ]; then
            echo -e "${YELLOW}⚠️  Key not found at $SSH_KEY${NC}"
            exit 1
        fi
        ;;
    3)
        echo "Generating new SSH key..."
        ssh-keygen -t ed25519 -C "crpbot-$(date +%Y%m%d)" -f ~/.ssh/crpbot_key
        SSH_KEY=~/.ssh/crpbot_key
        echo -e "${GREEN}✅ Key generated: $SSH_KEY${NC}"
        echo ""
        echo "⚠️  IMPORTANT: Copy public key to your cloud server:"
        echo ""
        echo "Run this command to copy:"
        echo "  ssh-copy-id -i $SSH_KEY.pub $REMOTE_SERVER"
        echo ""
        echo "Or manually add this public key to server's ~/.ssh/authorized_keys:"
        cat $SSH_KEY.pub
        echo ""
        read -p "Press Enter when done..."
        ;;
esac

# Ensure key has correct permissions
if [ -n "$SSH_KEY" ]; then
    chmod 600 "$SSH_KEY"
    chmod 644 "$SSH_KEY.pub" 2>/dev/null || true
    echo -e "${GREEN}✅ SSH key permissions set${NC}"
fi

# Step 3: Test Connection
echo ""
echo -e "${BLUE}Step 3: Testing SSH Connection${NC}"
echo ""

if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$REMOTE_SERVER" "echo 'Connection successful'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH connection successful!${NC}"
else
    echo -e "${YELLOW}⚠️  SSH connection failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Verify server IP: $SERVER_IP"
    echo "  2. Verify username: $SSH_USER"
    echo "  3. Copy public key to server:"
    echo "     ssh-copy-id -i $SSH_KEY.pub $REMOTE_SERVER"
    echo "  4. Check server firewall allows SSH (port 22)"
    echo ""
    read -p "Try again? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 4: Create SSH Config
echo ""
echo -e "${BLUE}Step 4: SSH Config Setup${NC}"
echo ""
read -p "Create SSH config entry for easy access? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter SSH config alias (e.g., crpbot-cloud): " SSH_ALIAS

    mkdir -p ~/.ssh
    touch ~/.ssh/config

    # Check if alias already exists
    if grep -q "Host $SSH_ALIAS" ~/.ssh/config; then
        echo -e "${YELLOW}⚠️  Alias '$SSH_ALIAS' already exists in ~/.ssh/config${NC}"
        echo "Please edit manually or choose different alias"
    else
        cat >> ~/.ssh/config <<EOF

# CRPBot Cloud Server
Host $SSH_ALIAS
    HostName $SERVER_IP
    User $SSH_USER
    IdentityFile $SSH_KEY
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
        echo -e "${GREEN}✅ SSH config created${NC}"
        echo ""
        echo "You can now connect with: ssh $SSH_ALIAS"
        REMOTE_SERVER="$SSH_ALIAS"
    fi
fi

# Step 5: Save Configuration
echo ""
echo -e "${BLUE}Step 5: Saving Configuration${NC}"
echo ""

# Save remote server
echo "$REMOTE_SERVER" > "$PROJECT_ROOT/.cloud_server"
echo -e "${GREEN}✅ Saved server: $REMOTE_SERVER${NC}"

# Save SSH key path
if [ -n "$SSH_KEY" ]; then
    echo "$SSH_KEY" > "$PROJECT_ROOT/.ssh_key"
    echo -e "${GREEN}✅ Saved SSH key path${NC}"
fi

# Add to .gitignore
if ! grep -q ".cloud_server" "$PROJECT_ROOT/.gitignore"; then
    echo ".cloud_server" >> "$PROJECT_ROOT/.gitignore"
    echo ".ssh_key" >> "$PROJECT_ROOT/.gitignore"
    echo -e "${GREEN}✅ Added config files to .gitignore${NC}"
fi

# Step 6: Summary & Next Steps
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup Complete! ✅                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration Summary:${NC}"
echo "  Remote Server: $REMOTE_SERVER"
echo "  SSH Key: $SSH_KEY"
echo "  Config saved: $PROJECT_ROOT/.cloud_server"
echo ""
echo -e "${CYAN}Quick Commands:${NC}"
echo "  Connect:       ssh $REMOTE_SERVER"
echo "  Deploy:        ./scripts/deploy_to_cloud.sh"
echo "  Sync to cloud: ./scripts/sync_to_cloud.sh"
echo "  Sync from cloud: ./scripts/sync_from_cloud.sh"
echo "  Check status:  ./scripts/check_both_environments.sh"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Review DUAL_ENVIRONMENT_SETUP.md"
echo "  2. Review PRE_MIGRATION_CHECKLIST.md"
echo "  3. Run: ./scripts/deploy_to_cloud.sh"
echo ""

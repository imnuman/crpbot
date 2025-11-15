#!/bin/bash
set -e

# Sync models between local and cloud (or via S3)
# Usage: ./scripts/sync_models.sh [local-to-cloud|cloud-to-local|to-s3|from-s3]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

DIRECTION="${1:-}"
SSH_KEY=""
REMOTE_SERVER=""

if [ -f "$PROJECT_ROOT/.cloud_server" ]; then
    REMOTE_SERVER=$(cat "$PROJECT_ROOT/.cloud_server")
fi

if [ -z "$DIRECTION" ]; then
    echo "Model Sync Script"
    echo ""
    echo "Usage: $0 <direction>"
    echo ""
    echo "Directions:"
    echo "  local-to-cloud  - Upload models from local to cloud server"
    echo "  cloud-to-local  - Download models from cloud to local"
    echo "  to-s3           - Upload models to S3"
    echo "  from-s3         - Download models from S3"
    echo ""
    echo "Examples:"
    echo "  $0 to-s3          # Upload to S3 (recommended)"
    echo "  $0 from-s3        # Download from S3"
    echo "  $0 local-to-cloud # Direct transfer (slow)"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Model Sync                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

case "$DIRECTION" in
    to-s3)
        echo -e "${BLUE}📤 Uploading models to S3...${NC}"
        cd "$PROJECT_ROOT"
        aws s3 sync models/ s3://crpbot-market-data-dev/models/ \
            --exclude "*.gitkeep" \
            --exclude "__pycache__/*" \
            --exclude ".DS_Store"
        echo -e "${GREEN}✅ Models uploaded to S3${NC}"
        echo "Location: s3://crpbot-market-data-dev/models/"
        ;;

    from-s3)
        echo -e "${BLUE}📥 Downloading models from S3...${NC}"
        cd "$PROJECT_ROOT"
        mkdir -p models
        aws s3 sync s3://crpbot-market-data-dev/models/ models/ \
            --exclude "*.gitkeep"
        echo -e "${GREEN}✅ Models downloaded from S3${NC}"
        echo "Location: $PROJECT_ROOT/models/"
        ;;

    local-to-cloud)
        if [ -z "$REMOTE_SERVER" ]; then
            echo -e "${RED}❌ No remote server configured${NC}"
            echo "Save server: echo 'user@server' > .cloud_server"
            exit 1
        fi

        echo -e "${BLUE}📤 Uploading models to cloud server...${NC}"
        echo "This may take several minutes for large models..."

        if [ -n "$SSH_KEY" ]; then
            rsync -avz --progress -e "ssh -i $SSH_KEY" \
                "$PROJECT_ROOT/models/" "$REMOTE_SERVER:~/crpbot/models/"
        else
            rsync -avz --progress \
                "$PROJECT_ROOT/models/" "$REMOTE_SERVER:~/crpbot/models/"
        fi

        echo -e "${GREEN}✅ Models uploaded to cloud${NC}"
        ;;

    cloud-to-local)
        if [ -z "$REMOTE_SERVER" ]; then
            echo -e "${RED}❌ No remote server configured${NC}"
            echo "Save server: echo 'user@server' > .cloud_server"
            exit 1
        fi

        echo -e "${BLUE}📥 Downloading models from cloud server...${NC}"
        echo "This may take several minutes for large models..."

        mkdir -p "$PROJECT_ROOT/models"

        if [ -n "$SSH_KEY" ]; then
            rsync -avz --progress -e "ssh -i $SSH_KEY" \
                "$REMOTE_SERVER:~/crpbot/models/" "$PROJECT_ROOT/models/"
        else
            rsync -avz --progress \
                "$REMOTE_SERVER:~/crpbot/models/" "$PROJECT_ROOT/models/"
        fi

        echo -e "${GREEN}✅ Models downloaded from cloud${NC}"
        ;;

    *)
        echo -e "${RED}❌ Unknown direction: $DIRECTION${NC}"
        echo "Use: local-to-cloud, cloud-to-local, to-s3, or from-s3"
        exit 1
        ;;
esac

echo ""
echo "Model sizes:"
du -sh "$PROJECT_ROOT/models" 2>/dev/null || echo "No models directory"
echo ""

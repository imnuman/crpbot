#!/bin/bash

# Generate perfect introduction prompt for Claude Code on cloud server
# Usage: ./scripts/claude_intro.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Claude Code Introduction Generator               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check environment
cd "$PROJECT_ROOT"

echo -e "${BLUE}Analyzing environment...${NC}"
echo ""

# Gather environment info
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 || echo "Not found")
GIT_COMMIT=$(git log --oneline -1 2>/dev/null || echo "No git repo")
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "No git repo")
ENV_FILE="❌ Missing"
[ -f .env ] && ENV_FILE="✅ Present"
DB_PASS="❌ Missing"
[ -f .db_password ] && DB_PASS="✅ Present"
AWS_CREDS="❌ Missing"
[ -f ~/.aws/credentials ] && AWS_CREDS="✅ Present"
DATA_SIZE=$(du -sh data/ 2>/dev/null | cut -f1 || echo "0")
MODEL_COUNT=$(ls models/*.pt 2>/dev/null | wc -l || echo "0")
VENV_STATUS="❌ Not activated"
[[ "$VIRTUAL_ENV" != "" ]] && VENV_STATUS="✅ Activated"

# Check tests
TEST_STATUS="⏳ Not run yet"
if pytest tests/unit/ -q > /dev/null 2>&1; then
    TEST_STATUS="✅ Passing"
else
    TEST_STATUS="⚠️ Some failures"
fi

# Generate context summary
echo -e "${GREEN}Environment Status:${NC}"
echo "  Python: $PYTHON_VERSION"
echo "  Git Branch: $GIT_BRANCH"
echo "  Git Commit: $GIT_COMMIT"
echo "  Virtual Env: $VENV_STATUS"
echo "  .env File: $ENV_FILE"
echo "  .db_password: $DB_PASS"
echo "  AWS Credentials: $AWS_CREDS"
echo "  Data Directory: $DATA_SIZE"
echo "  Models: $MODEL_COUNT files"
echo "  Tests: $TEST_STATUS"
echo ""

# Check if CLAUDE.md exists
if [ ! -f "CLAUDE.md" ]; then
    echo -e "${YELLOW}⚠️  Warning: CLAUDE.md not found${NC}"
    echo "Claude won't have project context!"
    echo ""
fi

# Generate prompt
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Perfect Introduction Prompt for Claude:${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

cat <<EOF
Hi Claude! Welcome to the CRPBot cloud server environment.

📋 IMPORTANT: Please read CLAUDE.md first - it contains complete project instructions and architecture.

🔍 Current Environment Status:
- Location: $(pwd)
- Python: $PYTHON_VERSION
- Git: $GIT_BRANCH @ $GIT_COMMIT
- Virtual Environment: $VENV_STATUS
- Credentials: .env $ENV_FILE, .db_password $DB_PASS
- AWS Config: $AWS_CREDS
- Data: $DATA_SIZE in data/ directory
- Models: $MODEL_COUNT model files
- Tests: $TEST_STATUS

📚 Key Documentation to Read:
1. CLAUDE.md - Complete project guide (READ THIS FIRST!)
2. CLOUD_SERVER_SETUP.md - Cloud environment setup
3. PHASE1_COMPLETE_NEXT_STEPS.md - Current status and next steps
4. MASTER_SUMMARY.md - Project overview

🎯 Project Context (from CLAUDE.md):
- Name: CRPBot (Cryptocurrency Trading AI)
- Purpose: FTMO challenge compliance with ML ensemble
- Architecture: LSTM + Transformer + RL models
- Current Phase: Phase 1 Complete (AWS infrastructure deployed)
- Status: Ready for model validation and paper trading

🛠️ Your Capabilities Here:
- Read all project files and code
- Understand architecture from CLAUDE.md
- Help with debugging, development, and deployment
- Suggest improvements and best practices
- Reference AWS RDS, S3, Redis infrastructure

⚠️ Important Notes:
- This is the CLOUD SERVER (not local development)
- All credentials are present and secured
- Connected to production AWS infrastructure (RDS, S3, Redis)
- Changes here should be committed and synced with local environment

🚀 Your First Tasks:
1. Read CLAUDE.md thoroughly to understand the project
2. Verify the environment status above looks correct
3. Check if there are any issues that need immediate attention
4. Summarize the project status and suggest next steps

Ready to help! What would you like to work on?
EOF

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

# Save to file
INTRO_FILE="/tmp/claude_intro_$(date +%Y%m%d_%H%M%S).txt"
cat > "$INTRO_FILE" <<EOF
Hi Claude! Welcome to the CRPBot cloud server environment.

📋 IMPORTANT: Please read CLAUDE.md first - it contains complete project instructions and architecture.

🔍 Current Environment Status:
- Location: $(pwd)
- Python: $PYTHON_VERSION
- Git: $GIT_BRANCH @ $GIT_COMMIT
- Virtual Environment: $VENV_STATUS
- Credentials: .env $ENV_FILE, .db_password $DB_PASS
- AWS Config: $AWS_CREDS
- Data: $DATA_SIZE in data/ directory
- Models: $MODEL_COUNT model files
- Tests: $TEST_STATUS

📚 Key Documentation to Read:
1. CLAUDE.md - Complete project guide (READ THIS FIRST!)
2. CLOUD_SERVER_SETUP.md - Cloud environment setup
3. PHASE1_COMPLETE_NEXT_STEPS.md - Current status and next steps
4. MASTER_SUMMARY.md - Project overview

🎯 Project Context (from CLAUDE.md):
- Name: CRPBot (Cryptocurrency Trading AI)
- Purpose: FTMO challenge compliance with ML ensemble
- Architecture: LSTM + Transformer + RL models
- Current Phase: Phase 1 Complete (AWS infrastructure deployed)
- Status: Ready for model validation and paper trading

🛠️ Your Capabilities Here:
- Read all project files and code
- Understand architecture from CLAUDE.md
- Help with debugging, development, and deployment
- Suggest improvements and best practices
- Reference AWS RDS, S3, Redis infrastructure

⚠️ Important Notes:
- This is the CLOUD SERVER (not local development)
- All credentials are present and secured
- Connected to production AWS infrastructure (RDS, S3, Redis)
- Changes here should be committed and synced with local environment

🚀 Your First Tasks:
1. Read CLAUDE.md thoroughly to understand the project
2. Verify the environment status above looks correct
3. Check if there are any issues that need immediate attention
4. Summarize the project status and suggest next steps

Ready to help! What would you like to work on?
EOF

echo -e "${GREEN}✅ Introduction saved to: $INTRO_FILE${NC}"
echo ""
echo -e "${YELLOW}To use:${NC}"
echo "  1. Start Claude Code: ${CYAN}claude-code .${NC}"
echo "  2. Copy and paste the prompt above"
echo "  3. Or cat the file: ${CYAN}cat $INTRO_FILE${NC}"
echo ""
echo -e "${BLUE}Quick copy command:${NC}"
echo "  ${CYAN}cat $INTRO_FILE | pbcopy${NC}  # On Mac"
echo "  ${CYAN}cat $INTRO_FILE | xclip -selection clipboard${NC}  # On Linux with xclip"
echo "  ${CYAN}cat $INTRO_FILE${NC}  # Then manually copy"
echo ""

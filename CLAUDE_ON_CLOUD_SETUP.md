# Setting Up Claude Code on Cloud Server

Complete guide to connect Claude Code to your cloud server with full project context.

---

## 🎯 Overview

Claude Code will have access to:
- ✅ All code in `/home/ubuntu/crpbot` (or your path)
- ✅ **CLAUDE.md** - Project instructions and architecture
- ✅ All documentation files
- ✅ Git history and structure
- ✅ Environment configuration

---

## 🚀 Method 1: Claude Code CLI (Recommended)

### Step 1: Install Claude Code CLI on Cloud Server

```bash
# Connect to your cloud server
ssh user@your-server-ip

# Check if Claude Code CLI is available
# Visit: https://docs.anthropic.com/claude/docs/claude-code

# Example installation (check official docs for latest method):

# If available via npm:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic/claude-code-cli

# Or if available as standalone binary:
# Follow official installation instructions

# Verify installation
claude-code --version
```

### Step 2: Authenticate Claude Code

```bash
# Initialize Claude Code with your API key
claude-code auth login

# Or set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Save to .bashrc for persistence
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### Step 3: Navigate to Project and Start

```bash
# Go to project directory
cd ~/crpbot

# Start Claude Code in this directory
claude-code .

# Claude Code will automatically read CLAUDE.md and understand your project!
```

---

## 🖥️ Method 2: VS Code Remote-SSH (Alternative)

If Claude Code integrates with VS Code:

### Step 1: Setup on Local Machine

```bash
# On your local machine, install VS Code
# Install extensions:
# 1. Remote - SSH
# 2. Claude Code extension (if available)
```

### Step 2: Connect to Cloud Server

```bash
# In VS Code:
# 1. Press F1 or Ctrl+Shift+P
# 2. Type "Remote-SSH: Connect to Host"
# 3. Enter: user@your-server-ip
# 4. Or select your saved SSH config (crpbot-cloud)
```

### Step 3: Open Project

```bash
# In VS Code connected to remote:
# File → Open Folder → /home/ubuntu/crpbot
# Claude Code extension will work on the remote server
```

---

## 📋 Method 3: Web-Based IDE (Alternative)

### Setup code-server (VS Code in Browser)

```bash
# On cloud server
curl -fsSL https://code-server.dev/install.sh | sh

# Configure
mkdir -p ~/.config/code-server
cat > ~/.config/code-server/config.yaml <<EOF
bind-addr: 0.0.0.0:8080
auth: password
password: your-secure-password-here
cert: false
EOF

# Start code-server
sudo systemctl enable --now code-server@$USER

# Access in browser: http://your-server-ip:8080
# Then install Claude Code extension if available
```

---

## 🎯 Giving Claude Context on First Use

When you first start Claude Code on the cloud server, use this introduction:

### Option A: Quick Start Command

```bash
cd ~/crpbot
claude-code "I'm working on the CRPBot project. Please read CLAUDE.md to understand the project architecture and current status. Then help me set up and verify everything is working."
```

### Option B: Interactive Session

```bash
cd ~/crpbot
claude-code .

# Then type this as your first prompt:
```

**First Prompt to Claude**:
```
Hi Claude! I've just connected you to the CRPBot cloud server. Here's what you need to know:

1. Please read CLAUDE.md - this contains all project instructions, architecture, and current status
2. Read CLOUD_SERVER_SETUP.md to understand the cloud environment
3. This is a dual-environment setup - code can be edited on both local and cloud
4. All credentials are synced (.env, .db_password, AWS credentials)

Current task: Help me verify the cloud server setup is complete and everything works correctly.

Start by:
1. Reading CLAUDE.md
2. Checking if all dependencies are installed
3. Verifying database and AWS connections work
4. Running tests to ensure everything is operational
```

---

## 📝 Creating a Welcome Script for Claude

I'll create a script that generates the perfect introduction for Claude:

```bash
# This script is at: scripts/claude_intro.sh
./scripts/claude_intro.sh
```

This will:
1. ✅ Check environment status
2. ✅ Generate context summary
3. ✅ Create perfect prompt for Claude
4. ✅ Show what Claude should know

---

## 🔍 What Claude Can Access

Claude Code running on your cloud server can read:

### Configuration & Documentation
- ✅ `CLAUDE.md` - **Most important** - project guide
- ✅ `README.md` - Quick start
- ✅ `CLOUD_SERVER_SETUP.md` - Server setup guide
- ✅ `CLOUD_SERVER_QUICKSTART.md` - Daily commands
- ✅ `MIGRATION_GUIDE.md` - Migration details
- ✅ All other .md files

### Code & Structure
- ✅ All Python code in `apps/`, `libs/`, `scripts/`
- ✅ Test files in `tests/`
- ✅ Configuration files (`pyproject.toml`, `Makefile`)
- ✅ Git history and branches

### Environment Files (that Claude can reference)
- ✅ `.env` (credentials - Claude won't expose these)
- ✅ `.db_password`
- ✅ Model files in `models/`
- ✅ Data files in `data/`

### What Claude Cannot Do
- ❌ Cannot make AWS API calls directly (you run the commands)
- ❌ Cannot access files outside project directory
- ❌ Cannot directly execute commands (it suggests, you run)

---

## 💡 Best Practices for Working with Claude on Cloud

### 1. Always Start with Context

```bash
# Good first prompt:
"Read CLAUDE.md and help me [specific task]"

# Or:
"I'm working on the cloud server. Check CLOUD_SERVER_SETUP.md and verify everything is set up correctly."
```

### 2. Reference Documentation

```bash
# When asking for help:
"Based on the architecture in CLAUDE.md, help me debug the runtime"
"Following the process in MIGRATION_GUIDE.md, help me sync models"
```

### 3. Specify Environment

```bash
# Be clear about which environment:
"We're on the cloud server. Help me train models for BTC."
"This is the production environment. Help me deploy carefully."
```

### 4. Use Claude for Planning

```bash
"Read PHASE1_COMPLETE_NEXT_STEPS.md and create a plan for today's work"
"Review the current status in CLAUDE.md and suggest next steps"
```

---

## 🎬 Example First Session

```bash
# Connect to cloud server
ssh crpbot-cloud

# Activate environment
cd ~/crpbot
source .venv/bin/activate

# Start Claude Code
claude-code .

# First prompt:
"""
Hi Claude! Welcome to the CRPBot cloud server.

Context:
- Read CLAUDE.md for complete project overview
- This is a cryptocurrency trading AI for FTMO challenges
- Uses LSTM + Transformer ensemble models
- Phase 1 complete: AWS infrastructure deployed
- Currently at: Model validation phase

Your mission today:
1. Verify cloud server setup is complete
2. Check all credentials and connections work
3. Help me validate the GPU-trained models
4. Prepare for paper trading phase

Start by reading CLAUDE.md and summarizing the current project status.
"""

# Claude will then read CLAUDE.md and respond with understanding!
```

---

## 🔄 Daily Workflow with Claude on Cloud

### Morning Check-in
```bash
ssh crpbot-cloud
cd ~/crpbot
claude-code "Review the system status and suggest today's priorities based on PHASE1_COMPLETE_NEXT_STEPS.md"
```

### During Development
```bash
claude-code "Help me implement [feature]. Reference the architecture in CLAUDE.md."
```

### Before Committing
```bash
claude-code "Review my changes and check if they follow project standards in CLAUDE.md"
```

### Troubleshooting
```bash
claude-code "The tests are failing. Check the logs and help me debug based on project structure in CLAUDE.md"
```

---

## 🔐 Security Notes

### What's Safe to Share with Claude
- ✅ Code structure and architecture
- ✅ Error messages and logs
- ✅ Documentation
- ✅ Test results

### What to Be Careful With
- ⚠️ Database passwords (Claude can see .env but won't expose)
- ⚠️ AWS credentials (same as above)
- ⚠️ API keys (Claude respects security)

Claude Code is designed to:
- **Read** credentials to help you debug connections
- **Not expose** credentials in responses
- **Suggest** how to use them securely

---

## 📊 Verifying Claude Has Context

After connecting Claude, ask:

```bash
"What do you know about this project? Summarize based on CLAUDE.md"
```

Claude should respond with:
- ✅ Project name (CRPBot)
- ✅ Purpose (FTMO crypto trading)
- ✅ Architecture (LSTM + Transformer + RL ensemble)
- ✅ Current phase (Phase 1 complete)
- ✅ Next steps (model validation)

If Claude doesn't have this context:
```bash
"Please read CLAUDE.md now and tell me what you learned"
```

---

## 🛠️ Troubleshooting

### Claude Can't Find Files
```bash
# Verify you're in the project directory
pwd  # Should show: /home/ubuntu/crpbot or similar

# Verify CLAUDE.md exists
ls -la CLAUDE.md

# Restart Claude in correct directory
cd ~/crpbot
claude-code .
```

### Claude Doesn't Remember Context
```bash
# Remind Claude at start of each session
"Read CLAUDE.md to understand the project context"

# Or reference specific docs
"Check CLOUD_SERVER_SETUP.md for the setup steps"
```

### Command Not Found
```bash
# Verify installation
which claude-code

# If not installed, check official docs:
# https://docs.anthropic.com/claude/docs/claude-code
```

---

## 📚 Quick Command Reference

```bash
# Connect to server
ssh crpbot-cloud

# Go to project
cd ~/crpbot

# Start Claude with context
claude-code "Read CLAUDE.md and help me with [task]"

# Or interactive
claude-code .
# Then: "Read CLAUDE.md and summarize the project"

# Exit Claude
# Type: exit or Ctrl+D
```

---

## 🎯 Next Steps

1. **Install Claude Code CLI** on cloud server
2. **Authenticate** with your API key
3. **Navigate** to `~/crpbot`
4. **Start Claude** with context prompt
5. **Verify** Claude understands the project
6. **Start working** on your tasks!

All documentation is already in the repository, so Claude will have everything it needs once you connect it to the project directory.

---

## 💡 Pro Tip

Create an alias for quick access:

```bash
# Add to ~/.bashrc
echo 'alias claude="cd ~/crpbot && claude-code ."' >> ~/.bashrc
source ~/.bashrc

# Now just type:
claude
# And you're ready to work!
```

---

**Claude Code will be your AI pair programmer on the cloud server, with full knowledge of your project through CLAUDE.md!** 🚀

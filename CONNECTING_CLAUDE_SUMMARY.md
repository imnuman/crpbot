# Connecting Claude Code to Cloud Server - Complete Guide

Simple guide to get Claude Code working on your cloud server with full project context.

---

## 🎯 The Simple Answer

Claude Code will **automatically** have full project context because:

1. ✅ **CLAUDE.md exists in the repo** - This is the main project guide
2. ✅ **All documentation is in the repo** - Claude can read everything
3. ✅ **.claude-context file added** - Quick context reference
4. ✅ **Helper script created** - Generates perfect introduction

**Result**: Claude knows everything about your project instantly!

---

## 🚀 Quick Setup (3 Steps)

### Step 1: On Local Machine - Sync Files to Cloud

```bash
cd /home/numan/crpbot

# Sync credentials
./scripts/sync_credentials.sh user@your-server-ip

# Push latest code (includes CLAUDE.md and all docs)
git add .
git commit -m "Add Claude context files"
git push origin main
```

### Step 2: On Cloud Server - Pull Latest Code

```bash
# You're already connected, right?
cd ~/crpbot
git pull origin main

# Verify CLAUDE.md is there
ls -la CLAUDE.md .claude-context CLAUDE_ON_CLOUD_SETUP.md
```

### Step 3: Start Claude with Context

```bash
# Generate the perfect introduction
./scripts/claude_intro.sh

# This will show you a prompt like:
# "Hi Claude! Welcome to CRPBot. Read CLAUDE.md first..."

# Start Claude Code
claude-code .

# Paste the introduction prompt
# Claude will read CLAUDE.md and understand everything!
```

---

## 💡 What Claude Will Know

When Claude Code starts in `~/crpbot`, it can read:

### Automatically Available
- ✅ **CLAUDE.md** - Complete project guide (architecture, commands, status)
- ✅ **.claude-context** - Quick reference summary
- ✅ All code in `apps/`, `libs/`, `scripts/`
- ✅ All documentation files
- ✅ Git history and structure
- ✅ Test files
- ✅ Configuration files

### What Claude Understands
- 🎯 Project: CRPBot - FTMO crypto trading AI
- 🏗️ Architecture: LSTM + Transformer + RL ensemble
- 📊 Current Phase: Phase 1 complete, ready for validation
- 🔧 Tech Stack: Python 3.10, PyTorch, PostgreSQL, Redis, S3
- 📍 Environment: Cloud server (production AWS infrastructure)
- 🎯 Next Steps: Model validation → Paper trading → Live

---

## 🎬 Example First Session

### Your Commands
```bash
# Connect to cloud server (via Terminus or SSH)
ssh user@your-server-ip

# Go to project
cd ~/crpbot
source .venv/bin/activate

# Generate introduction
./scripts/claude_intro.sh
```

### Copy & Paste This to Claude
The script outputs a perfect prompt. Here's what it looks like:

```
Hi Claude! Welcome to the CRPBot cloud server environment.

📋 IMPORTANT: Please read CLAUDE.md first - it contains complete project instructions and architecture.

🔍 Current Environment Status:
- Location: /home/ubuntu/crpbot
- Python: 3.10.x
- Git: main @ abc1234
- Virtual Environment: ✅ Activated
- Credentials: .env ✅ Present, .db_password ✅ Present
- AWS Config: ✅ Present
- Data: 764MB in data/ directory
- Models: 10 model files
- Tests: ✅ Passing

[... full context ...]

🚀 Your First Tasks:
1. Read CLAUDE.md thoroughly to understand the project
2. Verify the environment status above looks correct
3. Check if there are any issues that need immediate attention
4. Summarize the project status and suggest next steps

Ready to help! What would you like to work on?
```

### Claude's Response
Claude will:
1. ✅ Read CLAUDE.md (all project info)
2. ✅ Understand the architecture
3. ✅ Know the current status
4. ✅ Summarize what it learned
5. ✅ Be ready to help!

---

## 📋 Different Ways to Start Claude

### Option 1: Quick Start (Recommended)
```bash
cd ~/crpbot
./scripts/claude_intro.sh  # Shows perfect prompt
claude-code .               # Start Claude
# Paste the prompt shown
```

### Option 2: Simple Start
```bash
cd ~/crpbot
claude-code "Read CLAUDE.md and help me validate the GPU models"
```

### Option 3: Interactive
```bash
cd ~/crpbot
claude-code .

# Then type:
"Read CLAUDE.md to understand the project, then help me with model validation"
```

### Option 4: VS Code Remote-SSH
```bash
# On local machine in VS Code:
# 1. Install Remote-SSH extension
# 2. Connect to your server
# 3. Open folder: ~/crpbot
# 4. Use Claude Code extension
# Claude automatically has access to all files
```

---

## 🔍 Verifying Claude Has Context

After starting Claude, ask:

```
"What do you know about this project? Summarize the key points from CLAUDE.md"
```

**Claude should respond with**:
- ✅ Project name: CRPBot
- ✅ Purpose: FTMO crypto trading with ML
- ✅ Architecture: LSTM + Transformer + RL ensemble
- ✅ Current phase: Phase 1 complete
- ✅ Next steps: Model validation
- ✅ Key features: FTMO compliance, rate limiting, kill switch

**If Claude doesn't have this**, simply say:
```
"Please read CLAUDE.md now"
```

---

## 🎯 Common Use Cases

### Daily Check-in
```bash
claude-code "Review PHASE1_COMPLETE_NEXT_STEPS.md and suggest today's priorities"
```

### Code Review
```bash
claude-code "Review the changes in apps/runtime/ and check against CLAUDE.md standards"
```

### Debugging
```bash
claude-code "The tests are failing. Help me debug based on the architecture in CLAUDE.md"
```

### Feature Development
```bash
claude-code "I want to add hourly win rate tracking. Based on CLAUDE.md architecture, where should this go?"
```

### Deployment Help
```bash
claude-code "Read PHASE1_COMPLETE_NEXT_STEPS.md and help me deploy the validated models"
```

---

## 🔐 Security & Best Practices

### What's Automatically Handled
- ✅ Claude can read `.env` but won't expose credentials
- ✅ Claude respects security in responses
- ✅ Credentials stay secure
- ✅ All files readable within project directory

### What to Remember
- 📝 This is the **cloud/production** environment
- 📝 Changes should be **committed and synced**
- 📝 Test before deploying
- 📝 Follow FTMO compliance rules

---

## 📚 Key Files Claude Should Read

### Must Read
1. **CLAUDE.md** - Main project guide ⭐⭐⭐
2. **PHASE1_COMPLETE_NEXT_STEPS.md** - Current status
3. **MASTER_SUMMARY.md** - Complete overview

### Reference as Needed
4. **CLOUD_SERVER_SETUP.md** - Server setup
5. **CLOUD_SERVER_QUICKSTART.md** - Daily commands
6. **README.md** - Quick start
7. **MIGRATION_GUIDE.md** - Migration details

### Code Files
- `apps/trainer/main.py` - Training entry point
- `apps/runtime/main.py` - Runtime entry point
- `apps/runtime/aws_runtime.py` - AWS-integrated runtime
- `scripts/evaluate_model.py` - Model evaluation

---

## 🛠️ Troubleshooting

### Claude says "I don't see CLAUDE.md"
```bash
# Verify file exists
cd ~/crpbot
ls -la CLAUDE.md

# Pull latest from git
git pull origin main

# Restart Claude in correct directory
pwd  # Should show ~/crpbot
claude-code .
```

### Claude doesn't understand the project
```bash
# Explicitly ask Claude to read
"Please read CLAUDE.md, PHASE1_COMPLETE_NEXT_STEPS.md, and MASTER_SUMMARY.md. Then summarize what you learned."
```

### Wrong directory
```bash
# Make sure you're in the project
cd ~/crpbot
pwd  # Verify
claude-code .
```

---

## 🎉 You're All Set!

### What You Have Now
1. ✅ **CLAUDE.md** in repo - Claude's main guide
2. ✅ **.claude-context** - Quick reference
3. ✅ **claude_intro.sh** - Generates perfect prompts
4. ✅ **CLAUDE_ON_CLOUD_SETUP.md** - Detailed setup guide
5. ✅ All documentation in repo

### What Happens When You Start Claude
1. ✅ Claude reads CLAUDE.md automatically
2. ✅ Understands full project context
3. ✅ Knows current status and next steps
4. ✅ Ready to help immediately
5. ✅ References docs as needed

### Your Simple Workflow
```bash
# Connect to cloud
ssh your-server

# Go to project
cd ~/crpbot && source .venv/bin/activate

# Generate intro (optional - helps Claude)
./scripts/claude_intro.sh

# Start Claude
claude-code .

# Paste intro or just say:
"Read CLAUDE.md and help me with [your task]"

# Claude is now your AI pair programmer with full context!
```

---

## 🚀 Next Steps

1. **Push these new files to git**:
   ```bash
   # On local machine
   cd /home/numan/crpbot
   git add .
   git commit -m "Add Claude context files for cloud server"
   git push origin main
   ```

2. **Pull on cloud server**:
   ```bash
   # On cloud server
   cd ~/crpbot
   git pull origin main
   ```

3. **Start Claude with context**:
   ```bash
   ./scripts/claude_intro.sh
   claude-code .
   ```

4. **Begin working**!

---

## 💡 Pro Tips

### Save Time with Aliases
```bash
# Add to ~/.bashrc on cloud server
alias claude-start='cd ~/crpbot && source .venv/bin/activate && ./scripts/claude_intro.sh && claude-code .'

# Then just type:
claude-start
```

### Create Project Templates
```bash
# Save common prompts
echo "Read CLAUDE.md and help me debug tests" > ~/claude-prompts/debug.txt
echo "Read CLAUDE.md and review my code changes" > ~/claude-prompts/review.txt
```

### Quick Context Refresh
```bash
# If Claude seems to forget context mid-session
"Refresh your understanding by reading CLAUDE.md again"
```

---

## ✨ The Magic

The beauty of this setup:
- 🎯 **Zero manual configuration** - Everything in git
- 📚 **Full context automatic** - CLAUDE.md does it all
- 🔄 **Always in sync** - Git keeps both environments aligned
- 🚀 **Instant onboarding** - New team members get same context
- 💯 **Consistent experience** - Same on local and cloud

**CLAUDE.md is the single source of truth that makes this all work!**

---

Ready to connect Claude to your cloud server? Just follow the 3 steps at the top! 🎉

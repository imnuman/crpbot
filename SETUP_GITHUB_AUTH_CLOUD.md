# Setup GitHub Auth on Cloud Server - Quick Fix

**Issue**: Cloud Claude can't push to GitHub (commit afdb584 stuck locally)
**Solution**: Configure GitHub auth on cloud server (5 minutes)

---

## 🚀 Quick Setup (Choose One Method)

### Method 1: Personal Access Token (Recommended - 3 minutes)

On cloud server:

```bash
# 1. Configure git user
git config --global user.name "Cloud Claude"
git config --global user.email "noreply@anthropic.com"

# 2. Use Personal Access Token for authentication
# Generate token at: https://github.com/settings/tokens
# Scopes needed: repo (full control)

# Store credentials
git config --global credential.helper store

# Next push will ask for credentials once, then remember
git push origin main
# Username: imnuman
# Password: <paste your Personal Access Token>
```

### Method 2: SSH Key (Better Security - 5 minutes)

On cloud server:

```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -C "cloud-claude@crpbot" -f ~/.ssh/github_cloud -N ""

# 2. Display public key
cat ~/.ssh/github_cloud.pub

# 3. Add to GitHub
# Copy the public key output
# Go to: https://github.com/settings/ssh/new
# Title: "CRPBot Cloud Server"
# Key: <paste public key>
# Click "Add SSH key"

# 4. Configure SSH
cat >> ~/.ssh/config <<EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_cloud
EOF

# 5. Test SSH connection
ssh -T git@github.com

# 6. Update remote URL to use SSH
cd /root/crpbot
git remote set-url origin git@github.com:imnuman/crpbot.git

# 7. Configure git user
git config --global user.name "Cloud Claude"
git config --global user.email "noreply@anthropic.com"

# 8. Test push
git push origin main
```

---

## ✅ After Setup

Test that it works:

```bash
# Create a test commit
cd /root/crpbot
echo "# Test" >> .test
git add .test
git commit -m "test: verify GitHub auth works"
git push origin main

# If successful:
git rm .test
git commit -m "chore: remove test file"
git push origin main
```

---

## 🔄 For Now (Temporary Workaround)

If you don't have time to set up auth right now:

**Cloud Claude**: Just commit locally, we'll push from local machine

```bash
# On cloud
git add .
git commit -m "your message"
# Don't push - just commit

# On local (I'll do this)
# Create patch from cloud's commits and apply locally
```

---

## 📋 Personal Access Token Setup (Detailed)

If using Method 1, here's how to create the token:

1. Go to GitHub: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: "CRPBot Cloud Server"
4. Expiration: 90 days (or No expiration)
5. Scopes: Check "repo" (full control of private repositories)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again!)

Use this token as the password when pushing:
```bash
git push origin main
Username: imnuman
Password: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # ← Your token
```

Git will remember it after first use.

---

## 🎯 Recommendation

**Use Method 1** (Personal Access Token) for now - it's faster.
**Upgrade to Method 2** (SSH key) later for better security.

Either way, setup takes <5 minutes and solves the push issue permanently.

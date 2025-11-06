# GitHub Token Setup Guide

## For GitHub Actions (CI/CD)

GitHub Actions automatically uses `GITHUB_TOKEN` (provided by GitHub) for most operations. You **don't need** to set up a personal access token for basic CI/CD workflows.

### When you DO need a Personal Access Token

You only need a Personal Access Token if:
1. **Deploying to VPS via GitHub Actions** (using the deploy workflow)
2. **Pushing to protected branches** from CI
3. **Accessing private repositories** from CI

### Setting up GitHub Secrets (for VPS deployment)

If you're using the GitHub Actions deployment workflow (`.github/workflows/deploy_vps.yml`), you need to add these secrets:

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add these secrets:

#### Required Secrets for VPS Deployment:

- **`VPS_HOST`**: Your VPS IP address (e.g., `203.0.113.10`)
- **`VPS_USER`**: SSH username (e.g., `ubuntu`)
- **`VPS_SSH_KEY`**: Your **private** SSH key contents (the entire key, not the public key)

#### Optional: Personal Access Token (if needed)

If you need a Personal Access Token for other operations:

1. Go to: **GitHub Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **"Generate new token (classic)"**
3. Select scopes:
   - `repo` (full control of private repositories)
   - `workflow` (update GitHub Action workflows)
4. Copy the token (you won't see it again!)
5. Add it as a secret named `GH_TOKEN` in your repository secrets

### Using Secrets in GitHub Actions

Secrets are accessed in workflows like this:
```yaml
- name: Deploy to VPS
  env:
    VPS_HOST: ${{ secrets.VPS_HOST }}
    VPS_USER: ${{ secrets.VPS_USER }}
    SSH_KEY: ${{ secrets.VPS_SSH_KEY }}
```

## For Local Development

For local development, you typically don't need a GitHub token unless:
- You're pushing to protected branches
- You're using GitHub CLI (`gh`)

### GitHub CLI Setup (Optional)

If you want to use GitHub CLI locally:

```bash
# Install GitHub CLI
sudo apt install gh  # or download from https://cli.github.com

# Authenticate
gh auth login

# Follow the prompts
```

## Security Best Practices

1. **Never commit tokens to git** - Always use `.env` (gitignored) or GitHub Secrets
2. **Use fine-grained tokens** when possible (newer token type)
3. **Rotate tokens regularly**
4. **Use tokens with minimal required scopes**
5. **Never share tokens in chat/email**

## Summary

- ✅ **For CI/CD**: GitHub provides `GITHUB_TOKEN` automatically
- ✅ **For VPS deployment**: Add `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` as secrets
- ✅ **For local development**: Usually not needed (unless using protected branches)


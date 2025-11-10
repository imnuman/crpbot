# Bugbot Integration Guide

Bugbot is a GitHub App that provides automated code reviews for pull requests. Follow the steps below to enable it for the CRPBot repository and use it effectively alongside Cursor, Claude, and Amazon Q.

---

## 1. Installation Requirements

1. **GitHub Admin Access** – You need owner/admin permissions on the repository or organization.
2. **Install the GitHub App**  
   - Visit [https://github.com/apps/bugbot](https://github.com/apps/bugbot)  
   - Click **Install** and select the `imnuman/crpbot` repository.
3. **Permissions** – Grant Bugbot access to:
   - Pull requests (read/write comments)
   - Checks/statuses (to report results)
   - Contents (read-only) so it can inspect code
4. **Optional Configuration** – Bugbot reads an optional `.github/bugbot.yml` for advanced settings (severity filters, labels, etc.). We currently rely on default behaviour; add the config later if needed.

> ✅ Once installed, Bugbot will appear in the “Checks” tab on PRs and can be mentioned in comments.

---

## 2. How to Request a Review

1. Open a pull request with clear title/description.
2. Add any context Bugbot should know (risk areas, new dependencies) in the PR body.
3. Trigger Bugbot by commenting on the PR (common commands):
   - `@Bugbot review` – full analysis
   - `@Bugbot re-review` – rerun after changes
   - `@Bugbot focus <file or area>` – optional, if you want targeted feedback
4. Bugbot will post findings as review comments or a summary. Address each finding, push fixes, and ask Bugbot to re-review if needed.

---

## 3. Recommended Workflow

| Stage | Cursor | Claude | Bugbot | Amazon Q |
|-------|--------|--------|--------|----------|
| Local development | Implement feature/fix, push branch | - | - | - |
| Pull request | Create PR with checklist | Prepare to review | Trigger Bugbot review | Verify infra impact (if relevant) |
| Review loop | Apply fixes based on feedback | Review Bugbot findings, add human commentary | Provide automated defect report | Adjust AWS resources if required |
| Merge | Squash/merge after approvals | Final QA check | Optionally rerun before merge | Monitor deployment post-merge |

Tips:
- Run tests locally before requesting Bugbot to avoid noisy failures.
- Mention Bugbot early in the review to catch regressions before human review.
- Capture action items from Bugbot in the PR description or task list.

---

## 4. Troubleshooting

- **Bugbot does not respond** – Confirm the GitHub App is installed on the repo and the comment syntax is correct (`@Bugbot ...`). Bots ignore comments from forks without permissions.
- **Permission errors** – Revisit the app settings and ensure it has access to pull requests and checks.
- **False positives** – Leave a PR comment explaining why the issue is safe, then proceed after human review.
- **Audit trail** – Bugbot posts results under the Checks tab; review logs there if comments are missing.

---

## 5. Next Steps

1. Confirm with the repository owner that the Bugbot app is installed.
2. Add a PR template reminder (optional) to ping Bugbot during reviews.
3. Update `docs/WORKFLOW_SETUP.md` if our process changes (e.g., new commands or config).

With Bugbot in place, we’ll have automated defect detection complementing Cursor’s changes, Claude’s review, and Amazon Q’s infrastructure checks.***


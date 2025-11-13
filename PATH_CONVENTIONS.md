# Path Conventions - Local vs Cloud

## Environment Paths

| Environment | Project Root | User |
|-------------|--------------|------|
| **Local** | `/home/numan/crpbot` | numan |
| **Cloud** | `/root/crpbot` | root |

## ✅ Path-Agnostic Code (GOOD)

### Use Relative Paths
```python
# When running from project root
data_file = 'data/raw/BTC-USD_1m.parquet'
model_file = 'models/lstm_BTC_USD_1m.pt'
```

### Use Dynamic Project Root
```python
from pathlib import Path

# Find project root from any module
project_root = Path(__file__).resolve().parent.parent.parent
db_password = project_root / '.db_password'
models_dir = project_root / 'models'
```

### Use Environment Variables
```python
import os
project_root = os.getenv('CRPBOT_ROOT', os.getcwd())
```

### Use Current Working Directory
```python
import os
# Assumes you always run from project root
data_dir = os.path.join(os.getcwd(), 'data', 'raw')
```

## ❌ Hardcoded Paths (BAD)

```python
# DON'T DO THIS - breaks on cloud server
with open('/home/numan/crpbot/.db_password') as f:
    password = f.read()

# DON'T DO THIS - breaks on local machine
models_dir = '/root/crpbot/models'
```

## Fixed Files

### ✅ `apps/runtime/aws_runtime.py`
**Changed from:**
```python
with open('/home/numan/crpbot/.db_password', 'r') as f:
models_dir = '/home/numan/crpbot/models'
```

**Changed to:**
```python
self.project_root = Path(__file__).resolve().parent.parent.parent
db_password_file = self.project_root / '.db_password'
models_dir = self.project_root / 'models'
```

## Shell Scripts - Best Practices

### Use Script Directory
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Now use $PROJECT_ROOT
cd "$PROJECT_ROOT"
source .venv/bin/activate
```

### Use PWD
```bash
#!/bin/bash
# Assumes script runs from project root
cd "$(dirname "$0")/.."
DATA_DIR="$(pwd)/data"
```

## Documentation Files

Documentation files (*.md) can reference example paths but should note both environments:

```markdown
# Example - adjust path for your environment:
Local: /home/numan/crpbot/.env
Cloud: /root/crpbot/.env
```

## QC Review Guidelines

When reviewing code from Cloud Claude, Local Claude should verify:
- ✅ No hardcoded `/root/crpbot` paths
- ✅ Uses relative paths or dynamic root finding
- ✅ Works in both environments

When reviewing code from Local Claude, Cloud Claude should verify:
- ✅ No hardcoded `/home/numan/crpbot` paths
- ✅ Uses relative paths or dynamic root finding
- ✅ Works in both environments

## Git Best Practices

### Never Commit
- `.cloud_server` (contains cloud IP)
- `.ssh_key` (contains SSH key path)
- Any files with hardcoded environment-specific paths

### Always Use Relative Paths In
- Python imports: `from libs.config import settings`
- Config files: `data_dir = "./data/raw"`
- Scripts: `cd "$(dirname "$0")"`

## Testing Path Compatibility

Run this test on both environments:
```python
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
assert (project_root / '.env').exists(), "Can't find .env"
assert (project_root / 'models').exists(), "Can't find models dir"
print(f"✅ Project root detected: {project_root}")
```

Expected output:
- Local: `✅ Project root detected: /home/numan/crpbot`
- Cloud: `✅ Project root detected: /root/crpbot`

## Summary

**Golden Rule**: Never hardcode full paths. Always use:
1. Relative paths (when working directory is known)
2. Dynamic root detection (`Path(__file__).resolve().parent...`)
3. Environment variables (for flexibility)

This ensures code works seamlessly in both local and cloud environments.

Build Blueprint (dev-first, ship-ready)
0) Repo layout (copy/paste)
trading-ai/
├─ apps/
│  ├─ trainer/                 # LSTM/Transformer/GAN/RL training
│  │  ├─ main.py
│  │  ├─ data_pipeline.py
│  │  ├─ models/               # model defs
│  │  ├─ train/                # train loops & schedulers
│  │  └─ eval/                 # backtests & metrics
│  ├─ runtime/                 # VPS runtime: scanning + signals + auto-learning
│  │  ├─ main.py
│  │  ├─ ftmo_rules.py
│  │  ├─ confidence.py
│  │  ├─ telegram_bot.py
│  │  └─ auto_learner.py
│  └─ mt5_bridge/              # FTMO/MT5 connectors (isolated)
│     └─ bridge.py
├─ libs/
│  ├─ features/                # OHLCV, ATR, spread, etc.
│  ├─ rl_env/                  # PPO Gym env with spread/slippage
│  └─ synth/                   # GAN data utilities
├─ infra/
│  ├─ docker/                  # Dockerfiles
│  ├─ devcontainer/            # .devcontainer (VS Code/Cursor)
│  ├─ actions/                 # GitHub Actions workflows
│  ├─ systemd/                 # service units for VPS
│  └─ scripts/                 # shell tools
├─ data/                       # .gitignore; use DVC/LFS for versions
├─ models/                     # saved weights (DVC/LFS tracked)
├─ notebooks/                  # experiments / EDA
├─ tests/                      # pytest; unit + smoke + e2e sims
├─ pyproject.toml              # use uv/poetry; pinned deps
├─ Makefile                    # task runner
├─ .pre-commit-config.yaml
├─ .env.example
└─ README.md


Why this layout: maps cleanly to the blueprint’s components (LSTM, Transformer, RL, FTMO rules, Telegram, auto-learning) and splits trainer (GPU work) from runtime (VPS bot) so you can iterate independently

Blueprint

Blueprint

.

1) Branching, PRs, and automation

Branches:

main = deployable; dev = integration; feature branches feat/*, fix/*.

Required PR checks (GitHub Actions):

Lint/type: Ruff + mypy.

Tests: unit + smoke + a tiny 5-minute backtest sim.

Security: bandit (fast).

Packaging: build wheel to ensure deps are sane.

CODEOWNERS: route AI core to you; infra to you; MT5 to you (solo = still nice for Claude routing).

Claude review prompts (put in PR template):

“Audit RL env for reward leakage, improper look-ahead, or data snooping.”

“Check confidence calibration logic for overfitting and leakage against validation split.”

“Verify FTMO rule checks are enforced in all execution codepaths (daily/total loss).”

“Flag any external state (wall-clock, exchange latency) not simulated in backtests.”

2) Dev environment (Cursor-friendly)

Package/tooling:

Python 3.10, uv (or Poetry) for fast, locked installs.

pre-commit (ruff, isort, black, mypy, bandit).

.env.example (keep secrets out of git):

BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
FTMO_LOGIN=
FTMO_PASS=
FTMO_SERVER=
DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
CONFIDENCE_THRESHOLD=0.75


Makefile (quick targets):

.PHONY: setup sync fmt lint test unit smoke train rl run-bot
setup:        ## first run
	uv sync && pre-commit install
fmt:
	ruff check --fix . && ruff format .
lint:
	ruff check . && mypy .
test:
	pytest -q
smoke:        ## tiny sim/backtest (5 mins)
	pytest -q tests/smoke
run-bot:
	python apps/runtime/main.py
train:
	python apps/trainer/main.py --task lstm --coin BTC
rl:
	python apps/trainer/main.py --task ppo --steps 200000


Devcontainer (optional but nice for consistency):

Image: mcr.microsoft.com/devcontainers/python:3.10

Installs: uv/poetry, ta-lib build deps, MetaTrader5 wheel prereqs.

3) Data + FTMO parity

Data: use Binance 1m candles (2020–2025) for BTC/ETH/BNB; add ATR, spread, volume, session features.

FTMO parity: build a spread/slippage layer to match FTMO execution when training and simulating; keep it in libs/rl_env/execution_model.py. This is mission-critical for realistic validation as the blueprint notes (expect slight accuracy drop after true execution modelling)

Blueprint

.

Versioning: DVC (S3 remote) for data/ and models/. Git LFS if you won’t set up DVC day 1.

4) Models & training flow (map to your GPUs)

Targets from the blueprint:

LSTM (direction, 15-min horizon), Transformer (trend strength), GAN for synthetic extension, RL (PPO) for decision/action set

Blueprint

.

Training passes (automated by flags):

# LSTM per coin
python apps/trainer/main.py --task lstm --coin BTC --epochs 10
python apps/trainer/main.py --task lstm --coin ETH --epochs 10
python apps/trainer/main.py --task lstm --coin BNB --epochs 10

# Transformer
python apps/trainer/main.py --task transformer --epochs 8

# GAN synth (capped to <=20% of total set)
python apps/trainer/main.py --task gan --epochs 10 --max_synth_ratio 0.2

# RL PPO with execution model
python apps/trainer/main.py --task ppo --steps 8_000_000 --exec ftmo


Logging: write scalars to runs/ (TensorBoard) + CSV for CI smoke snapshot.

Cursor Debugger: focus on:

exploding gradients (clip in PPO),

NaNs from TA features,

train/val split leakage.

5) Backtests, calibration, and safety rails

Backtests: run per-coin and combined; enforce that validation accuracy meets a floor (e.g., ≥ 0.68 per blueprint) before the model is “promotable” to runtime

Blueprint

.

Confidence system: ensemble(LSTM, Transformer, RL) + historical pattern adjustment; include a conservative bias (-5%) exactly as in the blueprint to avoid over-promise

Blueprint

.

FTMO rules: library with daily/total loss guardrails + position sizing helper used everywhere (training sims, runtime, signals)

Blueprint

.

6) Runtime (VPS) & Telegram

Runtime loop (every 2 minutes): scan coins → predict → score → enforce FTMO limits → emit signal to Telegram → start auto-tracking

Blueprint

.

Commands: /check, /stats, /ftmo_status, /threshold <n> (match blueprint)

Blueprint

.

Systemd units (auto-restart); logs to journald and rotate to S3 nightly.

Observability: minimal — structured JSON logs + a /healthz endpoint for uptime checks.

7) Observation & micro-testing

Micro-lot live tests (0.01 lots, $1–$2 risk) for 2–3 days; track delta between simulated and real fills, adjust slippage model; target ≥ 68% micro win rate before full signal mode

Blueprint

Blueprint

.

8) Validation & go-live

Simulate FTMO challenges (20 runs). Require ≥ 70% pass rate to proceed; record avg days to pass; then run final confidence calibration (±5% error) before switching to LIVE mode

Blueprint

.

Default runtime threshold = 0.75 (user overridable) to hit the blueprint’s 78–83% tier targets

Blueprint

.

GitHub Actions (drop-in)

.github/workflows/ci.yml

name: ci
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy .
      - run: uv run pytest -q
      - name: Smoke backtest (5 min)
        run: uv run pytest -q tests/smoke


.github/workflows/runtime-docker.yml (optional: build image on tag)

name: build-runtime
on:
  push:
    tags: ['v*.*.*']
jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: infra/docker/runtime.Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/trading-ai-runtime:${{ github.ref_name }}

VPS deploy (systemd)

infra/systemd/trading-ai.service

[Unit]
Description=Trading AI runtime
After=network.target

[Service]
WorkingDirectory=/opt/trading-ai
EnvironmentFile=/opt/trading-ai/.env
ExecStart=/usr/bin/python3 apps/runtime/main.py
Restart=always
RestartSec=5
User=trading
Group=trading
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target

What to change / add to improve your dev setup

High-impact

DVC with S3 remote for data/ and models/ — reproducibility and rollback become trivial.

uv (or Poetry) with lockfile — deterministic, fast installs; perfect with Cursor’s refactors.

Pre-commit (ruff/mypy/bandit) — catches 80% of bugs before you even run tests.

Tiny smoke backtest in CI — protects you from silent accuracy regressions.

COGS split (apps vs libs) — makes Cursor Debugger traces legible and PRs smaller.

Nice-to-have
6) Devcontainer — ensures Claude sees the same env you do; one-click reproduce.
7) Dockerized runtime — pushes a single GHCR image to your VPS; fewer “works on my box” issues.
8) Minimal metrics — CSV + TensorBoard is enough; add Prometheus only if needed.
9) Claude PR rules — require “AI review” label before merge on dev → main.
10) Secrets — use GitHub Environments + OIDC for cloud writes; keep .env local only.

When you start training
11) Lambda/Vast training scripts with --resume and --save-every N to avoid losing progress.
12) Gradient sanity checks in PPO (norms/NaN detectors) to keep RL stable.

Execution order (10–14 days realistic)

Scaffold repo + CI + pre-commit + Makefile (0.5d)

Data ingestion + FTMO parity layer (1–1.5d)

Blueprint

LSTM/Transformer baselines + eval gates (1.5–2d)

Blueprint

GAN synth (≤20%) + sanity checks (1d)

Blueprint

RL env + PPO with exec model + convergence (2–3d)

Blueprint

Confidence & auto-learning + backtests (1d)

Blueprint

Blueprint

Runtime + Telegram + VPS systemd (0.5–1d)

Blueprint

Observation + micro-lot tests (2–3d)

Blueprint

Validation sims + final calibration + go-live (0.5–1d)

Blueprint

This sequence preserves the blueprint’s technical intent (models, FTMO rules, confidence tiers, micro-tests, validation gate) while fitting your toolchain (Cursor + GitHub + Claude + Cursor Debugger)

Blueprint

Blueprint

.
Analyzed
Always show details
# Create a repo skeleton for "trading-ai" with essential files and stubs.
import os, json, textwrap, zipfile, io, pathlib

root = "/mnt/data/trading-ai"
paths = [
    "apps/trainer/models",
    "apps/trainer/train",
    "apps/trainer/eval",
    "apps/runtime",
    "apps/mt5_bridge",
    "libs/features",
    "libs/rl_env",
    "libs/synth",
    ".github/workflows",
    "infra/docker",
    "infra/devcontainer",
    "infra/systemd",
    "infra/scripts",
    "data",
    "models",
    "notebooks",
    "tests/smoke",
]

for p in paths:
    os.makedirs(os.path.join(root, p), exist_ok=True)

# Helper to write file
def w(rel, content):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(textwrap.dedent(content).lstrip())

# Top-level files
w(".gitignore", """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.venv/
.env
# Data & models (use DVC/LFS if desired)
data/**
models/**
!data/.gitkeep
!models/.gitkeep
# Misc
.DS_Store
.idea/
.vscode/
runs/
dist/
build/
""")

w(".env.example", """
BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
FTMO_LOGIN=
FTMO_PASS=
FTMO_SERVER=
DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
CONFIDENCE_THRESHOLD=0.75
""")

w("README.md", """
# trading-ai

Repo skeleton for FTMO-focused crypto signal generation (LSTM + Transformer + PPO RL)
with FTMO rules, confidence calibration, Telegram runtime, and auto-learning.

## Quick start
```bash
make setup        # install deps & pre-commit
make smoke        # run 5-min smoke backtest test
make run-bot      # start runtime loop (stub)


""")

w("pyproject.toml", """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "trading-ai"
version = "0.0.1"
description = "FTMO-oriented crypto signals with DL + RL"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
"numpy",
"pandas",
"scipy",
"pydantic>=2",
"python-telegram-bot>=13,<14",
"matplotlib",
"scikit-learn",
"torch>=2",
"gymnasium",
"tqdm",
"loguru",
"pyyaml",
"requests",
"ta",
"psycopg[binary]",
"sqlalchemy",
"pytest",
]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
""")

w("Makefile", """
.PHONY: setup fmt lint test smoke run-bot train rl

setup:
\tpip install -U pip
\tpip install uv pre-commit
\tuv pip install -e .
\tpre-commit install

fmt:
\truff check --fix . || true
\truff format . || true

lint:
\truff check .

test:
\tpytest

smoke:
\tpytest tests/smoke -q

run-bot:
\tpython apps/runtime/main.py

train:
\tpython apps/trainer/main.py --task lstm --coin BTC --epochs 1

rl:
\tpython apps/trainer/main.py --task ppo --steps 1000
""")

w(".pre-commit-config.yaml", """
repos:

repo: https://github.com/astral-sh/ruff-pre-commit

rev: v0.6.9
hooks:

id: ruff
args: [--fix]

id: ruff-format

repo: https://github.com/pre-commit/mirrors-mypy

rev: v1.13.0
hooks:

id: mypy
additional_dependencies: ["pydantic>=2"]

repo: https://github.com/PyCQA/bandit

rev: 1.7.8
hooks:

id: bandit
args: ["-q", "-r", "."]
""")

GitHub Actions

w(".github/workflows/ci.yml", """
name: ci
on: [push, pull_request]
jobs:
build-test:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
with:
python-version: '3.10'
- name: Install uv
run: pip install uv
- name: Install deps
run: uv pip install -e .
- name: Lint
run: uv run ruff check .
- name: Format check
run: uv run ruff format --check .
- name: Tests
run: uv run pytest -q
- name: Smoke backtest (5 min)
run: uv run pytest -q tests/smoke
""")

w(".github/workflows/runtime-docker.yml", """
name: build-runtime
on:
push:
tags: ['v*..']
jobs:
docker:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: docker/setup-buildx-action@v3
- uses: docker/login-action@v3
with:
registry: ghcr.io
username: ${{ github.actor }}
password: ${{ secrets.GITHUB_TOKEN }}
- uses: docker/build-push-action@v6
with:
context: .
file: infra/docker/runtime.Dockerfile
push: true
tags: ghcr.io/${{ github.repository_owner }}/trading-ai-runtime:${{ github.ref_name }}
""")

Infra

w("infra/docker/runtime.Dockerfile", """
FROM python:3.10-slim
WORKDIR /opt/trading-ai
COPY pyproject.toml README.md ./
RUN pip install -U pip && pip install uv && uv pip install .
COPY apps ./apps
COPY libs ./libs
COPY .env.example ./.env.example
CMD ["python", "apps/runtime/main.py"]
""")

w("infra/devcontainer/devcontainer.json", """
{
"name": "trading-ai",
"image": "mcr.microsoft.com/devcontainers/python:3.10",
"features": {
"ghcr.io/devcontainers/features/git:1": {}
},
"postCreateCommand": "pip install uv pre-commit && uv pip install -e . && pre-commit install"
}
""")

w("infra/systemd/trading-ai.service", """
[Unit]
Description=Trading AI runtime
After=network.target

[Service]
WorkingDirectory=/opt/trading-ai
EnvironmentFile=/opt/trading-ai/.env
ExecStart=/usr/bin/python3 /opt/trading-ai/apps/runtime/main.py
Restart=always
RestartSec=5
User=trading
Group=trading
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
""")

w("infra/scripts/bootstrap.sh", """
#!/usr/bin/env bash
set -euo pipefail
sudo useradd -r -s /usr/sbin/nologin trading || true
sudo mkdir -p /opt/trading-ai
sudo chown -R trading:trading /opt/trading-ai
echo "Bootstrap complete"
""")

w("infra/scripts/seed_db.py", """
from sqlalchemy import create_engine, text
import os

DB_URL = os.getenv("DB_URL", "sqlite:///tradingai.db")
engine = create_engine(DB_URL)
with engine.begin() as conn:
conn.execute(text("CREATE TABLE IF NOT EXISTS patterns(id INTEGER PRIMARY KEY, name TEXT, wins INT DEFAULT 0, total INT DEFAULT 0)"))
print("DB seeded (patterns table).")
""")

Data & models keepers

w("data/.gitkeep", "")
w("models/.gitkeep", "")
w("notebooks/.gitkeep", "")

Trainer stubs

w("apps/trainer/main.py", """
import argparse
from loguru import logger

def train_lstm(coin: str, epochs: int):
logger.info(f"Training LSTM for {coin} for {epochs} epochs (stub).")

def train_transformer(epochs: int):
logger.info(f"Training Transformer for {epochs} epochs (stub).")

def train_gan(epochs: int, max_ratio: float):
logger.info(f"Training GAN for {epochs} epochs (max synth ratio {max_ratio}) (stub).")

def train_ppo(steps: int, exec_model: str):
logger.info(f"Training PPO for {steps} steps with exec model {exec_model} (stub).")

if name == "main":
p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["lstm", "transformer", "gan", "ppo"])
p.add_argument("--coin", default="BTC")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--max_synth_ratio", type=float, default=0.2)
p.add_argument("--exec", dest="exec_model", default="ftmo")
args = p.parse_args()

Always show details
if args.task == "lstm":
    train_lstm(args.coin, args.epochs)
elif args.task == "transformer":
    train_transformer(args.epochs)
elif args.task == "gan":
    train_gan(args.epochs, args.max_synth_ratio)
elif args.task == "ppo":
    train_ppo(args.steps, args.exec_model)


""")

w("apps/trainer/data_pipeline.py", """
from typing import Tuple
import pandas as pd

def load_clean_data(symbol: str) -> pd.DataFrame:
# TODO: implement Binance/FTMO parity pipeline
return pd.DataFrame({"close": [1,2,3], "volume": [10,20,30]})
""")

w("apps/trainer/models/init.py", "")
w("apps/trainer/models/lstm.py", """
import torch.nn as nn

class LSTMModel(nn.Module):
def init(self, input_dim=8, hidden=64):
super().init()
self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)
self.out = nn.Linear(hidden, 1)

Always show details
def forward(self, x):
    y,_ = self.rnn(x)
    return self.out(y[:,-1,:])


""")
w("apps/trainer/models/transformer.py", """
import torch.nn as nn

class TinyTransformer(nn.Module):
def init(self, d_model=64, nhead=4, dim_feedforward=64, num_layers=2):
super().init()
layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
self.out = nn.Linear(d_model, 1)
def forward(self, x):
h = self.enc(x)
return self.out(h[:,-1,:])
""")
w("apps/trainer/models/gan.py", """
class DummyGAN:
def fit(self, data): # stub
return
""")
w("apps/trainer/train/loops.py", """
def train_loop(model, data_loader, epochs=1): # stub
for _ in range(epochs):
pass
""")
w("apps/trainer/eval/backtest.py", """
def run_backtest(minutes=120) -> float:
"""Return a dummy win-rate for smoke tests."""
# This should be replaced with a real backtest using FTMO exec model.
return 0.70
""")

Runtime stubs

w("apps/runtime/main.py", """
import os, time
from loguru import logger
from .ftmo_rules import check_daily_loss, check_total_loss
from .confidence import score_confidence
from .telegram_bot import send_message

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

def loop_once():
# Dummy state → pretend we generated one HIGH signal
confidence = score_confidence(0.72, 0.70, 0.68)
if confidence >= THRESHOLD and check_daily_loss(100000, -500) and check_total_loss(100000, 0):
send_message(f"[SMOKE] HIGH signal @ {round(confidence*100,1)}%")

if name == "main":
logger.info("Runtime starting (stub).")
for _ in range(3): # short run for smoke
loop_once()
time.sleep(1)
logger.info("Runtime exiting (stub).")
""")

w("apps/runtime/ftmo_rules.py", """
def check_daily_loss(account_size: float, daily_pl: float) -> bool:
max_loss = account_size * 0.045
return daily_pl > -max_loss

def check_total_loss(account_size: float, total_pl: float) -> bool:
max_loss = account_size * 0.09
return total_pl > -max_loss
""")

w("apps/runtime/confidence.py", """
def score_confidence(lstm_prob: float, trans_score: float, rl_prob: float) -> float:
# Simple ensemble + conservative bias (-5%)
ensemble = 0.35lstm_prob + 0.40trans_score + 0.25*rl_prob
return max(0.0, ensemble - 0.05)
""")

w("apps/runtime/telegram_bot.py", """
import os
from loguru import logger

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str):
# Stub: avoid external calls during tests
logger.info(f"TELEGRAM → {CHAT_ID}: {text}")
""")

w("apps/runtime/auto_learner.py", """
def record_result(signal_id: str, won: bool):
# stub
return
""")

MT5 bridge stub

w("apps/mt5_bridge/bridge.py", """
def get_ftmo_data(symbol: str, timeframe: str, bars: int):
# Stub placeholder
return []
""")

libs

w("libs/features/ta_features.py", """
def add_basic_features(df):
# stub: add ATR/spread etc.
return df
""")

w("libs/rl_env/execution_model.py", """
def simulate_fill(price: float, direction: str, spread_bps: float = 12.0, slippage_bps: float = 3.0) -> float:
"""Apply FTMO-like spread/slippage (bps)."""
bps = (spread_bps + slippage_bps)/10000.0
return price * (1 + bps if direction == "LONG" else 1 - bps)
""")

w("libs/rl_env/env.py", """
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
metadata = {"render.modes": []}
def init(self):
super().init()
self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
self.action_space = spaces.Discrete(3)

Always show details
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    return self.observation_space.sample(), {}

def step(self, action):
    obs = self.observation_space.sample()
    reward = float(np.random.randn() * 0.01)
    done = False
    return obs, reward, done, False, {}


""")

w("libs/synth/gan_utils.py", """
def generate_synthetic(df, ratio=0.2):
# stub
return df
""")

Tests

w("tests/init.py", "")
w("tests/test_smoke.py", """
from apps.trainer.eval.backtest import run_backtest

def test_backtest_winrate_floor():
assert run_backtest() >= 0.65
""")
w("tests/smoke/test_backtest_smoke.py", """
from apps.trainer.eval.backtest import run_backtest

def test_smoke_backtest():
wr = run_backtest()
assert 0.50 <= wr <= 0.90
""")

Zip it for easy download

zip_path = "/mnt/data/trading-ai-skeleton.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
for dirpath, _, filenames in os.walk(root):
for fn in filenames:
full = os.path.join(dirpath, fn)
rel = os.path.relpath(full, "/mnt/data")
zf.write(full, rel)

zip_path

Always show details
Result
'/mnt/data/trading-ai-skeleton.zip'

Done. Repo skeleton’s ready.

Download: trading-ai-skeleton.zip

What you get
trading-ai/
├─ apps/
│  ├─ trainer/              # LSTM/Transformer/GAN/RL training stubs
│  ├─ runtime/              # FTMO rules, confidence, Telegram (stubbed)
│  └─ mt5_bridge/           # MT5/FTMO connector (stub)
├─ libs/                    # features, RL env, exec model (FTMO-like)
├─ infra/                   # Dockerfile, devcontainer, systemd, scripts
├─ .github/workflows/       # CI + runtime image build
├─ tests/                   # unit + smoke backtest
├─ data/, models/, notebooks/ (.gitkeep)
├─ pyproject.toml, Makefile, .env.example, .pre-commit-config.yaml
└─ README.md, .gitignore

Quick start
unzip trading-ai-skeleton.zip
cd trading-ai
make setup          # installs uv+deps, sets up pre-commit
make smoke          # runs tiny smoke tests (dummy backtest)
make run-bot        # runs a short stub runtime loop

Next steps (Cursor + Claude)

Open folder in Cursor → “Install deps” → run make setup.

Create dev branch. Start with:

apps/trainer/data_pipeline.py → real Binance/FTMO parity.

libs/rl_env/execution_model.py → replace stub with measured FTMO spreads/slippage.

apps/runtime/telegram_bot.py → swap stub logger for python-telegram-bot send.

Push, open PR → let Claude review with your prompts (RL leakage, FTMO guardrails, confidence calibration).

Tag v0.1.0 when runtime is stable to build/push the Docker image via GH Actions.




Analyzed
Always show details
# Add DVC remote template and deployment scripts, update Makefile, and repackage the zip.
import os, textwrap, zipfile

root = "/mnt/data/trading-ai"

def w(rel, content):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(textwrap.dedent(content).lstrip())

# DVC template files and setup script
w("DVC_README.md", """
# DVC Setup (Template)

This repo recommends using DVC to version large artifacts in `data/` and `models/` with an S3 remote.

## One-time setup

1. Install DVC with S3 support:
   ```bash
   pip install 'dvc[s3]'


Initialize DVC in this repo:

Always show details
dvc init
git add .dvc .gitignore
git commit -m "chore: init dvc"


Configure an S3 remote (replace with your bucket/region):

Always show details
dvc remote add -d s3store s3://YOUR_BUCKET_NAME/trading-ai
dvc remote modify s3store region YOUR_AWS_REGION
# Optional: custom endpoint for non-AWS S3 (e.g., Wasabi/MinIO)
# dvc remote modify s3store endpointurl https://s3.YOUR_PROVIDER.com


Track large dirs:

Always show details
dvc add data models
git add data.dvc models.dvc .gitignore
git commit -m "chore: track data & models via dvc"


Push artifacts:

Always show details
dvc push


Pull on a new machine:

Always show details
dvc pull


Keep secrets (AWS creds) out of git. Use environment variables or AWS profiles/roles.
""")

w("infra/scripts/setup_dvc.sh", """
#!/usr/bin/env bash
set -euo pipefail

BUCKET="${1:-your-bucket-name}"
PREFIX="${2:-trading-ai}"
REGION="${3:-us-east-1}"
ENDPOINT="${4:-}" # optional (e.g., https://s3.us-east-1.wasabisys.com
)

echo "[i] Installing DVC (s3)"
pip install 'dvc[s3]'

if [ ! -d ".dvc" ]; then
echo "[i] dvc init"
dvc init
git add .dvc .gitignore
git commit -m "chore: init dvc" || true
fi

echo "[i] Configure remote s3store -> s3://${BUCKET}/${PREFIX} (region: ${REGION})"
dvc remote add -d s3store "s3://${BUCKET}/${PREFIX}" || dvc remote modify s3store url "s3://${BUCKET}/${PREFIX}"
dvc remote modify s3store region "${REGION}"
if [ -n "${ENDPOINT}" ]; then
dvc remote modify s3store endpointurl "${ENDPOINT}"
fi

echo "[i] dvc add data models"
dvc add data models

echo "[i] Commit DVC metas"
git add data.dvc models.dvc .gitignore
git commit -m "chore: track data & models with DVC" || true

echo "[i] To push artifacts: dvc push"
""")
os.chmod(os.path.join(root, "infra/scripts/setup_dvc.sh"), 0o755)

Systemd deploy script using rsync

w("infra/scripts/deploy_vps.sh", """
#!/usr/bin/env bash
set -euo pipefail

Usage: ./infra/scripts/deploy_vps.sh ubuntu@YOUR_VPS_IP /opt/trading-ai

HOST="${1:?remote host e.g. ubuntu@1.2.3.4}"
DEST="${2:-/opt/trading-ai}"

echo "[i] Building runtime image not required (systemd runs from source)."
echo "[i] Rsyncing sources to ${HOST}:${DEST}"
ssh -o StrictHostKeyChecking=no "${HOST}" "sudo mkdir -p '${DEST}' && sudo chown -R $USER '$DEST'"
rsync -az --delete
--exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'models' --exclude 'pycache'
./ "${HOST}:${DEST}/"

echo "[i] Installing Python deps on remote via uv"
ssh "${HOST}" "cd '${DEST}' && python3 -m pip install -U pip uv && uv pip install -e ."

echo "[i] Installing systemd unit"
scp infra/systemd/trading-ai.service "${HOST}:/tmp/trading-ai.service"
ssh "${HOST}" "sudo mv /tmp/trading-ai.service /etc/systemd/system/trading-ai.service && sudo systemctl daemon-reload && sudo systemctl enable trading-ai --now"

echo '[i] Tail logs: ssh %s "sudo journalctl -u trading-ai -f --no-pager"' "${HOST}"
""")
os.chmod(os.path.join(root, "infra/scripts/deploy_vps.sh"), 0o755)

Optional installer to create service user and env file placeholder

w("infra/systemd/install_service.sh", """
#!/usr/bin/env bash
set -euo pipefail
DEST="/opt/trading-ai"
sudo useradd -r -s /usr/sbin/nologin trading || true
sudo mkdir -p "$DEST"
sudo chown -R $USER "$DEST"
sudo tee /etc/systemd/system/trading-ai.service >/dev/null <<'EOF'
[Unit]
Description=Trading AI runtime
After=network.target

[Service]
WorkingDirectory=/opt/trading-ai
EnvironmentFile=/opt/trading-ai/.env
ExecStart=/usr/bin/python3 /opt/trading-ai/apps/runtime/main.py
Restart=always
RestartSec=5
User=trading
Group=trading
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "[i] Service installed. Place your code in $DEST and create $DEST/.env"
""")
os.chmod(os.path.join(root, "infra/systemd/install_service.sh"), 0o755)

Update Makefile with dvc/deploy targets (append if not present)

makefile_path = os.path.join(root, "Makefile")
with open(makefile_path, "a") as f:
f.write(textwrap.dedent("""

--- DVC & Deploy helpers ---

.PHONY: dvc-init dvc-add dvc-push deploy

dvc-init:
\tbash infra/scripts/setup_dvc.sh your-bucket-name trading-ai us-east-1

dvc-add:
\tdvc add data models && git add data.dvc models.dvc .gitignore && git commit -m "chore: dvc track data/models" || true

dvc-push:
\tdvc push

deploy:
\tbash infra/scripts/deploy_vps.sh ubuntu@YOUR_VPS_IP /opt/trading-ai
"""))

Re-zip

zip_path = "/mnt/data/trading-ai-skeleton-with-dvc-deploy.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
for dirpath, _, filenames in os.walk(root):
for fn in filenames:
full = os.path.join(dirpath, fn)
rel = os.path.relpath(full, "/mnt/data")
zf.write(full, rel)

zip_path

Always show details
Result
'/mnt/data/trading-ai-skeleton-with-dvc-deploy.zip'

All set — added a DVC template and a turnkey VPS deploy script.

Download:

trading-ai-skeleton-with-dvc-deploy.zip

What’s new

DVC (S3)

DVC_README.md – step-by-step setup for dvc[s3], remote config, add/push/pull.

infra/scripts/setup_dvc.sh – one-liner bootstrap:

bash infra/scripts/setup_dvc.sh your-bucket trading-ai us-east-1


(Supports optional endpoint for Wasabi/MinIO.)

Systemd deploy to VPS

infra/scripts/deploy_vps.sh – rsync code to remote, install deps with uv, install/enable systemd unit.

# usage
bash infra/scripts/deploy_vps.sh ubuntu@YOUR_VPS_IP /opt/trading-ai
# then view logs
ssh ubuntu@YOUR_VPS_IP "sudo journalctl -u trading-ai -f --no-pager"


infra/systemd/install_service.sh – optional helper to pre-install the unit and create the service user.

Makefile additions

make dvc-init – runs the DVC setup script with defaults.

make dvc-add and make dvc-push – quick DVC flows.

make deploy – wraps the rsync + systemd deploy.

Suggested flow

DVC on (local):

make dvc-init                      # or run setup_dvc.sh with your bucket/region
export AWS_PROFILE=your_profile    # or set AWS creds env vars
dvc push


Deploy (to your VPS):

make deploy         # uses deploy_vps.sh (edit host/ip once)
# Tail logs
ssh ubuntu@YOUR_VPS_IP "sudo journalctl -u trading-ai -f --no-pager"


Run-time .env on VPS (required):

/opt/trading-ai/.env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
FTMO_LOGIN=...
FTMO_PASS=...
FTMO_SERVER=...
CONFIDENCE_THRESHOLD=0.75
DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai



Analyzed
Always show details
# Add a GitHub Actions deploy workflow that ships code to a VPS over SSH/SCP
import os, textwrap, zipfile

root = "/mnt/data/trading-ai"

def w(rel, content):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(textwrap.dedent(content).lstrip())

w(".github/workflows/deploy_vps.yml", """
name: deploy-vps

on:
  workflow_dispatch:
    inputs:
      dest:
        description: "Remote destination path"
        required: false
        default: "/opt/trading-ai"

jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      DEST: ${{ github.event.inputs.dest }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set DEST if empty
        run: |
          if [ -z "${DEST}" ]; then
            echo "DEST=/opt/trading-ai" >> $GITHUB_ENV
          fi

      - name: Upload sources to VPS
        uses: appleboy/scp-action@v0.1.7
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          source: |
            .
            !.git/**
            !.venv/**
            !data/**
            !models/**
            !__pycache__/**
          target: ${{ env.DEST }}
          overwrite: true
          strip_components: 0

      - name: Remote deploy steps
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            set -euo pipefail
            cd "${DEST}"
            # Ensure Python/tooling present
            if ! command -v python3 >/dev/null 2>&1; then
              sudo apt-get update -y && sudo apt-get install -y python3 python3-pip
            fi
            python3 -m pip install -U pip uv
            uv pip install -e .
            # Install or update systemd unit
            sudo mv infra/systemd/trading-ai.service /etc/systemd/system/trading-ai.service || true
            sudo systemctl daemon-reload
            sudo systemctl enable trading-ai || true
            sudo systemctl restart trading-ai || true
            echo "[ok] Deployed to ${DEST} and (re)started service"
""")

# Update README with deploy instructions
readme_path = os.path.join(root, "README.md")
with open(readme_path, "a") as f:
    f.write(textwrap.dedent("""

## GitHub Actions: One-click Deploy to VPS

This repo includes `.github/workflows/deploy_vps.yml` which:
- Copies sources to your VPS via SCP
- Installs Python deps with `uv`
- Installs/restarts the `trading-ai` systemd service

### Setup secrets (in GitHub → Settings → Secrets and variables → Actions)
- `VPS_HOST` = e.g., `203.0.113.10`
- `VPS_USER` = e.g., `ubuntu`
- `VPS_SSH_KEY` = contents of your **private** SSH key (PEM)

### Run it
- Go to GitHub → Actions → **deploy-vps** → **Run workflow**
- Optional input: destination path (default `/opt/trading-ai`)

> Ensure `/opt/trading-ai/.env` exists on the VPS with your runtime secrets.
"""))

# Re-zip the repo
zip_path = "/mnt/data/trading-ai-skeleton-with-dvc-deploy-actions.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, "/mnt/data")
            zf.write(full, rel)

zip_path

Result
'/mnt/data/trading-ai-skeleton-with-dvc-deploy-actions.zip'

Deployed the CI “button.”

Download:

trading-ai-skeleton-with-dvc-deploy-actions.zip

What I added

GitHub Actions: one-click deploy

.github/workflows/deploy_vps.yml

Trigger: workflow_dispatch (manual run from Actions tab)

Inputs: dest (defaults to /opt/trading-ai)

Uses appleboy/scp-action to copy the repo to your VPS (excludes .git, .venv, data/, models/)

Uses appleboy/ssh-action to:

Install pip/uv if needed

uv pip install -e .

Install/refresh systemd unit

enable + restart service

README updates

Clear instructions for required repo secrets:

VPS_HOST (e.g., 203.0.113.10)

VPS_USER (e.g., ubuntu)

VPS_SSH_KEY (paste your private key contents)

How to use it

Add secrets in GitHub → Settings → Secrets and variables → Actions:

VPS_HOST, VPS_USER, VPS_SSH_KEY

On the VPS, create runtime env:

sudo mkdir -p /opt/trading-ai
sudo nano /opt/trading-ai/.env


Fill in the tokens/credentials.

Run the workflow: GitHub → Actions → deploy-vps → Run → (optional) set dest.

Watch logs:

ssh ubuntu@YOUR_VPS_IP "sudo journalctl -u trading-ai -f --no-pager"

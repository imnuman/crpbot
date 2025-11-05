.PHONY: setup sync fmt lint test unit smoke train rl run-bot help
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Trading AI - Makefile Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First run: install dependencies and pre-commit hooks
	pip install -U pip
	pip install uv pre-commit
	uv pip install -e .
	uv pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Setup complete! Run 'make help' to see all commands."

sync: ## Sync dependencies with uv
	uv pip sync pyproject.toml

fmt: ## Format code with ruff
	ruff check --fix .
	ruff format .

lint: ## Run linting (ruff + mypy)
	ruff check .
	mypy .

test: ## Run all tests
	pytest

unit: ## Run unit tests only
	pytest tests/ -k "not smoke"

smoke: ## Run smoke tests (5-min backtest)
	pytest tests/smoke -q

train: ## Train LSTM model (example: make train COIN=BTC EPOCHS=10)
	python apps/trainer/main.py --task lstm --coin $(or $(COIN),BTC) --epochs $(or $(EPOCHS),1)

rl: ## Train RL model (example: make rl STEPS=1000)
	python apps/trainer/main.py --task ppo --steps $(or $(STEPS),1000)

run-bot: ## Start runtime loop
	python apps/runtime/main.py

# DVC & Deploy helpers
.PHONY: dvc-init dvc-add dvc-push deploy rollback-model

dvc-init: ## Initialize DVC with S3 remote (requires BUCKET, REGION env vars)
	bash infra/scripts/setup_dvc.sh $(or $(BUCKET),your-bucket-name) trading-ai $(or $(REGION),us-east-1)

dvc-add: ## Add data/models to DVC tracking
	dvc add data models && git add data.dvc models.dvc .gitignore && git commit -m "chore: dvc track data/models" || true

dvc-push: ## Push DVC artifacts to remote
	dvc push

deploy: ## Deploy to VPS (requires VPS_HOST env var)
	bash infra/scripts/deploy_vps.sh $(or $(VPS_HOST),ubuntu@YOUR_VPS_IP) /opt/trading-ai

rollback-model: ## Rollback to previous model version (requires TAG env var)
	@if [ -z "$(TAG)" ]; then \
		echo "❌ Error: TAG is required. Usage: make rollback-model TAG=v1.0.0"; \
		exit 1; \
	fi
	bash infra/scripts/rollback_model.sh $(TAG)


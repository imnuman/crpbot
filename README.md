# HYDRA 3.0 - Multi-Agent AI Trading System

A multi-agent cryptocurrency trading system with 4 competing AI engines (DeepSeek, Claude, Grok, Gemini) that generate, vote on, and evolve trading strategies through tournament-based competition.

## Architecture

```
                    HYDRA 3.0 RUNTIME
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  │ Engine A │ │ Engine B │ │ Engine C │ │ Engine D │
    │  │ DeepSeek │ │  Claude  │ │   Grok   │ │  Gemini  │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
    │       │            │            │            │
    │       └────────────┴─────┬──────┴────────────┘
    │                          │
    │                          ▼
    │  ┌─────────────────────────────────────────────┐
    │  │         STRATEGY MEMORY (80/20)             │
    │  │       80% Exploit / 20% Explore             │
    │  └─────────────────────────────────────────────┘
    │                          │
    │                          ▼
    │  ┌─────────────────────────────────────────────┐
    │  │           CONSENSUS ENGINE                   │
    │  │     Weight-aware voting + Tiebreaker        │
    │  └─────────────────────────────────────────────┘
    │                          │
    │                          ▼
    │  ┌─────────────────────────────────────────────┐
    │  │         TOURNAMENT SYSTEM                    │
    │  │     Rankings → Weights (40/30/20/10%)       │
    │  └─────────────────────────────────────────────┘
    │                          │
    │                          ▼
    │  ┌─────────────────────────────────────────────┐
    │  │           GUARDIAN (9 Rules)                │
    │  │   Daily 2% | Max DD 6% | Emergency 3%       │
    │  └─────────────────────────────────────────────┘
    │                          │
    │                          ▼
    │  ┌─────────────────────────────────────────────┐
    │  │             PAPER TRADER                    │
    │  │    Simulates trades, tracks P&L             │
    │  └─────────────────────────────────────────────┘
    │                                                 │
    └─────────────────────────────────────────────────┘
```

## Features

### Multi-Agent Competition
- **4 AI Engines**: DeepSeek, Claude, Grok, Gemini
- **Tournament System**: Performance-based weight allocation (40/30/20/10%)
- **Consensus Voting**: Weighted votes with confidence tiebreaker
- **Strategy Evolution**: Genetic breeding of winning strategies

### Strategy Memory
- **80/20 Exploit/Explore**: Reuse winning strategies 80% of the time
- **Performance Scoring**: `win_rate * sqrt(trades) * (1 + avg_rr/10)`
- **Persistent Storage**: JSON database per engine/asset/regime

### Risk Management (Guardian)
| Rule | Limit | Action |
|------|-------|--------|
| Daily Loss | 2% | Stop trading |
| Max Drawdown | 6% | Survival mode (50% size) |
| Emergency | 3% daily | 24hr shutdown |
| Risk/Trade | 1% max | Auto-adjust size |
| Max Positions | 3 | Block new trades |

### Monitoring Stack
- **Grafana**: Dashboards at http://server:3000
- **Prometheus**: Metrics at http://server:9090
- **Loki**: Log aggregation for agent conversations
- **Alertmanager**: Risk alerts and notifications

## Quick Start

```bash
# Clone and setup
git clone https://github.com/imnuman/crpbot.git
cd crpbot
make setup

# Configure environment
cp .env.example .env
# Edit .env with your API keys (DeepSeek, Claude, Grok, Gemini, Coinbase)

# Start HYDRA runtime (paper trading)
docker compose up -d

# Or run directly
uv run python apps/runtime/hydra_runtime.py --paper --assets BTC-USD ETH-USD SOL-USD
```

## Configuration

### Required API Keys
```bash
# .env file
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
GEMINI_API_KEY=AIza...
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...
```

### Data Source
- **Primary**: Coinbase Advanced Trade API (OHLCV data)
- **Market Context**: CoinGecko API (sentiment, market cap)

## Project Structure

```
crpbot/
├── apps/
│   ├── runtime/
│   │   └── hydra_runtime.py      # Main HYDRA orchestrator
│   └── dashboard_web/            # Web dashboard
├── libs/
│   ├── hydra/
│   │   ├── engines/              # 4 AI engine implementations
│   │   ├── consensus.py          # Voting system
│   │   ├── guardian.py           # Risk management
│   │   ├── strategy_memory.py    # Strategy database
│   │   ├── tournament_manager.py # Tournament system
│   │   ├── paper_trader.py       # Trade simulation
│   │   └── regime_detector.py    # Market regime classification
│   ├── data/
│   │   └── coinbase.py           # Market data client
│   └── monitoring/
│       └── metrics.py            # Prometheus metrics
├── monitoring/
│   ├── docker-compose.yml        # Grafana/Prometheus/Loki
│   └── grafana/dashboards/       # Pre-built dashboards
└── data/hydra/
    └── strategy_database.json    # Persistent strategy memory
```

## Monitoring

### Grafana Dashboards
- **Command Center**: System status, P&L, win rate
- **Engine Analytics**: Per-engine performance comparison
- **Risk Dashboard**: Drawdown, position sizing, Guardian status
- **Regime Analytics**: Market state classification

### Log Queries (Loki)
```
{container="hydra-runtime"}                    # All logs
{container="hydra-runtime"} |= "Gladiator A"   # DeepSeek only
{container="hydra-runtime"} |= "votes"         # Vote decisions
{container="hydra-runtime"} |= "CONSENSUS"     # Trade signals
```

## Development

```bash
make fmt          # Format code (ruff)
make lint         # Lint (ruff + mypy)
make test         # Run tests
```

## Signal Flow

1. **Market Data** - Fetch OHLCV from Coinbase
2. **Regime Detection** - Classify as TRENDING/RANGING/CHOPPY
3. **Strategy Selection** - 80% exploit memory / 20% explore new
4. **Strategy Generation** - Each engine proposes a strategy
5. **Voting Round** - All engines vote on all strategies
6. **Consensus** - Weight-aware voting with confidence tiebreaker
7. **Guardian Check** - Validate against 9 sacred rules
8. **Paper Trade** - Execute simulated trade
9. **Performance Update** - Record outcome, update rankings
10. **Strategy Evolution** - Top strategies breed, losers die

## Position Sizing

| Consensus | Engines Agree | Size Modifier |
|-----------|---------------|---------------|
| UNANIMOUS | 4/4 | 100% |
| STRONG | 3/4 | 75% |
| WEAK | 2/4 | 50% |
| NO_TRADE | <2/4 | 0% |

## License

Private - Proprietary trading system

---

**Status**: Production (Paper Trading)
**Version**: HYDRA 3.0
**Last Updated**: December 2024

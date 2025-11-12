-- CryptoBot Production Database Schema
-- PostgreSQL 15.4

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS metrics;

-- =====================================================
-- TRADING SCHEMA: Operational trading data
-- =====================================================

-- Trades table: Execution history
CREATE TABLE trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT')),
    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20, 8) NOT NULL CHECK (price > 0),
    executed_price DECIMAL(20, 8),
    executed_quantity DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED')),
    exchange_order_id VARCHAR(100),
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Metadata
    strategy_version VARCHAR(50),
    model_version VARCHAR(50),
    dry_run BOOLEAN DEFAULT false,

    -- Indexes
    CONSTRAINT trades_symbol_created_idx UNIQUE (symbol, created_at)
);

CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_status ON trading.trades(status);
CREATE INDEX idx_trades_created_at ON trading.trades(created_at DESC);
CREATE INDEX idx_trades_executed_at ON trading.trades(executed_at DESC) WHERE executed_at IS NOT NULL;

-- Signals table: Model predictions and trading signals
CREATE TABLE trading.signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    direction SMALLINT NOT NULL CHECK (direction IN (-1, 0, 1)),  -- -1: SHORT, 0: NEUTRAL, 1: LONG
    probability DECIMAL(5, 4) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),

    -- Signal metadata
    model_type VARCHAR(50) NOT NULL,  -- 'lstm', 'transformer', 'ensemble'
    model_version VARCHAR(50) NOT NULL,
    feature_version VARCHAR(50),

    -- Market context at signal time
    market_price DECIMAL(20, 8) NOT NULL,
    volatility DECIMAL(10, 6),
    spread_bps DECIMAL(10, 2),

    -- Risk parameters
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    position_size_pct DECIMAL(5, 4),

    -- Signal processing
    filtered BOOLEAN DEFAULT false,
    filter_reason VARCHAR(255),
    executed BOOLEAN DEFAULT false,
    trade_id UUID REFERENCES trading.trades(id),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT signals_symbol_created_idx UNIQUE (symbol, created_at)
);

CREATE INDEX idx_signals_symbol ON trading.signals(symbol);
CREATE INDEX idx_signals_direction ON trading.signals(direction) WHERE direction != 0;
CREATE INDEX idx_signals_created_at ON trading.signals(created_at DESC);
CREATE INDEX idx_signals_executed ON trading.signals(executed) WHERE executed = false;
CREATE INDEX idx_signals_model_version ON trading.signals(model_version);

-- Positions table: Current and historical positions
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
    entry_price DECIMAL(20, 8) NOT NULL CHECK (entry_price > 0),
    current_price DECIMAL(20, 8),

    -- P&L tracking
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,

    -- Risk management
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    max_drawdown_pct DECIMAL(5, 4),

    -- Position status
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'STOPPED', 'LIQUIDATED')),

    -- Related trades
    entry_trade_id UUID REFERENCES trading.trades(id),
    exit_trade_id UUID REFERENCES trading.trades(id),

    -- Timestamps
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Metadata
    strategy_version VARCHAR(50),
    model_version VARCHAR(50)
);

CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_positions_status ON trading.positions(status) WHERE status = 'OPEN';
CREATE INDEX idx_positions_opened_at ON trading.positions(opened_at DESC);
CREATE INDEX idx_positions_closed_at ON trading.positions(closed_at DESC) WHERE closed_at IS NOT NULL;

-- Account state table: FTMO guardrails and account balance
CREATE TABLE trading.account_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_balance DECIMAL(20, 8) NOT NULL CHECK (account_balance >= 0),
    initial_balance DECIMAL(20, 8) NOT NULL,

    -- Daily tracking
    daily_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    daily_trades INTEGER NOT NULL DEFAULT 0,
    daily_volume DECIMAL(20, 8) NOT NULL DEFAULT 0,

    -- FTMO limits
    max_daily_loss_pct DECIMAL(5, 4) NOT NULL DEFAULT 0.05,  -- 5%
    max_total_loss_pct DECIMAL(5, 4) NOT NULL DEFAULT 0.10,  -- 10%
    daily_loss_limit DECIMAL(20, 8),
    total_loss_limit DECIMAL(20, 8),

    -- Guardrail status
    guardrails_active BOOLEAN DEFAULT true,
    trading_enabled BOOLEAN DEFAULT true,
    last_reset_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_account_state_updated_at ON trading.account_state(updated_at DESC);

-- =====================================================
-- ML SCHEMA: Model registry and metadata
-- =====================================================

-- Models table: Trained model metadata
CREATE TABLE ml.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'lstm', 'transformer'
    version VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Model architecture
    input_size INTEGER NOT NULL,
    hidden_size INTEGER,
    num_layers INTEGER,
    architecture_params JSONB,

    -- Training metadata
    training_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    epochs_trained INTEGER,

    -- Performance metrics
    train_accuracy DECIMAL(5, 4),
    val_accuracy DECIMAL(5, 4),
    test_accuracy DECIMAL(5, 4),
    calibration_error DECIMAL(5, 4),

    -- Backtest results
    backtest_sharpe DECIMAL(10, 6),
    backtest_win_rate DECIMAL(5, 4),
    backtest_max_drawdown DECIMAL(5, 4),

    -- Deployment
    status VARCHAR(20) NOT NULL DEFAULT 'TRAINING' CHECK (status IN ('TRAINING', 'VALIDATING', 'TESTED', 'PRODUCTION', 'ARCHIVED')),
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,

    -- Storage
    s3_uri VARCHAR(255) NOT NULL,
    mlflow_run_id VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT models_name_version_unique UNIQUE (model_name, version)
);

CREATE INDEX idx_models_symbol ON ml.models(symbol);
CREATE INDEX idx_models_status ON ml.models(status);
CREATE INDEX idx_models_created_at ON ml.models(created_at DESC);
CREATE INDEX idx_models_test_accuracy ON ml.models(test_accuracy DESC) WHERE test_accuracy IS NOT NULL;

-- Feature importance table
CREATE TABLE ml.feature_importance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml.models(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    importance_score DECIMAL(10, 6) NOT NULL,
    rank INTEGER,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT feature_importance_model_feature_unique UNIQUE (model_id, feature_name)
);

CREATE INDEX idx_feature_importance_model_id ON ml.feature_importance(model_id);
CREATE INDEX idx_feature_importance_rank ON ml.feature_importance(model_id, rank);

-- Training runs table
CREATE TABLE ml.training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml.models(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,

    -- Training configuration
    hyperparameters JSONB NOT NULL,
    data_start_date DATE NOT NULL,
    data_end_date DATE NOT NULL,

    -- Resource usage
    training_duration_seconds INTEGER,
    gpu_type VARCHAR(50),
    training_cost_usd DECIMAL(10, 4),

    -- Results
    final_loss DECIMAL(10, 6),
    best_val_accuracy DECIMAL(5, 4),
    early_stopped BOOLEAN DEFAULT false,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'STARTED' CHECK (status IN ('STARTED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    error_message TEXT,

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- MLflow tracking
    mlflow_experiment_id VARCHAR(100),
    mlflow_run_id VARCHAR(100)
);

CREATE INDEX idx_training_runs_symbol ON ml.training_runs(symbol);
CREATE INDEX idx_training_runs_started_at ON ml.training_runs(started_at DESC);
CREATE INDEX idx_training_runs_status ON ml.training_runs(status);

-- =====================================================
-- METRICS SCHEMA: Performance tracking
-- =====================================================

-- Daily metrics aggregation
CREATE TABLE metrics.daily_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_date DATE NOT NULL,
    symbol VARCHAR(20),

    -- Trading metrics
    trades_count INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    total_volume DECIMAL(20, 8) DEFAULT 0,

    -- Performance metrics
    win_rate DECIMAL(5, 4),
    sharpe_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(5, 4),

    -- Model metrics
    signals_generated INTEGER DEFAULT 0,
    signals_executed INTEGER DEFAULT 0,
    signals_filtered INTEGER DEFAULT 0,
    avg_confidence DECIMAL(5, 4),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT daily_metrics_date_symbol_unique UNIQUE (metric_date, COALESCE(symbol, 'ALL'))
);

CREATE INDEX idx_daily_metrics_date ON metrics.daily_metrics(metric_date DESC);
CREATE INDEX idx_daily_metrics_symbol ON metrics.daily_metrics(symbol) WHERE symbol IS NOT NULL;

-- System health metrics
CREATE TABLE metrics.system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component VARCHAR(50) NOT NULL,  -- 'kafka', 'redis', 'rds', 'model_inference', etc.
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 6) NOT NULL,
    metric_unit VARCHAR(20),

    -- Thresholds
    warning_threshold DECIMAL(20, 6),
    critical_threshold DECIMAL(20, 6),
    status VARCHAR(20) CHECK (status IN ('OK', 'WARNING', 'CRITICAL')),

    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT system_health_component_metric_ts UNIQUE (component, metric_name, timestamp)
);

CREATE INDEX idx_system_health_component ON metrics.system_health(component);
CREATE INDEX idx_system_health_timestamp ON metrics.system_health(timestamp DESC);
CREATE INDEX idx_system_health_status ON metrics.system_health(status) WHERE status != 'OK';

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to all tables with updated_at
CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trading.trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_signals_updated_at BEFORE UPDATE ON trading.signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_account_state_updated_at BEFORE UPDATE ON trading.account_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON ml.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Calculate P&L on position update
CREATE OR REPLACE FUNCTION calculate_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.current_price IS NOT NULL THEN
        IF NEW.side = 'LONG' THEN
            NEW.unrealized_pnl = (NEW.current_price - NEW.entry_price) * NEW.quantity;
        ELSE  -- SHORT
            NEW.unrealized_pnl = (NEW.entry_price - NEW.current_price) * NEW.quantity;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_positions_pnl BEFORE INSERT OR UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION calculate_position_pnl();

-- =====================================================
-- VIEWS
-- =====================================================

-- Active positions view
CREATE VIEW trading.active_positions AS
SELECT
    p.*,
    t.symbol AS trade_symbol,
    t.executed_price AS entry_executed_price,
    ROUND((p.unrealized_pnl / (p.entry_price * p.quantity) * 100)::numeric, 2) AS unrealized_pnl_pct
FROM trading.positions p
LEFT JOIN trading.trades t ON p.entry_trade_id = t.id
WHERE p.status = 'OPEN';

-- Recent signals view
CREATE VIEW trading.recent_signals AS
SELECT
    s.*,
    m.model_name,
    m.test_accuracy AS model_accuracy,
    t.status AS trade_status,
    t.executed_price
FROM trading.signals s
LEFT JOIN ml.models m ON s.model_version = m.version AND s.symbol = m.symbol
LEFT JOIN trading.trades t ON s.trade_id = t.id
WHERE s.created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY s.created_at DESC;

-- Production models view
CREATE VIEW ml.production_models AS
SELECT
    m.*,
    COUNT(DISTINCT s.id) AS signals_generated,
    AVG(s.confidence) AS avg_signal_confidence,
    COUNT(DISTINCT CASE WHEN s.executed = true THEN s.id END) AS signals_executed
FROM ml.models m
LEFT JOIN trading.signals s ON m.version = s.model_version AND m.symbol = s.symbol
WHERE m.status = 'PRODUCTION'
GROUP BY m.id;

-- Trading performance view
CREATE VIEW metrics.trading_performance AS
SELECT
    symbol,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE unrealized_pnl > 0) AS winning_trades,
    COUNT(*) FILTER (WHERE unrealized_pnl < 0) AS losing_trades,
    ROUND((COUNT(*) FILTER (WHERE unrealized_pnl > 0)::numeric / COUNT(*)::numeric * 100)::numeric, 2) AS win_rate_pct,
    SUM(unrealized_pnl + realized_pnl) AS total_pnl,
    AVG(unrealized_pnl + realized_pnl) AS avg_pnl_per_trade,
    MAX(unrealized_pnl + realized_pnl) AS best_trade,
    MIN(unrealized_pnl + realized_pnl) AS worst_trade
FROM trading.positions
WHERE status = 'CLOSED'
GROUP BY symbol;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert initial account state
INSERT INTO trading.account_state (
    account_balance,
    initial_balance,
    max_daily_loss_pct,
    max_total_loss_pct,
    daily_loss_limit,
    total_loss_limit
) VALUES (
    100000.00,  -- $100k initial balance
    100000.00,
    0.05,  -- 5% max daily loss
    0.10,  -- 10% max total loss
    5000.00,  -- $5k daily loss limit
    10000.00  -- $10k total loss limit
);

-- Grant permissions (adjust based on your security requirements)
GRANT USAGE ON SCHEMA trading TO crpbot_admin;
GRANT USAGE ON SCHEMA ml TO crpbot_admin;
GRANT USAGE ON SCHEMA metrics TO crpbot_admin;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO crpbot_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO crpbot_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO crpbot_admin;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO crpbot_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml TO crpbot_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO crpbot_admin;

-- Create read-only user for analytics
-- CREATE USER crpbot_readonly WITH PASSWORD 'your_readonly_password';
-- GRANT USAGE ON SCHEMA trading, ml, metrics TO crpbot_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA trading, ml, metrics TO crpbot_readonly;

COMMENT ON SCHEMA trading IS 'Operational trading data: trades, signals, positions, account state';
COMMENT ON SCHEMA ml IS 'Machine learning models, training runs, feature importance';
COMMENT ON SCHEMA metrics IS 'Performance metrics and system health monitoring';

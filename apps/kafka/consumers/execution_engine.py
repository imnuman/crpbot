"""Execution engine consumer for trade execution.

Consumes: signals.calibrated
Produces: trades.orders.pending

Applies FTMO guardrails, position sizing, and generates orders.
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from apps.kafka.config.topics import (
    SIGNALS_CALIBRATED,
    TRADES_ORDERS_PENDING,
    get_topic_name,
)
from apps.kafka.consumers.base import KafkaConsumerBase
from apps.kafka.producers.base import KafkaProducerBase


class ExecutionEngine:
    """Execution engine for trade execution with FTMO guardrails."""

    def __init__(
        self,
        symbols: list[str],
        bootstrap_servers: str = "localhost:9092",
        account_balance: float = 100000.0,  # $100k FTMO account
        max_daily_loss_pct: float = 0.05,  # 5% max daily loss
        max_total_loss_pct: float = 0.10,  # 10% max total loss
        position_size_pct: float = 0.02,  # 2% risk per trade
        min_confidence: float = 0.65,  # Minimum confidence to trade
        dry_run: bool = True,  # Dry run mode (no actual orders)
    ):
        """Initialize execution engine.

        Args:
            symbols: List of symbols to trade
            bootstrap_servers: Kafka broker addresses
            account_balance: Starting account balance
            max_daily_loss_pct: Maximum daily loss percentage
            max_total_loss_pct: Maximum total loss percentage
            position_size_pct: Position size as percentage of account
            min_confidence: Minimum signal confidence to execute
            dry_run: If True, only log orders without executing
        """
        self.symbols = symbols
        self.bootstrap_servers = bootstrap_servers
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_loss_pct = max_total_loss_pct
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence
        self.dry_run = dry_run

        # Initialize Kafka consumer and producer
        input_topics = [get_topic_name(SIGNALS_CALIBRATED)]

        self.consumer = KafkaConsumerBase(
            topics=input_topics,
            group_id="execution-engine",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
        )

        self.producer = KafkaProducerBase(bootstrap_servers=bootstrap_servers)

        # Track daily P&L (resets at UTC 00:00)
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()

        # Track open positions
        self.open_positions: dict[str, dict] = {}

        # Track execution stats
        self.stats = {
            "signals_consumed": 0,
            "orders_generated": 0,
            "orders_rejected": 0,
            "total_pnl": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
        }

        logger.info(
            f"Initialized ExecutionEngine: symbols={symbols}, "
            f"account=${account_balance:,.2f}, dry_run={dry_run}"
        )

    def start(self) -> None:
        """Start consuming signals and executing trades."""
        logger.info("Starting execution engine...")

        try:
            self.consumer.consume(
                message_handler=self._process_signal,
                poll_timeout=1.0,
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def _process_signal(self, message: dict, metadata: dict) -> None:
        """Process a trading signal and generate order.

        Args:
            message: Signal message from Kafka
            metadata: Message metadata
        """
        try:
            symbol = message.get("symbol")
            signal = message.get("signal")

            if not symbol or not signal:
                logger.warning("Invalid message format")
                return

            self.stats["signals_consumed"] += 1

            # Reset daily P&L if new day
            self._check_daily_reset()

            # Check FTMO guardrails
            if not self._check_guardrails():
                logger.warning("Guardrails violated, rejecting signal")
                self.stats["orders_rejected"] += 1
                return

            # Filter signal
            if signal.get("filtered", False):
                logger.debug(f"Signal for {symbol} filtered out (low confidence)")
                self.stats["orders_rejected"] += 1
                return

            # Check confidence threshold
            confidence = signal.get("confidence", 0)
            if confidence < self.min_confidence:
                logger.debug(
                    f"Signal for {symbol} below confidence threshold: {confidence:.2f} < {self.min_confidence}"
                )
                self.stats["orders_rejected"] += 1
                return

            # Check if we already have an open position for this symbol
            if symbol in self.open_positions:
                logger.debug(
                    f"Already have open position for {symbol}, skipping signal"
                )
                return

            # Generate order
            order = self._generate_order(symbol, signal)

            if order:
                # Produce order to Kafka
                self._produce_order(order)
                self.stats["orders_generated"] += 1

                # Track open position
                self.open_positions[symbol] = {
                    "order_id": order["order_id"],
                    "direction": signal["direction"],
                    "entry_price": order["entry_price"],
                    "size": order["quantity"],
                    "entry_time": datetime.utcnow(),
                }

                # Log progress
                if self.stats["orders_generated"] % 10 == 0:
                    logger.info(
                        f"Execution stats: {self.stats['orders_generated']} orders generated, "
                        f"{self.stats['orders_rejected']} rejected, "
                        f"Daily P&L: ${self.daily_pnl:,.2f}, "
                        f"Total P&L: ${self.stats['total_pnl']:,.2f}"
                    )

        except Exception as e:
            logger.error(f"Error processing signal: {e}")

    def _check_daily_reset(self) -> None:
        """Reset daily P&L at UTC 00:00."""
        current_date = datetime.utcnow().date()

        if current_date > self.last_reset_date:
            logger.info(
                f"New trading day. Resetting daily P&L. Yesterday: ${self.daily_pnl:,.2f}"
            )
            self.daily_pnl = 0.0
            self.last_reset_date = current_date

    def _check_guardrails(self) -> bool:
        """Check FTMO guardrails.

        Returns:
            True if safe to trade, False if guardrails violated
        """
        # Check daily loss limit
        daily_loss_limit = self.account_balance * self.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            logger.warning(
                f"Daily loss limit violated: ${self.daily_pnl:,.2f} < -${daily_loss_limit:,.2f}"
            )
            return False

        # Check total loss limit
        total_loss = self.account_balance - self.initial_balance
        total_loss_limit = self.initial_balance * self.max_total_loss_pct

        if total_loss < -total_loss_limit:
            logger.warning(
                f"Total loss limit violated: ${total_loss:,.2f} < -${total_loss_limit:,.2f}"
            )
            return False

        return True

    def _generate_order(self, symbol: str, signal: dict) -> Optional[dict]:
        """Generate order from signal.

        Args:
            symbol: Symbol to trade
            signal: Signal dictionary

        Returns:
            Order dictionary or None if order generation failed
        """
        try:
            direction = signal.get("direction")
            confidence = signal.get("confidence")

            if direction == 0:  # Neutral
                return None

            # Calculate position size (Kelly criterion simplified)
            risk_amount = self.account_balance * self.position_size_pct

            # Adjust position size based on confidence
            adjusted_risk = risk_amount * confidence

            # TODO: Get current market price from market data
            # For now, use a placeholder
            market_price = 50000.0  # Placeholder

            # Calculate quantity
            quantity = adjusted_risk / market_price

            # Generate order ID
            order_id = f"{symbol}_{datetime.utcnow().isoformat()}_{direction}"

            # Calculate stop loss and take profit
            # Using 2% stop loss and 3% take profit (1.5:1 R:R)
            stop_loss_pct = 0.02
            take_profit_pct = 0.03

            if direction == 1:  # Long
                side = "BUY"
                stop_loss = market_price * (1 - stop_loss_pct)
                take_profit = market_price * (1 + take_profit_pct)
            else:  # Short
                side = "SELL"
                stop_loss = market_price * (1 + stop_loss_pct)
                take_profit = market_price * (1 - take_profit_pct)

            order = {
                "order_id": order_id,
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "order_type": "MARKET",
                "quantity": float(quantity),
                "entry_price": float(market_price),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "confidence": float(confidence),
                "signal": signal,
                "dry_run": self.dry_run,
            }

            logger.info(
                f"Generated order: {symbol} {side} {quantity:.4f} @ ${market_price:.2f} "
                f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}, confidence: {confidence:.2f})"
            )

            return order

        except Exception as e:
            logger.error(f"Error generating order for {symbol}: {e}")
            return None

    def _produce_order(self, order: dict) -> None:
        """Produce order to Kafka.

        Args:
            order: Order dictionary
        """
        topic = get_topic_name(TRADES_ORDERS_PENDING)

        # Produce to Kafka
        self.producer.produce(
            topic=topic,
            value=order,
            key=order["symbol"],
            headers={
                "symbol": order["symbol"],
                "side": order["side"],
                "dry_run": "true" if self.dry_run else "false",
            },
        )

        logger.debug(f"Produced order to {topic}: {order['order_id']}")

    def stop(self) -> None:
        """Stop the execution engine."""
        logger.info(f"Stopping execution engine. Final stats: {self.stats}")
        logger.info(
            f"Final account balance: ${self.account_balance:,.2f} "
            f"(P&L: ${self.stats['total_pnl']:,.2f})"
        )
        self.consumer.close()
        self.producer.close()


def main():
    """Main entry point for execution engine."""
    # Configuration
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    bootstrap_servers = "localhost:9092"

    # FTMO account settings
    account_balance = 100000.0  # $100k
    max_daily_loss_pct = 0.05  # 5% max daily loss
    max_total_loss_pct = 0.10  # 10% max total loss
    position_size_pct = 0.02  # 2% risk per trade
    min_confidence = 0.65  # 65% minimum confidence

    # Create and start execution engine
    engine = ExecutionEngine(
        symbols=symbols,
        bootstrap_servers=bootstrap_servers,
        account_balance=account_balance,
        max_daily_loss_pct=max_daily_loss_pct,
        max_total_loss_pct=max_total_loss_pct,
        position_size_pct=position_size_pct,
        min_confidence=min_confidence,
        dry_run=True,  # Dry run mode
    )

    try:
        engine.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        engine.stop()


if __name__ == "__main__":
    main()

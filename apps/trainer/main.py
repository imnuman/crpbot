"""Training entry point for LSTM, Transformer, GAN, and RL models."""

import argparse

from loguru import logger


def train_lstm(coin: str, epochs: int):
    """Train LSTM model for a specific coin."""
    from apps.trainer.train.train_lstm import train_lstm_for_coin

    # Convert coin symbol to format expected (e.g., BTC -> BTC-USD)
    symbol = coin if "-" in coin else f"{coin}-USD"

    train_lstm_for_coin(
        symbol=symbol,
        interval="1m",
        epochs=epochs,
        batch_size=32,
        sequence_length=60,
        horizon=15,
    )


def train_transformer(epochs: int):
    """Train Transformer model."""
    from apps.trainer.train.train_transformer import train_transformer_multi_coin

    train_transformer_multi_coin(
        symbols=["BTC-USD", "ETH-USD"],
        interval="1m",
        epochs=epochs,
        batch_size=16,
        sequence_length=100,
        horizon=15,
    )


def train_gan(epochs: int, max_ratio: float):
    """Train GAN for synthetic data generation."""
    logger.info(f"Training GAN for {epochs} epochs (max synth ratio {max_ratio}) (stub).")
    # TODO: Implement GAN training
    pass


def train_ppo(steps: int, exec_model: str):
    """Train PPO RL model."""
    logger.info(f"Training PPO for {steps} steps with exec model {exec_model} (stub).")
    # TODO: Implement PPO training
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--task", required=True, choices=["lstm", "transformer", "gan", "ppo"])
    parser.add_argument("--coin", default="BTC", help="Coin symbol (for LSTM)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--steps", type=int, default=1000, help="RL training steps")
    parser.add_argument(
        "--max_synth_ratio", type=float, default=0.2, help="Max synthetic data ratio"
    )
    parser.add_argument("--exec", dest="exec_model", default="ftmo", help="Execution model")

    args = parser.parse_args()

    if args.task == "lstm":
        train_lstm(args.coin, args.epochs)
    elif args.task == "transformer":
        train_transformer(args.epochs)
    elif args.task == "gan":
        train_gan(args.epochs, args.max_synth_ratio)
    elif args.task == "ppo":
        train_ppo(args.steps, args.exec_model)

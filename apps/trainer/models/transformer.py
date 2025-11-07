"""Transformer model for trend strength prediction."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerTrendModel(nn.Module):
    """
    Transformer model for predicting trend strength.

    Architecture:
    - Input: Sequence of feature vectors
    - Transformer encoder with self-attention
    - Output: Trend strength (continuous value 0-1)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Number of input features
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer (trend strength: 0-1)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            mask: Optional attention mask (for padding)

        Returns:
            Tensor of shape (batch_size, 1) with trend strength (0-1)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask if needed
        if mask is not None:
            # Convert mask to format expected by transformer (True for padding)
            src_key_padding_mask = ~mask  # (batch_size, seq_len)
        else:
            src_key_padding_mask = None

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Use the last time step for prediction
        x = x[-1]  # (batch_size, d_model)

        # Output layer
        output = self.output_layer(x)

        return output

    def predict_trend(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict trend strength given input sequence.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size) or (sequence_length, input_size)

        Returns:
            Tensor of predicted trend strengths (0-1)
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            trend_strength = self.forward(x)
        return trend_strength.squeeze()


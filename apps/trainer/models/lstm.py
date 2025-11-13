"""LSTM model for direction prediction (15-min horizon)."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class LSTMDirectionModel(nn.Module):
    """
    LSTM model for predicting price direction (up/down) with 15-minute horizon.

    Architecture:
    - Input: Sequence of feature vectors (one per time step)
    - LSTM layers: 2-3 layers with dropout
    - Output: Binary classification (up/down probability)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.35,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTM model with improved architecture.

        Args:
            input_size: Number of input features (31 for our feature set)
            hidden_size: LSTM hidden state size (128 for better capacity)
            num_layers: Number of LSTM layers (3 for deeper learning)
            dropout: Dropout probability (0.35 for better regularization)
            bidirectional: Whether to use bidirectional LSTM (True for better context)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output layer (binary classification: up/down)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Single output: probability of up
            nn.Sigmoid(),  # Output probability between 0 and 1
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            lengths: Optional tensor of actual sequence lengths for each sample in batch

        Returns:
            Tensor of shape (batch_size, 1) with probability of upward movement
        """
        # LSTM forward pass
        if lengths is not None:
            # Pack sequences for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(x_packed)
            # Use the last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]
        else:
            lstm_out, (hidden, _) = self.lstm(x)
            # Use the last hidden state
            if self.bidirectional:
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]

        # Fully connected layers
        output = self.fc(hidden)

        return output

    def predict_direction(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict direction (up/down) given input sequence.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size) or (sequence_length, input_size)
            threshold: Probability threshold for up prediction (default: 0.5)

        Returns:
            Tensor of predicted directions (1 for up, 0 for down)
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            probabilities = self.forward(x)
            predictions = (probabilities >= threshold).long()
        return predictions.squeeze()


"""LSTM and Transformer sequence models for streamflow prediction.

Both models accept inputs of shape (batch, seq_len, num_features) where
num_features = num_dynamic_forcings + num_static_attributes.  Static attributes
are tiled along the time dimension in the data pipeline (see data.py) before
being passed here, so the model sees them at every timestep.

Architecture notes
------------------
LSTM
  Input  → stacked LSTM layers → take last hidden state → Linear → scalar
  - Strengths : captures long-range temporal dependencies, handles irregular
                seasonality well in hydrology; well-studied for rainfall-runoff.
  - Weakness  : sequential computation limits parallelism; hidden state is a
                fixed-size bottleneck that may lose distant information.

Transformer
  Input → linear embedding → positional encoding → Transformer encoder layers
        → mean-pool over time → Linear → scalar
  - Strengths : parallel training; explicit attention over the full context
                window; no vanishing-gradient issue.
  - Weakness  : quadratic memory w.r.t. seq_len; requires more data to train
                well compared to LSTM; positional encoding may not capture
                hydrologic seasonality as naturally as recurrence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# LSTM model
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Multi-layer LSTM for one-step-ahead streamflow prediction.

    Parameters
    ----------
    input_size   : number of input features per timestep
    hidden_size  : LSTM hidden state size
    num_layers   : number of stacked LSTM layers
    dropout      : dropout probability applied between LSTM layers (0 to disable)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, 1)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden  = lstm_out[:, -1, :]   # take last timestep
        last_hidden  = self.dropout(last_hidden)
        out = self.fc(last_hidden)           # (batch, 1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Transformer model
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for one-step-ahead streamflow prediction.

    Parameters
    ----------
    input_size  : number of input features per timestep
    d_model     : internal embedding dimension (must be divisible by nhead)
    nhead       : number of attention heads
    num_layers  : number of TransformerEncoderLayer blocks
    dim_ff      : feedforward hidden dimension inside each encoder layer
    dropout     : dropout probability
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, 1)
        """
        x = self.input_proj(x)              # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)             # (batch, seq_len, d_model)
        # Mean-pool over the time dimension
        x = x.mean(dim=1)                   # (batch, d_model)
        x = self.dropout(x)
        out = self.fc(x)                    # (batch, 1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    num_features: int,
    **kwargs,
) -> nn.Module:
    """Instantiate a model by name.

    Parameters
    ----------
    model_type  : 'lstm' or 'transformer'
    num_features: total input features (dynamic + static)
    **kwargs    : forwarded to the model constructor
    """
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMModel(input_size=num_features, **kwargs)
    elif model_type in ("transformer", "tf"):
        return TransformerModel(input_size=num_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'lstm' or 'transformer'.")

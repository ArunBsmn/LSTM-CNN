"""Model architectures for the LSTM-CNN imagined speech framework.

Four variants are provided for the component comparison study described
in Chapter 4.  All variants share identical building blocks and an
identical MLP classification head, ensuring that performance differences
reflect the feature extractor ordering rather than the classifier.

Classes
-------
LSTMBlock   — stacked LSTM layers with per-layer ReLU activation.
CNNBlock    — stacked 1-D Conv → ReLU → MaxPool → BatchNorm blocks.
MLPHead     — shared flatten + fully-connected classification head.
LSTMCNN     — LSTM first, CNN second (proposed architecture).
CNNLSTM     — CNN first, LSTM second (ablation).
LSTMOnly    — LSTM feature extractor only (ablation).
CNNOnly     — CNN feature extractor only (ablation).

Default hyperparameters match the published study.  Any parameter can
be overridden by passing a partial config dict; unspecified keys fall
back to the defaults defined in each class.

Input convention
----------------
All wrappers expect tensors of shape (batch, 1, samples), i.e. a
single-channel time series produced by core_dataset.make_channelwise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

# ── Shared default hyperparameters ───────────────────────────────────────────

_DEFAULT_LSTM_DIMS:     List[int] = [32, 64, 64, 32, 16]
_DEFAULT_CNN_CHANNELS:  List[int] = [32, 64]
_DEFAULT_CNN_KERNELS:   List[int] = [3, 3]
_DEFAULT_FC_DIMS:       List[int] = []          # no hidden FC layer by default
_DEFAULT_POOL_KERNEL:   int       = 2
_DEFAULT_POOL_STRIDE:   int       = 2
_DEFAULT_BIDIRECTIONAL: bool      = False
_DEFAULT_DROPOUT:       float     = 0.0         # no dropout (no overfitting observed)


def _cfg(user: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Return user[key] if present and non-None/non-empty, else default."""
    if user is None:
        return default
    val = user.get(key)
    if val is None or (isinstance(val, list) and len(val) == 0):
        return default
    return val


# ── Shared building blocks ────────────────────────────────────────────────────

class LSTMBlock(nn.Module):
    """Stacked LSTM layers with ReLU activation after each layer.

    Processes input of shape (batch, seq_len, input_dim) and returns
    output of shape (batch, seq_len, lstm_dims[-1]).

    Args:
        input_dim:     Number of input features at each time step.
        lstm_dims:     Hidden sizes for each LSTM layer.
        bidirectional: If True, use bidirectional LSTM.
        dropout:       Dropout between intermediate LSTM layers.
    """

    def __init__(
        self,
        input_dim:     int,
        lstm_dims:     List[int],
        bidirectional: bool  = _DEFAULT_BIDIRECTIONAL,
        dropout:       float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.activation    = nn.ReLU()
        self.bidirectional = bidirectional

        sizes = [input_dim] + lstm_dims
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size  = sizes[i],
                hidden_size = sizes[i + 1],
                batch_first = True,
                bidirectional = bidirectional,
                dropout     = dropout if i < len(sizes) - 2 else 0.0,
            )
            for i in range(len(sizes) - 1)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, hidden_dim × num_directions)
        """
        for layer in self.layers:
            x, _ = layer(x)
            x    = self.activation(x)
        return x


class CNNBlock(nn.Module):
    """Stacked 1-D Conv → ReLU → MaxPool → BatchNorm blocks.

    Each block halves the temporal dimension via MaxPool(kernel=2, stride=2).
    Processes input of shape (batch, input_dim, seq_len) and returns
    output of shape (batch, cnn_channels[-1], seq_len // 2^n_layers).

    Args:
        input_dim:    Number of input channels.
        cnn_channels: Output channels for each convolutional block.
        kernel_sizes: Kernel width for each convolutional block.
        dropout:      Channel-wise dropout after BatchNorm.
    """

    def __init__(
        self,
        input_dim:    int,
        cnn_channels: List[int],
        kernel_sizes: List[int],
        dropout:      float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes), (
            "cnn_channels and kernel_sizes must have the same length."
        )
        in_ch  = input_dim
        blocks = []
        for out_ch, k in zip(cnn_channels, kernel_sizes):
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=_DEFAULT_POOL_KERNEL,
                             stride=_DEFAULT_POOL_STRIDE),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(dropout),
            ))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, input_dim, seq_len)

        Returns:
            (batch, cnn_channels[-1], seq_len // 2^n_blocks)
        """
        return self.blocks(x)


class MLPHead(nn.Module):
    """Flatten + fully-connected classification head.

    Shared across all four model variants.  A single linear layer
    (no hidden FC) is used by default; additional hidden layers can
    be added via *fc_dims*.

    Args:
        input_size:  Flattened feature dimensionality.
        num_classes: Number of output classes.
        fc_dims:     Hidden layer widths between input and output.
        dropout:     Dropout between hidden layers (ignored if fc_dims=[]).
    """

    def __init__(
        self,
        input_size:  int,
        num_classes: int,
        fc_dims:     List[int] = _DEFAULT_FC_DIMS,
        dropout:     float     = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        dims   = [input_size] + list(fc_dims) + [num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # not the final projection
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, input_size)

        Returns:
            (batch, num_classes)  — raw logits
        """
        return self.net(x)


# ── Model wrappers ────────────────────────────────────────────────────────────

class LSTMCNN(nn.Module):
    """Proposed architecture: LSTM layers → CNN blocks → MLP head.

    LSTM layers encode long-range temporal dependencies first; CNN blocks
    then refine localised patterns in the temporally enriched feature
    space.  This ordering is validated in the component comparison study.

    Args:
        num_classes:  Number of output classes.
        data_length:  Temporal length of the input (samples).
        input_dim:    Number of input channels (1 for channel-wise inputs).
        config:       Optional dict overriding any default hyperparameter.

    Config keys (all optional)
    --------------------------
    lstm_dims     list[int]   Hidden sizes per LSTM layer.
    cnn_channels  list[int]   Output channels per CNN block.
    cnn_kernels   list[int]   Kernel widths per CNN block.
    fc_dims       list[int]   Hidden sizes in the MLP head.
    bidirectional bool        Bidirectional LSTM flag.
    dropout       float       Dropout rate (applied in all sub-modules).
    """

    def __init__(
        self,
        num_classes: int,
        data_length: int,
        input_dim:   int                   = 1,
        config:      Optional[Dict] = None,
    ) -> None:
        super().__init__()

        lstm_dims    = _cfg(config, "lstm_dims",     _DEFAULT_LSTM_DIMS)
        cnn_channels = _cfg(config, "cnn_channels",  _DEFAULT_CNN_CHANNELS)
        cnn_kernels  = _cfg(config, "cnn_kernels",   _DEFAULT_CNN_KERNELS)
        fc_dims      = _cfg(config, "fc_dims",       _DEFAULT_FC_DIMS)
        bidi         = _cfg(config, "bidirectional", _DEFAULT_BIDIRECTIONAL)
        dropout      = _cfg(config, "dropout",       _DEFAULT_DROPOUT)

        lstm_out = lstm_dims[-1] * (2 if bidi else 1)

        self.lstm = LSTMBlock(input_dim, lstm_dims, bidi, dropout)
        self.cnn  = CNNBlock(lstm_out, cnn_channels, cnn_kernels, dropout)

        n_pool       = len(cnn_channels)
        flat_size    = cnn_channels[-1] * (data_length // (2 ** n_pool))
        self.head    = MLPHead(flat_size, num_classes, fc_dims, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, samples)

        Returns:
            (batch, num_classes)
        """
        x = x.permute(0, 2, 1)          # → (batch, samples, 1)
        x = self.lstm(x)                 # → (batch, samples, lstm_out)
        x = x.permute(0, 2, 1)          # → (batch, lstm_out, samples)
        x = self.cnn(x)                  # → (batch, cnn_ch, samples//2^n)
        x = torch.flatten(x, 1)
        return self.head(x)


class CNNLSTM(nn.Module):
    """Ablation: CNN blocks → LSTM layers → MLP head.

    Reversed component ordering used in the comparison study to confirm
    that premature spatial processing disrupts temporal structure.

    Args:
        num_classes:  Number of output classes.
        data_length:  Temporal length of the input (samples).
        input_dim:    Number of input channels.
        config:       Optional dict overriding any default hyperparameter.
    """

    def __init__(
        self,
        num_classes: int,
        data_length: int,
        input_dim:   int                   = 1,
        config:      Optional[Dict] = None,
    ) -> None:
        super().__init__()

        lstm_dims    = _cfg(config, "lstm_dims",     _DEFAULT_LSTM_DIMS)
        cnn_channels = _cfg(config, "cnn_channels",  _DEFAULT_CNN_CHANNELS)
        cnn_kernels  = _cfg(config, "cnn_kernels",   _DEFAULT_CNN_KERNELS)
        fc_dims      = _cfg(config, "fc_dims",       _DEFAULT_FC_DIMS)
        bidi         = _cfg(config, "bidirectional", _DEFAULT_BIDIRECTIONAL)
        dropout      = _cfg(config, "dropout",       _DEFAULT_DROPOUT)

        n_pool       = len(cnn_channels)
        cnn_out_len  = data_length // (2 ** n_pool)
        lstm_out     = lstm_dims[-1] * (2 if bidi else 1)

        self.cnn  = CNNBlock(input_dim, cnn_channels, cnn_kernels, dropout)
        self.lstm = LSTMBlock(cnn_channels[-1], lstm_dims, bidi, dropout)

        flat_size = lstm_out * cnn_out_len
        self.head = MLPHead(flat_size, num_classes, fc_dims, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, samples)

        Returns:
            (batch, num_classes)
        """
        x = self.cnn(x)                  # → (batch, cnn_ch, samples//2^n)
        x = x.permute(0, 2, 1)          # → (batch, samples//2^n, cnn_ch)
        x = self.lstm(x)                 # → (batch, samples//2^n, lstm_out)
        x = torch.flatten(x, 1)
        return self.head(x)


class LSTMOnly(nn.Module):
    """Ablation: LSTM layers only → MLP head.

    Args:
        num_classes:  Number of output classes.
        data_length:  Temporal length of the input (samples).
        input_dim:    Number of input channels.
        config:       Optional dict overriding any default hyperparameter.
    """

    def __init__(
        self,
        num_classes: int,
        data_length: int,
        input_dim:   int                   = 1,
        config:      Optional[Dict] = None,
    ) -> None:
        super().__init__()

        lstm_dims = _cfg(config, "lstm_dims",     _DEFAULT_LSTM_DIMS)
        fc_dims   = _cfg(config, "fc_dims",       _DEFAULT_FC_DIMS)
        bidi      = _cfg(config, "bidirectional", _DEFAULT_BIDIRECTIONAL)
        dropout   = _cfg(config, "dropout",       _DEFAULT_DROPOUT)

        lstm_out  = lstm_dims[-1] * (2 if bidi else 1)
        flat_size = lstm_out * data_length

        self.lstm = LSTMBlock(input_dim, lstm_dims, bidi, dropout)
        self.head = MLPHead(flat_size, num_classes, fc_dims, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, samples)

        Returns:
            (batch, num_classes)
        """
        x = x.permute(0, 2, 1)          # → (batch, samples, 1)
        x = self.lstm(x)                 # → (batch, samples, lstm_out)
        x = torch.flatten(x, 1)
        return self.head(x)


class CNNOnly(nn.Module):
    """Ablation: CNN blocks only → MLP head.

    Args:
        num_classes:  Number of output classes.
        data_length:  Temporal length of the input (samples).
        input_dim:    Number of input channels.
        config:       Optional dict overriding any default hyperparameter.
    """

    def __init__(
        self,
        num_classes: int,
        data_length: int,
        input_dim:   int                   = 1,
        config:      Optional[Dict] = None,
    ) -> None:
        super().__init__()

        cnn_channels = _cfg(config, "cnn_channels", _DEFAULT_CNN_CHANNELS)
        cnn_kernels  = _cfg(config, "cnn_kernels",  _DEFAULT_CNN_KERNELS)
        fc_dims      = _cfg(config, "fc_dims",      _DEFAULT_FC_DIMS)
        dropout      = _cfg(config, "dropout",      _DEFAULT_DROPOUT)

        n_pool    = len(cnn_channels)
        flat_size = cnn_channels[-1] * (data_length // (2 ** n_pool))

        self.cnn  = CNNBlock(input_dim, cnn_channels, cnn_kernels, dropout)
        self.head = MLPHead(flat_size, num_classes, fc_dims, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, samples)

        Returns:
            (batch, num_classes)
        """
        x = self.cnn(x)                  # → (batch, cnn_ch, samples//2^n)
        x = torch.flatten(x, 1)
        return self.head(x)


# ── Registry (used by driver scripts) ────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "LSTMCNN":  LSTMCNN,
    "CNNLSTM":  CNNLSTM,
    "LSTMOnly": LSTMOnly,
    "CNNOnly":  CNNOnly,
}
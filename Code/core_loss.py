"""Cross-entropy loss with optional class-weight support.

A thin wrapper around nn.CrossEntropyLoss that accepts pre-computed
class weights and exposes a consistent interface for the training loop
in core_train.py.

Public API
----------
WeightedCELoss   — cross-entropy with inverse-frequency class weights.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class WeightedCELoss(nn.Module):
    """Cross-entropy loss with optional inverse-frequency class weights.

    Wraps ``nn.CrossEntropyLoss`` and exposes the same forward signature.
    Weights are computed externally (see ``core_train._class_weights``) and
    passed at construction time so that the loss object is stateless during
    the training loop.

    Args:
        weight:     Optional float tensor of shape ``(n_classes,)`` with
                    per-class loss weights.  When None, all classes are
                    weighted equally.
        label_smoothing: Label smoothing factor in ``[0, 1)``.  Default 0.0
                    (no smoothing), matching the published study.

    Example::

        weights = compute_class_weights(train_loader, device)
        criterion = WeightedCELoss(weight=weights)
        loss = criterion(logits, targets)
    """

    def __init__(
        self,
        weight:          Optional[Tensor] = None,
        label_smoothing: float            = 0.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight          = weight,
            label_smoothing = label_smoothing,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits:  Raw model output of shape ``(batch, n_classes)``.
            targets: Integer ground-truth labels of shape ``(batch,)``.

        Returns:
            Scalar loss tensor.
        """
        return self.ce(logits, targets)

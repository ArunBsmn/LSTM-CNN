"""Grad-CAM implementation for the LSTM-CNN imagined speech framework.

Registers forward and backward hooks on a target convolutional layer to
capture feature maps and gradients, then combines them into a normalised
temporal heatmap following Selvaraju et al. (2017).

The implementation is model-agnostic: any nn.Module that contains at
least one Conv1d layer can be targeted.  By default the last Conv1d
layer is used, which corresponds to the highest-level spatial features
learned by the CNN block.

Public API
----------
GradCAM          — context manager that computes per-sample heatmaps.
batch_heatmaps   — convenience wrapper for DataLoader-level heatmap
                   generation across an entire dataset split.

References
----------
Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from
deep networks via gradient-based localisation. ICCV.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


# ── Hook storage ──────────────────────────────────────────────────────────────

class _GradCAMHooks:
    """Stores the forward activations and backward gradients for one layer."""

    def __init__(self) -> None:
        self.activations: Optional[Tensor] = None
        self.gradients:   Optional[Tensor] = None
        self._fwd_handle  = None
        self._bwd_handle  = None

    def register(self, layer: nn.Module) -> "_GradCAMHooks":
        self._fwd_handle = layer.register_forward_hook(self._save_activations)
        self._bwd_handle = layer.register_full_backward_hook(self._save_gradients)
        return self

    def remove(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _save_activations(
        self,
        _module: nn.Module,
        _input:  tuple,
        output:  Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module:   nn.Module,
        _grad_in:  tuple,
        grad_out:  tuple,
    ) -> None:
        self.gradients = grad_out[0].detach()


# ── Layer selection helper ─────────────────────────────────────────────────────

def _find_last_conv1d(model: nn.Module) -> nn.Module:
    """Return the last Conv1d layer found in a depth-first traversal.

    Args:
        model: Any nn.Module.

    Returns:
        The last Conv1d sub-module encountered.

    Raises:
        ValueError: If no Conv1d layer is found.
    """
    last: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last = module
    if last is None:
        raise ValueError("No Conv1d layer found in the model.")
    return last


# ── GradCAM context manager ───────────────────────────────────────────────────

class GradCAM:
    """Compute Grad-CAM temporal heatmaps for a trained LSTM-CNN model.

    Hooks are registered on *target_layer* (defaulting to the last
    Conv1d layer) for the duration of the context.  Each call to
    :meth:`compute` returns a normalised heatmap of shape
    ``(batch, seq_len)`` aligned with the original input length.

    Args:
        model:        Trained model in eval mode.
        target_layer: Conv1d module to hook.  If None, the last Conv1d
                      layer is selected automatically.
        device:       Compute device.

    Example::

        with GradCAM(model, device=device) as gcam:
            for x, y in loader:
                maps = gcam.compute(x.to(device), target_class=y.to(device))
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: Optional[nn.Module] = None,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.device = device or torch.device("cpu")
        self._layer = target_layer if target_layer is not None \
                      else _find_last_conv1d(model)
        self._hooks = _GradCAMHooks()

    def __enter__(self) -> "GradCAM":
        self._hooks.register(self._layer)
        self.model.eval()
        return self

    def __exit__(self, *_: object) -> None:
        self._hooks.remove()

    def compute(
        self,
        x:            Tensor,
        target_class: Optional[Tensor] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmaps for a batch of inputs.

        Args:
            x:            Input tensor of shape ``(batch, 1, samples)``,
                          already on *self.device*.
            target_class: Integer class indices of shape ``(batch,)``.
                          If None, the predicted class is used for each
                          sample.

        Returns:
            Float32 array of shape ``(batch, samples)`` with values in
            ``[0, 1]``, where 1 indicates maximum discriminative
            importance.  The heatmap is bilinearly interpolated from the
            CNN feature-map resolution back to the original sample length.
        """
        self.model.zero_grad()
        x = x.to(self.device).requires_grad_(False)

        logits: Tensor = self.model(x)           # (batch, n_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1)  # (batch,)
        target_class = target_class.to(self.device)

        # One-hot score for the target class.
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        score = (logits * one_hot).sum()
        score.backward()

        # activations: (batch, channels, feature_len)
        # gradients:   (batch, channels, feature_len)
        acts  = self._hooks.activations   # (B, C, L)
        grads = self._hooks.gradients     # (B, C, L)

        # Channel weights: global average pooling of gradients.
        weights = grads.mean(dim=2, keepdim=True)  # (B, C, 1)

        # Weighted combination + ReLU.
        cam = (weights * acts).sum(dim=1)          # (B, L)
        cam = torch.clamp(cam, min=0.0)

        # Upsample to original input length.
        orig_len = x.shape[2]
        cam = cam.unsqueeze(1)                     # (B, 1, L)
        cam = torch.nn.functional.interpolate(
            cam, size=orig_len, mode="linear", align_corners=False
        )
        cam = cam.squeeze(1)                       # (B, orig_len)

        # Normalise each sample independently to [0, 1].
        cam_np = cam.cpu().numpy().astype(np.float32)
        mins   = cam_np.min(axis=1, keepdims=True)
        maxs   = cam_np.max(axis=1, keepdims=True)
        denom  = np.where(maxs - mins > 0, maxs - mins, 1.0)
        cam_np = (cam_np - mins) / denom

        return cam_np


# ── DataLoader-level convenience wrapper ──────────────────────────────────────

def batch_heatmaps(
    model:        nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    target_layer: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmaps for every sample in *loader*.

    Processes the DataLoader in batches and concatenates results.

    Args:
        model:        Trained model.
        loader:       DataLoader yielding ``(x, y)`` pairs.
        device:       Compute device.
        target_layer: Conv1d layer to hook (last Conv1d if None).

    Returns:
        Tuple of three arrays:

        * ``heatmaps`` — shape ``(n_samples, seq_len)``, values in [0, 1].
        * ``signals``  — shape ``(n_samples, seq_len)``, raw input signals.
        * ``labels``   — shape ``(n_samples,)``, integer ground-truth labels.
    """
    all_maps:    List[np.ndarray] = []
    all_signals: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []

    with GradCAM(model, target_layer=target_layer, device=device) as gcam:
        for x, y in loader:
            maps = gcam.compute(x.to(device), target_class=y.to(device))
            all_maps.append(maps)
            all_signals.append(x.squeeze(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return (
        np.concatenate(all_maps,    axis=0),
        np.concatenate(all_signals, axis=0),
        np.concatenate(all_labels,  axis=0),
    )

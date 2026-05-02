"""Training, evaluation, and metric logging for the LSTM-CNN framework.

All functions operate in the subject-dependent paradigm described in
Chapter 4: a single model is trained per subject with a 70:15:15
stratified split, Adam optimiser, cross-entropy loss with class weights,
and early stopping (patience=5, tolerance=1e-3).

Public API
----------
train_model   — full training loop; saves best checkpoint and metrics.
evaluate      — accuracy + loss on any DataLoader (no gradient).
compute_metrics — weighted precision, recall, F1.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Internal helpers ──────────────────────────────────────────────────────────

def _class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    Args:
        loader: Training DataLoader.
        device: Target device for the weight tensor.

    Returns:
        Float tensor of shape (n_classes,).
    """
    all_targets: List[int] = []
    for _, targets in loader:
        all_targets.extend(targets.numpy().tolist())
    targets_arr = np.array(all_targets)
    classes, counts = np.unique(targets_arr, return_counts=True)
    weights = 1.0 / counts.astype(np.float32)
    weights /= weights.sum()
    weight_tensor = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    weight_tensor[classes] = torch.from_numpy(weights)
    return weight_tensor.to(device)


def _weighted_metrics(
    all_targets:     np.ndarray,
    all_predictions: np.ndarray,
) -> Tuple[float, float, float]:
    """Weighted precision, recall, and F1 from flattened prediction arrays.

    Args:
        all_targets:     Ground-truth integer labels.
        all_predictions: Predicted integer labels.

    Returns:
        (weighted_precision, weighted_recall, weighted_f1)
    """
    classes = np.unique(np.concatenate([all_targets, all_predictions]))
    total   = len(all_targets)
    wp = wr = wf = 0.0
    for cls in classes:
        tp  = int(np.sum((all_predictions == cls) & (all_targets == cls)))
        fp  = int(np.sum((all_predictions == cls) & (all_targets != cls)))
        fn  = int(np.sum((all_predictions != cls) & (all_targets == cls)))
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        w   = np.sum(all_targets == cls) / total
        wp += p * w
        wr += r * w
        wf += f1 * w
    return wp, wr, wf


def _save_metrics(metrics: Dict[str, List], path: str) -> None:
    """Append per-epoch metrics to a CSV file (creates file if absent).

    Args:
        metrics: Dict mapping column names to lists of per-epoch values.
        path:    Destination CSV path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header  = list(metrics.keys())
    rows    = list(zip(*metrics.values()))
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def _save_test_metrics(
    subject:   str,
    loss:      float,
    acc:       float,
    precision: float,
    recall:    float,
    f1:        float,
    path:      str,
) -> None:
    """Append final test metrics for one subject to a summary CSV.

    Args:
        subject:   Subject identifier string.
        loss:      Test loss.
        acc:       Test accuracy (fraction).
        precision: Weighted test precision.
        recall:    Weighted test recall.
        f1:        Weighted test F1.
        path:      Destination CSV path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("subject,loss,accuracy,precision,recall,f1\n")
        f.write(
            f"{subject},{loss:.6f},{acc * 100:.4f},"
            f"{precision:.6f},{recall:.6f},{f1:.6f}\n"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
) -> Tuple[float, float]:
    """Accuracy and average loss over a DataLoader without gradient.

    Args:
        model:     Trained model in eval mode after this call.
        loader:    DataLoader to evaluate.
        criterion: Loss function (must accept logits, targets).
        device:    Compute device.

    Returns:
        (accuracy_fraction, mean_loss_per_batch)
    """
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y     = x.to(device), y.to(device)
            logits   = model(x)
            loss     = criterion(logits, y)
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            total_loss += loss.item()
    return correct / total, total_loss / len(loader)


def compute_metrics(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Weighted precision, recall, and F1 over a DataLoader.

    Args:
        model:  Trained model.
        loader: DataLoader to evaluate.
        device: Compute device.

    Returns:
        (weighted_precision, weighted_recall, weighted_f1)
    """
    model.eval()
    all_preds:   List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
    return _weighted_metrics(np.array(all_targets), np.array(all_preds))


def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    num_epochs:   int         = 50,
    lr:           float       = 5e-4,
    patience:     int         = 5,
    min_delta:    float       = 1e-3,
    device:       Optional[torch.device] = None,
    subject:      str         = "subject",
    dataset:      str         = "dataset",
    model_path:   Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> dict:
    """Train one subject-dependent model with early stopping.

    Cross-entropy loss weights are computed from the training split.
    The best checkpoint (lowest validation loss) is restored before
    returning and optionally saved to *model_path*.

    Args:
        model:        Uninitialised or freshly constructed model on CPU.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        test_loader:  Test DataLoader.
        num_epochs:   Maximum training epochs.
        lr:           Adam learning rate.
        patience:     Early stopping patience in epochs.
        min_delta:    Minimum validation-loss improvement to reset patience.
        device:       Compute device (auto-detected if None).
        subject:      Subject identifier for logging and file names.
        dataset:      Dataset name for logging and file names.
        model_path:   If given, best weights are saved here as a .pth file.
        metrics_path: If given, per-epoch CSV and test summary CSV are saved
                      under this directory.

    Returns:
        Best model state_dict (can be reloaded with model.load_state_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    weights   = _class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    epoch_metrics: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc":  [], "val_acc":  [], "test_acc":  [],
        "train_prec": [], "val_prec": [], "test_prec": [],
        "train_rec":  [], "val_rec":  [], "test_rec":  [],
        "train_f1":   [], "val_f1":   [], "test_f1":   [],
    }

    for epoch in range(num_epochs):
        # ── Training pass ────────────────────────────────────────────────
        model.train()
        correct = total = 0
        running_loss = 0.0
        for x, y in train_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds    = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
            running_loss += loss.item()

        train_acc  = correct / total
        train_loss = running_loss / len(train_loader)
        tr_p, tr_r, tr_f1 = compute_metrics(model, train_loader, device)

        # ── Validation and test passes ───────────────────────────────────
        val_acc,  val_loss  = evaluate(model, val_loader,  criterion, device)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        vl_p, vl_r, vl_f1  = compute_metrics(model, val_loader,  device)
        ts_p, ts_r, ts_f1  = compute_metrics(model, test_loader, device)

        # ── Logging ──────────────────────────────────────────────────────
        print(
            f"[{dataset}|{subject}] Epoch {epoch:03d} — "
            f"Train {train_acc * 100:.2f}% / {train_loss:.4f}  "
            f"Val {val_acc * 100:.2f}% / {val_loss:.4f}  "
            f"Test {test_acc * 100:.2f}% / {test_loss:.4f}"
        )
        for key, val in zip(epoch_metrics, [
            train_loss, val_loss,  test_loss,
            train_acc,  val_acc,   test_acc,
            tr_p, vl_p, ts_p,
            tr_r, vl_r, ts_r,
            tr_f1, vl_f1, ts_f1,
        ]):
            epoch_metrics[key].append(val)

        # ── Early stopping and checkpoint ────────────────────────────────
        if val_loss < best_val_loss - min_delta:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Restore best weights ─────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Persist artefacts ────────────────────────────────────────────────
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_state, model_path)

    if metrics_path is not None:
        tr_csv = os.path.join(metrics_path, f"{dataset}_{subject}_train.csv")
        ts_csv = os.path.join(metrics_path, "test_summary.csv")
        _save_metrics(epoch_metrics, tr_csv)
        # Final test metrics from the best-restored model
        final_acc, final_loss = evaluate(model, test_loader, criterion, device)
        final_p, final_r, final_f1 = compute_metrics(model, test_loader, device)
        _save_test_metrics(
            subject, final_loss, final_acc,
            final_p, final_r, final_f1, ts_csv,
        )

    return best_state
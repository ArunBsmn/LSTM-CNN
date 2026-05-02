"""LSTM-CNN training pipeline — KARAOne dataset.

Configure DATA_PATH and RESULTS_ROOT below, then run via main_driver.py.
All hyperparameters are defined in DATA_CFG, MODEL_CFG, and TRAIN_CFG;
no external config files are required.

Runs one subject at a time in subject-dependent mode: a separate model
is trained for each (subject, session) pair.  KARAOne provides five
binary tasks (B1--B5) and one 11-class multi-class (MC) task; the task
is selected by setting MODEL_CFG["task"] in the driver.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch

from core_dataset  import make_channelwise, make_loaders, preprocess_trials
from core_loaders  import load_karaone
from core_model    import MODEL_REGISTRY
from core_train    import train_model
from core_utils    import Timer, save_metadata, save_model_summary, set_all_seeds

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH    = "/path/to/KARAOne"   # root directory for KARAOne CSVs
RESULTS_ROOT = "/path/to/results/karaone"

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CFG: Dict[str, Any] = {
    "subjects":   ["MM05", "MM08", "MM09", "MM10", "MM11",
                   "MM12", "MM14", "MM15", "MM16", "MM18",
                   "MM19", "MM20", "MM21", "P02"],
    "td":         5,        # trial duration (seconds)
    "task":       "MC",     # "B1"–"B5" or "MC" (set by driver)
    "load_rest":  False,    # set True to enable signal analysis
}

MODEL_CFG: Dict[str, Any] = {
    "arch":         "LSTMCNN",   # key into MODEL_REGISTRY
    "lstm_dims":    [32, 64, 64, 32, 16],
    "cnn_channels": [32, 64],
    "cnn_kernels":  [3, 3],
    "fc_dims":      [],
    "bidirectional": False,
    "dropout":      0.0,
}

TRAIN_CFG: Dict[str, Any] = {
    "batch_size": 64,
    "lr":         5e-4,
    "num_epochs": 50,
    "patience":   5,
    "min_delta":  1e-3,
    "seed":       37,
}

# ── Task label mapping (KARAOne) ───────────────────────────────────────────────

# KARAOne raw labels are phoneme/word integers; re-map to 0-based class indices.
_TASK_CLASSES: Dict[str, List[int]] = {
    "B1": [1, 2],   "B2": [3, 4],   "B3": [5, 6],
    "B4": [7, 8],   "B5": [9, 10],
    "MC": list(range(1, 12)),
}


def _filter_task(
    data:    "np.ndarray",    # type: ignore[name-defined]
    targets: "np.ndarray",    # type: ignore[name-defined]
    task:    str,
) -> tuple:
    import numpy as np
    classes   = _TASK_CLASSES[task]
    mask      = np.isin(targets, classes)
    data      = data[mask]
    targets   = targets[mask]
    # Re-map to 0-based indices.
    label_map = {c: i for i, c in enumerate(classes)}
    targets   = np.vectorize(label_map.__getitem__)(targets)
    return data, targets


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(cfg_override: Dict[str, Any] | None = None) -> None:
    """Run the KARAOne training pipeline.

    Args:
        cfg_override: Optional dict that overrides keys in DATA_CFG,
                      MODEL_CFG, or TRAIN_CFG.  Used by the driver script.
    """
    import numpy as np

    # Apply any driver overrides.
    dcfg = {**DATA_CFG,  **(cfg_override or {}).get("data",  {})}
    mcfg = {**MODEL_CFG, **(cfg_override or {}).get("model", {})}
    tcfg = {**TRAIN_CFG, **(cfg_override or {}).get("train", {})}

    set_all_seeds(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch   = MODEL_REGISTRY[mcfg["arch"]]
    task   = dcfg["task"]

    for subject in dcfg["subjects"]:
        print(f"\n{'='*60}")
        print(f"KARAOne | {task} | {subject}")
        print(f"{'='*60}")

        with Timer() as t:
            # Load and preprocess.
            raw_data, targets = load_karaone(
                subject   = subject,
                data_path = DATA_PATH,
                td        = dcfg["td"],
                load_rest = False,
            )
            data_clean, targets_clean = preprocess_trials(
                raw_data, targets, td=dcfg["td"]
            )
            data_task, targets_task = _filter_task(
                data_clean, targets_clean, task
            )
            windows, labels = make_channelwise(data_task, targets_task)

            n_classes   = len(np.unique(labels))
            data_length = windows.shape[2]

            train_loader, val_loader, test_loader = make_loaders(
                windows, labels,
                batch_size = tcfg["batch_size"],
                seed       = tcfg["seed"],
            )

            # Build model.
            model = arch(
                num_classes = n_classes,
                data_length = data_length,
                config      = mcfg,
            )

            # Output paths.
            out_dir = os.path.join(RESULTS_ROOT, task, subject)
            os.makedirs(out_dir, exist_ok=True)
            model_path   = os.path.join(out_dir, "best_model.pth")
            metrics_path = out_dir
            summary_path = os.path.join(out_dir, "model_summary.txt")

            save_model_summary(model, (1, data_length), summary_path, device)

            # Train.
            train_model(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                test_loader  = test_loader,
                num_epochs   = tcfg["num_epochs"],
                lr           = tcfg["lr"],
                patience     = tcfg["patience"],
                min_delta    = tcfg["min_delta"],
                device       = device,
                subject      = subject,
                dataset      = f"KARAOne_{task}",
                model_path   = model_path,
                metrics_path = metrics_path,
            )

            # Save metadata.
            save_metadata(
                {
                    "dataset":    "KARAOne",
                    "task":       task,
                    "subject":    subject,
                    "arch":       mcfg["arch"],
                    "n_classes":  n_classes,
                    "data_length": data_length,
                    "seed":       tcfg["seed"],
                },
                os.path.join(out_dir, "metadata.txt"),
            )

        print(f"[{subject}] Done in {t.elapsed_str}")

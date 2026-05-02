"""LSTM-CNN training pipeline — ASU Speech Imagery dataset.

Configure DATA_PATH and RESULTS_ROOT below, then run via main_driver.py.
All hyperparameters are defined in DATA_CFG, MODEL_CFG, and TRAIN_CFG;
no external config files are required.

ASU is organised into four subtasks (N1--N4) covering long words, short
words, vowels, and mixed word pairs.  A separate model is trained per
subject in subject-dependent mode.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch

from core_dataset  import make_channelwise, make_loaders, preprocess_trials
from core_loaders  import load_asu
from core_model    import MODEL_REGISTRY
from core_train    import train_model
from core_utils    import Timer, save_metadata, save_model_summary, set_all_seeds

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH    = "/path/to/ASU"       # root directory for ASU CSVs
RESULTS_ROOT = "/path/to/results/asu"

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CFG: Dict[str, Any] = {
    "subjects":  [f"sub_{i}b" for i in range(1, 17)],
    "td":        5,
    "task":      "N4",     # "N1", "N2", "N3", or "N4" (set by driver)
    "load_rest": False,
}

MODEL_CFG: Dict[str, Any] = {
    "arch":          "LSTMCNN",
    "lstm_dims":     [32, 64, 64, 32, 16],
    "cnn_channels":  [32, 64],
    "cnn_kernels":   [3, 3],
    "fc_dims":       [],
    "bidirectional": False,
    "dropout":       0.0,
}

TRAIN_CFG: Dict[str, Any] = {
    "batch_size": 64,
    "lr":         5e-4,
    "num_epochs": 50,
    "patience":   5,
    "min_delta":  1e-3,
    "seed":       37,
}

# ── Task definitions (ASU channel/class counts) ────────────────────────────────

# ASU N1-N4 class counts (from dataset documentation).
_TASK_N_CLASSES: Dict[str, int] = {
    "N1": 2,   # long words
    "N2": 2,   # short words
    "N3": 3,   # vowels
    "N4": 2,   # mixed word pairs
}


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(cfg_override: Dict[str, Any] | None = None) -> None:
    """Run the ASU training pipeline.

    Args:
        cfg_override: Optional dict that overrides keys in DATA_CFG,
                      MODEL_CFG, or TRAIN_CFG.  Used by the driver script.
    """
    import numpy as np

    dcfg = {**DATA_CFG,  **(cfg_override or {}).get("data",  {})}
    mcfg = {**MODEL_CFG, **(cfg_override or {}).get("model", {})}
    tcfg = {**TRAIN_CFG, **(cfg_override or {}).get("train", {})}

    set_all_seeds(tcfg["seed"])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch       = MODEL_REGISTRY[mcfg["arch"]]
    task       = dcfg["task"]
    n_classes  = _TASK_N_CLASSES[task]

    for subject in dcfg["subjects"]:
        print(f"\n{'='*60}")
        print(f"ASU | {task} | {subject}")
        print(f"{'='*60}")

        with Timer() as t:
            raw_data, targets = load_asu(
                subject   = subject,
                data_path = os.path.join(DATA_PATH, task),
                td        = dcfg["td"],
                load_rest = False,
            )
            data_clean, targets_clean = preprocess_trials(
                raw_data, targets,
                n_channels  = raw_data.shape[1] // int(dcfg["td"] * 256),
                aux_indices = [],
                bad_index   = None,
                td          = dcfg["td"],
            )
            windows, labels = make_channelwise(data_clean, targets_clean)

            data_length = windows.shape[2]

            train_loader, val_loader, test_loader = make_loaders(
                windows, labels,
                batch_size = tcfg["batch_size"],
                seed       = tcfg["seed"],
            )

            model = arch(
                num_classes = n_classes,
                data_length = data_length,
                config      = mcfg,
            )

            out_dir = os.path.join(RESULTS_ROOT, task, subject)
            os.makedirs(out_dir, exist_ok=True)
            model_path   = os.path.join(out_dir, "best_model.pth")
            metrics_path = out_dir
            summary_path = os.path.join(out_dir, "model_summary.txt")

            save_model_summary(model, (1, data_length), summary_path, device)

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
                dataset      = f"ASU_{task}",
                model_path   = model_path,
                metrics_path = metrics_path,
            )

            save_metadata(
                {
                    "dataset":     "ASU",
                    "task":        task,
                    "subject":     subject,
                    "arch":        mcfg["arch"],
                    "n_classes":   n_classes,
                    "data_length": data_length,
                    "seed":        tcfg["seed"],
                },
                os.path.join(out_dir, "metadata.txt"),
            )

        print(f"[{subject}] Done in {t.elapsed_str}")

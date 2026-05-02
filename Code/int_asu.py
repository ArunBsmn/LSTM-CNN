"""Interpretation pipeline — ASU Speech Imagery dataset.

Generates Grad-CAM heatmaps, applies VBTM and VATM masking at the
optimal threshold, and writes per-subject interpretation outputs.
Requires trained weights saved by main_asu.py.

Configure WEIGHTS_ROOT, RESULTS_ROOT, and DATA_PATH below, then run
via int_driver.py.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import torch

from core_dataset  import make_channelwise, make_loaders, preprocess_trials
from core_gradcam  import batch_heatmaps
from core_loaders  import load_asu
from core_model    import MODEL_REGISTRY
from core_train    import evaluate
from core_utils    import set_all_seeds
from int_signal    import analyse_subject, ttest_iws

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH    = "/path/to/ASU"
WEIGHTS_ROOT = "/path/to/results/asu"
RESULTS_ROOT = "/path/to/results/asu/interpretation"

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CFG: Dict[str, Any] = {
    "subjects":  [f"sub_{i}b" for i in range(1, 17)],
    "td":        5,
    "task":      "N4",
    "load_rest": True,
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

INT_CFG: Dict[str, Any] = {
    "seed":           37,
    "batch_size":     64,
    "sfreq":          256,
    "n_thresh_steps": 100,
    "arch_variants":  ["LSTMCNN", "CNNLSTM", "LSTMOnly", "CNNOnly"],
}

_TASK_N_CLASSES: Dict[str, int] = {
    "N1": 2, "N2": 2, "N3": 3, "N4": 2,
}


def _apply_mask(signals, heatmaps, threshold, mode):
    masked = signals.copy()
    if mode == "vbtm":
        masked[heatmaps < threshold] = 0.0
    elif mode == "vatm":
        masked[heatmaps >= threshold] = 0.0
    return masked


def run_subject(
    subject: str,
    task:    str,
    device:  torch.device,
    cfg_override: Dict[str, Any] | None = None,
) -> None:
    dcfg = {**DATA_CFG,  **(cfg_override or {}).get("data",  {})}
    mcfg = {**MODEL_CFG, **(cfg_override or {}).get("model", {})}
    icfg = {**INT_CFG,   **(cfg_override or {}).get("int",   {})}

    set_all_seeds(icfg["seed"])

    raw_think, targets, raw_rest = load_asu(
        subject   = subject,
        data_path = os.path.join(DATA_PATH, task),
        td        = dcfg["td"],
        load_rest = True,
    )

    # ASU has no auxiliary channels or bad channels to strip.
    n_channels  = raw_think.shape[1] // int(dcfg["td"] * icfg["sfreq"])
    data_clean, targets_clean = preprocess_trials(
        raw_think, targets,
        n_channels  = n_channels,
        aux_indices = [],
        bad_index   = None,
        td          = dcfg["td"],
    )
    rest_clean, _ = preprocess_trials(
        raw_rest,
        np.zeros(raw_rest.shape[0], dtype=np.int64),
        n_channels  = n_channels,
        aux_indices = [],
        bad_index   = None,
        td          = dcfg["td"],
    )

    windows, labels = make_channelwise(data_clean, targets_clean)
    n_classes       = _TASK_N_CLASSES[task]
    data_length     = windows.shape[2]

    _, _, test_loader = make_loaders(
        windows, labels,
        batch_size = icfg["batch_size"],
        seed       = icfg["seed"],
    )

    arch  = MODEL_REGISTRY[mcfg["arch"]]
    model = arch(num_classes=n_classes, data_length=data_length,
                 config=mcfg).to(device)
    model_path = os.path.join(WEIGHTS_ROOT, task, subject, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Signal analysis.
    band_durs = analyse_subject(data_clean, rest_clean, sfreq=icfg["sfreq"])
    all_durs  = [d for durs in band_durs.values() for d in durs]
    t_stat, p_val = ttest_iws(all_durs, trial_duration=float(dcfg["td"]))
    iws_min_s = int(min(all_durs) * icfg["sfreq"]) if all_durs else 0
    iws_max_s = int(max(all_durs) * icfg["sfreq"]) if all_durs else data_length

    print(f"  [{subject}] IWS range: {iws_min_s}–{iws_max_s} samples  "
          f"(t={t_stat:.2f}, p={p_val:.2e})")

    heatmaps, signals, gt_labels = batch_heatmaps(model, test_loader, device)

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    criterion = nn.CrossEntropyLoss()

    thresholds  = np.linspace(0.0, 1.0, icfg["n_thresh_steps"])
    best_thresh = 0.0
    best_acc    = 0.0
    best_n_s    = data_length

    for thresh in thresholds:
        masked  = _apply_mask(signals, heatmaps, thresh, "vbtm")
        n_samps = int((heatmaps >= thresh).sum(axis=1).mean())
        if not (iws_min_s <= n_samps <= iws_max_s):
            continue
        x_m = torch.from_numpy(masked[:, np.newaxis, :]).float()
        y_m = torch.from_numpy(gt_labels).long()
        ml  = DataLoader(TensorDataset(x_m, y_m), batch_size=icfg["batch_size"])
        acc, _ = evaluate(model, ml, criterion, device)
        if acc > best_acc or (acc == best_acc and n_samps < best_n_s):
            best_acc    = acc
            best_thresh = thresh
            best_n_s    = n_samps

    out_dir = os.path.join(RESULTS_ROOT, task, subject)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "heatmaps.npy"), heatmaps)
    np.save(os.path.join(out_dir, "signals.npy"),  signals)
    np.save(os.path.join(out_dir, "labels.npy"),   gt_labels)

    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(f"optimal_threshold = {best_thresh:.6f}\n")
        f.write(f"vbtm_accuracy     = {best_acc * 100:.4f}\n")
        f.write(f"retained_samples  = {best_n_s}\n")
        f.write(f"t_statistic       = {t_stat:.4f}\n")
        f.write(f"p_value           = {p_val:.6e}\n")

    results_rows = []
    for variant in icfg["arch_variants"]:
        v_arch  = MODEL_REGISTRY[variant]
        v_model = v_arch(num_classes=n_classes, data_length=data_length,
                         config=mcfg).to(device)
        v_path  = os.path.join(WEIGHTS_ROOT, task, subject, f"{variant}_best.pth")
        if not os.path.exists(v_path):
            continue
        v_model.load_state_dict(torch.load(v_path, map_location=device))
        v_model.eval()
        for mode in ["vbtm", "vatm"]:
            masked = _apply_mask(signals, heatmaps, best_thresh, mode)
            x_m = torch.from_numpy(masked[:, np.newaxis, :]).float()
            y_m = torch.from_numpy(gt_labels).long()
            ml  = DataLoader(TensorDataset(x_m, y_m),
                             batch_size=icfg["batch_size"])
            acc, _ = evaluate(v_model, ml, criterion, device)
            results_rows.append(f"{variant},{mode},{acc * 100:.4f}")

    if results_rows:
        with open(os.path.join(out_dir, "masking_results.csv"), "w") as f:
            f.write("arch,mode,accuracy\n")
            f.write("\n".join(results_rows) + "\n")


def run(cfg_override: Dict[str, Any] | None = None) -> None:
    dcfg   = {**DATA_CFG, **(cfg_override or {}).get("data", {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task   = dcfg["task"]
    for subject in dcfg["subjects"]:
        run_subject(subject, task, device, cfg_override)

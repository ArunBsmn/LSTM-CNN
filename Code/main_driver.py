"""Entry point for LSTM-CNN main training experiments.

Configure RUN_CFG below to select the dataset, task, and subjects,
then run from inside the Code/ directory:

    cd Code
    python main_driver.py

All pipeline scripts (main_karaone.py, main_asu.py) are invoked via
their ``run()`` function; no command-line arguments are required.
"""
from __future__ import annotations

from typing import Any, Dict, List

import main_asu
import main_karaone

# ── Run configuration ──────────────────────────────────────────────────────────

RUN_CFG: Dict[str, Any] = {
    # Which dataset to run: "karaone" or "asu".
    "dataset": "karaone",

    # Task identifier.
    #   KARAOne: "B1", "B2", "B3", "B4", "B5", or "MC".
    #   ASU:     "N1", "N2", "N3", or "N4".
    "task": "MC",

    # Subset of subjects to run (None = all subjects in the pipeline).
    "subjects": None,

    # Model architecture: "LSTMCNN", "CNNLSTM", "LSTMOnly", or "CNNOnly".
    "arch": "LSTMCNN",
}

# ── Dispatch ───────────────────────────────────────────────────────────────────

def _build_override(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert RUN_CFG into override dicts for the pipeline ``run()`` call."""
    data_override: Dict[str, Any] = {"task": run_cfg["task"]}
    if run_cfg.get("subjects") is not None:
        data_override["subjects"] = run_cfg["subjects"]
    return {
        "data":  data_override,
        "model": {"arch": run_cfg["arch"]},
        "train": {},
    }


def main() -> None:
    override = _build_override(RUN_CFG)
    if RUN_CFG["dataset"].lower() == "karaone":
        main_karaone.run(override)
    elif RUN_CFG["dataset"].lower() == "asu":
        main_asu.run(override)
    else:
        raise ValueError(
            f"Unknown dataset '{RUN_CFG['dataset']}'. "
            "Choose 'karaone' or 'asu'."
        )


if __name__ == "__main__":
    main()

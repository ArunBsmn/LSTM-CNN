"""Entry point for interpretation, masking, and post-hoc analysis.

Configure RUN_CFG below, then run from inside the Code/ directory:

    cd Code
    python int_driver.py

Requires trained model weights saved by a main-experiment run.
Set WEIGHTS_ROOT and RESULTS_ROOT in int_karaone.py / int_asu.py
before running.  Signal analysis (int_signal.py) is invoked
automatically from within the interpretation pipelines.

Post-hoc statistical tests (int_posthoc.py) are run after interpretation
outputs are available.
"""
from __future__ import annotations

from typing import Any, Dict

import int_asu
import int_karaone
import int_posthoc

# ── Run configuration ──────────────────────────────────────────────────────────

RUN_CFG: Dict[str, Any] = {
    # Dataset: "karaone" or "asu".
    "dataset": "karaone",

    # Task identifier.
    "task": "MC",

    # Subset of subjects to interpret (None = all).
    "subjects": None,

    # Model architecture used during training.
    "arch": "LSTMCNN",

    # Whether to run post-hoc statistical tests after interpretation.
    "run_posthoc": True,
}

# ── Dispatch ───────────────────────────────────────────────────────────────────

def main() -> None:
    data_override: Dict[str, Any] = {"task": RUN_CFG["task"]}
    if RUN_CFG.get("subjects") is not None:
        data_override["subjects"] = RUN_CFG["subjects"]

    override = {
        "data":  data_override,
        "model": {"arch": RUN_CFG["arch"]},
    }

    if RUN_CFG["dataset"].lower() == "karaone":
        int_karaone.run(override)
    elif RUN_CFG["dataset"].lower() == "asu":
        int_asu.run(override)
    else:
        raise ValueError(
            f"Unknown dataset '{RUN_CFG['dataset']}'. "
            "Choose 'karaone' or 'asu'."
        )

    if RUN_CFG.get("run_posthoc", False):
        int_posthoc.run({"dataset": RUN_CFG["dataset"],
                         "task":    RUN_CFG["task"]})


if __name__ == "__main__":
    main()

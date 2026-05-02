# LSTM-CNN

**An explainable deep learning framework for temporal event localisation in imagined speech EEG.**

LSTM-CNN processes IS-EEG signals through stacked LSTM layers that encode long-range temporal dependencies, followed by CNN blocks that refine local spatial patterns in the enriched feature space. A shared MLP classification head is used across all model variants. Grad-CAM heatmaps generated from the trained model drive a threshold-based masking framework — VBTM and VATM — to isolate and validate the discriminative temporal regions corresponding to the imagined word segment.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Repository Structure

```bash
LSTM-CNN/
├── Code/
│   ├── core_dataset.py      # EEG preprocessing, channel-wise windowing, train/val/test splits
│   ├── core_loaders.py      # Dataset loaders for KARAOne and ASU datasets
│   ├── core_model.py        # LSTM-CNN, CNN-LSTM, LSTM, CNN variants + shared MLP head
│   ├── core_loss.py         # Cross-entropy loss with class-weight support
│   ├── core_train.py        # Training loop, evaluation, early stopping, metric logging
│   ├── core_utils.py        # Reproducibility, timing, metadata serialisation, model summary
│   ├── core_gradcam.py      # Grad-CAM implementation with gradient hooks on CNN layers
│   │
│   ├── main_karaone.py      # LSTM-CNN training pipeline — KARAOne dataset
│   ├── main_asu.py          # LSTM-CNN training pipeline — ASU dataset
│   ├── main_driver.py       # Entry point for main training runs
│   │
│   ├── int_signal.py        # Signal analysis: band decomposition, power envelopes, IWS duration
│   ├── int_karaone.py       # Interpretation pipeline — KARAOne dataset
│   ├── int_asu.py           # Interpretation pipeline — ASU dataset
│   ├── int_driver.py        # Entry point for interpretation and masking runs
│   │
│   └── int_posthoc.py       # Post-hoc statistics: Friedman, Nemenyi, Wilcoxon; box plots
│
├── CITATION.cff
├── requirements.txt
└── README.md
```

All scripts live under `Code/` and are intentionally flat within it — all imports resolve within the same directory with no relative-path dependencies. Run all driver scripts from inside `Code/`.

---

## Datasets

Two imagined-speech EEG benchmarks are supported.

**KARAOne**
Five binary classification tasks (B1--B5) and an 11-class multi-class (MC) task. Preprocessing consists of ICA-based artefact removal followed by downsampling to 256 Hz. Raw `.csv` files are expected under a single root directory; `core_loaders.py` resolves the session subfolder layout automatically. [KARAOne Dataset](https://doi.org/10.3389/fnins.2015.00090)

**ASU Speech Imagery Dataset**
Four subtasks (N1--N4) covering long words, short words, vowels, and mixed word pairs (2-class and 3-class). Preprocessing additionally includes a bandpass filter from 8--70 Hz and a notch filter at 60 Hz prior to ICA and downsampling. Raw `.csv` files are expected under a single root directory. [ASU Dataset](https://doi.org/10.1088/1741-2552/aa8235)

---

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

All imports use only the standard library plus the packages listed in `requirements.txt`. No additional installation steps are required.

---

## Usage

Each experiment group has a dedicated driver script. Configure the run by editing the relevant config dicts, then execute the driver from inside the `Code/` directory.

### Main experiments (LSTM-CNN training)

```bash
cd Code
python main_driver.py
```

Set `DATA_PATH` and `RESULTS_ROOT` at the top of `main_karaone.py` or `main_asu.py` before running.

### Interpretation, masking, and post-hoc analysis

```bash
python int_driver.py
```

Requires trained weights saved by a main-experiment run. Set `WEIGHTS_ROOT` and `RESULTS_ROOT` in `int_karaone.py` / `int_asu.py`. Signal analysis (`int_signal.py`) runs independently of trained weights and requires only the raw dataset path.

Interpretation outputs per subject:

| Output file | Description |
| --- | --- |
| `gradcam_*.svg` | Grad-CAM temporal heatmap |
| `masking_curve_*.svg` | Classification accuracy vs. threshold curve |
| `vbtm_results_*.csv` | VBTM masking accuracy across all model components |
| `vatm_results_*.csv` | VATM masking accuracy across all model components |
| `noise_heatmap_*.svg` | Grad-CAM heatmaps under controlled SNR degradation |
| `tsne_*.svg` | t-SNE embedding of model representations |
| `roc_*.svg` | ROC curves with AUC scores |

---

## Configuration

Each pipeline script (`main_karaone.py`, `main_asu.py`, `int_karaone.py`, `int_asu.py`) contains three self-contained config dicts — `DATA_CFG`, `MODEL_CFG`, and `TRAIN_CFG` — alongside a `DATA_PATH` / `RESULTS_ROOT` header block. No external config files are required; all hyperparameters are co-located with the pipeline that uses them.

Driver scripts expose a `RUN_CFG` dict for selecting dataset, task, and subjects without modifying the pipeline files.

Key hyperparameters (defaults match the published study):

| Parameter | Value |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 5 × 10⁻⁴ |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Early stopping tolerance | 1 × 10⁻³ |
| Train / val / test split | 70 : 15 : 15 |
| Sampling frequency | 256 Hz |

---

## Reproducibility

`core_utils.set_all_seeds` fixes seeds for Python, NumPy, and PyTorch (including `cudnn.deterministic`). All pipelines call this before data loading. The default seed across all experiments is `37`.

Normalisation statistics are always fitted on the training split only and applied to validation and test splits, recomputed per subject.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lstmcnn_iseeg2025,
  author    = {Arun Balasubramanian and Kartik Pandey and Gautam Veer and Debasis Samanta},
  title     = {Explainable artificial intelligence-based identification of the localized events in imagined speech electroencephalogram},
  journal   = {Computers and Electrical Engineering},
  volume    = {127},
  pages     = {110608},
  year      = {2025},
  doi       = {10.1016/j.compeleceng.2025.110608},
  publisher = {Elsevier}
}
```

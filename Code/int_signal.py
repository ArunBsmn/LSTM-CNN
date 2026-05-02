"""Signal analysis for estimating the imagined word segment (IWS) duration.

Implements the methodology described in Chapter 4, Section 3.2:

1. Decompose raw EEG into five canonical frequency bands (delta, theta,
   alpha, beta, gamma) using fourth-order Butterworth bandpass filters
   applied zero-phase (forward-backward).
2. Compute the power envelope of each band via the Hilbert transform.
3. Smooth the envelope with a Savitzky-Golay filter (100 ms window).
4. Normalise against resting-state statistics (z-score baseline).
5. Threshold the normalised envelope to detect contiguous active segments;
   discard segments shorter than 10 ms.
6. Record segment durations and report per-band mean IWS duration.
7. Run a one-sample t-test against the full trial duration to confirm
   temporal sparsity of the IWS.

This module runs independently of trained model weights and requires
only raw dataset paths.

Public API
----------
analyse_subject   — full pipeline for one subject; returns per-band durations.
analyse_dataset   — aggregate analysis across all subjects in a dataset.
ttest_iws         — one-sample t-test: H0 = mean duration == full trial.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats

# ── Frequency band definitions ────────────────────────────────────────────────

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 70.0),
}

_FILTER_ORDER:      int   = 4        # Butterworth filter order
_SG_WINDOW_MS:      int   = 100     # Savitzky-Golay window length (ms)
_SG_POLY_ORDER:     int   = 3       # Savitzky-Golay polynomial order
_MIN_SEGMENT_MS:    int   = 10      # minimum valid segment length (ms)
_ACTIVITY_THRESH:   float = 0.0     # z-score threshold for active segments
_DEFAULT_SFREQ:     int   = 256     # sampling frequency (Hz)


# ── Band decomposition helpers ────────────────────────────────────────────────

def _butter_bandpass(
    low_hz:  float,
    high_hz: float,
    sfreq:   int,
    order:   int = _FILTER_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a zero-phase Butterworth bandpass filter.

    Args:
        low_hz:  Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        sfreq:   Sampling frequency in Hz.
        order:   Filter order.

    Returns:
        (b, a) filter coefficient arrays.
    """
    nyq  = sfreq / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq
    low  = np.clip(low,  1e-4, 1.0 - 1e-4)
    high = np.clip(high, 1e-4, 1.0 - 1e-4)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a


def _power_envelope(x: np.ndarray) -> np.ndarray:
    """Magnitude envelope of the analytic signal via Hilbert transform.

    Args:
        x: 1-D signal array.

    Returns:
        Power envelope of the same shape as *x*.
    """
    return np.abs(signal.hilbert(x))


def _smooth_envelope(
    env:   np.ndarray,
    sfreq: int = _DEFAULT_SFREQ,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter to smooth the power envelope.

    Window length corresponds to ``_SG_WINDOW_MS`` ms; it is forced to
    an odd integer as required by scipy.

    Args:
        env:   1-D power envelope array.
        sfreq: Sampling frequency in Hz.

    Returns:
        Smoothed envelope of the same length as *env*.
    """
    window = int(_SG_WINDOW_MS * sfreq / 1000)
    if window % 2 == 0:
        window += 1
    window = max(window, _SG_POLY_ORDER + 1)
    return signal.savgol_filter(env, window_length=window,
                                polyorder=_SG_POLY_ORDER)


def _zscore_normalise(
    think_env: np.ndarray,
    rest_mean: float,
    rest_std:  float,
) -> np.ndarray:
    """Z-score normalise thinking-state envelope against resting-state stats.

    Args:
        think_env:  Smoothed thinking-state envelope (1-D).
        rest_mean:  Mean of the resting-state envelope.
        rest_std:   Standard deviation of the resting-state envelope.

    Returns:
        Z-score normalised envelope.
    """
    if rest_std < 1e-10:
        rest_std = 1e-10
    return (think_env - rest_mean) / rest_std


# ── Segment detection ─────────────────────────────────────────────────────────

def _detect_segments(
    z_env:    np.ndarray,
    sfreq:    int   = _DEFAULT_SFREQ,
    thresh:   float = _ACTIVITY_THRESH,
    min_ms:   int   = _MIN_SEGMENT_MS,
) -> List[float]:
    """Detect contiguous above-threshold segments and return their durations.

    Args:
        z_env:  Z-score normalised envelope (1-D).
        sfreq:  Sampling frequency in Hz.
        thresh: Activity threshold in z-score units.
        min_ms: Minimum segment length in milliseconds.

    Returns:
        List of segment durations in seconds.
    """
    min_samples = int(min_ms * sfreq / 1000)
    active      = z_env > thresh

    durations: List[float] = []
    in_seg   = False
    seg_start = 0

    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg    = True
            seg_start = i
        elif not a and in_seg:
            length = i - seg_start
            if length >= min_samples:
                durations.append(length / sfreq)
            in_seg = False

    # Handle segment that runs to the end of the signal.
    if in_seg:
        length = len(active) - seg_start
        if length >= min_samples:
            durations.append(length / sfreq)

    return durations


# ── Per-subject analysis ──────────────────────────────────────────────────────

def analyse_subject(
    think_data: np.ndarray,
    rest_data:  np.ndarray,
    sfreq:      int = _DEFAULT_SFREQ,
) -> Dict[str, List[float]]:
    """Run the full signal analysis pipeline for one subject.

    Computes per-band IWS segment durations by averaging over all trials
    and all EEG channels.

    Args:
        think_data: Thinking-state array of shape
                    ``(n_trials, n_channels, samples)``.
        rest_data:  Resting-state array of shape
                    ``(n_rest_trials, n_channels, samples)``.
        sfreq:      Sampling frequency in Hz.

    Returns:
        Dict mapping band name to a list of detected segment durations
        (in seconds) across all trials and channels.
    """
    band_durations: Dict[str, List[float]] = {b: [] for b in BANDS}

    # Pre-compute resting-state envelope statistics per band and channel.
    rest_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    n_ch = rest_data.shape[1]

    for band, (lo, hi) in BANDS.items():
        b, a = _butter_bandpass(lo, hi, sfreq)
        rest_means = np.zeros(n_ch)
        rest_stds  = np.zeros(n_ch)
        for ch in range(n_ch):
            ch_envs: List[np.ndarray] = []
            for trial in rest_data:
                filtered = signal.filtfilt(b, a, trial[ch])
                env      = _power_envelope(filtered)
                ch_envs.append(_smooth_envelope(env, sfreq))
            rest_all         = np.concatenate(ch_envs)
            rest_means[ch]   = rest_all.mean()
            rest_stds[ch]    = rest_all.std()
        rest_stats[band] = (rest_means, rest_stds)

    # Analyse thinking-state trials.
    for trial in think_data:
        for band, (lo, hi) in BANDS.items():
            b, a = _butter_bandpass(lo, hi, sfreq)
            r_means, r_stds = rest_stats[band]
            for ch in range(n_ch):
                filtered = signal.filtfilt(b, a, trial[ch])
                env      = _power_envelope(filtered)
                smooth   = _smooth_envelope(env, sfreq)
                z_env    = _zscore_normalise(smooth, r_means[ch], r_stds[ch])
                durs     = _detect_segments(z_env, sfreq)
                band_durations[band].extend(durs)

    return band_durations


# ── Dataset-level aggregation ─────────────────────────────────────────────────

def analyse_dataset(
    all_band_durations: List[Dict[str, List[float]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-subject band-duration dicts into dataset-level statistics.

    Args:
        all_band_durations: List of dicts as returned by :func:`analyse_subject`,
                            one per subject.

    Returns:
        Dict mapping band name to a sub-dict with keys ``mean``, ``std``,
        ``min``, and ``max`` (all in seconds).
    """
    merged: Dict[str, List[float]] = {b: [] for b in BANDS}
    for subj_durs in all_band_durations:
        for band, durs in subj_durs.items():
            merged[band].extend(durs)

    summary: Dict[str, Dict[str, float]] = {}
    for band, durs in merged.items():
        arr = np.array(durs) if durs else np.array([0.0])
        summary[band] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "n":    len(durs),
        }
    return summary


# ── Statistical test ──────────────────────────────────────────────────────────

def ttest_iws(
    durations:      List[float],
    trial_duration: float = 5.0,
) -> Tuple[float, float]:
    """One-sample t-test: H0 = mean IWS duration == full trial duration.

    Tests whether imagined speech activity spans the entire trial or is
    significantly shorter, as described in Chapter 4, Section 3.2.

    Args:
        durations:      Flat list of IWS segment durations (seconds).
        trial_duration: Full trial duration to test against (default 5.0 s).

    Returns:
        Tuple ``(t_statistic, p_value)``.  A very negative t with p ≈ 0
        confirms that the mean IWS duration is significantly less than
        the full trial (H1 accepted, H0 rejected).
    """
    if len(durations) < 2:
        warnings.warn("Fewer than 2 durations — t-test unreliable.")
        return float("nan"), float("nan")
    t_stat, p_val = stats.ttest_1samp(durations, popmean=trial_duration)
    return float(t_stat), float(p_val)

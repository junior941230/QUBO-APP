import numpy as np
from scipy.signal import welch


BANDS = (
    ("Delta", 0.5, 4),
    ("Theta", 4, 8),
    ("Alpha", 8, 12),
    ("Beta", 12, 30),
    ("Gamma", 30, 40),
)


def extract_band_power(epoch, sfreq=256, log_power=False, relative_power=False):
    """Return flattened channel band powers in the legacy band-major order."""
    freqs, psd = welch(epoch, fs=sfreq, nperseg=sfreq)
    powers = np.stack([
        psd[:, np.logical_and(freqs >= low, freqs <= high)].mean(axis=1)
        for _, low, high in BANDS
    ], axis=1)
    if relative_power:
        powers = powers / np.maximum(powers.sum(axis=1, keepdims=True), 1e-12)
    if log_power:
        powers = np.log10(np.maximum(powers, 1e-12))
    return powers.T.reshape(-1)


def robust_normalize_features(features):
    """Scale each feature using the recording median and interquartile range."""
    features = np.asarray(features, dtype=float)
    if features.size == 0:
        return features.copy()
    median = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return (features - median) / scale


def add_causal_context(features, window_epochs):
    """Append a trailing mean that never crosses a recording boundary."""
    features = np.asarray(features, dtype=float)
    window_epochs = int(window_epochs)
    if window_epochs <= 1 or len(features) == 0:
        return features.copy()
    cumulative = np.vstack([np.zeros((1, features.shape[1])), features.cumsum(axis=0)])
    starts = np.maximum(0, np.arange(len(features)) + 1 - window_epochs)
    ends = np.arange(1, len(features) + 1)
    means = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    return np.hstack([features, means])

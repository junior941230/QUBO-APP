"""EDF preprocessing and per-record feature extraction."""

import gc
import os
import warnings

from joblib import Parallel, delayed
import mne
import numpy as np

from config import EPOCH_DURATION_SECONDS
from core.channels import CANONICAL_CHB_CHANNELS, build_channel_plan
from core.options import ExperimentOptions
from features import (
    BANDS,
    add_causal_context,
    extract_band_power,
    robust_normalize_features,
)


DURATION = EPOCH_DURATION_SECONDS
STANDARD_CHB_CHANNELS = list(CANONICAL_CHB_CHANNELS)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


def normalize_chb_channels(raw):
    """Return a RawArray with the canonical 18-channel bipolar montage."""
    normalized = []
    for instruction in build_channel_plan(raw.ch_names):
        if instruction[0] == "direct":
            normalized.append(raw.get_data(picks=[instruction[1]])[0])
        else:
            _, anode, cathode = instruction
            pair = raw.get_data(picks=[anode, cathode])
            normalized.append(pair[0] - pair[1])
    info = mne.create_info(
        ch_names=list(CANONICAL_CHB_CHANNELS),
        sfreq=float(raw.info["sfreq"]),
        ch_types=["eeg"] * len(CANONICAL_CHB_CHANNELS),
    )
    return mne.io.RawArray(np.vstack(normalized), info, verbose=False)


def process_all_files(
    file_list,
    seizure_times_by_file,
    n_jobs=-1,
    options=None,
):
    """Preprocess EDF files independently and preserve file boundaries."""
    options = options or ExperimentOptions()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(preprocess_one_file)(
            path,
            seizure_times_by_file.get(os.path.basename(path), []),
            options,
        )
        for path in file_list
    )
    file_names = [os.path.basename(path) for path in file_list]
    features = {name: result[0] for name, result in zip(file_names, results)}
    labels = {name: result[1] for name, result in zip(file_names, results)}
    return features, labels


def processAllFiles(fileList, seizureTimesDict, nJobs=-1, options=None):
    """Compatibility alias for the original public API."""
    return process_all_files(fileList, seizureTimesDict, nJobs, options)


def preprocess_one_file(edf_path, seizure_times, options=None):
    """Load one EDF recording, filter it, and create epoch-level features."""
    options = options or ExperimentOptions()
    source_raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw = normalize_chb_channels(source_raw)
    del source_raw
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False, n_jobs=1)
    raw.notch_filter(freqs=60.0, verbose=False, n_jobs=1)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=DURATION, overlap=0.0, verbose=False
    )
    epoch_data = epochs.get_data(copy=False)
    sfreq = int(round(float(raw.info["sfreq"])))
    features = np.asarray([
        extract_band_power(
            epoch,
            sfreq=sfreq,
            log_power=options.log_power,
            relative_power=options.relative_power,
        )
        for epoch in epoch_data
    ])
    expected = len(CANONICAL_CHB_CHANNELS) * len(BANDS)
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(
            f"Unexpected feature shape {features.shape}; expected (epochs, {expected})"
        )
    if options.robust_normalize:
        features = robust_normalize_features(features)
    if options.temporal_context_seconds > 0:
        window_epochs = max(1, round(options.temporal_context_seconds / DURATION))
        features = add_causal_context(features, window_epochs)

    labels = np.zeros(len(features), dtype=int)
    for start, end in seizure_times:
        start_index = max(0, int(start / DURATION))
        end_index = min(len(features), int(end / DURATION))
        labels[start_index:end_index] = 1

    del raw, epochs, epoch_data
    gc.collect()
    return features, labels

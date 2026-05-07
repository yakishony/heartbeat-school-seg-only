import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt

from env import RATE


NORMALIZE_TO_GLOBAL_MAX = False     # Whether to normalize by global max vs per-recording max
BANDPASS_LOWCUT = 60.0              # Bandpass filter low cutoff (Hz)
BANDPASS_HIGHCUT = 150.0            # Bandpass filter high cutoff (Hz)


def find_max_of_all_recordings_for_normalization(dataset):
    """Find the global maximum absolute amplitude across all recordings (for global normalization)."""
    max_values = []
    for rec in dataset.values():
        max_values.append(np.max(np.abs(rec['signal'])))
    return np.max(max_values)


def normalize_signal(signal, global_max=None):
    """Normalize signal to [-1, 1]. If global_max is None, normalizes by the signal's own max."""
    if global_max is None:
        max_value = np.max(np.abs(signal))
    else:
        max_value = global_max
    return signal / max_value


def bandpass_filter(signal, BANDPASS_LOWCUT=BANDPASS_LOWCUT, BANDPASS_HIGHCUT=BANDPASS_HIGHCUT, fs=RATE, order=4):
    """Apply zero-phase Butterworth bandpass filter (keeps frequencies between LOWCUT and HIGHCUT)."""
    nyq = 0.5 * fs  # Nyquist frequency = half the sampling rate
    # butter(): design the filter coefficients; filtfilt(): apply forward+backward for zero phase shift
    b, a = butter(order, [BANDPASS_LOWCUT / nyq, BANDPASS_HIGHCUT / nyq], btype='band')
    return filtfilt(b, a, signal)


def normalize_dataset(dataset):
    """Normalize all recordings in the dataset. Returns (normalized_dataset, global_max)."""
    if NORMALIZE_TO_GLOBAL_MAX:
        global_max = find_max_of_all_recordings_for_normalization(dataset)
    else:
        global_max = None  # each recording normalized by its own max
    normalized = {}
    for rec_id, rec in dataset.items():
        normalized[rec_id] = {
            **rec,                                                    # copy all fields (sr, type, etc.)
            'signal': normalize_signal(rec['signal'], global_max),   # overwrite signal with normalized version
            'y': rec['y'].copy(),
        }
    return normalized, global_max


def bandpass_filter_dataset(dataset):
    """Apply bandpass filter to every recording in the dataset."""
    filtered = {}
    for rec_id, rec in dataset.items():
        filtered[rec_id] = {
            **rec,
            'signal': bandpass_filter(rec['signal']),
        }
    return filtered

def downsample_signal(signal, factor):
    """Downsample by taking every factor-th sample."""
    return signal[::factor]


def downsample_dataset(dataset, factor):
    """Downsample every recording (both signal and labels) in the dataset."""
    downsampled = {}
    for rec_id, rec in dataset.items():
        downsampled[rec_id] = {
            **rec,
            'signal': downsample_signal(rec['signal'], factor),
            'y': downsample_signal(rec['y'], factor),  # labels must be downsampled too to stay aligned
        }
    return downsampled


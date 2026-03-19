import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt

from env import RATE


NORMALIZE_TO_GLOBAL_MAX = True
LOWCUT = 61.0
HIGHCUT = 160.0


def find_max_of_all_recordings_for_normalization(dataset):
    max_values = []
    for rec in dataset.values():
        max_values.append(np.max(np.abs(rec['signal'])))
    return np.max(max_values)


def normalize_signal(data, global_max = None):
    if global_max is None:
        max_value = np.max(np.abs(data))
    else:
        max_value = global_max
    return data / max_value


def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=RATE, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def normalize_dataset(dataset):
    if NORMALIZE_TO_GLOBAL_MAX:
        global_max = find_max_of_all_recordings_for_normalization(dataset)
    else:
        global_max = None
    normalized = {}
    for rec_id, rec in dataset.items():
        normalized[rec_id] = {
            **rec,
            'signal': normalize_signal(rec['signal'], global_max),
            'y': rec['y'].copy(),
        }
    return normalized, global_max


def bandpass_filter_dataset(dataset):
    filtered = {}
    for rec_id, rec in dataset.items():
        filtered[rec_id] = {
            **rec,
            'signal': bandpass_filter(rec['signal']),
        }
    return filtered


def downsample_dataset(dataset, factor):
    downsampled = {}
    for rec_id, rec in dataset.items():
        downsampled[rec_id] = {
            **rec,
            'signal': rec['signal'][::factor],
            'y': rec['y'][::factor],
        }
    return downsampled


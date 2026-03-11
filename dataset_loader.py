"""
Lazy tf.data.Dataset loader that reads per-recording .npy files from disk.
Only one batch lives in RAM at a time.
"""
import numpy as np
import tensorflow as tf

from env import DATA_FOR_ML

BATCH_SIZE = 32


def list_recording_ids(data_dir=DATA_FOR_ML):
    signal_dir = data_dir / "signals"
    return sorted(p.stem for p in signal_dir.glob("*.npy"))


def _load_npy_pair(rec_id_tensor):
    """tf.py_function helper: load signal + label arrays for one recording."""
    rec_id = rec_id_tensor.numpy().decode("utf-8")
    signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id}.npy")  # float32
    label = np.load(DATA_FOR_ML / "labels" / f"{rec_id}.npy")    # int64
    return signal, label


def load_pair_wrapper(rec_id):
    signal, label = tf.py_function(
        _load_npy_pair,
        [rec_id],
        [tf.float32, tf.int64],
    )
    signal.set_shape([None])
    label.set_shape([None])
    return signal, label


def make_dataset(batch_size=BATCH_SIZE, shuffle=True):
    """
    Returns a tf.data.Dataset of (signal, label) pairs loaded lazily from disk.

    Each element has variable length — apply your own windowing / padding
    before calling .batch() if needed.
    """
    rec_ids = list_recording_ids()
    ds = tf.data.Dataset.from_tensor_slices(rec_ids)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(rec_ids), reshuffle_each_iteration=True)

    ds = ds.map(load_pair_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # NOTE: recordings have variable lengths.
    # Add your windowing/padding step here before batching, e.g.:
    #   ds = ds.flat_map(window_fn)
    # Then:
    #   ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


if __name__ == "__main__":
    ds = make_dataset(shuffle=False)
    for signal, label in ds.take(1):
        print(f"signal shape: {signal.shape}, dtype: {signal.dtype}")
        print(f"label  shape: {label.shape},  dtype: {label.dtype}")
        print(f"unique labels: {np.unique(label.numpy())}")

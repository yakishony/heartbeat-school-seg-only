"""
Orchestrate the full pipeline: load → normalize → bandpass → save .npy → tf.data.Dataset.
"""
import gc
import shutil

import matplotlib.pyplot as plt
import numpy as np

from env import DATA_FOR_ML, DOWNSAMPLE_FACTOR
from get_data import load_dataset
from prepare_data import normalize_dataset, bandpass_filter_dataset, downsample_dataset
from split_data_into_fixed_length_recordings import split_data_into_fixed_length_recordings
from understand_data import plot_fft
from utils.plot_utils import plot_recording_before_and_after


def save_dataset_as_npy(dataset, out_dir=DATA_FOR_ML):
    """Save each recording's signal and labels as separate .npy files."""
    if out_dir.exists():
        shutil.rmtree(out_dir) # remove the directory if it exists
    signal_dir = out_dir / "signals"
    label_dir = out_dir / "labels"
    signal_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for rec_id, rec in dataset.items(): # saving each recording as a separate .npy file 
        # so that it can be loaded into RAM seperately
        np.save(signal_dir / f"{rec_id}.npy", rec["signal"].astype(np.float32))
        np.save(label_dir / f"{rec_id}.npy", rec["y"].astype(np.int64))

    print(f"Saved {len(dataset)} recordings to {out_dir}")


def run():
    # 1. Load
    dataset_raw, missing = load_dataset()
    print(f"Loaded {len(dataset_raw)} recordings ({len(missing)} missing annotations)")

    # 2. Split into fixed length recordings
    dataset_split, count_deleted_recordings, count_full_0_splits = split_data_into_fixed_length_recordings(dataset_raw)
    # del dataset_raw
    # gc.collect()

    # 3. Normalize
    dataset_normalized, global_max = normalize_dataset(dataset_split)
    del dataset_split
    gc.collect()
    print(f"Normalized (global_max={global_max:.4f})")
    signal_id = plot_fft(dataset_normalized, title="Before Bandpass Filtering")
   
    # 4. Bandpass filter
    dataset_bandpassed = bandpass_filter_dataset(dataset_normalized)
    del dataset_normalized
    gc.collect()
    print("Bandpass filtered")
    plot_fft(dataset_bandpassed, signal_id=signal_id, title="After Bandpass Filtering")
    

    # 5. Downsample (safe after bandpass — no aliasing)
    dataset_downsampled = downsample_dataset(dataset_bandpassed, DOWNSAMPLE_FACTOR)
    del dataset_bandpassed
    gc.collect()
    print(f"Downsampled by {DOWNSAMPLE_FACTOR}x")

    # 6. Save processed dataset as .npy files
    save_dataset_as_npy(dataset_downsampled) # will remove the directory if it already exists

    # 7. Free dataset from RAM
    del dataset_downsampled
    gc.collect()
    print("Freed dataset from RAM")


if __name__ == "__main__":
    run()

"""
Orchestrate the full pipeline: load → normalize → bandpass → save .npy → tf.data.Dataset.
"""
import gc

import matplotlib.pyplot as plt
import numpy as np

from env import DATA_FOR_ML
from get_data import load_dataset
from prepare_data import normalize_dataset, bandpass_filter_dataset
from utils.plot_utils import plot_recording_before_and_after


def save_dataset_as_npy(dataset, out_dir=DATA_FOR_ML):
    """Save each recording's signal and labels as separate .npy files."""
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

    # 2. Normalize
    dataset_normalized, global_max = normalize_dataset(dataset_raw)
    del dataset_raw
    gc.collect()
    print(f"Normalized (global_max={global_max:.4f})")

    # 3. Bandpass filter
    dataset_bandpassed = bandpass_filter_dataset(dataset_normalized)
    del dataset_normalized
    gc.collect()
    print("Bandpass filtered")

    # 4. Save processed dataset as .npy files
    save_dataset_as_npy(dataset_bandpassed)

    # 5. Free dataset from RAM
    del dataset_bandpassed
    gc.collect()
    print("Freed dataset from RAM")


if __name__ == "__main__":
    run()

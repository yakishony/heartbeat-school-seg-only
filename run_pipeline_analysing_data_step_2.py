"""
Orchestrate the full pipeline: load → normalize → bandpass → save .npy → tf.data.Dataset.
"""
import gc

import matplotlib.pyplot as plt
import numpy as np

from env import DOWNSAMPLE_FACTOR
from  download_data_and_annotate_step1 import load_dataset_raw, save_dataset_as_npy
from run_pipline_analysing_utils import normalize_dataset, bandpass_filter_dataset, downsample_dataset
from split_data_into_fixed_length_recordings import split_data_into_fixed_length_recordings_without_unannotated




def run():
    # 1. Load
    dataset_raw, missing = load_dataset_raw()
    print(f"Loaded {len(dataset_raw)} recordings ({len(missing)} missing annotations)")

    # 2. Split into fixed length recordings (less unannotated)
    dataset_split = split_data_into_fixed_length_recordings_without_unannotated(dataset_raw)
    del dataset_raw
    gc.collect()

    # 3. Normalize
    dataset_normalized, global_max = normalize_dataset(dataset_split)
    del dataset_split
    gc.collect()
    print(f"Normalized (global_max={global_max:.4f})" if global_max is not None else "Normalized (per-recording)")
    
    # 4. Bandpass filter
    dataset_bandpassed = bandpass_filter_dataset(dataset_normalized)
    del dataset_normalized
    gc.collect()
    print("Bandpass filtered")


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

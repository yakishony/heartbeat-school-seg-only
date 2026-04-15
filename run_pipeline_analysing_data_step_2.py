"""
Orchestrate the full pipeline: load → normalize → bandpass → save .npy → tf.data.Dataset.
"""
import gc

import matplotlib.pyplot as plt
import numpy as np

from env import DOWNSAMPLE_FACTOR
from  download_data_and_annotate_step1 import load_dataset_raw, save_dataset_as_npy
from run_pipline_analysing_utils import normalize_dataset, bandpass_filter_dataset, downsample_dataset
from split_data_into_fixed_length_recordings import split_data_into_fixed_length_recordings_without_unrecognized
from utils.plot_utils import plot_recording, plot_segmented_signal_interactive




def run():
    example_rec_id = "50348_AV_1"
    # 1. Load
    dataset_raw, missing = load_dataset_raw()
    print(f"Loaded {len(dataset_raw)} recordings ({len(missing)} missing annotations)")
    # plot_recording(example_rec_id, dataset_raw, name="raw", sa_fig=True)

    # 2. Split into fixed length recordings (less unrecognized)
    dataset_split = split_data_into_fixed_length_recordings_without_unrecognized(dataset_raw)
    del dataset_raw
    gc.collect()

    # find all splits of example_rec_id
    example_rec_id_splits = [rec_id for rec_id in dataset_split.keys() if example_rec_id in rec_id]
    for rec_id in example_rec_id_splits:
        # plot_recording(rec_id, dataset_split, name="split", sa_fig=True)
        pass

    # 3. Normalize
    dataset_normalized, global_max = normalize_dataset(dataset_split)
    del dataset_split
    gc.collect()
    print(f"Normalized (global_max={global_max:.4f})" if global_max is not None else "Normalized (per-recording)")
    # plot_recording(example_rec_id_splits[0], dataset_normalized, name="normalized", sa_fig=True)
    
    # 4. Bandpass filter
    dataset_bandpassed = bandpass_filter_dataset(dataset_normalized)
    del dataset_normalized
    gc.collect()
    print("Bandpass filtered")
    # plot_recording(example_rec_id_splits[0], dataset_bandpassed, name="bandpassed", sa_fig=True)

    # 5. Downsample (safe after bandpass — no aliasing)
    # plot_recording(example_rec_id_splits[0], dataset_bandpassed, name="before downsampled zoomed in",xl=[1.4, 1.54], sa_fig=True)
    dataset_downsampled = downsample_dataset(dataset_bandpassed, DOWNSAMPLE_FACTOR)
    del dataset_bandpassed
    gc.collect()
    print(f"Downsampled by {DOWNSAMPLE_FACTOR}x")
    # plot_recording(example_rec_id_splits[0], dataset_downsampled, name="after downsampled zoomed in",xl=[1.4, 1.54], sa_fig=True)
    # plot_recording(example_rec_id_splits[0], dataset_downsampled, name="downsampled", sa_fig=True)
    fig = plot_segmented_signal_interactive(dataset_downsampled[example_rec_id_splits[0]]['signal'], dataset_downsampled[example_rec_id_splits[0]]['y'])
    fig.show()

    # 6. Save processed dataset as .npy files
    save_dataset_as_npy(dataset_downsampled) # will remove the directory if it already exists

    # 7. Free dataset from RAM
    del dataset_downsampled
    gc.collect()
    print("Freed dataset from RAM")


if __name__ == "__main__":
    run()

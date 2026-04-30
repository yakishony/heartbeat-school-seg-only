import os
import pickle
import shutil

import kagglehub
import numpy as np
import pandas as pd
from scipy.io import wavfile

from env import DATA_DOWNLOADED, DATA_FOR_ML, DATA_RAW_DIR
from pathlib import Path



def download_dataset(local_dir=DATA_DOWNLOADED):
    """Download dataset from Kaggle, or reuse local copy if it exists."""
    local_dir = Path(local_dir)
    if local_dir.is_dir() and any(local_dir.rglob("*.wav")): # to prevent downloading the dataset again if it already exists
        print(f"Using cached dataset at {local_dir}")
        return str(local_dir)

    print("Downloading dataset from Kaggle...")
    src = kagglehub.dataset_download("bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2")
    print(f"Copying to {local_dir} for future reuse...")
    shutil.copytree(src, local_dir, dirs_exist_ok=True)
    return str(local_dir)

def build_sample_labels(tsv_path, signal_length, sr):
    """Convert interval annotations to per-sample labels."""
    y = np.zeros(signal_length, dtype=np.int64)
    annotated = np.zeros(signal_length, dtype=bool)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "label"])

    for _, row in df.iterrows():
        start_sample = max(int(row["start"] * sr), 0)
        end_sample = min(int(row["end"] * sr), signal_length)
        y[start_sample:end_sample] = int(row["label"])
        annotated[start_sample:end_sample] = True

    n_unannotated = int((~annotated).sum())
    n_annotated_as_0 = int((annotated & (y == 0)).sum())
    return y, n_unannotated, n_annotated_as_0

def index_tsv_files(path):
    tsv_files = {}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".tsv"):
                tsv_files[f.replace(".tsv", "")] = os.path.join(root, f)
    return tsv_files


def load_dataset_raw(path_to_load_data_from=None, use_cache=True): # (save to pickle as dict) and returns the dict and the list of missing annotations
    """ returns pickle file with dataset as dict(rec_id: {'signal': signal, 'sr': sr, 'y': y, 'type': type}) 
    and missing annotations as list(rec_id) """

    if use_cache and DATA_RAW_DIR.exists():
        print(f"Loading cached dataset from {DATA_RAW_DIR}")
        with open(DATA_RAW_DIR, "rb") as f:
            return pickle.load(f)

    if path_to_load_data_from is None:
        path_to_load_data_from = download_dataset()

    tsv_files = index_tsv_files(path_to_load_data_from)
    dataset = {}
    missing_annotations = []
    total_samples = 0
    total_unannotated = 0
    total_annotated_as_0 = 0

    for root, _, files in os.walk(path_to_load_data_from):
        for f in files:
            if not f.endswith(".wav"):
                continue

            rec_id = f.replace(".wav", "")
            if rec_id not in tsv_files:
                missing_annotations.append(rec_id)
                continue

            sr, signal = wavfile.read(os.path.join(root, f))
            if signal.ndim > 1:
                signal = signal[:, 0]

            y, n_unannotated, n_annotated_as_0 = build_sample_labels(tsv_files[rec_id], len(signal), sr)
            total_samples += len(signal)
            total_unannotated += n_unannotated
            total_annotated_as_0 += n_annotated_as_0
            dataset[rec_id] = {"signal": signal, "sr": sr, "y": y, 'type': rec_id.split('_')[1]}

    print(f"Unannotated (not covered by any TSV row, y=0 by default): "
          f"{total_unannotated/total_samples:.1%} ({total_unannotated}/{total_samples} samples)")
    print(f"Annotated as 0 (explicitly labeled 0 in TSV): "
          f"{total_annotated_as_0/total_samples:.1%} ({total_annotated_as_0}/{total_samples} samples)")

    if use_cache:
        print(f"Caching dataset to {DATA_RAW_DIR}")
        with open(DATA_RAW_DIR, "wb") as f:
            pickle.dump((dataset, missing_annotations), f)

    return dataset, missing_annotations

def save_dataset_as_npy(dataset, out_dir=DATA_FOR_ML): # (save to npy files)
    """Save each recording's signal and segmentation labels as .npy."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    signal_dir = out_dir / "signals"
    label_dir = out_dir / "labels"
    signal_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for rec_id, rec in dataset.items():
        np.save(signal_dir / f"{rec_id}.npy", rec["signal"].astype(np.float32))
        np.save(label_dir / f"{rec_id}.npy", rec["y"].astype(np.int64))

    print(f"Saved {len(dataset)} recordings to {out_dir}")



if __name__ == "__main__":
    dataset, missing = load_dataset_raw()
    print(f"Loaded recordings: {len(dataset)}")
    print(f"Missing annotations: {len(missing)}")

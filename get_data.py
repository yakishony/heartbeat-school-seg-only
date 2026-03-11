import os
import shutil

import kagglehub
import numpy as np
import pandas as pd
from scipy.io import wavfile

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_dataset(local_dir=LOCAL_DATA_DIR):
    """Download dataset from Kaggle, or reuse local copy if it exists."""
    if os.path.isdir(local_dir) and any(f.endswith(".wav") for f in os.listdir(local_dir)):
        print(f"Using cached dataset at {local_dir}")
        return local_dir

    print("Downloading dataset from Kaggle...")
    src = kagglehub.dataset_download("bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2")
    print(f"Copying to {local_dir} for future reuse...")
    shutil.copytree(src, local_dir, dirs_exist_ok=True)
    return local_dir


def build_sample_labels(tsv_path, signal_length, sr):
    """Convert interval annotations to per-sample labels."""
    y = np.zeros(signal_length, dtype=np.int64)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "label"])

    for _, row in df.iterrows():
        start_sample = max(int(row["start"] * sr), 0)
        end_sample = min(int(row["end"] * sr), signal_length)
        y[start_sample:end_sample] = int(row["label"])

    return y


def index_tsv_files(path):
    tsv_files = {}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".tsv"):
                tsv_files[f.replace(".tsv", "")] = os.path.join(root, f)
    return tsv_files


def load_dataset(path=None):
    if path is None:
        path = download_dataset()

    tsv_files = index_tsv_files(path)
    dataset = {}
    missing_annotations = []

    for root, _, files in os.walk(path):
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

            y = build_sample_labels(tsv_files[rec_id], len(signal), sr)
            dataset[rec_id] = {"signal": signal, "sr": sr, "y": y}

    return dataset, missing_annotations


if __name__ == "__main__":
    dataset, missing = load_dataset()
    print(f"Loaded recordings: {len(dataset)}")
    print(f"Missing annotations: {len(missing)}")
    for rec_id, rec in list(dataset.items())[:3]:
        print(f"  {rec_id}: sr={rec['sr']}, len={len(rec['signal'])}, labels={np.unique(rec['y'])}")

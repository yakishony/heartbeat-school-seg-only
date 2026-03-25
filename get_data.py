import os
import pickle
import shutil

import kagglehub
import numpy as np
import pandas as pd
from scipy.io import wavfile

from env import DATA_DOWNLOADED, NUMERIC_CATEGORIZED_MURMUR, RAW_DATA_DIR
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


def build_murmur_map(txt_dir=None):
    """Parse patient .txt files and return {patient_id: murmur} dict."""
    if txt_dir is None:
        txt_dir = DATA_DOWNLOADED
    txt_dir = Path(txt_dir)
    murmur_map = {}
    for txt_file in txt_dir.glob("*.txt"):
        patient_id = txt_file.stem
        for line in txt_file.read_text().splitlines():
            if line.startswith("#Murmur:"):
                murmur_map[patient_id] = NUMERIC_CATEGORIZED_MURMUR[line.split(":")[1].strip()]
                break
    print(f"Loaded murmur labels for {len(murmur_map)} patients")
    return murmur_map


def load_dataset(path_to_load_data_from=None, use_cache=True): # what are the missing annotations? 
    """ returns pickle file with dataset as dict(rec_id: {'signal': signal, 'sr': sr, 'y': y, 'type': type, 'murmur': murmur}) 
    and missing annotations as list(rec_id) """

    if use_cache and RAW_DATA_DIR.exists():
        print(f"Loading cached dataset from {RAW_DATA_DIR}")
        with open(RAW_DATA_DIR, "rb") as f:
            return pickle.load(f)

    if path_to_load_data_from is None:
        path_to_load_data_from = download_dataset()

    tsv_files = index_tsv_files(path_to_load_data_from)
    murmur_map = build_murmur_map()
    dataset = {}
    missing_annotations = []

    for root, _, files in os.walk(path_to_load_data_from):
        for f in files:
            if not f.endswith(".wav"):
                continue

            rec_id = f.replace(".wav", "")
            if rec_id not in tsv_files:
                missing_annotations.append(rec_id)
                continue

            patient_id = rec_id.split("_")[0]
            murmur = murmur_map[patient_id]

            sr, signal = wavfile.read(os.path.join(root, f))
            if signal.ndim > 1:
                signal = signal[:, 0]

            y = build_sample_labels(tsv_files[rec_id], len(signal), sr)
            dataset[rec_id] = {"signal": signal, "sr": sr, "y": y, 'type': rec_id.split('_')[1], 'murmur': murmur}

    if use_cache:
        print(f"Caching dataset to {RAW_DATA_DIR}")
        with open(RAW_DATA_DIR, "wb") as f:
            pickle.dump((dataset, missing_annotations), f)

    return dataset, missing_annotations




if __name__ == "__main__":
    dataset, missing = load_dataset()
    print(f"Loaded recordings: {len(dataset)}")
    print(f"Missing annotations: {len(missing)}")
    for rec_id, rec in list(dataset.items())[:3]:
        print(f"  {rec_id}: sr={rec['sr']}, len={len(rec['signal'])}, labels={np.unique(rec['y'])}")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import FIGURES_DIR, RATE, TYPES, CLASSES, CATEGORY_NAMES, DATA_FOR_ML_X2
from download_data_and_annotate_step1 import load_dataset_raw
from run_pipline_analysing_utils import BANDPASS_LOWCUT, BANDPASS_HIGHCUT 


def dataset_to_summary_df(dataset: dict) -> pd.DataFrame:
    """One-row-per-recording summary with rec_id, type, sr, n_samples, length_sec."""
    return pd.DataFrame([
        {'rec_id': rec_id, 'type': rec['type'], 'sr': rec['sr'],
         'n_samples': len(rec['signal']), 'length_sec': len(rec['signal']) / rec['sr']}
        for rec_id, rec in dataset.items()
    ])

def build_summary(dataset):
    """Build a summary of the dataset - to see the distribution of recordings by type, length, and rate"""
    df = dataset_to_summary_df(dataset)
    df.to_csv(FIGURES_DIR / "recordings_summary.csv", index=False)
    print(f"Saved {FIGURES_DIR / 'recordings_summary.csv'}  ({len(df)} rows)")

    assert set(df['type'].unique()) == set(TYPES)
    print('Recording types:', df['type'].unique().tolist())

    assert df['sr'].nunique() == 1
    assert df['sr'].iloc[0] == RATE
    print('Global rate is', RATE)

    longest = df.loc[df['length_sec'].idxmax()]
    shortest = df.loc[df['length_sec'].idxmin()]
    print(f"Longest recording: {longest['length_sec']:.1f}s (ID: {longest['rec_id']})")
    print(f"Shortest recording: {shortest['length_sec']:.1f}s (ID: {shortest['rec_id']})")
    return df


def plot_stacked_histogram_by_type(df):
    """Plot the distribution of recording lengths by type(TV, MV, PV, Phc, AV) - to see if some types have longer or shorter recordings"""
    plt.figure(figsize=(12, 6))
    labels = df['type'].unique().tolist()
    values = [df.loc[df['type'] == t, 'length_sec'].values for t in labels]
    plt.hist(values, bins=50, stacked=True, label=labels)
    plt.xlabel("Length (seconds)")
    plt.ylabel("Frequency")
    plt.title("Stacked Histogram of Recording Lengths by Type")
    plt.legend()
    plt.savefig(FIGURES_DIR / "fig_histogram.png", dpi=100)
    print(f"Saved {FIGURES_DIR / 'fig_histogram.png'}")

def plot_type_pie(df):
    """Plot the distribution of recording types(TV, MV, PV, Phc, AV)"""
    type_counts = df['type'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Recording TYPES")
    plt.savefig(FIGURES_DIR / "fig_pie.png", dpi=100)
    print(f"Saved {FIGURES_DIR / 'fig_pie.png'}")


def plot_category_pie(dataset, name="fig_category_pie"):
    """Plot the distribution of time samples by category (unannotated, S1, systolic, S2, diastole)"""
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for rec in dataset.values():
        for c in CLASSES:
            counts[c] += np.sum(rec['y'] == c)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=CATEGORY_NAMES, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Time Samples by Category")
    plt.savefig(FIGURES_DIR / (name + ".png"), dpi=100)
    plt.show()
    print(f"Saved {FIGURES_DIR / (name + '.png')}")

def plot_unannotated_by_timestamp():
    """For each timestamp, plot the % of recordings labeled 'unannotated' (class 0). - to understand where unannotated samples are"""
    
    label_dir = DATA_FOR_ML_X2 / "labels"
    label_files = sorted(label_dir.glob("*.npy"))

    labels = np.stack([np.load(f) for f in label_files])  # (n_recordings, n_timestamps)
    pct_unannotated = np.mean(labels == 0, axis=0) * 100  # % per timestamp

    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(pct_unannotated)), pct_unannotated, width=1.0, edgecolor='none')
    plt.xlabel("Timestamp (sample index)")
    plt.ylabel("% recordings labeled unannotated")
    plt.title("Percentage of Recordings with Unannotated Label per Timestamp")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unannotated_by_timestamp.png", dpi=100)
    plt.show()
    print(f"Saved {FIGURES_DIR / 'fig_unannotated_by_timestamp.png'}")


def plot_fft(data_set=None, signal_id=None, title="FFT", sr=RATE):
    """Plot the frequency spectrum (FFT) of a single recording with bandpass filter boundaries."""

    if data_set is None:
        data_set, _ = load_dataset_raw()
    if signal_id is None:
        signal_id = random.choice(list(data_set.keys()))

    signal = data_set[signal_id]['signal']
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)        # frequency bins (0 to Nyquist)
    magnitudes = np.abs(np.fft.rfft(signal)) / n   # normalized magnitude spectrum

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, magnitudes)
    # show the bandpass filter boundaries as red dashed lines
    plt.axvline(BANDPASS_LOWCUT, color='red', linestyle='--', linewidth=1.5, label=f'Lowcut {BANDPASS_LOWCUT} Hz')
    plt.axvline(BANDPASS_HIGHCUT, color='red', linestyle='--', linewidth=1.5, label=f'Highcut {BANDPASS_HIGHCUT} Hz')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.xlim(0, sr / 2)  # show up to Nyquist frequency
    plt.legend()
    plt.tight_layout()
    name = f"fig_{title}_{signal_id}.png"
    plt.savefig(FIGURES_DIR / name , dpi=100)
    plt.show()

    print(f"Saved {FIGURES_DIR / name }")
    return signal_id

if __name__ == "__main__":
    pass

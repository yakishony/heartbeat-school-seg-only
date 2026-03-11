import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import load_dataset
from env import FIGURES_DIR, RATE, TYPES
from utils.plot_utils import plot_recording
from utils.data_utils import dataset_to_summary_df


def build_summary(dataset):
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


def plot_pie(df):
    type_counts = df['type'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Recording TYPES")
    plt.savefig(FIGURES_DIR / "fig_pie.png", dpi=100)
    print(f"Saved {FIGURES_DIR / 'fig_pie.png'}")


def plot_global_vs_good_max(dataset):
    max_signals_by_type = {t: [] for t in TYPES}
    good_max_by_type = {t: [] for t in TYPES}
    for rec in dataset.values():
        t = rec['type']
        max_signals_by_type[t].append(np.max(np.abs(rec['signal'])))
        resolved = rec['y'] != 0
        if resolved.any():
            good_max_by_type[t].append(np.max(np.abs(rec['signal'][resolved])))
        else:
            good_max_by_type[t].append(0.0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(TYPES)))
    type_colors = dict(zip(TYPES, colors))

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    for t in TYPES:
        ax_scatter.scatter(max_signals_by_type[t], good_max_by_type[t],
                           alpha=0.4, s=10, color=type_colors[t], label=t)

    all_max = [v for vals in max_signals_by_type.values() for v in vals]
    all_good = [v for vals in good_max_by_type.values() for v in vals]
    max_val = max(max(all_max), max(all_good))
    ax_scatter.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    ax_scatter.set_xlabel("Global Max |signal|")
    ax_scatter.set_ylabel("Good Max |signal| (y ≠ 0)")
    ax_scatter.legend()

    ax_hist_x.hist([max_signals_by_type[t] for t in TYPES], bins=50,
                   stacked=True, color=[type_colors[t] for t in TYPES], label=TYPES, edgecolor='black')
    ax_hist_x.tick_params(labelbottom=False)
    ax_hist_x.set_ylabel("Count")

    ax_hist_y.hist([good_max_by_type[t] for t in TYPES], bins=50,
                   stacked=True, color=[type_colors[t] for t in TYPES], orientation='horizontal', edgecolor='black')
    ax_hist_y.tick_params(labelleft=False)
    ax_hist_y.set_xlabel("Count")

    fig.suptitle("Global Max vs Resolved-Region Max", y=0.92)
    fig.savefig(FIGURES_DIR / "fig_global_vs_good_max.png", dpi=100)
    print(f"Saved {FIGURES_DIR / 'fig_global_vs_good_max.png'}")


def plot_example(dataset, df):
    example_id = df['rec_id'].iloc[2]
    plot_recording(example_id, dataset, [4.3, 4.5], [-3000, 3000])


if __name__ == "__main__":
    dataset, missing = load_dataset()
    print(f"Loaded {len(dataset)} recordings ({len(missing)} missing annotations)")

    df = build_summary(dataset)
    plot_stacked_histogram_by_type(df)
    plot_pie(df)
    plot_global_vs_good_max(dataset)
    plot_example(dataset, df)
    plt.show()

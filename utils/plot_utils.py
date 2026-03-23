# Updated colors and descriptive labels

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from env import FIGURES_DIR


LABEL_COLORS = {
    0: "gray",
    1: "red",     # S1
    2: "orange",  # systole
    3: "blue",    # S2
    4: "green"    # diastole
}

LABEL_NAMES = {
    0: "unrecognized",
    1: "S1",
    2: "systole",
    3: "S2",
    4: "diastole"
}


def plot_segmented_signal_interactive(signal, y, sr, height=450, width=1100):
    """Return an interactive Plotly figure with color-coded segmentation."""
    t = np.arange(len(signal)) / sr
    fig = go.Figure()
    for label, color in LABEL_COLORS.items():
        mask = y == label
        if not mask.any():
            continue
        fig.add_trace(go.Scattergl(
            x=t[mask], y=signal[mask],
            mode="markers", marker=dict(size=2, color=color),
            name=LABEL_NAMES[label],
        ))
    fig.update_layout(
        height=height, width=width,
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=40, b=40),
        dragmode="zoom",
    )
    return fig


def plot_signal_on_ax(ax, signal, y, sr, s=1, show_labels=True):
    """Plot a color-coded signal on a given axes."""
    t = np.arange(len(signal)) / sr
    for label, color in LABEL_COLORS.items():
        mask = y == label
        ax.scatter(t[mask], signal[mask], s=s, c=color,
                   label=LABEL_NAMES[label] if show_labels else None)


def plot_recording(rec_id, dataset, xl=None, yl=None, sa_fig: bool = False):
    data = dataset[rec_id]
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_signal_on_ax(ax, data["signal"], data["y"], data["sr"])
    if xl:
        ax.set_xlim(xl)
    if yl:
        ax.set_ylim(yl)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Recording {rec_id} — segmentation")
    ax.legend(markerscale=6)
    fig.tight_layout()
    if sa_fig:
        path = FIGURES_DIR / f"fig_{rec_id}.png"
        fig.savefig(path, dpi=100)
        print(f"Saved {path}")


def plot_recording_before_and_after(rec_id, dataset_before, dataset_after,
                                    label="processing", xl=None):
    """Plot a single recording before and after any processing step."""
    sig_before = dataset_before[rec_id]['signal']
    sig_after = dataset_after[rec_id]['signal']
    sr = dataset_before[rec_id]['sr']
    t_before = np.arange(len(sig_before)) / sr
    t_after = np.arange(len(sig_after)) / sr

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t_before, sig_before, '.-', color='steelblue', linewidth=0.4, label=f"Before {label}")
    ax.plot(t_after, sig_after, '.-', color='tomato', linewidth=0.4, label=f"After {label}")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Recording {rec_id} — {label}")
    ax.legend()

    if xl:
        ax.set_xlim(xl)

    fig.tight_layout()
    path = FIGURES_DIR / f"fig_{label}_{rec_id}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig

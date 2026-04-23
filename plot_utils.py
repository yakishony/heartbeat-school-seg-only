import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from env import FIGURES_DIR, RATE


# Color mapping for each segmentation class
LABEL_COLORS = {
    0: "gray",  # unannotated
    1: "red",     # S1
    2: "orange",  # systole
    3: "blue",    # S2
    4: "green"    # diastole
}

# Display names for each segmentation class
LABEL_NAMES = {
    0: "unannotated",
    1: "S1",
    2: "systole",
    3: "S2",
    4: "diastole"
}

# for application
def plot_plain_signal_interactive(signal, sr, height=450, width=1100):
    """Plain amplitude-vs-time Plotly figure."""
    t = np.arange(len(signal)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=signal,
        mode="markers", marker=dict(size=2, color="steelblue"),
        name="signal",
    ))
    fig.update_layout(
        height=height, width=width,
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        margin=dict(l=50, r=20, t=40, b=40),
        dragmode="zoom",
    )
    return fig

# for application
def plot_segmented_signal_interactive(signal, y, sr, height=450, width=1100):
    """an interactive Plotly figure with color-coded segmentation."""
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


def plot_segmented_signal_on_ax(ax, signal, y, name=None, s=1, show_labels=True):
    """Plot a color-coded signal if segmented is True, otherwise plot the original signal on a given axes."""
    t = np.arange(len(signal)) / RATE
    if name is None:
        name = "segmented_signal"
    for label, color in LABEL_COLORS.items():
        mask = y == label
        ax.scatter(t[mask], signal[mask], s=s, c=color,
                   label=LABEL_NAMES[label] if show_labels else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{name} segmented")


def plot_recording(rec_id, dataset, name, xl=None, yl=None, sa_fig: bool = False):
    data = dataset[rec_id]
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_segmented_signal_on_ax(ax, data["signal"], data["y"])
    if xl:
        ax.set_xlim(xl)
    if yl:
        ax.set_ylim(yl)
    ax.set_title(f"Recording {rec_id} {name}")
    ax.legend(markerscale=6)
    fig.tight_layout()
    if sa_fig:
        path = FIGURES_DIR / f"fig_{rec_id}_{name}.png"
        fig.savefig(path, dpi=100)
        print(f"Saved {path}")


import numpy as np
import matplotlib.pyplot as plt
from get_data import load_dataset
from env import FIGURES_DIR, RATE
from utils.plot_utils import plot_signal_on_ax

dataset, _ = load_dataset()

rec_ids = list(dataset.keys())
np.random.seed(10)
sample_ids = np.random.choice(rec_ids, size=12, replace=False)

START = 4.3
DURATION = 3  # seconds
n_samples = int(DURATION * RATE)
n_start = int(START * RATE)

fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True, sharey=True)

for ax, rec_id in zip(axes.flat, sample_ids):
    data = dataset[rec_id]
    plot_signal_on_ax(ax, data["signal"][n_start:n_start+n_samples], data["y"][n_start:n_start+n_samples],
                      data["sr"], s=0.3)
    ax.set_title(rec_id, fontsize=8)
    ax.set_ylim(-30000, 30000)
    ax.tick_params(labelsize=7)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
axes[3, 0].set_ylabel("Amplitude")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, markerscale=8, fontsize=9)
fig.suptitle("12 Sample Recordings (first 3 s)", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIGURES_DIR / "fig_multi_recordings.png", dpi=150)
print(f"Saved {FIGURES_DIR / 'fig_multi_recordings.png'}")
plt.show()



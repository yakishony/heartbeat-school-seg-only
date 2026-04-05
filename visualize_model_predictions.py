"""
Load a trained checkpoint, predict on validation samples, plot ground-truth vs prediction.
"""
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from env import DATA_FOR_ML, FIGURES_DIR, RATE_DS, CATEGORY_NAMES, MURMUR_NAMES
from ML import load_all_ids
from utils.plot_utils import LABEL_COLORS, LABEL_NAMES
CHECKPOINT = "checkpoints/less_unrecognized_murmur_cls_model8/best.keras"
N_SAMPLES = 5
SEED = 7
GAP = 0.3  # gap between truth and prediction traces (in signal units)


def plot_truth_and_pred(ax, signal, y_true, y_pred, sr, rec_id):
    t = np.arange(len(signal)) / sr
    sig_range = signal.max() - signal.min()
    shift = sig_range + GAP # a shift that is adjusted to each recording
    for label, color in LABEL_COLORS.items():
        mask_true = y_true == label
        mask_pred = y_pred == label
        ax.scatter(t[mask_true], signal[mask_true], s=0.3, c=color, label=LABEL_NAMES[label])
        ax.scatter(t[mask_pred], signal[mask_pred] - shift, s=0.3, c=color)
    ax.axhline(signal.min() - GAP / 2, color="k", linewidth=0.3, linestyle="--")
    ax.text(0.01, 0.95, "Truth", fontsize=7, va="top", transform=ax.transAxes)
    ax.text(0.01, 0.45, "Prediction", fontsize=7, va="top", transform=ax.transAxes)
    ax.set_title(rec_id, fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)



def plot_confusion_matrix(model, batch_size=128):
    name = input("Enter a name for the confusion matrix figure: ")
    all_ids = load_all_ids()
    print(f"Predicting on {len(all_ids)} recordings...")
    all_true, all_pred = [], []
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        signals = [np.load(DATA_FOR_ML / "signals" / f"{rid}.npy") for rid in batch_ids]
        labels = [np.load(DATA_FOR_ML / "labels" / f"{rid}.npy") for rid in batch_ids]
        X = np.stack(signals)[..., np.newaxis]
        seg_preds, _ = model.predict(X, verbose=0)
        preds = seg_preds.argmax(axis=-1)
        all_true.append(np.concatenate(labels))
        all_pred.append(preds.ravel())
        print(f"  {min(i + batch_size, len(all_ids))}/{len(all_ids)}")
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CATEGORY_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    disp1 = ConfusionMatrixDisplay(cm_norm, display_labels=CATEGORY_NAMES)
    disp1.plot(ax=ax1, cmap="Blues", colorbar=False, values_format=".2f")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            disp1.text_[i][j].set_text(str(cm[i, j]))
    ax1.set_title("Counts")
    ConfusionMatrixDisplay(cm_norm, display_labels=CATEGORY_NAMES).plot(ax=ax2, cmap="Blues", values_format=".2f", colorbar=False)
    ax2.set_title("Normalized (row)")
    fig.suptitle(f"Confusion Matrix — {name}", fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / f"fig_confusion_matrix_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


def plot_murmur_confusion_matrix(model, batch_size=128):
    name = input("Enter a name for the murmur confusion matrix figure: ")
    all_ids = load_all_ids()
    print(f"Predicting murmur on {len(all_ids)} recordings...")
    all_true, all_pred = [], []
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        signals = [np.load(DATA_FOR_ML / "signals" / f"{rid}.npy") for rid in batch_ids]
        murmurs = [np.load(DATA_FOR_ML / "murmurs" / f"{rid}.npy") for rid in batch_ids]
        X = np.stack(signals)[..., np.newaxis]
        _, murmur_preds = model.predict(X, verbose=0)
        all_true.append(np.array(murmurs))
        all_pred.append(murmur_preds.argmax(axis=-1))
        print(f"  {min(i + batch_size, len(all_ids))}/{len(all_ids)}")
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(MURMUR_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    disp1 = ConfusionMatrixDisplay(cm_norm, display_labels=MURMUR_NAMES)
    disp1.plot(ax=ax1, cmap="Oranges", colorbar=False, values_format=".2f")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            disp1.text_[i][j].set_text(str(cm[i, j]))
    ax1.set_title("Counts")
    ConfusionMatrixDisplay(cm_norm, display_labels=MURMUR_NAMES).plot(ax=ax2, cmap="Oranges", values_format=".2f", colorbar=False)
    ax2.set_title("Normalized (row)")
    fig.suptitle(f"Confusion Matrix — Murmur ({name})", fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / f"fig_murmur_confusion_matrix_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


def main():
    model = keras.models.load_model(CHECKPOINT, compile=False)
    # print(f"Loaded model from {CHECKPOINT}")

    # all_ids = load_all_ids()
    # _, val_ids = train_test_split(all_ids, test_size=VAL_SPLIT, random_state=42)

    # rng = np.random.default_rng(SEED)
    # sample_ids = rng.choice(val_ids, size=N_SAMPLES, replace=False)

    # fig, axes = plt.subplots(N_SAMPLES, 1, figsize=(16, 3 * N_SAMPLES), sharex=True)

    # for i, rec_id in enumerate(sample_ids):
    #     signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id}.npy")
    #     y_true = np.load(DATA_FOR_ML / "labels" / f"{rec_id}.npy")

    #     pred = model.predict(signal.reshape(1, -1, 1), verbose=0)
    #     y_pred = pred.argmax(axis=-1).squeeze()

    #     plot_truth_and_pred(axes[i], signal, y_true, y_pred, RATE_DS, rec_id)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=5, markerscale=8, fontsize=9)
    # axes[-1].set_xlabel("Time (s)")
    # fig.suptitle("Ground Truth (top) vs Prediction (bottom)", fontsize=13, y=1.01)
    # fig.tight_layout(rect=[0, 0, 1, 0.97])

    # out = FIGURES_DIR / "fig_predictions.png"
    # fig.savefig(out, dpi=150, bbox_inches="tight")
    # print(f"Saved {out}")
    # plt.show()

    # Confusion matrices on all data
    plot_confusion_matrix(model)
    plot_murmur_confusion_matrix(model)

if __name__ == "__main__":
    main()

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
CHECKPOINT = "checkpoints/normalization_pre_rec_split_regular_model10/best.keras"


def plot_training_curves(history, save_path=None):
    h = history.history
    epochs = list(range(1, len(h["loss"]) + 1))

    is_multi = "seg_accuracy" in h

    if is_multi:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(epochs, h["seg_accuracy"], 'o-', label='Train')
        axes[0, 0].plot(epochs, h["val_seg_accuracy"], 'o-', label='Val')
        axes[0, 0].set_title('Segmentation Accuracy')

        axes[0, 1].plot(epochs, h["seg_loss"], 'o-', label='Train')
        axes[0, 1].plot(epochs, h["val_seg_loss"], 'o-', label='Val')
        axes[0, 1].set_title('Segmentation Loss')

        axes[0, 2].axis('off')

        axes[1, 0].plot(epochs, h["murmur_accuracy"], 'o-', label='Train')
        axes[1, 0].plot(epochs, h["val_murmur_accuracy"], 'o-', label='Val')
        axes[1, 0].set_title('Murmur Accuracy')

        axes[1, 1].plot(epochs, h["murmur_loss"], 'o-', label='Train')
        axes[1, 1].plot(epochs, h["val_murmur_loss"], 'o-', label='Val')
        axes[1, 1].set_title('Murmur Loss')

        axes[1, 2].plot(epochs, h["murmur_recall_present"], 'o-', label='Train Recall')
        axes[1, 2].plot(epochs, h["val_murmur_recall_present"], 'o-', label='Val Recall')
        axes[1, 2].plot(epochs, h["murmur_precision_present"], 's--', label='Train Precision')
        axes[1, 2].plot(epochs, h["val_murmur_precision_present"], 's--', label='Val Precision')
        axes[1, 2].set_title('Murmur Present: Recall & Precision')

        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            if ax.get_legend_handles_labels()[1]:
                ax.legend()
            ax.grid(True)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, h["accuracy"], 'o-', label='Train')
        ax1.plot(epochs, h["val_accuracy"], 'o-', label='Val')
        ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)

        ax2.plot(epochs, h["loss"], 'o-', label='Train')
        ax2.plot(epochs, h["val_loss"], 'o-', label='Val')
        ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def annotate_precision_recall(ax, cm):
    """Add per-class recall (right of rows), precision (below columns) and accuracy (bottom right) to confusion matrix."""
    n = cm.shape[0]
    recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    precision = np.diag(cm) / cm.sum(axis=0).clip(min=1)

    for i in range(n):
        ax.text(n - 0.5 + 0.4, i, f"{recall[i]:.2f}",
                ha="left", va="center", fontsize=8, fontweight="bold", color="green")
    for j in range(n):
        ax.text(j, n - 0.5 + 0.4, f"{precision[j]:.2f}",
                ha="center", va="top", fontsize=8, fontweight="bold", color="purple")

    ax.text(n - 0.5 + 0.4, -0.7, "Recall", ha="left", va="center",
            fontsize=8, fontstyle="italic", color="green")
    ax.text(-0.7, n - 0.5 + 0.4, "Prec.", ha="center", va="top",
            fontsize=8, fontstyle="italic", color="purple")

    accuracy = np.diag(cm).sum() / cm.sum()
    ax.text(n - 0.5 + 0.4, n - 0.5 + 0.4, f"Acc:\n{accuracy:.2f}",
            ha="left", va="top", fontsize=8, fontweight="bold", color="darkred")
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

    disp = ConfusionMatrixDisplay(cm_norm, display_labels=CATEGORY_NAMES)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False)
    annotate_precision_recall(disp.ax_, cm)
    disp.ax_.set_title(f"Normalized Confusion Matrix — {name}", fontsize=13)
    disp.figure_.set_size_inches(8, 5.5)
    disp.figure_.tight_layout()

    out = FIGURES_DIR / f"fig_confusion_matrix_{name}.png"
    disp.figure_.savefig(out, dpi=150, bbox_inches="tight")
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

    disp = ConfusionMatrixDisplay(cm_norm, display_labels=MURMUR_NAMES)
    disp.plot(cmap="Oranges", values_format=".2f", colorbar=False)
    annotate_precision_recall(disp.ax_, cm)
    disp.ax_.set_title(f"Normalized Confusion Matrix — Murmur ({name})", fontsize=13)
    disp.figure_.set_size_inches(7, 4.5)
    disp.figure_.tight_layout()

    out = FIGURES_DIR / f"fig_murmur_confusion_matrix_{name}.png"
    disp.figure_.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()



def main():
    model = keras.models.load_model(CHECKPOINT, compile=False)
    # Confusion matrices on all data
    plot_confusion_matrix(model)
    plot_murmur_confusion_matrix(model)

if __name__ == "__main__":
    main()

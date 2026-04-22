import json
from pathlib import Path
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ML import BATCH_SIZE
from env import DATA_FOR_ML, FIGURES_DIR, CATEGORY_NAMES, RATE
from ML_utils import load_all_ids, split_ids
from plot_utils import LABEL_COLORS, LABEL_NAMES
CHECKPOINT = Path("checkpoints/model_13/best.keras")

def plot_training_curves(history, name=None):
    h = history if isinstance(history, dict) else history.history # if history is the dict from json - no need to preform history.history on it
    epochs = list(range(1, len(h["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, h["accuracy"], 'o-', label='Train')
    ax1.plot(epochs, h["val_accuracy"], 'o-', label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, h["loss"], 'o-', label='Train')
    ax2.plot(epochs, h["val_loss"], 'o-', label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Training Curves {name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
    
    out = FIGURES_DIR / f"fig_training_curves_{name}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

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

GAP = 0.3  # vertical gap between truth and prediction traces (in signal amplitude units)
def plot_truth_and_pred(ax, signal, y_true, y_pred, rec_id):
    t = np.arange(len(signal)) / RATE
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


SPLIT_NAMES = ("all", "train", "val", "test")

def _get_ids_for_split(data_path, split):
    if split == "all":
        return load_all_ids(data_path)
    train_ids, val_ids, test_ids = split_ids()
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    parts = split.split("+") if "+" in split else [split]
    ids = []
    for p in parts:
        assert p in split_map, f"unknown split '{p}', use train/val/test/all"
        ids.extend(split_map[p])
    return ids


def plot_confusion_matrix(model, data_path, name=None, split="all", batch_size=BATCH_SIZE):
    if name is None:
        name = input("Enter a name for the confusion matrix figure: ")
    ids = _get_ids_for_split(data_path, split)
    print(f"Predicting on {len(ids)} recordings (split={split})...")
    all_true, all_pred = [], []

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        signals = [np.load(data_path / "signals" / f"{id}.npy") for id in batch_ids] # a len=batch_size list of signals with SAMPLES_NUM len
        labels = [np.load(data_path / "labels" / f"{id}.npy") for id in batch_ids] # a len=batch_size list of labels with SAMPLES_NUM len
        X = np.stack(signals)[..., np.newaxis] # [batch_size, samples_num, 1] - transforming into a form that the  Keras conv1D models expect.(matrix) 
        seg_preds = model.predict(X, verbose=0) # [batch_size, samples_num, num_seg_classes] - num_seg_classes - softmax return probabilities
        preds = seg_preds.argmax(axis=-1) # [batch_size, samples_num] - the class with the highest probability
        all_true.append(np.concatenate(labels)) # concatenate all arrays in the list into one 1D array
        all_pred.append(preds.reshape(-1)) # reshape the array to 1D array
        print(f"  {min(i + batch_size, len(ids))}/{len(ids)}")
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CATEGORY_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(cm_norm, display_labels=CATEGORY_NAMES)
    disp.plot(cmap="Blues", values_format=".2f", colorbar=False)
    annotate_precision_recall(disp.ax_, cm)
    disp.ax_.set_title(f"Normalized Confusion Matrix — {name} [{split}]", fontsize=13)
    disp.figure_.set_size_inches(8, 5.5)
    disp.figure_.tight_layout()

    out = FIGURES_DIR / f"fig_confusion_matrix_{name}_{split}.png"
    disp.figure_.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()
    return all_true, all_pred


def main():
    pass
if __name__ == "__main__":
    main()

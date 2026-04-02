import matplotlib.pyplot as plt


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

import matplotlib.pyplot as plt


def plot_training_curves(history, save_path=None):
    h = history.history
    epochs = list(range(1, len(h["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, h["accuracy"], 'o-', label='Train')
    ax1.plot(epochs, h["val_accuracy"], 'o-', label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Epochs'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, h["loss"], 'o-', label='Train')
    ax2.plot(epochs, h["val_loss"], 'o-', label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Epochs'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

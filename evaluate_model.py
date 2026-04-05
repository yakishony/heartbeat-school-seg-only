import argparse
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


def evaluate_best(run_name: str):
    """Build fresh model, load weights from best.keras, evaluate on train/val/test."""
    from ML import (
        build_model, split_ids, make_tf_dataset, weighted_murmur_loss,
        SparseRecall, SparsePrecision, BATCH_SIZE, SEG_LOSS_WEIGHT, MURMUR_LOSS_WEIGHT,
    )
    from env import DATA_FOR_ML

    checkpoint_dir = DATA_FOR_ML.parent / "checkpoints" / run_name
    best_path = checkpoint_dir / "best.keras"
    if not best_path.exists():
        raise FileNotFoundError(f"No best.keras in {checkpoint_dir}")

    model = build_model()
    model.load_weights(best_path)
    model.compile(
        loss={"seg": "sparse_categorical_crossentropy", "murmur": weighted_murmur_loss},
        loss_weights={"seg": SEG_LOSS_WEIGHT, "murmur": MURMUR_LOSS_WEIGHT},
        metrics={
            "seg": "accuracy",
            "murmur": [
                "accuracy",
                SparseRecall(class_id=0, name="recall_present"),
                SparsePrecision(class_id=0, name="precision_present"),
            ],
        },
    )

    train_ids, val_ids, test_ids = split_ids()
    for split_name, ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        ds = make_tf_dataset(ids, BATCH_SIZE, shuffle=False)
        res = model.evaluate(ds, return_dict=True, verbose=0)
        print(f"\n── {split_name} ({len(ids)} samples) ──")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")


def list_runs():
    from env import DATA_FOR_ML
    checkpoint_root = DATA_FOR_ML.parent / "checkpoints"
    runs = sorted(p.name for p in checkpoint_root.iterdir()
                  if p.is_dir() and (p / "best.keras").exists())
    if not runs:
        print("No runs with best.keras found.")
    else:
        print("Available runs:")
        for i, r in enumerate(runs, 1):
            print(f"  {i}. {r}")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", nargs="?", default=None,
                        help="Run name to evaluate (omit to list available runs)")
    args = parser.parse_args()
    if args.run_name is None:
        runs = list_runs()
        if runs:
            choice = input("Enter run number or name: ").strip()
            args.run_name = runs[int(choice) - 1] if choice.isdigit() else choice
        else:
            exit(1)
    evaluate_best(args.run_name)

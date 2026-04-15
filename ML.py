# keras: high-level API on top of TF for defining and training neural networks
import json
import keras
# layers: building blocks (Conv1D, Dense, etc.); Model: class that wires layers into a trainable network
from keras import layers, Model
# Path: object-oriented filesystem paths (e.g. Path("a") / "b" → "a/b")
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from ML_utils import make_tf_dataset, split_ids, compute_seg_class_weights, weighted_loss
from env import (
    DATA_FOR_ML, CLASSES
)
from split_data_into_fixed_length_recordings import SAMPLES_NUM

# ── Hyperparameters ──
NUM_SEG_CLASSES = len(CLASSES)

BATCH_SIZE = 64
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-3
POOL_1 = 4        
POOL_2 = 4  
ReduceLROnPlateau_PATIENCE = 3       
ReduceLROnPlateau_FACTOR = 0.6
ReduceLROnPlateau_MIN_LR = 1e-6

# ── Model ──
def build_model(seq_len=SAMPLES_NUM, num_seg_classes=NUM_SEG_CLASSES):
    inp = layers.Input(shape=(seq_len, 1))

    # ── Encoder block 1 ──
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    skip1 = x                                        # (2000, 64)
    x = layers.MaxPool1D(pool_size=POOL_1)(x)        # (500, 64)

    # ── Encoder block 2 ──
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    skip2 = x                                        # (500, 128)
    x = layers.MaxPool1D(pool_size=POOL_2)(x)        # (125, 128)

    # ── Bottleneck: two BiGRU ──
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    )(x)
                                                  # (125, 256)
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    )(x)
                                                  # (125, 256)
    # ── Decoder block 1 ──
    x = layers.UpSampling1D(size=POOL_2)(x)           # (500, 256)
    x = layers.Concatenate()([x, skip2])               # (500, 384)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x) # (500, 64)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # ── Decoder block 2 ──
    x = layers.UpSampling1D(size=POOL_1)(x)            # (2000, 64)
    x = layers.Concatenate()([x, skip1])                # (2000, 128)

    seg_out = layers.Dense(
        num_seg_classes, activation="softmax", name="seg"
    )(x)                                                # (2000, 5)

    return Model(inputs=inp, outputs=seg_out)

def main(resume_checkpoint_path: str | None = None, from_scratch: bool = False):
    train_ids, val_ids, test_ids = split_ids()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    train_ds = make_tf_dataset(train_ids, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_ids, BATCH_SIZE, shuffle=False)
    test_ds = make_tf_dataset(test_ids, BATCH_SIZE, shuffle=False)
    seg_weights_tensor = compute_seg_class_weights(train_ids)

    initial_epoch = 0
    if resume_checkpoint_path:
        print(f"Loading model from: {resume_checkpoint_path}")
        model = keras.models.load_model(resume_checkpoint_path)
        if from_scratch:
            run_name = input("Enter a name for this training run: ").strip()
        else:
            initial_epoch = int(Path(resume_checkpoint_path).stem.split("_")[-1])
            run_name = Path(resume_checkpoint_path).parent.name
    else:
        model = build_model()
        run_name = input("Enter a name for this training run: ").strip()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), 
            loss=weighted_loss(seg_weights_tensor),
            metrics=["accuracy"]
        )
    model.summary()

    checkpoint_dir = DATA_FOR_ML.parent / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {checkpoint_dir} (resuming from epoch {initial_epoch})")

    # save model description to a txt file(with the ML.py and run_pipeline_analysing_data_step_2.py code)
    model_description = checkpoint_dir / "model_description.txt"
    if not model_description.exists():
        description = input("Enter a description for this run (or press Enter to skip): ").strip()
        with open(model_description, "w") as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Description: {description}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("ML.py\n")
            f.write("=" * 60 + "\n")
            f.write(Path(__file__).read_text())
            f.write("\n" + "=" * 60 + "\n")
            f.write("run_pipeline_analysing_data_step_2.py\n")
            f.write("=" * 60 + "\n")
            f.write((Path(__file__).parent / "run_pipeline_analysing_data_step_2.py").read_text())
        print(f"Run notes saved → {model_description}")

    metrics_csv = checkpoint_dir / "metrics.csv"
    callbacks = [
        keras.callbacks.CSVLogger(str(metrics_csv), append=(initial_epoch > 0)), # append=True(initial_epoch > 0) - append to the existing file instead of overwriting it
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"), save_freq="epoch"),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_dir / "best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=ReduceLROnPlateau_FACTOR, patience=ReduceLROnPlateau_PATIENCE, min_lr=ReduceLROnPlateau_MIN_LR, verbose=1), # verbose controlls how much info is being printed out
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    results = model.evaluate(test_ds, return_dict=True) # note that if the model ran all 
    # the way to EPOCHS num, it will return the results of the last epoch - not necessarily the best one
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    # save history and results to a json file
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history.history, f)
    with open(checkpoint_dir / "results.json", "w") as f:
        json.dump(results, f)
    from visualize_model import plot_training_curves
    plot_training_curves(history, name=run_name)
    
    # plot confusion matrix 
    from visualize_model import plot_confusion_matrix
    plot_confusion_matrix(model)

if __name__ == "__main__":
    # To train fresh:          main()
    # To resume training:      main(resume_checkpoint="checkpoints/<run>/epoch_NNN.keras")
    # To reload & retrain:     main(resume_checkpoint="checkpoints/<run>/epoch_NNN.keras", from_scratch=True)
    main()

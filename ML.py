"""
Conv1D-BiGRU encoder-decoder for PCG segmentation.
Architecture: Conv1D+MaxPool (2000→125) → BiGRU → Upsample+Conv1D (125→2000) with skip connections
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.model_selection import train_test_split
from evaluate_model import plot_training_curves
from env import DATA_FOR_ML, CLASSES
from split_data_into_fixed_length_recordings import SAMPLES_NUM

# ── Hyperparameters ──
NUM_CLASSES = len(CLASSES)     # 0=background, 1=S1, 2=systolic, 3=S2, 4=diastolic
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
POOL_1 = 4           # 2000 → 500
POOL_2 = 4           # 500 → 125


# ── Data loading ──
def load_all_ids():
    signal_dir = DATA_FOR_ML / "signals"
    return sorted(p.stem for p in signal_dir.glob("*.npy"))


def _load_npy_pair(rec_id_tensor):
    rec_id = rec_id_tensor.numpy().decode("utf-8")
    signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id}.npy")
    label = np.load(DATA_FOR_ML / "labels" / f"{rec_id}.npy")
    return signal.reshape(-1, 1), label


def load_pair_wrapper(rec_id):
    signal, label = tf.py_function(
        _load_npy_pair,
        [rec_id],
        [tf.float32, tf.int64],
    )
    signal.set_shape([SAMPLES_NUM, 1])
    label.set_shape([SAMPLES_NUM])
    return signal, label


def make_tf_dataset(rec_ids, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(rec_ids)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(rec_ids), reshuffle_each_iteration=True)
    ds = ds.map(load_pair_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model ──
def build_model(seq_len=SAMPLES_NUM, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(seq_len, 1))

    # Encoder (skips saved before pooling, U-Net style)
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    skip1 = x                                        # (2000, 64)
    x = layers.MaxPool1D(pool_size=POOL_1)(x)        # (500, 64)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    skip2 = x                                        # (500, 128)
    x = layers.MaxPool1D(pool_size=POOL_2)(x)        # (125, 128)

    # BiGRU
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # Decoder (upsample then concat, U-Net style)
    x = layers.UpSampling1D(size=POOL_2)(x)          # (500, 256)
    x = layers.Concatenate()([x, skip2])              # (500, 384)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.UpSampling1D(size=POOL_1)(x)          # (2000, 64)
    x = layers.Concatenate()([x, skip1])              # (2000, 128)
    x = layers.Conv1D(num_classes, 1, activation="softmax")(x)

    return Model(inputs=inp, outputs=x)


# ── Train ──
def main(resume_checkpoint: str | None = None):
    all_ids = load_all_ids()
    # Split into train (70%), val (15%), test (15%)
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.30, random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=42
    )
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_ds = make_tf_dataset(train_ids, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_ids, BATCH_SIZE, shuffle=False)

    initial_epoch = 0
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
        model = keras.models.load_model(resume_checkpoint)
        initial_epoch = int(Path(resume_checkpoint).stem.split("_")[-1])
        run_name = Path(resume_checkpoint).parent.name
    else:
        model = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        run_name = input("Enter a name for this training run: ").strip()

    model.summary()

    checkpoint_dir = DATA_FOR_ML.parent / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {checkpoint_dir} (resuming from epoch {initial_epoch})")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
            save_freq="epoch",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    plot_training_curves(history, save_path=f"figures/fig_training_curves_{run_name}.png")

    test_signals = np.stack([np.load(DATA_FOR_ML / "signals" / f"{rid}.npy").reshape(-1, 1) for rid in test_ids])
    test_labels = np.stack([np.load(DATA_FOR_ML / "labels" / f"{rid}.npy") for rid in test_ids])
    test_loss, test_accuracy = model.evaluate(test_signals, test_labels)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)

"""
Conv1D-BiGRU encoder-decoder for PCG segmentation.
Architecture: Conv1D+MaxPool (4000→250) → BiGRU → Upsample+Conv1D (250→4000)
"""
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
POOL_1 = 4           # 4000 → 1000
POOL_2 = 4           # 1000 → 250


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

    # Encoder
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPool1D(pool_size=POOL_1)(x)                    # 4000 → 1000

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPool1D(pool_size=POOL_2)(x)                    # 1000 → 250

    # BiGRU on reduced sequence
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)                                                          # (250, 256)

    # Decoder
    x = layers.UpSampling1D(size=POOL_2)(x)                      # 250 → 1000
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.UpSampling1D(size=POOL_1)(x)                      # 1000 → 4000
    x = layers.Conv1D(num_classes, 1, activation="softmax")(x)   # (4000, 5)

    return Model(inputs=inp, outputs=x)


# ── Train ──
def main():
    all_ids = load_all_ids()
    train_ids, val_ids = train_test_split(
        all_ids, test_size=VAL_SPLIT, random_state=42
    )
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_ds = make_tf_dataset(train_ids, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_ids, BATCH_SIZE, shuffle=False)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), # if the square-root of the sum of all gradients is greater than the clipnorm, it will be clipped to 1
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    run_name = input("Enter a name for this training run: ").strip()
    checkpoint_dir = DATA_FOR_ML.parent / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
            save_freq="epoch",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),# if the val - loss does not improve for 2 epochs, the learning rate will be reduced by half to take more careful steps
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1
        ), # if the val - loss does not improve for 5 epochs, the training will be stopped
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_training_curves(history, save_path=f"figures/fig_training_curves_{run_name}.png")


if __name__ == "__main__":
    main()

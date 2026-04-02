"""
Conv1D-BiGRU dual-head model for PCG analysis.
Head 1 (seg):    per-sample segmentation  (S1/S2/systolic/diastolic/background)
Head 2 (murmur): recording-level murmur classification (Present/Absent/Unknown)
Architecture: shared CNN+BiGRU encoder → U-Net decoder (seg) + GlobalAvgPool→Dense (murmur)
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.model_selection import GroupShuffleSplit

from evaluate_model import plot_training_curves
from env import (
    DATA_FOR_ML, CLASSES, NUM_MURMUR_CLASSES, MURMUR_CLASS_WEIGHTS,
)
from split_data_into_fixed_length_recordings import SAMPLES_NUM

_MURMUR_WEIGHTS_TENSOR = tf.constant(
    [MURMUR_CLASS_WEIGHTS[i] for i in range(NUM_MURMUR_CLASSES)],
    dtype=tf.float32,
)


class SparseRecall(keras.metrics.Metric):
    """Recall for a single class given sparse integer labels and softmax output."""
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name or f"recall_c{class_id}", **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight("tp", initializer="zeros")
        self.fn = self.add_weight("fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)
        is_class = tf.equal(y_true, self.class_id)
        self.tp.assign_add(tf.reduce_sum(tf.cast(is_class & tf.equal(pred, self.class_id), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(is_class & tf.not_equal(pred, self.class_id), tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-7)

    def reset_state(self):
        self.tp.assign(0)
        self.fn.assign(0)


class SparsePrecision(keras.metrics.Metric):
    """Precision for a single class given sparse integer labels and softmax output."""
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name or f"precision_c{class_id}", **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight("tp", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)
        pred_class = tf.equal(pred, self.class_id)
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_class & tf.equal(y_true, self.class_id), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_class & tf.not_equal(y_true, self.class_id), tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-7)

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)


# ── Hyperparameters ──
NUM_SEG_CLASSES = len(CLASSES)
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
POOL_1 = 4           # 2000 → 500
POOL_2 = 4           # 500 → 125
SEG_LOSS_WEIGHT = 1.0
MURMUR_LOSS_WEIGHT = 0.5


# ── Data loading ──
def load_all_ids():
    signal_dir = DATA_FOR_ML / "signals"
    return sorted(p.stem for p in signal_dir.glob("*.npy"))


def _load_triplet(rec_id_tensor):
    rec_id = rec_id_tensor.numpy().decode("utf-8")
    signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id}.npy").reshape(-1, 1)
    seg_label = np.load(DATA_FOR_ML / "labels" / f"{rec_id}.npy")
    murmur = np.load(DATA_FOR_ML / "murmurs" / f"{rec_id}.npy").reshape(())
    return signal, seg_label, murmur


def load_triplet_wrapper(rec_id):
    signal, seg_label, murmur = tf.py_function(
        _load_triplet,
        [rec_id],
        [tf.float32, tf.int64, tf.int64],
    )
    signal.set_shape([SAMPLES_NUM, 1])
    seg_label.set_shape([SAMPLES_NUM])
    murmur.set_shape([])
    return signal, {"seg": seg_label, "murmur": murmur}


def make_tf_dataset(rec_ids, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(rec_ids)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(rec_ids), reshuffle_each_iteration=True)
    ds = ds.map(load_triplet_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model ──
def build_model(seq_len=SAMPLES_NUM, num_seg_classes=NUM_SEG_CLASSES,
                num_murmur_classes=NUM_MURMUR_CLASSES):
    inp = layers.Input(shape=(seq_len, 1))

    # Shared encoder
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

    # Shared BiGRU bottleneck
    bottleneck = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)                                              # (125, 256)

    # ── Head 1: Segmentation decoder (U-Net style) ──
    seg = layers.UpSampling1D(size=POOL_2)(bottleneck) # (500, 256)
    seg = layers.Concatenate()([seg, skip2])            # (500, 384)
    seg = layers.Conv1D(64, 5, padding="same", activation="relu")(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Dropout(0.2)(seg)

    seg = layers.UpSampling1D(size=POOL_1)(seg)        # (2000, 64)
    seg = layers.Concatenate()([seg, skip1])            # (2000, 128)
    seg_out = layers.Conv1D(
        num_seg_classes, 1, activation="softmax", name="seg"
    )(seg)                                              # (2000, 5)

    # ── Head 2: Murmur classification ──
    cls = layers.GlobalAveragePooling1D()(bottleneck)   # (256,)
    cls = layers.Dense(64, activation="relu")(cls)
    cls = layers.Dropout(0.3)(cls)
    cls_out = layers.Dense(
        num_murmur_classes, activation="softmax", name="murmur"
    )(cls)                                              # (3,)

    return Model(inputs=inp, outputs=[seg_out, cls_out])


def weighted_murmur_loss(y_true, y_pred):
    """Sparse categorical crossentropy with per-class weights for murmur head."""
    weights = tf.gather(_MURMUR_WEIGHTS_TENSOR, tf.cast(y_true, tf.int32))
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return loss * weights


# ── Train ──
def main(resume_checkpoint: str | None = None):
    all_ids = load_all_ids()
    patient_ids = [rid.split("_")[0] for rid in all_ids]

    # Patient-level split: train 70%, val 15%, test 15%
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(all_ids, groups=patient_ids))
    train_ids = [all_ids[i] for i in train_idx]

    temp_ids_list = [all_ids[i] for i in temp_idx]
    temp_patients = [patient_ids[i] for i in temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_ids_list, groups=temp_patients))
    val_ids = [temp_ids_list[i] for i in val_idx]
    test_ids = [temp_ids_list[i] for i in test_idx]

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
            loss={
                "seg": "sparse_categorical_crossentropy",
                "murmur": weighted_murmur_loss,
            },
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
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
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

    # ── Test evaluation ──
    test_ds = make_tf_dataset(test_ids, BATCH_SIZE, shuffle=False)
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)

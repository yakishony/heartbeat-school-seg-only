# Path: object-oriented filesystem paths (e.g. Path("a") / "b" → "a/b")
from pathlib import Path
# numpy: numerical array library used for loading .npy files
import numpy as np
# tensorflow: the deep-learning framework that runs training, graphs, and GPU ops
import tensorflow as tf
# keras: high-level API on top of TF for defining and training neural networks
import keras
# layers: building blocks (Conv1D, Dense, etc.); Model: class that wires layers into a trainable network
from keras import layers, Model
# GroupShuffleSplit: splits data into train/test while keeping all samples
# from the same group (here, patient) on the same side of the split,
# so the model is never tested on data from a patient it trained on.
from sklearn.model_selection import GroupShuffleSplit

from env import (
    DATA_FOR_ML, CLASSES, NUM_MURMUR_CLASSES,
)
from split_data_into_fixed_length_recordings import SAMPLES_NUM

# Recall = TP / (TP + FN) — "of all actual positives, how many did we catch?"
# We need a custom class because Keras built-in recall doesn't natively handle
# sparse labels with multi-class softmax output for a single target class.
class SparseRecall(keras.metrics.Metric):
    """Recall for a single class given sparse integer labels and softmax output."""
    def __init__(self, class_id, name=None, **kwargs):
        # super().__init__: call the parent Metric class constructor to register this metric
        super().__init__(name=name or f"recall_c{class_id}", **kwargs)
        self.class_id = class_id
        # add_variable: creates a persistent scalar (shape=()) that survives across batches,
        # initialized to 0. We accumulate true-positives and false-negatives here.
        self.tp = self.add_variable(name="tp", shape=(), initializer="zeros")
        self.fn = self.add_variable(name="fn", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # argmax: picks the index of the highest probability from softmax output → predicted class
        # tf.cast: converts the result to int64 so it matches y_true's dtype
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        # squeeze: removes extra dimensions of size 1 (e.g. shape (N,1) → (N,))
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)
        # is_class: boolean mask — True where the ground-truth label equals our target class
        is_class = tf.equal(y_true, self.class_id)
        # TP: ground truth IS this class AND prediction IS this class
        # reduce_sum: sums all True values across the batch into a single count
        self.tp.assign(self.tp + tf.reduce_sum(tf.cast(is_class & tf.equal(pred, self.class_id), tf.float32)))
        # FN: ground truth IS this class BUT prediction is NOT this class (we missed it)
        self.fn.assign(self.fn + tf.reduce_sum(tf.cast(is_class & tf.not_equal(pred, self.class_id), tf.float32)))

    def result(self):
        # 1e-7: tiny number to prevent division by zero when TP + FN = 0
        return self.tp / (self.tp + self.fn + 1e-7)

    def get_config(self):
        # get_config: returns a dict of constructor args so Keras can serialize/save the metric
        return {**super().get_config(), "class_id": self.class_id}

    def reset_state(self):
        # Called at the start of each epoch to zero out the counters
        self.tp.assign(0)
        self.fn.assign(0)


# Precision = TP / (TP + FP) — "of all samples we predicted as this class, how many actually were?"
# Same structure as SparseRecall but counts false-positives instead of false-negatives.
class SparsePrecision(keras.metrics.Metric):
    """Precision for a single class given sparse integer labels and softmax output."""
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name or f"precision_c{class_id}", **kwargs)
        self.class_id = class_id
        self.tp = self.add_variable(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_variable(name="fp", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)
        # pred_class: boolean mask — True where the MODEL PREDICTED this class
        # (contrast with SparseRecall which masks on ground truth)
        pred_class = tf.equal(pred, self.class_id)
        # TP: we predicted this class AND it really is this class
        self.tp.assign(self.tp + tf.reduce_sum(tf.cast(pred_class & tf.equal(y_true, self.class_id), tf.float32)))
        # FP: we predicted this class BUT it's actually a different class (false alarm)
        self.fp.assign(self.fp + tf.reduce_sum(tf.cast(pred_class & tf.not_equal(y_true, self.class_id), tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-7)

    def get_config(self):
        return {**super().get_config(), "class_id": self.class_id}

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)


# ── Hyperparameters ──
NUM_SEG_CLASSES = len(CLASSES)

BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
POOL_1 = 4        
POOL_2 = 4         
SEG_LOSS_WEIGHT = 1.0
MURMUR_LOSS_WEIGHT = 0.1


# ── Data loading ──
# Overall flow: scan directory for rec IDs → split into train/val/test →
#  build a tf.data.Dataset that lazily loads .npy files one at a time.

def load_all_ids():
    """Return a sorted list of all recording ID strings available on disk."""
    signal_dir = DATA_FOR_ML / "signals"
    # glob("*.npy"): finds every file ending with .npy in the directory.
    # path.stem: strips the directory and extension, leaving just the filename
    #   (e.g. Path("signals/rec_001.npy").stem → "rec_001").
    # sorted(): puts the IDs in alphabetical order for reproducibility.
    return sorted(path.stem for path in signal_dir.glob("*.npy"))


def _load_triplet(rec_id_tensor):
    """Load one recording's three .npy files. Called via tf.py_function (see wrapper below)."""
    # rec_id arrives as a TF string tensor (a TF wrapper around bytes).
    # .numpy() extracts the raw bytes; .decode("utf-8") converts to a normal Python string.
    rec_id_str = rec_id_tensor.numpy().decode("utf-8")
    # np.load: reads a .npy file into a numpy array.
    # reshape(-1, 1): turns a flat 1-D array of shape (2000,) into (2000, 1)
    #   because Conv1D expects input shape (time_steps, channels).
    signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id_str}.npy").reshape(-1, 1)
    # seg_label: integer label per time-step (shape (2000,)), values 0–4
    seg_label = np.load(DATA_FOR_ML / "labels" / f"{rec_id_str}.npy")
    # murmur: a single integer (0, 1, or 2). reshape(()) makes it a scalar (shape ())
    #   instead of a 1-element array, which is what the loss function expects.
    murmur = np.load(DATA_FOR_ML / "murmurs" / f"{rec_id_str}.npy").reshape(())
    return signal, seg_label, murmur


def load_triplet_wrapper(rec_id):
    """Bridge between the TF data pipeline and our plain-Python _load_triplet loader."""
    # tf.py_function: lets us call any regular Python function (here _load_triplet)
    #   from inside a TF data pipeline. TF pipelines normally only run TF ops,
    #   but we need plain Python to call np.load / do file I/O.
    # [rec_id]: the arguments to pass to _load_triplet.
    # [tf.float32, tf.int64, tf.int64]: the dtypes of the three return values
    #   (signal is float, seg_label and murmur are integers).
    signal, seg_label, murmur = tf.py_function(
        _load_triplet,
        [rec_id],
        [tf.float32, tf.int64, tf.int64],
    )
    # tf.py_function erases shape information (TF doesn't know what shapes
    # the Python function will return). set_shape tells TF the exact shape
    # so that downstream ops (batching, model layers) can validate dimensions.
    signal.set_shape([SAMPLES_NUM, 1])    # (2000, 1)
    seg_label.set_shape([SAMPLES_NUM])    # (2000,)
    murmur.set_shape([])                  # scalar
    # Return format expected by model.fit: (inputs, targets_dict).
    # The dict keys "seg" and "murmur" must match the layer names in the model
    # so Keras routes each target to the correct output head and loss function.
    return signal, {"seg": seg_label, "murmur": murmur}


def load_pair_wrapper(rec_id):
    """Like load_triplet_wrapper but returns (signal, seg_label) only — no murmur."""
    signal, seg_label, _ = tf.py_function(
        _load_triplet,
        [rec_id],
        [tf.float32, tf.int64, tf.int64],
    )
    signal.set_shape([SAMPLES_NUM, 1])
    seg_label.set_shape([SAMPLES_NUM])
    return signal, seg_label


def make_tf_dataset(rec_ids, batch_size=BATCH_SIZE, shuffle=True, seg_only=False):
    # from_tensor_slices: takes a list and creates a dataset that yields one element at a time.
    # Here each element is a single rec_id string (e.g. "rec_001").
    ds = tf.data.Dataset.from_tensor_slices(rec_ids)
    if shuffle:
        # shuffle: randomizes the order elements are yielded.
        # buffer_size: how many elements TF holds in memory to pick randomly from.
        #   A smaller buffer would only shuffle within a sliding window, biasing order.
        # reshuffle_each_iteration: when True, produces a fresh random order every epoch
        #   (otherwise the same shuffled order repeats).
        ds = ds.shuffle(buffer_size=len(rec_ids), reshuffle_each_iteration=True)
    # map: applies a function to each element, transforming each rec_id string
    #   into a (signal, {seg_label, murmur}) tuple by loading the .npy files.
    # num_parallel_calls=AUTOTUNE: lets TF decide how many elements to process
    #   in parallel (using multiple CPU threads) for speed.
    loader = load_pair_wrapper if seg_only else load_triplet_wrapper
    ds = ds.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
    # batch: collects individual samples into groups of batch_size,
    #   so the model receives e.g. 64 samples at once per training step.
    # prefetch: loads the next batch in the background while the GPU trains
    #   on the current batch, so data loading doesn't become a bottleneck.
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
    
# --class weights--

def _balanced_class_weights(counts):
    """sklearn-style 'balanced' weights: total / (n_classes * count_per_class).
    Mean weight across classes = 1.0, so overall loss scale is unchanged."""
    n_classes = len(counts)
    total = counts.sum()
    return total / (n_classes * counts + 1e-8)


def compute_seg_class_weights(rec_ids, num_classes=len(CLASSES)):
    """Compute 1-proportion weights for segmentation from training labels."""
    # count the propotions:
    counts = np.zeros(num_classes, dtype=np.float64)
    for rid in rec_ids:
        labels = np.load(DATA_FOR_ML / "labels" / f"{rid}.npy")
        for c in range(num_classes):
            counts[c] += np.sum(labels == c)
    weights = _balanced_class_weights(counts)
    print("Seg class counts:", dict(enumerate(counts.astype(int))))
    print("Seg class weights:", dict(enumerate(np.round(weights, 3))))
    return tf.constant(weights, dtype=tf.float32)


def compute_murmur_class_weights(rec_ids, num_classes=NUM_MURMUR_CLASSES):
    """Compute balanced weights for murmur from training labels."""
    # count the propotions:
    counts = np.zeros(num_classes, dtype=np.float64)
    for rid in rec_ids:
        m = np.load(DATA_FOR_ML / "murmurs" / f"{rid}.npy").item()
        counts[m] += 1
    weights = _balanced_class_weights(counts)
    print("Murmur class counts:", dict(enumerate(counts.astype(int))))
    print("Murmur class weights:", dict(enumerate(np.round(weights, 3))))
    return tf.constant(weights, dtype=tf.float32)


def weighted_loss(weights_tensor):
    """Returns a named loss function that penalizes mistakes on rare classes more heavily."""
    def loss_fn(y_true, y_pred):
        weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss * weights
    return loss_fn


def split_ids():
    """Split recording IDs into train (70%), val (15%), test (15%)
    while keeping all recordings from the same patient on the same side.
    This prevents data leakage: the model never validates/tests on a patient it trained on."""
    all_ids = load_all_ids()
    # Extract patient ID from recording ID. Rec IDs look like "PatientID_segment",
    # so splitting on "_" and taking the first part gives the patient.
    patient_ids = [rid.split("_")[0] for rid in all_ids]

    # GroupShuffleSplit: like train_test_split but respects groups.
    #   All recordings with the same patient_id stay together.
    #   n_splits=1: we only need one split (not k-fold cross-validation).
    #   test_size=0.30: 30% of patients go to the "temp" pool (will become val+test).
    #   random_state=42: fixed seed for reproducibility (same split every run).
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    # next(): GroupShuffleSplit is a generator; next() retrieves the single split.
    # split() returns indices (not data), so train_idx/temp_idx are arrays of positions.
    train_idx, temp_idx = next(gss1.split(all_ids, groups=patient_ids))
    # Use the indices to select the actual rec_id strings.
    train_ids = [all_ids[i] for i in train_idx]

    # Now split the 30% temp pool in half → 15% val + 15% test,
    # again keeping patients grouped.
    temp_ids_list = [all_ids[i] for i in temp_idx]
    temp_patients = [patient_ids[i] for i in temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_ids_list, groups=temp_patients))
    val_ids = [temp_ids_list[i] for i in val_idx]
    test_ids = [temp_ids_list[i] for i in test_idx]

    return train_ids, val_ids, test_ids

# ── Model ──
def build_model(seq_len=SAMPLES_NUM, num_seg_classes=NUM_SEG_CLASSES):
    inp = layers.Input(shape=(seq_len, 1))

    # ── Encoder block 1 ──
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    skip1 = x                                        # (4000, 64)
    x = layers.MaxPool1D(pool_size=POOL_1)(x)        # (1000, 64)

    # ── Encoder block 2 ──
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    skip2 = x                                        # (1000, 128)
    x = layers.MaxPool1D(pool_size=POOL_2)(x)        # (250, 128)

    # ── Bottleneck: single BiGRU ──
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    )(x)                                              # (250, 256)

    # ── Decoder block 1 ──
    x = layers.UpSampling1D(size=POOL_2)(x)           # (1000, 256)
    x = layers.Concatenate()([x, skip2])               # (1000, 384)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # ── Decoder block 2 ──
    x = layers.UpSampling1D(size=POOL_1)(x)            # (4000, 64)
    x = layers.Concatenate()([x, skip1])                # (4000, 128)
    seg_out = layers.Conv1D(
        num_seg_classes, 1, activation="softmax", name="seg"
    )(x)                                                # (4000, 5)

    return Model(inputs=inp, outputs=seg_out)


# ── Train ──
def main_with_murmur(resume_checkpoint_path: str | None = None, from_scratch: bool = False):
    train_ids, val_ids, test_ids = split_ids()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # we calculate the weights based on the imbalace of the train data
    seg_weights_tensor = compute_seg_class_weights(train_ids)
    murmur_weights_tensor = compute_murmur_class_weights(train_ids)

    train_ds = make_tf_dataset(train_ids, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_ids, BATCH_SIZE, shuffle=False)

    initial_epoch = 0
    if resume_checkpoint_path: # if we are resuming training from a checkpoint
        print(f"Loading model from: {resume_checkpoint_path}")
        model = keras.models.load_model(resume_checkpoint_path)
        if from_scratch: # if we are training from scratch
            initial_epoch = 0
            run_name = input("Enter a name for this training run: ").strip()
        else: # if we are resuming training from a certain epoch
            initial_epoch = int(Path(resume_checkpoint_path).stem.split("_")[-1])
            run_name = Path(resume_checkpoint_path).parent.name
    else:
        model = build_model()
        # compile: configures the model for training by specifying:
        model.compile(
            # Adam: an optimizer that adapts the learning rate per-parameter.
            #   It combines momentum (keeps moving in the same direction) with
            #   RMSProp (scales steps by recent gradient magnitude).
            # clipnorm=1.0: if the total gradient magnitude exceeds 1.0, scale it down.
            #   Prevents "exploding gradients" that can destabilize training.
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            # loss: which loss function to use for each output head.
            # Keys "seg" and "murmur" match the layer names in the model.
            loss={
                "seg": weighted_loss(seg_weights_tensor),
                "murmur": weighted_loss(murmur_weights_tensor),
            },
            # loss_weights: how much each head's loss contributes to the total.
            # total_loss = 1.0 * seg_loss + 0.5 * murmur_loss
            loss_weights={"seg": SEG_LOSS_WEIGHT, "murmur": MURMUR_LOSS_WEIGHT}, # weights of each classifiaction in the multi-task loss
            # metrics: quantities to track during training (don't affect optimization,
            # just printed/logged so you can monitor performance).
            metrics={
                "seg": "accuracy",
                "murmur": [
                    "accuracy",
                    # class_id=0 corresponds to "Present" murmur —
                    # we track recall and precision specifically for the positive class.
                    SparseRecall(class_id=0, name="recall_present"),
                    SparsePrecision(class_id=0, name="precision_present"),
                ],
            },
        )
        run_name = input("Enter a name for this training run: ").strip()

    # summary: prints a table of every layer, its output shape, and parameter count.
    model.summary()

    # Create a directory to save model checkpoints during training.
    # parents=True: create intermediate directories if needed.
    # exist_ok=True: don't error if it already exists.
    checkpoint_dir = DATA_FOR_ML.parent / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {checkpoint_dir} (resuming from epoch {initial_epoch})")

    # Callbacks: functions Keras calls automatically at certain points during training
    # (end of epoch, when a metric improves, etc.).
    metrics_csv = checkpoint_dir / "metrics.csv"
    callbacks = [
        # CSVLogger: writes loss and metric values to a CSV file after each epoch.
        # append=True when resuming so we don't overwrite previous epochs' data.
        keras.callbacks.CSVLogger(str(metrics_csv), append=(initial_epoch > 0)), # append=True when resuming so we don't overwrite previous epochs' data.
        # ModelCheckpoint (every epoch): saves the full model after each epoch.
        # {epoch:03d} formats the epoch number with 3 digits (e.g. epoch_007.keras).
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
            save_freq="epoch",
        ),
        # ModelCheckpoint (best only): saves the model only when val_loss improves.
        # This ensures we always have the best-performing version available.
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # ReduceLROnPlateau: if val_loss hasn't improved for `patience` epochs,
        # multiply the learning rate by `factor` (0.5 = halve it).
        # Smaller learning rate → finer weight adjustments → can escape plateaus.
        # min_lr: never reduce below this value.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
        # EarlyStopping: if val_loss hasn't improved for `patience` epochs, stop training.
        # restore_best_weights: after stopping, roll back the model weights to the epoch
        # with the lowest val_loss (not the last epoch, which may have overfit).
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
    ]

    # model.fit: the main training loop. For each epoch, iterates over all batches
    # in train_ds, computes loss, updates weights, then evaluates on val_ds.
    # Returns a History object containing loss/metric values per epoch.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        # epochs: maximum number of epochs to train (may stop earlier via EarlyStopping).
        epochs=EPOCHS,
        # initial_epoch: which epoch number to start from (0 for fresh, or the resumed epoch).
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # Plot loss and metric curves over epochs and save to a PNG file.
    from visualize_model import plot_training_curves
    plot_training_curves(history, save_path=f"figures/fig_training_curves_{run_name}.png")

    # ── Test evaluation ──
    # Evaluate on the held-out test set (never seen during training or validation).
    test_ds = make_tf_dataset(test_ids, BATCH_SIZE, shuffle=False)
    # model.evaluate: runs a forward pass on every test batch and computes loss + metrics.
    # return_dict=True: returns results as a dict {metric_name: value} instead of a list.
    results = model.evaluate(test_ds, return_dict=True) # note that if the model ran all 
    # the way to EPOCHS num, it will return the results of the last epoch - not necessarily the best one
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    # plot confusion matrix of both murmur classes and segmentation classes
    from visualize_model import plot_confusion_matrix, plot_murmur_confusion_matrix
    plot_confusion_matrix(model)
    plot_murmur_confusion_matrix(model)

def main_only_seg(resume_checkpoint_path: str | None = None, from_scratch: bool = False):
    train_ids, val_ids, test_ids = split_ids()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    train_ds = make_tf_dataset(train_ids, BATCH_SIZE, shuffle=True, seg_only=True)
    val_ds = make_tf_dataset(val_ids, BATCH_SIZE, shuffle=False, seg_only=True)
    test_ds = make_tf_dataset(test_ids, BATCH_SIZE, shuffle=False, seg_only=True)
    seg_weights_tensor = compute_seg_class_weights(train_ids)

    initial_epoch = 0
    if resume_checkpoint_path:
        print(f"Loading model from: {resume_checkpoint_path}")
        model = keras.models.load_model(resume_checkpoint_path)
        if from_scratch:
            initial_epoch = 0
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
    metrics_csv = checkpoint_dir / "metrics.csv"
    callbacks = [
        keras.callbacks.CSVLogger(str(metrics_csv), append=(initial_epoch > 0)),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"), save_freq="epoch"),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_dir / "best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    from visualize_model import plot_training_curves
    plot_training_curves(history, save_path=f"figures/fig_training_curves_{run_name}.png")
    test_ds = make_tf_dataset(test_ids, BATCH_SIZE, shuffle=False)
    results = model.evaluate(test_ds, return_dict=True) # note that if the model ran all 
    # the way to EPOCHS num, it will return the results of the last epoch - not necessarily the best one
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    # plot confusion matrix 
    from visualize_model import plot_confusion_matrix
    plot_confusion_matrix(model)

if __name__ == "__main__":
    # To train fresh:          main()
    # To resume training:      main(resume_checkpoint="checkpoints/<run>/epoch_NNN.keras")
    # To reload & retrain:     main(resume_checkpoint="checkpoints/<run>/epoch_NNN.keras", from_scratch=True)
    main_only_seg()

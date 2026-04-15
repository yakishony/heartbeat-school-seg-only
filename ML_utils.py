import keras
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
import numpy as np


from env import CLASSES, DATA_FOR_ML

# ── Data loading ──
# Overall flow: scan directory for rec IDs → split into train/val/test →
#  build a tf.data.Dataset that lazily loads .npy files one at a time.

def load_all_ids(data_path=DATA_FOR_ML):
    """Return a sorted list of all recording ID strings available on disk."""
    signal_dir = data_path / "signals"
    # glob("*.npy"): finds every file ending with .npy in the directory.
    # path.stem: strips the directory and extension, leaving just the filename
    #   (e.g. Path("signals/rec_001.npy").stem → "rec_001").
    # sorted(): puts the IDs in alphabetical order for reproducibility.
    return sorted(path.stem for path in signal_dir.glob("*.npy"))

def load_npy_pair(rec_id):
    signal = np.load(DATA_FOR_ML / "signals" / f"{rec_id}.npy")
    label = np.load(DATA_FOR_ML / "labels" / f"{rec_id}.npy")
    return signal, label

def _data_generator(rec_ids):
    for rec_id in rec_ids:
        signal, label = load_npy_pair(rec_id)
        signal = signal.astype(np.float32)[..., np.newaxis]  # (SAMPLES_NUM,) → (SAMPLES_NUM, 1)
        label = label.astype(np.int32)
        yield signal, label


def make_tf_dataset(rec_ids, batch_size, shuffle=True):
    from split_data_into_fixed_length_recordings import SAMPLES_NUM
    generator_func = lambda: _data_generator(rec_ids)  # Lambda returns a fresh generator each epoch so tf.data can cycle multiple epochs.

    dataset = tf.data.Dataset.from_generator(
        generator_func,
        output_signature=(
            tf.TensorSpec(shape=(SAMPLES_NUM, 1), dtype=tf.float32), # just reasurfing the shapes and dtypes for tf.data.Dataset.from_generator
            tf.TensorSpec(shape=(SAMPLES_NUM,),    dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(rec_ids))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --split ids--
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

# --class weights--

def _balanced_class_weights(counts):
    """sklearn-style 'balanced' weights: total / (n_classes * count_per_class).
    Mean weight across classes = 1.0, so overall loss scale is unchanged."""
    n_classes = len(counts)
    total_counts = counts.sum()
    return total_counts / (n_classes * counts + 1e-8)


def compute_seg_class_weights(rec_ids, num_classes=len(CLASSES), background_weight_is_1=True):
    """Compute 1-proportion weights for segmentation from training labels."""
    # count the propotions:
    counts = np.zeros(num_classes, dtype=np.float64)
    for rid in rec_ids:
        labels = np.load(DATA_FOR_ML / "labels" / f"{rid}.npy")
        for c in range(num_classes):
            counts[c] += np.sum(labels == c)
    weights = _balanced_class_weights(counts)
    if background_weight_is_1:
        weights[0] = 1.0  
    print("Seg class counts:", dict(enumerate(counts.astype(int))))
    print("Seg class weights:", dict(enumerate(np.round(weights, 3))))
    return tf.constant(weights, dtype=tf.float32)


def weighted_loss(weights_tensor):
    """Returns a named loss function that penalizes mistakes on rare classes more heavily."""
    def loss_fn(y_true, y_pred):
        weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss * weights
    return loss_fn

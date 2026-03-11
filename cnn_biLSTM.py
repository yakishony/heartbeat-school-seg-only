
"""# CNN + biLSTM"""

BATCH_SIZE = 32

# Optimized Pipeline:
# 1. Shuffle filenames FIRST (very light on RAM)
# 2. Map (load files) in parallel
# 3. Batch and Prefetch

# Shuffle the paths first
dataset_shuffled = filepath_dataset.shuffle(buffer_size=len(signal_filepaths), reshuffle_each_iteration=True)

# Map (load data) - num_parallel_calls lets the CPU prepare batches while GPU trains
dataset_mapped = dataset_shuffled.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and Prefetch
dataset_configured = dataset_mapped.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Verify
for signals_batch, labels_batch in dataset_configured.take(1):
    print(f"Batch signals shape: {signals_batch.shape}")
    print(f"Batch labels shape: {labels_batch.shape}")
    break

print("Optimized TensorFlow Dataset configured.")

"""## Update Model Training

### Subtask:
Modify the `full_model.fit()` call to use the newly created `tf.data.Dataset`.

**Reasoning**:
I will modify the existing `full_model.fit()` call to use the `dataset_configured` as the input, which is already set up for efficient loading, batching, and prefetching.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, TimeDistributed, UpSampling1D, BatchNormalization, Activation

# Input: 10 seconds at 4000Hz
input_signal = Input(shape=(TARGET_SAMPLES, 1), name='input_signal') # (40000, 1)

# --- Feature Extractor (More aggressive downsampling) ---
# Goal: Reduce sequence length from 40,000 to approx 625 for the LSTM
# Total reduction needed: 64x (2 * 4 * 8)

x = Conv1D(32, kernel_size=5, padding='same')(input_signal)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=4, padding='same')(x) # 40000 -> 10000

x = Conv1D(64, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=4, padding='same')(x) # 10000 -> 2500

x = Conv1D(128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=4, padding='same')(x) # 2500 -> 625

# --- BiLSTM Part ---
# Now sequence length is 625, which is very fast for LSTM
bi_lstm = Bidirectional(LSTM(128, return_sequences=True), name='bi_lstm_features')(x)

# --- Upsampling to original sequence length ---
# We went down by 4*4*4 = 64, so we need to go up by 64.
# Upsampling1D simply repeats values.
upsampled = UpSampling1D(size=64, name='upsample_to_original')(bi_lstm)

# --- Segmentation Head ---
segmentation_output = TimeDistributed(Dense(5, activation='softmax'), name='segmentation_head')(upsampled)

# --- Full Model ---
full_model = Model(inputs=input_signal, outputs=segmentation_output, name='optimized_segmentation_model')

full_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

full_model.summary()

# # Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os

EPOCHS = 5
checkpoint_dir = '/content/drive/MyDrive/דפיקות לב פרו/training_history_pro'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras.weights.h5')

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                 save_weights_only=True,
                                                 save_freq='epoch',
                                                 verbose=1)

# Train the model with the new callback
history = full_model.fit(
    dataset_configured,
          epochs=EPOCHS,
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

import pandas as pd
import matplotlib.pyplot as plt

# Convert the history.history dict to a pandas DataFrame
hist_df = pd.DataFrame(history.history)

# Display the history
print("Training History:")
print(hist_df)

# Plot training accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(hist_df['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(hist_df['loss'], label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
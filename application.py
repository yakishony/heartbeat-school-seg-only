"""
Gradio app for PCG heartbeat segmentation.
Upload a WAV file → bandpass filter → normalize → downsample → model predict → color-coded plot.
"""
import numpy as np
import gradio as gr
import keras

from scipy.io import wavfile

from run_pipline_analysing_utils import bandpass_filter, normalize_signal, downsample_signal
from plot_utils import plot_segmented_signal_interactive
from env import RATE, DOWNSAMPLE_FACTOR, RATE_DS
from split_data_into_fixed_length_recordings import SAMPLES_NUM, LEN_REC

CHECKPOINT = "checkpoints/downsampled_4x_reducedoverfit_testadded_model4/epoch_032.keras" # checkpoint should change to model13
model = keras.models.load_model(CHECKPOINT)
SUPPORTED_RATES = {RATE, RATE_DS}  # {4000, 1000}

def seperate_signal_into_2s_chunks(signal, signal_sr):
    n = len(signal)
    time = n / signal_sr

    if time < LEN_REC:
        raise gr.Error(f"Signal too short — need at least {SAMPLES_NUM / RATE_DS:.1f}s ")
    
    is_tail = time % LEN_REC > 0
    tail_length_time = time % LEN_REC
    overlap_samples = int((LEN_REC - tail_length_time) * signal_sr)
    overlap_samples_index_in_signal = None

    chunks = []

    chunks_num = int(time // LEN_REC) 
    for i in range(chunks_num):
        start = int(i * signal_sr * LEN_REC)
        end = int(start + signal_sr * LEN_REC)
        chunk = signal[start:end]
        chunks.append(chunk)
    if is_tail:
        start = int((time - LEN_REC) * signal_sr)
        end = n
        chunk = signal[start:end]
        chunks.append(chunk)
        overlap_samples_index_in_signal = (start, start + overlap_samples)
        
    return chunks, overlap_samples_index_in_signal


def from_wav_to_predicted_signal_plot(audio_path):
    sr, signal = wavfile.read(audio_path)

    if sr not in SUPPORTED_RATES:
        raise gr.Error(f"Sample rate must be {RATE} or {RATE_DS} Hz, got {sr} Hz")
    if signal.ndim > 1:
        signal = signal[:, 0] # take the first channel if there are multiple channels(ofter in stereo recordings) - coverts to mono
    signal = signal.astype(np.float64)

    # preprocess the signal
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, fs=sr)

    # downsample the signal if it is 4000 Hz:
    if sr == RATE:
        signal = downsample_signal(signal, DOWNSAMPLE_FACTOR)
        sr = sr // DOWNSAMPLE_FACTOR

    # split into chunks so that we will be able to insert data to the fixed sized model:
    chunks, overlap_samples_index_in_signal = seperate_signal_into_2s_chunks(signal, sr)
    full_predicted_signal = np.zeros(len(signal))
    chunk_len = len(chunks[0])

    # predict each chunk and insert the prediction to the full predicted signal:
    n_full_chunks = int(len(signal) / chunk_len)
    for chunk_num, chunk in enumerate(chunks):
        pred = model.predict(chunk.reshape(1, -1, 1), verbose=0)
        y_pred = pred.argmax(axis=-1).squeeze()
        if chunk_num < n_full_chunks:
            full_predicted_signal[chunk_num*chunk_len:(chunk_num+1)*chunk_len] = y_pred
        else:
            overlap_end = overlap_samples_index_in_signal[1] - overlap_samples_index_in_signal[0]
            full_predicted_signal[overlap_samples_index_in_signal[1]:] = y_pred[overlap_end:]
    
    
    return plot_segmented_signal_interactive(signal, full_predicted_signal, sr)


with gr.Blocks(title="Heartbeat PCG Segmentation") as demo:
    gr.Markdown("# Heartbeat PCG Segmentation")
    gr.Markdown(
        "Upload a heart sound WAV recording (4 000 or 1 000 Hz, ≥ 2 s). "
        "The model segments it into **S1**, **systole**, **S2**, and **diastole**.  \n"
        "Use the **scroll wheel to zoom** and **drag to pan** on the plot."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Upload PCG (.wav)")
            btn = gr.Button("Segment", variant="primary")
        with gr.Column(scale=3):
            plot_out = gr.Plot(label="Segmentation Result")
    btn.click(fn=from_wav_to_predicted_signal_plot, inputs=audio_in, outputs=plot_out)

if __name__ == "__main__":
    demo.launch()

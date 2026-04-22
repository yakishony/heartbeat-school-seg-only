"""
Gradio app for PCG heartbeat segmentation.
Upload a WAV file → bandpass filter → normalize → downsample → model predict → color-coded plot.
"""
import numpy as np
import gradio as gr
import keras

from scipy.io import wavfile
from scipy.signal import resample

from run_pipline_analysing_utils import bandpass_filter, normalize_signal, downsample_signal
from plot_utils import plot_segmented_signal_interactive, plot_plain_signal_interactive
from env import RATE, DOWNSAMPLE_FACTOR, RATE_DS
from split_data_into_fixed_length_recordings import SAMPLES_NUM, LEN_REC

# Load model once at server startup — stays in memory for all requests
CHECKPOINT = "checkpoints/model_13/best.keras"
model = keras.models.load_model(CHECKPOINT, compile=False)
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


def preprocess_wav(audio_path):
    """Load, filter, normalize, downsample. Returns (signal, sr)."""
    sr, signal = wavfile.read(audio_path)

    if not sr % RATE_DS == 0:
        raise gr.Error(f"Sampling Rate is not supported")
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = signal.astype(np.float64)

    # preprocess the signal
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, fs=sr)

    if sr != RATE_DS:
        if sr % RATE_DS == 0:
            signal = downsample_signal(signal, sr // RATE_DS)
        else:
            # resample for sample rates not evenly divisible by RATE_DS (e.g. 44100 Hz from microphone)
            num_samples = int(len(signal) * RATE_DS / sr) # the samples number in RATE_DS (RATE_DS * recording_lengs_s)
            signal = resample(signal, num_samples) # resample the signal to the new sample rate
        sr = RATE_DS

    return signal, sr


def on_upload(audio_path):
    """Show plain amplitude plot on file upload and cache the processed signal."""
    if audio_path is None:
        return gr.update(), None, None
    signal, sr = preprocess_wav(audio_path)
    fig = plot_plain_signal_interactive(signal, sr)
    return fig, signal.tolist(), sr # return the signal as a list to be used in the on_segment function - beacse gardio only supports that


def on_segment(signal_list, sr):
    """Run segmentation model on cached signal and return colored plot."""
    if signal_list is None:
        raise gr.Error("Upload a WAV file first.")
    signal = np.array(signal_list)

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
        else: # we are at the tail of the signal - we need to insert the prediction of the none overlap area
            overlap_end = overlap_samples_index_in_signal[1] - overlap_samples_index_in_signal[0]
            full_predicted_signal[overlap_samples_index_in_signal[1]:] = y_pred[overlap_end:]

    return plot_segmented_signal_interactive(signal, full_predicted_signal, sr=sr)


HIDE_SHARE_CSS = "button.share-btn, .share-button, button[title='Share'] { display: none !important; }"  # CSS rule injected into the Gradio page to hide the non-functional share button on the audio widget

with gr.Blocks(title="Heartbeat PCG Segmentation", css=HIDE_SHARE_CSS) as demo:
    gr.Markdown("# Heartbeat PCG Segmentation")
    gr.Markdown(
        "Upload a heart sound WAV recording (≥ 2 s). "
        "The model segments it into **S1**, **systole**, **S2**, **diastole**, and **unrecognized**.  \n"
        "Use the **scroll wheel to zoom** and **drag to pan** on the plot."
    )
    signal_state = gr.State(None)
    sr_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Upload PCG (.wav)", editable=False)
            btn = gr.Button("Segment", variant="primary")
        with gr.Column(scale=3):
            plot_out = gr.Plot(label="Segmentation Result")

    audio_in.change(fn=on_upload, inputs=audio_in, outputs=[plot_out, signal_state, sr_state])  # triggered when a file is uploaded
    audio_in.stop_recording(fn=on_upload, inputs=audio_in, outputs=[plot_out, signal_state, sr_state])  # triggered when microphone recording stops
    btn.click(fn=on_segment, inputs=[signal_state, sr_state], outputs=plot_out)  # triggered when "Segment" button is clicked

if __name__ == "__main__":
    demo.launch()

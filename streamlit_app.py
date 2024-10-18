import streamlit as st
import os
os.system("pip install tensorflow")
import tensorflow as tf
import numpy as np
os.system("pip install librosa")
import librosa
os.system("pip install matplotlib")
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.saved_model.load('YAMNet/YamNet')  # Update with your actual model path

classes = [
    "queen not present", 
    "queen present and newly accepted",
    "queen present and rejected",
    "queen present or original queen"
]

# Define a function to predict the Queen bee status
def predict_queen_status(audio_file):
    waveform, sr = librosa.load(audio_file, sr=16000)
    if waveform.shape[0] % 16000 != 0:
        waveform = np.concatenate([waveform, np.zeros(16000)])
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    prediction = classes[class_scores.argmax()]
    return prediction, waveform

# Define a function to plot the waveform and Mel spectrogram
def plot_waveform_and_spectrogram(waveform,sr=16000):
    #waveform, sr = librosa.load(audio, sr=16000)
    fig, axes = plt.subplots(2, figsize=(12, 8))
    # Plot the waveform
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform)
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, len(waveform)])
    
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Plot the Mel spectrogram
    img = librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Mel-Frequency Spectrogram')

    # Add a colorbar
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    # Show the plots
    plt.tight_layout()
    return fig

# Streamlit App Interface
st.title("🐝 Queen Bee Status Prediction")
st.write(
    "Upload an audio file to predict the Queen bee's status in the hive using a trained YAMNet model. "
    "This app analyzes the audio data and classifies the status of the Queen bee."
)

# Upload audio file
uploaded_audio = st.file_uploader("Upload an audio file (.wav or .mp3)", type=("wav", "mp3"))

if uploaded_audio:
    # Predict the Queen bee status
    prediction, waveform = predict_queen_status(uploaded_audio)
    
    # Display the prediction
    st.write(f"Predicted Queen Bee Status: **{prediction}**")
    
    # Plot the waveform and Mel spectrogram
    st.write("### Audio Analysis")
    #plot_waveform_and_spectrogram(waveform)
    st.pyplot(plot_waveform_and_spectrogram(waveform, 16000))
else:
    st.info("Please upload an audio file to continue.", icon="🔍")

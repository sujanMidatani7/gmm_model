import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
# Load the encoder classifier model

def cosine_similarity(x1, x2):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm_x1 = torch.norm(x1, dim=-1)
    norm_x2 = torch.norm(x2, dim=-1)
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    return cosine_similarity

def load_audio(file):
    # Load audio file using Torchaudio
    
        waveform, sample_rate = torchaudio.load(file)
        return waveform, sample_rate
    

def analyze_audio(file):
    waveform, sample_rate = load_audio(file)
    if waveform is None:
        return

    # Encode the audio signal using the x-vector model
    mfcc1 = librosa.feature.mfcc(y=waveform.numpy(), sr=16000, n_mfcc=20, n_fft=4096, hop_length=11)
    delta_mfcc1 = librosa.feature.delta(mfcc1)
    delta2_mfcc1 = librosa.feature.delta(mfcc1, order=2)
    ddmfcc1 = np.vstack((mfcc1, delta_mfcc1, delta2_mfcc1))
    ddmfcc1 = (ddmfcc1 - np.mean(ddmfcc1)) / np.std(ddmfcc1)
#     st.write(ddmfcc1.shape)
    X1 = np.reshape(ddmfcc1.T, (ddmfcc1.shape[1]*ddmfcc1.shape[0]*ddmfcc1.shape[2], 1))
    n_components = 25

    gmm1 = GaussianMixture(n_components=n_components, covariance_type='diag')
    gmm1.fit(X1)
    features1 = np.zeros(n_components)
    for i in range(25):
         var1 = np.diag(gmm1.covariances_[i])
         weight1 = gmm1.weights_[i]
         features1[i] = np.sum(weight1 * np.log(var1) + 0.5 * np.log(2 * np.pi * np.e))

    return torch.from_numpy(features1)
    # Display the embeddings
#     st.write("The x-vector embeddings are:")
#     st.write(embeddings_xvect)

# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparision of two audio samples using gmm model.")

# Create audio input component
audio_file1 = st.file_uploader("Choose 1st audio  file", type=["mp3", "wav", "flac"])
gmm1=torch.rand(2, 3,1)
gmm2=torch.rand(2, 3,1)
# Analyze audio properties when file is uploaded
if audio_file1 is not None:
    gmm1=analyze_audio(audio_file1)
audio_file2=st.file_uploader("Choose 2nd audio  file", type=["mp3", "wav", "flac"])
if audio_file2 is not None:
    gmm2=analyze_audio(audio_file2)
st.write("the similarity of the given two audio files is:")
st.write(cosine_similarity(gmm1,gmm2)[0][0])

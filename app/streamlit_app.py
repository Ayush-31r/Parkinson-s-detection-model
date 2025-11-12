import streamlit as st
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import tempfile
import os

# -------------------- Config --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_rate = 22050
n_mels = 64

# -------------------- Model Definition --------------------
class CNNVoiceClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNVoiceClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = CNNVoiceClassifier(num_classes=2)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "../Models/cnn_voice_classifier_state.pth")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ✅ Initialize the model
model = load_model()

# -------------------- Mel Spectrogram Transform --------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=n_mels
)

# -------------------- Helper Functions --------------------
def chunk_waveform(waveform, sr, chunk_sec=5):
    chunk_len = chunk_sec * sr
    chunks = []
    total_len = waveform.shape[-1]
    for start in range(0, total_len, chunk_len):
        end = start + chunk_len
        chunk = waveform[:, start:end]
        if chunk.shape[-1] < chunk_len:
            pad = torch.zeros((waveform.shape[0], chunk_len - chunk.shape[-1]))
            chunk = torch.cat([chunk, pad], dim=-1)
        chunks.append(chunk)
    return chunks

def predict_parkinsons(waveform, model):
    chunks = chunk_waveform(waveform, sample_rate)
    preds = []
    for chunk in chunks:
        mel = mel_transform(chunk)
        mel_db = torchaudio.functional.amplitude_to_DB(
            mel, multiplier=10.0, amin=1e-10, db_multiplier=0
        )
        mel_db = mel_db.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(mel_db)
            prob = torch.softmax(output, dim=1)[:, 1].item()
            preds.append(prob)
    return np.mean(preds)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Parkinson’s Voice Detector", layout="centered")
st.title("🎙️ Parkinson’s Voice Detector")

st.markdown(
    """
    Upload a `.wav` voice recording to estimate Parkinson’s likelihood.  
    **Note:** Longer samples (≈ 1 minute) yield more accurate and stable predictions.
    """
)

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path)

    waveform_data, sr = sf.read(temp_path, dtype='float32')
    waveform = torch.tensor(waveform_data.T).unsqueeze(0) if waveform_data.ndim == 1 else torch.tensor(waveform_data.T)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    prob = predict_parkinsons(waveform, model)
    st.success(f"🧠 Parkinson’s likelihood: {prob*100:.2f}%")

st.markdown("---")
st.caption("Model: CNN on Mel Spectrograms | Built by Ayush Rai")

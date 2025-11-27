# streamlit_ui.py

import requests
import streamlit as st
import pandas as pd
from pathlib import Path

from src.preprocessing import PROCESSED_DATA_DIR
from src.model import train_best_model

API_BASE = API_BASE = "https://mlpipeline-hiphop.onrender.com"

st.set_page_config(page_title="Hip-Hop Era Classifier", layout="wide")

st.title("Hip-Hop Era Classifier Dashboard")

# ---- Model uptime ----
st.subheader("Model Uptime & Status")
try:
    health = requests.get(f"{API_BASE}/health").json()
    st.success(f"API Status: {health['status']}, Uptime: {health['uptime_seconds']:.1f} seconds")
except Exception as e:
    st.error(f"Could not reach API at {API_BASE}: {e}")

# ---- Data visualizations ----
st.subheader("Data Visualizations")

train_csv = PROCESSED_DATA_DIR / "train_features.csv"
if train_csv.exists():
    df = pd.read_csv(train_csv)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tempo distribution by era**")
        st.bar_chart(df.groupby("era")["tempo"].mean())

    with col2:
        st.markdown("**Spectral centroid by era**")
        st.bar_chart(df.groupby("era")["spectral_centroid"].mean())

    st.markdown("**MFCC mean (0) by era**")
    st.bar_chart(df.groupby("era")["mfcc_mean_0"].mean())

    st.info(
        "Interpretation: higher average tempo and spectral centroid generally "
        "indicate more energetic, brighter tracks in modern/trap eras compared "
        "to golden age hip-hop."
    )
else:
    st.warning("Processed train features not found. Run preprocessing first.")

# ---- Prediction demo ----
st.subheader("Predict Era for a Single Audio File")

audio_file = st.file_uploader("Upload an audio file (.mp3/.wav)", type=["mp3", "wav", "flac"])
if st.button("Predict") and audio_file is not None:
    files = {"file": (audio_file.name, audio_file.getvalue())}
    resp = requests.post(f"{API_BASE}/predict-audio", files=files)
    if resp.status_code == 200:
        out = resp.json()
        st.success(f"Predicted Era: **{out['predicted_label']}**")
        st.json(out["probabilities"])
    else:
        st.error(f"Error from API: {resp.text}")

# ---- Bulk upload + retrain ----
st.subheader("Bulk Upload for Retraining")

csv_file = st.file_uploader("Upload new training data CSV", type=["csv"], key="retrain_csv")
if st.button("Upload CSV for Retrain") and csv_file is not None:
    files = {"file": (csv_file.name, csv_file.getvalue())}
    resp = requests.post(f"{API_BASE}/upload-training-data", files=files)
    st.write(resp.json())

if st.button("Trigger Retrain"):
    resp = requests.post(f"{API_BASE}/retrain")
    st.write(resp.json())


# src/prediction.py

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from .preprocessing import extract_audio_features, get_feature_names, SAMPLE_RATE, DURATION
from .model import MODEL_DIR
from .preprocessing import DATA_DIR


def load_model_components():
    model = joblib.load(MODEL_DIR / "hiphop_era_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    metadata = joblib.load(MODEL_DIR / "model_metadata.pkl")
    feature_names = metadata["feature_names"]
    return model, scaler, label_encoder, feature_names, metadata


def predict_audio_era(audio_path: Path) -> Dict[str, Any]:
    """
    Given a single audio file, run the full preprocessing and prediction pipeline.

    Returns:
        dict with predicted_label, probabilities, and metadata.
    """
    model, scaler, label_encoder, feature_names, _ = load_model_components()

    # Extract features
    feats = extract_audio_features(audio_path)
    # Make sure shape is (1, n_features)
    feats = feats.reshape(1, -1)

    # Scale
    feats_scaled = scaler.transform(feats)

    # Predict
    pred_encoded = model.predict(feats_scaled)[0]
    proba = model.predict_proba(feats_scaled)[0]

    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    proba_dict = {
        cls: float(p) for cls, p in zip(label_encoder.classes_, proba)
    }

    return {
        "file_path": str(audio_path),
        "predicted_label": pred_label,
        "probabilities": proba_dict,
    }


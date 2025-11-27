# src/api.py

import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .prediction import predict_audio_era
from .model import retrain_from_csv, train_best_model, MODEL_DIR
from .preprocessing import DATA_DIR

app = FastAPI(title="Hip-Hop Era Classifier API")

# For UI / local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

APP_START_TIME = time.time()
LATEST_RETRAIN_CSV: Optional[Path] = None


@app.on_event("startup")
def on_startup():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # You can optionally check whether model exists and train if missing


@app.get("/health")
def health():
    uptime_seconds = time.time() - APP_START_TIME
    return {"status": "ok", "uptime_seconds": uptime_seconds}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav", ".flac")):
        raise HTTPException(status_code=400, detail="Please upload an audio file (.mp3/.wav/.flac).")

    temp_path = Path("/tmp") / file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    result = predict_audio_era(temp_path)
    return result


@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Accept a CSV file with the same schema as train_features.csv,
    including 'era' as the label column.
    """
    global LATEST_RETRAIN_CSV
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    retrain_dir = DATA_DIR / "retrain_uploads"
    retrain_dir.mkdir(parents=True, exist_ok=True)
    out_path = retrain_dir / file.filename
    df.to_csv(out_path, index=False)

    LATEST_RETRAIN_CSV = out_path
    return {"message": "CSV uploaded", "path": str(out_path)}


@app.post("/retrain")
def retrain_model():
    """
    Trigger retraining using the last uploaded CSV (via /upload-training-data).
    """
    if LATEST_RETRAIN_CSV is None:
        raise HTTPException(status_code=400, detail="No retrain CSV uploaded yet.")

    result = retrain_from_csv(LATEST_RETRAIN_CSV)
    return {"message": "Model retrained", "metrics": result}


# src/preprocessing.py

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import train_test_split

# ---- Directories (match notebook) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Constants (match notebook) ----
SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42


def assign_era(date_str: str) -> str:
    """Assign era label given a date string (same logic as notebook)."""
    if pd.isna(date_str):
        return None
    try:
        year = pd.to_datetime(date_str).year
        if year < 1995:
            return "golden_age"
        elif year < 2005:
            return "bling_era"
        elif year < 2015:
            return "trap_rise"
        else:
            return "modern"
    except Exception:
        return None


def organize_dataset_by_era(
    hiphop_tracks: pd.DataFrame, audio_dir: Path, output_dir: Path
) -> None:
    """
    Organize audio files into train/test splits by era
    (same behavior as notebook's organize_dataset_by_era).
    """
    output_dir = Path(output_dir)
    train_df, test_df = train_test_split(
        hiphop_tracks,
        test_size=TEST_SIZE,
        stratify=hiphop_tracks["era"],
        random_state=RANDOM_STATE,
    )

    # Ensure dirs exist
    for split, df in [("train", train_df), ("test", test_df)]:
        for era_name in df["era"].unique():
            era_dir = output_dir / split / era_name
            era_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink files from FMA structure into our train/test tree.
    # NOTE: In the notebook you were copying from FMA's folder layout.
    # Here we assume that RAW_DATA_DIR mirrors that, but you can adapt as needed.

    for split, df in [("train", train_df), ("test", test_df)]:
        print(f"Organizing {split} files...")
        for idx, row in df.iterrows():
            era = row["era"]
            track_id = str(idx).zfill(6)
            # FMA: audio/<first three digits>/<track_id>.mp3
            src = audio_dir / track_id[:3] / f"{track_id}.mp3"
            dst = output_dir / split / era / f"{track_id}.mp3"
            if src.exists():
                if not dst.exists():
                    os.link(src, dst)  # hard link; use shutil.copy2 if needed


def extract_audio_features(audio_path: Path) -> np.ndarray:
    """
    Extract features from an audio file.

    Returns:
        numpy array of 43 features
    """
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=DURATION)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    # Rhythmic
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # 12 chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            np.array([spectral_centroid]).flatten(),
            np.array([spectral_rolloff]).flatten(),
            np.array([spectral_bandwidth]).flatten(),
            np.array([zero_crossing_rate]).flatten(),
            np.array([tempo]).flatten(),
            chroma_mean,
        ]
    )
    return features


def get_feature_names() -> List[str]:
    """Return the feature name list used in the notebook."""
    return (
        [f"mfcc_mean_{i}" for i in range(N_MFCC)]
        + [f"mfcc_std_{i}" for i in range(N_MFCC)]
        + [
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_bandwidth",
            "zero_crossing_rate",
            "tempo",
        ]
        + [f"chroma_{i}" for i in range(12)]
    )


def process_dataset(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Build the feature dataframe for a given split ('train' or 'test').

    Mirrors the notebook's process_dataset: loops through era folders,
    extracts features, and builds a DataFrame with columns:
    feature_names + ['era', 'file_path'].
    """
    split_dir = Path(data_dir) / split
    feature_names = get_feature_names()

    features_list = []
    labels_list = []
    file_paths = []

    for era_dir in sorted(split_dir.iterdir()):
        if not era_dir.is_dir():
            continue
        era_label = era_dir.name
        for audio_file in era_dir.glob("*.mp3"):
            try:
                feats = extract_audio_features(audio_file)
            except Exception as e:
                print(f"Skipping {audio_file} due to error: {e}")
                continue
            features_list.append(feats)
            labels_list.append(era_label)
            file_paths.append(str(audio_file))

    df = pd.DataFrame(features_list, columns=feature_names)
    df["era"] = labels_list
    df["file_path"] = file_paths

    print(f"Extracted features from {len(df)} files")
    print(f"Feature dimensions: {df.shape}")

    return df


def save_processed_features() -> None:
    """High-level helper to process train/test and save CSVs (as in notebook)."""
    train_df = process_dataset(DATA_DIR, split="train")
    test_df = process_dataset(DATA_DIR, split="test")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_features.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_features.csv", index=False)

    print("Saved processed features to:", PROCESSED_DATA_DIR)


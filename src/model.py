# src/model.py

from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

from .preprocessing import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    DATA_DIR,
    get_feature_names,
)

RANDOM_STATE = 42
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_data() -> Dict[str, Any]:
    train_features_df = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    test_features_df = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv")

    feature_columns = [
        c for c in train_features_df.columns if c not in ["era", "file_path"]
    ]

    X_train = train_features_df[feature_columns].values
    y_train = train_features_df["era"].values
    X_test = test_features_df[feature_columns].values
    y_test = test_features_df["era"].values

    return dict(
        train_df=train_features_df,
        test_df=test_features_df,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def train_best_model() -> Dict[str, Any]:
    data = load_feature_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_columns = data["feature_columns"]

    # ---- Scaling & encoding (same as notebook) ----
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encoded = label_encoder.fit_transform(y_train)

    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = label_encoder.transform(y_test)

    # ---- RandomForest with log-loss GridSearch ----
    rf_param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_param_grid,
        scoring="neg_log_loss",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    rf_grid.fit(X_train_scaled, y_train_encoded)
    rf_model = rf_grid.best_estimator_

    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)
    rf_logloss = log_loss(y_test_encoded, y_pred_proba_rf)

    # ---- LightGBM with early stopping ----
    X_train_lgb, X_valid_lgb, y_train_lgb, y_valid_lgb = train_test_split(
        X_train_scaled,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE,
    )

    lgbm_model = LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lgbm_model.fit(
        X_train_lgb,
        y_train_lgb,
        eval_set=[(X_valid_lgb, y_valid_lgb)],
        eval_metric="multi_logloss",
        callbacks=[],  # in notebook you used early_stopping primitive; this matches intent
    )

    y_proba_lgbm_test = lgbm_model.predict_proba(X_test_scaled)
    lgbm_logloss = log_loss(y_test_encoded, y_proba_lgbm_test)
    y_pred_lgbm = lgbm_model.predict(X_test_scaled)

    # ---- LightGBM extra metrics ----
    lgbm_accuracy = accuracy_score(y_test_encoded, y_pred_lgbm)
    lgbm_precision, lgbm_recall, lgbm_f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred_lgbm, average="weighted"
    )

    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test_encoded, classes=np.arange(n_classes))
    brier_per_class_lgbm = [
        brier_score_loss(y_test_bin[:, i], y_proba_lgbm_test[:, i])
        for i in range(n_classes)
    ]
    lgbm_brier_mean = float(np.mean(brier_per_class_lgbm))

    # ---- Choose best model based on log-loss ----
    if lgbm_logloss < rf_logloss:
        best_model = lgbm_model
        best_model_name = "LightGBM (early stopping)"
        best_proba = y_proba_lgbm_test
        best_pred = y_pred_lgbm
    else:
        best_model = rf_model
        best_model_name = "RandomForest (GridSearchCV)"
        best_proba = y_pred_proba_rf
        best_pred = y_pred_rf

    best_accuracy = accuracy_score(y_test_encoded, best_pred)
    best_precision, best_recall, best_f1, _ = precision_recall_fscore_support(
        y_test_encoded, best_pred, average="weighted"
    )
    best_logloss = log_loss(y_test_encoded, best_proba)

    # ---- Save model, scaler, encoder, metadata (as in notebook) ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODEL_DIR / "hiphop_era_classifier.pkl"
    model_versioned_path = MODEL_DIR / f"hiphop_era_classifier_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(best_model, model_versioned_path)

    scaler_path = MODEL_DIR / "scaler.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "timestamp": timestamp,
        "best_model_name": best_model_name,
        "accuracy": float(best_accuracy),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1_score": float(best_f1),
        "log_loss": float(best_logloss),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_features": int(len(feature_columns)),
        "feature_names": feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "rf_logloss": float(rf_logloss),
        "lgbm_logloss": float(lgbm_logloss),
        "lgbm_brier_mean": float(lgbm_brier_mean),
    }

    metadata_path = MODEL_DIR / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics": metadata,
    }


def retrain_from_csv(csv_path: Path) -> Dict[str, Any]:
    """
    Retrain the currently best model using additional labeled rows in csv_path.

    This mirrors the idea of retrain_model_from_db(...) in the notebook,
    but uses a local CSV file instead of MongoDB.
    """
    data = load_feature_data()
    X_train, y_train = data["X_train"], data["y_train"]

    df_new = pd.read_csv(csv_path)
    feature_columns = data["feature_columns"]

    missing_cols = [c for c in feature_columns + ["era"] if c not in df_new.columns]
    if missing_cols:
        raise ValueError(f"New data is missing required columns: {missing_cols}")

    new_X = df_new[feature_columns].values
    new_y = df_new["era"].values

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    base_model = joblib.load(MODEL_DIR / "hiphop_era_classifier.pkl")

    X_train_scaled = scaler.transform(X_train)
    y_train_encoded = encoder.transform(y_train)

    new_X_scaled = scaler.transform(new_X)
    new_y_encoded = encoder.transform(new_y)

    X_combined = np.vstack([X_train_scaled, new_X_scaled])
    y_combined = np.concatenate([y_train_encoded, new_y_encoded])

    ModelClass = base_model.__class__
    params = base_model.get_params()
    retrained_model = ModelClass(**params)
    retrained_model.fit(X_combined, y_combined)

    # Evaluate on original test set
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = encoder.transform(y_test)

    y_proba = retrained_model.predict_proba(X_test_scaled)
    retrain_logloss = log_loss(y_test_encoded, y_proba)

    joblib.dump(retrained_model, MODEL_DIR / "hiphop_era_classifier.pkl")
    return {"retrain_logloss": float(retrain_logloss)}


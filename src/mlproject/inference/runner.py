from __future__ import annotations

import datetime as dt
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("outputs/model_artifacts/best_model.joblib")
TEST_FEATURES_PATH = Path("src/mlproject/data/test_features.csv")
OUT_DIR = Path("outputs")


def run_inference(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_features_path: str | Path = TEST_FEATURES_PATH,
    out_dir: str | Path = OUT_DIR,
) -> str:
    """
    Orchestrates batch inference:
    1) preprocess raw -> clean
    2) feature engineering -> features
    3) load best model and predict on test features
    4) write outputs/YYYYMMDD_predictions.csv
    """
    model_path = Path(model_path)
    test_features_path = Path(test_features_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess
    subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)

    # 2) Feature engineering
    subprocess.run([sys.executable, "scripts/feature_engineering.py"], check=True)

    # 3) Load model + predict
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run uv run train first.")

    if not test_features_path.exists():
        raise FileNotFoundError(f"Test features not found: {test_features_path}.")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_features_path)

    # Keep id if present
    id_col = "id" if "id" in test_df.columns else None
    X_test = test_df.copy()

    preds_log1p = model.predict(X_test)
    preds = np.expm1(preds_log1p)  # undo log1p used in training

    # 4) Save predictions
    stamp = dt.datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"{stamp}_predictions.csv"

    if id_col:
        out_df = pd.DataFrame({id_col: test_df[id_col], "trip_duration": preds})
    else:
        out_df = pd.DataFrame({"trip_duration": preds})

    out_df.to_csv(out_path, index=False)
    return str(out_path)
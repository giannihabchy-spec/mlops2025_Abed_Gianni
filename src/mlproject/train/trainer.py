from _future_ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainConfig:
    train_features_path: str = "src/mlproject/data/train_features.csv"
    target: str = "trip_duration"
    id_col: str = "id"
    test_size: float = 0.2
    random_state: int = 42
    artifacts_dir: str = "outputs/model_artifacts"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def get_models(random_state: int = 42) -> Dict[str, object]:
    # 5 different regression models
    return {
        "ridge": Ridge(alpha=5.0),  # Ridge has no random_state
        "lasso": Lasso(alpha=0.0005, random_state=random_state, max_iter=20000),
        "elasticnet": ElasticNet(
            alpha=0.001, l1_ratio=0.3, random_state=random_state, max_iter=20000
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=None,
            random_state=random_state,
        ),
    }


def load_training_data(cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(cfg.train_features_path)

    if cfg.target not in data.columns:
        raise ValueError(
            f"Target '{cfg.target}' not found in {cfg.train_features_path}"
        )

    y = data[cfg.target].astype(float)
    X = data.drop(columns=[cfg.target], errors="ignore")

    if cfg.id_col in X.columns:
        X = X.drop(columns=[cfg.id_col], errors="ignore")

    return X, y


def train_and_select(cfg: TrainConfig) -> Dict[str, object]:
    X, y_raw = load_training_data(cfg)

    # Stabilize target (heavy-tailed)
    y = np.log1p(y_raw.to_numpy())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    preprocessor = _build_preprocessor(X_train)
    models = get_models(cfg.random_state)

    rows: List[dict] = []
    best_name = None
    best_score = None
    best_pipe = None

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_val)
        score = _rmse(y_val, pred)  # RMSE on log1p target

        rows.append({"model": name, "rmse_log1p": score})

        if best_score is None or score < best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    results = (
        pd.DataFrame(rows)
        .sort_values("rmse_log1p", ascending=True)
        .reset_index(drop=True)
    )

    return {
        "best_model_name": best_name,
        "best_rmse_log1p": float(best_score),
        "best_pipeline": best_pipe,
        "results": results,
    }


def save_artifacts(cfg: TrainConfig, payload: Dict[str, object]) -> Dict[str, str]:
    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "best_model.joblib"
    results_path = out_dir / "results.csv"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(payload["best_pipeline"], model_path)

    results_df: pd.DataFrame = payload["results"]
    results_df.to_csv(results_path, index=False)

    metrics = {
        "train_features_path": cfg.train_features_path,
        "target": cfg.target,
        "metric": "rmse_log1p",
        "best_model": payload["best_model_name"],
        "best_rmse_log1p": payload["best_rmse_log1p"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {
        "model_path": str(model_path),
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
    }


def run_training(cfg: TrainConfig | None = None) -> Dict[str, str]:
    cfg = cfg or TrainConfig()
    payload = train_and_select(cfg)
    paths = save_artifacts(cfg, payload)
    return paths
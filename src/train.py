"""Train and compare every registered regression algorithm on the housing dataset.

Run with: python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data import build_preprocessor, get_features_and_target, load_data
from src.metrics import mae, mdape, r2, rmse
from src.models import MODEL_REGISTRY, slugify

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def build_pipeline(build_model) -> Pipeline:
    # Prices are heavily right-skewed, so every model is fitted on log(price);
    # predict() applies exp() and returns naira.
    model = TransformedTargetRegressor(regressor=build_model(), func=np.log, inverse_func=np.exp)
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def r2_log(y_true, y_pred) -> float:
    # R² on log(price): the headline metric, since naira-scale R² is dominated by the
    # handful of most expensive listings.
    return r2(np.log(y_true), np.log(y_pred))


def train_all(df: pd.DataFrame | None = None, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, build_model in MODEL_REGISTRY.items():
        pipeline = build_pipeline(build_model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # CV on the training split only, so the test set stays unseen by every fit.
        cv_scores = cross_val_score(
            build_pipeline(build_model),
            X_train,
            y_train,
            cv=cv,
            scoring=make_scorer(r2_log),
        )

        results.append(
            {
                "Model": name,
                "R2_log": r2_log(y_test, y_pred),
                "RMSE": rmse(y_test, y_pred),
                "MAE": mae(y_test, y_pred),
                "MdAPE": mdape(y_test, y_pred),
                "CV_R2_log_mean": float(cv_scores.mean()),
                "CV_R2_log_std": float(cv_scores.std()),
            }
        )

        joblib.dump(pipeline, artifacts_dir / f"{slugify(name)}.joblib")
        print(
            f"Trained {name}: R2_log={results[-1]['R2_log']:.3f}  "
            f"MAE={results[-1]['MAE']:,.0f}  MdAPE={results[-1]['MdAPE']:.1%}"
        )

    metrics_df = pd.DataFrame(results).sort_values("R2_log", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(artifacts_dir / "metrics.csv", index=False)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    return metrics_df


if __name__ == "__main__":
    metrics_df = train_all()
    print("\nModel comparison (sorted by test R2_log):")
    print(metrics_df.to_string(index=False))

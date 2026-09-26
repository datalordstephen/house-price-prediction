"""Train and compare every registered regression algorithm on the housing dataset.

Run with: python -m src.train
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data import build_preprocessor, get_features_and_target, load_data
from src.metrics import mae, r2, rmse
from src.models import MODEL_REGISTRY, slugify

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


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
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", build_model())]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(
            Pipeline(steps=[("preprocess", build_preprocessor()), ("model", build_model())]),
            X,
            y,
            cv=cv,
            scoring="r2",
        )

        results.append(
            {
                "Model": name,
                "RMSE": rmse(y_test, y_pred),
                "MAE": mae(y_test, y_pred),
                "R2": r2(y_test, y_pred),
                "CV_R2_mean": float(cv_scores.mean()),
                "CV_R2_std": float(cv_scores.std()),
            }
        )

        joblib.dump(pipeline, artifacts_dir / f"{slugify(name)}.joblib")
        print(f"Trained {name}: RMSE={results[-1]['RMSE']:.0f}  R2={results[-1]['R2']:.3f}")

    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(artifacts_dir / "metrics.csv", index=False)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    return metrics_df


if __name__ == "__main__":
    metrics_df = train_all()
    print("\nModel comparison (sorted by test R2):")
    print(metrics_df.to_string(index=False))

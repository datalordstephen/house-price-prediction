"""Train and compare several regression algorithms on the housing dataset.

Run with: python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data import build_preprocessor, get_features_and_target, load_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42

MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge Regression": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Neural Net (MLP)": lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def train_all() -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, factory in MODEL_FACTORIES.items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", factory())]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(
            Pipeline(steps=[("preprocess", build_preprocessor()), ("model", factory())]),
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

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, MODELS_DIR / f"{slug}.joblib")
        print(f"Trained {name}: RMSE={results[-1]['RMSE']:.0f}  R2={results[-1]['R2']:.3f}")

    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    return metrics_df


if __name__ == "__main__":
    metrics_df = train_all()
    print("\nModel comparison (sorted by test R2):")
    print(metrics_df.to_string(index=False))

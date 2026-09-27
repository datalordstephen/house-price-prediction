"""Data loading and preprocessing for the Nigerian housing dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clean_nig_housing_dset.csv"

TARGET = "Price_NGN"
CATEGORICAL_COLS = ["City", "Property_Type", "Ownership", "Condition", "Available_Utilities"]
NUMERIC_COLS = ["Latitude", "Longitude", "Bedrooms", "Bathrooms", "Size_sqm"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )

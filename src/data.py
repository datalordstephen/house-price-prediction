"""Data loading, cleaning and preprocessing for the Nigerian housing dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nigeria_houses_data.csv"

TARGET = "price"
CATEGORICAL_COLS = ["title", "town", "state"]
NUMERIC_COLS = ["bedrooms", "bathrooms", "toilets", "parking_space"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# Fixed price bounds (₦), roughly the 1st and 99.5th percentiles of the deduplicated
# data. Constants rather than load-time quantiles so the row set never drifts.
PRICE_MIN = 5_000_000
PRICE_MAX = 2_000_000_000


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Raw rows: {len(df):,}")

    # Exact duplicates are repeated listings; left in, the same listing would land in
    # both the train and test sets.
    df = df.drop_duplicates()
    print(f"After dropping duplicates: {len(df):,}")

    df = df[df[TARGET].between(PRICE_MIN, PRICE_MAX)]
    print(f"After keeping {PRICE_MIN:,} <= {TARGET} <= {PRICE_MAX:,}: {len(df):,}")

    return df.reset_index(drop=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                # Categories seen fewer than 10 times (about half the towns) share one
                # "infrequent" column, which unseen categories also map to at predict time.
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
                CATEGORICAL_COLS,
            ),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )

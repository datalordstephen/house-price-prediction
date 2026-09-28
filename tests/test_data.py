import pandas as pd

from src.data import (
    CATEGORICAL_COLS,
    MISLABELLED_STATE,
    NUMERIC_COLS,
    PRICE_MAX,
    PRICE_MIN,
    TARGET,
    build_preprocessor,
    clean_data,
    get_features_and_target,
    load_data,
)


def test_load_data_has_expected_columns():
    df = load_data()
    expected = ["title", "town", "state", "bedrooms", "bathrooms", "toilets", "parking_space", "price"]
    assert CATEGORICAL_COLS + NUMERIC_COLS + [TARGET] == expected
    for col in expected:
        assert col in df.columns
    assert len(df) > 0
    # load_data() returns cleaned data
    assert not df.duplicated().any()
    assert df[TARGET].between(PRICE_MIN, PRICE_MAX).all()
    assert (df["state"] != MISLABELLED_STATE).all()


def test_clean_data_drops_duplicates_and_out_of_range_prices(tiny_df):
    in_range = tiny_df.head(3).copy()
    in_range[TARGET] = [PRICE_MIN, 50_000_000, PRICE_MAX]  # bounds are inclusive
    too_cheap = in_range.iloc[[0]].assign(**{TARGET: PRICE_MIN - 1})
    too_dear = in_range.iloc[[0]].assign(**{TARGET: PRICE_MAX + 1})
    raw = pd.concat([in_range, in_range.iloc[[1]], too_cheap, too_dear], ignore_index=True)

    cleaned = clean_data(raw)

    assert len(cleaned) == 3
    assert not cleaned.duplicated().any()
    assert sorted(cleaned[TARGET]) == [PRICE_MIN, 50_000_000, PRICE_MAX]


def test_clean_data_drops_mislabelled_state(tiny_df):
    raw = tiny_df.head(4).copy()
    raw.loc[[0, 1], "state"] = MISLABELLED_STATE

    cleaned = clean_data(raw)

    assert len(cleaned) == 2
    assert MISLABELLED_STATE not in set(cleaned["state"])


def test_get_features_and_target(tiny_df):
    X, y = get_features_and_target(tiny_df)
    assert list(X.columns) == CATEGORICAL_COLS + NUMERIC_COLS
    assert y.name == TARGET
    assert len(X) == len(y) == len(tiny_df)


def test_build_preprocessor_expands_categoricals(tiny_df):
    X, _ = get_features_and_target(tiny_df)
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == len(X)
    # one-hot encoding must produce more columns than the raw numeric-only count
    assert transformed.shape[1] > len(NUMERIC_COLS)

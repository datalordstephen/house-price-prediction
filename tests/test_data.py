from src.data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    build_preprocessor,
    get_features_and_target,
    load_data,
)


def test_load_data_has_expected_columns():
    df = load_data()
    for col in CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]:
        assert col in df.columns
    assert len(df) > 0


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

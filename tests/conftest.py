import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

_CATEGORICAL_CHOICES = {
    "City": ["Lagos", "Abuja", "Kano"],
    "Property_Type": ["Duplex", "Studio", "Bungalow"],
    "Ownership": ["Sale", "Rent"],
    "Condition": ["New", "Old", "Fairly Used"],
    "Available_Utilities": ["Water", "Security", "Parking"],
}


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    data = {col: rng.choice(_CATEGORICAL_CHOICES[col], n) for col in CATEGORICAL_COLS}
    data.update(
        {
            "Latitude": rng.uniform(4.0, 13.0, n),
            "Longitude": rng.uniform(3.0, 13.0, n),
            "Bedrooms": rng.integers(1, 6, n).astype(float),
            "Bathrooms": rng.integers(1, 5, n).astype(float),
            "Size_sqm": rng.uniform(30.0, 500.0, n),
        }
    )
    assert set(NUMERIC_COLS) <= data.keys()
    data[TARGET] = rng.uniform(1e6, 5e8, n)
    return pd.DataFrame(data)

import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

# (town, state) pairs so every town belongs to exactly one state, as in the real data.
_LOCATIONS = [("Lekki", "Lagos"), ("Ikeja", "Lagos"), ("Gwarinpa", "Abuja")]
_TITLES = ["Detached Duplex", "Terraced Duplexes", "Block of Flats"]


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    # 60 rows keeps most categories above the encoder's min_frequency=10.
    rng = np.random.default_rng(0)
    n = 60
    towns, states = zip(*(_LOCATIONS[i] for i in rng.integers(0, len(_LOCATIONS), n)))
    data = {
        "title": rng.choice(_TITLES, n),
        "town": list(towns),
        "state": list(states),
    }
    assert set(CATEGORICAL_COLS) == data.keys()
    data.update({col: rng.integers(1, 10, n).astype(float) for col in NUMERIC_COLS})
    data[TARGET] = rng.uniform(1e7, 1e9, n)
    return pd.DataFrame(data)

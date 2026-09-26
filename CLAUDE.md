# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the venv before running anything (`.venv` already exists at repo root):

```bash
source .venv/bin/activate
pip install -r requirements.txt      # install/update deps
```

Train and compare all models (writes artifacts to `models/`):

```bash
python -m src.train
```

Launch the dashboard/predictor:

```bash
streamlit run app.py
```

There is no test suite, linter, or build step in this repo yet.

## Architecture

This is a small end-to-end ML project: train several regressors on Nigerian housing
listings, compare them, and serve the comparison + a live predictor through Streamlit.
The pipeline has three layers that must stay in sync:

- **`src/data.py`** — single source of truth for the schema: `CATEGORICAL_COLS`,
  `NUMERIC_COLS`, and `TARGET` (`Price_NGN`). `build_preprocessor()` returns the
  `ColumnTransformer` (one-hot for categoricals, `StandardScaler` for numerics) that
  every model pipeline uses. If a column is added/renamed in the CSV, update this
  file first — `train.py` and `app.py` both import their column lists from here.

- **`src/train.py`** — defines the model zoo in `MODEL_FACTORIES` (Linear, Ridge,
  Decision Tree, Random Forest, Gradient Boosting, and an MLP as the "neural net"
  entry). Each model is wrapped in its own `Pipeline(preprocess, model)` so every
  model sees identical features — never fit a bare estimator on raw `X` outside a
  pipeline, or the comparison stops being apples-to-apples. `train_all()` fits every
  pipeline, scores it (RMSE/MAE/R² on a held-out split, plus 5-fold CV R²), and
  writes both the fitted pipeline (`models/<slug>.joblib`) and the comparison table
  (`models/metrics.csv` / `.json`). The model name → filename slug is
  `name.lower().replace(" ", "_")` with parens stripped — `app.py` derives the same
  slug to load each pipeline, so don't rename a model without checking both files.

- **`app.py`** — Streamlit UI with two tabs: a comparison view (bar charts + table
  from `models/metrics.csv`) and a predictor (builds a one-row `DataFrame` from form
  inputs and calls `pipeline.predict`). `get_metrics_and_models()` auto-trains via
  `train_all()` if `models/` doesn't exist yet, so a fresh checkout works with just
  `streamlit run app.py` (no separate training step required, though running
  `python -m src.train` explicitly is faster to iterate on model changes since the
  Streamlit cache won't retrain on every rerun).

### Data note

`data/clean_nig_housing_dset.csv` has near-zero correlation between every feature
(size, bedrooms, bathrooms, lat/long, city, property type) and `Price_NGN` — all
models currently score negative R² (worse than predicting the mean). This is a
property of the dataset, not a bug in the pipeline; don't "fix" it by overfitting
or adding synthetic feature engineering. Categorical columns also contain inconsistent
spellings (e.g. `Duplex`/`Duplx`, `Apartment`/`Appartment`) that are intentionally
left as-is and one-hot encoded separately rather than merged.

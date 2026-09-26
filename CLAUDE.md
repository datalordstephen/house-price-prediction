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

Regenerate the README's comparison charts after retraining (reads `models/metrics.csv`, writes to `assets/`):

```bash
python -m scripts.generate_report_assets
```

Run the test suite (install dev deps first — `pytest` isn't in the runtime `requirements.txt`):

```bash
pip install -r requirements-dev.txt
python -m pytest            # -k <name> to run a single test
```

There is no linter or build step in this repo yet.

## Architecture

This is a small end-to-end ML project: train several regressors on Nigerian housing
listings, compare them, and serve the comparison + a live predictor through Streamlit.
The pipeline has four layers that must stay in sync:

- **`src/data.py`** — single source of truth for the schema: `CATEGORICAL_COLS`,
  `NUMERIC_COLS`, and `TARGET` (`Price_NGN`). `build_preprocessor()` returns the
  `ColumnTransformer` (one-hot for categoricals, `StandardScaler` for numerics) that
  every model pipeline uses. If a column is added/renamed in the CSV, update this
  file first — `train.py` and `app.py` both import their column lists from here.

- **`src/models/`** — one module per algorithm (`linear_regression.py`,
  `ridge_regression.py`, `decision_tree.py`, `random_forest.py`,
  `gradient_boosting.py`, `neural_net.py`), each exposing just `NAME: str` and
  `build() -> estimator`. Hyperparameters live in the individual module (e.g. Random
  Forest's `n_estimators=200, max_depth=12`); `src/config.py` holds the one constant
  shared across all of them (`RANDOM_STATE`). `src/models/__init__.py` collects them
  into `MODEL_REGISTRY` (an insertion-ordered `{name: build_fn}` dict — this order
  drives table/chart ordering elsewhere) and owns `slugify()`, the single name→filename
  mapping used to name `.joblib` artifacts. To add a new algorithm: create a module
  with that same `NAME`/`build()` contract and add it to `_MODEL_MODULES` — nothing
  else needs to change.

- **`src/metrics.py`** — plain `rmse`/`mae`/`r2` functions, kept dependency-free
  (numpy only) so they're trivial to unit test in isolation from sklearn pipelines.

- **`src/train.py`** — the orchestrator: for each `(name, build_model)` in
  `MODEL_REGISTRY`, wraps it in its own `Pipeline(preprocess, model)` so every model
  sees identical features — never fit a bare estimator on raw `X` outside a pipeline,
  or the comparison stops being apples-to-apples. `train_all(df=None, models_dir=MODELS_DIR)`
  fits every pipeline, scores it (RMSE/MAE/R² on a held-out split, plus 5-fold CV R²),
  and writes both the fitted pipeline (`models/<slug>.joblib`, slug from
  `src.models.slugify`) and the comparison table (`models/metrics.csv` / `.json`). The
  optional `df`/`models_dir` args exist so tests can inject a tiny synthetic dataset
  and a `tmp_path` without touching the real CSV or `models/`.

- **`app.py`** — Streamlit UI with two tabs: a comparison view (bar charts + table
  from `models/metrics.csv`) and a predictor (builds a one-row `DataFrame` from form
  inputs and calls `pipeline.predict`). `get_metrics_and_models()` auto-trains via
  `train_all()` if `models/` doesn't exist yet, so a fresh checkout works with just
  `streamlit run app.py` (no separate training step required, though running
  `python -m src.train` explicitly is faster to iterate on model changes since the
  Streamlit cache won't retrain on every rerun).

### Tests

`tests/conftest.py` provides a `tiny_df` fixture (40 synthetic rows matching the
real schema) so the suite never trains on the full 1862-row CSV or writes into the
real `models/` dir — `test_train.py` passes `tiny_df` + pytest's `tmp_path` into
`train_all()` for that reason. `test_models.py` checks the registry shape/contract
rather than any model's actual accuracy (accuracy is expected to be poor — see the
Data note below).

### Data note

`data/clean_nig_housing_dset.csv` has near-zero correlation between every feature
(size, bedrooms, bathrooms, lat/long, city, property type) and `Price_NGN` — all
models currently score negative R² (worse than predicting the mean). This is a
property of the dataset, not a bug in the pipeline; don't "fix" it by overfitting
or adding synthetic feature engineering. Categorical columns also contain inconsistent
spellings (e.g. `Duplex`/`Duplx`, `Apartment`/`Appartment`) that are intentionally
left as-is and one-hot encoded separately rather than merged.

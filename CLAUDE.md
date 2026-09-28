# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the venv before running anything (`.venv` at repo root; needs Python 3.12+ because
the pinned numpy requires it):

```bash
source .venv/bin/activate
pip install -r requirements.txt      # install/update deps (pinned direct dependencies)
```

Train and compare all models (writes artifacts to `artifacts/`):

```bash
python -m src.train
```

Launch the dashboard/predictor:

```bash
streamlit run app.py
```

Regenerate the README's charts after retraining (reads `artifacts/metrics.csv` and the
cleaned dataset, writes to `assets/`):

```bash
python -m scripts.generate_report_assets
```

Run the test suite (install dev deps first — `pytest` isn't in the runtime `requirements.txt`):

```bash
pip install -r requirements-dev.txt
python -m pytest            # -k <name> to run a single test
```

There is no linter or build step in this repo yet. `.github/workflows/tests.yml` runs
`python -m pytest` on Python 3.12 on every push to `main` and every pull request.

## Architecture

This is a small end-to-end ML project: train several regressors on real Nigerian housing
listings (Kaggle "Nigeria Houses and Prices", scraped from nigeriapropertycentre.com),
compare them, and serve the comparison + a live predictor through Streamlit. The
pipeline has four layers that must stay in sync:

- **`src/data.py`** — single source of truth for the schema and cleaning:
  `DATA_PATH` (`data/nigeria_houses_data.csv`), `CATEGORICAL_COLS`
  (`title`, `town`, `state`), `NUMERIC_COLS` (`bedrooms`, `bathrooms`, `toilets`,
  `parking_space`) and `TARGET` (`price`, in naira). `load_data()` always runs
  `clean_data()`, which (printing row counts at each step) drops exact duplicate rows
  (24,326 → 13,888; otherwise the same listing leaks into train and test) and keeps
  `PRICE_MIN <= price <= PRICE_MAX` (₦5M–₦2B → 13,714 rows), then drops
  `state == MISLABELLED_STATE` ("Anambara", whose 141 listings are really Lagos/Abuja/
  Rivers/... towns → 13,573 rows). The bounds are fixed module constants, deliberately
  not quantiles computed at load time.
  `build_preprocessor()` returns the `ColumnTransformer` every model uses:
  `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` for
  categoricals (about half of the 183 towns have < 10 listings and share an
  "infrequent" column) and `StandardScaler` for numerics. If a column is added/renamed
  in the CSV, update this file first — `train.py`, `app.py` and the tests import their
  column lists from here.

- **`src/models/`** — one module per algorithm (`linear_regression.py`,
  `ridge_regression.py`, `decision_tree.py`, `random_forest.py`,
  `gradient_boosting.py`, `neural_net.py`), each exposing just `NAME: str` and
  `build() -> estimator` (the bare estimator — no target transform, no preprocessing).
  Hyperparameters live in the individual module; `src/config.py` holds the one constant
  shared across them (`RANDOM_STATE`). `src/models/__init__.py` collects them into
  `MODEL_REGISTRY` (an insertion-ordered `{name: build_fn}` dict, 6 models, with
  Linear Regression — the baseline the others are compared against — first), owns
  `slugify()` (the single name→filename mapping for `.joblib` artifacts) and
  `MODEL_COLORS` (one fixed colour per model, used by both `app.py` and the report
  script). To add a
  new algorithm: create a module with the same `NAME`/`build()` contract, add it to
  `_MODEL_MODULES`, and give it a `MODEL_COLORS` entry (a test checks every model has one).

- **`src/metrics.py`** — plain `rmse`/`mae`/`r2`/`mdape` functions (MdAPE = median
  absolute percentage error, returned as a fraction), kept dependency-free
  (numpy only) so they're trivial to unit test in isolation from sklearn pipelines.

- **`src/train.py`** — the orchestrator. `build_pipeline(build_model)` is the one place
  that wraps a model: `Pipeline(preprocess, TransformedTargetRegressor(regressor=build_model(),
  func=np.log, inverse_func=np.exp))`. Prices are heavily right-skewed, so every model
  is fitted on log(price); `predict()` still returns naira. Use this helper for every
  fit — never fit a bare estimator on raw `X`, or put the log transform in a model
  module, or the comparison stops being apples-to-apples. `train_all(df=None,
  artifacts_dir=ARTIFACTS_DIR)` does an 80/20 split, fits each pipeline on the train
  split and reports on the test split: `R2_log` (R² on log(price) — the headline
  metric), `RMSE` and `MAE` (both in naira), `MdAPE` (the typical % error), plus
  `CV_R2_log_mean`/`_std` from 5-fold CV run on `X_train` only, scored with
  `make_scorer(r2_log)`. The table is
  sorted by `R2_log` and written to `artifacts/metrics.csv` / `.json`, alongside each
  fitted pipeline at `artifacts/<slug>.joblib`. The optional `df`/`artifacts_dir` args
  exist so tests can inject a tiny synthetic dataset and a `tmp_path` without touching
  the real CSV or `artifacts/`. (`artifacts/` — not `models/` — precisely to avoid
  colliding with the `src/models/` package name.)

- **`app.py`** — Streamlit UI with two tabs: a comparison view (R2_log and RMSE bar
  charts + full metrics table from `artifacts/metrics.csv`) and a predictor (state →
  towns filtered to that state → property type, plus integer inputs bounded by the
  data's min/max; builds a one-row `DataFrame` and calls `pipeline.predict`). Tree
  models also get an
  overall feature-importance chart, read from
  `pipeline.named_steps["model"].regressor_` because the estimator sits inside the
  `TransformedTargetRegressor`. Use `width="stretch"`, not the deprecated
  `use_container_width`. `get_metrics_and_models()` auto-trains via `train_all()` if
  `artifacts/` doesn't exist yet, so a fresh checkout works with just
  `streamlit run app.py`.

- **`scripts/generate_report_assets.py`** — writes `assets/model_comparison.png`
  (R2_log + RMSE per model), `price_by_bedrooms.png` and `price_by_title.png`
  (median price per group) for the README.

### Tests

`tests/conftest.py` provides a `tiny_df` fixture (60 synthetic rows matching the real
schema — enough that `min_frequency=10` doesn't fold every category into "infrequent")
so the suite never trains on the full dataset or writes into the real `artifacts/`
dir — `test_train.py` passes `tiny_df` + pytest's `tmp_path` into `train_all()` for
that reason. `test_models.py` checks the registry shape/contract (6 models, Linear
Regression first, a colour per model) rather than any model's accuracy. `test_data.py` covers
`clean_data()` (duplicates and both price bounds) and loads the real CSV once.

### Data notes

- Current results (test set): MLP best at R2_log ≈ 0.72 (CV ≈ 0.70), with the Linear
  Regression baseline, Ridge and Gradient Boosting around 0.69; MAE ≈ ₦68–71M,
  MdAPE ≈ 33–35%. If a change moves these a lot, suspect a pipeline bug (e.g. leakage
  or a dropped transform) before celebrating or tuning.
- Because the target is log(price), back-transformed predictions are median-like:
  mean prediction is ~20% below the mean actual price. This is expected, not a bug.
  MdAPE is the metric to quote for typical error (MAE is inflated by the priciest
  listings).
- `title` values are the site's own categories (`Detached Duplex`, `Terraced Duplexes`,
  `Block of Flats`, ...); keep them as-is.
- Some town names exist in more than one state, which is why the app filters towns by
  the chosen state.
- The previous dataset (`clean_nig_housing_dset.csv`) was removed because it looked
  synthetic — every model scored negative R² on it. Don't reintroduce it.

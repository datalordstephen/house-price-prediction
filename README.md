# Nigerian House Price Prediction

Predicts the asking price of Nigerian houses from their listing details, comparing
six regression algorithms, from linear regression (the baseline) to a neural
network. A Streamlit dashboard shows the comparison and runs a live predictor.

## Dataset

[Nigeria Houses and Prices Dataset](https://www.kaggle.com/datasets/abdullahiyunus/nigeria-houses-and-prices-dataset)
on Kaggle (`data/nigeria_houses_data.csv`): 24,326 listings scraped from
nigeriapropertycentre.com.

| Column | Role |
|---|---|
| `title` (property type), `town`, `state` | categorical features |
| `bedrooms`, `bathrooms`, `toilets`, `parking_space` | numeric features |
| `price` (₦) | target |

### Cleaning

`src/data.py::clean_data()` runs on every load:

| Step | Rows |
|---|---:|
| Raw CSV | 24,326 |
| Drop exact duplicate rows (10,438 repeated listings) | 13,888 |
| Keep ₦5M ≤ price ≤ ₦2B | 13,714 |
| Drop state "Anambara" (mislabelled: its towns are in Lagos, Abuja, Rivers, ...) | 13,573 |

Without the duplicate removal, the same listing would appear in both the train and
test sets. The price bounds are fixed constants, roughly the 1st and 99.5th
percentiles. They remove implausible values such as a ₦1.8 trillion typo.

Towns with fewer than 10 listings (about half of the 183) share a single
"infrequent" one-hot column.

Price rises clearly with bedroom count and property type:

![Median price by bedrooms](assets/price_by_bedrooms.png)
![Median price by property type](assets/price_by_title.png)

### Why a log target

Prices are heavily right-skewed: the median is ₦75M but the top listings reach ₦2B.
Every model is fitted on `log(price)` using `TransformedTargetRegressor`, and its
predictions are converted back to naira. This keeps the handful of most expensive
listings from dominating the fit and lets errors scale with price. For the same
reason, the headline metric is **R² on log(price)**.

## Setup

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train and compare all models. This writes the fitted pipelines and the metrics to
`artifacts/`:

```bash
python -m src.train
```

Launch the dashboard. It trains the models on first run if `artifacts/` doesn't
exist yet:

```bash
streamlit run app.py
```

Regenerate the charts in this README after retraining:

```bash
python -m scripts.generate_report_assets
```

## Results

Each model is trained inside the same `Pipeline`: one-hot encoding for the
categorical features, standard scaling for the numeric ones, and a log-price target.
The data is split 80/20 into train and test sets. Cross-validation (5-fold) runs on
the training split only.

![Model comparison chart](assets/model_comparison.png)

| Model | R² (log price) | RMSE (₦M) | MAE (₦M) | MdAPE | CV R² (log price), mean ± std |
|---|---:|---:|---:|---:|---:|
| **Neural Net (MLP)** | 0.720 | 161.4 | 67.7 | 32.9% | 0.704 ± 0.019 |
| Ridge Regression | 0.689 | 166.3 | 71.0 | 35.0% | 0.692 ± 0.015 |
| Linear Regression *(baseline)* | 0.688 | 166.1 | 71.1 | 35.0% | 0.691 ± 0.015 |
| Gradient Boosting | 0.685 | 166.7 | 70.1 | 35.3% | 0.684 ± 0.011 |
| Random Forest | 0.683 | 164.3 | 70.7 | 35.2% | 0.676 ± 0.017 |
| Decision Tree | 0.606 | 171.5 | 75.6 | 38.7% | 0.601 ± 0.019 |

R² is measured on log price, the scale the models are trained on. RMSE and MAE are
in naira, after converting predictions back from the log scale. MdAPE is the median
absolute percentage error: half of the test-set predictions are within that
percentage of the listed price.

MAE is dragged up by a small number of very expensive listings (the median test price
is ₦75M, the mean ₦156M), so MdAPE (about ±33–35%) is the better measure of a
model's typical error. Converting predictions back from log price gives roughly the
*median* price for a given listing, not the mean, so the models under-predict the
average price by about 20% (mean prediction ÷ mean actual price is 0.77–0.83 on the
test set).

**Best model: Neural Net (MLP)**, with R² = 0.720 on log price (0.704 in
cross-validation) and the lowest RMSE, MAE and MdAPE. It beats the Linear Regression
baseline (0.688) only modestly, and by just 0.013 in cross-validation. The more
flexible models therefore add little over a linear fit on these features. Much of
the remaining error likely comes from what the listings don't record: floor area,
age, condition and exact location.

## Why we changed datasets

The project originally used `clean_nig_housing_dset.csv`, which turned out to look
synthetic:

- property type was independent of bedroom count;
- rent and sale prices had identical distributions;
- every model scored a negative R², doing worse than predicting the mean.

The Kaggle dataset consists of real scraped listings, and its prices follow the
features you'd expect.

## Project structure

- `src/data.py`: schema, cleaning, and the shared preprocessing pipeline
- `src/models/`: one module per algorithm, plus the registry and chart colours
- `src/train.py`: trains, scores, and saves every model
- `src/metrics.py`: RMSE, MAE, R²
- `app.py`: Streamlit dashboard (model comparison + interactive predictor)
- `scripts/generate_report_assets.py`: regenerates the charts in this README

# Report & Slides Guide: Nigerian House Price Prediction

A quick reference for writing the project report and slides. It walks through each
stage of the ML life cycle and says what we did and why. Unless stated otherwise,
every number here comes from the held-out test set. The charts are in `assets/`,
and the full details are in `README.md`.

## At a glance

| | |
|---|---|
| Task | Predict a house's asking price (₦) from its listing details (regression) |
| Data | 24,326 Kaggle listings from nigeriapropertycentre.com → 13,573 after cleaning |
| Models | 6: Linear Regression (baseline), Ridge, Decision Tree, Random Forest, Gradient Boosting, Neural Net (MLP) |
| Best model | Neural Net (MLP): R² = 0.720 on log price, half of predictions within ±32.9% |
| Deliverable | Streamlit web app: model comparison + live price predictor |

---

## 1. Problem definition

- **Goal:** estimate the asking price of a Nigerian house from the details in its
  listing: property type, town, state, and the number of bedrooms, bathrooms, toilets
  and parking spaces.
- **Type:** supervised regression, because the target (`price`) is a continuous
  amount in naira.
- **Success criteria:** beat a simple Linear Regression baseline, and report the
  error in a form a non-technical reader understands ("typically within ±X%").

## 2. Data collection

- **Source:** [Nigeria Houses and Prices Dataset](https://www.kaggle.com/datasets/abdullahiyunus/nigeria-houses-and-prices-dataset)
  (Kaggle). It has 24,326 listings scraped from nigeriapropertycentre.com and 8
  columns: `bedrooms`, `bathrooms`, `toilets`, `parking_space`, `title` (property
  type), `town`, `state`, `price`.
- **We replaced our first dataset.** The original file (1,862 listings) looked
  synthetic:
  - property type had no relationship with bedroom count;
  - rent and sale prices had identical distributions;
  - every model scored a *negative* R², meaning it did worse than always guessing the
    average price.

  *This makes a good slide: checking data quality before trusting any model.*

## 3. Data cleaning

| Step | Rows left | Why |
|---|---:|---|
| Raw data | 24,326 | |
| Remove exact duplicate rows | 13,888 | 10,438 (43%) were repeated listings. Left in, the same house could appear in both the training and test sets, which inflates scores. |
| Keep prices from ₦5M to ₦2B | 13,714 | Removes typos and implausible values (the raw maximum was ₦1.8 trillion). The bounds are roughly the 1st and 99.5th percentiles. |
| Drop state "Anambara" | 13,573 | A mislabelled state: its 141 listings are towns in Lagos, Abuja, Rivers and other states (Lekki, Ajah, Ikoyi, ...), not Anambra. |

There were no missing values. The cleaning code is `clean_data()` in `src/data.py`.

## 4. Exploratory data analysis

- **Prices are heavily right-skewed.** The median is ₦75M but the mean is ₦157M
  (skewness 3.84). A few very expensive listings pull the average up. Taking the log
  of the price brings the skewness down to 0.30.
- **Price rises with bedrooms and property type.** The median price is ₦25–45M for
  1–3 bedrooms and ₦350M for 9 bedrooms. By property type, it ranges from ₦22M
  (terraced bungalow) to ₦130M (detached duplex). Figures: `assets/price_by_bedrooms.png`,
  `assets/price_by_title.png`.
- **The data is concentrated.** Lagos and Abuja make up 85% of the listings. There
  are 183 towns, and about half of them have fewer than 10 listings.

## 5. Feature engineering & preprocessing

- **Categorical features** (`title`, `town`, `state`) are one-hot encoded. Categories
  with fewer than 10 listings are grouped into a single "infrequent" column, so rare
  towns don't each get a column with almost no data behind it. This gives 111 input
  features in total.
- **Numeric features** (bedrooms, bathrooms, toilets, parking) are standardised
  (StandardScaler).
- **Log target:** every model is trained on `log(price)`, and its predictions are
  converted back to naira. This handles the skew and stops the most expensive
  houses from dominating training.
- All models share the same preprocessing pipeline, so the comparison is fair.

## 6. Model selection & training

| Model | Why include it | Key settings |
|---|---|---|
| Linear Regression | Baseline: the simplest reasonable model | defaults |
| Ridge Regression | Linear model with regularisation | alpha = 1.0 |
| Decision Tree | Simple non-linear model, easy to interpret | max depth 8 |
| Random Forest | Ensemble of trees; reduces overfitting | 200 trees, max depth 12 |
| Gradient Boosting | Trees built sequentially to correct earlier errors | sklearn defaults |
| Neural Net (MLP) | Most flexible model | 2 hidden layers (64, 32), early stopping |

- **Split:** 80% training (10,858 rows) and 20% test (2,715 rows), with a fixed
  random seed (42) so the results are reproducible.
- **Cross-validation:** 5-fold, on the training set only. The test set stays unseen
  until the final evaluation.
- No hyperparameter tuning was done. The settings above were chosen by hand.

## 7. Evaluation

**Metrics (what to say on the slide):**
- **R² (log price):** the share of variation in log price that the model explains.
  0 means no better than guessing the average, and 1 is perfect. This is our headline
  metric.
- **RMSE / MAE (₦):** the average size of the error in naira. RMSE punishes large
  misses more heavily than MAE.
- **MdAPE:** median absolute percentage error. Half of the predictions are within
  this percentage of the real price. This is the clearest "typical error" figure.
- **CV R²:** R² averaged over the 5 cross-validation folds, which shows the result
  isn't a lucky split.

| Model | R² (log) | RMSE (₦M) | MAE (₦M) | MdAPE | CV R² (log) |
|---|---:|---:|---:|---:|---:|
| **Neural Net (MLP)** | **0.720** | **161.4** | **67.7** | **32.9%** | 0.704 ± 0.019 |
| Ridge Regression | 0.689 | 166.3 | 71.0 | 35.0% | 0.692 ± 0.015 |
| Linear Regression *(baseline)* | 0.688 | 166.1 | 71.1 | 35.0% | 0.691 ± 0.015 |
| Gradient Boosting | 0.685 | 166.7 | 70.1 | 35.3% | 0.684 ± 0.011 |
| Random Forest | 0.683 | 164.3 | 70.7 | 35.2% | 0.676 ± 0.017 |
| Decision Tree | 0.606 | 171.5 | 75.6 | 38.7% | 0.601 ± 0.019 |

Figure: `assets/model_comparison.png`

**Key findings:**
1. **The MLP is the best model on every metric**, but only modestly. It beats the
   linear baseline by 0.032 R² on the test set and by 0.013 in cross-validation.
   About 0.70 is the realistic figure to quote.
2. **The single Decision Tree is clearly the weakest.** One tree overfits, while the
   ensembles (Random Forest, Gradient Boosting) fix most of that.
3. **Location, bedrooms and property type drive price.** In the tree models, these
   three account for about 90% of feature importance: town about 35%, bedrooms about
   28%, property type about 27%. Ikoyi is the single most important town.
4. **MAE looks large (about ₦68M)** because a few very expensive listings drag it up.
   MdAPE (about 33%) better reflects the typical error.
5. **The models under-predict the average price by about 20%.** A model trained on
   log price predicts something close to the *median* price for a listing, not the
   mean. This is a known side effect of the log transform, not a bug.

## 8. Deployment

- A **Streamlit web app** (`streamlit run app.py`) with two tabs:
  - **Model Comparison:** charts and the full metrics table.
  - **Predict a Price:** choose a state, then a town within it, then the property type
    and room counts. You get a price estimate from any of the six models. Tree models
    also show which features matter most overall.
- The trained models are saved to `artifacts/`. If they're missing, the app trains
  them automatically on first launch.

## 9. Maintenance & reproducibility

- **Automated tests:** 13 `pytest` tests cover data cleaning, metrics, the model
  registry and training. GitHub Actions runs them on every push and pull request.
- **Reproducible:** fixed random seed, pinned library versions in
  `requirements.txt`, and one command to retrain (`python -m src.train`).
- **Extensible:** adding a new model means adding one small file in `src/models/`.

## Limitations & future work

- **Missing information:** the listings don't include floor area, age, condition or
  exact location. These probably explain much of the remaining ±33% error.
- **Geographic bias:** 85% of the data is from Lagos and Abuja, so predictions for
  other states are less reliable.
- **Asking prices, not sale prices:** we model what sellers ask, which may differ
  from what buyers pay.
- **No tuning:** hyperparameter search (e.g. grid search with cross-validation) could
  improve the tree models and the MLP.
- **Next steps:** add richer features, tune the models, and show a price *range*
  instead of a single number.

## Suggested slide outline

1. **Title & team**
2. **Problem:** why predict house prices, and why it's a regression task
3. **Data:** Kaggle source, 8 columns, why we replaced the first dataset
4. **Cleaning:** the table from section 3 (24,326 → 13,573)
5. **EDA:** the skew (median vs mean) plus the two price charts
6. **Preprocessing:** one-hot encoding, scaling, log target
7. **Models:** the six models and the 80/20 + 5-fold CV setup
8. **Results:** `model_comparison.png` plus the metrics table
9. **Insights:** the five key findings
10. **Demo:** Streamlit app screenshots (comparison tab + a prediction)
11. **Limitations & future work**

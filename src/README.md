# Models in src/

This README describes the models and helper utilities implemented in the `src/` folder, how they transform inputs, and where to look for the key functions and artifacts used across training and inference.

## High-level contract (inputs / outputs)

- Inputs:
  - Prices CSV: must contain a date-like column (one of `date`, `datetime`, `timestamp`), and OHLCV-like columns (columns starting with `open`, `high`, `low`, `close`, `volume`). A `ticker` column is supported if the file contains multiple tickers.
  - News CSV (optional for news-aware models): should include a publication date column (the code uses `pub_date` / `pub_day`) and a text field (default column name `text`). Optionally a numeric sentiment column can be supplied and its name passed via `--news_score_col`.

- Outputs (per-model, under `output/<TICKER>/<model_folder>/`):
  - `model.keras` — saved Keras model (TensorFlow SavedModel/keras HDF5-like file).
  - `scaler.joblib` — MinMaxScaler fit on the training price features (used for consistent inference).
  - `predictions.csv` — CSV containing the test predictions (columns `date`, `y_true`, `y_pred`).
  - `metrics.json` — JSON with MAE, MAPE, simple accuracy proxy and feature metadata.


## Shared utilities and conventions

- Feature selection: `pick_features()` picks the first column whose lowercase starts with `open`, `high`, `low`, `close`, `volume` in that order. The `close` column is used as the prediction target.
- Scaling: `MinMaxScaler` (from scikit-learn) is used in all training scripts. For inference, `infer.py` tries to load `scaler.joblib` from the model folder; if not found it will fit a MinMaxScaler on the provided prices at inference time (this preserves functionality but may create distribution shifts).
- Sequences: `make_sequences_multifeat()` produces input sequences of shape (N_samples, seq_len, n_features) and corresponding targets. For MLP models the sequences are flattened to shape (N_samples, seq_len * n_features).
- Default hyperparameters across scripts: seq_len=20, epochs=200, batch_size=64. Optimizers/losses are set per-model (see below).


## Models

Each model file lives in `src/` and can be run directly as a script (e.g. `python src/lstm_advanced.py --ticker AMZN --prices test_data/AMZN.csv`). The important files are:

- `src/mlp_advanced.py`
  - Architecture: Multilayer Perceptron applied to flattened time windows.
  - Input: sequences flattened to (seq_len * n_features,), i.e. all features in the window concatenated.
  - Network:
    - Input layer: shape=(seq_len * n_features,)
    - Dense(256, relu)
    - Dropout(0.2)
    - Dense(128, relu)
    - Dropout(0.1)
    - Dense(1, linear)
  - Optimizer: AdamW
  - When to use: baseline model that ignores temporal recurrence but can be faster to train and smaller on-device.
  - Key functions: `build_mlp()`, `make_sequences_multifeat()`.

- `src/lstm_advanced.py`
  - Architecture: stacked LSTM layers over the input window (no external sentiment features).
  - Input: sequences of shape (seq_len, n_features)
  - Network:
    - Input layer: (seq_len, n_features)
    - LSTM(96, return_sequences=True)
    - Dropout(0.2)
    - LSTM(48, return_sequences=True)
    - Dropout(0.1)
    - LSTM(24)
    - Dense(1, linear)
  - Optimizer: Nadam
  - Key functions: `build_lstm()`, `make_sequences_multifeat()`.

- `src/lstm_news_advanced.py`
  - Architecture: same LSTM stack as `lstm_advanced`, but the input features are augmented with a daily sentiment value (one extra feature).
  - How sentiment is used: compute_daily_sentiment(news_df, ...) aggregates news per-day and produces a single scalar sentiment per day. `align_with_sentiment()` aligns each price row to the sentiment for that day. `make_sequences_multifeat_with_sent()` attaches the prediction-day sentiment into every step of the input window (by default) producing sequences of shape (seq_len, n_features + 1).
  - Network: identical LSTM stack as `lstm_advanced` but `n_features_plus1 = n_features + 1`.
  - Sentiment sources:
    - If you pass `--news_score_col NAME` and that column exists in the news CSV, the code uses its daily mean directly.
    - Otherwise it tries VADER (if the `vaderSentiment` package is installed) and falls back to neutral (zero) sentiment if VADER isn't available.
  - Optimizer: AdamW
  - Key functions: `compute_daily_sentiment()`, `align_with_sentiment()`, `make_sequences_multifeat_with_sent()`.

- `src/lstm_finbert.py`
  - Purpose: identical training/inference flow to `lstm_news_advanced.py` but with an attempt to compute sentiment using ProsusAI/FinBERT (via Hugging Face Inference API) before falling back to VADER.
  - HF behavior:
    - If an environment variable `HUGGINGFACE_API_KEY` (or `HF_API_KEY` / `HUGGINGFACE_TOKEN`) is present the code will call the Hugging Face Inference API for `ProsusAI/finbert` in batches, map the returned labels to signed scores (+score for "positive", -score for "negative", 0 otherwise), and average per day.
    - The HF path uses small batching and a small sleep between batches to reduce rate-limit pressure. The code prints HF API errors to stderr and falls back gracefully if the HF call fails.
  - Fallbacks: if HF key not present or HF inference fails then VADER is attempted; if that is unavailable the code uses zero (neutral) sentiment per day.
  - Key functions: `compute_daily_sentiment()` (HF + fallbacks), `align_with_sentiment()`, `make_sequences_multifeat_with_sent()`.


## How to integrate sentiment + price features

- The news-aware models (LSTM+News and LSTM FinBERT) expect the news CSV to contain a per-article `pub_date`/`pub_day` column that can be converted to the price date format (by default `%Y-%m-%d`). The helper `compute_daily_sentiment()` returns a series indexed by day (string dates) with the average sentiment for that day.
- `align_with_sentiment()` will merge that daily series to the price dates and produce a column vector of shape (n_rows, 1) which is then appended as an extra feature to the price feature matrix. The sequence builder will align the sentiment value for the target day (the day being predicted) with the corresponding input window.


## Inference and integration with `infer.py`

- `infer.py` (top-level script) contains a `ModelWrapper` that loads `<output>/<TICKER>/<model_folder>/model.keras` and `<...>/scaler.joblib` when present. It re-creates the same sequence shapes used during training (calls the sequences builders above) and writes predictions to `results/<TICKER>/test/prediction/<model_name>_predictions.csv`.
- If the model requires sentiment, `infer.py` will attempt to import sentiment helpers from `src/lstm_news_advanced` and then from `src/lstm_finbert` (so either code path will satisfy the dependency).
- If a `scaler.joblib` is not present, `infer.py` will create and fit a `MinMaxScaler` on the supplied prices and emit a warning; results will still be written but may not be strictly comparable to training-time scaling.


## Practical tips and troubleshooting

- HF FinBERT limits: the Hugging Face Inference API is subject to rate limits and quota. Provide a valid key in `HUGGINGFACE_API_KEY` (or `HF_API_KEY` / `HUGGINGFACE_TOKEN`) and be mindful of the per-request batch sizes. If you hit rate limits, the code will print errors and fall back to VADER.
- NYT news fetcher: use `fetch_nyt_test_news.py` to build `test_news_data/<TICKER>.csv` files. That script supports key rotation and backoff; ensure you provide valid NYT keys.
- Reproducing results: save the `scaler.joblib` and `model.keras` produced by training to ensure inference uses the same scaling. When sharing models, include the `metrics.json` so downstream evaluation/visualization can label model accuracy.
- Missing or noisy columns: the `pick_features()` logic uses prefix matching. If your CSV has different column names, rename to conventional names or update the helper functions.


## Files to inspect

- `src/mlp_advanced.py` — MLP training; `build_mlp()`
- `src/lstm_advanced.py` — LSTM training; `build_lstm()`
- `src/lstm_news_advanced.py` — LSTM with news; `compute_daily_sentiment()`, `align_with_sentiment()`, `make_sequences_multifeat_with_sent()`
- `src/lstm_finbert.py` — FinBERT-backed sentiment path; same helpers as `lstm_news_advanced.py` but `compute_daily_sentiment()` calls HF first.


## Quick checklist before running training or inference

- [ ] `prices.csv` or `test_data/<TICKER>.csv` present and contains date + OHLCV-like columns.
- [ ] For news-aware models: `news.csv` or `test_news_data/<TICKER>.csv` present and contains `pub_date` / `pub_day` and `text`.
- [ ] If using FinBERT: set `HUGGINGFACE_API_KEY` (or `HF_API_KEY` / `HUGGINGFACE_TOKEN`).
- [ ] For reproducible inference: keep `scaler.joblib` saved alongside `model.keras`.


If you'd like, I can also add a small diagram or a short example walkthrough showing the exact sequence of calls infer.py makes to these helpers, or generate a minimal unit test that checks sequence-building outputs for a small synthetic input. Which would you prefer next?
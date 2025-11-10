# Stock prediction project

This repository contains three model implementations (MLP, LSTM, LSTM+News) for short-term stock price prediction, training utilities, inference and evaluation tools, and simple analysis scripts. The workspace layout (important files/folders):

- `mlp_advanced.py`, `lstm_advanced.py`, `lstm_news_advanced.py` — training scripts for each model.
- `infer.py` — lightweight inference wrapper to load a saved model + scaler and run predictions on any prices CSV.
- `evaluate_predictions.py` — evaluates predictions CSV(s) and writes results into `results/<TICKER>/`.
- `analyze_predictions.py`, `combine_returns.py` — helper analysis scripts used in the repository.
- `output/` — model outputs organized by ticker and model (e.g. `output/MSFT/mlp_advanced/`). Each model output folder will contain:
  - `model.keras` (saved Keras model)
  - `scaler.joblib` (saved MinMaxScaler used at training time)
  - `predictions.csv` (predictions for the configured test range)
  - `metrics.json` (MAE, MAPE, other metadata)
- `results/` — evaluation summaries and combined tables created by `evaluate_predictions.py` and helpers.
- `prices.csv`, `news.csv` — example input data used by the training scripts.
- `requirements.txt` — Python dependencies (install in a virtualenv).

## Quick setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Verify your Python + TensorFlow install. Training requires TensorFlow; GPU optional.

## Train a model (one-off per ticker)

Each training script has CLI arguments. The scripts default to training on `--ticker MSFT` and saving under `output/<TICKER>/<model_folder>`.

Examples:

- Train the MLP on the default ticker (uses date-range splitting by default):

```bash
python3 mlp_advanced.py --ticker MSFT
```

- Train the LSTM for a different ticker and custom ranges:

```bash
python3 lstm_advanced.py --ticker GOOGL --train-start 2019-01-01 --train-end 2021-12-31 --test-start 2022-01-01 --test-end 2022-06-30
```

- Train the LSTM+News variant (point `--news` to a CSV of news items):

```bash
python3 lstm_news_advanced.py --ticker AMZN --news news.csv
```

Notes:
- The scripts create `output/<TICKER>/<model_folder>/` and save `model.keras` and `scaler.joblib`. The scaler is saved so you can run inference later with identical feature scaling.
- By default the training code fits a MinMaxScaler on the entire `feats` array; the scripts now save that scaler so later inference can reuse it.

## Run the saved model on other datasets (single-line predict)

If you trained a model (e.g. `output/MSFT/mlp_advanced/model.keras`) you can reuse it on any other prices CSV with `infer.py`.

CLI example (preferred — uses saved scaler if present):

```bash
python3 infer.py --model_dir output/MSFT/mlp_advanced --prices path/to/test_prices.csv --outdir output/NEW/mlp_eval
```

This will write `output/NEW/mlp_eval/predictions.csv` containing `date,y_true,y_pred` (same format as training outputs).

Programmatic usage (from Python):

```python
from infer import ModelWrapper
mw = ModelWrapper(model_dir='output/MSFT/mlp_advanced')
df = mw.predict_from_csv('path/to/test_prices.csv')  # DataFrame with date, y_true, y_pred
```

Notes:
- If `scaler.joblib` is present in the model folder, `infer.py` will use it. If not, it will fit a new MinMaxScaler on the provided prices CSV (this changes input scaling vs training and may affect predictions).
- For LSTM+News, if the model expects a sentiment feature, pass `--news path/to/news.csv` to compute daily sentiment aligned with prices. If omitted, sentiment defaults to zero.

## Evaluate predictions and compute returns

`evaluate_predictions.py` evaluates predictions CSVs and writes summaries to `results/<TICKER>/`.

Batch-evaluate the three models for a ticker (the script searches common output locations):

```bash
python3 evaluate_predictions.py --ticker MSFT
```

Evaluate a single CSV and save to a chosen filename:

```bash
python3 evaluate_predictions.py --csv output/MSFT/mlp_advanced/predictions.csv --ticker MSFT --out mlp_returns.csv
```

The evaluation computes three strategy end-values starting from `initial_cash` (default 10000):
- `buy_and_hold`: buy at the first actual price and hold to last price
- `predicted_signal`: be fully invested only on days where the model predicts the price will rise (y_pred[t] > y_pred[t-1])
- `actual_signal`: use the realized (true) up-days as the signal (not realistic forward-looking, but a useful benchmark)

Helper scripts:
- `analyze_predictions.py` — computes sign accuracy, invested days and per-invested-day returns for each predictions file.
- `combine_returns.py` — merges per-model return CSVs into `results/<TICKER>/test/combined_returns.csv` (and `combined_returns.md`/`combined_returns.txt` in some cases).

## Reproduce the repo results (recommended quick steps)

1. Train models for a ticker (or reuse pre-trained in `output/<TICKER>/` if present).
2. Run `infer.py` on any test prices you want.
3. Run `evaluate_predictions.py --ticker <TICKER>` to write `results/<TICKER>/*.csv`.
4. Optionally run `python3 combine_returns.py` to build a combined table across models.

## File formats

- predictions CSV: `date,y_true,y_pred` (date format YYYY-MM-DD)
- model folder: `model.keras` + `scaler.joblib` + `metrics.json` + `predictions.csv`
- results summary CSVs: contain columns `initial_cash,buy_and_hold,predicted_signal,actual_signal`

## Troubleshooting

- Missing packages: ensure `venv` is active and `pip install -r requirements.txt` completed.
- TensorFlow errors (GPU drivers): switch to CPU TF wheel or ensure CUDA/cuDNN versions match.
- Input CSV parsing errors: `load_prices()` tries to auto-detect date and OHLC(V)-like columns; ensure your file has a date column named `date`/`datetime`/`timestamp` (case-insensitive) and price columns named like `open,high,low,close,volume` or similar prefixes.
- If inference predictions look wrong, confirm `scaler.joblib` exists in the model folder and is used by `infer.py` (it will print a warning if it fits a new scaler instead).

## Next steps / suggestions

- Add transaction costs and position sizing to `evaluate_predictions.py` for more realistic return estimates.
- Add unit/integration tests for end-to-end training -> inference -> evaluation flows.
- Add a small experiment harness to sweep hyperparameters (LR, dropout, hidden sizes) and collect metrics.
- Persist training logs (TensorBoard) and a reproducibility manifest (git SHA, python env) alongside model outputs.

If you want, I can also:
- Add a one-line shell wrapper to call `model.predict(test.csv)` directly (thin wrapper around `infer.py`).
- Create a PDF/HTML report that includes prediction plots and the combined returns table.

---
Happy to expand any section or add example commands tailored to the tickers you care about.

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

# Stock prediction project

This repository contains three model implementations (MLP, LSTM, LSTM+News) for short-term stock price prediction, plus training, inference and evaluation utilities and small analysis helpers.

High-level layout
- `mlp_advanced.py`, `lstm_advanced.py`, `lstm_news_advanced.py` — training scripts that save model artifacts under `output/<TICKER>/<model_folder>/`.
- `infer.py` — lightweight inference wrapper to load a saved `model.keras` + `scaler.joblib` (if present) and produce `predictions.csv` (columns: `date,y_true,y_pred`).
- `evaluate_predictions.py` — evaluates prediction CSV(s) and writes per-model return summaries under `results/<TICKER>/` (train vs test locations described below).
- `combine_returns.py`, `analyze_predictions.py` — helper scripts to merge or analyze per-model results.
- `output/` — model outputs organized by ticker and model (e.g. `output/MSFT/mlp_advanced/`). Typical files in a model folder:
  - `model.keras` — saved Keras model
  - `scaler.joblib` — saved MinMaxScaler used at training time (optional but recommended)
  - `predictions.csv` — predictions for the configured test range (date,y_true,y_pred)
  - `metrics.json` — training/validation metrics
- `results/` — evaluation summaries and combined tables created by `evaluate_predictions.py` and helpers.

## Quick setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Verify Python + TensorFlow installation. Training needs TensorFlow; GPU is optional.

## Training

Each training script accepts CLI args and saves artifacts under `output/<TICKER>/<model_folder>/` by default (e.g. `output/MSFT/mlp_advanced/`).

Examples:

```bash
python3 mlp_advanced.py --ticker MSFT
python3 lstm_advanced.py --ticker GOOGL --train-start 2019-01-01 --train-end 2021-12-31 --test-start 2022-01-01 --test-end 2022-06-30
python3 lstm_news_advanced.py --ticker AMZN --news news.csv
```

Notes:
- Training scripts save `model.keras` and (when enabled) `scaler.joblib`. Keeping the scaler is important for consistent inference.

## Inference (updated)

`infer.py` is the main lightweight inference wrapper. Important behavior (updated):

- `--ticker` is required when using the CLI form (this repository standardizes on per-ticker workflows).
- If `--model` is provided (e.g. `mlp_advanced`), `infer.py` will run only that model. Otherwise it will run all model subfolders found under `output/{ticker}/` (e.g. `mlp_advanced`, `lstm_advanced`, `lstm_news_advanced`).
- Default prices file: `test_data/{ticker}.csv` (used when `--prices` is omitted).
- Outputs (when run via the CLI) are written into `results/{ticker}/test/prediction/{model_name}_predictions.csv`. The script will create the output folder if necessary.
- If a model folder contains `scaler.joblib`, `infer.py` will use it; otherwise it fits a new MinMaxScaler on the provided prices file (this may change model inputs vs training-time scaling and produce different predictions).
- For sentiment-aware models (LSTM+News), pass `--news path/to/news.csv` to compute daily sentiment; otherwise sentiment defaults to zero.

Examples:

Run all models for a ticker (default prices file):

```bash
python3 infer.py --ticker MSFT
```

Run a single model:

```bash
python3 infer.py --ticker MSFT --model mlp_advanced
```

Use custom prices or news:

```bash
python3 infer.py --ticker MSFT --prices path/to/prices.csv --news path/to/news.csv
```

Programmatic usage (same as before):

```python
from infer import ModelWrapper
mw = ModelWrapper(model_dir='output/MSFT/mlp_advanced')
df = mw.predict_from_csv('path/to/test_prices.csv')  # DataFrame with date, y_true, y_pred
```

## Evaluation (updated)

`evaluate_predictions.py` computes three strategy end-values (buy & hold, predicted-signal, actual-signal) from predictions CSVs that contain `date,y_true,y_pred`.

New behavior and locations:

- Test mode: pass `--test yes` (or `--test y|true|1`) to evaluate prediction CSVs placed under `results/{TICKER}/test/`.
  - The script searches common locations: `results/{TICKER}/test/prediction`, `results/{TICKER}/test/predictions`, and `results/{TICKER}/test` for CSVs.
  - Evaluation outputs are written to `results/{TICKER}/test/returns/{model_name}_returns.csv`.
  - If a single `--csv` is provided with `--test yes`, that CSV is evaluated and the summary is written into the test returns folder.

- Train/default mode (no `--test`): the script keeps the previous behavior of looking for model prediction CSVs under `output/{TICKER}/{model_folder}/predictions.csv` and writes per-model returns to `results/{TICKER}/train/{model_name}_returns.csv`.

Examples:

Evaluate test predictions for a ticker (scan test/prediction(s) and write returns):

```bash
python3 evaluate_predictions.py --ticker AMZN --test yes
```

Evaluate a single CSV into test returns:

```bash
python3 evaluate_predictions.py --ticker AMZN --csv results/AMZN/test/prediction/mlp_advanced_predictions.csv --test yes
```

Evaluate training/production prediction CSVs (default):

```bash
python3 evaluate_predictions.py --ticker AMZN
```

Output CSV format for return summaries (one row): `initial_cash,buy_and_hold,predicted_signal,actual_signal`.

Helper scripts:
- `analyze_predictions.py` — compute accuracy/sign-level stats per predictions file.
- `combine_returns.py` — merge per-model return CSVs (e.g. build `results/{TICKER}/test/combined_returns.csv`).

## Reproduce the repo results (recommended quick steps)

1. Train models (or reuse saved `output/<TICKER>/` artifacts).
2. Run inference with `infer.py --ticker <TICKER>` (optionally `--model <model_folder>` or `--prices <file>`).
3. For test evaluation, run `evaluate_predictions.py --ticker <TICKER> --test yes` to process `results/{TICKER}/test/...` and write `results/{TICKER}/test/returns`.
4. Optionally run `python3 combine_returns.py` to produce a combined table across models.

## File formats

- predictions CSV: `date,y_true,y_pred` (date format YYYY-MM-DD)
- model folder: `model.keras` + `scaler.joblib` + `metrics.json` + `predictions.csv`
- returns summary CSV: `initial_cash,buy_and_hold,predicted_signal,actual_signal`

## Troubleshooting

- Missing packages: ensure `venv` is active and `pip install -r requirements.txt` completed.
- TensorFlow errors (GPU drivers): switch to CPU TF wheel or ensure CUDA/cuDNN versions match.
- Input CSV parsing errors: `load_prices()` tries to auto-detect date and OHLC-like columns; ensure your file has a date column named `date`/`datetime`/`timestamp` (case-insensitive) and price columns named like `open,high,low,close,volume` or similar.
- If inference predictions look wrong, confirm `scaler.joblib` exists in the model folder and is used by `infer.py` (it prints a warning if it fits a new scaler instead).

## Next steps / suggestions

- Add transaction costs and position sizing to `evaluate_predictions.py` for more realistic return estimates.
- Add unit/integration tests for end-to-end training -> inference -> evaluation flows.
- Add an experiment harness to sweep hyperparameters (LR, dropout, hidden sizes) and collect metrics.
- Persist training logs (TensorBoard) and a reproducibility manifest (git SHA, python env) alongside model outputs.

If you'd like, I can also add a combined CSV summary automatically after evaluation (e.g. `results/{TICKER}/test/combined_returns.csv`) or update `visualize_predictions.py` to read the new returns folders.

---
Updated to reflect the repository's current inference & evaluation layout (inference now targets `results/{TICKER}/test/prediction` outputs; evaluation supports `--test` and separate test/train returns folders).

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
This repository contains multiple model implementations (MLP, LSTM, LSTM+News, and LSTM+FinBERT) for short-term stock price prediction, plus training, inference, news-fetching, evaluation and visualization utilities.

What this project does (end-to-end)
- Train models that predict next-day (or next-step) prices using historical price features and optionally news-derived sentiment.
- Run saved models on test price CSVs to produce per-model `predictions.csv` (columns: `date,y_true,y_pred`).
- Evaluate those predictions using simple trading strategies (buy-and-hold, predicted-signal, actual-signal) and save per-model return summaries.
- Visualize model predictions vs true prices and compare models on a single plot.

Key files and what they do
- `src/mlp_advanced.py` — training and artifacts for the MLP baseline; saves model and `predictions.csv` under `output/{TICKER}/mlp_advanced/`.
- `src/lstm_advanced.py` — LSTM training script (no external sentiment); saves artifacts under `output/{TICKER}/lstm_advanced/`.
- `src/lstm_news_advanced.py` — LSTM that consumes a daily sentiment feature computed from news; saves artifacts under `output/{TICKER}/lstm_news_advanced/`.
- `src/lstm_finbert.py` — LSTM that computes sentiment using ProsusAI/finbert via the Hugging Face Inference API (or falls back to VADER/local heuristics). Saves artifacts under `output/{TICKER}/lstm_finbert/`.
- `infer.py` — Lightweight inference wrapper and `ModelWrapper` that loads a saved `model.keras` and `scaler.joblib` and runs predictions on any prices CSV. CLI notes:
  - `--ticker` (required) selects the ticker directory under `output/` to find models.
  - `--model` runs a single model folder (e.g. `mlp_advanced`); omitted runs all model subfolders under `output/{TICKER}`.
  - Default prices file: `test_data/{TICKER}.csv` when `--prices` not provided.
  - For sentiment-aware models (`lstm_news_advanced`, `lstm_finbert`) the wrapper will look for `test_news_data/{TICKER}.csv` (if present) or use `--news` to point to news CSV.
  - CLI output: writes `results/{TICKER}/test/prediction/{model_name}_predictions.csv` by default.
- `fetch_nyt_test_news.py` — fetch helper that queries the New York Times Article Search API for a set of tickers/company names and writes per-ticker news CSVs into `test_news_data/` (columns: `ticker,company,pub_date,headline,lead,pub_dt,pub_day,text,url`). Supports rotating NYT keys, configurable sleeps and pages to avoid rate limits.
- `eval/evaluate_predictions.py` — Evaluates `predictions.csv` files and writes per-model returns summaries.
  - Test mode (`--test yes`) reads prediction CSVs from `results/{TICKER}/test/prediction/` and writes summaries to `results/{TICKER}/test/returns/{model}_returns.csv`.
  - Default (train) mode reads predictions from `output/{TICKER}/{model_folder}/predictions.csv` and writes to `results/{TICKER}/train/{model}_returns.csv`.
- `eval/visualize_predictions.py` — Loads per-model predictions (test or train) and creates a combined comparison plot saved under `results/{TICKER}/...`. The script now includes `LSTM FinBERT` in the comparison.
- `eval/combine_returns.py` (helper) — merges per-model return CSVs into a single `combined_returns.csv` (useful for tables/reports).

Data layout expectations
- `output/{TICKER}/{model_folder}/` should contain:
  - `model.keras` — Keras SavedModel or .keras file
  - `scaler.joblib` — MinMaxScaler used at training time (recommended)
  - `predictions.csv` — prediction outputs (date,y_true,y_pred)
- `test_news_data/{TICKER}.csv` — per-ticker news CSVs produced by `fetch_nyt_test_news.py` (used by sentiment models)
- `results/{TICKER}/test/prediction/` — inference outputs from `infer.py` (one file per model)
- `results/{TICKER}/test/returns/` — evaluation summaries (per-model) when `--test yes` is used

Quick setup (recommended)
1. Create and activate a virtual environment, then install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Install extra sentiment dependencies if you want local VADER or transformers-based scoring:

```bash
pip install vaderSentiment
# or for local transformers (large models):
pip install transformers[sentencepiece] torch
```

Step-by-step: fetch news, run inference, evaluate, visualize
1) Fetch test news (optional but recommended for sentiment models)

Set your NYT keys and run the fetch script (this writes one file per ticker under `test_news_data/`):

```bash
export NYT_API_KEYS="your_key1,your_key2"
python3 fetch_nyt_test_news.py --tickers aapl,amzn,brk-b,googl,jnj,jpm,meta,msft,nvda,tsla \
  --start 2022-08-01 --end 2023-07-31 --out-dir test_news_data --per-ticker-sleep 6 --page-sleep 3 --max-pages 20
```

2) Run inference

To run all models for a ticker (outputs are written to `results/{TICKER}/test/prediction`):

```bash
python3 infer.py --ticker MSFT
```

To run a single model:

```bash
python3 infer.py --ticker MSFT --model lstm_finbert
```

If running `lstm_finbert` or `lstm_news_advanced`, ensure `test_news_data/{TICKER}.csv` exists or pass `--news path/to/news.csv`.

3) Evaluate predictions

Evaluate test predictions and write returns to `results/{TICKER}/test/returns`:

```bash
python3 eval/evaluate_predictions.py --ticker MSFT --test yes
```

Evaluate training/production prediction CSVs (reads from `output/...`):

```bash
python3 eval/evaluate_predictions.py --ticker MSFT
```

You can also evaluate a single CSV directly and save the summary:

```bash
python3 eval/evaluate_predictions.py --ticker MSFT --csv results/MSFT/test/prediction/mlp_advanced_predictions.csv --test yes
```

4) Visualize model comparisons

Create and save a combined plot comparing true prices vs model predictions (test or train):

```bash
python3 eval/visualize_predictions.py --ticker MSFT --test yes
```

This will create `results/{TICKER}/test/model_comparison_plot.png` (or the train equivalent).

Notes, tips and troubleshooting
- If `infer.py` fits a new scaler (no `scaler.joblib` present) you will get different scaling than training-time; keep `scaler.joblib` with model artifacts when possible.
- If Hugging Face inference is used by `lstm_finbert`, set `HUGGINGFACE_API_KEY` (or `HF_API_KEY`) env var. The script will fallback to VADER if API calls fail.
- If you hit rate limits while fetching news, increase `--page-sleep` and `--per-ticker-sleep` or supply multiple `NYT_API_KEYS` to rotate.
- Filenames and folders are configurable in the scripts; if you prefer alternative naming (e.g., `lstm-finbert` folder), update the script calls accordingly.

Development and next steps
- Add transaction costs / position sizing to `eval/evaluate_predictions.py` for more realistic backtests.
- Add unit and integration tests for the end-to-end pipeline.
- Optionally implement local `transformers` inference for `lstm_finbert` to avoid HF API rate limits (requires GPU/CPU resources and disk for model weights).

Contact / help
If you want me to wire up combined reports, add per-ticker incremental fetch/append behavior, or change file naming conventions, tell me which change and I will update the scripts.

---
This README was updated to reflect recent additions: `lstm_finbert`, per-ticker `test_news_data/`, `fetch_nyt_test_news.py`, rotated NYT key support, test/train `results` layout, and visualization/evaluation updates.

#!/usr/bin/env python3
"""
Evaluate trading performance based on a predictions CSV produced by the MLP script.

It expects a CSV with at least these columns: date, y_true, y_pred
- date: parseable by pandas.to_datetime
- y_true: actual price (target)
- y_pred: predicted price

Strategies computed:
- buy_and_hold: buy at first actual price, hold to the last date
- predicted_signal: for each day t>0, if y_pred[t] > y_pred[t-1] then stay/invested for t-1->t and capture actual return; otherwise stay in cash
- actual_signal: same as predicted_signal but using y_true signals (benchmark)

Usage:
    python evaluate_predictions.py --csv outputs_mlp_advanced/predictions.csv

Outputs printed to stdout and saved to outputs_evaluation.csv by default.
"""

import argparse
import os
import pandas as pd
import numpy as np


def evaluate(df, init_cash=10000.0):
    # ensure sorted
    df = df.sort_values("date").reset_index(drop=True)
    prices = df["y_true"].astype(float).values
    preds = df["y_pred"].astype(float).values

    if len(prices) < 2:
        raise ValueError("Need at least 2 rows to evaluate returns.")

    # buy and hold
    bh_final = init_cash * (prices[-1] / prices[0])

    # predicted-signal strategy: if predicted[t] > predicted[t-1] then be invested across t-1->t
    val_pred = init_cash
    for t in range(1, len(prices)):
        if preds[t] > preds[t - 1]:
            ret = prices[t] / prices[t - 1]
            val_pred *= ret
        # else hold cash (no change)

    # actual-signal strategy: same but using actual price movement as signal
    val_actual_signal = init_cash
    for t in range(1, len(prices)):
        if prices[t] > prices[t - 1]:
            ret = prices[t] / prices[t - 1]
            val_actual_signal *= ret

    results = {
        "initial_cash": init_cash,
        "buy_and_hold": bh_final,
        "predicted_signal": val_pred,
        "actual_signal": val_actual_signal,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Single predictions CSV to evaluate (optional)")
    ap.add_argument("--ticker", required=True, help="Ticker name used for results subfolder")
    ap.add_argument("--cash", type=float, default=10000.0)
    ap.add_argument("--out", default=None, help="Output filename for single-csv mode (optional)")
    ap.add_argument("--test", default="no", help="If 'yes', evaluate test predictions found under results/{ticker}/test (default: no)")
    args = ap.parse_args()
    # Determine mode: test vs train
    is_test = str(args.test).lower() in ("yes", "y", "true", "1")

    # Candidate locations for each model's predictions in training/batch mode (default behavior)
    models = {
        "mlp": os.path.join("output", args.ticker, "mlp_advanced", "predictions.csv"),
        "lstm": os.path.join("output", args.ticker, "lstm_advanced", "predictions.csv"),
        "lstm_news": os.path.join("output", args.ticker, "lstm_news_advanced", "predictions.csv")
    }

    # output base directories
    results_root = os.path.join("results", args.ticker)
    if is_test:
        # test mode: read predictions from results/<ticker>/test/(prediction|predictions|)
        sd=os.path.join(results_root, "test", "prediction")
        out_base = os.path.join(results_root, "test")
    else:
        # train/default mode: write to results/<ticker>/train
        out_base = os.path.join(results_root, "train")

    os.makedirs(out_base, exist_ok=True)

    def _load_and_prepare(path):
        df = pd.read_csv(path)
        required = {"date", "y_true", "y_pred"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {required}. Found: {df.columns.tolist()}")
        df = df.dropna(subset=["y_true", "y_pred"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    # single-file mode: evaluate provided CSV and save into either test returns or train results
    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(args.csv)
        df = _load_and_prepare(args.csv)
        res = evaluate(df, init_cash=args.cash)
        print("Evaluation results:")
        print(f"  initial cash: {res['initial_cash']:.2f}")
        print(f"  buy & hold final: {res['buy_and_hold']:.2f}")
        print(f"  predicted-signal final: {res['predicted_signal']:.2f}")
        print(f"  actual-signal final: {res['actual_signal']:.2f}")
        # determine out path
        base = os.path.splitext(os.path.basename(args.csv))[0]
        if args.out:
            out_name = args.out
        else:
            out_name = f"{base}_returns.csv"
        out_path = os.path.join(out_name)
        pd.DataFrame([res]).to_csv(out_path, index=False)
        print(f"Saved summary to {out_path}")
        return

    # batch mode:
    if is_test:
        # collect CSV prediction files from test search dirs
        found_files = []
        for fn in os.listdir(sd):
            if not fn.lower().endswith('.csv'):
                continue
            found_files.append(os.path.join(sd, fn))

        for path in sorted(found_files):
            try:
                df = _load_and_prepare(path)
                res = evaluate(df, init_cash=args.cash)
                # derive model name from filename
                base = os.path.splitext(os.path.basename(path))[0]
                # normalize names like mlp_pred, mlp_predictions -> mlp
                model_name = base
                for suffix in ['_predictions', '_pred', '_preds', '_predictions.csv', '_pred.csv']:
                    if model_name.endswith(suffix.replace('.csv','')):
                        model_name = model_name[: -len(suffix.replace('.csv',''))]
                out_path = os.path.join(out_base, f"{model_name}_returns.csv")
                pd.DataFrame([res]).to_csv(out_path, index=False)
                print(f"Saved {model_name} -> {out_path}")
            except Exception as e:
                print(f"Error evaluating {path}: {e}")
    else:
        # train/default mode: use known output locations (from training/inference output folders)
        for model_name, candidates in models.items():
            # candidates may be a single path or a list of possible paths
            if isinstance(candidates, (list, tuple)):
                found = None
                for p in candidates:
                    if os.path.exists(p):
                        found = p
                        break
                if not found:
                    print(f"Skipping {model_name}: none of candidate paths found")
                    continue
                path = found
            else:
                path = candidates
                if not os.path.exists(path):
                    print(f"Skipping {model_name}: {path} not found")
                    continue

            try:
                df = _load_and_prepare(path)
                res = evaluate(df, init_cash=args.cash)
                out_path = os.path.join(out_base, f"{model_name}_returns.csv")
                pd.DataFrame([res]).to_csv(out_path, index=False)
                print(f"Saved {model_name} -> {out_path}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")


if __name__ == "__main__":
    main()

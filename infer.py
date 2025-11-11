#!/usr/bin/env python3
"""
Lightweight inference wrapper to load a saved Keras model + scaler and run predictions on any prices CSV.

Usage:
  # CLI: provide model folder or model file and prices CSV
  python3 infer.py --model_dir output/MSFT/mlp_advanced --prices test_prices.csv --outdir output/NEW/mlp_eval

Programmatic:
  from infer import ModelWrapper
  mw = ModelWrapper(model_dir='output/MSFT/mlp_advanced')
  df = mw.predict_from_csv('test_prices.csv')  # returns DataFrame with date,y_true,y_pred

Notes:
- If the model expects an extra sentiment feature (lstm_news), pass --news to compute sentiment; otherwise sentiment defaults to 0.
- If a scaler.joblib exists in the model folder it will be used; otherwise the script fits a new scaler on the provided prices (warning: this differs from training scaler).
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# reuse helper functions from your scripts
from mlp_advanced import load_prices, make_sequences_multifeat
try:
    from lstm_news_advanced import compute_daily_sentiment, align_with_sentiment, make_sequences_multifeat_with_sent
except Exception:
    # if lstm_news_advanced not importable for some reason, define fallback that raises when needed
    compute_daily_sentiment = None
    align_with_sentiment = None
    make_sequences_multifeat_with_sent = None


class ModelWrapper:
    def __init__(self, model_dir=None, model_path=None):
        # find model.keras
        if model_path is None and model_dir is None:
            raise ValueError('Provide model_dir or model_path')
        if model_path is None:
            # look for model.keras in model_dir or its immediate children
            cand = []
            if os.path.isdir(model_dir):
                # direct
                p = os.path.join(model_dir, 'model.keras')
                if os.path.exists(p):
                    cand.append(p)
                # look deeper one level
                for sub in glob.glob(os.path.join(model_dir, '*')):
                    p2 = os.path.join(sub, 'model.keras')
                    if os.path.exists(p2):
                        cand.append(p2)
            if not cand:
                raise FileNotFoundError(f'No model.keras found under {model_dir}')
            model_path = cand[0]
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.model = tf.keras.models.load_model(model_path)
        # try load scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

    def _prepare(self, prices_csv, seq_len=None, news_csv=None):
        feats, dates, feat_cols, target_idx = load_prices(prices_csv, ticker=None)
        n_features = feats.shape[1]
        # scaler
        if self.scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(feats)
            print('Warning: no saved scaler found; fitted a new scaler on provided prices file')
        else:
            scaler = self.scaler
        feats_scaled = scaler.transform(feats)

        # inspect model input shape
        in_shape = self.model.input_shape  # e.g. (None, seq_len, nfeat) or (None, input_dim)
        if len(in_shape) == 3:
            model_seq_len = in_shape[1]
            # check if model expects extra sentiment (nfeat+1)
            model_nfeat = in_shape[2]
            if model_nfeat == n_features:
                X_all, y_all = make_sequences_multifeat(feats_scaled, model_seq_len, target_idx)
            else:
                # expects extra sentiment feature
                if news_csv is None:
                    print('Model expects sentiment feature but no news CSV provided; using zeros for sentiment')
                    # create zero sentiment
                    sentiment = np.zeros((len(feats_scaled),1), dtype=np.float32)
                else:
                    if compute_daily_sentiment is None:
                        raise RuntimeError('lstm_news utilities not available to compute sentiment; provide news or implement compute_daily_sentiment')
                    news = pd.read_csv(news_csv)
                    daily_sent = compute_daily_sentiment(news, ticker=None, day_col='pub_day', text_col='text')
                    sentiment = align_with_sentiment(dates, feats, daily_sent)
                X_all, y_all = make_sequences_multifeat_with_sent(feats_scaled, sentiment, model_seq_len, target_idx)
            X_for_pred = X_all
            dates_y = dates[model_seq_len:]
            seq_len_used = model_seq_len
        elif len(in_shape) == 2:
            input_dim = in_shape[1]
            # infer seq_len
            if input_dim % n_features == 0:
                model_seq_len = input_dim // n_features
            else:
                if seq_len is None:
                    raise ValueError('Cannot infer seq_len for MLP; provide --seq_len')
                model_seq_len = seq_len
            X_all, y_all = make_sequences_multifeat(feats_scaled, model_seq_len, target_idx)
            X_for_pred = X_all.reshape((X_all.shape[0], -1))
            dates_y = dates[model_seq_len:]
            seq_len_used = model_seq_len
        else:
            raise ValueError(f'Unsupported model.input_shape: {in_shape}')

        return X_for_pred, y_all, dates_y, scaler, feat_cols, target_idx, seq_len_used

    def predict_from_csv(self, prices_csv, news_csv=None, seq_len=None, save_to=None):
        X_for_pred, y_all, dates_y, scaler, feat_cols, target_idx, seq_len_used = self._prepare(prices_csv, seq_len=seq_len, news_csv=news_csv)
        preds_scaled = self.model.predict(X_for_pred, verbose=0)
        # invert predictions & truth
        tmp = np.zeros((len(preds_scaled), len(feat_cols)), dtype=np.float32)
        tmp[:, target_idx] = preds_scaled.flatten()
        preds_inv = scaler.inverse_transform(tmp)[:, target_idx].reshape(-1,1)

        tmp2 = np.zeros((len(y_all), len(feat_cols)), dtype=np.float32)
        tmp2[:, target_idx] = y_all.flatten()
        y_inv = scaler.inverse_transform(tmp2)[:, target_idx].reshape(-1,1)

        df_out = pd.DataFrame({
            'date': pd.Series(dates_y).astype(str),
            'y_true': y_inv.flatten(),
            'y_pred': preds_inv.flatten()
        })
        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            df_out.to_csv(save_to, index=False)
            print('Saved predictions to', save_to)
        return df_out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default=None, help='Folder containing model.keras (or parent of model folder)')
    ap.add_argument('--model_path', default=None, help='Exact model file path to load')
    ap.add_argument('--prices', required=True)
    ap.add_argument('--news', default=None)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    if args.outdir:
        outp = os.path.join(args.outdir)
    else:
        # default: save next to model
        md = args.model_path or args.model_dir
        if md is None:
            raise ValueError('Provide model_dir or model_path')
        if args.model_path is None:
            # derive model file
            # search
            cand = []
            if os.path.isdir(args.model_dir):
                p = os.path.join(args.model_dir, 'model.keras')
                if os.path.exists(p):
                    cand.append(p)
                for sub in glob.glob(os.path.join(args.model_dir, '*')):
                    p2 = os.path.join(sub, 'model.keras')
                    if os.path.exists(p2):
                        cand.append(p2)
            if not cand:
                raise FileNotFoundError(f'No model.keras found under {args.model_dir}')
            model_path = cand[0]
        else:
            model_path = args.model_path
        outp = os.path.join(os.path.dirname(model_path), 'predictions.csv')

    wrapper = ModelWrapper(model_dir=args.model_dir, model_path=args.model_path)
    df = wrapper.predict_from_csv(args.prices, news_csv=args.news, save_to=outp)
    print(df.head())

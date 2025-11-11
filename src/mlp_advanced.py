
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
import tensorflow as tf

def pick_features(df, wanted=("open","high","low","close","volume")):
    cols = []
    lower = {c.lower(): c for c in df.columns}
    for w in wanted:
        # pick the first column whose lowercase startswith the wanted name
        cand = [lower[c] for c in lower if c.startswith(w)]
        if cand:
            cols.append(cand[0])
    return cols

def load_prices(path, ticker=None, date_key_candidates=("date","datetime","timestamp")):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # date column
    date_col = None
    lower = {c.lower(): c for c in df.columns}
    for k in date_key_candidates:
        if k in lower:
            date_col = lower[k]; break
    if date_col is None:
        raise ValueError("No date/datetime/timestamp column found.")
    # ticker selection
    if "ticker" in [c.lower() for c in df.columns] and ticker is not None:
        tcol = [c for c in df.columns if c.lower()=="ticker"][0]
        df = df[df[tcol].astype(str)==str(ticker)]
    # pick OHLCV
    feat_cols = pick_features(df)
    if len(feat_cols) < 2:
        raise ValueError("Could not find enough OHLCV-like columns.")
    # ensure sorting and clean
    df = df[[date_col]+feat_cols].dropna()
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col].values)
    feats = df[feat_cols].astype(float).values
    # target = close-like
    target_col = [c for c in feat_cols if c.lower().startswith("close")][0]
    target_idx = feat_cols.index(target_col)
    return feats, dates, feat_cols, target_idx

def make_sequences_multifeat(arr, seq_len, target_idx):
    X, y = [], []
    for i in range(len(arr)-seq_len):
        X.append(arr[i:i+seq_len, :])
        y.append(arr[i+seq_len, target_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1,1)
    return X, y

def build_mlp(input_dim, lr=1e-3):
    tf.random.set_seed(1234)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr)
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="prices.csv")
    ap.add_argument("--news", default="news.csv")
    ap.add_argument("--ticker", default="MSFT")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.85, help="(deprecated) fraction split if date ranges not used")
    ap.add_argument("--train-start", default="2019-01-01", help="training start date (inclusive)")
    ap.add_argument("--train-end", default="2021-12-31", help="training end date (inclusive)")
    ap.add_argument("--test-start", default="2022-01-01", help="test start date (inclusive)")
    ap.add_argument("--test-end", default="2022-12-31", help="test end date (inclusive)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    ap.add_argument("--outdir", default=os.path.join("output", args.ticker,"mlp_advanced"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feats, dates, feat_cols, target_idx = load_prices(args.prices, args.ticker)

    # parse date ranges from arguments (these refer to the date of the target value,
    # i.e. the date at index i+seq_len when creating sequences)
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    test_start = pd.to_datetime(args.test_start)
    test_end = pd.to_datetime(args.test_end)

    scaler = MinMaxScaler()
    feats_scaled = scaler.fit_transform(feats)

    X_all, y_all = make_sequences_multifeat(feats_scaled, args.seq_len, target_idx)
    dates_y = dates[args.seq_len:]

    # select sequences by the date of their target (dates_y)
    mask_train = (dates_y >= train_start) & (dates_y <= train_end)
    mask_test = (dates_y >= test_start) & (dates_y <= test_end)

    if mask_train.sum() == 0:
        raise ValueError(f"No training samples found in range {train_start.date()} to {train_end.date()}")
    if mask_test.sum() == 0:
        raise ValueError(f"No test samples found in range {test_start.date()} to {test_end.date()}")

    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_test, y_test = X_all[mask_test], y_all[mask_test]
    dates_test = dates_y[mask_test]

    # flatten sequences for MLP
    X_train_f = X_train.reshape((X_train.shape[0], -1))
    X_test_f = X_test.reshape((X_test.shape[0], -1))

    model = build_mlp(X_train_f.shape[1], lr=1e-3)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]

    model.fit(
        X_train_f, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0
    )

    preds_scaled = model.predict(X_test_f, verbose=0)

    # inverse only the target dimension
    # build a scaler for target from original scaler by transforming a zero-array and replacing the column
    y_test_inv = preds_inv = None
    # For inversion, transform unit vectors method:
    # scale back using feature-wise min/max stored in scaler
    # We'll reconstruct by building a temp array with the target in the right column.
    def invert(col_scaled):
        tmp = np.zeros((len(col_scaled), feats.shape[1]), dtype=np.float32)
        tmp[:, target_idx] = col_scaled.flatten()
        inv = scaler.inverse_transform(tmp)[:, target_idx].reshape(-1,1)
        return inv

    y_test_inv = invert(y_test)
    preds_inv = invert(preds_scaled)

    mae = mean_absolute_error(y_test_inv, preds_inv)
    mape = mean_absolute_percentage_error(y_test_inv, preds_inv)
    acc = 1 - mape

    pd.DataFrame({
        "date": dates_test.astype(str),
        "y_true": y_test_inv.flatten(),
        "y_pred": preds_inv.flatten()
    }).to_csv(f"{args.outdir}/predictions.csv", index=False)

    with open(f"{args.outdir}/metrics.json","w") as f:
        json.dump({"mae": float(mae), "mape": float(mape), "accuracy_proxy": float(acc),
                   "features": feat_cols, "target_idx": target_idx}, f, indent=2)

    model.save(f"{args.outdir}/model.keras")
    # save scaler for consistent inference later
    try:
        joblib.dump(scaler, f"{args.outdir}/scaler.joblib")
    except Exception:
        pass
    print(f"[MLP Advanced] MAE={mae:.6f}  MAPE={mape:.6f}  Acc≈{acc:.6f}")
    print(f"Saved under {args.outdir}/")

if __name__ == "__main__":
    main()

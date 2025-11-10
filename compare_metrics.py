
import os, json
import pandas as pd

def load_metrics(name, path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {"Model": name, **data}
    else:
        return {"Model": name, "mae": None, "mape": None, "accuracy_proxy": None}

def main():
    base_dir = os.path.dirname(__file__)
    paths = {
        "MLP Advanced": os.path.join(base_dir, "data", "outputs_mlp_adv_aapl", "metrics.json"),
        "LSTM Advanced": os.path.join(base_dir, "data", "outputs_lstm_adv_aapl", "metrics.json"),
        "LSTM + News Advanced": os.path.join(base_dir, "data", "outputs_lstm_news_adv_aapl", "metrics.json")
    }

    records = [load_metrics(name, path) for name, path in paths.items()]
    df = pd.DataFrame(records)
    print("\nModel Comparison:\n")
    print(df.to_string(index=False))

    out_path = os.path.join(base_dir, "data", "model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison CSV to: {out_path}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Ticker symbol (required)")
    ap.add_argument("--test" , default="NO", help="Path to test CSV file (default: results/{ticker}/test/prediction/...)")
    args = ap.parse_args()
    base_dir = os.path.dirname(__file__)
    if( args.test=="yes" or args.test=="YES" or args.test=="Yes" ):
        paths = {
        "MLP Advanced": os.path.join(base_dir,"results", args.ticker ,"test","prediction","mlp_advanced_predictions.csv"),
        "LSTM Advanced": os.path.join(base_dir, "results", args.ticker ,"test","prediction","lstm_advanced_predictions.csv"),
        "LSTM + News Advanced": os.path.join(base_dir, "results", args.ticker ,"test","prediction","lstm_news_advanced_predictions.csv")
        }
    else:
        paths = {
            "MLP Advanced": os.path.join(base_dir,"output", args.ticker ,"mlp_advanced", "predictions.csv"),
            "LSTM Advanced": os.path.join(base_dir, "output", args.ticker ,"lstm_advanced", "predictions.csv"),
            "LSTM + News Advanced": os.path.join(base_dir, "output", args.ticker ,"lstm_news_advanced", "predictions.csv")
        }
    # print(paths)
    dfs = {}
    for name, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            dfs[name] = df[["date", "y_pred"]].rename(columns={"y_pred": f"{name} Pred"})
        else:
            print(f"⚠️ Missing: {path}")

    # Use MLP as base for merging
    if not dfs:
        print("❌ No prediction files found.")
        return

    base_df = pd.read_csv(paths["MLP Advanced"])
    base_df["date"] = pd.to_datetime(base_df["date"])
    merged = base_df[["date", "y_true"]].copy()
    for name, df in dfs.items():
        merged = pd.merge(merged, df, on="date", how="inner")

    plt.figure(figsize=(14, 7))
    plt.plot(merged["date"], merged["y_true"], label="True Price", color="black", linewidth=2.5)
    if "MLP Advanced Pred" in merged:
        plt.plot(merged["date"], merged["MLP Advanced Pred"], '--', label="MLP Pred", color="blue")
    if "LSTM Advanced Pred" in merged:
        plt.plot(merged["date"], merged["LSTM Advanced Pred"], '--', label="LSTM Pred", color="green")
    if "LSTM + News Advanced Pred" in merged:
        plt.plot(merged["date"], merged["LSTM + News Advanced Pred"], '--', label="LSTM+News Pred", color="red")

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{args.ticker} Price Prediction Comparison (True vs Models)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(args.test=="yes" or args.test=="YES" or args.test=="Yes" ):
        out_path = os.path.join(base_dir,"results", args.ticker,"test","model_comparison_plot.png")
    else:
        out_path = os.path.join(base_dir,"results", args.ticker,"train","model_comparison_plot.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Combined comparison plot saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()

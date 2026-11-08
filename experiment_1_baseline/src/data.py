"""Data download, feature engineering and chronological splitting."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
# Chronological split. Test period is never used for tuning / model selection.
TRAIN_END = "2017-12-31"
VAL_END = "2019-12-31"
START, END = "2005-01-01", "2025-01-01"

FEATURES = ["ret1", "ret5", "ret10", "ret20", "ma_ratio", "px_ma20", "vol_z", "volat20"]


def download(ticker="SPY", start=START, end=END, force=False):
    """Download daily OHLCV from Yahoo! Finance (cached as CSV in data/)."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path) and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path)
    return df


def add_features(df):
    """Features use only information available at the close of day t."""
    d = df.copy()
    c = d["Close"]
    lr = np.log(c).diff()
    d["ret1"] = lr
    d["ret5"] = np.log(c / c.shift(5))
    d["ret10"] = np.log(c / c.shift(10))
    d["ret20"] = np.log(c / c.shift(20))
    d["ma_ratio"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    d["px_ma20"] = c / c.rolling(20).mean() - 1
    lv = np.log(d["Volume"])
    d["vol_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    d["volat20"] = lr.rolling(20).std()
    d = d.dropna()
    # scale to roughly unit range (fixed constants, no look-ahead / fitting on test)
    scale = {"ret1": 0.01, "ret5": 0.02, "ret10": 0.03, "ret20": 0.05,
             "ma_ratio": 0.02, "px_ma20": 0.03, "vol_z": 1.0, "volat20": 0.01}
    for k, s in scale.items():
        d[k] = (d[k] / s).clip(-5, 5)
    return d


def load_splits(ticker="SPY", force=False):
    df = add_features(download(ticker, force=force))
    return {
        "train": df[:TRAIN_END],
        "val": df[TRAIN_END:VAL_END].iloc[1:],
        "test": df[VAL_END:].iloc[1:],
    }


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for k, v in load_splits(t, force=True).items():
        print(k, v.index[0].date(), v.index[-1].date(), len(v))

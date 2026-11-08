"""Data download, feature engineering (price / macro / cross-asset) and walk-forward folds.

All features at row t use only information available at the close of day t.
Scaling uses fixed constants (nothing is fitted on validation/test data).
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
START, END = "1998-01-01", "2025-01-01"
MA_WINDOWS = [5, 10, 20, 30, 50, 100, 200]

PRICE_F = ["ret1", "ret5", "ret10", "ret20", "ma_ratio", "px_ma20", "vol_z", "volat20"]
MACRO_F = ["vix_lvl", "vix_chg5", "y10_lvl", "y10_chg20", "curve"]
CROSS_F = ["usd_ret20", "gold_ret20", "oil_ret20"]
SECTOR_F = ["xly_xlp20", "xlu_rel20", "xlf_rel20", "iwm_rel20"]
BOND_F = ["tlt_ret20", "credit20", "copper20"]
VOL_F = ["skew_lvl", "vxn_vix", "move_lvl"]
VARIANTS = {"price": PRICE_F, "macro": PRICE_F + MACRO_F, "cross": PRICE_F + MACRO_F + CROSS_F,
            "macro_sector": PRICE_F + MACRO_F + SECTOR_F, "macro_bonds": PRICE_F + MACRO_F + BOND_F,
            "macro_vol": PRICE_F + MACRO_F + VOL_F, "all": PRICE_F + MACRO_F + SECTOR_F + BOND_F + VOL_F}
EXTRA_F = CROSS_F + SECTOR_F + BOND_F + VOL_F

# Expanding-window walk-forward: test blocks; validation = the 2 years before the block;
# training = everything before the validation years (from 1998).
TEST_BLOCKS = [(2008, 2009), (2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017),
               (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2024)]


def _path(sym):
    return os.path.join(DATA_DIR, sym.replace("^", "_").replace("=", "_").replace("/", "_") + ".csv")


def download(sym, force=False):
    p = _path(sym)
    if os.path.exists(p) and not force:
        return pd.read_csv(p, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(sym, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(p)
    return df


def _clip(x, s):
    return (x / s).replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def build(ticker="SPY", variant="price"):
    px = download(ticker)
    idx = px.index
    d = px[["Open", "Close", "Volume"]].copy()
    c = d["Close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=idx)
    f["ret1"] = _clip(lr, 0.01)
    f["ret5"] = _clip(np.log(c / c.shift(5)), 0.02)
    f["ret10"] = _clip(np.log(c / c.shift(10)), 0.03)
    f["ret20"] = _clip(np.log(c / c.shift(20)), 0.05)
    f["ma_ratio"] = _clip(c.rolling(5).mean() / c.rolling(20).mean() - 1, 0.02)
    f["px_ma20"] = _clip(c / c.rolling(20).mean() - 1, 0.03)
    lv = np.log(d["Volume"].replace(0, np.nan))
    f["vol_z"] = _clip((lv - lv.rolling(60).mean()) / lv.rolling(60).std(), 1.0)
    f["volat20"] = _clip(lr.rolling(20).std(), 0.01)

    def aux(sym):
        return download(sym)["Close"].reindex(idx).ffill()

    vix, tnx, irx = aux("^VIX"), aux("^TNX"), aux("^IRX")
    f["vix_lvl"] = _clip(vix / 20 - 1, 0.5)
    f["vix_chg5"] = _clip(np.log(vix / vix.shift(5)), 0.2)
    f["y10_lvl"] = _clip(tnx - 3.0, 1.5)
    f["y10_chg20"] = _clip(tnx - tnx.shift(20), 0.3)
    f["curve"] = _clip(tnx - irx, 1.5)
    usd, gold, oil = aux("DX-Y.NYB"), aux("GC=F"), aux("CL=F")
    f["usd_ret20"] = _clip(usd / usd.shift(20) - 1, 0.03)
    f["gold_ret20"] = _clip(gold / gold.shift(20) - 1, 0.05)
    f["oil_ret20"] = _clip(oil / oil.shift(20) - 1, 0.10)

    r20 = lambda x: x / x.shift(20) - 1
    spy20 = r20(c)
    xly, xlp, xlu, xlf, iwm = (aux(t) for t in ["XLY", "XLP", "XLU", "XLF", "IWM"])
    f["xly_xlp20"] = _clip(r20(xly) - r20(xlp), 0.04)
    f["xlu_rel20"] = _clip(r20(xlu) - spy20, 0.04)
    f["xlf_rel20"] = _clip(r20(xlf) - spy20, 0.04)
    f["iwm_rel20"] = _clip(r20(iwm) - spy20, 0.04)
    tlt, lqd, ief, cu = (aux(t) for t in ["TLT", "LQD", "IEF", "HG=F"])
    f["tlt_ret20"] = _clip(r20(tlt), 0.04)
    f["credit20"] = _clip(r20(lqd) - r20(ief), 0.02)
    f["copper20"] = _clip(r20(cu), 0.08)
    skew, vxn, move = aux("^SKEW"), aux("^VXN"), aux("^MOVE")
    f["skew_lvl"] = _clip(skew - 125, 10)
    f["vxn_vix"] = _clip(vxn / vix - 1, 0.1)
    f["move_lvl"] = _clip(move - 90, 30)

    feats = VARIANTS[variant]
    d = pd.concat([d, f[PRICE_F + MACRO_F + EXTRA_F]], axis=1)
    d["rf"] = (irx.clip(lower=0) / 100).fillna(0.0)          # annual risk-free (3m T-bill)
    for w in MA_WINDOWS:
        d[f"sma{w}"] = c.rolling(w).mean()
    d = d.dropna(subset=PRICE_F + ["sma200"])
    d[MACRO_F + EXTRA_F] = d[MACRO_F + EXTRA_F].fillna(0.0)  # series that start after 1998 -> neutral 0
    d.attrs["feats"] = feats
    return d


def fold_split(df, k):
    a, b = TEST_BLOCKS[k]
    y = df.index.year
    out = {"train": df[y < a - 2], "val": df[(y >= a - 2) & (y < a)], "test": df[(y >= a) & (y <= b)]}
    for v in out.values():
        v.attrs["feats"] = df.attrs["feats"]
    return out


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for s in [t, "^VIX", "^TNX", "^IRX", "DX-Y.NYB", "GC=F", "CL=F", "XLY", "XLP", "XLU", "XLF", "IWM", "TLT", "LQD", "IEF", "HG=F", "^SKEW", "^VXN", "^MOVE"]:
        download(s)
    d = build(t, "all")
    print(d.shape, d.index.min().date(), d.index.max().date())
    for k in range(len(TEST_BLOCKS)):
        s = fold_split(d, k)
        print(k, TEST_BLOCKS[k], {n: len(x) for n, x in s.items()})

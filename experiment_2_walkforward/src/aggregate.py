"""Aggregate walk-forward results: stitched out-of-sample metrics, per-fold table, block-bootstrap tests.
usage: python aggregate.py --variants price macro cross [--cost 0.0005] [--tag ""]
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import stats_from_returns

ORDER = ["DQN-ens", "DQN-single", "Buy&Hold", "VolMatch", "MA-tuned", "Random-TM"]
SUBSETS = {"ALL folds (2008-2024)": range(0, 9), "CLEAN folds (2014-2024, after tuning window)": range(3, 9)}


def stitched(d, folds):
    d = d[d.fold.isin(list(folds))]
    rows = []
    for (s, sd), g in d.groupby(["strategy", "seed"]):
        g = g.sort_values(["fold", "date"])
        m = stats_from_returns(g.ret.values, g.rf.values, g.expo.values)
        m.update(strategy=s, seed=sd); rows.append(m)
    r = pd.DataFrame(rows)
    cols = ["total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_exposure"]
    mean = r.groupby("strategy")[cols].mean()
    sd = r.groupby("strategy")[cols].std()
    mean = mean.reindex([o for o in ORDER if o in mean.index])
    return mean, sd.reindex(mean.index), r


def series(d, folds, strat):
    g = d[(d.strategy == strat) & d.fold.isin(list(folds))].sort_values(["fold", "date"])
    return g.ret.values, g.rf.values


def boot(a, b, rf, n=4000, block=21, seed=0):
    """Paired block bootstrap of (sharpe_a - sharpe_b) and annualised return difference."""
    rng = np.random.default_rng(seed)
    T = len(a); nb = int(np.ceil(T / block))
    sh = lambda x, f: (x - f).mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)
    ds, dr = [], []
    for _ in range(n):
        st = rng.integers(0, T - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:T]
        ds.append(sh(a[idx], rf[idx]) - sh(b[idx], rf[idx]))
        dr.append((a[idx].mean() - b[idx].mean()) * 252)
    ds, dr = np.array(ds), np.array(dr)
    pt_s = sh(a, rf) - sh(b, rf); pt_r = (a.mean() - b.mean()) * 252
    return pt_s, np.percentile(ds, [2.5, 97.5]), (ds > 0).mean(), pt_r, np.percentile(dr, [2.5, 97.5])


def fmt(mean, sd):
    t = mean.copy().astype(object)
    for s in mean.index:
        for c in mean.columns:
            v, e = mean.loc[s, c], sd.loc[s, c]
            pct = c != "sharpe"
            f = (lambda x: f"{100*x:.1f}%") if pct else (lambda x: f"{x:.2f}")
            t.loc[s, c] = f(v) + (f" ±{f(e)}" if pd.notna(e) else "")
    return t


def main(variants, cost, tag):
    out = []
    for v in variants:
        p = f"results/{tag}{v}_SPY_c{cost}_daily.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        out.append(f"\n# Variant: {v}  (cost {cost*1e4:.0f} bps/side)\n")
        for name, folds in SUBSETS.items():
            mean, sd, _ = stitched(d, folds)
            out.append(f"## {name}\nStitched out-of-sample, mean ± std across seeds (single-run rows have no ±).\n")
            out.append(fmt(mean, sd).to_markdown() + "\n")
            e, erf = series(d, folds, "DQN-ens")
            out.append("Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline\n")
            out.append("| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |\n|---|---|---|---|---|---|")
            for b in ["Buy&Hold", "VolMatch", "MA-tuned"]:
                br, _ = series(d, folds, b)
                ps, ci, pp, pr, cr = boot(e, br, erf)
                out.append(f"| {b} | {ps:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {pp:.2f} | {100*pr:+.1f}% | [{100*cr[0]:+.1f}%, {100*cr[1]:+.1f}%] |")
            rt = d[(d.strategy == "Random-TM") & d.fold.isin(list(folds))]
            rsh = [stats_from_returns(g.sort_values(["fold", "date"]).ret.values, g.sort_values(["fold", "date"]).rf.values)["sharpe"]
                   for _, g in rt.groupby("seed")]
            es = stats_from_returns(e, erf)["sharpe"]
            out.append(f"\nRandom-TM Sharpe over 20 seeds: mean {np.mean(rsh):+.2f}, 95th pct {np.percentile(rsh,95):+.2f}; "
                       f"DQN-ens Sharpe {es:+.2f} -> beats {100*np.mean(np.array(rsh) < es):.0f}% of random runs.\n")
        # per-fold total return
        f = d.groupby(["fold", "strategy", "seed"]).apply(lambda g: (1 + g.ret).prod() - 1, include_groups=False).reset_index(name="tr")
        pv = f.groupby(["fold", "strategy"]).tr.mean().unstack().reindex(columns=ORDER)
        pv.index = [f"{a}-{b}" for a, b in [(2008, 2009), (2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2024)]]
        out.append("## Per-fold total return (single/random = mean over seeds)\n")
        out.append((100 * pv).round(1).astype(str).add("%").to_markdown() + "\n")
    txt = "\n".join(out)
    open(f"results/{tag}summary_{'_'.join(variants)}_c{cost}.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["price"]); ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); main(a.variants, a.cost, a.tag)

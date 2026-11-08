"""Compare all feature variants on identical walk-forward test windows.
usage (from Project_4_v2/): python src/compare.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import series, boot, stitched
from metrics import stats_from_returns

VARS = ["price", "macro", "cross", "macro_sector", "macro_bonds", "macro_vol", "all"]
SUB = {"ALL folds 2008-2024": range(9), "CLEAN folds 2014-2024": range(3, 9)}


def main(cost=0.0005):
    D = {v: pd.read_csv(f"results/{v}_SPY_c{cost}_daily.csv") for v in VARS if os.path.exists(f"results/{v}_SPY_c{cost}_daily.csv")}
    out = [f"# Variant comparison (SPY, walk-forward, {cost*1e4:.0f} bps/side)\n"]
    for name, folds in SUB.items():
        base = D["macro"]
        bh, rf = series(base, folds, "Buy&Hold")
        mt, _ = series(base, folds, "MA-tuned")
        sb = stats_from_returns(bh, rf)
        out.append(f"## {name}\nReference: Buy&Hold Sharpe {sb['sharpe']:.2f}, ann. return {100*sb['ann_return']:.1f}%, vol {100*sb['ann_vol']:.1f}%, maxDD {100*sb['max_drawdown']:.1f}%; "
                   f"MA-tuned Sharpe {stats_from_returns(mt, rf)['sharpe']:.2f}.\n")
        out.append("| variant | #feat | ENS Sharpe | ENS ann.ret | ENS vol | ENS maxDD | single-seed Sharpe (mean±sd) | vs Buy&Hold: Sharpe diff [95% CI], P>0 | vs macro: Sharpe diff [95% CI], P>0 | beats % of random |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        from data import VARIANTS
        em, emrf = series(D["macro"], folds, "DQN-ens")
        for v, d in D.items():
            e, erf = series(d, folds, "DQN-ens")
            s = stats_from_returns(e, erf)
            sing = stitched(d, folds)[2]
            sing = sing[sing.strategy == "DQN-single"].sharpe.values
            ps, ci, pp, _, _ = boot(e, bh, erf)
            if v == "macro":
                vm = "(reference)"
            else:
                ps2, ci2, pp2, _, _ = boot(e, em, erf)
                vm = f"{ps2:+.2f} [{ci2[0]:+.2f}, {ci2[1]:+.2f}], {pp2:.2f}"
            rt = d[(d.strategy == "Random-TM") & d.fold.isin(list(folds))]
            rsh = [stats_from_returns(g.sort_values(["fold", "date"]).ret.values, g.sort_values(["fold", "date"]).rf.values)["sharpe"] for _, g in rt.groupby("seed")]
            out.append(f"| {v} | {len(VARIANTS[v])+1} | {s['sharpe']:.2f} | {100*s['ann_return']:.1f}% | {100*s['ann_vol']:.1f}% | {100*s['max_drawdown']:.1f}% | "
                       f"{sing.mean():.2f} ± {sing.std(ddof=1):.2f} | {ps:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], {pp:.2f} | {vm} | {100*np.mean(np.array(rsh) < s['sharpe']):.0f}% |")
        out.append("")
    # per-fold ensemble Sharpe-free view: total return per fold
    rows = {}
    for v, d in D.items():
        g = d[d.strategy == "DQN-ens"].groupby("fold").ret.apply(lambda x: (1 + x).prod() - 1)
        rows[v] = g
    rows["Buy&Hold"] = D["macro"][D["macro"].strategy == "Buy&Hold"].groupby("fold").ret.apply(lambda x: (1 + x).prod() - 1)
    rows["MA-tuned"] = D["macro"][D["macro"].strategy == "MA-tuned"].groupby("fold").ret.apply(lambda x: (1 + x).prod() - 1)
    pf = pd.DataFrame(rows)
    pf.index = ["2008-09", "2010-11", "2012-13", "2014-15", "2016-17", "2018-19", "2020-21", "2022-23", "2024"]
    out.append("## Per-fold total return of the ensemble\n")
    out.append((100 * pf).round(1).astype(str).add("%").to_markdown() + "\n")
    txt = "\n".join(out)
    open("results/round2_comparison.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

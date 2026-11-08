import numpy as np


def stats_from_returns(r, rf_daily, expo=None, trades=None, cost=None):
    """r: daily net simple returns; rf_daily: daily risk-free. Sharpe is on excess returns."""
    r, rf = np.asarray(r, float), np.asarray(rf_daily, float)
    eq = np.cumprod(1 + r)
    ex = r - rf
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    out = {
        "total_return": eq[-1] - 1,
        "ann_return": eq[-1] ** (252 / len(r)) - 1,
        "ann_vol": sd * np.sqrt(252),
        "sharpe": ex.mean() / (sd + 1e-12) * np.sqrt(252),
        "max_drawdown": (eq / np.maximum.accumulate(eq) - 1).min(),
    }
    if expo is not None:
        out["avg_exposure"] = float(np.mean(expo))
    if trades is not None:
        out["trades"] = trades
    if cost is not None:
        out["total_cost"] = cost
    return out


def evaluate(env):
    return stats_from_returns(env.rets, env.rfs, env.expo, env.n_trades, env.total_cost)

import numpy as np

def evaluate(env, rf=0.0):
    v = np.array(env.values)
    r = v[1:] / v[:-1] - 1
    ann = 252
    ret = v[-1] / v[0] - 1
    vol = r.std(ddof=1) * np.sqrt(ann)
    sharpe = (r.mean() - rf / ann) / (r.std(ddof=1) + 1e-12) * np.sqrt(ann)
    dd = (v / np.maximum.accumulate(v) - 1).min()
    return {"total_return": ret, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": dd,
            "trades": env.n_trades, "total_cost": env.total_cost, "final_value": v[-1]}

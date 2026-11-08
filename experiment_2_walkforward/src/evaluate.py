"""Walk-forward evaluation on the TEST blocks only.

Strategies per fold (all share the same env: next-open execution, costs, cash interest):
  DQN-ens   : average-Q ensemble of all seeds           DQN-single : each seed on its own
  Buy&Hold  : 100% invested from the start
  VolMatch  : constant exposure = DQN-ens' average exposure in that fold (rebalanced daily, same costs)
  MA-tuned  : SMA(fast/slow) crossover, pair chosen on the VALIDATION window of that fold
  Random-TM : random exposure changes with the same trade frequency as DQN-ens (20 seeds)
Outputs results/<variant>_<ticker>_daily.csv (every daily return) and _folds.csv.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build, fold_split, TEST_BLOCKS
from env import TradingEnv, LEVELS
from agent import Ensemble
from metrics import evaluate

MA_GRID = [(5, 20), (10, 30), (10, 50), (20, 100), (50, 200)]


def run_agent(agent, df, cost):
    env = TradingEnv(df, cost=cost); s, _ = env.reset(); done = False
    while not done:
        s, _, done, _, _ = env.step(agent.act(s, greedy=True))
    return env


def run_const(df, cost, w, force=False):
    env = TradingEnv(df, cost=cost); env.reset(); done = False
    while not done:
        _, _, done, _, _ = env.step_level(w, force=force)
    return env


def run_ma(df, cost, fast, slow):
    env = TradingEnv(df, cost=cost); env.reset(); done = False
    while not done:
        t = env.t
        _, _, done, _, _ = env.step_level(1.0 if env.sma[fast][t] > env.sma[slow][t] else 0.0)
    return env


def run_random(df, cost, p, rng):
    env = TradingEnv(df, cost=cost); env.reset(); done = False
    while not done:
        w = float(rng.choice(LEVELS)) if rng.random() < p else env.level
        _, _, done, _, _ = env.step_level(w)
    return env


def main(variant, ticker, seeds, cost, models_dir="models", hidden=64, tag=""):
    df = build(ticker, variant)
    n_in = len(df.attrs["feats"]) + 1
    daily, rows = [], []

    def rec(strategy, fold, seed, env, dates):
        m = evaluate(env); m.update(strategy=strategy, fold=fold, seed=seed); rows.append(m)
        daily.append(pd.DataFrame({"date": dates[1:1 + len(env.rets)], "ret": env.rets, "rf": env.rfs,
                                   "expo": env.expo, "strategy": strategy, "fold": fold, "seed": seed}))

    for k in range(len(TEST_BLOCKS)):
        sp = fold_split(df, k); te, va = sp["test"], sp["val"]
        paths = sorted(glob.glob(f"{models_dir}/{tag}{variant}_{ticker}_f{k}_s*.pt"))[:seeds]
        if not paths:
            continue
        for i, p in enumerate(paths):
            rec("DQN-single", k, i, run_agent(Ensemble([p], n_in, hidden), te, cost), te.index)
        ens = run_agent(Ensemble(paths, n_in, hidden), te, cost)
        rec("DQN-ens", k, 0, ens, te.index)
        rec("Buy&Hold", k, 0, run_const(te, cost, 1.0), te.index)
        w = float(np.mean(ens.expo))
        rec("VolMatch", k, 0, run_const(te, cost, w, force=True), te.index)
        best = max(MA_GRID, key=lambda fs: evaluate(run_ma(va, cost, *fs))["sharpe"])
        rec("MA-tuned", k, 0, run_ma(te, cost, *best), te.index)
        p_sw = max(ens.n_trades, 1) / len(ens.rets)
        for sd in range(20):
            rec("Random-TM", k, sd, run_random(te, cost, p_sw, np.random.default_rng(sd)), te.index)
        print(f"fold {k} {TEST_BLOCKS[k]} done: seeds={len(paths)} MA={best} volmatch_w={w:.2f}", flush=True)
    os.makedirs("results", exist_ok=True)
    pd.concat(daily).to_csv(f"results/{tag}{variant}_{ticker}_c{cost}_daily.csv", index=False)
    pd.DataFrame(rows).to_csv(f"results/{tag}{variant}_{ticker}_c{cost}_folds.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="price"); ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.variant, a.ticker, a.seeds, a.cost, hidden=a.hidden, tag=a.tag)

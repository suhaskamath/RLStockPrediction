"""Evaluate trained DQN seeds vs baselines on the held-out TEST period; make tables & plots."""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data import load_splits
from env import TradingEnv
from agent import DQNAgent
from baselines import run_policy, buy_and_hold, make_random, make_ma_crossover
from metrics import evaluate
from train import greedy_run

N_OBS = 11  # 8 market features + position, cash fraction, portfolio value


def dqn_envs(ticker, cost, df, n_seeds):
    envs = []
    for sd in range(n_seeds):
        p = f"models/dqn_{ticker}_c{cost}_s{sd}.pt"
        if not os.path.exists(p): continue
        ag = DQNAgent(N_OBS); ag.load(p)
        envs.append(greedy_run(ag, df, cost))
    return envs


def main(ticker="SPY", cost=0.001, seeds=5, split="test"):
    df = load_splits(ticker)[split]
    rows, curves = [], {}
    def add(name, env):
        rows.append({"strategy": name, **evaluate(env)}); curves.setdefault(name, []).append(env.values)
    add("Buy & Hold", run_policy(TradingEnv(df, cost=cost), buy_and_hold))
    add("MA crossover (5/20)", run_policy(TradingEnv(df, cost=cost), make_ma_crossover()))
    for sd in range(seeds):
        add("Random", run_policy(TradingEnv(df, cost=cost), make_random(sd)))
    for env in dqn_envs(ticker, cost, df, seeds):
        add("DQN", env)
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv(f"results/{ticker}_{split}_c{cost}_runs.csv", index=False)
    cols = ["total_return", "ann_vol", "sharpe", "max_drawdown", "trades", "total_cost"]
    summ = res.groupby("strategy")[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(f"results/{ticker}_{split}_c{cost}_summary.csv")
    pd.set_option("display.width", 200)
    print(f"\n== {ticker} {split} cost={cost} (mean over runs) ==")
    print(res.groupby("strategy")[cols].mean().round(3))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); idx = df.index
    for name, cs in curves.items():
        a = np.array(cs) / 10_000; m = a.mean(0)
        ax[0].plot(idx, m, label=name); ax[1].plot(idx, m - 1, label=name)
        if len(cs) > 1: ax[0].fill_between(idx, a.min(0), a.max(0), alpha=.15)
    ax[0].set_title(f"{ticker} portfolio value, x initial ({split}, cost={cost})"); ax[1].set_title("Cumulative return")
    for x in ax: x.legend(); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"results/{ticker}_{split}_c{cost}_portfolio.png", dpi=130); plt.close()
    p0 = f"models/dqn_{ticker}_c{cost}_s0.pt"
    if os.path.exists(p0):
        ag = DQNAgent(N_OBS); ag.load(p0)
        env = TradingEnv(df, cost=cost); s, _ = env.reset(); done = False; buys, sells = [], []
        while not done:
            a = ag.act(s, greedy=True); pos = env.shares > 0
            if a == 1 and not pos: buys.append(env.t)
            if a == 2 and pos: sells.append(env.t)
            s, _, done, _, _ = env.step(a)
        plt.figure(figsize=(12, 4)); plt.plot(idx, env.prices, lw=1, color="gray")
        plt.scatter(idx[buys], env.prices[buys], marker="^", c="g", s=25, label="buy")
        plt.scatter(idx[sells], env.prices[sells], marker="v", c="r", s=25, label="sell")
        plt.legend(); plt.title(f"DQN trades (seed 0) - {ticker} {split}"); plt.tight_layout()
        plt.savefig(f"results/{ticker}_{split}_c{cost}_trades.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY"); ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--split", default="test")
    a = ap.parse_args(); main(a.ticker, a.cost, a.seeds, a.split)

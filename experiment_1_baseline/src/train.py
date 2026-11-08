"""Train DQN with several seeds; select checkpoint on VALIDATION data only."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import load_splits
from env import TradingEnv
from agent import DQNAgent
from metrics import evaluate


def greedy_run(agent, df, cost):
    env = TradingEnv(df, cost=cost)
    s, _ = env.reset(); done = False
    while not done:
        s, _, done, _, _ = env.step(agent.act(s, greedy=True))
    return env


def train(ticker="SPY", seed=0, episodes=40, cost=0.001, out="models", tag="", **kw):
    sp = load_splits(ticker)
    env = TradingEnv(sp["train"], cost=cost)
    agent = DQNAgent(env.observation_space.shape[0], seed=seed, **kw)
    best, hist = -1e9, []
    path = os.path.join(out, f"dqn_{ticker}_c{cost}_s{seed}{tag}.pt")
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done = False; tot = 0
        while not done:
            a = agent.act(s)
            s2, r, done, _, _ = env.step(a)
            agent.mem.push(s, a, r, s2, float(done)); agent.learn()
            s, tot = s2, tot + r
        agent.end_episode()
        val = evaluate(greedy_run(agent, sp["val"], cost))["sharpe"]
        hist.append({"ep": ep, "train_logret": tot, "val_sharpe": val, "eps": agent.eps})
        if val > best:
            best = val; agent.save(path)
        print(f"[{ticker} s{seed}] ep{ep:02d} train_logret={tot:+.3f} val_sharpe={val:+.2f} eps={agent.eps:.2f}", flush=True)
    return path, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=40); ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--no-double", action="store_true")
    a = ap.parse_args()
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    logs = {}
    for sd in range(a.seeds):
        _, logs[sd] = train(a.ticker, sd, a.episodes, a.cost, lr=a.lr, gamma=a.gamma, double=not a.no_double)
    json.dump(logs, open(f"results/train_log_{a.ticker}_c{a.cost}.json", "w"))

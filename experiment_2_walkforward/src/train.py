"""Train one DQN (variant, ticker, fold, seed).

Anti-memorisation: every episode is a RANDOM WINDOW of the training history (random start, random
initial exposure) instead of replaying the same full path.
Robust selection: validation Sharpe is measured every `eval_every` episodes on the whole validation
period; the checkpoint with the best 2-point-smoothed validation Sharpe is kept (test data never used).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build, fold_split
from env import TradingEnv, LEVELS
from agent import DQNAgent
from metrics import evaluate

DEFAULT = dict(episodes=120, window=504, eval_every=10, lr=3e-4, gamma=0.95, hidden=64, wd=1e-4, cost=0.0005, train_freq=2)


def greedy_run(agent, df, cost):
    env = TradingEnv(df, cost=cost)
    s, _ = env.reset(); done = False
    while not done:
        s, _, done, _, _ = env.step(agent.act(s, greedy=True))
    return env


def train_one(variant, ticker, fold, seed, tag="", out="models", **over):
    cfg = {**DEFAULT, **over}
    path = os.path.join(out, f"{tag}{variant}_{ticker}_f{fold}_s{seed}.pt")
    logp = path.replace(".pt", ".json")
    if os.path.exists(logp):
        return json.load(open(logp))
    df = build(ticker, variant)
    sp = fold_split(df, fold)
    tr, va = sp["train"], sp["val"]
    env = TradingEnv(tr, cost=cfg["cost"])
    rng = np.random.default_rng(1000 * fold + seed)
    agent = DQNAgent(env.observation_space.shape[0], lr=cfg["lr"], gamma=cfg["gamma"],
                     hidden=cfg["hidden"], wd=cfg["wd"], seed=seed)
    E, W = cfg["episodes"], min(cfg["window"], len(tr) - 2)
    best, vals, t0, k = -1e9, [], time.time(), 0
    for ep in range(E):
        agent.eps = max(0.05, 1.0 - 0.95 * ep / (0.6 * E))
        start = int(rng.integers(0, len(tr) - W - 1))
        s, _ = env.reset(start=start, length=W, level0=float(rng.choice(LEVELS))); done = False
        while not done:
            a = agent.act(s)
            s2, r, done, _, _ = env.step(a)
            agent.mem.push(s, a, r, s2, float(done)); k += 1
            if k % cfg["train_freq"] == 0: agent.learn()
            s = s2
        if (ep + 1) % cfg["eval_every"] == 0:
            v = evaluate(greedy_run(agent, va, cfg["cost"]))["sharpe"]
            vals.append(float(v))
            sm = float(np.mean(vals[-2:]))
            if sm > best:
                best = sm; agent.save(path)
    log = {"variant": variant, "ticker": ticker, "fold": fold, "seed": seed, "cfg": cfg,
           "val_sharpe": vals, "best_smoothed": best, "last4_mean": float(np.mean(vals[-4:])),
           "secs": time.time() - t0}
    json.dump(log, open(logp, "w"))
    return log


def _job(a):
    v, t, f, s, tag, out, over = a
    return train_one(v, t, f, s, tag=tag, out=out, **over)

"""Benchmark strategies. Each returns a list of actions evaluated through the same env."""
import numpy as np
from env import HOLD, BUY, SELL


def run_policy(env, policy):
    s, _ = env.reset()
    done = False
    while not done:
        s, _, done, _, _ = env.step(policy(env, s))
    return env


def buy_and_hold(env, s):
    return BUY if env.t == 0 else HOLD


def make_random(seed):
    rng = np.random.default_rng(seed)
    return lambda env, s: int(rng.integers(3))


def make_ma_crossover(fast=5, slow=20):
    """Long when SMA(fast) > SMA(slow), flat otherwise (uses prices up to day t)."""
    def pol(env, s):
        if env.t < slow:
            return HOLD
        p = env.prices[: env.t + 1]
        up = p[-fast:].mean() > p[-slow:].mean()
        invested = env.shares > 0
        return BUY if (up and not invested) else SELL if (not up and invested) else HOLD
    return pol

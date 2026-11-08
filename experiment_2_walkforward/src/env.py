"""Single-asset, long-only trading env with target-exposure actions and realistic timing.

Timing : features are computed at the close of day t; the decision is EXECUTED AT THE OPEN of day t+1
         (no trading at a price you have already used to compute the signal).
Actions: 0/1/2 -> target exposure 0% / 50% / 100% of equity in the asset (rest in cash).
         Choosing the same level as before = no trade (exposure just drifts with the price).
Costs  : `cost` x traded notional (fee + spread + slippage, per side).
Cash   : earns the 3-month T-bill rate (daily).
Reward : log(V_{t+1} / V_t), net of costs and including cash interest and the overnight gap.
State  : market features at t + current exposure fraction. (No portfolio-value feature: it leaked
         the position in the episode and was out-of-range at test time in v1.)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data import MA_WINDOWS

LEVELS = (0.0, 0.5, 1.0)


class TradingEnv(gym.Env):
    def __init__(self, df, cost=0.0005, initial_cash=10_000.0):
        super().__init__()
        self.O = df["Open"].values.astype(np.float64)
        self.C = df["Close"].values.astype(np.float64)
        self.rf = (df["rf"].values / 252.0).astype(np.float64)
        self.X = df[df.attrs["feats"]].values.astype(np.float32)
        self.sma = {w: df[f"sma{w}"].values for w in MA_WINDOWS}
        self.dates = df.index
        self.cost, self.cash0, self.n = cost, initial_cash, len(df)
        self.action_space = spaces.Discrete(len(LEVELS))
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.X.shape[1] + 1,), np.float32)

    def _exposure(self):
        v = self.cash + self.shares * self.C[self.t]
        return self.shares * self.C[self.t] / v

    def _obs(self):
        return np.concatenate([self.X[self.t], [self._exposure()]]).astype(np.float32)

    def reset(self, seed=None, options=None, start=0, length=None, level0=0.0):
        super().reset(seed=seed)
        self.t0 = self.t = start
        self.end = self.n - 1 if length is None else min(self.n - 1, start + length)
        self.cash = self.cash0 * (1 - level0)
        self.shares = self.cash0 * level0 / self.C[start]
        self.level = level0
        self.n_trades, self.total_cost = 0, 0.0
        self.values, self.rets, self.rfs, self.expo = [self.cash0], [], [], []
        return self._obs(), {}

    def step_level(self, w, force=False):
        t = self.t
        v_prev = self.cash + self.shares * self.C[t]
        o, c = self.O[t + 1], self.C[t + 1]
        paid = 0.0
        if force or w != self.level:
            v_open = self.cash + self.shares * o
            traded = abs(w * v_open - self.shares * o)
            paid = traded * self.cost
            if traded > 1e-9:
                self.n_trades += 1
            v_after = v_open - paid
            self.shares = w * v_after / o
            self.cash = (1 - w) * v_after
            self.level = w
        self.cash *= 1.0 + self.rf[t]
        v1 = self.cash + self.shares * c
        self.total_cost += paid
        self.t += 1
        self.values.append(v1)
        self.rets.append(v1 / v_prev - 1.0)
        self.rfs.append(self.rf[t])
        self.expo.append(self.shares * c / v1)
        done = self.t >= self.end
        return self._obs(), float(np.log(v1 / v_prev)), done, False, {}

    def step(self, action):
        return self.step_level(LEVELS[int(action)])

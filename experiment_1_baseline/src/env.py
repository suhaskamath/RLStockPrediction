"""Gymnasium-style single-asset trading environment (long-only, all-in/all-out).

Actions: 0 = hold, 1 = buy (invest all cash), 2 = sell (liquidate all shares).
Invalid actions (buy with no cash, sell with no shares) are treated as hold.
Shares are fractional; a buy spends all cash, a sell liquidates all shares.
Decision at close of day t, executed at close of day t, reward over t -> t+1:
    r_t = log(V_{t+1} / V_t)
The fee is deducted from cash/shares when the trade happens, so it is already inside V_{t+1}.
Do NOT subtract the cost again in the reward, or it would be counted twice.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data import FEATURES

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    def __init__(self, df, cost=0.001, initial_cash=10_000.0, window_feats=FEATURES):
        super().__init__()
        self.prices = df["Close"].values.astype(np.float64)
        self.X = df[window_feats].values.astype(np.float32)
        self.dates = df.index
        self.cost, self.cash0 = cost, initial_cash
        self.n = len(self.prices)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.X.shape[1] + 3,), np.float32)

    def _value(self):
        return self.cash + self.shares * self.prices[self.t]

    def _obs(self):
        v = self._value()
        pos = self.shares * self.prices[self.t] / v          # fraction in stock
        cash_frac = self.cash / v
        pv = np.log(v / self.cash0)                          # normalised portfolio value
        return np.concatenate([self.X[self.t], [pos, cash_frac, pv]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.cash, self.shares = 0, self.cash0, 0.0
        self.n_trades, self.total_cost = 0, 0.0
        self.values = [self.cash0]
        return self._obs(), {}

    def step(self, action):
        p = self.prices[self.t]
        v_pre = self._value()
        paid = 0.0
        if action == BUY and self.cash > 1e-8:
            spend = self.cash
            paid = spend * self.cost / (1 + self.cost)       # fee on notional
            self.shares += (spend - paid) / p
            self.cash = 0.0
            self.n_trades += 1
        elif action == SELL and self.shares > 1e-12:
            gross = self.shares * p
            paid = gross * self.cost
            self.cash += gross - paid
            self.shares = 0.0
            self.n_trades += 1
        self.total_cost += paid
        self.t += 1
        v1 = self._value()
        # log growth of the portfolio, net of the transaction cost paid on this step
        reward = float(np.log(v1 / v_pre))
        self.values.append(v1)
        done = self.t >= self.n - 1
        return self._obs(), reward, done, False, {"value": v1, "cost": paid}

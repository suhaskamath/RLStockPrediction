"""DQN / Double-DQN agent: online net, target net, replay buffer, epsilon-greedy."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, n_in, n_out=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, *tr):
        self.buf.append(tr)

    def sample(self, k):
        b = random.sample(self.buf, k)
        s, a, r, s2, d = map(np.array, zip(*b))
        return (torch.tensor(s, dtype=torch.float32), torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32), torch.tensor(s2, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, n_in, lr=1e-3, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.97,
                 buffer=50_000, batch=64, target_every=500, hidden=64, double=True, seed=0):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.q, self.tq = QNet(n_in, hidden=hidden), QNet(n_in, hidden=hidden)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.mem = Replay(buffer)
        self.gamma, self.batch, self.target_every, self.double = gamma, batch, target_every, double
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

    def act(self, s, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(3)
        with torch.no_grad():
            return int(self.q(torch.tensor(s).unsqueeze(0)).argmax(1))

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def learn(self):
        if len(self.mem) < self.batch:
            return
        s, a, r, s2, d = self.mem.sample(self.batch)
        r = r * 100  # rewards are small log-returns; scale for stable gradients
        qsa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(s2).argmax(1, keepdim=True)
                nxt = self.tq(s2).gather(1, a2).squeeze(1)
            else:
                nxt = self.tq(s2).max(1).values
            y = r + self.gamma * nxt * (1 - d)
        loss = nn.functional.smooth_l1_loss(qsa, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_every == 0:
            self.tq.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path)); self.q.eval()

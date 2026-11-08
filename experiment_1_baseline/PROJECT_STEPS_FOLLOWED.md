# Experiment 1: Steps Followed

This document records how we built experiment 1, what it ran on, how the pieces fit together, and what we found.


## 1. Computing Environment

We ran everything in a JupyterHub container on Linux (Ubuntu, kernel 6.8, x86_64) with 8 CPU cores. Training used the CPU only. The code never moves data to a GPU, and a short test run used about six times more CPU time than clock time, which fits multi threaded CPU work. Each training job was limited to 2 threads, and several jobs ran at the same time.

| Item | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0 |
| Gymnasium | 1.2.2 |
| pandas | 3.0.5 |
| yfinance | 1.8.0 |

The market data is real daily history from Yahoo! Finance for 2005 to 2024. The simulator is our own Gymnasium style environment (`src/env.py`) that replays that history.


## 2. Steps Followed

1. We read the project proposal and checked which packages were already installed.
2. We installed `yfinance` and downloaded SPY for 2005 to 2024. We chose SPY because the proposal specifies it.
3. We wrote the code for data, environment, agent, baselines, metrics, training, and evaluation.
4. We changed the reward so that the trade cost comes out of the portfolio value, and the reward is the log of end value over pre trade value.
5. We ran a smoke test of the baselines and a two episode training run.
6. We ran the full SPY pipeline with three cost settings (0%, 0.1%, 0.5%), 5 seeds each, 40 episodes per seed, and then evaluated on the test period.
7. We wrote the README while training ran.
8. We extended the pipeline to AAPL, MSFT, AMZN, and TSLA with a 0.1% cost and 5 seeds each. TSLA trains from September 2010 because it listed that year.
9. We recorded the tables, portfolio and trade plots, and training logs in `results/`.

We did not clone or open the Q Trader repository. The design comes from the proposal, and the code was written from scratch.


## 3. Architecture

### Pipeline

```
 Yahoo! Finance (yfinance)
          |
          v
 +-------------------+     data/<TICKER>.csv
 |  src/data.py      |---> 8 features per day, scaled with fixed constants
 +---------+---------+
           |  chronological split
           |  train 2005-2017 | validation 2018-2019 | test 2020-2024
           v
 +-------------------+   state (11)    +----------------------------+
 |  src/env.py       |---------------->|  src/agent.py (Double DQN) |
 |  cash, shares,    |<----------------|  online net, target net,   |
 |  cost, reward     |  action + reward|  replay buffer, eps-greedy |
 +-------------------+                 +-------------+--------------+
                                                     |
                                    src/train.py: 5 seeds, best validation Sharpe kept
                                                     v
                                          models/dqn_<T>_c<cost>_s<seed>.pt
                                                     |
 +--------------------------------+     +------------+---------------+
 | src/evaluate.py (test period)  |<----| src/baselines.py           |
 | DQN + baselines, src/metrics.py|     | buy and hold, random, MA   |
 +----------------+---------------+     +----------------------------+
                  v
   results/: summary tables, per run tables, portfolio plots, trade plots
```

### Q network

```
 state (11 inputs)
   8 market features: ret1, ret5, ret10, ret20, MA ratio, price/MA20, volume z score, volatility
   3 portfolio values: position fraction, cash fraction, log portfolio value
        |
   Linear(11 -> 64) + ReLU
   Linear(64 -> 64) + ReLU
   Linear(64 -> 3)  -> Q(hold), Q(buy), Q(sell); the largest value is the action
```

### Settings

| Component | Setting |
|---|---|
| Actions | 0 hold, 1 buy all cash, 2 sell everything |
| Reward | log(V(t+1) / V(t)) with the cost deducted (scaled by 100 inside the learner) |
| Algorithm | Double DQN, Huber loss, Adam with learning rate 0.001, gradient clipping at 5 |
| Discount factor | 0.95 |
| Exploration | Epsilon from 1.0 to 0.05, multiplied by 0.97 each episode |
| Replay buffer and batch | 50,000 and 64 |
| Target network | Copied every 500 updates |
| Training | 40 episodes per seed, seeds 0 to 4 |
| Model selection | Best validation Sharpe ratio, never the test period |
| Starting cash and cost | $10,000 and 0.1% per trade |


## 4. Results

The test period runs from 2 January 2020 to 31 December 2024 (1,258 trading days). Numbers are averages over 5 seeds for the DQN and random strategies. Buy and hold and the moving average are deterministic.

| Ticker | Buy and hold | Moving average 5/20 | DQN | Random |
|---|---|---|---|---|
| SPY | 94% / 0.74 | 88% / 1.06 | 41% / 0.44 | -8% / -0.09 |
| AAPL | 244% / 0.94 | 179% / 1.11 | 151% / 0.82 | 27% / 0.29 |
| MSFT | 174% / 0.82 | 21% / 0.29 | 97% / 0.65 | 13% / 0.18 |
| AMZN | 131% / 0.65 | 70% / 0.55 | 37% / 0.31 | -7% / 0.01 |
| TSLA | 1307% / 1.12 | 1105% / 1.27 | 347% / 0.76 | 437% / 0.72 |

Each cell shows total return and Sharpe ratio.

More detail for SPY with a 0.1% cost:

| Strategy | Return | Volatility | Sharpe | Max drawdown | Trades | Fees paid |
|---|---|---|---|---|---|---|
| Buy and hold | 94.4% | 21.0% | 0.74 | -33.7% | 1 | $10 |
| Moving average | 88.1% | 12.7% | 1.06 | -16.4% | 62 | $896 |
| DQN (mean of 5) | 40.5% (std 38) | 17.0% | 0.44 | -30.6% | 205 | $2,335 |
| Random (mean of 5) | -7.9% | 14.5% | -0.09 | -38.1% | 416 | $3,825 |

Effect of the trading cost on the SPY DQN:

| Cost per trade | Return | Sharpe | Trades | Buy and hold return |
|---|---|---|---|---|
| 0% | 69.3% | 0.73 | 459 | 94.6% |
| 0.1% | 40.5% | 0.44 | 205 | 94.4% |
| 0.5% | 53.0% | 0.59 | 77 | 93.6% |

At the highest cost the agent traded much less, which probably explains why it did better than at 0.1%. We did not test this directly, and the difference is within the seed to seed spread.

What we take from this:

* The DQN beat random trading on four of the five assets, but it did not beat buy and hold on any of them. The 2020 to 2024 period was a strong bull market, which is hard to beat by moving in and out.
* The DQN had lower volatility than buy and hold, but not enough to raise its Sharpe ratio.
* The results depend heavily on the seed. On SPY the five seeds returned between 3% and 90%.
* The simple 5/20 moving average rule had the best Sharpe ratio on SPY, AAPL, and TSLA.

Plots of portfolio value and of the buy and sell points are in `results/`.


## 5. Problems We Found Later

A review of these results turned up weaknesses that led to experiment 2.

* **The agent memorized the training data.** Every episode replayed the same 3,213 days of SPY. The training return was between 30 and 90 times the starting money, while buy and hold made 2.9 times over the same years.
* **The state included the portfolio value.** The agent could use it to tell where it was in an episode, and its range in training was very different from its range in testing.
* **Model selection was optimistic.** We kept the best of 40 noisy validation scores on only 502 days, and every seed scored lower on the test period than on validation.
* **Only one test period and five seeds.** A single bull market with a large spread between seeds cannot support strong conclusions.
* **The comparisons were weak.** Random trading made about 416 trades, the moving average was not tuned, and cash earned nothing.
* **Trading was optimistic.** The agent traded at the same close that it used to compute its signal.
* **The stocks were chosen after the fact.** AAPL, MSFT, AMZN, and TSLA were all big winners over the test period.

Experiment 2 addresses these points. Its log is in `../experiment_2_walkforward/IMPROVEMENT_LOG.md`.

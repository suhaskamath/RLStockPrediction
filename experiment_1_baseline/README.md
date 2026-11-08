# Experiment 1: Baseline DQN Trader

## 1. Overview

This experiment is the first implementation of our project proposal (`../docs/`). A Double DQN agent chooses to buy, sell, or hold SPY each day, and we compare it with buy and hold, random trading, and a moving average crossover on a test period that the agent never saw during training. It is a historical simulation, so we make no claim about real world profit.

The agent did not beat buy and hold. We later found weaknesses in this setup, such as the agent memorizing its training data, and we fixed them in experiment 2. The step by step write-up is in `PROJECT_STEPS_FOLLOWED.md`.


## 2. Data and Method

We download daily SPY prices and volume from Yahoo! Finance with `yfinance` and cache them in `data/SPY.csv`. The split is by date. Training uses 2005 to 2017, validation uses 2018 to 2019 to choose the saved model, and testing uses 2020 to 2024. The test period is not used for features or tuning.

* **State (11 values).** Returns over 1, 5, 10, and 20 days, the ratio of the 5 day to the 20 day moving average, price relative to the 20 day average, a volume z score, and 20 day volatility. We add the position fraction, the cash fraction, and the log of the portfolio value. Every feature uses data up to day t only.
* **Actions.** Hold, buy (put all cash into shares), or sell (sell all shares). Buying with no cash or selling with no shares counts as hold.
* **Reward.** log(V(t+1) / V(t)), with the cost of any trade taken out of the portfolio value.
* **Agent.** Double DQN with two hidden layers of 64 units, a replay buffer of 50,000, batches of 64, and a target network copied every 500 updates. Epsilon falls from 1.0 to 0.05 (multiplied by 0.97 each episode), the discount factor is 0.95, and the learning rate is 0.001. Each seed trains for 40 episodes.
* **Reliability.** We train 5 random seeds and report the mean and standard deviation.


## 3. Files

| File | Purpose |
|---|---|
| `src/data.py` | Downloads data, builds features, makes the chronological split |
| `src/env.py` | Trading environment with cash, shares, portfolio value, transaction cost, and invalid action handling |
| `src/agent.py` | DQN and Double DQN agent |
| `src/baselines.py` | Buy and hold, random, and moving average crossover (5/20) |
| `src/metrics.py` | Return, volatility, maximum drawdown, Sharpe ratio, trades, and total cost |
| `src/train.py` | Trains several seeds and keeps the checkpoint that scores best on validation |
| `src/evaluate.py` | Produces the test tables and plots in `results/` |
| `run_all.sh` | Runs the whole pipeline, including the transaction cost test (0%, 0.1%, 0.5%) |
| `run_stocks.sh` | Repeats the pipeline for AAPL, MSFT, AMZN, and TSLA |


## 4. How to Run

```
pip install -r requirements.txt
./run_all.sh
./run_stocks.sh
```

For a single stock: `python src/data.py AAPL`, then `python src/train.py --ticker AAPL --seeds 5`, then `python src/evaluate.py --ticker AAPL`.


## 5. Results

All results are on the 2020 to 2024 test period with a 0.1% cost per trade, averaged over 5 seeds for the DQN and random strategies. Each cell shows total return and Sharpe ratio.

| Ticker | Buy and hold | Moving average | DQN | Random |
|---|---|---|---|---|
| SPY | 94% / 0.74 | 88% / 1.06 | 41% / 0.44 | -8% / -0.09 |
| AAPL | 244% / 0.94 | 179% / 1.11 | 151% / 0.82 | 27% / 0.29 |
| MSFT | 174% / 0.82 | 21% / 0.29 | 97% / 0.65 | 13% / 0.18 |
| AMZN | 131% / 0.65 | 70% / 0.55 | 37% / 0.31 | -7% / 0.01 |
| TSLA | 1307% / 1.12 | 1105% / 1.27 | 347% / 0.76 | 437% / 0.72 |

The DQN earned less than buy and hold on every asset. It beat random trading on SPY, AAPL, MSFT, and AMZN but not on TSLA. On SPY the five seeds ranged from 3% to 90% return, so the average is not very reliable. Trading costs also took a large share of the gains, because the agent made between 200 and 380 trades. With no cost the SPY return was 69%, with a 0.1% cost it was 41%, and with a 0.5% cost it was 53%, where the agent learned to trade much less.

TSLA only began trading in 2010, so its training period is shorter (2010 to 2017). The results for each asset, with plots of portfolio value and trades, are in `results/`.

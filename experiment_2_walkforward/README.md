# Experiment 2: Walk Forward DQN Trader

## 1. Overview

This experiment rebuilds the training and testing from experiment 1 after we found that its results were weaker than they looked. The agent is still a Double DQN trading SPY, but it now trains on random windows of history, chooses its model more carefully, and is tested on nine separate periods from 2008 to 2024. It is compared against benchmarks that are harder to beat than the ones in experiment 1.

The full record of what we did and why is in `IMPROVEMENT_LOG.md`.


## 2. What Changed from Experiment 1

* **Training.** Each episode is a random 504 day window of the training history with a random starting position. Before, every episode replayed the same 3,213 days and the agent memorized them.
* **State.** We removed the portfolio value from the state, because it told the agent where it was in an episode. The state is now the market features plus the current exposure.
* **Model selection.** We measure validation performance every 10 episodes, smooth it over two checks, and keep the best checkpoint. We train 10 seeds and combine them by averaging their Q values.
* **Testing.** We use nine walk forward periods (2008 to 2009, 2010 to 2011, up to 2024), each trained only on earlier data, and we use bootstrap confidence intervals instead of single numbers.
* **Benchmarks.** Buy and hold, a constant exposure version with the same average holding as the DQN, a moving average whose windows are chosen on validation data, and random trading with the same number of trades as the DQN.
* **Trading.** Decisions are made at the close and carried out at the next open. The cost is 0.05% per trade. Cash earns the 3 month T-bill rate. The agent holds 0%, 50%, or 100% of its money in the asset. Shares are fractional, and each decision trades just enough shares to reach the chosen weight. The 0.05% fee is deducted from the portfolio at the moment of the trade, and the reward is the log change in portfolio value, so the cost is counted once. Buy and hold is run in the same environment over the same nine test periods and with the same cost.
* **Data.** We tested several extra data groups on top of price and volume: VIX and interest rates, dollar, gold, and oil, sector ETFs, bonds and credit, and volatility indices.


## 3. Results

The table shows the Sharpe ratio of the 10 seed ensemble. Buy and hold scored 0.55 over all nine periods and 0.72 over 2014 to 2024.

| Data given to the agent | 2008 to 2024 | 2014 to 2024 | Repeat with 10 new seeds |
|---|---|---|---|
| Price and volume only | 0.31 | 0.44 | 0.37 / 0.42 |
| Plus VIX and yields ("macro") | 0.47 | 0.68 | 0.33 / 0.43 |
| Plus dollar, gold, oil | 0.27 | 0.43 | not repeated |
| Plus sector rotation | 0.32 | 0.49 | not repeated |
| Plus bonds, credit, copper | 0.31 | 0.35 | not repeated |
| Plus SKEW, VXN, MOVE | 0.25 | 0.41 | not repeated |
| All of the above | 0.30 | 0.36 | not repeated |

No version of the DQN reliably beat buy and hold, the constant exposure benchmark, or random trading with the same number of trades. The macro variant looked best, but when we trained it again with 10 new seeds its 2014 to 2024 score dropped from 0.68 to 0.43, the same as the price only agent. We therefore treat the first result as luck. Adding more data sources did not help, and some combinations were significantly worse than buy and hold. The agent also lost money in 2008 to 2009 in every version, when a tuned moving average gained 23%.

The results also depend on the trading cost. For the macro agent, at 0.02% per trade the Sharpe ratio was about the same as buy and hold, and at 0.1% it was clearly lower. We did not retrain the agent for each cost.


## 4. Files

* `src/` contains the code: `data.py` (features and the nine periods), `env.py`, `agent.py`, `train.py`, `run_train.py` (parallel training), `tune.py` (the validation only search), `evaluate.py`, `aggregate.py`, `compare.py`, and `metrics.py`.
* `run_experiment.sh` lists the commands we ran, in order.
* `results/` holds the summary tables, the comparison of all variants, the tuning and training logs, and `folds/`, which has the metrics of every strategy in every period.

The market data, the model checkpoints, and the per day return files (about 11 MB each) are not stored here. The script recreates them. Run the scripts from this folder, because they use relative paths.

`tune.py` sets the learning rate of the baseline configurations to 0.001. We ran the search before we changed the default to 0.0003, and this setting makes it repeat exactly.

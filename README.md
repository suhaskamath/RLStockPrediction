# Deep Q Learning Driven Stock Trader Bot

## 1. Project Overview and Motivation

For this project we built a Deep Q Network, or DQN, that makes short term trading decisions. Each day the agent looks at recent market information and its current position and chooses how much of the asset to hold. It receives a reward based on how the portfolio value changes after trading costs. This is a historical simulation, so we are not claiming that the model would earn real money.

We started with SPY, the ETF that follows the S&P 500. In the first experiment we also ran the same approach on four individual stocks (AAPL, MSFT, AMZN, and TSLA) to see whether it worked across different assets.

The repository holds two experiments. The first follows our proposal closely. When we reviewed its results we found several weaknesses in how the agent was trained and how it was tested, so the second experiment rebuilds the training and the evaluation. We trust the second experiment more, and its conclusions are the ones we report.


## 2. Research Question and Objectives

Our main research question is: can a DQN agent learn to choose trading actions that perform competitively against standard trading strategies when tested on new historical data? Experiment 2 adds a second question: does extra market data, such as volatility, interest rates, and sector information, help the agent?

* Build a custom trading environment with cash, holdings, portfolio value, and transaction costs.
* Train a DQN with an online Q network, a target network, a replay buffer, and epsilon greedy exploration.
* Compare the DQN with buy and hold, random trading, and a moving average crossover strategy.
* Evaluate return, volatility, maximum drawdown, Sharpe ratio, trade count, and transaction costs.
* Use chronological splits and repeated random seeds so that we do not draw conclusions from one lucky run.


## 3. The Two Experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Folder | `experiment_1_baseline` | `experiment_2_walkforward` |
| Agent | Double DQN, buy / sell / hold | Double DQN, hold 0%, 50%, or 100% in the asset |
| Test data | One period, 2020 to 2024 | Nine periods from 2008 to 2024 (walk forward) |
| Seeds | 5 | 10, combined into an ensemble |
| Benchmarks | Buy and hold, random, moving average 5/20 | Buy and hold, a constant exposure version, a tuned moving average, random trading with the same trade count |
| Trading | Same day close, 0.1% cost, cash earns nothing | Next day open, 0.05% cost, cash earns the T-bill rate |
| Data | Price and volume | Price and volume plus VIX, yields, dollar, gold, oil, sectors, bonds, and volatility indices |
| Write-up | `PROJECT_STEPS_FOLLOWED.md` | `IMPROVEMENT_LOG.md` |


## 4. Results

On SPY in experiment 1, the DQN returned 41% over 2020 to 2024 against 94% for buy and hold, with a Sharpe ratio of 0.44 against 0.74. It did beat random trading, but it did not beat buy and hold on any of the five assets. The result also varied a lot between seeds, with SPY returns ranging from 3% to 90%.

In experiment 2 we found that the first experiment's agent had memorized its training data. Its training return was 30 to 90 times the starting money, while buy and hold made 2.9 times. After fixing this and testing on nine separate periods, the DQN no longer showed a reliable edge. The table shows the Sharpe ratio of the 10 seed ensemble. Buy and hold scored 0.55 over all nine periods and 0.72 over 2014 to 2024.

| Data given to the agent | 2008 to 2024 | 2014 to 2024 | Repeat with 10 new seeds |
|---|---|---|---|
| Price and volume only | 0.31 | 0.44 | 0.37 / 0.42 |
| Plus VIX and yields ("macro") | 0.47 | 0.68 | 0.33 / 0.43 |
| Plus dollar, gold, oil | 0.27 | 0.43 | not repeated |
| Plus sector rotation | 0.32 | 0.49 | not repeated |
| Plus bonds, credit, copper | 0.31 | 0.35 | not repeated |
| Plus SKEW, VXN, MOVE | 0.25 | 0.41 | not repeated |
| All of the above | 0.30 | 0.36 | not repeated |

The macro variant looked best at first, so we repeated it with new seeds. The score fell from 0.68 to 0.43, which is the same as the price only agent, so we treat the first result as luck. None of the extra data sources improved the agent. The DQN also did not protect us in market drops. Every version lost money in 2008 to 2009, and several lost money in 2022 to 2023, while a simple moving average rule gained 23% and 18% in those two periods. At a cost of 0.02% per trade the agent roughly matched buy and hold, and at 0.1% it was clearly worse.

Our conclusion is that, in this setup, the DQN does not reliably beat simply holding the index. We report this as a historical simulation finding only.


## 5. Limitations

Experiment 2 only tests SPY. The multi stock test in experiment 1 used four stocks that we picked after knowing they had done well, which favors buy and hold. Our hyperparameter search was small, and each network trained for only 120 short episodes on a CPU. Even the 10 seed ensemble moves by about 0.2 in Sharpe ratio between repeats, which is as large as the differences we were trying to measure. Daily price direction is also hard to predict in general. Finally, our simulation leaves out issues such as slippage, taxes, and liquidity, and good historical results would not guarantee future performance.


## 6. How to Run

We used Python 3.13 and the packages in `requirements.txt`. Training ran on a CPU only. Run the scripts from inside each experiment folder.

* Experiment 1: `./run_all.sh`, then `./run_stocks.sh` for the extra stocks.
* Experiment 2: `./run_experiment.sh`. This takes about two hours on 8 cores.

Data comes from Yahoo! Finance through `yfinance`, so the numbers may shift slightly if Yahoo revises its history. Market data, model checkpoints, and the per day return files are not stored in the repository, and the scripts recreate them.


## 7. Tools

We used Python, NumPy, Pandas, Matplotlib, SciPy, PyTorch, yfinance, and a custom trading environment based on the Gymnasium style. Claude Code was used as a coding assistant to help write and run the experiments.


## References

1. Mnih, V., et al. (2015). Human level control through deep reinforcement learning. Nature, 518, 529 to 533. https://doi.org/10.1038/nature14236
2. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1). https://doi.org/10.1609/aaai.v30i1.10295
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
4. Lu, E. Q Trader: An implementation of Q learning applied to short term stock trading. GitHub. https://github.com/edwardhdlu/q-trader. Used as a reference for the problem setup only. We did not copy its code.
5. Yahoo! Finance. Historical market data. https://finance.yahoo.com/

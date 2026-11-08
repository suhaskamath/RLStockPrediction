# Experiment 2: Log of What We Did and Why

This log records each step of the rebuild, the reason for it, and what we found. We kept it up to date as we went, including the mistakes and one conclusion we later had to withdraw.


## 1. Why We Rebuilt the Experiment

A review of experiment 1 showed that its conclusions were weaker than they looked. The problems and our fix for each are below.

| Problem in experiment 1 | What we did in experiment 2 |
|---|---|
| The agent replayed one price path in every episode and memorized it. Its training return was 30 to 90 times the starting money, against 2.9 times for buy and hold. | Each episode is a random 504 day window with a random starting position. |
| The portfolio value in the state acted as a clock and was out of range at test time. | We removed it. The state is the market features plus current exposure. |
| We kept the best of 40 noisy validation scores, and every seed did worse on the test period than on validation. | We smooth the validation score and combine 10 seeds into an ensemble. |
| One test period (a bull market) and 5 seeds. | Nine walk forward periods from 2008 to 2024, 10 seeds, and bootstrap confidence intervals. |
| Weak benchmarks, trading at the same close used for the signal, and no interest on cash. | Harder benchmarks, trading at the next open, a 0.05% cost, and interest on cash. |
| Only price and volume as inputs. | Several extra data groups to test. |

We did not address four things in this round: choosing the stocks before looking at results, testing the claim that the agent finds peaks and troughs, trying a recurrent network, and pinning package versions. We pinned the versions later, when we put the repository together.


## 2. Folders and Environment

The original project was left as it was. We took a read only backup of it and did the new work in a separate folder.

We used the same JupyterHub container as before (Linux, 8 cores, CPU only). The packages were Python 3.13, PyTorch 2.11.0, Gymnasium 1.2.2, pandas 3.0.5, and SciPy 1.18.0. The `yfinance` package (1.8.0) was no longer installed in the container, so we reinstalled it. The first version of the project had only worked because the data was already cached. Training ran as 7 parallel processes with one thread each.


## 3. Data

We downloaded the series below from Yahoo! Finance for 1998 to 2024. Everything is computed from data available at the close of day t, and the scaling uses fixed constants, so nothing is fitted on validation or test data.

* **Target:** SPY.
* **Macro:** the VIX, the 10 year yield, and the 3 month T-bill rate (also used as the rate that cash earns).
* **Cross asset:** the dollar index, gold, and oil. Oil futures went negative in April 2020, so we use simple percentage changes for oil and clip the result.
* **Sector (round 2):** XLY, XLP, XLU, XLF, and IWM, measured against SPY.
* **Bonds (round 2):** TLT, LQD, IEF, and copper.
* **Volatility (round 2):** the CBOE SKEW, the VXN, and the MOVE index.

Some series start after 1998 (gold and oil in 2000, the bond ETFs in 2002), and we fill the missing early values with zero. In the earliest period about 55% of the training days have zero filled bond and MOVE values. This mostly affects the early periods, not 2014 to 2024. We left out HYG, VVIX, and VIX3M because they start in 2006 or 2007. Fold 0 would then be tested on inputs it had never seen.


## 4. Walk Forward Periods

We use an expanding window. Each test period lasts two years, validation is the two years before it, and training is everything earlier, starting in 1998. For example, the first period trains on 1998 to 2005, validates on 2006 to 2007, and tests on 2008 to 2009. The nine test periods are 2008 to 2009, 2010 to 2011, 2012 to 2013, 2014 to 2015, 2016 to 2017, 2018 to 2019, 2020 to 2021, 2022 to 2023, and 2024.


## 5. Code

| File | Purpose |
|---|---|
| `src/data.py` | Downloads data, builds the feature sets, makes the periods |
| `src/env.py` | Trading environment with next open execution, trade costs, interest on cash, and 0%, 50%, or 100% exposure |
| `src/agent.py` | Double DQN with weight decay, and the seed ensemble |
| `src/metrics.py` | Return, volatility, Sharpe ratio on excess returns, maximum drawdown, and exposure |
| `src/train.py`, `src/run_train.py` | Trains one network with random windows and smoothed validation selection; the second file runs many jobs in parallel and can resume |
| `src/tune.py` | Validation only hyperparameter search |
| `src/evaluate.py` | Test period evaluation of the DQN and all benchmarks |
| `src/aggregate.py`, `src/compare.py` | Combines the periods and runs the bootstrap tests |


## 6. Log of Steps

**Step 1. Backup and new folder.** We copied the first project to a read only backup and started a new folder for the new code. The original folder was not changed.

**Step 2. Data check.** The first download test failed because `yfinance` was missing. We reinstalled it and confirmed that all the series we needed came back without gaps.

**Step 3. Rewriting the code.** We wrote the new files from scratch so that the new project would not quietly inherit logic from the old one.

**Step 4. Correctness checks.** We tested the code before spending training time. Buy and hold inside the environment returned 52.66% for 2020 to 2021, which matches the raw price calculation exactly. Holding only cash returned 0.37%, the T-bill rate at the time, with no trades. The feature ranges were sensible. A training step ran at about 627 steps per second per core, so we made the agent update every second step to keep the full run near 40 minutes.

**Step 5. Hyperparameter search.** We tried six settings on the first three periods with three seeds each (54 runs). We scored each by the average of its last four validation Sharpe ratios, not its best one, to avoid picking a lucky peak.

| Setting | Mean validation Sharpe |
|---|---|
| Learning rate 0.0003 | 0.226 |
| Discount factor 0.99 | 0.038 |
| Discount factor 0.90 | -0.067 |
| Weight decay 0.001 | -0.122 |
| Original settings (learning rate 0.001) | -0.201 |
| 128 hidden units | -0.229 |

We chose a learning rate of 0.0003 and left everything else at the original values. All of these scores are close to zero, so even the best setting is weak. The search used validation years 2006 to 2011, which are also test years for the first two periods, so those two periods are slightly optimistic. That is why we also report 2014 to 2024 on its own, a range that the search never saw.

**Step 6. Main training.** We trained 3 data variants in 9 periods with 10 seeds each, which is 270 networks, in about 35 minutes.

**Step 7. Evaluation.** We evaluated every variant on the test periods. Two small bugs came up in the aggregation script, one where strategies with a single run lost their rows and one where a table package was missing. We fixed both. We also launched the cost tests from the wrong folder at first, because the scripts use relative paths, and reran them from the project root.


## 7. Round 1 Results

The table shows the stitched out of sample Sharpe ratio, with annual return in brackets. "Clean" means 2014 to 2024.

| Strategy | 2008 to 2024 | 2014 to 2024 |
|---|---|---|
| DQN ensemble, price only | 0.31 (4.9%) | 0.44 (7.2%) |
| DQN ensemble, macro | 0.47 (8.3%) | 0.68 (11.2%) |
| DQN ensemble, cross asset | 0.27 (4.4%) | 0.43 (6.9%) |
| Buy and hold | 0.55 (10.6%) | 0.72 (13.2%) |
| Constant exposure, same average as the DQN | 0.55 (8.8%) | 0.70 (10.7%) |
| Tuned moving average | 0.49 (6.8%) | 0.46 (7.1%) |
| Random with the same trade count | 0.36 | 0.47 |

For the macro agent, none of the bootstrap intervals for the Sharpe difference excluded zero. Against buy and hold the interval was -0.07 (from -0.25 to 0.11) over all periods and -0.04 (from -0.30 to 0.26) for 2014 to 2024. The price only agent was worse than buy and hold and beat only 30% of the random runs over all periods.

Other findings from round 1:

* Averaging the seeds helps. The ensemble Sharpe ratio was higher than the average single seed in every variant.
* The agent did not protect us in crashes. In 2008 to 2009 the macro agent lost 11%, the price only agent 6%, and the cross asset agent 23%, against a 19.5% loss for buy and hold. The tuned moving average gained 23%.
* The macro agent had lower volatility than buy and hold (15.0% against 17.1%) but also a lower return.
* At the macro agent's 0.02% cost the ensemble Sharpe was 0.54 over all periods against 0.55 for buy and hold. At 0.05% it was 0.47, and at 0.1% it was 0.36. We did not retrain at each cost.

At the time we wrote that macro data might help. We withdrew this claim after step 9.


## 8. Round 2: More Data

Before running round 2 we wrote down how we would read the results, so that we could not bend the rules afterwards. We added four groups on top of the macro data: sector rotation, bonds and credit, volatility indices, and all of them together. Everything else stayed the same as round 1. Our rules were:

1. The main question was whether any new variant beats the macro variant and buy and hold on 2014 to 2024 with a confidence interval that excludes zero.
2. With seven variants being compared, a single best result would count as a lead and not a finding.
3. We would not drop or add features after seeing the results.

**Step 8. Results.** We trained 360 more networks in about 47 minutes. A check we ran before training caught a bug where our first edit had dropped the price columns, and we fixed it. We also confirmed that the macro features were unchanged, so the round 1 models stayed valid.

| Variant | Sharpe, 2008 to 2024 | Sharpe, 2014 to 2024 |
|---|---|---|
| Price only | 0.31 | 0.44 |
| Macro | 0.47 | 0.68 |
| Cross asset | 0.27 | 0.43 |
| Macro plus sector | 0.32 | 0.49 |
| Macro plus bonds | 0.31 | 0.35 |
| Macro plus volatility indices | 0.25 | 0.41 |
| Everything | 0.30 | 0.36 |

No new data source helped. Every version with more features scored below the macro variant, and the combined version was among the worst. Under our rules nothing qualified. Three variants (bonds, volatility indices, and everything) were significantly worse than buy and hold for 2014 to 2024.

The macro variant now looked like an outlier. The other six all sat in the range of random trading with the same number of trades (average Sharpe 0.36 and 0.47). With seven variants one of them will look best by chance, and more inputs making things worse is what overfitting to noise looks like. We decided to test this.

**Step 9. Repeat with new seeds.** We retrained the macro and price only variants with 10 completely new seeds (180 networks, about 23 minutes).

| Ensemble Sharpe | 2008 to 2024, first seeds | 2008 to 2024, new seeds | 2014 to 2024, first seeds | 2014 to 2024, new seeds |
|---|---|---|---|---|
| Price only | 0.31 | 0.37 | 0.44 | 0.42 |
| Macro | 0.47 | 0.33 | 0.68 | 0.43 |
| Buy and hold | 0.55 | 0.55 | 0.72 | 0.72 |

The macro result did not repeat. With new seeds it fell to the same level as the price only agent, and it was significantly worse than buy and hold in both ranges. A single 10 seed ensemble moves by about 0.2 in Sharpe ratio between repeats, which is as large as every difference we tested in rounds 1 and 2.

Individual networks are steadier. Across all 20 seeds the macro networks averaged 0.35 against 0.25 for price only networks over all periods (p = 0.003), and 0.40 against 0.34 for 2014 to 2024 (p = 0.12). This test treats the seeds as independent although they share the same market data, and macro was picked as the best of seven. We treat it as weak evidence at most.

**We withdraw the statement from round 1 that macro features help.** What the data supports is that macro data may give a small gain per network (about 0.1 in Sharpe ratio), and that the near tie with buy and hold in round 1 came from a lucky ensemble.


## 9. Conclusions

1. After fixing the problems in experiment 1, no DQN variant reliably beat buy and hold, the constant exposure benchmark, or random trading with the same trade count on SPY from 2008 to 2024.
2. More data did not help. Six extra feature groups did no better than the macro or price only agents, and the best looking result did not repeat.
3. The agent did not protect us in the 2008 to 2009 and 2022 to 2023 drops, where a simple tuned trend rule did better.
4. The changes to the method were what mattered. Random window training, ensembling, honest benchmarks, walk forward periods, and bootstrap intervals changed our conclusion from "this looks promising" to "we found no reliable edge".


## 10. Limitations and Next Steps

We only tested SPY in this experiment. The hyperparameter search was small, the networks were trained for only 120 short episodes, and the periods are few enough that differences smaller than about 0.25 in Sharpe ratio cannot be resolved. We also compared seven variants and reported the best, which is why we repeated the best one.

Next steps we would try:

* A multi stock test with the stocks chosen before looking at results.
* Predicting volatility instead of price direction, which is easier to forecast and still useful for sizing positions.
* A recurrent or windowed network.
* A direct test of whether the agent buys near lows and sells near highs.

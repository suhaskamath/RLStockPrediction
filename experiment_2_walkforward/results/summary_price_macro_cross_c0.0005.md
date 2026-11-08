
# Variant: price  (cost 5 bps/side)

## ALL folds (2008-2024)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 125.8%         | 4.9%         | 15.9%       | 0.31       | -34.3%         | 61.3%          |
| DQN-single | 70.8% ±42.6%   | 3.0% ±1.6%   | 14.9% ±0.8% | 0.20 ±0.10 | -39.2% ±7.1%   | 56.4% ±5.0%    |
| Buy&Hold   | 448.1%         | 10.6%        | 20.0%       | 0.55       | -51.9%         | 100.0%         |
| VolMatch   | 224.9%         | 7.2%         | 12.3%       | 0.53       | -29.1%         | 61.3%          |
| MA-tuned   | 203.5%         | 6.8%         | 12.7%       | 0.49       | -33.7%         | 66.3%          |
| Random-TM  | 147.4% ±80.8%  | 5.2% ±1.8%   | 13.0% ±0.7% | 0.37 ±0.13 | -30.9% ±7.8%   | 50.6% ±3.0%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.23 | [-0.52, +0.02] | 0.04 | -6.0% | [-11.3%, -1.2%] |
| VolMatch | -0.22 | [-0.50, +0.04] | 0.05 | -1.6% | [-5.5%, +2.1%] |
| MA-tuned | -0.18 | [-0.61, +0.22] | 0.21 | -1.3% | [-6.5%, +4.2%] |

Random-TM Sharpe over 20 seeds: mean +0.37, 95th pct +0.63; DQN-ens Sharpe +0.31 -> beats 30% of random runs.

## CLEAN folds (2014-2024, after tuning window)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 114.3%         | 7.2%         | 14.4%       | 0.44       | -26.2%         | 72.6%          |
| DQN-single | 70.4% ±22.2%   | 4.9% ±1.3%   | 13.1% ±1.1% | 0.32 ±0.10 | -28.2% ±7.4%   | 62.4% ±8.6%    |
| Buy&Hold   | 290.1%         | 13.2%        | 17.1%       | 0.72       | -33.7%         | 100.0%         |
| VolMatch   | 188.4%         | 10.1%        | 12.8%       | 0.70       | -25.4%         | 72.6%          |
| MA-tuned   | 111.7%         | 7.1%         | 13.3%       | 0.46       | -33.7%         | 72.1%          |
| Random-TM  | 92.3% ±46.2%   | 5.9% ±2.1%   | 11.3% ±0.8% | 0.42 ±0.17 | -24.6% ±4.2%   | 50.5% ±3.3%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.27 | [-0.58, +0.02] | 0.03 | -5.9% | [-10.5%, -1.0%] |
| VolMatch | -0.25 | [-0.54, +0.03] | 0.04 | -2.5% | [-6.0%, +1.3%] |
| MA-tuned | -0.02 | [-0.49, +0.35] | 0.46 | +0.3% | [-4.9%, +5.9%] |

Random-TM Sharpe over 20 seeds: mean +0.42, 95th pct +0.68; DQN-ens Sharpe +0.44 -> beats 70% of random runs.

## Per-fold total return (single/random = mean over seeds)

|           | DQN-ens   | DQN-single   | Buy&Hold   | VolMatch   | MA-tuned   | Random-TM   |
|:----------|:----------|:-------------|:-----------|:-----------|:-----------|:------------|
| 2008-2009 | -6.1%     | -13.1%       | -19.5%     | -7.2%      | 23.3%      | 3.0%        |
| 2010-2011 | 15.5%     | 10.8%        | 15.3%      | 7.1%       | -4.5%      | 5.7%        |
| 2012-2013 | -2.8%     | 3.8%         | 51.3%      | 13.3%      | 21.8%      | 18.7%       |
| 2014-2015 | -2.4%     | 0.6%         | 15.7%      | 8.2%       | -4.8%      | 4.9%        |
| 2016-2017 | 14.1%     | 12.0%        | 37.9%      | 28.3%      | 24.0%      | 15.5%       |
| 2018-2019 | 17.9%     | 15.1%        | 24.2%      | 17.6%      | 12.8%      | 8.4%        |
| 2020-2021 | 42.3%     | 23.4%        | 52.6%      | 37.1%      | 19.4%      | 23.9%       |
| 2022-2023 | -11.1%    | -7.6%        | 2.3%       | 3.3%       | 17.5%      | 3.0%        |
| 2024-2024 | 29.1%     | 16.8%        | 26.1%      | 24.6%      | 13.3%      | 14.7%       |


# Variant: macro  (cost 5 bps/side)

## ALL folds (2008-2024)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 287.2%         | 8.3%         | 17.9%       | 0.47       | -45.5%         | 74.5%          |
| DQN-single | 160.7% ±66.9%  | 5.6% ±1.7%   | 16.3% ±0.7% | 0.35 ±0.10 | -37.4% ±5.2%   | 66.1% ±4.6%    |
| Buy&Hold   | 448.1%         | 10.6%        | 20.0%       | 0.55       | -51.9%         | 100.0%         |
| VolMatch   | 317.5%         | 8.8%         | 15.4%       | 0.55       | -40.3%         | 74.5%          |
| MA-tuned   | 203.5%         | 6.8%         | 12.7%       | 0.49       | -33.7%         | 66.3%          |
| Random-TM  | 135.3% ±51.4%  | 5.0% ±1.4%   | 13.0% ±1.1% | 0.36 ±0.11 | -33.4% ±7.7%   | 50.4% ±3.1%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.07 | [-0.25, +0.11] | 0.19 | -2.4% | [-6.1%, +0.8%] |
| VolMatch | -0.08 | [-0.25, +0.09] | 0.16 | -0.0% | [-3.1%, +2.8%] |
| MA-tuned | -0.02 | [-0.44, +0.35] | 0.47 | +2.2% | [-3.5%, +8.0%] |

Random-TM Sharpe over 20 seeds: mean +0.36, 95th pct +0.51; DQN-ens Sharpe +0.47 -> beats 85% of random runs.

## CLEAN folds (2014-2024, after tuning window)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 220.0%         | 11.2%        | 15.0%       | 0.68       | -34.0%         | 72.9%          |
| DQN-single | 99.8% ±37.1%   | 6.4% ±1.8%   | 14.0% ±0.6% | 0.40 ±0.12 | -31.8% ±2.1%   | 64.9% ±5.7%    |
| Buy&Hold   | 290.1%         | 13.2%        | 17.1%       | 0.72       | -33.7%         | 100.0%         |
| VolMatch   | 204.1%         | 10.7%        | 13.6%       | 0.70       | -31.9%         | 72.9%          |
| MA-tuned   | 111.7%         | 7.1%         | 13.3%       | 0.46       | -33.7%         | 72.1%          |
| Random-TM  | 96.5% ±43.2%   | 6.1% ±2.2%   | 11.0% ±1.4% | 0.47 ±0.21 | -23.9% ±7.7%   | 50.5% ±3.4%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.04 | [-0.30, +0.26] | 0.39 | -2.1% | [-6.3%, +2.3%] |
| VolMatch | -0.02 | [-0.27, +0.23] | 0.42 | +0.7% | [-2.7%, +4.3%] |
| MA-tuned | +0.21 | [-0.19, +0.61] | 0.85 | +4.0% | [-1.2%, +10.1%] |

Random-TM Sharpe over 20 seeds: mean +0.47, 95th pct +0.71; DQN-ens Sharpe +0.68 -> beats 85% of random runs.

## Per-fold total return (single/random = mean over seeds)

|           | DQN-ens   | DQN-single   | Buy&Hold   | VolMatch   | MA-tuned   | Random-TM   |
|:----------|:----------|:-------------|:-----------|:-----------|:-----------|:------------|
| 2008-2009 | -11.4%    | -2.2%        | -19.5%     | -12.5%     | 23.3%      | -4.7%       |
| 2010-2011 | 1.2%      | 1.2%         | 15.3%      | 12.8%      | -4.5%      | 7.8%        |
| 2012-2013 | 34.9%     | 32.0%        | 51.3%      | 39.1%      | 21.8%      | 18.6%       |
| 2014-2015 | 16.6%     | 4.6%         | 15.7%      | 13.2%      | -4.8%      | 5.3%        |
| 2016-2017 | 22.5%     | 18.1%        | 37.9%      | 25.7%      | 24.0%      | 16.5%       |
| 2018-2019 | 15.0%     | 11.2%        | 24.2%      | 15.6%      | 12.8%      | 9.7%        |
| 2020-2021 | 53.7%     | 29.4%        | 52.6%      | 49.1%      | 19.4%      | 25.2%       |
| 2022-2023 | 16.7%     | -1.2%        | 2.3%       | 4.8%       | 17.5%      | 1.7%        |
| 2024-2024 | 8.5%      | 13.9%        | 26.1%      | 18.4%      | 13.3%      | 15.2%       |


# Variant: cross  (cost 5 bps/side)

## ALL folds (2008-2024)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 106.0%         | 4.4%         | 17.1%       | 0.27       | -50.7%         | 69.4%          |
| DQN-single | 88.1% ±114.5%  | 3.1% ±2.8%   | 16.0% ±0.9% | 0.20 ±0.17 | -45.1% ±6.7%   | 63.2% ±4.2%    |
| Buy&Hold   | 448.1%         | 10.6%        | 20.0%       | 0.55       | -51.9%         | 100.0%         |
| VolMatch   | 295.9%         | 8.5%         | 14.1%       | 0.57       | -38.4%         | 69.4%          |
| MA-tuned   | 203.5%         | 6.8%         | 12.7%       | 0.49       | -33.7%         | 66.3%          |
| Random-TM  | 126.4% ±55.8%  | 4.8% ±1.5%   | 13.2% ±1.0% | 0.34 ±0.11 | -35.6% ±7.8%   | 50.7% ±3.4%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.28 | [-0.48, -0.08] | 0.00 | -6.3% | [-10.5%, -2.5%] |
| VolMatch | -0.30 | [-0.49, -0.11] | 0.00 | -3.4% | [-6.7%, -0.0%] |
| MA-tuned | -0.22 | [-0.65, +0.18] | 0.14 | -1.6% | [-7.3%, +4.0%] |

Random-TM Sharpe over 20 seeds: mean +0.34, 95th pct +0.48; DQN-ens Sharpe +0.27 -> beats 25% of random runs.

## CLEAN folds (2014-2024, after tuning window)
Stitched out-of-sample, mean ± std across seeds (single-run rows have no ±).

| strategy   | total_return   | ann_return   | ann_vol     | sharpe     | max_drawdown   | avg_exposure   |
|:-----------|:---------------|:-------------|:------------|:-----------|:---------------|:---------------|
| DQN-ens    | 106.9%         | 6.9%         | 14.2%       | 0.43       | -35.4%         | 67.0%          |
| DQN-single | 74.8% ±68.0%   | 4.8% ±3.2%   | 13.4% ±0.9% | 0.30 ±0.24 | -29.7% ±5.5%   | 61.3% ±5.5%    |
| Buy&Hold   | 290.1%         | 13.2%        | 17.1%       | 0.72       | -33.7%         | 100.0%         |
| VolMatch   | 187.2%         | 10.1%        | 12.0%       | 0.73       | -28.1%         | 67.0%          |
| MA-tuned   | 111.7%         | 7.1%         | 13.3%       | 0.46       | -33.7%         | 72.1%          |
| Random-TM  | 97.3% ±43.7%   | 6.2% ±2.2%   | 11.3% ±1.2% | 0.45 ±0.21 | -25.3% ±6.4%   | 50.7% ±3.6%    |

Block-bootstrap (21-day blocks, 4000 resamples): DQN-ens minus baseline

| vs | Sharpe diff | 95% CI | P(diff>0) | Ann. return diff | 95% CI |
|---|---|---|---|---|---|
| Buy&Hold | -0.29 | [-0.59, +0.02] | 0.03 | -6.3% | [-11.4%, -1.6%] |
| VolMatch | -0.30 | [-0.59, -0.04] | 0.01 | -2.7% | [-6.6%, +0.9%] |
| MA-tuned | -0.03 | [-0.46, +0.32] | 0.41 | -0.1% | [-5.2%, +4.8%] |

Random-TM Sharpe over 20 seeds: mean +0.45, 95th pct +0.68; DQN-ens Sharpe +0.43 -> beats 40% of random runs.

## Per-fold total return (single/random = mean over seeds)

|           | DQN-ens   | DQN-single   | Buy&Hold   | VolMatch   | MA-tuned   | Random-TM   |
|:----------|:----------|:-------------|:-----------|:-----------|:-----------|:------------|
| 2008-2009 | -22.8%    | -17.4%       | -19.5%     | -11.5%     | 23.3%      | -8.6%       |
| 2010-2011 | -1.7%     | 1.8%         | 15.3%      | 11.7%      | -4.5%      | 7.2%        |
| 2012-2013 | 31.2%     | 22.1%        | 51.3%      | 39.4%      | 21.8%      | 18.2%       |
| 2014-2015 | 4.4%      | 3.7%         | 15.7%      | 12.3%      | -4.8%      | 4.0%        |
| 2016-2017 | 16.3%     | 13.3%        | 37.9%      | 27.1%      | 24.0%      | 20.0%       |
| 2018-2019 | 19.4%     | 11.0%        | 24.2%      | 16.9%      | 12.8%      | 9.4%        |
| 2020-2021 | 19.6%     | 19.4%        | 52.6%      | 42.0%      | 19.4%      | 18.7%       |
| 2022-2023 | -1.2%     | -3.7%        | 2.3%       | 5.6%       | 17.5%      | 5.2%        |
| 2024-2024 | 20.7%     | 13.6%        | 26.1%      | 14.7%      | 13.3%      | 15.3%       |

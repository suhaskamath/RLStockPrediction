# v2 Improvement Log - what was done, and why

Living document. Every action on the v2 rebuild is recorded here with its reason. New entries are appended at the bottom
under "Log (chronological)"; the status table is kept current.

## Why v2 exists
A critical review of v1 (`/home/jovyan/Project_4`, summarised in `PROJECT_STEPS_FOLLOWED.md`) found that its conclusions were
weaker than they looked:

| # | v1 problem (evidence) | v2 fix |
|---|---|---|
| 1 | Agent memorised one price path: every episode replayed the same 3,213 days; training log-return was +3.4..+4.5 (30-90x) vs 2.9x for buy-and-hold | Each episode is a **random window** (504 days) of training history with a random starting exposure |
| 2 | State contained log portfolio value: acts as a clock inside an episode, and was out of range at test time | Removed. State = market features + current exposure only |
| 3 | Checkpoint = best of 40 noisy validation Sharpes on 502 days; every seed scored lower on test than on validation | Validation measured every 10 episodes, **2-point-smoothed**; plus **ensemble of 10 seeds** (average Q-values) so one lucky seed can't decide the result |
| 4 | 5 seeds, one test path, 2020-24 bull market only | **Walk-forward**: 9 expanding-window folds covering 2008-2024 (incl. 2008 crash, 2022 bear), 10 seeds, block-bootstrap confidence intervals |
| 5 | Weak / arbitrary baselines (random with ~416 trades, untuned MA, no cash interest) | Added **VolMatch** (same average exposure as the DQN), **MA-tuned** (parameters chosen on validation only), **Random-TM** (same trade frequency as the DQN, 20 seeds); cash earns T-bill rate |
| 6 | Trades executed at the same close used to compute the signal; flat 10 bps cost | Decision at close t, **execution at open t+1**; 5 bps per side (fee + spread); cash earns 3-month T-bill rate |
| 7 | Hyperparameters never tuned; all-in/all-out only | Small validation-only sweep; actions are now target exposure **0% / 50% / 100%** |
| 8 | Price-only features | Added feature variants: `price`, `macro` (+VIX, yields, curve), `cross` (+dollar, gold, oil) to test whether other data helps |

Not changed in this round (listed so it is not forgotten): stock-universe fix (pre-specified random S&P 500 sample),
direct peak/trough precision-recall test, recurrent/windowed networks, git/version pinning.

## Folder layout
- `/home/jovyan/Project_4` - original v1 project, **left untouched**.
- `/home/jovyan/Project_4_v1_backup` - read-only (`chmod a-w`) full copy of v1 taken before any v2 work (7.3 MB).
- `/home/jovyan/Project_4_v2` - all new work: `src/`, `data/`, `models/`, `results/`, this log.

## Environment (v2 session)
Same JupyterHub container (Linux, 8 cores), Python 3.13, PyTorch 2.11.0+cu128 (**CPU only**; 1 thread per worker),
gymnasium 1.2.2, pandas 3.0.5, scipy 1.18.0. `yfinance` 1.8.0 had to be **re-installed** (`pip install --user yfinance`) because the
container no longer had it (v1 only worked from cached CSVs). Training runs 7 worker processes in parallel.

## Data
Downloaded with yfinance, 1998-01-01 to 2024-12-31, cached as CSV in `data/`:
SPY (target); `^VIX`; `^TNX` (10y yield); `^IRX` (3m T-bill, used as the cash rate); `DX-Y.NYB` (dollar index); `GC=F` (gold); `CL=F` (oil).
(`^TYX` was tested but is not used.) Gold/oil start in 2000, so earlier values are filled with 0 (neutral). Oil futures went negative in
April 2020, so oil returns are computed as simple % change and clipped to +-5 scaled units rather than using logs.
All features use only data up to the close of day t and are scaled with fixed constants (nothing is fitted on val/test).

## Walk-forward folds (expanding window, from 1998)
Test block = 2 years; validation = the 2 years before it; training = everything before that.
Folds: 2008-09, 2010-11, 2012-13, 2014-15, 2016-17, 2018-19, 2020-21, 2022-23, 2024. (e.g. fold 0: train 1998-2005, val 2006-07, test 2008-09.)

## Files in `src/`
| File | Role |
|---|---|
| `data.py` | download + features (3 variants) + fold splits |
| `env.py` | trading env: next-open execution, cost per traded notional, cash interest, target-exposure actions |
| `agent.py` | Double-DQN (AdamW, weight decay) + `Ensemble` (average Q across seeds) |
| `metrics.py` | return, vol, Sharpe on excess returns, max drawdown, exposure |
| `train.py` / `run_train.py` | random-window training with smoothed validation selection; parallel resumable driver |
| `tune.py` | validation-only hyper-parameter sweep |
| `evaluate.py` | walk-forward test evaluation of DQN + all baselines |
| `aggregate.py` | stitched out-of-sample metrics, per-fold table, block-bootstrap tests |

## Status
| Step | Status |
|---|---|
| Backup of v1 + new folder | done |
| Code rewrite (data/env/agent/train/eval) | done |
| Correctness checks | done |
| Hyper-parameter sweep (validation only) | done |
| Main training: 3 variants x 9 folds x 10 seeds = 270 nets | done (see final status at the bottom) |
| Evaluation + aggregation + bootstrap | done |
| Other-data comparison (price vs macro vs cross) | done |
| Final report with results | done (entries 7-16; see 15-16 for the replication that overturned entry 9's macro claim) |

---

## Log (chronological)

### 1. Backup and new folder
`cp -a Project_4 Project_4_v1_backup`, made read-only; created `Project_4_v2/`. *Why:* you asked to keep the current implementation and
put changes elsewhere. The original folder was also left unmodified, so there are two copies of v1.

### 2. Reinstalled yfinance and checked extra data
First data test failed (`No module named 'yfinance'`; also `python3 -I` hides user-site packages). Installed with `pip install --user yfinance`
and confirmed all 8 symbols return 1998-2024 data with no gaps (gold/oil from 2000). *Why:* the new-data request needs live downloads.

### 3. Rewrote the code (see table above)
Removed the copied v1 sources in `v2/src` and wrote new ones, so v2 does not inherit v1 logic by accident.

### 4. Correctness checks (before spending CPU time)
- Buy-and-hold inside the env on 2020-21 returned **52.66%**, identical to the raw price calculation Open[day 2] -> Close[last] (**52.66%**). This confirms timing and accounting.
- Holding only cash for 2020-21 returned **0.37%** (T-bill rate near zero then) with 0 trades, as expected.
- Feature ranges looked sane (clipped to +-5 scaled units).
- Speed: about **627 env-steps/s** per core with a learning step every step. *Decision:* learn every 2nd step (`train_freq=2`) to keep the full run to about 40 minutes.

### 5. Hyper-parameter sweep (validation only)
6 configs x folds 0-2 x 3 seeds = 54 runs (about 7.5 min). Score = mean of the last 4 validation Sharpe ratios (not the best one, to avoid selection bias).

| Config | Mean val Sharpe |
|---|---|
| **lr = 3e-4** | **+0.226** |
| gamma = 0.99 | +0.038 |
| gamma = 0.90 | -0.067 |
| weight decay 1e-3 | -0.122 |
| base (lr 1e-3, gamma 0.95, hidden 64, wd 1e-4) | -0.201 |
| hidden = 128 | -0.229 |

*Decision:* use **lr = 3e-4**, everything else at base. The margin is clear (+0.43 over base), but note all scores are low: even tuned, validation Sharpe is near zero.
*Caveat to keep in mind:* the sweep used validation years 2006-2011, which are also test years for folds 0-1 (2008-09, 2010-11). So folds 0-2 results are slightly
optimistic; the headline numbers therefore also report **"clean" folds 3-8 (2014-2024)**, which no tuning ever saw. Sweep models are in `models/tune/`.

### 6. Launched main training
`python3 src/run_train.py --variants price macro cross --folds 0-8 --seeds 10 --jobs 7` -> 270 networks, 120 episodes each, resumable, log in `results/train_main.log`.
*Why 10 seeds / 3 variants:* 10 seeds supports the ensemble and a seed-spread estimate; the three variants answer your "can other data help" question with a controlled comparison
(identical training, only the feature set differs).

### 7. Training finished
270/270 networks in 2,104 s (about 35 min). Checked: 270 `.pt` files present.

### 8. Evaluation and aggregation (problems met, and fixes)
- `evaluate.py` run for each variant (10 seeds, 5 bps cost) -> `results/<variant>_SPY_c0.0005_daily.csv` and `_folds.csv`.
- `aggregate.py` bug 1: single-run strategies (Buy&Hold etc.) have a NaN std, and my `dropna` removed them -> KeyError. Fixed by aligning the std table to the mean table.
- `aggregate.py` bug 2: `tabulate` missing for `to_markdown` -> `pip install --user tabulate`.
- Cost-sensitivity runs first failed (exit 1) because I launched them from `src/` while the scripts use relative `models/` and `results/` paths. Re-ran from the project root. (Lesson: always run from `Project_4_v2/`.)
- Full output: `results/summary_price_macro_cross_c0.0005.md`.

### 9. Results (SPY, walk-forward, 5 bps/side, cash earns T-bill rate)
Stitched out-of-sample Sharpe (ann. return). "Clean" = folds 2014-2024, never seen by the sweep.

| Strategy | ALL 2008-24 | CLEAN 2014-24 |
|---|---|---|
| **DQN-ens, price only** | 0.31 (4.9%) | 0.44 (7.2%) |
| **DQN-ens, macro** | **0.47 (8.3%)** | **0.68 (11.2%)** |
| DQN-ens, cross | 0.27 (4.4%) | 0.43 (6.9%) |
| Buy & Hold | 0.55 (10.6%) | 0.72 (13.2%) |
| VolMatch (same avg exposure as DQN) | 0.55 (8.8%) | 0.70 (10.7%) |
| MA-tuned | 0.49 (6.8%) | 0.46 (7.1%) |
| Random, same trade frequency (20 seeds) | 0.36 | 0.47 |

Block-bootstrap, DQN-ens(macro) minus baseline (Sharpe diff, 95% CI): vs Buy&Hold -0.07 [-0.25, +0.11] (all) / -0.04 [-0.30, +0.26] (clean);
vs VolMatch -0.08 [-0.25, +0.09] / -0.02 [-0.27, +0.23]; vs MA-tuned -0.02 [-0.44, +0.35] / +0.21 [-0.19, +0.61]. **No interval excludes zero**, i.e.
the macro DQN is statistically indistinguishable from buy-and-hold, not better.
Price-only DQN is worse than Buy&Hold / VolMatch (P(diff>0) = 0.03-0.05) and beats only 30% (all) / 70% (clean) of random-same-turnover runs.

Findings:
1. **Ensembling helps**: ensemble Sharpe (0.31 / 0.47 / 0.27) is above the mean single-seed Sharpe (0.20 / 0.35 / 0.20) for every variant, and the single-seed spread is large
   (cross variant total return 88% +-115%).
2. **Macro features help; cross-asset features do not.** macro-ens minus price-ens Sharpe: +0.16 [-0.12, +0.46] (all), +0.23 [-0.12, +0.65] (clean); macro minus cross: +0.20 [-0.00, +0.42] / +0.25 [-0.05, +0.58].
   Single-seed means (10 seeds): macro 0.35 +-0.10 vs price 0.20 +-0.10 (all), 0.40 vs 0.32 (clean). Direction is consistent, but the confidence intervals include zero, and I picked macro as best of 3 variants, so this is **suggestive, not proven**.
3. **The DQN does not protect in crashes.** 2008-09: macro DQN -11%, price DQN -6%, cross DQN -23% (Buy&Hold -19.5%); MA-tuned made +23%. 2022-23: price DQN -11%, MA-tuned +17.5%.
4. **Lower risk, lower return.** Macro DQN: volatility 15.0% vs 17.1%, max drawdown similar; return 11.2% vs 13.2% (clean).

### 10. Cost sensitivity (macro variant; models trained at 5 bps, only the evaluation cost changed)
| Eval cost/side | ALL: DQN-ens Sharpe vs Buy&Hold | CLEAN: DQN-ens Sharpe vs Buy&Hold |
|---|---|---|
| 2 bps | 0.54 vs 0.55 | 0.75 vs 0.72 |
| 5 bps | 0.47 vs 0.55 | 0.68 vs 0.72 |
| 10 bps | 0.36 vs 0.54 | 0.55 vs 0.72 |
The edge disappears as costs rise, so the result depends on cheap trading (SPY realistically costs 1-3 bps). Not retrained at each cost, so the 10 bps row is pessimistic.

### 11. Honest limits of v2
- Only SPY was tested in v2 (the 4-stock extension was not redone). Stock-universe fix is still open.
- Hyper-parameters were tuned on a small grid (6 configs, 3 folds, 3 seeds); folds 0-2 are slightly optimistic (see step 5).
- Three variants were compared and the best reported; the macro > price gap is not significant at 95%.
- 10 seeds and 9 folds give wide intervals (about +-0.25 Sharpe). Differences smaller than that cannot be resolved.
- Learning step every 2 env steps, 120 episodes per net: longer training was not explored.
- The peak/trough claim and recurrent networks are still untested. No git repo; `requirements.txt` still unpinned in v2 (not yet written).

## Status (final for this round)
| Step | Status |
|---|---|
| Backup of v1 + new folder | done |
| Code rewrite, correctness checks | done |
| Hyper-parameter sweep (validation only) | done |
| Main training (270 nets) | done |
| Evaluation, aggregation, bootstrap | done |
| Other-data comparison (price vs macro vs cross) | done: macro helps (not significant), cross does not |
| Cost sensitivity | done |
| Open items | stock-universe test, more data sources (see report), peak/trough test, version pinning |

---
## Round 2: more data sources (pre-registered BEFORE any results)

### 12. Plan and why
Round 1 found macro features (VIX + yields) helped (suggestive, not significant) and dollar/gold/oil did not. Round 2 tests four new groups, always **on top of macro**:

| Variant | Adds | Intuition |
|---|---|---|
| `macro_sector` | XLY-XLP (cyclical vs defensive), XLU, XLF, IWM relative 20d returns vs SPY | risk appetite / market breadth proxies |
| `macro_bonds` | TLT 20d return, LQD-IEF 20d relative return (credit vs Treasuries), copper 20d return | credit stress and growth signals |
| `macro_vol` | CBOE SKEW level, VXN/VIX ratio, MOVE (bond volatility) level | tail-risk and cross-market volatility |
| `all` | macro + all three groups | combined |

Excluded on purpose: HYG, ^VVIX, ^VIX3M (start 2006-07, after fold 0's training window, so fold 0 would be tested on inputs never seen in training).
Series that start after 1998 (TLT/LQD/IEF 2002, IWM 2000, VXN 2001, MOVE 2002) are filled with 0 (neutral) before they exist, as gold/oil were.
Everything else is **identical to round 1**: same folds, 10 seeds, 120 episodes, lr 3e-4, 5 bps cost, same evaluation and bootstrap.

**Pre-registered rules for reading the results** (written now so they cannot be bent later):
1. Primary question: does any new variant beat `macro` and Buy&Hold on the stitched CLEAN folds (2014-24) with a bootstrap 95% CI excluding zero?
2. Seven variants are now compared. With 4 new candidates, a P(diff>0) below about 0.99 is **not** convincing evidence; a single best-of-N winner will be treated as a lead, not a finding.
3. Nothing is tuned on these results. No feature will be dropped or added after seeing them within this round.

### 13. Implementation and checks
- Added `SECTOR_F`, `BOND_F`, `VOL_F` and the four variants in `data.py` (download list extended; 12 new symbols cached in `data/`).
- Bug found by my own check: first edit dropped the price columns from the concatenated frame (`KeyError: ret1...`). Fixed (`PRICE_F + MACRO_F + EXTRA_F`).
- Checks: 0 NaN / 0 inf over all 23 features; value ranges sane; **round-1 `macro` features are bit-identical** after the change, so round-1 models and results remain valid.
- Known weakness: in fold 0 (train 1998-2005) about **55% of training rows have zero-filled** bond/MOVE features (series start 2002). `vxn_vix` has a typical value near +2 after scaling, so "0" is not perfectly neutral for it before 2001. These affect early folds mostly; the CLEAN folds (2014-24) are unaffected.
- Launched: `python3 src/run_train.py --variants macro_sector macro_bonds macro_vol all --folds 0-8 --seeds 10 --jobs 7` -> 360 networks, log `results/train_round2.log` (expected about 45-50 min).

### 14. Round-2 results (360 nets trained in 2,805 s; evaluated; `results/round2_comparison.md`)
Stitched out-of-sample ensemble Sharpe (Buy&Hold: 0.55 ALL / 0.72 CLEAN):

| Variant | ALL | CLEAN | vs macro (CLEAN, P>0) |
|---|---|---|---|
| price | 0.31 | 0.44 | 0.10 |
| **macro** | **0.47** | **0.68** | reference |
| cross | 0.27 | 0.43 | 0.05 |
| macro_sector | 0.32 | 0.49 | 0.10 |
| macro_bonds | 0.31 | 0.35 | 0.01 |
| macro_vol | 0.25 | 0.41 | 0.04 |
| all | 0.30 | 0.36 | 0.04 |

**No new data source helped.** Every variant that adds features to `macro` is lower than `macro`, and `all` (23 features) is among the worst. Under the pre-registered rules (step 12) nothing qualifies:
no new variant beats `macro` or Buy&Hold. Several are significantly *worse* than Buy&Hold (macro_bonds, all, macro_vol: CI excludes zero on CLEAN).

**Reading with suspicion.** `macro` is an outlier: the other six cluster at 0.25-0.36 (ALL) and 0.35-0.49 (CLEAN), which is the same range as random same-turnover trading (mean 0.36 / 0.47). Out of seven variants, one will look
best by chance. Round 1's "macro helps" may therefore be selection luck, not signal. Adding more information that makes results *worse* is also what overfitting to noise looks like (more inputs, same amount of data).
**Test of this:** retrain `macro` and `price` with 10 completely new seeds (10-19, tag `rep_`) and check whether macro > price replicates. Launched; log `results/train_replication.log`.
(Code change for this: `--seed-start` option in `run_train.py`.)

### 15. Replication with 10 new seeds (macro and price; 180 nets, 1,407 s)
Seeds 10-19, tag `rep_`, otherwise identical. Evaluated the same way (`results/rep_*_daily.csv`).

| Ensemble Sharpe | ALL orig | ALL new seeds | CLEAN orig | CLEAN new seeds |
|---|---|---|---|---|
| price | 0.31 | 0.37 | 0.44 | 0.42 |
| macro | **0.47** | **0.33** | **0.68** | **0.43** |
| Buy & Hold | 0.55 | 0.55 | 0.72 | 0.72 |

- **The headline macro result did NOT replicate.** With new seeds the macro ensemble falls from 0.68 to 0.43 (CLEAN) and from 0.47 to 0.33 (ALL). macro(new) minus macro(orig): -0.24 [-0.52, -0.01] on CLEAN.
- New-seed macro vs new-seed price: -0.04 [-0.35, +0.26] (ALL), +0.02 [-0.31, +0.38] (CLEAN): **no difference**.
- Macro new-seed ensemble is *significantly worse* than Buy&Hold on both windows (ALL: -0.22 [-0.42, -0.05]; CLEAN: -0.28 [-0.54, -0.02]).
- A single 10-seed ensemble over 9 folds has a run-to-run Sharpe swing of roughly +-0.2, as large as every difference tested in rounds 1-2.
- Single-seed networks are steadier. Pooled over 20 seeds: macro 0.35 +- 0.10 vs price 0.25 +- 0.10 (ALL; Welch p = 0.003) and 0.40 vs 0.34 (CLEAN; p = 0.12). This is a small gain (about +0.1 Sharpe) that is only clear on ALL folds.
  Caveat: this test treats seeds as independent but they share the same market data, so it measures training noise only, not market luck; and macro was picked as best of 7. Treat it as weak evidence at most.

**Correction to step 9, finding 2:** "macro features help" is **withdrawn as a finding**. What the data supports is: macro may give a small average improvement per network (about +0.1 Sharpe, weak evidence), and the apparent near-parity with Buy&Hold in round 1 was a lucky ensemble.

### 16. Final conclusions of the v2 rebuild
1. After removing v1's flaws, **no DQN variant reliably beats Buy&Hold, a constant-exposure Buy&Hold, or random trading of equal frequency** on SPY 2008-2024 at 5 bps.
2. **More data did not help**: 6 additional feature sets (dollar/gold/oil, sectors, bonds/credit/copper, SKEW/VXN/MOVE, all combined) were no better than macro or price-only; the best round-1 result was luck.
3. The model does not protect in crashes (2008-09, 2022-23) where a simple tuned trend rule did.
4. What worked as method: random-window training, ensembling (reduces variance but not enough), honest baselines, walk-forward folds and bootstrap intervals. These changed the conclusion from "looks promising" to "no demonstrated edge".

## Status (final)
| Step | Status |
|---|---|
| Round 1: rebuild, sweep, 270 nets, evaluation | done |
| Round 2: 4 new data groups, 360 nets, comparison | done: no improvement |
| Replication with new seeds | done: macro result did not replicate |
| Open | multi-stock test with pre-fixed universe; sequence model; predicting something easier than daily direction (e.g. volatility); version pinning |

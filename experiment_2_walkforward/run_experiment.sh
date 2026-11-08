#!/usr/bin/env bash
# Reproduces experiment 2 (run from this folder). Training is CPU-only; ~35 min per 270 networks on 8 cores.
set -e
export OMP_NUM_THREADS=1
mkdir -p models results data

python3 src/data.py SPY                                   # download SPY + macro/sector/bond/vol series (yfinance)

python3 src/tune.py                                       # validation-only sweep (folds 0-2, 3 seeds)  ~8 min

# Round 1: price / macro / cross-asset  (3 variants x 9 folds x 10 seeds = 270 nets)
python3 src/run_train.py --variants price macro cross --folds 0-8 --seeds 10 --jobs 7
# Round 2: extra data groups on top of macro  (4 variants = 360 nets)
python3 src/run_train.py --variants macro_sector macro_bonds macro_vol all --folds 0-8 --seeds 10 --jobs 7
# Replication with 10 new seeds (180 nets)
python3 src/run_train.py --variants macro price --folds 0-8 --seeds 10 --seed-start 10 --tag rep_ --jobs 7

for v in price macro cross macro_sector macro_bonds macro_vol all; do python3 src/evaluate.py --variant $v --seeds 10; done
for v in price macro; do python3 src/evaluate.py --variant $v --seeds 10 --tag rep_; done
for c in 0.0002 0.001; do python3 src/evaluate.py --variant macro --seeds 10 --cost $c; done   # cost sensitivity

python3 src/aggregate.py --variants price macro cross      # tables + bootstrap tests
python3 src/compare.py                                     # all-variant comparison

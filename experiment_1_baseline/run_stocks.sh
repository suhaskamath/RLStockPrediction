#!/usr/bin/env bash
# Multi-stock extension: same pipeline on AAPL, MSFT, AMZN, TSLA (cost 0.1%).
export OMP_NUM_THREADS=2
for t in AAPL MSFT AMZN TSLA; do
  ( python3 src/data.py $t > results/data_$t.log 2>&1 && \
    python3 src/train.py --ticker $t --seeds 5 --episodes 40 > results/train_$t.log 2>&1 && \
    python3 src/evaluate.py --ticker $t --seeds 5 > results/eval_$t.log 2>&1 ) &
done
wait

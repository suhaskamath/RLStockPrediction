#!/usr/bin/env bash
# Full pipeline: data -> train (3 cost settings in parallel) -> evaluate on held-out test.
set -e
export OMP_NUM_THREADS=2
python3 src/data.py
for c in 0.001 0.0 0.005; do
  python3 src/train.py --seeds 5 --episodes 40 --cost $c > results/train_c$c.log 2>&1 &
done
wait
for c in 0.001 0.0 0.005; do python3 src/evaluate.py --cost $c --seeds 5; done

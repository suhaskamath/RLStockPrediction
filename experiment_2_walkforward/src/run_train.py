"""Parallel, resumable training driver.
usage: python run_train.py --variants price macro cross --folds 0-8 --seeds 10 --jobs 7 [--tag X --out models --set gamma=0.99 ...]
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_over(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = int(v) if v.isdigit() else float(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["price"])
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--folds", default="0-8")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="models")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args()
    lo, _, hi = a.folds.partition("-")
    folds = list(range(int(lo), int(hi or lo) + 1))
    over = parse_over(a.set)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(v, a.ticker, f, s, a.tag, a.out, over) for v in a.variants for f in folds for s in range(a.seed_start, a.seed_start + a.seeds)]
    from train import _job
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            print(f"[{done}/{len(jobs)}] {r['variant']} f{r['fold']} s{r['seed']} last4_val_sharpe={r['last4_mean']:+.2f} "
                  f"best={r['best_smoothed']:+.2f} ({r['secs']:.0f}s) elapsed={time.time()-t0:.0f}s", flush=True)

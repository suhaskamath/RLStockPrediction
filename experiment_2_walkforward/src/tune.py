"""Small hyper-parameter sweep. Scored ONLY on validation windows (mean of the last 4 validation Sharpes),
using the first 3 folds (validation years 2006-2011) and 3 seeds. Test windows are never touched here."""
import os, sys, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {"base": {}, "gamma99": {"gamma": 0.99}, "hid128": {"hidden": 128},
           "wd1e-3": {"wd": 1e-3}, "lr3e-4": {"lr": 3e-4}, "gamma90": {"gamma": 0.90}}
# The sweep was run when the default lr was 1e-3; pin it so the "base" config reproduces exactly.
CONFIGS = {n: {"lr": 1e-3, **o} for n, o in CONFIGS.items()}
if __name__ == "__main__":
    from train import _job
    jobs = [("price", "SPY", f, s, f"tune_{n}_", "models/tune", o) for n, o in CONFIGS.items() for f in (0, 1, 2) for s in range(3)]
    os.makedirs("models/tune", exist_ok=True)
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(max_workers=7, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); n = futs[fu][4][5:-1]
            res.setdefault(n, []).append(r["last4_mean"])
            print(f"[{i}/{len(jobs)}] {n} f{r['fold']} s{r['seed']} {r['last4_mean']:+.2f} {time.time()-t0:.0f}s", flush=True)
    summ = {n: sum(v) / len(v) for n, v in res.items()}
    json.dump(summ, open("results/tune_summary.json", "w"), indent=1)
    for n, v in sorted(summ.items(), key=lambda x: -x[1]): print(f"{n:8s} mean last-4 val Sharpe = {v:+.3f}")

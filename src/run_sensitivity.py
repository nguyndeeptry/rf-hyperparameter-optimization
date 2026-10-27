"""
Sensitivity analysis w.r.t. the search BUDGET (number of trials).
Question: how do solution quality and stability of Random Search / TPE change when the
budget is 30, 60 or 100 trials? (Baseline does not search, so it is not re-run.)

    python src/run_sensitivity.py --dataset breast_cancer --budgets 30 60 100 --seeds 3

Output: results/<dataset>/sensitivity_results.json (saved after every run; use --resume).
"""
import argparse
import json
import time

import config
from search_random import run_random_search
from search_tpe import run_tpe_search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, trials=False, seeds=True)
    parser.set_defaults(seeds=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[30, 60, 100])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    path = config.results_dir() / "sensitivity_results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if (args.resume and path.exists()) else []
    done = {(r["method"], r["budget"], r["seed"]) for r in results}

    t0 = time.time()
    for budget in args.budgets:
        config.N_TRIALS = budget
        for seed in config.SEEDS:
            for name, fn in (("random_search", run_random_search), ("tpe", run_tpe_search)):
                if (name, budget, seed) in done:
                    continue
                print(f"[budget {budget}, seed {seed}] {name} ...", flush=True)
                r = fn(seed)
                r["budget"] = budget
                results.append(r)
                path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()

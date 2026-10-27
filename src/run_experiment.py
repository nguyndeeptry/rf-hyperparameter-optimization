"""
Main experiment: Baseline vs Random Search vs TPE (Optuna), repeated over several seeds.

    python src/run_experiment.py --dataset breast_cancer
    python src/run_experiment.py --dataset wine --seeds 10 --trials 60

Outputs (results/<dataset>/): raw_results.json (every run + full trial history), run_meta.json
Use --resume to continue an interrupted run (finished seeds are skipped).
"""
import argparse
import json
import time

import config
import meta
from baseline import run_baseline
from search_random import run_random_search
from search_tpe import run_tpe_search

METHODS = (("baseline", run_baseline),
           ("random_search", run_random_search),
           ("tpe", run_tpe_search))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser)
    parser.add_argument("--resume", action="store_true",
                        help="skip (method, seed) pairs already stored in raw_results.json")
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    out_dir = config.results_dir()
    raw_path = out_dir / "raw_results.json"
    results = []
    if args.resume and raw_path.exists():
        results = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"Resuming: {len(results)} runs already stored")
    done = {(r["method"], r["seed"]) for r in results}

    print(f"Dataset={config.DATASET_NAME} trials={config.N_TRIALS} seeds={config.SEEDS}")
    t_start = time.time()
    for seed in config.SEEDS:
        for name, fn in METHODS:
            if (name, seed) in done:
                continue
            print(f"[seed {seed}] {name} ...", flush=True)
            results.append(fn(seed))
        raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")  # save after each seed

    meta.write_meta(out_dir / "run_meta.json",
                    {"total_time_sec_this_session": round(time.time() - t_start, 1)})
    print(f"Saved {len(results)} runs -> {raw_path}")


if __name__ == "__main__":
    main()

"""
5-minute demo of the OPTIMISATION part.

    python src/demo.py                        # 15 trials, seed 0, breast_cancer (~1-2 min)
    python src/demo.py --dataset wine --trials 25 --seed 3

What it shows
  1. the problem formulation (decision variables, objective, constraints, budget);
  2. a LIVE run of Baseline vs Random Search vs TPE with the same data, folds and budget,
     printing the best-so-far score while the search progresses;
  3. the best configuration of each method, its search cost, and ONE evaluation on the locked test set;
  4. a saved convergence plot  (results/demo/fig_demo_convergence.png).
Saved full-budget results (results/<dataset>/) are summarised first, if they exist.
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import config
from baseline import run_baseline
from data import load_data
from search_random import run_random_search
from search_tpe import run_tpe_search


def show_formulation():
    s = config.SEARCH_SPACE
    print("=" * 70)
    print("PROBLEM FORMULATION")
    print("  decision variables x = (n_estimators, max_depth, min_samples_split, max_features)")
    print(f"  search space       : n_estimators in {s['n_estimators']}, max_depth in {s['max_depth']},")
    print(f"                       min_samples_split in {s['min_samples_split']}, max_features in {s['max_features']}")
    print(f"  objective f(x)     : maximise mean {config.CV_FOLDS}-fold CV {config.METRIC} on the TRAIN set")
    print(f"  constraints/budget : x in search space; exactly {config.N_TRIALS} evaluations per method")
    print("  protocol           : same split, same CV folds, same budget; test set locked until the end")
    print("=" * 70)


def show_saved_summary():
    path = config.results_dir(create=False) / "summary_table.csv"
    if not path.exists():
        print("(no saved full-budget results found - run src/run_all.py first)\n")
        return
    print(f"SAVED RESULTS ({config.DATASET_NAME}, full budget):")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(f"  {r['method']:14s} CV {float(r['cv_mean_avg']):.4f} +- {float(r['cv_mean_std']):.4f} "
                  f"({r['n_seeds']} seeds)  time {float(r['time_avg_sec']):.1f}s")
    print()


def test_f1(result, X_train, y_train, X_test, y_test, seed):
    keys = ("n_estimators", "max_depth", "min_samples_split", "max_features")
    params = {k: result["best_params"][k] for k in keys if k in result["best_params"]}
    model = RandomForestClassifier(random_state=seed, **params).fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test), average="macro")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, seeds=False)
    parser.set_defaults(trials=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    show_formulation()
    show_saved_summary()

    X_train, X_test, y_train, y_test = load_data()
    print(f"LIVE RUN: dataset={config.DATASET_NAME}, seed={args.seed}, budget={config.N_TRIALS} trials\n")

    res = {}
    t0 = time.time()
    res["baseline"] = run_baseline(args.seed)
    print(f"baseline        default config       CV={res['baseline']['cv_mean']:.4f}  "
          f"({time.time() - t0:.1f}s)")
    for name, fn in (("random_search", run_random_search), ("tpe", run_tpe_search)):
        print(f"\n{name}: searching {config.N_TRIALS} configurations ...", flush=True)
        r = fn(args.seed)
        best = -np.inf
        for t in sorted(r["trial_history"], key=lambda t: t["trial"]):
            if t["score"] > best:
                best = t["score"]
                print(f"  trial {t['trial'] + 1:3d}: new best CV={best:.4f}  params={t['params']}")
        print(f"  -> finished in {r['time_sec']:.1f}s, best CV={r['cv_mean']:.4f}")
        res[name] = r

    print("\nRESULT (same data, same folds, same budget)")
    print(f"{'method':14s} {'best CV':>8s} {'search time':>12s} {'test F1 (1 eval)':>17s}   best configuration")
    for name in ("baseline", "random_search", "tpe"):
        r = res[name]
        f1 = test_f1(r, X_train, y_train, X_test, y_test, args.seed)
        params = {k: v for k, v in r["best_params"].items()
                  if k in ("n_estimators", "max_depth", "min_samples_split", "max_features")}
        print(f"{name:14s} {r['cv_mean']:8.4f} {r['time_sec']:11.1f}s {f1:17.4f}   {params}")

    out = config.results_dir("demo")
    plt.figure(figsize=(7, 4.5))
    for name in ("random_search", "tpe"):
        hist = sorted(res[name]["trial_history"], key=lambda t: t["trial"])
        plt.plot(np.arange(1, len(hist) + 1), np.maximum.accumulate([t["score"] for t in hist]),
                 marker="o", ms=3, label=name)
    plt.axhline(res["baseline"]["cv_mean"], color="gray", linestyle="--", label="baseline")
    plt.xlabel("Trial"); plt.ylabel("Best CV macro-F1 so far")
    plt.title(f"Demo convergence ({config.DATASET_NAME}, seed {args.seed})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "fig_demo_convergence.png", dpi=150); plt.close()
    print(f"\nSaved plot -> {out / 'fig_demo_convergence.png'}")
    print("Note: one seed and a small budget only illustrate the mechanism; "
          "conclusions come from the multi-seed experiment.")


if __name__ == "__main__":
    main()

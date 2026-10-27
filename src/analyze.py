"""
Analyse results/<dataset>/raw_results.json.

Outputs (results/<dataset>/):
  summary_table.csv      mean/std/best/worst CV score and runtime per method
  fig_convergence.png    best-so-far CV score vs trial (mean +- std over seeds)
  fig_boxplot.png        distribution of the final CV score over seeds (stability)
  fig_boxplot_time.png   distribution of the search runtime over seeds
  best_params.csv        best configuration found for every (method, seed)
  trial_stats.csv        mean sampled hyper-parameters over ALL trials (what each method explores)

    python src/analyze.py --dataset breast_cancer
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

METHODS = ("baseline", "random_search", "tpe")


def group_by_method(data):
    groups = {}
    for r in data:
        groups.setdefault(r["method"], []).append(r)
    return {m: groups[m] for m in METHODS if m in groups}


def summary_rows(groups):
    rows = []
    for m, recs in groups.items():
        cv = np.array([r["cv_mean"] for r in recs])
        t = np.array([r["time_sec"] for r in recs])
        rows.append({"method": m, "n_seeds": len(recs),
                     "cv_mean_avg": cv.mean(), "cv_mean_std": cv.std(),
                     "cv_mean_best": cv.max(), "cv_mean_worst": cv.min(),
                     "time_avg_sec": t.mean(), "time_std_sec": t.std()})
    return rows


def best_so_far(history):
    best, curve = -np.inf, []
    for t in sorted(history, key=lambda x: x["trial"]):
        best = max(best, t["score"])
        curve.append(best)
    return np.array(curve)


def plot_convergence(groups, path):
    plt.figure(figsize=(7, 5))
    for m in ("random_search", "tpe"):
        recs = groups.get(m)
        if not recs:
            continue
        curves = np.stack([best_so_far(r["trial_history"]) for r in recs])
        mean, std = curves.mean(0), curves.std(0)
        x = np.arange(1, len(mean) + 1)
        plt.plot(x, mean, label=m)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    if "baseline" in groups:
        plt.axhline(np.mean([r["cv_mean"] for r in groups["baseline"]]),
                    color="gray", linestyle="--", label="baseline")
    plt.xlabel("Trial")
    plt.ylabel("Best CV macro-F1 so far")
    plt.title(f"Convergence on {config.DATASET_NAME} (mean +- std over seeds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_box(groups, key, ylabel, title, path):
    methods = list(groups)
    plt.figure(figsize=(6, 5))
    plt.boxplot([[r[key] for r in groups[m]] for m in methods], tick_labels=methods, showmeans=True)
    plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def write_best_params(groups, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "cv_f1_macro", "n_estimators", "max_depth",
                    "min_samples_split", "max_features"])
        for m in ("random_search", "tpe"):
            for r in sorted(groups.get(m, []), key=lambda r: r["seed"]):
                bp = r["best_params"]
                w.writerow([m, r["seed"], f"{r['cv_mean']:.4f}", bp["n_estimators"],
                            bp["max_depth"], bp["min_samples_split"], bp["max_features"]])


def write_trial_stats(groups, path):
    rows = []
    for m in ("random_search", "tpe"):
        trials = [t["params"] for r in groups.get(m, []) for t in r["trial_history"]]
        if trials:
            rows.append([m, len(trials),
                         np.mean([p["n_estimators"] for p in trials]),
                         np.mean([p["max_depth"] for p in trials]),
                         np.mean([p["min_samples_split"] for p in trials])])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_trials_total", "mean_n_estimators", "mean_max_depth",
                    "mean_min_samples_split"])
        w.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, trials=False, seeds=False)
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    out = config.results_dir()
    groups = group_by_method(json.loads((out / "raw_results.json").read_text(encoding="utf-8")))

    rows = summary_rows(groups)
    header = ["method", "n_seeds", "cv_mean_avg", "cv_mean_std", "cv_mean_best",
              "cv_mean_worst", "time_avg_sec", "time_std_sec"]
    with open(out / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])

    print(f"=== {config.DATASET_NAME}: CV macro-F1 over seeds ===")
    for r in rows:
        print(f"{r['method']:14s} {r['cv_mean_avg']:.4f}+-{r['cv_mean_std']:.4f}  "
              f"best {r['cv_mean_best']:.4f}  worst {r['cv_mean_worst']:.4f}  "
              f"time {r['time_avg_sec']:.1f}+-{r['time_std_sec']:.1f}s")

    plot_convergence(groups, out / "fig_convergence.png")
    plot_box(groups, "cv_mean", "CV macro-F1", f"Stability over seeds ({config.DATASET_NAME})",
             out / "fig_boxplot.png")
    plot_box(groups, "time_sec", "Search runtime (s)", f"Runtime over seeds ({config.DATASET_NAME})",
             out / "fig_boxplot_time.png")
    write_best_params(groups, out / "best_params.csv")
    stats = write_trial_stats(groups, out / "trial_stats.csv")
    for m, n, ne, md, ms in stats:
        print(f"{m:14s} mean n_estimators over all trials = {ne:.1f}")
    print(f"Saved tables and figures -> {out}")


if __name__ == "__main__":
    main()

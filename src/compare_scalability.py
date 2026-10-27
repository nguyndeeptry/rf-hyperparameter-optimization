"""
Compare the two datasets side by side (generalisation / scalability study).
Needs results/breast_cancer/ and results/wine/ (run_experiment + evaluate_test + analyze).

    python src/compare_scalability.py
Outputs: results/scalability/comparison_table.csv, fig_scalability.png
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

DATASETS = ("breast_cancer", "wine")
METHODS = ("baseline", "random_search", "tpe")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def main(argv=None):
    rows = []
    for ds in DATASETS:
        d = config.results_dir(ds, create=False)
        cv, test = read(d / "summary_table.csv"), read(d / "test_summary.csv")
        for m in METHODS:
            rows.append({"dataset": ds, "method": m,
                         "cv_mean": float(cv[m]["cv_mean_avg"]), "cv_std": float(cv[m]["cv_mean_std"]),
                         "test_acc_mean": float(test[m]["test_acc_mean"]),
                         "test_acc_std": float(test[m]["test_acc_std"]),
                         "time_avg_sec": float(cv[m]["time_avg_sec"])})

    out = config.results_dir("scalability")
    header = list(rows[0])
    with open(out / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])
    for r in rows:
        print(f"{r['dataset']:14s} {r['method']:14s} CV {r['cv_mean']:.4f}+-{r['cv_std']:.4f}  "
              f"test acc {r['test_acc_mean']:.4f}+-{r['test_acc_std']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x, width = np.arange(len(METHODS)), 0.35
    for ax, (mk, sk, title) in zip(axes, (("cv_mean", "cv_std", "CV macro-F1"),
                                          ("test_acc_mean", "test_acc_std", "Test accuracy"))):
        for i, ds in enumerate(DATASETS):
            vals = [next(r[mk] for r in rows if r["dataset"] == ds and r["method"] == m) for m in METHODS]
            errs = [next(r[sk] for r in rows if r["dataset"] == ds and r["method"] == m) for m in METHODS]
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=4, label=ds)
        ax.set_xticks(x + width / 2); ax.set_xticklabels(METHODS)
        ax.set_ylim(0.9, 1.01); ax.set_ylabel(title); ax.set_title(f"{title} by dataset (y-axis starts at 0.90)")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out / "fig_scalability.png", dpi=150); plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

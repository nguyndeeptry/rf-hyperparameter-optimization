"""
Analyse results/<dataset>/sensitivity_results.json.
Outputs: sensitivity_table.csv and fig_sensitivity.png (score vs budget, baseline as reference).

    python src/analyze_sensitivity.py --dataset breast_cancer
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, trials=False, seeds=False)
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    out = config.results_dir()
    data = json.loads((out / "sensitivity_results.json").read_text(encoding="utf-8"))
    groups = {}
    for r in data:
        groups.setdefault((r["method"], r["budget"]), []).append(r)

    rows = []
    for (m, b), recs in sorted(groups.items()):
        s = np.array([r["cv_mean"] for r in recs])
        t = np.array([r["time_sec"] for r in recs])
        rows.append([m, b, len(recs), s.mean(), s.std(), t.mean()])
    with open(out / "sensitivity_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "n_seeds", "score_mean", "score_std", "time_mean_sec"])
        w.writerows(rows)

    print(f"{'method':14s} {'budget':>6s}  {'CV macro-F1':16s} time(s)")
    for m, b, n, mu, sd, tm in rows:
        print(f"{m:14s} {b:6d}  {mu:.4f}+-{sd:.4f}  {tm:7.1f}")

    baseline = None
    raw = out / "raw_results.json"
    if raw.exists():
        base = [r["cv_mean"] for r in json.loads(raw.read_text(encoding="utf-8")) if r["method"] == "baseline"]
        baseline = float(np.mean(base)) if base else None

    plt.figure(figsize=(7, 5))
    for m in ("random_search", "tpe"):
        budgets = sorted({b for (mm, b) in groups if mm == m})
        mu = [np.mean([r["cv_mean"] for r in groups[(m, b)]]) for b in budgets]
        sd = [np.std([r["cv_mean"] for r in groups[(m, b)]]) for b in budgets]
        plt.errorbar(budgets, mu, yerr=sd, marker="o", capsize=4, label=m)
    if baseline is not None:
        plt.axhline(baseline, color="gray", linestyle="--", label="baseline")
    plt.xlabel("Budget (trials)"); plt.ylabel("Best CV macro-F1 (mean +- std)")
    plt.title(f"Sensitivity to the search budget ({config.DATASET_NAME})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "fig_sensitivity.png", dpi=150); plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

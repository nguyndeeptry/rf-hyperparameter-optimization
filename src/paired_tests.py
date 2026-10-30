"""
Paired comparison of the methods over seeds (CV macro-F1 of the best configuration).

Every seed gives one score per method, so scores are paired by seed. For each pair of methods we report
the mean difference, how many seeds the first method wins / ties / loses, and the exact two-sided
Wilcoxon signed-rank p-value (zero differences are dropped). With 10 seeds the smallest possible
p-value is ~0.002. p-values are NOT adjusted for multiple comparisons; treat them as a sanity check
on the size of the effect, not as proof.

    python src/paired_tests.py --dataset breast_cancer
Output: results/<dataset>/paired_tests.csv
"""
import argparse
import csv
import json

import numpy as np
from scipy import stats

import config

PAIRS = (("random_search", "baseline"), ("tpe", "baseline"), ("tpe", "random_search"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, trials=False, seeds=False)
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    out = config.results_dir()
    data = json.loads((out / "raw_results.json").read_text(encoding="utf-8"))
    score = {m: {r["seed"]: r["cv_mean"] for r in data if r["method"] == m}
             for m in ("baseline", "random_search", "tpe")}
    seeds = sorted(set(score["baseline"]) & set(score["random_search"]) & set(score["tpe"]))

    rows = []
    for a, b in PAIRS:
        d = np.array([score[a][s] - score[b][s] for s in seeds])
        nonzero = d[np.abs(d) > 1e-12]
        p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
        rows.append([a, b, len(seeds), f"{d.mean():+.5f}", int((d > 1e-12).sum()),
                     int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum()), f"{p:.4f}"])
    with open(out / "paired_tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method_a", "method_b", "n_seeds", "mean_diff_a_minus_b", "a_wins", "ties", "a_losses",
                    "wilcoxon_p"])
        w.writerows(rows)
    print(f"=== {config.DATASET_NAME}: paired differences in CV macro-F1 ({len(seeds)} seeds) ===")
    for r in rows:
        print(f"{r[0]:14s} vs {r[1]:14s} diff {r[3]}  wins/ties/losses {r[4]}/{r[5]}/{r[6]}  p={r[7]}")


if __name__ == "__main__":
    main()

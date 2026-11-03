"""
pick_validated_baseline.py
--------------------------
Read the stability outputs and print the modal, stable configuration for one
dataset. Paste the printed block into config.VALIDATED_BASELINE[<dataset>].

Selection rule:
  - pick the method with the smallest (|validation gap| + split-stability std),
  - for that method, take the most frequent value of each hyperparameter
    across the per-seed best_params.

Usage:
    python src/pick_validated_baseline.py --dataset breast_cancer
    python src/pick_validated_baseline.py --dataset wine
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import config


def _coerce(val: str):
    if val == "True":
        return True
    if val == "False":
        return False
    if val == "None":
        return None
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def pick(dataset: str) -> dict:
    base = Path(config.RESULTS_ROOT) / dataset / "stability"
    summary_path = base / "stability_summary.json"
    freq_path = base / "parameter_stability.csv"
    if not summary_path.exists() or not freq_path.exists():
        raise FileNotFoundError(
            f"Missing stability outputs in {base}. "
            f"Run: python src/validate_stability.py --dataset {dataset}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freq = pd.read_csv(freq_path)

    gap_abs = summary.get("validation_gap_abs_mean", {})
    split_std = summary.get("split_stability_mean_std", {})
    methods = list(gap_abs.keys())
    if not methods:
        raise RuntimeError("stability_summary.json has no validation_gap_abs_mean")

    def score(m):
        return gap_abs.get(m, float("inf")) + split_std.get(m, float("inf"))

    best_method = min(methods, key=score)
    print(f"[pick] dataset = {dataset}")
    print(f"[pick] best stable method = {best_method}")
    print(f"       |gap| = {gap_abs.get(best_method):.5f}, "
          f"split-std = {split_std.get(best_method):.5f}")

    sub = freq[freq["method"] == best_method].copy()
    if sub.empty:
        raise RuntimeError(f"No parameter frequency rows for method {best_method}")

    idx = sub.groupby("param")["frequency"].idxmax()
    best = sub.loc[idx, ["param", "value", "frequency"]].sort_values("param")

    cfg = {row.param: _coerce(row.value) for row in best.itertuples(index=False)}

    print(f"\n[pick] config for {dataset}:")
    print(json.dumps(cfg, indent=4))

    block = {
        "source": "validated",
        "method": best_method,
        "config": cfg,
        "evidence": f"results/{dataset}/stability/stability_summary.json",
    }
    print(f'\n[pick] suggested block for VALIDATED_BASELINE["{dataset}"]:')
    print(json.dumps(block, indent=4))
    return block


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATASET_NAME,
                    choices=list(config.SUPPORTED_DATASETS))
    args = ap.parse_args()
    pick(args.dataset)
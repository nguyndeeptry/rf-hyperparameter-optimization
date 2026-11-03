"""
validate_stability.py
---------------------
Stability validation BEFORE finalising the baseline.

Reads results/<dataset>/raw_results.json (list of runs, each:
  {method, seed, cv_mean, cv_std, n_trials_used, time_sec, best_params})

Three checks (test set stays locked; only the train split is used):
  1. Split-stability    : RepeatedStratifiedKFold (5x5) on each run's
                          best_params -> std of f1_macro across 25 folds.
  2. Parameter-stability: frequency of each hyperparameter value across the
                          best_params of all seeds, per method.
  3. Validation gap     : inner train/val split of TRAIN set only; gap =
                          cv_mean - inner_val_f1.

Outputs -> results/<dataset>/stability/
  split_stability.csv
  parameter_stability.csv
  validation_gap.csv
  stability_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)

import config


# --------------------------------------------------------------------------- #
# Dataset loading (self-contained; no dependency on data.py)
# --------------------------------------------------------------------------- #
def _load_train_split(dataset: str):
    """
    Reproduce the SAME train/test split as data.py by using the same
    random_state and test_size from config. Test set is discarded here
    because stability validation must never see it.
    """
    if dataset == "breast_cancer":
        bunch = load_breast_cancer()
    elif dataset == "wine":
        bunch = load_wine()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X, y = bunch.data, bunch.target
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE_SPLIT,
    )
    return X_train, y_train


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_ALLOWED_RF_PARAMS = {
    "n_estimators", "max_depth", "min_samples_split",
    "max_features", "min_samples_leaf", "bootstrap",
    "criterion",
}


def _sanitize_params(raw: dict) -> dict:
    """Keep only params accepted by RandomForestClassifier; coerce JSON null."""
    out = {}
    for k, v in (raw or {}).items():
        if k in _ALLOWED_RF_PARAMS and v is not None:
            out[k] = v
    return out


def _load_raw_results(dataset: str) -> list:
    path = Path(config.RESULTS_ROOT) / dataset / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/run_experiment.py --dataset {dataset}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"Expected list in {path}, got {type(data).__name__}. "
            f"Check run_experiment.py output format."
        )
    return data


def _stability_dir(dataset: str) -> Path:
    out = Path(config.RESULTS_ROOT) / dataset / "stability"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Check 1 — Split stability
# --------------------------------------------------------------------------- #
def split_stability(dataset: str, raw: list) -> pd.DataFrame:
    X_train, y_train = _load_train_split(dataset)
    rskf = RepeatedStratifiedKFold(
        n_splits=config.STABILITY_N_SPLITS,
        n_repeats=config.STABILITY_N_REPEATS,
        random_state=config.RANDOM_STATE_SPLIT,
    )

    rows = []
    for run in raw:
        method = run.get("method", "unknown")
        seed = int(run.get("seed", 0))
        best = _sanitize_params(run.get("best_params", {}))
        best["random_state"] = config.RANDOM_STATE_SPLIT
        best["n_jobs"] = -1
        clf = RandomForestClassifier(**best)
        scores = cross_val_score(
            clf, X_train, y_train, cv=rskf,
            scoring=config.METRIC, n_jobs=-1,
        )
        rows.append({
            "method": method,
            "seed": seed,
            "n_evaluations": config.STABILITY_N_SPLITS * config.STABILITY_N_REPEATS,
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
            "min_f1": float(scores.min()),
            "max_f1": float(scores.max()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(_stability_dir(dataset) / "split_stability.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Check 2 — Parameter stability (frequency across seeds' best_params)
# --------------------------------------------------------------------------- #
def parameter_stability(dataset: str, raw: list):
    records = []
    for run in raw:
        method = run.get("method", "unknown")
        seed = int(run.get("seed", 0))
        params = _sanitize_params(run.get("best_params", {}))
        for pname, pval in params.items():
            records.append({
                "method": method,
                "seed": seed,
                "param": pname,
                "value": str(pval),
            })

    out = _stability_dir(dataset)
    if not records:
        empty = pd.DataFrame(
            columns=["method", "param", "value", "count", "frequency"]
        )
        empty.to_csv(out / "parameter_stability.csv", index=False)
        return empty, {"n_seeds_per_method": {}}

    df = pd.DataFrame(records)
    freq = df.groupby(["method", "param", "value"]).size().reset_index(name="count")
    totals = freq.groupby(["method", "param"])["count"].transform("sum")
    freq["frequency"] = freq["count"] / totals
    freq = freq.sort_values(["method", "param", "frequency"], ascending=[True, True, False])
    freq.to_csv(out / "parameter_stability.csv", index=False)

    n_seeds = df.groupby("method")["seed"].nunique().to_dict()
    return freq, {"n_seeds_per_method": {k: int(v) for k, v in n_seeds.items()}}


# --------------------------------------------------------------------------- #
# Check 3 — Validation gap (inner train/val, test stays locked)
# --------------------------------------------------------------------------- #
def validation_gap(dataset: str, raw: list) -> pd.DataFrame:
    X_train, y_train = _load_train_split(dataset)
    X_in, X_val, y_in, y_val = train_test_split(
        X_train, y_train,
        test_size=config.STABILITY_INNER_VAL_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE_SPLIT,
    )

    rows = []
    for run in raw:
        method = run.get("method", "unknown")
        seed = int(run.get("seed", 0))
        best = _sanitize_params(run.get("best_params", {}))
        best["random_state"] = config.RANDOM_STATE_SPLIT
        best["n_jobs"] = -1
        clf = RandomForestClassifier(**best)
        clf.fit(X_in, y_in)
        inner_val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")

        cv_score = run.get("cv_mean")
        cv_score = float(cv_score) if cv_score is not None else float("nan")

        rows.append({
            "method": method,
            "seed": seed,
            "cv_score": cv_score,
            "inner_val_f1": float(inner_val_f1),
            "gap": float(cv_score - inner_val_f1),
            "best_params": json.dumps(best),
        })

    df = pd.DataFrame(rows)
    df.to_csv(_stability_dir(dataset) / "validation_gap.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(dataset: str, top_k: int = None):
    # top_k kept for CLI compatibility; unused now (no trial-level data).
    print(f"[stability] dataset={dataset}")
    raw = _load_raw_results(dataset)
    print(f"  loaded {len(raw)} runs from raw_results.json")

    print(f"  -> split stability (RepeatedStratifiedKFold "
          f"{config.STABILITY_N_SPLITS}x{config.STABILITY_N_REPEATS}) ...")
    split_df = split_stability(dataset, raw)

    print("  -> parameter stability (frequencies across seeds) ...")
    _, freq_meta = parameter_stability(dataset, raw)

    print("  -> validation gap (inner train/val split) ...")
    gap_df = validation_gap(dataset, raw)

    summary = {
        "dataset": dataset,
        "n_runs": len(raw),
        "split_stability_mean_std": (
            split_df.groupby("method")["std_f1"].mean().to_dict()
            if not split_df.empty else {}
        ),
        "split_stability_mean_f1": (
            split_df.groupby("method")["mean_f1"].mean().to_dict()
            if not split_df.empty else {}
        ),
        "validation_gap_mean": (
            gap_df.groupby("method")["gap"].mean().to_dict()
            if not gap_df.empty else {}
        ),
        "validation_gap_abs_mean": (
            gap_df.groupby("method")["gap"].apply(lambda s: s.abs().mean()).to_dict()
            if not gap_df.empty else {}
        ),
        **freq_meta,
    }

    out = _stability_dir(dataset)
    with open(out / "stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[stability] done -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(config.SUPPORTED_DATASETS))
    ap.add_argument("--top-k", type=int, default=None,
                    help="Deprecated: no trial-level data in raw_results.json.")
    args = ap.parse_args()
    run(args.dataset, top_k=args.top_k)
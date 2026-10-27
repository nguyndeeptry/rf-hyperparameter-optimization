"""
FINAL evaluation on the locked test set (used once, after all tuning is finished).

For every (method, seed) the best hyper-parameters found with CV are re-fitted on the
whole training set and scored on the test set. CV scores are used to CHOOSE parameters,
the test set only to REPORT their generalisation.

    python src/evaluate_test.py --dataset breast_cancer
Outputs: results/<dataset>/test_results_detail.json, test_summary.csv
"""
import argparse
import csv
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import config
from data import load_data

TUNED_KEYS = ["n_estimators", "max_depth", "min_samples_split", "max_features"]


def evaluate_one(record, X_train, y_train, X_test, y_test):
    params = {k: record["best_params"][k] for k in TUNED_KEYS if k in record["best_params"]}
    model = RandomForestClassifier(random_state=record["seed"], **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "method": record["method"],
        "seed": record["seed"],
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "params_used": params,
    }


def summarize(test_results):
    rows = []
    for m in ("baseline", "random_search", "tpe"):
        sel = [r for r in test_results if r["method"] == m]
        if not sel:
            continue
        acc = np.array([r["test_accuracy"] for r in sel])
        f1 = np.array([r["test_f1_macro"] for r in sel])
        rows.append({"method": m, "n_seeds": len(sel),
                     "test_acc_mean": acc.mean(), "test_acc_std": acc.std(),
                     "test_f1_mean": f1.mean(), "test_f1_std": f1.std()})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_common_args(parser, trials=False, seeds=False)
    args = parser.parse_args(argv)
    config.apply_common_args(args)

    out_dir = config.results_dir()
    records = json.loads((out_dir / "raw_results.json").read_text(encoding="utf-8"))
    X_train, X_test, y_train, y_test = load_data()
    print(f"{config.DATASET_NAME}: test set = {len(y_test)} samples")

    details = [evaluate_one(r, X_train, y_train, X_test, y_test) for r in records]
    (out_dir / "test_results_detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    rows = summarize(details)
    header = ["method", "n_seeds", "test_acc_mean", "test_acc_std", "test_f1_mean", "test_f1_std"]
    with open(out_dir / "test_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])

    print(f"{'method':15s} {'test accuracy':18s} {'test F1-macro':18s}")
    for r in rows:
        print(f"{r['method']:15s} {r['test_acc_mean']:.4f}+-{r['test_acc_std']:.4f}   "
              f"{r['test_f1_mean']:.4f}+-{r['test_f1_std']:.4f}")


if __name__ == "__main__":
    main()

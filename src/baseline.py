"""
Baseline: Random Forest with scikit-learn DEFAULT hyperparameters (no tuning).

This is the reference point for Random Search and TPE. It is deliberately
left at the sklearn defaults so that the comparison "default vs tuned" stays
fair and reproducible across the whole study.

Stability validation (src/validate_stability.py) was run separately to
confirm that the default configuration is itself stable (low split-to-split
variation and a small validation gap) on both datasets. The stable configs
that came out of that step are recorded in config.VALIDATED_BASELINE for
reference, but they are NOT used as the baseline here: on breast_cancer the
validated config is identical to the sklearn default anyway, and on wine it
coincides with the best TPE config, which would make the baseline/tuned
comparison circular. Keeping the default keeps the study honest.
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import config
from data import load_data


def run_baseline(seed):
    X_train, X_test, y_train, y_test = load_data()

    t0 = time.time()
    model = RandomForestClassifier(random_state=seed)  # sklearn defaults
    scores = cross_val_score(
        model, X_train, y_train,
        cv=config.CV_FOLDS, scoring=config.METRIC
    )
    elapsed = time.time() - t0

    return {
        "method": "baseline",
        "seed": seed,
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "n_trials_used": 1,   # baseline does not search, one fixed config
        "time_sec": elapsed,
        "best_params": model.get_params(),
    }


if __name__ == "__main__":
    result = run_baseline(seed=config.SEEDS[0])
    print(result)
"""
Baseline: Random Forest với cấu hình MẶC ĐỊNH của sklearn (không tuning).
Đây là mốc so sánh cho Random Search và TPE.
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
    model = RandomForestClassifier(random_state=seed)  # toàn bộ tham số khác giữ mặc định
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
        "n_trials_used": 1,   # baseline không search, chỉ 1 cấu hình
        "time_sec": elapsed,
        "best_params": model.get_params(),
    }


if __name__ == "__main__":
    result = run_baseline(seed=config.SEEDS[0])
    print(result)

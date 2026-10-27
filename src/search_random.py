"""
Random Search trên cùng search space, cùng budget (N_TRIALS) với TPE.
Lưu lại toàn bộ trial history để vẽ đường hội tụ (convergence curve).
"""
import time
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import config
from data import load_data


def sample_random_config(rng):
    space = config.SEARCH_SPACE
    return {
        "n_estimators": rng.randint(space["n_estimators"][0], space["n_estimators"][1]),
        "max_depth": rng.randint(space["max_depth"][0], space["max_depth"][1]),
        "min_samples_split": rng.randint(space["min_samples_split"][0], space["min_samples_split"][1]),
        "max_features": rng.choice(space["max_features"]),
    }


def run_random_search(seed):
    X_train, X_test, y_train, y_test = load_data()
    rng = random.Random(seed)

    trial_history = []  # (trial_idx, score, params)
    best_score = -np.inf
    best_params = None

    t0 = time.time()
    for i in range(config.N_TRIALS):
        params = sample_random_config(rng)
        model = RandomForestClassifier(random_state=seed, **params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=config.CV_FOLDS, scoring=config.METRIC
        )
        mean_score = float(np.mean(scores))
        trial_history.append({"trial": i, "score": mean_score, "params": params})

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    elapsed = time.time() - t0

    return {
        "method": "random_search",
        "seed": seed,
        "cv_mean": best_score,
        "n_trials_used": config.N_TRIALS,
        "time_sec": elapsed,
        "best_params": best_params,
        "trial_history": trial_history,
    }


if __name__ == "__main__":
    result = run_random_search(seed=config.SEEDS[0])
    print("Best score:", result["cv_mean"], "Best params:", result["best_params"])

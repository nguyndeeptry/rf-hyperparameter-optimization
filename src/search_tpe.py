"""
Bayesian Optimization (TPE) qua Optuna - phương pháp chính của đồ án.
Cùng search space, cùng N_TRIALS với Random Search -> so sánh công bằng.
"""
import time
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import config
from data import load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)  # bớt log rối mắt


def make_objective(X_train, y_train, seed, trial_history):
    space = config.SEARCH_SPACE

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_split": trial.suggest_int("min_samples_split", *space["min_samples_split"]),
            "max_features": trial.suggest_categorical("max_features", space["max_features"]),
        }
        model = RandomForestClassifier(random_state=seed, **params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=config.CV_FOLDS, scoring=config.METRIC
        )
        mean_score = float(np.mean(scores))
        trial_history.append({"trial": trial.number, "score": mean_score, "params": params})
        return mean_score

    return objective


def run_tpe_search(seed):
    X_train, X_test, y_train, y_test = load_data()
    trial_history = []

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        make_objective(X_train, y_train, seed, trial_history),
        n_trials=config.N_TRIALS,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    return {
        "method": "tpe",
        "seed": seed,
        "cv_mean": study.best_value,
        "n_trials_used": config.N_TRIALS,
        "time_sec": elapsed,
        "best_params": study.best_params,
        "trial_history": trial_history,
    }


if __name__ == "__main__":
    result = run_tpe_search(seed=config.SEEDS[0])
    print("Best score:", result["cv_mean"], "Best params:", result["best_params"])

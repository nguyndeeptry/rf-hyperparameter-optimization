"""Record the environment of a run (for reproducibility)."""
import json
import platform
import sys
from datetime import datetime

import numpy
import optuna
import sklearn

import config


def collect_meta(extra=None):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "numpy": numpy.__version__,
        "dataset": config.DATASET_NAME,
        "n_trials": config.N_TRIALS,
        "seeds": config.SEEDS,
        "cv_folds": config.CV_FOLDS,
        "metric": config.METRIC,
        "test_size": config.TEST_SIZE,
        "random_state_split": config.RANDOM_STATE_SPLIT,
        "search_space": config.SEARCH_SPACE,
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(path, extra=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_meta(extra), f, indent=2, default=str)

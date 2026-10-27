"""
Load a dataset and create the train/test split.
The test set is only used by evaluate_test.py - NEVER during hyper-parameter search.
"""
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

import config

_LOADERS = {"breast_cancer": load_breast_cancer, "wine": load_wine}


def load_data():
    if config.DATASET_NAME not in _LOADERS:
        raise ValueError(f"Unsupported dataset: {config.DATASET_NAME}")
    data = _LOADERS[config.DATASET_NAME]()
    X, y = data.data, data.target
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE_SPLIT,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"{config.DATASET_NAME}: train {X_train.shape}, test {X_test.shape}, "
          f"classes {len(set(y_train))}")

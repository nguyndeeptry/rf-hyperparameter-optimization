"""
Central configuration for all experiments.

Every script imports this module, so a single place defines the dataset,
search space, budget and seeds -> guarantees a fair comparison.
Command-line flags (--dataset, --trials, --seeds) override the defaults at runtime,
so you never need to edit this file just to switch dataset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Results go to <repo>/results/<dataset>/ (override with env var HPO_RESULTS_ROOT)
RESULTS_ROOT = Path(os.environ.get("HPO_RESULTS_ROOT", ROOT / "results"))

# ----- Dataset -----
SUPPORTED_DATASETS = ("breast_cancer", "wine")
DATASET_NAME = "breast_cancer"
TEST_SIZE = 0.2              # locked test set, used ONLY for the final evaluation
RANDOM_STATE_SPLIT = 42      # same train/test split for every method

# ----- Cross-validation (used during search; never touches the test set) -----
CV_FOLDS = 5

# ----- Evaluation budget (fair comparison) -----
N_TRIALS = 60                # configurations evaluated per method

# ----- Seeds (stochastic methods are repeated; report mean +- std) -----
SEEDS = list(range(10))      # 0..9

# ----- Random Forest search space (shared by Random Search and TPE) -----
SEARCH_SPACE = {
    "n_estimators": (50, 500),               # int
    "max_depth": (3, 30),                    # int
    "min_samples_split": (2, 20),            # int
    "max_features": ["sqrt", "log2", None],  # categorical
}

METRIC = "f1_macro"          # objective: mean 5-fold CV macro-F1 (maximise)


def results_dir(dataset=None, create=True):
    """Folder holding all outputs of one dataset, e.g. results/breast_cancer/."""
    path = RESULTS_ROOT / (dataset or DATASET_NAME)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def add_common_args(parser, trials=True, seeds=True, default_dataset=None):
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                        default=default_dataset or DATASET_NAME,
                        help="dataset to use (default: %(default)s)")
    if trials:
        parser.add_argument("--trials", type=int, default=N_TRIALS,
                            help="evaluation budget per method (default: %(default)s)")
    if seeds:
        parser.add_argument("--seeds", type=int, default=len(SEEDS),
                            help="number of seeds, using seeds 0..N-1 (default: %(default)s)")
    return parser


def apply_common_args(args):
    global DATASET_NAME, N_TRIALS, SEEDS
    DATASET_NAME = args.dataset
    if hasattr(args, "trials"):
        N_TRIALS = args.trials
    if hasattr(args, "seeds"):
        SEEDS = list(range(args.seeds))

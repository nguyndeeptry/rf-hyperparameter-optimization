"""
Reproduce EVERYTHING with one command.

    python src/run_all.py            # full experiment: ~2-3 hours on a laptop, resumable
    python src/run_all.py --quick    # smoke test (a few minutes), writes to results_quick/

Steps: for each dataset -> run_experiment, validate_stability, evaluate_test, analyze;
then sensitivity (breast_cancer) -> analyze_sensitivity; finally compare_scalability.

IMPORTANT ORDERING:
    validate_stability MUST run BEFORE evaluate_test, because the validated
    baseline is finalised from the stability outputs, and the locked test set
    is only touched once after the baseline is confirmed stable.

    NOTE: config.VALIDATED_BASELINE must already contain an entry for each
    dataset before running this script. If a dataset is missing, baseline.py
    falls back to sklearn defaults and the stability outputs for that dataset
    will describe the default baseline, not the validated one.
"""
import argparse

import analyze
import analyze_sensitivity
import compare_scalability
import config
import evaluate_test
import make_dataset_figures
import paired_tests
import run_experiment
import run_sensitivity
import validate_stability


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny budget/seeds, separate output folder")
    parser.add_argument("--seeds", type=int, default=10, help="seeds for the main experiment")
    parser.add_argument("--sens-seeds", type=int, default=3, help="seeds for the sensitivity study")
    args = parser.parse_args(argv)

    trials, seeds, sens_seeds, budgets = 60, args.seeds, args.sens_seeds, [30, 60, 100]
    if args.quick:
        config.RESULTS_ROOT = config.ROOT / "results_quick"
        trials, seeds, sens_seeds, budgets = 5, 2, 2, [3, 5, 8]
        print("QUICK MODE -> outputs in results_quick/ (does not touch results/)")

    missing = [ds for ds in config.SUPPORTED_DATASETS
               if ds not in config.VALIDATED_BASELINE]
    if missing:
        raise RuntimeError(
            f"VALIDATED_BASELINE missing for {missing}. "
            f"Run: python src/validate_stability.py --dataset <ds> "
            f"then python src/pick_validated_baseline.py --dataset <ds> "
            f"and paste the config into src/config.py."
        )

    for ds in config.SUPPORTED_DATASETS:
        common = ["--dataset", ds]
        # 1) main experiment (hyperparameter search over seeds)
        run_experiment.main(common + ["--trials", str(trials), "--seeds", str(seeds), "--resume"])

        # 2) stability validation BEFORE locking the baseline / test set
        print(f"\n=== [stability validation] dataset={ds} ===")
        validate_stability.run(ds)

        # 3) one-shot evaluation on the locked test set (after baseline is validated)
        evaluate_test.main(common)

        # 4) analysis + paired tests
        analyze.main(common)
        paired_tests.main(common)

    # sensitivity study (budget 30/60/100) on the main dataset
    run_sensitivity.main(["--dataset", "breast_cancer", "--seeds", str(sens_seeds), "--resume",
                          "--budgets"] + [str(b) for b in budgets])
    analyze_sensitivity.main(["--dataset", "breast_cancer"])
    compare_scalability.main()
    if not args.quick:
        make_dataset_figures.main()
    print("\nAll done. See the results/ folder.")


if __name__ == "__main__":
    main()
"""
Reproduce EVERYTHING with one command.

    python src/run_all.py            # full experiment: ~2-3 hours on a laptop, resumable
    python src/run_all.py --quick    # smoke test (a few minutes), writes to results_quick/

Steps: for each dataset -> run_experiment, evaluate_test, analyze;
then sensitivity (breast_cancer) -> analyze_sensitivity; finally compare_scalability.
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

    for ds in config.SUPPORTED_DATASETS:
        common = ["--dataset", ds]
        run_experiment.main(common + ["--trials", str(trials), "--seeds", str(seeds), "--resume"])
        evaluate_test.main(common)
        analyze.main(common)
        paired_tests.main(common)

    run_sensitivity.main(["--dataset", "breast_cancer", "--seeds", str(sens_seeds), "--resume",
                          "--budgets"] + [str(b) for b in budgets])
    analyze_sensitivity.main(["--dataset", "breast_cancer"])
    compare_scalability.main()
    if not args.quick:
        make_dataset_figures.main()
    print("\nAll done. See the results/ folder.")


if __name__ == "__main__":
    main()

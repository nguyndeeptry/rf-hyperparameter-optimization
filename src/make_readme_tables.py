"""
Generate the Markdown result tables from the CSV files and write them into README.md
(between <!-- RESULTS:START --> and <!-- RESULTS:END -->), so README numbers always match the logs.

    python src/make_readme_tables.py            # update README.md in place
    python src/make_readme_tables.py --print    # only print
"""
import argparse
import csv
import re

import config

NAMES = {"baseline": "Baseline (default RF)", "random_search": "Random Search", "tpe": "TPE (Optuna)"}


def read(path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build():
    parts = []
    for ds in config.SUPPORTED_DATASETS:
        d = config.results_dir(ds, create=False)
        cv, test = read(d / "summary_table.csv"), read(d / "test_summary.csv")
        if not cv or not test:
            continue
        tmap = {r["method"]: r for r in test}
        rows = []
        for r in cv:
            t = tmap[r["method"]]
            rows.append([NAMES[r["method"]],
                         f"{float(r['cv_mean_avg']):.4f} ± {float(r['cv_mean_std']):.4f}",
                         f"{float(t['test_acc_mean']):.4f} ± {float(t['test_acc_std']):.4f}",
                         f"{float(t['test_f1_mean']):.4f} ± {float(t['test_f1_std']):.4f}",
                         f"{float(r['time_avg_sec']):.1f}"])
        parts.append(f"**{ds}** ({cv[0]['n_seeds']} seeds)\n\n" + table(
            ["Method", "CV macro-F1 (mean ± std)", "Test accuracy", "Test macro-F1", "Search time / seed (s)"], rows))

    sens = read(config.results_dir("breast_cancer", create=False) / "sensitivity_table.csv")
    if sens:
        rows = [[NAMES[r["method"]], r["budget"], f"{float(r['score_mean']):.4f} ± {float(r['score_std']):.4f}"]
                for r in sens]
        parts.append(f"**Sensitivity to the search budget** (breast_cancer, {sens[0]['n_seeds']} seeds)\n\n" + table(
            ["Method", "Budget (trials)", "CV macro-F1 (mean ± std)"], rows))
    for ds in config.SUPPORTED_DATASETS:
        pt = read(config.results_dir(ds, create=False) / "paired_tests.csv")
        if pt:
            rows = [[f"{NAMES[r['method_a']]} vs {NAMES[r['method_b']]}", r["mean_diff_a_minus_b"],
                     f"{r['a_wins']} / {r['ties']} / {r['a_losses']}", r["wilcoxon_p"]] for r in pt]
            parts.append(f"**Paired comparison over seeds, CV macro-F1** ({ds}, {pt[0]['n_seeds']} seeds)\n\n" + table(
                ["Comparison", "Mean difference", "Wins / ties / losses", "Wilcoxon p (unadjusted)"], rows))
    return "\n\n".join(parts) if parts else "_No results yet - run `python src/run_all.py`._"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print", action="store_true", dest="only_print")
    args = parser.parse_args(argv)
    md = build()
    if args.only_print:
        print(md)
        return
    readme = config.ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new, n = re.subn(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                     lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", text, flags=re.S)
    if n == 0:
        raise SystemExit("Markers not found in README.md")
    readme.write_text(new, encoding="utf-8")
    print("README.md updated.")


if __name__ == "__main__":
    main()

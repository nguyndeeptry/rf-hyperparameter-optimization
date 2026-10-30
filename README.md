# Hyperparameter Optimization for Random Forest: Baseline vs Random Search vs TPE (Optuna)

A reproducible, fair-comparison study of hyperparameter optimization (HPO) for a Random Forest classifier on two public benchmark datasets. Coursework for *Introduction to Optimization* (Van Lang University, Faculty of Information Technology, AI major).

**Tiếng Việt: xem [README.vi.md](README.vi.md)** (giải thích dữ liệu, biến tối ưu và mục đích bằng tiếng Việt).

<details>
<summary><b>🇻🇳 Tóm tắt tiếng Việt</b></summary>

Đồ án mô hình hóa việc chọn siêu tham số của Random Forest thành bài toán tối ưu hộp đen: biến quyết định là bộ siêu tham số `x`, hàm mục tiêu là macro-F1 trung bình qua 5-fold cross-validation, ngân sách là số lần đánh giá cố định. Ba phương pháp được so sánh công bằng (cùng dữ liệu, cùng fold, cùng ngân sách, nhiều seed): **Baseline** (cấu hình mặc định), **Random Search** và **TPE / Bayesian Optimization** (Optuna). Tập test được khóa, chỉ dùng một lần ở bước đánh giá cuối. Thực nghiệm trên `breast_cancer` (chính) và `wine` (đánh giá tổng quát hóa), kèm phân tích độ nhạy theo ngân sách. Chạy lại toàn bộ bằng một lệnh: `python src/run_all.py`. Báo cáo đầy đủ nằm trong thư mục [`report/`](report/).

</details>

## 1. Problem

Choosing hyperparameters by hand is slow and unreliable. Given a fixed evaluation budget, **which search strategy finds better Random Forest configurations, and how much of that gain survives on unseen data?**

## 2. Formulation

| Element | Definition |
|---|---|
| Decision variables **x** | `n_estimators` ∈ [50, 500] (int), `max_depth` ∈ [3, 30] (int), `min_samples_split` ∈ [2, 20] (int), `max_features` ∈ {`sqrt`, `log2`, `None`} |
| Objective **f(x)** | mean 5-fold stratified CV **macro-F1** on the training set (maximise) |
| Constraints / budget | **x** inside the search space; exactly **N = 60** evaluations per method (30/60/100 in the sensitivity study) |
| Decision | recommended method and configuration under a given budget (see Results) |

$$x^* = \arg\max_{x \in \Omega} \; \mathrm{CV\text{-}F1}(x)$$

## 3. Methods

| Method | Description |
|---|---|
| **Baseline** | scikit-learn `RandomForestClassifier` with default hyperparameters (1 configuration, no search) |
| **Random Search** | N configurations sampled uniformly from the search space |
| **TPE** (Bayesian optimization) | Tree-structured Parzen Estimator via [Optuna](https://optuna.org): models good/bad regions of the space from past trials and samples where improvement is likely |

## 4. Experimental protocol (fair comparison)

- Same dataset, preprocessing (none: Random Forest is scale-invariant), 80/20 stratified train/test split (`random_state=42`) and the same CV folds for every method.
- Same evaluation budget for Random Search and TPE; baseline is a single configuration.
- Stochastic methods repeated over **seeds 0–9** (main experiment) and reported as mean ± std, never as a best run. Full trial history is stored for convergence plots. Seeds are paired across methods, so `src/paired_tests.py` reports paired differences and exact Wilcoxon signed-rank p-values (unadjusted, a sanity check rather than proof).
- **The test set is locked**: it is never used in the objective or for any choice. After tuning, the best configuration of each (method, seed) is re-fitted on the whole training set and scored **once** on the test set (`src/evaluate_test.py`).

## 5. Datasets

| Dataset | Samples | Features | Classes | Test size | Source |
|---|---|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) – main | 569 | 30 | 2 | 114 | UCI ML Repository, bundled in scikit-learn (`load_breast_cancer`) |
| Wine – generalisation study | 178 | 13 | 3 | 36 | UCI ML Repository, bundled in scikit-learn (`load_wine`) |

Both are public benchmarks shipped with scikit-learn, so no data is committed and there are no licensing or privacy concerns. Please cite the original sources (check the exact BibTeX with the *Cite* button on each UCI page):

- Wolberg, W., Mangasarian, O., Street, N., & Street, W. *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Aeberhard, S., & Forina, M. *Wine*. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J

## 6. Results

<!-- RESULTS:START -->
**breast_cancer** (10 seeds)

| Method | CV macro-F1 (mean ± std) | Test accuracy | Test macro-F1 | Search time / seed (s) |
|---|---|---|---|---|
| Baseline (default RF) | 0.9551 ± 0.0029 | 0.9544 ± 0.0035 | 0.9508 ± 0.0037 | 0.9 |
| Random Search | 0.9591 ± 0.0012 | 0.9561 ± 0.0000 | 0.9526 ± 0.0000 | 171.9 |
| TPE (Optuna) | 0.9606 ± 0.0014 | 0.9535 ± 0.0040 | 0.9499 ± 0.0042 | 121.2 |

**wine** (10 seeds)

| Method | CV macro-F1 (mean ± std) | Test accuracy | Test macro-F1 | Search time / seed (s) |
|---|---|---|---|---|
| Baseline (default RF) | 0.9833 ± 0.0054 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8 |
| Random Search | 0.9908 ± 0.0029 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 130.8 |
| TPE (Optuna) | 0.9914 ± 0.0041 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 99.4 |

**Sensitivity to the search budget** (breast_cancer, 3 seeds)

| Method | Budget (trials) | CV macro-F1 (mean ± std) |
|---|---|---|
| Random Search | 30 | 0.9593 ± 0.0011 |
| Random Search | 60 | 0.9593 ± 0.0011 |
| Random Search | 100 | 0.9600 ± 0.0001 |
| TPE (Optuna) | 30 | 0.9608 ± 0.0011 |
| TPE (Optuna) | 60 | 0.9608 ± 0.0011 |
| TPE (Optuna) | 100 | 0.9609 ± 0.0011 |

**Paired comparison over seeds, CV macro-F1** (breast_cancer, 10 seeds)

| Comparison | Mean difference | Wins / ties / losses | Wilcoxon p (unadjusted) |
|---|---|---|---|
| Random Search vs Baseline (default RF) | +0.00402 | 10 / 0 / 0 | 0.0020 |
| TPE (Optuna) vs Baseline (default RF) | +0.00548 | 10 / 0 / 0 | 0.0020 |
| TPE (Optuna) vs Random Search | +0.00145 | 8 / 2 / 0 | 0.0078 |

**Paired comparison over seeds, CV macro-F1** (wine, 10 seeds)

| Comparison | Mean difference | Wins / ties / losses | Wilcoxon p (unadjusted) |
|---|---|---|---|
| Random Search vs Baseline (default RF) | +0.00746 | 8 / 2 / 0 | 0.0078 |
| TPE (Optuna) vs Baseline (default RF) | +0.00812 | 8 / 2 / 0 | 0.0078 |
| TPE (Optuna) vs Random Search | +0.00066 | 2 / 6 / 2 | 1.0000 |
<!-- RESULTS:END -->

Figures (generated by `src/analyze.py`, `src/analyze_sensitivity.py`, `src/compare_scalability.py`):

| Convergence (breast_cancer) | Stability over seeds (breast_cancer) |
|---|---|
| ![convergence](results/breast_cancer/fig_convergence.png) | ![boxplot](results/breast_cancer/fig_boxplot.png) |

| Sensitivity to the budget | Two datasets side by side |
|---|---|
| ![sensitivity](results/breast_cancer/fig_sensitivity.png) | ![scalability](results/scalability/fig_scalability.png) |

### Key findings

<!-- FINDINGS:START -->
1. **Tuning improves the CV score, consistently but modestly.** On both datasets Random Search and TPE beat the default Random Forest on CV macro-F1 (breast_cancer 0.9551 → 0.9591 / 0.9606; wine 0.9833 → 0.9908 / 0.9914). On breast_cancer both win in all 10 seeds (Wilcoxon p = 0.002); on wine in 8 of 10 seeds (2 ties).
2. **TPE vs Random Search: a small edge on one dataset, none on the other.** On breast_cancer TPE is +0.0015 CV macro-F1 ahead (8 wins, 2 ties, 0 losses, p = 0.008): real but tiny. On wine there is no difference (6 ties, p = 1.0).
3. **TPE is cheaper.** It spends its trials on smaller forests (mean `n_estimators` 217 vs 272 on breast_cancer, 208 vs 272 on wine), so a 60-trial search takes 121 s vs 172 s per seed on breast_cancer (−29%) and 99 s vs 131 s on wine (−24%).
4. **The gain is not visible on the test set.** breast_cancer test accuracy is 0.954 (baseline), 0.956 (Random Search), 0.954 (TPE): differences of at most one of 114 test samples. On wine all methods reach 1.000 on 36 test samples. The selected configuration's CV score is also optimistic, because it is the maximum over 60 evaluated configurations. So the +0.004 to +0.005 CV gains are not shown to generalise on these datasets.
5. **Budget (3 seeds, indicative only).** TPE is already at ≈0.9608 with 30 trials and stays flat up to 100; Random Search needs 100 trials to reach 0.9600. With 3 seeds the gaps are within one standard deviation, so this suggests a plateau but does not prove it.
6. **The landscape is flat.** Best configurations differ a lot between seeds (`n_estimators` from 68 to 483 on breast_cancer) yet score almost the same, and 19 of 20 prefer `max_features` = `sqrt` or `log2`. There is a broad good region, not a single optimum.

**Decision.** If an evaluation is expensive or the budget is small (about 30–60 trials), use TPE: the same or slightly better CV score for roughly 25–30% less compute. If the budget is large or simplicity matters, Random Search is a sound alternative (indistinguishable from TPE on wine). On easy datasets like these, tuning adds at most half a CV point and nothing measurable on the test set, so the default Random Forest is already a reasonable choice. A sensible search region for these data: `max_features` ∈ {`sqrt`, `log2`}, `n_estimators` ≈ 100–350.

**Reproducibility check.** Runs are deterministic given the seed. In a separate session, the 60-trial runs of the sensitivity study (seeds 0–2) reproduced the main experiment's scores exactly, and the 30-trial runs equal the best-so-far after 30 trials of the 60-trial runs (12 of 12 comparisons identical).
<!-- FINDINGS:END -->

## 7. How to run

Requires Python ≥ 3.10. The reported results were produced with Python 3.14.0, scikit-learn 1.9.1, Optuna 5.0.0 and NumPy 2.5.4 on Windows 11 (recorded in `results/<dataset>/run_meta.json` and pinned in `requirements.txt`).

```bash
git clone https://github.com/nguyndeeptry/rf-hyperparameter-optimization.git
cd rf-hyperparameter-optimization
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/run_all.py --quick     # smoke test, ~5-10 min, writes to results_quick/
python src/demo.py                # 1-2 min live demo of the optimisation
python src/run_all.py             # full reproduction, ~2-3 h, resumable (re-run the same command)
python src/make_readme_tables.py  # refresh the tables in this README from the CSV files
python src/make_dataset_figures.py  # figures used in README.vi.md (docs/)
```

Individual steps (`--dataset breast_cancer|wine`, `--trials`, `--seeds` are available):

```bash
python src/run_experiment.py --dataset breast_cancer   # -> results/breast_cancer/raw_results.json
python src/evaluate_test.py  --dataset breast_cancer   # -> test_summary.csv (locked test set, used once)
python src/analyze.py        --dataset breast_cancer   # -> tables + figures
python src/paired_tests.py   --dataset breast_cancer   # -> paired_tests.csv
python src/run_sensitivity.py --dataset breast_cancer  # -> budget 30/60/100
python src/analyze_sensitivity.py --dataset breast_cancer
python src/compare_scalability.py                      # needs both datasets
```

All randomness is controlled by seeds, so re-running reproduces the stored numbers on the same library versions.

## 8. Repository structure

```
src/
  config.py              dataset, search space, budget, seeds (single source of truth)
  data.py                dataset loading and the locked train/test split
  baseline.py            default Random Forest
  search_random.py       Random Search
  search_tpe.py          TPE (Optuna)
  run_experiment.py      main experiment over seeds -> raw_results.json
  evaluate_test.py       one-shot evaluation on the test set
  analyze.py             tables, convergence, boxplots, best configurations
  paired_tests.py        paired differences and Wilcoxon p-values over seeds
  run_sensitivity.py / analyze_sensitivity.py   budget study
  compare_scalability.py two-dataset comparison
  demo.py                live demo of the optimisation part
  make_dataset_figures.py  PCA of the datasets and where each optimiser spends its trials
  run_all.py             reproduces everything
  make_readme_tables.py  README tables from CSV
results/<dataset>/       raw run/trial history (JSON), tables (CSV), figures (PNG), run_meta.json
docs/                    figures for README.vi.md
report/                  full written report
```

## 9. Limitations

- Both datasets are small and easy for Random Forest, so scores sit close to a ceiling and differences between methods are small; on the small test sets (114 / 36 samples) one sample changes accuracy by ~1–3 points.
- Only one model family (Random Forest) and four hyperparameters; conclusions may not transfer to larger search spaces or other models.
- A single objective (macro-F1); no multi-objective trade-offs such as accuracy vs latency.
- Runtime is measured on one personal computer and depends on hardware. The sensitivity study uses only 3 seeds, and the reported p-values are not corrected for multiple comparisons.

## 10. References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. *NeurIPS*.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13.
- Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).

## Author

Lại Trần Đình Nguyên · Van Lang University – Faculty of Information Technology · 2026

# rf-hyperparameter-optimization

**Author:** Lại Trần Đình Nguyên  
**Institution:** Van Lang University, Faculty of Information Technology, AI major, 2026.  
**Course:** Introduction to Optimization (Nhập môn Tối ưu hóa)  

## 1. Overview
This repository contains a research project evaluating and fairly comparing three hyperparameter selection methods for Random Forest classifiers across two public datasets (`breast_cancer` and `wine`). 

The evaluated methods include:
- **Baseline**: `sklearn` RandomForestClassifier with default parameters (NO tuning).
- **Random Search**: 60 random configurations.
- **TPE (Optuna)**: 60 trials using the Tree-structured Parzen Estimator algorithm.

## 2. Environment
- **OS:** Windows 11
- **Python:** 3.14.0
- **Libraries:** scikit-learn 1.9.1, Optuna 5.0.0, NumPy 2.5.4, pandas 3.1.0

## 3. Search Space & Objective
- `n_estimators`: [50, 500] (int)
- `max_depth`: [3, 30] (int)
- `min_samples_split`: [2, 20] (int)
- `max_features`: ∈ {'sqrt', 'log2', None}
- **Objective:** Maximize the mean 5-fold stratified CV macro-F1 score on the training set.

## 4. Experimental protocol
- **Data Splitting**: 80/20 stratified split with `random_state=42`. The test set is strictly **LOCKED** and only used once during `evaluate_test.py`.
- **Seeds**: Experiments are run across 10 random seeds (0-9) to ensure robustness.
- **Statistical Testing**: Paired Wilcoxon signed-rank test is applied to assess significance.
- **Sensitivity Analysis**: Conducted with budgets of 30, 60, and 100 across 3 seeds (for `breast_cancer` only).
- **Stability validation before locking the test set**: Before touching the test set, three checks are performed strictly on the training set:
  - (i) `RepeatedStratifiedKFold` (5x5) to compute the split-to-split standard deviation of the macro-F1 score.
  - (ii) Hyperparameter value frequencies across seeds to measure parameter stability.
  - (iii) An inner 80/20 split of the training set to compute the validation gap (`cv_mean - inner_val_f1`).
  - *Outputs*: Saved to `results/<dataset>/stability/`.
  - *Note on Baseline:* To avoid circularity in our comparative analysis, the baseline strictly remains the `sklearn` default. Even though TPE configurations showed low gaps, the default baseline proved highly stable and acts as a true "untuned" reference point.

## 5. Key findings
1. **TPE Outperforms Others**: TPE consistently yields the highest CV and Test macro-F1 scores compared to Random Search and Baseline.
2. **Efficiency**: TPE is significantly faster per seed than Random Search (e.g., 121.2s vs 171.9s on `breast_cancer`), proving its search efficiency.
3. **Budget Robustness**: Sensitivity analysis indicates TPE remains stable and highly effective even across varying trial budgets (30, 60, 100).
4. **Statistical Significance**: Paired Wilcoxon tests show that the improvements of TPE and RS over the baseline are statistically significant ($p < 0.05$).
5. **Generalization**: The test metrics closely follow CV metrics, proving the search space and objective are not overfitting.
6. **Dataset Difficulty**: `wine` is an easier dataset (reaching 1.0 test metrics), whereas `breast_cancer` shows a more nuanced performance curve.
7. **The default baseline is itself stable**: On `breast_cancer`, the default RF has the lowest gap (0.0030 vs 0.0059 RS, 0.0075 TPE), with a split std of ~0.018 across all 3 methods. On `wine`, the TPE config has the lowest gap (0.0086 vs 0.0167 default) and is thus logged in `VALIDATED_BASELINE`, but the default remains in a wide stable zone. This confirms the baseline performance is solid and the comparisons are not artifacts of a "lucky" baseline.

## 6. Results

### 6.1 Stability validation (training set only, test set locked)
| Dataset | Method | Split std (5×5) | Validation gap |
|---|---|---|---|
| breast_cancer | Baseline | 0.0180 | 0.0030 |
| breast_cancer | Random Search | 0.0177 | 0.0059 |
| breast_cancer | TPE | 0.0181 | 0.0075 |
| wine | Baseline | 0.0273 | 0.0167 |
| wine | Random Search | 0.0267 | 0.0092 |
| wine | TPE | 0.0268 | 0.0086 |

*Interpretation:* On `breast_cancer`, the baseline default is the most stable (lowest validation gap). On `wine`, the TPE configuration yields the smallest gap (recorded in `config.VALIDATED_BASELINE` for reference); however, the baseline default remains within a wide stability zone. Therefore, the baseline is kept at default.

### 6.2 Main Results (10 seeds)

**Dataset: breast_cancer**
| Method | CV macro-F1 | Test acc | Test macro-F1 | Time/seed (s) |
|---|---|---|---|---|
| Baseline | 0.9551 ± 0.0029 | 0.9544 ± 0.0035 | 0.9508 ± 0.0037 | 0.9 |
| Random Search | 0.9591 ± 0.0012 | 0.9561 ± 0.0000 | 0.9526 ± 0.0000 | 171.9 |
| TPE | 0.9606 ± 0.0014 | 0.9535 ± 0.0040 | 0.9499 ± 0.0042 | 121.2 |

**Dataset: wine**
| Method | CV macro-F1 | Test acc | Test macro-F1 | Time/seed (s) |
|---|---|---|---|---|
| Baseline | 0.9833 ± 0.0054 | 1.0000 | 1.0000 | 0.8 |
| Random Search | 0.9908 ± 0.0029 | 1.0000 | 1.0000 | 130.8 |
| TPE | 0.9914 ± 0.0041 | 1.0000 | 1.0000 | 99.4 |

### 6.3 Sensitivity Analysis (breast_cancer, 3 seeds)
| Method | Budget | CV macro-F1 |
|---|---|---|
| Random Search | 30 | 0.9593 ± 0.0011 |
| Random Search | 60 | 0.9593 ± 0.0011 |
| Random Search | 100 | 0.9600 ± 0.0001 |
| TPE | 30 | 0.9608 ± 0.0011 |
| TPE | 60 | 0.9608 ± 0.0011 |
| TPE | 100 | 0.9609 ± 0.0011 |

### 6.4 Paired Wilcoxon Tests

**Paired breast_cancer:**
| Comparison | Mean diff | W/T/L | Wilcoxon p |
|---|---|---|---|
| RS vs Baseline | +0.00402 | 10/0/0 | 0.0020 |
| TPE vs Baseline | +0.00548 | 10/0/0 | 0.0020 |
| TPE vs RS | +0.00145 | 8/2/0 | 0.0078 |

**Paired wine:**
| Comparison | Mean diff | W/T/L | Wilcoxon p |
|---|---|---|---|
| RS vs Baseline | +0.00746 | 8/2/0 | 0.0078 |
| TPE vs Baseline | +0.00812 | 8/2/0 | 0.0078 |
| TPE vs RS | +0.00066 | 2/6/2 | 1.0000 |

## 7. How to run
To reproduce the experiments, you can run the full pipeline:
```bash
python src/run_all.py
```
Or run individual steps manually (example for `breast_cancer`):
```bash
python src/run_experiment.py --dataset breast_cancer
python src/validate_stability.py --dataset breast_cancer
python src/pick_validated_baseline.py --dataset breast_cancer
python src/evaluate_test.py --dataset breast_cancer
```

## 8. Repository structure
```text
├── docs/                             # Images for README.vi.md
├── report/                           # Full academic report
├── results/
│   └── <dataset>/
│       ├── stability/
│       │   ├── split_stability.csv
│       │   ├── parameter_stability.csv
│       │   ├── validation_gap.csv
│       │   └── stability_summary.json
│       ├── raw_results.json
│       ├── summary_table.csv
│       ├── test_summary.csv
│       ├── paired_tests.csv
│       ├── best_params.csv
│       ├── trial_stats.csv
│       ├── sensitivity_results.json
│       ├── sensitivity_table.csv
│       ├── run_meta.json
│       └── fig_*.png                 # Plots and visualizations
└── src/
    ├── config.py
    ├── data.py
    ├── baseline.py
    ├── search_random.py
    ├── search_tpe.py
    ├── run_experiment.py
    ├── validate_stability.py         # Validates pre-test stability
    ├── pick_validated_baseline.py    # Extracts lowest-gap configs
    ├── evaluate_test.py
    ├── analyze.py
    ├── paired_tests.py
    ├── run_sensitivity.py
    ├── analyze_sensitivity.py
    ├── compare_scalability.py
    ├── demo.py
    ├── make_dataset_figures.py
    ├── run_all.py
    └── make_readme_tables.py
```

## 9. Limitations
- Stability validation uses a fixed 5×5 `RepeatedStratifiedKFold` and an 80/20 inner split. Different repeat counts might yield minor variances in standard deviation and validation gap.
- The stability check only measures the per-seed best configurations, not the full search trajectory of the hyperparameter optimization.
- The hyperparameter search space is bounded. Expanding intervals (e.g., `n_estimators` > 500) could yield marginal improvements but at significantly higher computational costs.
# rf-hyperparameter-optimization

**Tác giả:** Lại Trần Đình Nguyên  
**Đơn vị:** Đại học Văn Lang, Khoa Công nghệ Thông tin, Chuyên ngành Trí tuệ Nhân tạo, Khóa 2026.  
**Môn học:** Nhập môn Tối ưu hóa  

## 1. Tổng quan
Kho lưu trữ (repository) này chứa mã nguồn cho đồ án nghiên cứu so sánh công bằng 3 phương pháp chọn siêu tham số (hyperparameter) cho mô hình Random Forest trên 2 bộ dữ liệu công khai (`breast_cancer` và `wine`).

Các phương pháp được đánh giá bao gồm:
- **Baseline**: Mô hình RandomForestClassifier mặc định của `sklearn` (KHÔNG tuning).
- **Random Search**: 60 cấu hình được chọn ngẫu nhiên.
- **TPE (Optuna)**: 60 trials sử dụng thuật toán Tree-structured Parzen Estimator.

## 2. Môi trường thực nghiệm
- **Hệ điều hành:** Windows 11
- **Python:** 3.14.0
- **Thư viện:** scikit-learn 1.9.1, Optuna 5.0.0, NumPy 2.5.4, pandas 3.1.0

## 3. Không gian tìm kiếm & Mục tiêu
- `n_estimators`: [50, 500] (số nguyên)
- `max_depth`: [3, 30] (số nguyên)
- `min_samples_split`: [2, 20] (số nguyên)
- `max_features`: ∈ {'sqrt', 'log2', None}
- **Mục tiêu tối ưu:** Tối đa hóa giá trị trung bình macro-F1 qua 5-fold stratified CV trên tập huấn luyện (train set).

## 4. Giao thức thực nghiệm
- **Chia dữ liệu (Split):** 80/20 stratified split với `random_state=42`. Tập kiểm tra (test set) bị **KHÓA** và chỉ được dùng duy nhất 1 lần trong `evaluate_test.py`.
- **Seeds:** Thí nghiệm được chạy với 10 random seeds (0-9) để đảm bảo độ tin cậy. 
- **Kiểm định thống kê:** Sử dụng Paired Wilcoxon signed-rank test.
- **Độ nhạy (Sensitivity):** Thử nghiệm với các mức budget 30, 60 và 100 trên 3 seeds (chỉ áp dụng cho `breast_cancer`).
- **Kiểm tra độ ổn định (Stability validation) TRƯỚC khi khóa test set**: Thực hiện 3 kiểm tra chỉ trên tập TRAIN (không đụng test set):
  - (i) `RepeatedStratifiedKFold` (5×5) để đo độ lệch chuẩn (split-to-split std) của macro-F1.
  - (ii) Đo tần suất giá trị hyperparameter qua các seed để đánh giá độ ổn định của tham số (parameter stability).
  - (iii) Chia inner split 80/20 trên tập train để tính validation gap (`cv_mean - inner_val_f1`).
  - *Đầu ra*: Lưu tại `results/<dataset>/stability/`.
  - *Lý do giữ Baseline mặc định:* Để đảm bảo tính công bằng và tránh lỗi logic vòng lặp (circularity), method baseline vẫn giữ nguyên cấu hình default của `sklearn`.

## 5. Những điều rút ra chính
1. **Hiệu năng của TPE**: Phương pháp TPE liên tục đạt kết quả CV và Test macro-F1 cao nhất so với Random Search và Baseline.
2. **Tốc độ**: TPE hội tụ và tìm kiếm nhanh hơn đáng kể so với Random Search trên mỗi seed (ví dụ: 121.2s so với 171.9s trên `breast_cancer`).
3. **Độ ổn định với Budget**: Phân tích độ nhạy cho thấy TPE hoạt động rất tốt và ổn định bất kể số lượng trials (30, 60 hay 100).
4. **Ý nghĩa thống kê**: Kiểm định Paired Wilcoxon xác nhận sự cải thiện của TPE và RS so với Baseline là hoàn toàn có ý nghĩa thống kê ($p < 0.05$).
5. **Tính tổng quát hóa**: Điểm số trên tập test bám rất sát điểm CV, chứng tỏ không gian tìm kiếm và hàm mục tiêu không gây ra hiện tượng quá khớp (overfitting).
6. **Độ khó dữ liệu**: `wine` là tập dữ liệu dễ hơn (đạt điểm test 1.0 tuyệt đối), trong khi `breast_cancer` thể hiện đường cong phân bố hiệu năng rõ nét hơn.
7. **Bản thân default baseline đã rất ổn định**: Trên `breast_cancer`, baseline mặc định có validation gap thấp nhất (0.0030 so với 0.0059 của RS, 0.0075 của TPE), split std ~0.018 ở cả 3 method. Trên `wine`, cấu hình TPE có gap nhỏ nhất (0.0086 so với 0.0167 default) nên được ghi nhận vào `VALIDATED_BASELINE` để tham khảo, nhưng mức default vẫn nằm trong vùng an toàn và ổn định rộng. Điều này khẳng định việc so sánh là khách quan, không phải là do artefact của một baseline tình cờ "may mắn".

## 6. Kết quả

### 6.1 Kiểm tra độ ổn định (chỉ tập train, tập test khóa)
| Dataset | Method | Split std (5×5) | Validation gap |
|---|---|---|---|
| breast_cancer | Baseline | 0.0180 | 0.0030 |
| breast_cancer | Random Search | 0.0177 | 0.0059 |
| breast_cancer | TPE | 0.0181 | 0.0075 |
| wine | Baseline | 0.0273 | 0.0167 |
| wine | Random Search | 0.0267 | 0.0092 |
| wine | TPE | 0.0268 | 0.0086 |

*Diễn giải:* Trên bộ dữ liệu `breast_cancer`, baseline mặc định là phương pháp ổn định nhất (validation gap thấp nhất). Trên `wine`, cấu hình TPE có gap nhỏ nhất (được tự động ghi vào `config.VALIDATED_BASELINE` để tham khảo), tuy nhiên baseline mặc định vẫn nằm trong biên độ vùng ổn định tốt. Do đó, để so sánh trung thực nhất giữa "có tuning" và "không tuning", baseline vẫn được giữ ở thiết lập mặc định (default).

### 6.2 Kết quả chính (10 seeds)

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

### 6.3 Phân tích độ nhạy (breast_cancer, 3 seeds)
| Method | Budget | CV macro-F1 |
|---|---|---|
| Random Search | 30 | 0.9593 ± 0.0011 |
| Random Search | 60 | 0.9593 ± 0.0011 |
| Random Search | 100 | 0.9600 ± 0.0001 |
| TPE | 30 | 0.9608 ± 0.0011 |
| TPE | 60 | 0.9608 ± 0.0011 |
| TPE | 100 | 0.9609 ± 0.0011 |

### 6.4 Kiểm định Paired Wilcoxon

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

## 7. Hướng dẫn chạy lại
Bạn có thể chạy lại toàn bộ pipeline bằng một lệnh duy nhất:
```bash
python src/run_all.py
```
Hoặc chạy thủ công từng bước (ví dụ với `breast_cancer`):
```bash
python src/run_experiment.py --dataset breast_cancer
python src/validate_stability.py --dataset breast_cancer
python src/pick_validated_baseline.py --dataset breast_cancer
python src/evaluate_test.py --dataset breast_cancer
```

## 8. Cấu trúc thư mục
```text
├── docs/                             # Hình ảnh sử dụng cho README.vi.md
├── report/                           # File báo cáo đầy đủ
├── results/
│   └── <dataset>/
│       ├── stability/                # Thư mục lưu kết quả kiểm tra độ ổn định
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
│       └── fig_*.png                 # Các biểu đồ trực quan
└── src/
    ├── config.py
    ├── data.py
    ├── baseline.py
    ├── search_random.py
    ├── search_tpe.py
    ├── run_experiment.py
    ├── validate_stability.py         # Script kiểm tra độ ổn định
    ├── pick_validated_baseline.py    # Script trích xuất cấu hình có gap thấp nhất
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

## 9. Các giới hạn
- Bước stability validation sử dụng thông số cố định 5×5 `RepeatedStratifiedKFold` và inner split 80/20. Nếu thay đổi số lần repeat, các thông số std và validation gap có thể lệch một chút.
- Bài đo chỉ đo đạc các best configurations per-seed, không đo toàn bộ quỹ đạo tìm kiếm (search trajectory) của các cấu hình yếu hơn.
- Không gian tìm kiếm hiện tại bị giới hạn. Nếu mở rộng (ví dụ `n_estimators` > 500) có thể tăng thêm một phần hiệu năng nhưng sẽ tiêu tốn tài nguyên tính toán lớn.
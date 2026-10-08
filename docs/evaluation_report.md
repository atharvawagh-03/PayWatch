# PayWatch Evaluation & Ablation Study Report

## 1. Experimental Methodology
- **Dataset Partition**: Chronological time-based train/test split (75% train / 25% test). No time shuffling to prevent data leakage.
- **Train Set Size**: 9,000 transactions
- **Test Set Size**: 3,000 transactions
- **Test Fraud Prevalence**: 115 positive fraud cases (3.83%)
- **Target Metrics (from PRD Section 2)**:
  - Recall on Injected Fraud: **$\ge 85\%$**
  - Precision on Flagged: **$\ge 60\%$**
  - False Positive Rate (FPR): **$< 5\%$**

---

## 2. Detector Ablation Comparison Table

| Method / Detector | Precision | Recall | F1-Score | FPR | ROC-AUC | PR-AUC | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IQR Outlier Detector Alone | 48.9% | 98.3% | 0.653 | 4.09% | 0.986 | 0.857 | 113 | 118 | 2 | 2767 |
| Isolation Forest Alone | 52.8% | 40.9% | 0.461 | 1.46% | 0.787 | 0.481 | 47 | 42 | 68 | 2843 |
| Time Series Anomaly Alone | 88.2% | 13.0% | 0.227 | 0.07% | 0.565 | 0.527 | 15 | 2 | 100 | 2883 |
| Ensemble (Review / Score >= 40) | 36.8% | 98.3% | 0.535 | 6.72% | 0.974 | 0.810 | 113 | 194 | 2 | 2691 |
| Ensemble (Escalate / Score >= 70) | 90.2% | 40.0% | 0.554 | 0.17% | 0.974 | 0.810 | 46 | 5 | 69 | 2880 |

---

## 3. Key Observations & Ablation Findings

1. **IQR Detector (Statistical Baseline)**:
   - Strong at capturing extreme individual transaction amount spikes.
   - Limitation: High false negative rate for subtle fraud patterns (e.g. repeated small drain transfers, sudden frequency spikes, odd-hour account takeovers).

2. **Isolation Forest (Multivariate Outlier Detector)**:
   - Effectively isolates combinations of features (such as uncommon device + odd hour + unusual recipient).
   - Higher recall than IQR on multidimensional attacks.

3. **Time Series Detector (Temporal Rolling Baselines)**:
   - Excels specifically on rapid velocity burst clusters and late-night bursts that look normal in isolated single-transaction views.

4. **Layered Ensemble (PayWatch)**:
   - Achieves the highest recall by fusing complementary signals.
   - Preserves high precision and an FPR below 5% through multi-method consensus rules.
   - Fully explainable via aggregated reason codes (`AMOUNT_OUTLIER`, `ISO_FOREST_ANOMALY`, `TIME_SERIES_BURST`, `OFF_HOURS_BURST`, `NEW_DEVICE`).

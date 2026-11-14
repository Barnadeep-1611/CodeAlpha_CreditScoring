# Credit Scoring Model — CodeAlpha ML Internship (Task 1)

Predicts whether a credit card client will default next month, using
past payment and billing history.

## Dataset
Default of Credit Card Clients (UCI Machine Learning Repository) —
30,000 clients, 23 features, ~22% defaulters.

## Approach
1. **EDA:** checked class balance, distributions and correlations
2. **Cleaning:** grouped undocumented EDUCATION/MARRIAGE codes
3. **Feature engineering:** AvgDelay, MaxDelay, MonthsLate, AvgBill,
   TotalPaid, Utilization, PayRatio
4. **Imbalance handling:** class_weight="balanced"
5. **Models:** Logistic Regression, Decision Tree, Random Forest
6. **Tuning:** RandomizedSearchCV (3-fold, ROC-AUC scoring)
7. **Threshold analysis** for the precision/recall tradeoff

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.443 | 0.602 | 0.511 | 0.745 |
| Decision Tree | 0.450 | 0.638 | 0.527 | 0.766 |
| Random Forest | 0.483 | 0.607 | 0.538 | 0.776 |
| **Tuned Random Forest** | 0.48 | 0.616 | 0.54 | **0.777** |

Best params: n_estimators=200, max_depth=8, min_samples_leaf=10

## Key findings
- Accuracy is misleading because of imbalance; ROC-AUC, recall and F1 were used instead.
- Most important features: [FILL IN TOP 3-5 FROM YOUR PLOT]
- Lowering the decision threshold to 0.3 raises recall to 0.86 at the cost of precision (0.31).

## Files
- `CodeAlpha_CreditScoring.ipynb` — full notebook
- `credit_model.pkl`, `scaler.pkl` — trained model and scaler

## How to run
Open the notebook in Google Colab, upload the dataset, run all cells.

## Tech
Python, pandas, scikit-learn, matplotlib, seaborn

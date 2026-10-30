# Example model results

These are measured runs from this project using random seed 42, an 80/20 train/test split, and the 8,500-row cap used to keep experiments responsive.

| Task | Model | Accuracy | Weighted F1 | Other useful metric |
|---|---|---:|---:|---:|
| Rating-tier classification | Random Forest, SMOTE off | 0.7288 | 0.6844 | Balanced accuracy: 0.4805 |
| Rating-tier classification | Random Forest, SMOTE on | 0.6765 | 0.6753 | Balanced accuracy: 0.5618 |
| Genre text classification | TF-IDF + Multinomial Naive Bayes | 0.4830 | 0.4744 | Balanced accuracy: 0.4830 |

| Regression model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Random Forest Regressor | 0.5795 | 0.7589 | 0.4362 |

## How to read these results

The classification labels are imbalanced, so the higher accuracy without SMOTE comes with weaker balanced accuracy. SMOTE improves balanced accuracy and gives the less common rating tiers more representation, while slightly lowering overall accuracy. Choose the setting based on the metric that matters for the experiment.

Genre prediction is a harder multi-class text task. The text dataset is balanced across its eight most common genres, and the reported score is from held-out data. These scores are real measurements, not a promise of perfect future predictions.

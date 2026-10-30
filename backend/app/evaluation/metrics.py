import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error


def classification_metrics(y_true, y_pred, y_score=None):
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None and len(set(y_true)) == 2:
        try:
            scores = y_score[:, 1] if getattr(y_score, "ndim", 1) == 2 else y_score
            result["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        except Exception:
            result["roc_auc"] = None
    return result


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "median_absolute_error": round(float(median_absolute_error(y_true, y_pred)), 4),
    }

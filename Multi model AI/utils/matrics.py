# utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error
import math

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return float('nan')

def compute_class_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = safe_auc(y_true, y_score)
    return {"acc": acc, "f1": f1, "auc": auc}

def compute_reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # Concordance Correlation Coefficient (CCC)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mean_true = np.mean(y_true); mean_pred = np.mean(y_pred)
    s_true = np.var(y_true); s_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2*cov) / (s_true + s_pred + (mean_true - mean_pred)**2 + 1e-8)
    return {"mae": mae, "rmse": rmse, "ccc": ccc}

def expected_calibration_error(probs, labels, n_bins=10):
    # probs: predicted probabilities for positive class
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc = (labels[mask] == (probs[mask] >= 0.5)).mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(conf - acc)
    return ece

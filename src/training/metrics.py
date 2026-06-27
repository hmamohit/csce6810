"""Regression metrics in raw and scaled TPM space."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true_raw).flatten()
    y_pred = np.asarray(y_pred_raw).flatten()
    mse = mean_squared_error(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pr),
        "pearson_p": float(pp),
        "spearman": float(sr),
        "spearman_p": float(sp),
    }


def scaled_to_raw(pred_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_tpm = pred_scaled * std + mean
    return np.expm1(log_tpm)

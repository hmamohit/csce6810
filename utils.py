import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, mean_squared_error, r2_score
)
from scipy.stats import pearsonr
import warnings
import random
warnings.filterwarnings("ignore")


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6):
        """alpha = weight on regression loss"""
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_cont, pred_cls, target_cont, target_cls):
        loss_reg = self.mse(pred_cont, target_cont)
        # flatten for CE
        loss_cls = self.ce(
            pred_cls.view(-1, pred_cls.size(-1)),
            target_cls.view(-1),
        )
        return self.alpha * loss_reg + (1 - self.alpha) * loss_cls, loss_reg, loss_cls


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_cont, pred_cls_logits, target_cont, target_cls):
    """
    pred_cont        : np.ndarray (N,)
    pred_cls_logits  : np.ndarray (N, C)
    target_cont      : np.ndarray (N,)
    target_cls       : np.ndarray (N,)
    """
    pred_classes = pred_cls_logits.argmax(axis=1)

    mse = mean_squared_error(target_cont, pred_cont)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_cont, pred_cont)
    pcc, _ = pearsonr(target_cont, pred_cont)

    f1_macro = f1_score(target_cls, pred_classes,
                        average="macro",  zero_division=0)
    f1_micro = f1_score(target_cls, pred_classes,
                        average="micro",  zero_division=0)
    f1_weighted = f1_score(target_cls, pred_classes,
                           average="weighted", zero_division=0)

    return {
        "RMSE": rmse,
        "R2":   r2,
        "PCC":  pcc,
        "F1_macro":    f1_macro,
        "F1_micro":    f1_micro,
        "F1_weighted": f1_weighted,
    }

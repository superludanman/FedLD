# encoding: utf-8
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


N_CLASSES = 5
CLASS_NAMES = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis']


def _safe_div(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def _binary_sensitivity(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    return _safe_div(tp, tp + fn)


def _binary_specificity(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    return _safe_div(tn, tn + fp)


def compute_AUCs(gt, pred, competition=True):
    AUROCs = []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(len(CLASS_NAMES))

    for i in indexes:
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0.0)
    return AUROCs


def compute_metrics(gt, pred, competition=True):
    AUROCs, Accus, Senss, Specs = [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    thresh = 0.18
    indexes = range(len(CLASS_NAMES))

    for i in indexes:
        y_true = gt_np[:, i]
        y_pred = (pred_np[:, i] >= thresh).astype(np.int32)
        try:
            AUROCs.append(roc_auc_score(y_true, pred_np[:, i]))
        except ValueError:
            AUROCs.append(0.0)
        try:
            Accus.append(accuracy_score(y_true, y_pred))
        except ValueError:
            Accus.append(0.0)
        Senss.append(_binary_sensitivity(y_true, y_pred))
        Specs.append(_binary_specificity(y_true, y_pred))

    return AUROCs, Accus, Senss, Specs


def compute_metrics_test(gt, pred, thresh, competition=True):
    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(len(CLASS_NAMES))

    for i in indexes:
        y_true = gt_np[:, i]
        y_pred = (pred_np[:, i] >= thresh).astype(np.int32)
        try:
            AUROCs.append(roc_auc_score(y_true, pred_np[:, i]))
        except ValueError:
            AUROCs.append(0.0)
        try:
            Accus.append(accuracy_score(y_true, y_pred))
        except ValueError:
            Accus.append(0.0)
        Senss.append(_binary_sensitivity(y_true, y_pred))
        Specs.append(_binary_specificity(y_true, y_pred))
        try:
            Pre.append(precision_score(y_true, y_pred, zero_division=0))
        except ValueError:
            Pre.append(0.0)
        try:
            F1.append(f1_score(y_true, y_pred, zero_division=0))
        except ValueError:
            F1.append(0.0)

    return AUROCs, Accus, Senss, Specs, Pre, F1

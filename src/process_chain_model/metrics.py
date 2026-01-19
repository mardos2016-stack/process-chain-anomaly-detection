from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute model quality metrics.

    Labels convention:
        1  = normal
        -1 = anomaly

    Primary metric: MCC.
    """
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)

    metrics: Dict[str, float] = {}

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)

    metrics["mcc"] = float(matthews_corrcoef(y_true_binary, y_pred_binary))
    metrics["precision"] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    metrics["recall"] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    denom = (tp + tn + fp + fn)
    metrics["accuracy"] = float((tp + tn) / denom) if denom > 0 else 0.0

    denom_fpr = (fp + tn)
    metrics["fpr"] = float(fp / denom_fpr) if denom_fpr > 0 else 0.0

    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_scores))
            metrics["pr_auc"] = float(average_precision_score(y_true_binary, y_scores))
        except Exception:
            # If scores are degenerate (e.g., all equal), AUC may fail
            pass

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print metrics."""
    print("\n" + "=" * 70)
    print("МЕТРИКИ МОДЕЛИ")
    print("=" * 70)

    print("\n Confusion Matrix:")
    print(
        f"   TP: {metrics.get('true_positives', 0):>5} | FP: {metrics.get('false_positives', 0):>5}"
    )
    print(
        f"   FN: {metrics.get('false_negatives', 0):>5} | TN: {metrics.get('true_negatives', 0):>5}"
    )

    print("\n Основные метрики:")
    print(
        f"   MCC (Matthews Correlation):  {metrics.get('mcc', 0):>7.4f}   ПЕРВИЧНАЯ"
    )
    print(f"   F1-Score:                    {metrics.get('f1_score', 0):>7.4f}")
    print(f"   Precision:                   {metrics.get('precision', 0):>7.4f}")
    print(f"   Recall:                      {metrics.get('recall', 0):>7.4f}")

    if "roc_auc" in metrics:
        print("\n AUC метрики:")
        print(f"   ROC-AUC:                     {metrics.get('roc_auc', 0):>7.4f}")
        print(f"   PR-AUC:                      {metrics.get('pr_auc', 0):>7.4f}")

    print("\n Дополнительно:")
    print(f"   Accuracy:                    {metrics.get('accuracy', 0):>7.4f}")
    print(f"   False Positive Rate:         {metrics.get('fpr', 0):>7.4f}")
    print("=" * 70 + "\n")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(labels: dict, probs: dict, threshold=0.5) -> dict:
    """Per-task detection metrics over labelled samples. Positive class (1) = malicious/fake."""
    results = {}
    for task, task_labels in labels.items():
        y = np.asarray(task_labels)
        p = np.asarray(probs[task])
        keep = y >= 0
        if not keep.any():
            results[task] = None
            continue
        y, p = y[keep], p[keep]
        pred = (p >= threshold).astype(int)
        results[task] = {
            "n": int(keep.sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            # False negatives are the costly error here: an attack that got through.
            "false_negative_rate": float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }
    return results

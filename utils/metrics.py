from typing import Sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def classification_report(y_true: Sequence[int], y_pred: Sequence[int]):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"precision": float(p), "recall": float(r), "f1": float(f)}

"""Classifier metrics for digits 1-9."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def expected_calibration_error(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1) + 1
    correct = (prediction == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < (hi if hi < 1 else hi + 1e-12))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_report(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = probs.argmax(axis=1) + 1
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "ece": expected_calibration_error(probs, labels),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=list(range(1, 10)))),
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(1, 10))).tolist(),
    }

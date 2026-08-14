"""开放集识别与未知聚类指标。"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
)


def compute_oscr(
    known_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    known_labels: np.ndarray,
) -> float:
    known_scores = np.max(known_probabilities, axis=1)
    unknown_scores = np.max(unknown_probabilities, axis=1)
    known_correct = np.argmax(known_probabilities, axis=1) == known_labels
    thresholds = np.r_[
        np.inf,
        np.sort(np.unique(np.r_[known_scores, unknown_scores]))[::-1],
        -np.inf,
    ]
    false_positive_rate = np.array(
        [np.mean(unknown_scores >= threshold) for threshold in thresholds]
    )
    correct_classification_rate = np.array(
        [
            np.mean(known_correct & (known_scores >= threshold))
            for threshold in thresholds
        ]
    )
    order = np.argsort(false_positive_rate, kind="stable")
    return float(
        np.trapz(
            correct_classification_rate[order], false_positive_rate[order]
        )
    )


def open_set_metrics(
    known_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    known_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    known_scores = np.max(known_probabilities, axis=1)
    unknown_scores = np.max(unknown_probabilities, axis=1)
    known_predictions = np.argmax(known_probabilities, axis=1)
    closed_accuracy = float(np.mean(known_predictions == known_labels))
    known_acceptance = float(np.mean(known_scores >= threshold))
    unknown_rejection = float(np.mean(unknown_scores < threshold))
    labels = np.r_[
        np.ones_like(known_scores, dtype=np.int64),
        np.zeros_like(unknown_scores, dtype=np.int64),
    ]
    scores = np.r_[known_scores, unknown_scores]
    return {
        "known_closed_set_accuracy": closed_accuracy,
        "known_acceptance_rate": known_acceptance,
        "unknown_rejection_rate": unknown_rejection,
        "open_binary_balanced_accuracy": 0.5 * (known_acceptance + unknown_rejection),
        "auroc": float(roc_auc_score(labels, scores)),
        "oscr": compute_oscr(
            known_probabilities, unknown_probabilities, known_labels
        ),
        "threshold": float(threshold),
    }


def purity_score(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    matrix = confusion_matrix(true_labels, cluster_labels)
    return float(np.sum(np.max(matrix, axis=0)) / np.sum(matrix))


def cluster_accuracy(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    size = int(max(true_labels.max(), cluster_labels.max()) + 1)
    matrix = np.zeros((size, size), dtype=np.int64)
    for predicted, truth in zip(cluster_labels, true_labels):
        matrix[predicted, truth] += 1
    rows, columns = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[rows, columns].sum() / len(true_labels))


def clustering_metrics(
    true_labels: np.ndarray, cluster_labels: np.ndarray
) -> Dict[str, float]:
    return {
        "nmi": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "ari": float(adjusted_rand_score(true_labels, cluster_labels)),
        "purity": purity_score(true_labels, cluster_labels),
        "cluster_accuracy": cluster_accuracy(true_labels, cluster_labels),
    }


def roc_points(
    known_scores: np.ndarray, unknown_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.r_[
        np.ones_like(known_scores, dtype=np.int64),
        np.zeros_like(unknown_scores, dtype=np.int64),
    ]
    scores = np.r_[known_scores, unknown_scores]
    false_positive, true_positive, _ = roc_curve(labels, scores)
    return false_positive, true_positive


"""
Confidence Metrics & Gating
===========================
Calculates token-level and span-level confidence metrics for early commitment.
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ConfidenceReport:
    average: float
    minimum: float
    maximum: float
    harmonic_mean: float
    entropy_weighted: float
    all_above_threshold: bool
    is_confident: bool


def calculate_confidence(
    token_confidences: List[float],
    threshold: float = 0.90,
    min_penalty_weight: float = 0.30
) -> ConfidenceReport:
    """
    Evaluates confidence over a tool-call token span.
    Uses a hybrid metric: (1 - w) * mean + w * min to ensure no single critical token is uncertain.
    """
    if not token_confidences:
        return ConfidenceReport(
            average=0.0,
            minimum=0.0,
            maximum=0.0,
            harmonic_mean=0.0,
            entropy_weighted=0.0,
            all_above_threshold=False,
            is_confident=False
        )

    avg = sum(token_confidences) / len(token_confidences)
    minimum = min(token_confidences)
    maximum = max(token_confidences)

    # Harmonic mean strongly penalizes any low outlier
    harmonic = len(token_confidences) / sum(1.0 / max(1e-4, c) for c in token_confidences)

    # Weighted confidence score
    hybrid_score = ((1.0 - min_penalty_weight) * avg) + (min_penalty_weight * minimum)

    # Entropy weighted metric
    # Higher confidence tokens get higher weight
    weights = [c ** 2 for c in token_confidences]
    weight_sum = sum(weights)
    entropy_weighted = sum(c * w for c, w in zip(token_confidences, weights)) / max(1e-6, weight_sum)

    is_confident = (avg >= threshold) and (minimum >= max(0.50, threshold - 0.20))

    return ConfidenceReport(
        average=avg,
        minimum=minimum,
        maximum=maximum,
        harmonic_mean=harmonic,
        entropy_weighted=entropy_weighted,
        all_above_threshold=minimum >= threshold,
        is_confident=is_confident
    )


def is_confidence_gated(
    token_confidences: List[float],
    threshold: float = 0.90
) -> bool:
    """Convenience boolean gate check."""
    report = calculate_confidence(token_confidences, threshold=threshold)
    return report.is_confident
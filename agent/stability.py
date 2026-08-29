"""
Stability Metrics & Gating
==========================
Tracks token equality and multi-step stability across diffusion denoising iterations.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StabilityReport:
    ratio: float
    stable_count: int
    total_tokens: int
    consecutive_stable_steps: int
    is_stable: bool


def calculate_token_stability(
    previous_tokens: List[str],
    current_tokens: List[str]
) -> List[bool]:
    """Returns boolean flag per token indicating equality between successive steps."""
    length = min(len(previous_tokens), len(current_tokens))
    stability = [previous_tokens[i] == current_tokens[i] for i in range(length)]
    
    # If current tokens has more elements, remainder are considered not stable
    if len(current_tokens) > length:
        stability.extend([False] * (len(current_tokens) - length))
        
    return stability


def token_stability_ratio(
    previous_tokens: List[str],
    current_tokens: List[str]
) -> float:
    """Calculates ratio of identical tokens between step t-1 and step t."""
    if not current_tokens:
        return 0.0
    if not previous_tokens:
        return 0.0

    flags = calculate_token_stability(previous_tokens, current_tokens)
    return sum(flags) / len(current_tokens)


def count_consecutive_stable_steps(
    token_history: List[List[str]]
) -> int:
    """
    Counts how many consecutive recent steps have had identical token sequences.
    """
    if len(token_history) < 2:
        return 1 if token_history else 0

    consecutive = 1
    current = token_history[-1]

    for prev in reversed(token_history[:-1]):
        if prev == current:
            consecutive += 1
        else:
            break

    return consecutive


def evaluate_stability(
    token_history: List[List[str]],
    threshold: float = 0.90,
    min_consecutive_steps: int = 1
) -> StabilityReport:
    """
    Evaluates stability of current token sequence against history.
    """
    if not token_history:
        return StabilityReport(
            ratio=0.0,
            stable_count=0,
            total_tokens=0,
            consecutive_stable_steps=0,
            is_stable=False
        )

    current_tokens = token_history[-1]
    total_tokens = len(current_tokens)

    if len(token_history) == 1:
        return StabilityReport(
            ratio=0.0,
            stable_count=0,
            total_tokens=total_tokens,
            consecutive_stable_steps=1,
            is_stable=False
        )

    previous_tokens = token_history[-2]
    flags = calculate_token_stability(previous_tokens, current_tokens)
    stable_count = sum(flags)
    ratio = stable_count / total_tokens if total_tokens > 0 else 0.0
    consecutive = count_consecutive_stable_steps(token_history)

    is_stable = (ratio >= threshold) and (consecutive >= min_consecutive_steps)

    return StabilityReport(
        ratio=ratio,
        stable_count=stable_count,
        total_tokens=total_tokens,
        consecutive_stable_steps=consecutive,
        is_stable=is_stable
    )
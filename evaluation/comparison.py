"""
Baseline vs DiffAgent Comparator
================================
"""

from typing import Dict, Any
from evaluation.metrics import calculate_metrics


def compare(baseline: Dict[str, Any], gated: Dict[str, Any]) -> Dict[str, Any]:
    """Generates comparative analysis between baseline and gated execution."""
    metrics = calculate_metrics(baseline, gated)
    
    is_winner = metrics["early_execution"] and metrics["routing_correct"]
    verdict = (
        f"DiffAgent fired {metrics['steps_saved']} steps earlier, "
        f"saving {metrics['step_savings_percent']}% of diffusion denoising steps "
        f"({metrics['time_saved_ms']}ms faster)."
        if is_winner
        else "Completed at full denoising steps."
    )

    return {
        "metrics": metrics,
        "winner": "DiffAgent ⚡" if is_winner else "Baseline",
        "verdict": verdict
    }
"""
DiffAgent Evaluation Metrics
============================
Computes quantitative performance metrics comparing Baseline dLLM vs DiffAgent Confidence-Gated dLLM.
"""

from typing import Dict, Any


def calculate_metrics(
    baseline: Dict[str, Any],
    gated: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes comparative step savings, latency reduction, and routing fidelity.
    """
    baseline_step = baseline.get("execution_step", 10)
    gated_step = gated.get("execution_step", 10)
    total_steps = baseline.get("total_steps", 10)

    baseline_latency = baseline.get("total_latency", 0.0)
    gated_latency = gated.get("total_latency", 0.0)

    conf = gated.get("execution_confidence", 0.0)
    stab = gated.get("execution_stability", 0.0)
    early_execution = gated.get("early_execution", False)

    baseline_tool = baseline.get("tool", "")
    gated_tool = gated.get("tool", "")
    routing_correct = (baseline_tool == gated_tool)

    # Step savings percentage
    step_savings = 0.0
    if baseline_step > 0:
        step_savings = max(0.0, ((baseline_step - gated_step) / baseline_step) * 100.0)

    # Latency savings percentage
    latency_savings = 0.0
    if baseline_latency > 0:
        latency_savings = max(0.0, ((baseline_latency - gated_latency) / baseline_latency) * 100.0)

    # Time saved in milliseconds
    time_saved_ms = max(0.0, (baseline_latency - gated_latency) * 1000.0)

    return {
        "baseline_execution_step": baseline_step,
        "gated_execution_step": gated_step,
        "total_steps": total_steps,
        "step_savings_percent": round(step_savings, 2),
        "steps_saved": max(0, baseline_step - gated_step),
        "baseline_latency": round(baseline_latency, 4),
        "gated_latency": round(gated_latency, 4),
        "latency_savings_percent": round(latency_savings, 2),
        "time_saved_ms": round(time_saved_ms, 1),
        "execution_confidence": round(conf, 4),
        "execution_stability": round(stab, 4),
        "early_execution": early_execution,
        "baseline_tool": baseline_tool,
        "gated_tool": gated_tool,
        "routing_correct": routing_correct
    }
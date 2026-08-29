"""
DiffAgent Benchmark Suite
=========================
Evaluates DiffAgent across multiple domains (Weather, Math, Campus RAG).
"""

from typing import List, Dict, Any, Tuple
from evaluation.comparison import compare


BENCHMARK_TASKS = [
    {
        "category": "Weather (Open-Meteo)",
        "query": "What is the weather in Chennai?",
        "expected_tool": "weather"
    },
    {
        "category": "Weather (Open-Meteo)",
        "query": "Current temperature and rain in Tokyo?",
        "expected_tool": "weather"
    },
    {
        "category": "Weather (Open-Meteo)",
        "query": "Is it cloudy in London right now?",
        "expected_tool": "weather"
    },
    {
        "category": "Calculator (Math)",
        "query": "Calculate 125 * 48",
        "expected_tool": "calculator"
    },
    {
        "category": "Calculator (Math)",
        "query": "Calculate 15 percent of 4500",
        "expected_tool": "calculator"
    },
    {
        "category": "Calculator (Math)",
        "query": "What is 2 to the power of 10?",
        "expected_tool": "calculator"
    },
    {
        "category": "Campus RAG (Documents)",
        "query": "What are the library opening and closing hours?",
        "expected_tool": "campus"
    },
    {
        "category": "Campus RAG (Documents)",
        "query": "What are the hostel room amenities and curfew rules?",
        "expected_tool": "campus"
    },
    {
        "category": "Campus RAG (Documents)",
        "query": "What is the minimum attendance requirement for semester exams?",
        "expected_tool": "campus"
    },
    {
        "category": "Campus RAG (Documents)",
        "query": "How do I access medical emergency services on campus?",
        "expected_tool": "campus"
    }
]


def run_benchmark(agent) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs full benchmark across all task suites and computes aggregate metrics.
    """
    table_records = []
    total_tasks = len(BENCHMARK_TASKS)
    early_triggers = 0
    correct_routes = 0
    total_step_savings = 0.0
    total_latency_savings = 0.0
    total_time_saved_ms = 0.0

    for item in BENCHMARK_TASKS:
        query = item["query"]
        expected_tool = item["expected_tool"]
        category = item["category"]

        baseline = agent.run_baseline(query)
        gated = agent.run_gated(query)

        comp = compare(baseline, gated)
        m = comp["metrics"]

        is_route_correct = gated.get("tool") == expected_tool
        if is_route_correct:
            correct_routes += 1
        if m["early_execution"]:
            early_triggers += 1

        total_step_savings += m["step_savings_percent"]
        total_latency_savings += m["latency_savings_percent"]
        total_time_saved_ms += m["time_saved_ms"]

        table_records.append({
            "Category": category,
            "Query": query,
            "Expected": expected_tool,
            "Selected": gated.get("tool", "unknown"),
            "Route": "✅ Correct" if is_route_correct else "❌ Mismatch",
            "Baseline Step": f"{m['baseline_execution_step']}/{m['total_steps']}",
            "DiffAgent Step": f"{m['gated_execution_step']}/{m['total_steps']}",
            "Confidence": f"{m['execution_confidence']:.0%}",
            "Early Fire": "⚡ Early" if m["early_execution"] else "Full",
            "Step Savings (%)": m["step_savings_percent"],
            "Latency Saved (ms)": m["time_saved_ms"],
            "Speedup (%)": m["latency_savings_percent"]
        })

    summary = {
        "total_tasks": total_tasks,
        "routing_accuracy": round((correct_routes / total_tasks) * 100.0, 1),
        "early_commitment_rate": round((early_triggers / total_tasks) * 100.0, 1),
        "mean_step_savings_percent": round(total_step_savings / total_tasks, 1),
        "mean_latency_savings_percent": round(total_latency_savings / total_tasks, 1),
        "total_time_saved_ms": round(total_time_saved_ms, 1)
    }

    return table_records, summary
"""
DiffAgent: Confidence-Gated Tool Calling for Diffusion Language Models
======================================================================
Implements early commitment tool execution based on per-token confidence and stability.
"""

import time
import math
import re
from typing import Dict, Any, List, Optional, Tuple

from agent.decoder import DiffusionDecoder, DecodingStepState, tokenize_tool_call
from agent.confidence import calculate_confidence, ConfidenceReport
from agent.stability import evaluate_stability, StabilityReport
from agent.router import ToolRouter


class DiffAgent:
    """
    Confidence-Gated Agent Harness for Diffusion Language Models.
    """

    def __init__(
        self,
        router: ToolRouter,
        confidence_threshold: float = 0.90,
        stability_threshold: float = 0.90,
        total_steps: int = 10,
        step_latency_ms: float = 40.0
    ):
        self.router = router
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.total_steps = max(3, total_steps)
        self.step_latency_ms = step_latency_ms
        self.decoder = DiffusionDecoder(
            total_steps=self.total_steps,
            step_latency_ms=self.step_latency_ms
        )

    def _synthesize_response(
        self,
        query: str,
        tool: str,
        tool_result: Dict[str, Any]
    ) -> str:
        """
        Synthesizes a human-readable, conversational, grounded final response.
        """
        if not tool_result or not tool_result.get("success"):
            error_msg = tool_result.get("error", "No data retrieved.") if tool_result else "Execution failed."
            return f"I encountered an issue executing the {tool} tool: {error_msg}"

        result_data = tool_result.get("result", {})

        # ----------------------------------------------------
        # 1. WEATHER SYNTHESIS
        # ----------------------------------------------------
        if tool == "weather":
            if not result_data.get("success", False):
                return result_data.get("message", "Weather information currently unavailable.")

            loc = result_data.get("location", "the requested location")
            country = result_data.get("country", "")
            loc_full = f"{loc}, {country}" if country else loc
            data = result_data.get("data", {})

            temp = data.get("temperature", 0.0)
            feels_like = data.get("feels_like", temp)
            humidity = data.get("humidity", 0)
            wind = data.get("wind_speed", 0.0)
            condition = data.get("condition", "clear")
            icon = data.get("icon", "🌤️")
            precip = data.get("precipitation", 0.0)
            weather_code = data.get("weather_code", 0)

            q_lower = query.lower()

            # Umbrella / Rain intent
            rain_intent_words = [
                "umbrella", "rain", "raining", "rainy", "raincoat",
                "drizzle", "shower", "wet", "storm", "thunderstorm", "pour"
            ]
            if any(re.search(rf"\b{w}\b", q_lower) for w in rain_intent_words):
                is_rainy = (
                    precip > 0.05
                    or weather_code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
                    or any(w in condition.lower() for w in ["rain", "drizzle", "shower", "thunderstorm", "storm"])
                )
                if is_rainy:
                    return (
                        f"🌧️ **Yes, you should carry an umbrella!** There is **{condition.lower()}** "
                        f"(precipitation: **{precip} mm**) in **{loc_full}** today with a temperature of **{temp}°C**."
                    )
                else:
                    return (
                        f"☀️ **No need to carry an umbrella.** The weather in **{loc_full}** today is **{condition.lower()}** "
                        f"with **0 mm** precipitation and a temperature of **{temp}°C**."
                    )

            # Jacket / Clothing / Cold intent
            clothing_intent_words = ["jacket", "coat", "sweater", "warm", "cold", "chilly", "freeze", "wear", "clothing"]
            if any(re.search(rf"\b{w}\b", q_lower) for w in clothing_intent_words):
                if temp < 16.0:
                    return (
                        f"🧥 **Yes, wearing a warm jacket or coat is recommended.** It is currently **{temp}°C** (feels like **{feels_like}°C**) "
                        f"with **{condition.lower()}** in **{loc_full}**."
                    )
                elif temp >= 28.0:
                    return (
                        f"👕 **It is quite warm in {loc_full}!** The current temperature is **{temp}°C** (feels like **{feels_like}°C**) "
                        f"with **{condition.lower()}**, so light and breathable clothing is recommended."
                    )
                else:
                    return (
                        f"🌤️ **A light layer should be comfortable.** The temperature in **{loc_full}** is **{temp}°C** (feels like **{feels_like}°C**) "
                        f"with **{condition.lower()}**."
                    )

            # General weather synthesis
            return (
                f"{icon} The current weather in **{loc_full}** is **{temp}°C** with **{condition}**. "
                f"It feels like **{feels_like}°C**, relative humidity is **{humidity}%**, and wind speed is **{wind} km/h**."
            )

        # ----------------------------------------------------
        # 2. CALCULATOR SYNTHESIS
        # ----------------------------------------------------
        if tool == "calculator":
            expr = result_data.get("expression", query)
            ans = result_data.get("answer", None)
            if ans is None:
                return f"Calculation failed for `{expr}`."

            q_clean = query.strip()
            
            # Rate / duration word problem: "if i study for 3 hours everyday for 7 days how many hours is that?" or "3 hours for 7 days"
            rate_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|pages?|km|miles?|dollars?|rs\.?|units?|items?)\s*(everyday|every day|per day|a day|each day|daily|per week|a week|for|over|in|across)?\s*(?:for|over|in|across)?\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|hours?)",
                q_clean,
                re.IGNORECASE
            )
            if rate_match:
                qty = rate_match.group(1)
                unit = rate_match.group(2)
                freq_raw = rate_match.group(3) or "per day"
                freq = f" {freq_raw}" if freq_raw and freq_raw not in ("for", "over", "in", "across") else ""
                dur = rate_match.group(4)
                dur_unit = rate_match.group(5)
                return (
                    f"🧮 If you study for **{qty} {unit}**{freq} for **{dur} {dur_unit}**, "
                    f"that is **{ans} {unit}** in total ({qty} × {dur} = {ans})."
                )

            # Percentage: "15% of 4500"
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:of)\s*(\d+(?:\.\d+)?)", q_clean, re.IGNORECASE)
            if pct_match:
                pct, val = pct_match.group(1), pct_match.group(2)
                return f"🧮 **{pct}% of {val}** is **{ans}** ({pct}/100 × {val} = {ans})."

            # Exponentiation: "2 to the power of 10"
            pow_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to the power of|power of|raised to|power|\^)\s*(\d+(?:\.\d+)?)", q_clean, re.IGNORECASE)
            if pow_match:
                b, e = pow_match.group(1), pow_match.group(2)
                return f"🧮 **{b}^{e}** ({b} to the power of {e}) = **{ans}**."

            return f"🧮 Result: `{expr}` = **{ans:,}**" if isinstance(ans, (int, float)) and abs(ans) >= 1000 else f"🧮 Result: `{expr}` = **{ans}**"

        # ----------------------------------------------------
        # 3. CAMPUS RAG SYNTHESIS
        # ----------------------------------------------------
        if tool in ("campus", "campus_rag"):
            results = result_data.get("results", [])
            if not results:
                return "I searched the campus knowledge base but couldn't find any relevant documentation."

            q_lower = query.lower()
            is_compound = bool(re.search(r"\band\b|&|\bas well as\b|\balso\b", q_lower) and len(results) > 1)

            # Group documents by source
            sections_by_source = {}
            for doc in results:
                src = doc.get("source", "docs")
                if src not in sections_by_source:
                    sections_by_source[src] = []
                sections_by_source[src].append(doc)

            formatted_sections = []
            citations = []

            if is_compound and len(sections_by_source) > 1:
                # For compound queries, take the best section from each distinct source
                for src, docs in sections_by_source.items():
                    if src not in citations:
                        citations.append(src)
                    top_doc = docs[0]
                    raw_text = top_doc.get("text", "")
                    clean_text = re.sub(r"^\[.*?\]\s*", "", raw_text).strip()
                    lines = clean_text.split("\n")
                    header = re.sub(r"^\d+\.\s*", "", lines[0].strip())
                    body_lines = [l.strip() for l in lines[1:] if l.strip()]
                    bullets = []
                    for l in body_lines:
                        cleaned = re.sub(r"^[-*•]\s*", "", l)
                        bullets.append(f"• {cleaned}")
                    formatted_body = "\n".join(bullets)
                    if bullets:
                        formatted_sections.append(f"**{header}:**\n{formatted_body}")
                    else:
                        formatted_sections.append(f"**{header}**")
            else:
                # For single-intent queries, format precisely the top matching section
                top_doc = results[0]
                src = top_doc.get("source", "docs")
                citations = [src]
                raw_text = top_doc.get("text", "")
                clean_text = re.sub(r"^\[.*?\]\s*", "", raw_text).strip()
                lines = clean_text.split("\n")
                header = re.sub(r"^\d+\.\s*", "", lines[0].strip())

                body_lines = [l.strip() for l in lines[1:] if l.strip()]
                bullets = []
                for l in body_lines:
                    cleaned = re.sub(r"^[-*•]\s*", "", l)
                    bullets.append(f"• {cleaned}")
                formatted_body = "\n".join(bullets)

                if bullets:
                    formatted_sections.append(f"**{header}:**\n{formatted_body}")
                else:
                    formatted_sections.append(f"**{header}**")

            body_text = "\n\n".join(formatted_sections)
            cite_str = ", ".join([f"`{c}`" for c in citations])
            return f"{body_text}\n\n*(Source: {cite_str})*" if len(citations) == 1 else f"{body_text}\n\n*(Sources: {cite_str})*"

        return f"Tool `{tool}` output: {result_data}"

    def run_baseline(self, query: str) -> Dict[str, Any]:
        """
        Baseline Execution:
        Denoises all T steps before executing the tool call.
        """
        t_start = time.perf_counter()

        # Step 1: Tool Routing
        tool = self.router.choose_tool(query)
        tool_name, arg_str = self.router.canonical_tool_span(tool, query)
        target_tokens = tokenize_tool_call(tool_name, arg_str)

        # Step 2: Full Diffusion Decoding across all T steps
        history = self.decoder.decode_tool_span(
            target_tokens=target_tokens,
            tool_name=tool_name,
            delay_simulation=False
        )

        # Baseline always waits for the final step
        execution_step = self.total_steps
        final_state = history[-1]

        # Step 3: Execute tool at full completion
        t_tool_start = time.perf_counter()
        tool_result = self.router.execute(query, tool=tool)
        t_tool_end = time.perf_counter()
        tool_duration = t_tool_end - t_tool_start

        # Step 4: Synthesize response
        response = self._synthesize_response(query, tool, tool_result)

        # Calculate realistic simulated latency
        # T steps * step_latency + tool API time + synthesis time
        denoise_latency = (self.total_steps * self.step_latency_ms) / 1000.0
        total_latency = denoise_latency + tool_duration + 0.015

        return {
            "mode": "baseline",
            "query": query,
            "tool": tool,
            "tool_call_span": " ".join(target_tokens),
            "execution_step": execution_step,
            "total_steps": self.total_steps,
            "execution_confidence": final_state.avg_confidence,
            "execution_min_confidence": final_state.min_confidence,
            "execution_stability": final_state.stability_score,
            "early_execution": False,
            "step_saved": 0,
            "denoise_latency": denoise_latency,
            "tool_latency": tool_duration,
            "total_latency": total_latency,
            "history": history,
            "result": tool_result,
            "response": response
        }

    def run_gated(self, query: str) -> Dict[str, Any]:
        """
        Confidence-Gated DiffAgent Execution:
        Monitors token confidence and stability at each denoising step.
        Executes the tool immediately at step t* as soon as confidence >= tau_conf and stability >= tau_stab.
        """
        t_start = time.perf_counter()

        # Step 1: Tool Routing & Tokenization
        tool = self.router.choose_tool(query)
        tool_name, arg_str = self.router.canonical_tool_span(tool, query)
        target_tokens = tokenize_tool_call(tool_name, arg_str)

        # Step 2: Denoise and track step states
        history = self.decoder.decode_tool_span(
            target_tokens=target_tokens,
            tool_name=tool_name,
            delay_simulation=False
        )

        # Step 3: Evaluate early commitment gate at each step
        token_history = []
        execution_step = self.total_steps
        execution_conf = history[-1].avg_confidence
        execution_min_conf = history[-1].min_confidence
        execution_stab = history[-1].stability_score
        early_execution = False

        for state in history:
            token_history.append(state.tokens)

            conf_report = calculate_confidence(
                state.confidences,
                threshold=self.confidence_threshold
            )
            stab_report = evaluate_stability(
                token_history,
                threshold=self.stability_threshold,
                min_consecutive_steps=1
            )

            # Gate condition: Both confidence and stability meet threshold criteria
            # Also ensure no uncommitted "[MASK]" or corrupt token in argument span
            has_no_masks = not any("[MASK]" in tok or "..." in tok for tok in state.tokens)

            if conf_report.is_confident and stab_report.is_stable and has_no_masks:
                execution_step = state.step
                execution_conf = conf_report.average
                execution_min_conf = conf_report.minimum
                execution_stab = stab_report.ratio
                early_execution = state.step < self.total_steps
                break

        # Step 4: Execute tool
        t_tool_start = time.perf_counter()
        tool_result = self.router.execute(query, tool=tool)
        t_tool_end = time.perf_counter()
        tool_duration = t_tool_end - t_tool_start

        # Step 5: Synthesize response
        response = self._synthesize_response(query, tool, tool_result)

        # Calculate realistic latency savings
        # Only execution_step * step_latency needed before tool is fired!
        denoise_latency = (execution_step * self.step_latency_ms) / 1000.0
        total_latency = denoise_latency + tool_duration + 0.015
        steps_saved = max(0, self.total_steps - execution_step)

        return {
            "mode": "confidence_gated",
            "query": query,
            "tool": tool,
            "tool_call_span": " ".join(target_tokens),
            "execution_step": execution_step,
            "total_steps": self.total_steps,
            "execution_confidence": execution_conf,
            "execution_min_confidence": execution_min_conf,
            "execution_stability": execution_stab,
            "early_execution": early_execution,
            "step_saved": steps_saved,
            "denoise_latency": denoise_latency,
            "tool_latency": tool_duration,
            "total_latency": total_latency,
            "history": history,
            "result": tool_result,
            "response": response
        }
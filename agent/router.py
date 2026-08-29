"""
DiffAgent Tool Router
=====================
Classifies user intent, generates canonical tool-call representations for diffusion decoding,
and dispatches execution to the registered tool functions.
"""

import re
from typing import Dict, Any, Callable, Tuple
from tools.weather import extract_location
from tools.calculator import extract_math_expression


class ToolRouter:
    """
    Autonomous tool router and argument binder.
    """

    def __init__(self):
        self.tools: Dict[str, Callable[[str], Any]] = {}

    def register(self, name: str, function: Callable[[str], Any]):
        """Registers a tool name with its execution handler."""
        self.tools[name] = function

    def choose_tool(self, query: str) -> str:
        """
        Classifies user query intent to select the appropriate tool.
        """
        query_lower = query.lower().strip()

        # ----------------------------------------------------
        # 1. WEATHER INTENT
        # ----------------------------------------------------
        weather_keywords = [
            "weather", "temperature", "temp", "rain", "raining", "rainy", "humidity",
            "forecast", "wind", "hot", "cold", "umbrella", "raincoat", "climate", "degree",
            "sunny", "cloudy", "storm", "snow", "celsius", "fahrenheit", "drizzle",
            "shower", "jacket", "coat", "chilly", "warm"
        ]
        if any(re.search(rf"\b{kw}\b", query_lower) for kw in weather_keywords):
            return "weather"

        # ----------------------------------------------------
        # 2. CALCULATOR INTENT
        # ----------------------------------------------------
        calculator_keywords = [
            "calculate", "calculator", "multiply", "multiplied", "divide",
            "divided", "addition", "add", "subtract", "percentage", "percent",
            "how much is", "plus", "minus", "times", "power of", "sqrt",
            "product of", "sum of", "difference between"
        ]
        has_calc_word = any(re.search(rf"\b{kw}\b", query_lower) for kw in calculator_keywords)
        has_math_symbols = bool(re.search(r"\d+\s*[-+*/^%]\s*\d+", query_lower))
        has_rate_word_problem = bool(re.search(r"\b\d+\s*(?:hours?|hrs?|pages?|km|miles?|dollars?|rs\.?|units?|items?|times?)\s*(?:everyday|every day|per day|a day|each day|daily|per week|a week|each week)\b", query_lower))
        has_how_many_math = bool(re.search(r"\bhow many (?:hours|days|weeks|items|pages|total|left|much)\b", query_lower) and re.search(r"\d+", query_lower))

        if has_calc_word or has_math_symbols or has_rate_word_problem or has_how_many_math:
            return "calculator"

        # ----------------------------------------------------
        # 3. CAMPUS RAG INTENT
        # ----------------------------------------------------
        campus_keywords = [
            "library", "hostel", "attendance", "academic", "semester", "exam",
            "campus", "student", "faculty", "course", "facility", "facilities",
            "service", "services", "transport", "bus", "medical", "doctor",
            "health", "mess", "food", "curfew", "grade", "gpa", "fee", "admission"
        ]
        if any(re.search(rf"\b{kw}\b", query_lower) for kw in campus_keywords):
            return "campus"

        # Default fallback to campus RAG
        return "campus"

    def canonical_tool_span(self, tool: str, query: str) -> Tuple[str, str]:
        """
        Generates the canonical tool call representation (tool_name, argument_string).
        """
        if tool == "weather":
            loc = extract_location(query)
            arg_str = f"location = '{loc}'"
            return "weather", arg_str

        elif tool == "calculator":
            expr = extract_math_expression(query)
            arg_str = f"expression = '{expr}'"
            return "calculator", arg_str

        elif tool == "campus":
            # Clean campus query
            clean_q = query.replace("'", "").replace('"', '').strip()
            arg_str = f"query = '{clean_q}'"
            return "campus", arg_str

        else:
            clean_q = query.replace("'", "").replace('"', '').strip()
            return tool, f"query = '{clean_q}'"

    def execute(self, query: str, tool: str = None) -> Dict[str, Any]:
        """
        Executes the appropriate tool function.
        """
        selected_tool = tool if tool else self.choose_tool(query)

        if selected_tool not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{selected_tool}' is not registered.",
                "tool": selected_tool,
                "result": None
            }

        try:
            result = self.tools[selected_tool](query)
            return {
                "success": True,
                "tool": selected_tool,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": selected_tool,
                "result": None
            }
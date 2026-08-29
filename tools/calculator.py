"""
Safe AST-Based Calculator Tool
==============================
Safely evaluates arithmetic expressions, percentages, and math queries.
"""

import ast
import operator
import re
from typing import Dict, Any, Union

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    raise ValueError(f"Unsupported AST node: {type(node)}")


def extract_math_expression(query: str) -> str:
    """Extracts and normalizes arithmetic expression from natural language query, including word problems."""
    query_clean = query.strip()

    # 1. Check for rate/duration word problem: "3 hours everyday for 7 days", "3 hours for 7 days", "5 miles a day for 10 days"
    rate_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|pages?|km|miles?|dollars?|rs\.?|units?|items?|times?)\s*(?:everyday|every day|per day|a day|each day|daily|per week|a week|each week|for|over|in|across)?\s*(?:for|over|in|across)?\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?|hours?|sessions?)",
        query_clean,
        re.IGNORECASE
    )
    if rate_match:
        val1 = rate_match.group(1)
        val2 = rate_match.group(2)
        return f"{val1} * {val2}"

    # 2. Check for percentage pattern: "15 percent of 4500" or "15% of 4500"
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:of)\s*(\d+(?:\.\d+)?)", query_clean, re.IGNORECASE)
    if pct_match:
        pct = pct_match.group(1)
        val = pct_match.group(2)
        return f"({pct} / 100) * {val}"

    # 3. Check for exponentiation: "2 to the power of 10", "5 power 3"
    pow_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to the power of|power of|raised to|power|\^)\s*(\d+(?:\.\d+)?)", query_clean, re.IGNORECASE)
    if pow_match:
        base = pow_match.group(1)
        exp = pow_match.group(2)
        return f"{base} ** {exp}"

    # 4. Check for sum/product of X and Y: "product of 12 and 15"
    prod_match = re.search(r"\bproduct of\s+(\d+(?:\.\d+)?)\s+(?:and)\s+(\d+(?:\.\d+)?)", query_clean, re.IGNORECASE)
    if prod_match:
        return f"{prod_match.group(1)} * {prod_match.group(2)}"

    sum_match = re.search(r"\bsum of\s+(\d+(?:\.\d+)?)\s+(?:and)\s+(\d+(?:\.\d+)?)", query_clean, re.IGNORECASE)
    if sum_match:
        return f"{sum_match.group(1)} + {sum_match.group(2)}"

    # 5. Check for standard math symbols and words
    cleaned = re.sub(r"\b(calculate|what is|how much is|evaluate|compute|equal to|equals|how many hours is that|how many hours in total|how many total|\?)\b", "", query_clean, flags=re.IGNORECASE).strip()
    
    # Replace math word operators
    cleaned = re.sub(r"\bplus\b", "+", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bminus\b", "-", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btimes\b|\bmultiplied by\b", "*", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdivided by\b", "/", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bto the power of\b|\bpower\b", "**", cleaned, flags=re.IGNORECASE)

    # Find valid math characters
    math_match = re.search(r"[-+*/().\d\s^%]+", cleaned)
    if math_match:
        expr = math_match.group(0).strip().replace("^", "**")
        # Ensure it has digits and at least one arithmetic operator
        if any(c.isdigit() for c in expr):
            return expr

    return "0"


def calculate(query: str) -> Dict[str, Any]:
    """Calculates mathematical result safely using python AST."""
    expr = extract_math_expression(query)
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        
        # Round float to clean precision if needed
        if isinstance(result, float) and result.is_integer():
            formatted_answer = int(result)
        elif isinstance(result, float):
            formatted_answer = round(result, 6)
        else:
            formatted_answer = result

        return {
            "success": True,
            "tool": "calculator",
            "expression": expr,
            "answer": formatted_answer,
            "raw_query": query
        }
    except Exception as e:
        return {
            "success": False,
            "tool": "calculator",
            "expression": expr,
            "error": str(e),
            "message": f"Could not evaluate expression: {expr}"
        }
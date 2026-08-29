"""
Diffusion Decoder Simulator for dLLMs
=====================================
Models discrete diffusion language model denoising dynamics:
- Multi-step parallel token denoising (Steps 1 ... T)
- Per-token confidence trajectory (sigmoid growth with noise decay)
- Per-token Shannon entropy decay
- Step-by-step token stability and masking dynamics

Note: Ready for drop-in replacement with real dLLM model weights/sampling.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DecodingStepState:
    """Represents the internal state of the diffusion decoder at step t."""
    step: int
    tokens: List[str]
    confidences: List[float]
    entropies: List[float]
    is_stable: List[bool]
    text: str
    avg_confidence: float
    min_confidence: float
    stability_score: float
    timestamp: float


class DiffusionDecoder:
    """
    Simulates discrete diffusion denoising trajectories for structured tool-calls.
    """

    def __init__(
        self,
        total_steps: int = 10,
        step_latency_ms: float = 40.0
    ):
        self.total_steps = max(3, total_steps)
        self.step_latency_ms = step_latency_ms

    def _corrupt_token(self, token: str, step: int) -> str:
        """Simulates noisy or uncommitted token states in early diffusion steps."""
        if len(token) <= 2:
            return "[MASK]"
        
        # Variants of noise in discrete diffusion (Masking / Noisy characters)
        progress = step / self.total_steps
        if progress < 0.25:
            return "[MASK]"
        elif progress < 0.50:
            if len(token) > 4:
                return f"{token[:2]}...{token[-1]}"
            return "[MASK]"
        else:
            return token

    def _calculate_token_entropy(self, conf: float) -> float:
        """Calculates binary Shannon entropy H(p) in nats."""
        p = max(1e-6, min(1.0 - 1e-6, conf))
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def _confidence_for_token(
        self,
        token: str,
        target_token: str,
        index: int,
        step: int,
        is_tool_name: bool = False,
        is_literal_arg: bool = False
    ) -> float:
        """
        Calculates realistic confidence based on diffusion step, token role, and alignment.
        - Tool keywords (e.g. 'weather', 'calculator') stabilize fastest.
        - Core arguments stabilize next.
        - Syntactic tokens stabilize progressively.
        """
        progress = step / self.total_steps
        
        # S-curve (sigmoid) progression of confidence across diffusion steps
        # x ranges from -4 (at step 0) to +4 (at total_steps)
        x = (progress - 0.45) * 8.0
        sigmoid_val = 1.0 / (1.0 + math.exp(-x))

        if is_tool_name:
            # Tool names have high prior attention
            base = 0.55 + 0.44 * sigmoid_val
        elif is_literal_arg:
            # Arguments stabilize smoothly
            base = 0.40 + 0.58 * sigmoid_val
        else:
            # Punctuation / syntax
            base = 0.45 + 0.53 * sigmoid_val

        # If token is currently corrupted or masked
        if token == "[MASK]":
            base = 0.15 + 0.15 * progress
        elif "..." in token:
            base = 0.45 + 0.25 * progress
        elif token != target_token:
            base = 0.30 + 0.20 * progress

        # Add deterministic micro-jitter to simulate distinct token variances
        jitter = (((index * 17 + step * 31) % 11) - 5) * 0.008
        conf = base + jitter

        return max(0.05, min(0.995, conf))

    def decode_tool_span(
        self,
        target_tokens: List[str],
        tool_name: str,
        delay_simulation: bool = False
    ) -> List[DecodingStepState]:
        """
        Generates full diffusion trajectory over total_steps for the given token sequence.
        """
        history: List[DecodingStepState] = []
        previous_tokens: List[str] = []

        # Identify token roles
        token_roles = []
        for i, token in enumerate(target_tokens):
            if token.lower() == tool_name.lower():
                token_roles.append("tool")
            elif any(c.isalnum() for c in token) and token not in ("query", "expression", "location", "top_k"):
                token_roles.append("arg")
            else:
                token_roles.append("syntax")

        # Stable step thresholds for each token (when each token settles)
        # Tool names stabilize around 35-45% of total steps, args around 45-60%, syntax around 60-70%
        settle_steps = []
        for i, role in enumerate(token_roles):
            if role == "tool":
                s = max(2, int(self.total_steps * 0.35))
            elif role == "arg":
                # args settle around step 4-6
                s = max(3, int(self.total_steps * (0.45 + ((i % 3) * 0.05))))
            else:
                s = max(3, int(self.total_steps * (0.55 + ((i % 2) * 0.08))))
            settle_steps.append(min(self.total_steps - 1, s))

        for step in range(1, self.total_steps + 1):
            t0 = time.perf_counter()

            # Generate step tokens based on settle schedule
            current_tokens = []
            for i, target_tok in enumerate(target_tokens):
                if step >= settle_steps[i]:
                    current_tokens.append(target_tok)
                else:
                    current_tokens.append(self._corrupt_token(target_tok, step))

            # Calculate per-token confidences
            confidences = []
            entropies = []
            for i, tok in enumerate(current_tokens):
                is_tool = token_roles[i] == "tool"
                is_arg = token_roles[i] == "arg"
                c = self._confidence_for_token(
                    token=tok,
                    target_token=target_tokens[i],
                    index=i,
                    step=step,
                    is_tool_name=is_tool,
                    is_literal_arg=is_arg
                )
                confidences.append(c)
                entropies.append(self._calculate_token_entropy(c))

            # Stability flags (compared to previous step)
            is_stable = []
            if previous_tokens and len(previous_tokens) == len(current_tokens):
                for prev, curr in zip(previous_tokens, current_tokens):
                    is_stable.append(prev == curr)
            else:
                is_stable = [False] * len(current_tokens)

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            min_conf = min(confidences) if confidences else 0.0
            stability_score = sum(is_stable) / len(is_stable) if is_stable else 0.0
            text_rep = " ".join(current_tokens)

            state = DecodingStepState(
                step=step,
                tokens=current_tokens,
                confidences=confidences,
                entropies=entropies,
                is_stable=is_stable,
                text=text_rep,
                avg_confidence=avg_conf,
                min_confidence=min_conf,
                stability_score=stability_score,
                timestamp=time.time()
            )
            history.append(state)
            previous_tokens = list(current_tokens)

            if delay_simulation and self.step_latency_ms > 0:
                time.sleep(self.step_latency_ms / 1000.0)

        return history


def tokenize_tool_call(tool: str, argument_str: str) -> List[str]:
    """Tokenizes a canonical tool call structure e.g. `weather ( location = 'Chennai' )`."""
    formatted = f"{tool} ( {argument_str} )"
    # Clean spacing around punctuation for crisp token boundaries
    formatted = (
        formatted.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("=", " = ")
        .replace("'", " ' ")
        .replace('"', ' " ')
        .replace(",", " , ")
    )
    tokens = [t for t in formatted.split() if t.strip()]
    return tokens
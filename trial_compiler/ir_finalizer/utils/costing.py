# costing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# OpenAI API pricing (per 1M tokens).
# Reference: https://platform.openai.com/docs/pricing
# IMPORTANT: These are OpenAI list prices; if you're billed via Azure, use Azure pricing instead.

@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float = 0.0

OPENAI_PRICING: Dict[str, ModelPricing] = {
    "gpt-5": ModelPricing(input_per_1m=1.25, output_per_1m=10.00, cached_input_per_1m=0.125),
    "gpt-4.1": ModelPricing(input_per_1m=2.00, output_per_1m=8.00, cached_input_per_1m=0.50),
    "gpt-4.1-mini": ModelPricing(input_per_1m=0.40, output_per_1m=1.60, cached_input_per_1m=0.10),
}

def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> Optional[float]:
    pr = OPENAI_PRICING.get(model)
    if pr is None:
        return None

    cached = max(0, min(prompt_tokens, cached_prompt_tokens))
    uncached = max(0, prompt_tokens - cached)

    cost = 0.0
    cost += (uncached / 1_000_000.0) * pr.input_per_1m
    if cached and pr.cached_input_per_1m:
        cost += (cached / 1_000_000.0) * pr.cached_input_per_1m
    cost += (completion_tokens / 1_000_000.0) * pr.output_per_1m
    return cost

def count_tokens(text: str, model: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Best-effort local token counting.

    - Most accurate: provider returns usage.prompt_tokens / completion_tokens
    - Fallback: tiktoken locally (may differ slightly vs server tokenizer)
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None, "tiktoken_not_installed"

    # tiktoken model map may not include very new names; attempt best-effort.
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text)), getattr(enc, "name", "encoding_for_model")
    except Exception:
        # Common fallback encoding for modern OpenAI models when model mapping is missing
        try:
            enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text)), "o200k_base_fallback"
        except Exception:
            return None, "tiktoken_failed"

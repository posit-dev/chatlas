"""Formatting helpers for __repr__ methods."""

from __future__ import annotations


def format_tokens(tokens: tuple[int, int, int] | None, cost: float | None) -> str:
    """Format token usage and cost for display."""
    if tokens is None:
        return ""
    input_tokens, output_tokens, cached = tokens
    result = f"input={input_tokens}"
    if cached > 0:
        result += f"+{cached}"
    result += f" output={output_tokens}"
    if cost is not None:
        result += f" cost=${cost:.4f}"
    return result

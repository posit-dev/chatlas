from __future__ import annotations

import copy
import importlib.resources as resources
from threading import Lock
from typing import TYPE_CHECKING

import orjson

from ._logging import logger
from ._typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from ._provider import Provider


class TokenUsage(TypedDict):
    """
    Token usage for a given provider (name).
    """

    name: str
    model: str
    input: int
    output: int
    cached_input: int
    cost: float | None


class ThreadSafeTokenCounter:
    def __init__(self):
        self._lock = Lock()
        self._tokens: dict[str, TokenUsage] = {}

    def log_tokens(
        self,
        name: str,
        model: str,
        tokens: tuple[int, int, int],
        variant: str = "",
    ) -> None:
        logger.info(
            f"Provider '{name}' generated a response of {tokens[1]} tokens "
            f"from an input of {tokens[0]} tokens and {tokens[2]} cached input tokens."
        )

        with self._lock:
            if name not in self._tokens:
                self._tokens[name] = {
                    "name": name,
                    "model": model,
                    "input": tokens[0],
                    "output": tokens[1],
                    "cached_input": tokens[2],
                    "cost": get_token_cost(name, model, tokens, variant),
                }
            else:
                self._tokens[name]["input"] += tokens[0]
                self._tokens[name]["output"] += tokens[1]
                self._tokens[name]["cached_input"] += tokens[2]
                new_cost = get_token_cost(name, model, tokens, variant)
                if new_cost is not None:
                    old_cost = self._tokens[name]["cost"]
                    if old_cost is None:
                        self._tokens[name]["cost"] = new_cost
                    else:
                        self._tokens[name]["cost"] = old_cost + new_cost

    def get_usage(self) -> list[TokenUsage] | None:
        with self._lock:
            if not self._tokens:
                return None
            # Create a deep copy to avoid external modifications
            return copy.deepcopy(list(self._tokens.values()))


# Global instance
_token_counter = ThreadSafeTokenCounter()


def tokens_log(
    provider: "Provider", tokens: tuple[int, int, int], variant: str = ""
) -> None:
    """
    Log token usage for a provider in a thread-safe manner.

    Parameters
    ----------
    provider
        The provider instance
    tokens
        A tuple of (input_tokens, output_tokens, cached_tokens)
    variant
        The pricing variant (e.g., "flex", "priority"). Defaults to "" (standard pricing).
    """
    _token_counter.log_tokens(provider.name, provider.model, tokens, variant)


def tokens_reset() -> None:
    """
    Reset the token usage counter
    """
    global _token_counter  # noqa: PLW0603
    _token_counter = ThreadSafeTokenCounter()


class TokenPrice(TypedDict):
    """
    Defines the necessary information to look up pricing for a given turn.
    """

    provider: str
    """The provider name (e.g., "OpenAI", "Anthropic", etc.)"""
    model: str
    """The model name (e.g., "gpt-3.5-turbo", "claude-2", etc.)"""
    cached_input: NotRequired[float]
    """The cost per user token in USD per million tokens for cached input"""
    input: float
    """The cost per user token in USD per million tokens"""
    output: NotRequired[float]
    """The cost per assistant token in USD per million tokens"""
    variant: NotRequired[str]
    """The pricing variant (e.g., "flex", "priority", "batches")"""


# Load in pricing pulled from ellmer
f = resources.files("chatlas").joinpath("data/prices.json").read_text(encoding="utf-8")
pricing_list: list[TokenPrice] = orjson.loads(f)


def get_price_info(name: str, model: str, variant: str = "") -> TokenPrice | None:
    """
    Get token pricing information given a provider name and model

    Note
    ----
    Only a subset of providers and models and currently supported.
    The pricing information derives from ellmer.

    Parameters
    ----------
    name
        The provider name (e.g., "OpenAI", "Anthropic", etc.)
    model
        The model name (e.g., "gpt-4.1", "claude-3-opus", etc.)
    variant
        The pricing variant (e.g., "flex", "priority"). Defaults to "" (standard pricing).

    Returns
    -------
    TokenPrice | None
    """
    # First, try to find an exact match with the variant
    result = next(
        (
            item
            for item in pricing_list
            if item["provider"] == name
            and item["model"] == model
            and item.get("variant", "") == variant
        ),
        None,
    )

    # If no exact match and variant was specified, fall back to baseline (empty variant)
    if result is None and variant:
        result = next(
            (
                item
                for item in pricing_list
                if item["provider"] == name
                and item["model"] == model
                and item.get("variant", "") == ""
            ),
            None,
        )

    return result


def get_token_cost(
    name: str,
    model: str,
    tokens: tuple[int, int, int],
    variant: str = "",
) -> float | None:
    """
    Compute the cost of a turn.

    Parameters
    ----------
    name
        The provider name (e.g., "OpenAI", "Anthropic", etc.)
    model
        The model name (e.g., "gpt-4.1", "claude-3-opus", etc.)
    tokens
        A tuple of (input_tokens, output_tokens, cached_input_tokens)
    variant
        The pricing variant (e.g., "flex", "priority"). Defaults to "" (standard pricing).

    Returns
    -------
    float | None
        The cost of the turn in USD, or None if the cost could not be calculated.
    """
    price = get_price_info(name, model, variant)
    if price is None:
        return None
    input_price = tokens[0] * (price["input"] / 1e6)
    output_price = tokens[1] * (price.get("output", 0) / 1e6)
    cached_price = tokens[2] * (price.get("cached_input", 0) / 1e6)
    return input_price + output_price + cached_price


def token_usage() -> list[TokenUsage] | None:
    """
    Report on token usage in the current session

    Call this function to find out the cumulative number of tokens that you
    have sent and received in the current session. The price will be shown if known

    Returns
    -------
    list[TokenUsage] | None
        A list of dictionaries with the following keys: "name", "input", "output", and "cost".
        If no cost data is available for the name/model combination chosen, then "cost" will be None.
        If no tokens have been logged, then None is returned.
    """
    return _token_counter.get_usage()

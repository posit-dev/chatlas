from __future__ import annotations

import copy
import importlib.resources as resources
import warnings
from threading import Lock
from typing import TYPE_CHECKING

import orjson

from ._logging import logger
from ._typing_extensions import TypedDict

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
    cost: float | None


class ThreadSafeTokenCounter:
    def __init__(self):
        self._lock = Lock()
        self._tokens: dict[str, TokenUsage] = {}

    def log_tokens(
        self, name: str, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        logger.info(
            f"Provider '{name}' generated a response of {output_tokens} tokens "
            f"from an input of {input_tokens} tokens."
        )

        with self._lock:
            if name not in self._tokens:
                self._tokens[name] = {
                    "name": name,
                    "model": model,
                    "input": input_tokens,
                    "output": output_tokens,
                    "cost": None,
                }
            else:
                self._tokens[name]["input"] += input_tokens
                self._tokens[name]["output"] += output_tokens

    def get_usage(self) -> list[TokenUsage] | None:
        with self._lock:
            if not self._tokens:
                return None
            # Create a deep copy to avoid external modifications
            return copy.deepcopy(list(self._tokens.values()))


# Global instance
_token_counter = ThreadSafeTokenCounter()


def tokens_log(provider: "Provider", tokens: tuple[int, int]) -> None:
    """
    Log token usage for a provider in a thread-safe manner.
    """
    _token_counter.log_tokens(provider.name, provider.model, tokens[0], tokens[1])


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
    model: str
    cached_input: float
    input: float
    output: float


# Load in pricing pulled from ellmer
f = resources.files("chatlas").joinpath("data/prices.json").read_text(encoding="utf-8")
PricingList: list[TokenPrice] = orjson.loads(f)


def get_token_pricing(name: str, model: str) -> TokenPrice | dict:
    """
    Get the token pricing for the chat if available based on the prices.json file.

    Returns
    -------
    dict[str, str | float]
        A dictionary with the token pricing for the chat. The keys are:
          - `"provider"`: The provider name (e.g., "OpenAI", "Anthropic", etc.).
          - `model`: The model name (e.g., "gpt-3.5-turbo", "claude-2", etc.).
          - `"input"`: The cost per user token in USD per million tokens.
          - `"output"`: The cost per assistant token in USD per million tokens.
    """
    result = next(
        (
            item
            for item in PricingList
            if item["provider"] == name and item["model"] == model
        ),
        {},
    )
    if not result:
        warnings.warn(
            f"Token pricing for the provider '{name}' and model '{model}' you selected is not available. "
            "Please check the provider's documentation."
        )

    return result


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
    tokens = _token_counter.get_usage()
    if tokens:
        for item in tokens:
            price = get_token_pricing(item["name"], item["model"])
            if price:
                item["cost"] = item["input"] * (price["input"] / 1e6) + item[
                    "output"
                ] * (price["output"] / 1e6)
            else:
                item["cost"] = None

    return tokens

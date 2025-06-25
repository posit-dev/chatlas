from __future__ import annotations

import copy
import warnings
import importlib.resources as resources
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
    input: int
    output: int


class ThreadSafeTokenCounter:
    def __init__(self):
        self._lock = Lock()
        self._tokens: dict[str, TokenUsage] = {}

    def log_tokens(self, name: str, input_tokens: int, output_tokens: int) -> None:
        logger.info(
            f"Provider '{name}' generated a response of {output_tokens} tokens "
            f"from an input of {input_tokens} tokens."
        )

        with self._lock:
            if name not in self._tokens:
                self._tokens[name] = {
                    "name": name,
                    "input": input_tokens,
                    "output": output_tokens,
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


def tokens_log(provider: Provider, tokens: tuple[int, int]) -> None:
    """
    Log token usage for a provider in a thread-safe manner.
    """
    _token_counter.log_tokens(provider.name, tokens[0], tokens[1])


def tokens_reset() -> None:
    """
    Reset the token usage counter
    """
    global _token_counter  # noqa: PLW0603
    _token_counter = ThreadSafeTokenCounter()


f = resources.files("chatlas").joinpath("data/prices.json").read_text(encoding="utf-8")
prices_json = orjson.loads(f)


def get_token_pricing(provider: Provider) -> dict[str, str | float]:
    """
    Get the token pricing for the chat if available based on the prices.json file.

    Returns
    -------
    dict[str, str | float]
        A dictionary with the token pricing for the chat. The keys are:
          - `"provider"`: The provider name (e.g., "OpenAI", "Anthropic", etc.).
          - `model`: The model name (e.g., "gpt-3.5-turbo", "claude-2", etc.).
          - `"input"`: The cost per user token in USD.
          - `"output"`: The cost per assistant token in USD.
    """
    result = next(
        (
            item
            for item in prices_json
            if item["provider"] == provider.name and item["model"] == provider.model
        ),
        {},
    )

    if not result:
        warnings.warn(
            f"Token pricing for the provider '{provider.name}' and model '{provider.model}' you selected is not available. "
            "Please check the provider's documentation."
        )

    return result


def get_token_cost(
    name: str, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """
    Get the cost of tokens for a given provider and model.

    Parameters
    ----------
    name : Provider
        The provider instance.
    model : str
        The model name.
    input_tokens : int
        The number of input tokens.
    output_tokens : int
        The number of output tokens.

    Returns
    -------
    float
        The cost of the tokens, or None if the cost is not known.
    """

    # return get_token_cost(provider.__name__, model, input_tokens, output_tokens)


def token_usage() -> list[TokenUsage] | None:
    """
    Report on token usage in the current session

    Call this function to find out the cumulative number of tokens that you
    have sent and received in the current session. The price will be shown if known

    Returns
    -------
    list[TokenUsage] | None
        A list of dictionaries with the following keys: "name", "input", and "output".
        If no tokens have been logged, then None is returned.
    """
    _token_counter.get_usage()

    return _token_counter.get_usage()

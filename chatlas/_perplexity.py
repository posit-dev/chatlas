from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._openai import ChatOpenAI
from ._turn import Turn
from ._utils import MISSING, MISSING_TYPE, inform_model_default

if TYPE_CHECKING:
    from ._openai import ChatCompletionArgs, ProviderClientArgs


def ChatPerplexity(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.perplexity.ai/",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ProviderClientArgs"] = None,
) -> Chat["ChatCompletionArgs"]:
    """
    Chat with a model hosted on perplexity.ai

    Sign up at <https://www.perplexity.ai>.

    This function is a lightweight wrapper around `ChatOpenAI` with
    the defaults tweaked for perplexity.ai.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-`None` values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `PERPLEXITY_API_KEY` environment
        variable.
    base_url
        The base URL to the endpoint; the default uses Perplexity's API.
    seed
        Optional integer seed that ChatGPT uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client
        constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Examples
    --------
    >>> from chatlas import ChatPerplexity
    >>> chat = ChatPerplexity()
    >>> chat.chat("What is the capital of France?")
    """
    if model is None:
        model = inform_model_default("llama-3.1-sonar-small-128k-online")
    if api_key is None:
        api_key = os.getenv("PERPLEXITY_API_KEY")

    return ChatOpenAI(
        system_prompt=system_prompt,
        turns=turns,
        model=model,
        api_key=api_key,
        base_url=base_url,
        seed=seed,
        kwargs=kwargs,
    )

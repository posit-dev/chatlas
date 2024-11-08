from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._openai import ChatOpenAI
from ._turn import Turn
from ._utils import inform_model_default
from .types import MISSING, MISSING_TYPE

if TYPE_CHECKING:
    from ._openai import ChatCompletionArgs, ProviderClientArgs


def ChatGithub(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://models.inference.ai.azure.com/",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ProviderClientArgs"] = None,
) -> Chat["ChatCompletionArgs"]:
    """
    Chat with a model hosted on the GitHub model marketplace.

    GitHub (via Azure) hosts a wide variety of open source models, some of
    which are fined tuned for specific tasks.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Sign up at <https://github.com/marketplace/models> to get an API key.
    You may need to apply for and be accepted into a beta access program.
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatGithub` requires the `openai` package (e.g., `pip install openai`).
    :::


    Examples
    --------

    ```python
    import os
    from chatlas import ChatGithub

    chat = ChatGithub(api_key=os.getenv("GITHUB_PAT"))
    chat.chat("What is the capital of France?")
    ```

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
        this directly, but instead set the `GITHUB_PAT` environment variable.
    base_url
        The base URL to the endpoint; the default uses Github's API.
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

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for the GitHub model marketplace.
    """
    if model is None:
        model = inform_model_default("gpt-4o-mini")
    if api_key is None:
        api_key = os.getenv("GITHUB_PAT")

    return ChatOpenAI(
        system_prompt=system_prompt,
        turns=turns,
        model=model,
        api_key=api_key,
        base_url=base_url,
        seed=seed,
        kwargs=kwargs,
    )

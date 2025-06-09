from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._openai import ChatOpenAI
from ._turn import Turn
from ._utils import MISSING, MISSING_TYPE

if TYPE_CHECKING:
    from ._openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatPerplexity(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.perplexity.ai/",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on perplexity.ai.

    Perplexity AI is a platform for running LLMs that are capable of
    searching the web in real-time to help them answer questions with
    information that may not have been available when the model was trained.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Sign up at <https://www.perplexity.ai> to get an API key.
    :::


    Examples
    --------

    ```python
    import os
    from chatlas import ChatPerplexity

    chat = ChatPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))
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

    Note
    ----
    This function is a lightweight wrapper around [](`chatlas.ChatOpenAI`) with
    the defaults tweaked for perplexity.ai.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatPerplexity(api_key="...")`)
    is the simplest way to get started, and is fine for interactive use, but is
    problematic for code that may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    PERPLEXITY_API_KEY=...
    ```

    ```python
    from chatlas import ChatPerplexity
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatPerplexity()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export PERPLEXITY_API_KEY=...
    ```
    """
    if model is None:
        model = log_model_default("llama-3.1-sonar-small-128k-online")
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

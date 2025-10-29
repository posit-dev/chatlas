from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai_completions import OpenAICompletionsProvider
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from ._provider_openai_completions import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatOpenRouter(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with one of the many models hosted on OpenRouter.

    OpenRouter provides access to a wide variety of language models from different providers
    through a unified API. Support for features depends on the underlying model that you use.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Sign up at <https://openrouter.ai> to get an API key.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatOpenRouter

    chat = ChatOpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use. See <https://openrouter.ai/models>
        for available models.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `OPENROUTER_API_KEY` environment variable.
    base_url
        The base URL to the endpoint; the default uses OpenRouter's API.
    seed
        Optional integer seed that the model uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for OpenRouter.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatOpenRouter(api_key="...")`)
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
    OPENROUTER_API_KEY=...
    ```

    ```python
    from chatlas import ChatOpenRouter
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatOpenRouter()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export OPENROUTER_API_KEY=...
    ```
    """
    if model is None:
        model = log_model_default("gpt-4.1")

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    kwargs2 = add_default_headers(kwargs or {})

    return Chat(
        provider=OpenAICompletionsProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            name="OpenRouter",
            kwargs=kwargs2,
        ),
        system_prompt=system_prompt,
    )


def add_default_headers(kwargs: "ChatClientArgs") -> "ChatClientArgs":
    headers = kwargs.get("default_headers", None)
    # https://openrouter.ai/docs/api-keys
    default_headers = {
        "HTTP-Referer": "https://posit-dev.github.io/chatlas",
        "X-Title": "chatlas",
        **(headers or {}),
    }
    return {"default_headers": default_headers, **kwargs}

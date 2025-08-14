from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai import OpenAIProvider
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatMistral(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.mistral.ai/v1/",
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Mistral's La Plateforme.

    Mistral AI provides high-performance language models through their API platform.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API credentials

    Get your API key from https://console.mistral.ai/api-keys.
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatMistral

    chat = ChatMistral(api_key=os.getenv("MISTRAL_API_KEY"))
    chat.chat("Tell me three jokes about statisticians")
    ```

    Known limitations
    -----------------

    * Tool calling may be unstable.
    * Images require a model that supports vision.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `MISTRAL_API_KEY` environment
        variable.
    base_url
        The base URL to the endpoint; the default uses Mistral AI.
    seed
        Optional integer seed that Mistral uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client
        constructor (Mistral uses OpenAI-compatible API).

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatMistral(api_key="...")`)
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
    MISTRAL_API_KEY=...
    ```

    ```python
    from chatlas import ChatMistral
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatMistral()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export MISTRAL_API_KEY=...
    ```
    """
    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    if model is None:
        model = log_model_default("mistral-large-latest")

    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")

    return Chat(
        provider=MistralProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class MistralProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        base_url: str = "https://api.mistral.ai/v1/",
        seed: Optional[int] = None,
        name: str = "Mistral",
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            name=name,
            kwargs=kwargs,
        )

    # Mistral is essentially OpenAI-compatible, with a couple small differences.
    # We _could_ bring in the Mistral SDK and use it directly for more precise typing,
    # etc., but for now that doesn't seem worth it.
    def _chat_perform_args(
        self, stream, turns, tools, data_model=None, kwargs=None
    ) -> "SubmitInputArgs":
        # Get the base arguments from OpenAI provider
        kwargs2 = super()._chat_perform_args(stream, turns, tools, data_model, kwargs)

        # Mistral doesn't support stream_options
        if "stream_options" in kwargs2:
            del kwargs2["stream_options"]

        # Mistral wants random_seed, not seed
        if seed := kwargs2.pop("seed", None):
            if isinstance(seed, int):
                kwargs2["extra_body"] = {"random_seed": seed}
            elif seed is not None:
                raise ValueError(
                    "MistralProvider only accepts an integer seed, or None."
                )

        return kwargs2

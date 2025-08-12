from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, cast

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai import OpenAIProvider
from ._turn import Turn
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatDeepSeek(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on DeepSeek.

    DeepSeek is a platform for AI inference with competitive pricing
    and performance.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Sign up at <https://platform.deepseek.com> to get an API key.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatDeepSeek

    chat = ChatDeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Known limitations
    --------------

    * Structured data extraction is not supported.
    * Images are not supported.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `DEEPSEEK_API_KEY` environment variable.
    base_url
        The base URL to the endpoint; the default uses DeepSeek's API.
    seed
        Optional integer seed that DeepSeek uses to try and make output more
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
    the defaults tweaked for DeepSeek.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatDeepSeek(api_key="...")`)
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
    DEEPSEEK_API_KEY=...
    ```

    ```python
    from chatlas import ChatDeepSeek
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatDeepSeek()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export DEEPSEEK_API_KEY=...
    ```
    """
    if model is None:
        model = log_model_default("deepseek-chat")

    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=DeepSeekProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            name="DeepSeek",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class DeepSeekProvider(OpenAIProvider):
    @staticmethod
    def _as_message_param(turns: list[Turn]) -> list["ChatCompletionMessageParam"]:
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionUserMessageParam,
        )

        params = OpenAIProvider._as_message_param(turns)

        # Content must be a string
        for i, param in enumerate(params):
            if param["role"] in ["assistant", "user"]:
                param = cast(
                    ChatCompletionAssistantMessageParam
                    | ChatCompletionUserMessageParam,
                    param,
                )
                contents = param.get("content", None)
                if not isinstance(contents, list):
                    continue
                params[i]["content"] = "".join(
                    content.get("text", "") for content in contents
                )

        return params

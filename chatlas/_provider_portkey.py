from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai import OpenAIProvider
from ._utils import drop_none

if TYPE_CHECKING:
    from ._provider_openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatPortkey(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    virtual_key: Optional[str] = None,
    base_url: str = "https://api.portkey.ai/v1",
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on PortkeyAI

    [PortkeyAI](https://portkey.ai/docs/product/ai-gateway/universal-api)
    provides an interface (AI Gateway) to connect through its Universal API to a
    variety of LLMs providers with a single endpoint.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Portkey credentials

    Follow the instructions at <https://portkey.ai/docs/introduction/make-your-first-request>
    to get started making requests to PortkeyAI. You will need to set the
    `PORTKEY_API_KEY` environment variable to your Portkey API key, and optionally
    the `PORTKEY_VIRTUAL_KEY` environment variable to your virtual key.
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatPortkey

    chat = ChatPortkey(api_key=os.getenv("PORTKEY_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

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
        this directly, but instead set the `PORTKEY_API_KEY` environment variable.
    virtual_key
        An (optional) virtual identifier, storing the LLM provider's API key. See
        [documentation](https://portkey.ai/docs/product/ai-gateway/virtual-keys).
        You generally should not supply this directly, but instead set the
        `PORTKEY_VIRTUAL_KEY` environment variable.
    base_url
        The base URL for the Portkey API. The default is suitable for most users.
    kwargs
        Additional arguments to pass to the OpenAIProvider, such as headers or
        other client configuration options.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    -----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for PortkeyAI.
    """
    if model is None:
        model = log_model_default("gpt-4.1")
    if api_key is None:
        api_key = os.getenv("PORTKEY_API_KEY")

    kwargs2 = add_default_headers(
        kwargs or {},
        api_key=api_key,
        virtual_key=virtual_key,
    )

    return Chat(
        provider=OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            name="Portkey",
            kwargs=kwargs2,
        ),
        system_prompt=system_prompt,
    )


def add_default_headers(
    kwargs: "ChatClientArgs",
    api_key: Optional[str] = None,
    virtual_key: Optional[str] = None,
) -> "ChatClientArgs":
    headers = kwargs.get("default_headers", None)
    default_headers = drop_none(
        {
            "x-portkey-api-key": api_key,
            "x-portkey-virtual-key": virtual_key,
            **(headers or {}),
        }
    )
    return {"default_headers": default_headers, **kwargs}


class PortkeyProvider(OpenAIProvider):
    def list_models(self):
        raise NotImplementedError(
            ".list_models() is not yet implemented for Portkey. "
            "To view model availability online, see https://portkey.ai/docs/product/model-catalog"
        )

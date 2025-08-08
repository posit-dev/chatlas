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

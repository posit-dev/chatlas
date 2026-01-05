from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion

from ._chat import Chat
from ._provider_openai import OpenAIProvider
from ._provider_openai_completions import OpenAICompletionsProvider
from ._utils import MISSING, MISSING_TYPE, is_testing, split_http_client_kwargs

if TYPE_CHECKING:
    from openai.types.responses import Response
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params.reasoning import Reasoning

    from .types.openai import (
        ChatAzureClientArgs,
        ResponsesSubmitInputArgs,
        SubmitInputArgs,
    )


def ChatAzureOpenAI(
    *,
    endpoint: str,
    deployment_id: str,
    api_version: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    reasoning: "Optional[ReasoningEffort | Reasoning]" = None,
    service_tier: Optional[
        Literal["auto", "default", "flex", "scale", "priority"]
    ] = None,
    kwargs: Optional["ChatAzureClientArgs"] = None,
) -> "Chat[ResponsesSubmitInputArgs, Response]":
    """
    Chat with a model hosted on Azure OpenAI.

    The [Azure OpenAI server](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
    hosts a number of open source models as well as proprietary models
    from OpenAI.

    Examples
    --------
    ```python
    import os
    from chatlas import ChatAzureOpenAI

    chat = ChatAzureOpenAI(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_id="REPLACE_WITH_YOUR_DEPLOYMENT_ID",
        api_version="YYYY-MM-DD",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    endpoint
        Azure OpenAI endpoint url with protocol and hostname, i.e.
        `https://{your-resource-name}.openai.azure.com`. Defaults to using the
        value of the `AZURE_OPENAI_ENDPOINT` environment variable.
    deployment_id
        Deployment id for the model you want to use.
    api_version
        The API version to use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `AZURE_OPENAI_API_KEY` environment
        variable.
    system_prompt
        A system prompt to set the behavior of the assistant.
    reasoning
        The reasoning effort (e.g., `"low"`, `"medium"`, `"high"`) for
        reasoning-capable models like the o and gpt-5 series. To use the default
        reasoning settings in a way that will work for multi-turn conversations,
        set this to an empty dictionary `{}`.
    service_tier
        Request a specific service tier. Options:
        - `"auto"` (default): uses the service tier configured in Project settings.
        - `"default"`: standard pricing and performance.
        - `"flex"`: slower and cheaper.
        - `"scale"`: batch-like pricing for high-volume use.
        - `"priority"`: faster and more expensive.
    kwargs
        Additional arguments to pass to the `openai.AzureOpenAI()` client constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    kwargs_chat: "ResponsesSubmitInputArgs" = {}

    if reasoning is not None:
        if isinstance(reasoning, str):
            reasoning = {"effort": reasoning, "summary": "auto"}
        kwargs_chat["reasoning"] = reasoning

    if service_tier is not None:
        kwargs_chat["service_tier"] = service_tier

    return Chat(
        provider=OpenAIAzureProvider(
            endpoint=endpoint,
            deployment_id=deployment_id,
            api_version=api_version,
            api_key=api_key,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
        kwargs_chat=kwargs_chat,
    )


class OpenAIAzureProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        deployment_id: str,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        name: str = "Azure/OpenAI",
        model: Optional[str] = "UnusedValue",
        kwargs: Optional["ChatAzureClientArgs"] = None,
    ):
        super().__init__(
            name=name,
            model=deployment_id,
            # The OpenAI() constructor will fail if no API key is present.
            # However, a dummy value is fine -- AzureOpenAI() handles the auth.
            api_key=api_key or "not-used",
        )

        kwargs_full: "ChatAzureClientArgs" = {
            "azure_endpoint": endpoint,
            "azure_deployment": deployment_id,
            "api_version": api_version,
            "api_key": api_key,
            **(kwargs or {}),
        }

        sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs_full)

        self._client = AzureOpenAI(**sync_kwargs)  # type: ignore
        self._async_client = AsyncAzureOpenAI(**async_kwargs)  # type: ignore


def ChatAzureOpenAICompletions(
    *,
    endpoint: str,
    deployment_id: str,
    api_version: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatAzureClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Azure OpenAI.

    This function exists mainly for historical reasons; new code should
    prefer `ChatAzureOpenAI()`, which uses the newer Responses API.
    """

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=OpenAIAzureCompletionsProvider(
            endpoint=endpoint,
            deployment_id=deployment_id,
            api_version=api_version,
            api_key=api_key,
            seed=seed,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class OpenAIAzureCompletionsProvider(OpenAICompletionsProvider):
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        deployment_id: str,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: int | None = None,
        name: str = "Azure/OpenAI",
        model: Optional[str] = "UnusedValue",
        kwargs: Optional["ChatAzureClientArgs"] = None,
    ):
        super().__init__(
            name=name,
            model=deployment_id,
            seed=seed,
            # The OpenAI() constructor will fail if no API key is present.
            # However, a dummy value is fine -- AzureOpenAI() handles the auth.
            api_key=api_key or "not-used",
        )

        kwargs_full: "ChatAzureClientArgs" = {
            "azure_endpoint": endpoint,
            "azure_deployment": deployment_id,
            "api_version": api_version,
            "api_key": api_key,
            **(kwargs or {}),
        }

        sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs_full)

        self._client = AzureOpenAI(**sync_kwargs)  # type: ignore
        self._async_client = AsyncAzureOpenAI(**async_kwargs)  # type: ignore

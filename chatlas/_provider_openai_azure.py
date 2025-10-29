from typing import TYPE_CHECKING, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion

from ._chat import Chat
from ._provider_openai import OpenAIProvider
from ._utils import MISSING, MISSING_TYPE, is_testing, split_http_client_kwargs

if TYPE_CHECKING:
    from .types.openai import ChatAzureClientArgs, SubmitInputArgs


def ChatAzureOpenAI(
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
        value of the `AZURE_OPENAI_ENDPOINT` envinronment variable.
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
    seed
        Optional integer seed that ChatGPT uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.AzureOpenAI()` client constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=OpenAIAzureProvider(
            endpoint=endpoint,
            deployment_id=deployment_id,
            api_version=api_version,
            api_key=api_key,
            seed=seed,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class OpenAIAzureProvider(OpenAIProvider):
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
            # The OpenAI() constructor will fail if no API key is present.
            # However, a dummy value is fine -- AzureOpenAI() handles the auth.
            api_key=api_key or "not-used",
        )

        self._seed = seed

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

import os
from typing import TYPE_CHECKING, Optional

import requests

from ._chat import Chat
from ._provider_openai import OpenAIProvider

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

    from .types.openai import ChatClientArgs


def ChatVllm(
    *,
    base_url: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: Optional[int] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat:
    """
    Chat with a model hosted by vLLM

    [vLLM](https://docs.vllm.ai/en/latest/) is an open source library that
    provides an efficient and convenient LLMs model server. You can use
    `ChatVllm()` to connect to endpoints powered by vLLM.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## vLLM runtime

    `ChatVllm` requires a vLLM server to be running somewhere (either on your
    machine or a remote server). If you want to run a vLLM server locally, see
    the [vLLM documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html).
    :::


    Parameters
    ----------
    base_url
        Base URL of the vLLM server (e.g., "http://localhost:8000/v1").
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        Model identifier to use.
    seed
        Random seed for reproducibility.
    api_key
        API key for authentication. If not provided, the `VLLM_API_KEY` environment
        variable will be used.
    kwargs
        Additional arguments to pass to the LLM client.

    Return
    ------
    Chat
        A chat object that retains the state of the conversation.

    Note
    -----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for PortkeyAI.
    """

    if api_key is None:
        api_key = get_vllm_key()

    if model is None:
        models = get_vllm_models(base_url, api_key)
        available_models = ", ".join(models)
        raise ValueError(f"Must specify model. Available models: {available_models}")

    return Chat(
        provider=VLLMProvider(
            base_url=base_url,
            model=model,
            seed=seed,
            api_key=api_key,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class VLLMProvider(OpenAIProvider):
    # Just like OpenAI but no strict
    @staticmethod
    def _tool_schema_json(
        schema: "ChatCompletionToolParam",
    ) -> "ChatCompletionToolParam":
        schema["function"]["strict"] = False
        return schema


def get_vllm_key() -> str:
    key = os.getenv("VLLM_API_KEY", os.getenv("VLLM_KEY"))
    if not key:
        raise ValueError("VLLM_API_KEY environment variable not set")
    return key


def get_vllm_models(base_url: str, api_key: Optional[str] = None) -> list[str]:
    if api_key is None:
        api_key = get_vllm_key()

    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/v1/models", headers=headers)
    response.raise_for_status()
    data = response.json()

    return [model["id"] for model in data["data"]]


# def chat_vllm_test(**kwargs) -> Chat:
#     """Create a test chat instance with default parameters."""
#     return ChatVllm(base_url="https://llm.nrp-nautilus.io/", model="llama3", **kwargs)

from __future__ import annotations

import os
import re
import urllib.request
from typing import TYPE_CHECKING, Optional

import orjson

from ._chat import Chat
from ._provider import ModelInfo
from ._provider_openai_completions import OpenAICompletionsProvider
from ._utils import is_testing

if TYPE_CHECKING:
    from ._provider_openai_completions import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatLMStudio(
    model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    base_url: str = "http://localhost:1234",
    api_key: Optional[str] = None,
    seed: Optional[int] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> "Chat[SubmitInputArgs, ChatCompletion]":
    """
    Chat with a local LM Studio model.

    [LM Studio](https://lmstudio.ai) makes it easy to run a wide variety of
    open-source models locally on Mac, Windows, and Linux. It is particularly
    notable for its excellent support for Apple's MLX inference engine, making
    it a compelling choice for local inference on Apple Silicon.


    Prerequisites
    -------------

    ::: {.callout-note}
    ## LM Studio runtime

    `ChatLMStudio` requires [LM Studio](https://lmstudio.ai/download) to be
    installed and running on your machine with at least one model loaded.
    :::

    ::: {.callout-note}
    ## Load a model

    Open LM Studio, load a model from the Discover tab, then start the local
    server from the Developer tab.
    :::


    Examples
    --------

    ```python
    from chatlas import ChatLMStudio

    chat = ChatLMStudio(model="zai-org/glm-4.7-flash")
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    model
        The model to use for the chat. If `None`, a list of locally loaded
        models will be printed.
    system_prompt
        A system prompt to set the behavior of the assistant.
    base_url
        The base URL to the endpoint. The default uses the LM Studio local
        server. You can also set the `LMSTUDIO_BASE_URL` environment variable
        to override this default.
    api_key
        An optional API key. LM Studio doesn't require credentials for local
        usage. If you're accessing an LM Studio instance behind a reverse proxy
        or secured endpoint that enforces bearer-token authentication, you can
        set the `LMSTUDIO_API_KEY` environment variable or provide a value here.
    seed
        Optional integer seed that helps to make output more reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for LM Studio.
    """
    base_url = os.getenv("LMSTUDIO_BASE_URL", base_url)
    base_url = re.sub("/+$", "", base_url)

    if api_key is None:
        api_key = os.getenv("LMSTUDIO_API_KEY", "")

    if not has_lmstudio(base_url, api_key=api_key):
        raise RuntimeError("Can't find locally running LM Studio.")

    models = lmstudio_model_info(base_url, api_key=api_key)
    model_ids = [m["id"] for m in models]

    if model is None:
        raise ValueError(
            f"Must specify model. Locally loaded models: {', '.join(model_ids)}"
        )

    if model not in model_ids:
        raise ValueError(
            f"Model '{model}' is not available in LM Studio. "
            f"Load the model using the LM Studio GUI. "
            f"Locally loaded models: {', '.join(model_ids)}"
        )

    if seed is None:
        seed = 1014 if is_testing() else None

    return Chat(
        provider=LMStudioProvider(
            api_key=api_key if api_key else "lmstudio",
            model=model,
            base_url=base_url,
            seed=seed,
            name="LM Studio",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class LMStudioProvider(OpenAICompletionsProvider):
    def __init__(self, *, api_key, model, base_url, seed, name, kwargs):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=f"{base_url}/v1",
            seed=seed,
            name=name,
            kwargs=kwargs,
        )
        self.base_url = base_url

    def list_models(self):
        return lmstudio_model_info(self.base_url)


def lmstudio_model_info(base_url: str, api_key: str = "") -> list[ModelInfo]:
    req = urllib.request.Request(f"{base_url}/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    response = urllib.request.urlopen(req)
    data = orjson.loads(response.read())
    models = data.get("data", [])
    if not models:
        return []

    return [{"id": model["id"]} for model in models]


def has_lmstudio(base_url: str = "http://localhost:1234", api_key: str = "") -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        urllib.request.urlopen(req)
        return True
    except Exception:
        return False

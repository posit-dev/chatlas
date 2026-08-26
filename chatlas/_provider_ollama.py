from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

import orjson

from ._chat import Chat
from ._provider import ModelInfo
from ._provider_openai_completions import OpenAICompletionsProvider
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from openai.types.shared.reasoning_effort import ReasoningEffort

    from ._provider_openai_completions import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatOllama(
    model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    options: Optional[dict[str, Any]] = None,
    reasoning_effort: "ReasoningEffort" = None,
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a local Ollama model.

    [Ollama](https://ollama.com) makes it easy to run a wide-variety of
    open-source models locally, making it a great choice for privacy
    and security.


    Prerequisites
    -------------

    ::: {.callout-note}
    ## Ollama runtime

    `ChatOllama` requires the [ollama](https://ollama.com/download) executable
    to be installed and running on your machine.
    :::

    ::: {.callout-note}
    ## Pull model(s)

    Once ollama is running locally, download a model from the command line
    (e.g. `ollama pull llama3.2`).
    :::


    Examples
    --------

    ```python
    from chatlas import ChatOllama

    chat = ChatOllama(model="llama3.2")
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    model
        The model to use for the chat. If `None`, a list of locally installed
        models will be printed.
    system_prompt
        A system prompt to set the behavior of the assistant.
    base_url
        The base URL to the endpoint; the default uses ollama's API.
    options
        Additional Ollama model options (e.g. `{"num_ctx": 8192}` to increase
        the context window size, which defaults to 2048). These are passed
        through to the request body of Ollama's OpenAI-compatible endpoint.
        See <https://docs.ollama.com/api/chat#request-body-options> for
        available options.
    reasoning_effort
        Enables extended "thinking" for models that support it (e.g. qwen3,
        gpt-oss). Which values are accepted is model-dependent -- qwen3 only
        distinguishes `"none"` from any other value (thinking on), while
        gpt-oss accepts `"low"`, `"medium"`, or `"high"` but ignores `"none"`.
        See <https://docs.ollama.com/capabilities/thinking> for details.
    seed
        Optional integer seed that helps to make output more reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAICompletions`) with
    the defaults tweaked for ollama.

    Limitations
    -----------
    `ChatOllama` currently doesn't work with streaming tools, and tool calling more
    generally doesn't seem to work very well with currently available models.
    """

    base_url = re.sub("/+$", "", base_url)

    is_local = is_local_ollama(base_url)

    models = ollama_model_info(base_url)
    if models is None:
        if is_local:
            raise RuntimeError("Can't find locally running ollama.")
        raise RuntimeError(f"Can't connect to ollama at {base_url}.")

    model_ids = [m["id"] for m in models]

    if model is None:
        raise ValueError(f"Must specify model. Available models: {', '.join(model_ids)}")

    # Model ids have any ":latest" tag stripped, so normalize the same way
    if re.sub(":latest$", "", model) not in model_ids:
        if is_local:
            raise ValueError(
                f"Model '{model}' is not installed locally. "
                f"Run `ollama pull {model}` in your terminal to install it. "
                f"Locally installed models: {', '.join(model_ids)}"
            )
        raise ValueError(
            f"Model '{model}' is not available on {base_url}. "
            f"Available models: {', '.join(model_ids)}"
        )

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    kwargs_chat: "SubmitInputArgs" = {}
    if reasoning_effort is not None:
        kwargs_chat["reasoning_effort"] = reasoning_effort
    if options is not None:
        kwargs_chat["extra_body"] = options

    return Chat(
        provider=OllamaProvider(
            api_key="ollama",  # ignored
            model=model,
            base_url=base_url,
            seed=seed,
            name="Ollama",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
        kwargs_chat=kwargs_chat,
    )


class OllamaProvider(OpenAICompletionsProvider):
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
        return ollama_model_info(self.base_url) or []


def ollama_model_info(base_url: str) -> Optional[list[ModelInfo]]:
    """
    Retrieve model info from ollama's `/api/tags` endpoint.

    Returns `None` if the endpoint can't be reached.
    """
    try:
        response = urllib.request.urlopen(url=f"{base_url}/api/tags")
    except Exception:
        return None

    data = orjson.loads(response.read())
    models = data.get("models", [])
    if not models:
        return []

    res: list[ModelInfo] = []
    for model in models:
        # TODO: add capabilities
        info: ModelInfo = {
            "id": re.sub(":latest$", "", model["name"]),
            "created_at": model["modified_at"],
            "size": model["size"],
        }
        res.append(info)

    return res


def is_local_ollama(base_url: str) -> bool:
    """Whether `base_url` points at a locally running ollama instance."""
    host = urllib.parse.urlparse(base_url).hostname
    return host in ("localhost", "127.0.0.1", "::1")


def has_ollama(base_url):
    return ollama_model_info(base_url) is not None

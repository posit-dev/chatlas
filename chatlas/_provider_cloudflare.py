from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai import OpenAIProvider
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from ._provider_openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatCloudflare(
    *,
    account: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Cloudflare Workers AI.

    Cloudflare Works AI hosts a variety of open-source AI models.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API credentials

    To use the Cloudflare API, you must have an Account ID and an Access Token,
    which you can obtain by following the instructions at
    <https://developers.cloudflare.com/workers-ai/get-started/rest-api/>.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatCloudflare

    chat = ChatCloudflare(
        api_key=os.getenv("CLOUDFLARE_API_KEY"),
        account=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    )
    chat.chat("What is the capital of France?")
    ```

    Known limitations
    -----------------

    - Tool calling does not appear to work.
    - Images don't appear to work.

    Parameters
    ----------
    account
        The Cloudflare account ID. You generally should not supply this directly,
        but instead set the `CLOUDFLARE_ACCOUNT_ID` environment variable.
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `CLOUDFLARE_API_KEY` environment
        variable.
    seed
        Optional integer seed that ChatGPT uses to try and make output more
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
    the defaults tweaked for Cloudflare.

    Note
    ----
    Pasting credentials into a chat constructor (e.g.,
    `ChatCloudflare(api_key="...", account="...")`) is the simplest way to get
    started, and is fine for interactive use, but is problematic for code that
    may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    CLOUDFLARE_API_KEY=...
    CLOUDFLARE_ACCOUNT_ID=...
    ```

    ```python
    from chatlas import ChatCloudflare
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatCloudflare()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export CLOUDFLARE_API_KEY=...
    export CLOUDFLARE_ACCOUNT_ID=...
    ```
    """
    # List at https://developers.cloudflare.com/workers-ai/models/
    # `@cf` appears to be part of the model name
    if model is None:
        model = log_model_default("@cf/meta/llama-3.3-70b-instruct-fp8-fast")

    if api_key is None:
        api_key = os.getenv("CLOUDFLARE_API_KEY")

    if account is None:
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID")

    if account is None:
        raise ValueError(
            "Cloudflare account ID is required. Set the CLOUDFLARE_ACCOUNT_ID "
            "environment variable or pass the `account` parameter."
        )

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    # https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/
    cloudflare_api = "https://api.cloudflare.com/client/v4/accounts"
    base_url = f"{cloudflare_api}/{account}/ai/v1/"

    return Chat(
        provider=CloudflareProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            name="Cloudflare",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


# TODO: need to treat tools like Google?


class CloudflareProvider(OpenAIProvider):
    pass

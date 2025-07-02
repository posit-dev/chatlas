from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._openai import OpenAIProvider

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

    from ._openai import ChatCompletion
    from .types.openai import SubmitInputArgs


def ChatDatabricks(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    workspace_client: Optional["WorkspaceClient"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Databricks.

    Databricks provides out-of-the-box access to a number of [foundation
    models](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html)
    and can also serve as a gateway for external models hosted by a third party.

    Prerequisites
    --------------

    ::: {.callout-note}
    ## Python requirements

    `ChatDatabricks` requires the `databricks-sdk` package: `pip install
    "chatlas[databricks]"`.
    :::

    ::: {.callout-note}
    ## Authentication

    `chatlas` delegates to the `databricks-sdk` package for authentication with
    Databricks. As such, you can use any of the authentication methods discussed
    here:

    https://docs.databricks.com/aws/en/dev-tools/sdk-python#authentication

    Note that Python-specific article points to this language-agnostic "unified"
    approach to authentication:

    https://docs.databricks.com/aws/en/dev-tools/auth/unified-auth

    There, you'll find all the options listed, but a simple approach that
    generally works well is to set the following environment variables:

    * `DATABRICKS_HOST`: The Databricks host URL for either the Databricks
      workspace endpoint or the Databricks accounts endpoint.
    * `DATABRICKS_TOKEN`: The Databricks personal access token.
    :::

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    workspace_client
        A `databricks.sdk.WorkspaceClient()` to use for the connection. If not
        provided, a new client will be created.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.
    """
    if model is None:
        model = log_model_default("databricks-claude-3-7-sonnet")

    return Chat(
        provider=DatabricksProvider(
            model=model,
            workspace_client=workspace_client,
        ),
        system_prompt=system_prompt,
    )


class DatabricksProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        model: str,
        name: str = "Databricks",
        workspace_client: Optional["WorkspaceClient"] = None,
    ):
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError:
            raise ImportError(
                "`ChatDatabricks()` requires the `databricks-sdk` package. "
                "Install it with `pip install databricks-sdk`."
            )

        import httpx
        from openai import AsyncOpenAI

        super().__init__(name=name, model=model)

        self._seed = None

        if workspace_client is None:
            workspace_client = WorkspaceClient()

        client = workspace_client.serving_endpoints.get_open_ai_client()

        self._client = client

        # The databricks sdk does currently expose an async client, but we can
        # effectively mirror what .get_open_ai_client() does internally.
        # Note also there is a open PR to add async support that does essentially
        # the same thing:
        # https://github.com/databricks/databricks-sdk-py/pull/851
        self._async_client = AsyncOpenAI(
            base_url=client.base_url,
            api_key="no-token",  # A placeholder to pass validations, this will not be used
            http_client=httpx.AsyncClient(auth=client._client.auth),
        )

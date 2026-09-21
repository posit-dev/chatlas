from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import httpx
from openai import AsyncOpenAI

from ._chat import Chat
from ._content import Content
from ._logging import log_model_default
from ._provider_openai_completions import OpenAICompletionsProvider
from ._provider_openai_generic import create_openai_client
from ._turn import AssistantTurn

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

    from ._provider_openai_completions import ChatCompletion
    from .types.openai import SubmitInputArgs


def _normalize_content_parts(parts: list[Any]) -> tuple[Optional[str], Optional[str]]:
    """Split a Databricks GPT-OSS typed content array into (text, reasoning).

    GPT-OSS endpoints can return `message.content` / `delta.content` as a list
    of typed parts (`{"type": "text", ...}`, `{"type": "reasoning", ...}`)
    rather than the plain string the OpenAI-compatible parser assumes. Text
    parts are concatenated back into a string; reasoning summaries are
    concatenated into a separate string so callers can feed it into the
    existing `reasoning` handling. Unrecognized part types are ignored, per
    this provider's own scope (no generic typed-content support).
    """
    text = ""
    reasoning = ""
    for part in parts:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == "text":
            text += part.get("text") or ""
        elif part_type == "reasoning":
            for summary in part.get("summary") or []:
                reasoning += summary.get("text") or ""
    return (text or None), (reasoning or None)


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
        provided, a new client will be created. Note that calling
        [`Chat.close()`](`chatlas.Chat.close`) does not close a
        caller-supplied `WorkspaceClient` -- it only closes the OpenAI-compatible
        client that chatlas derives from it.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.
    """
    if model is None:
        model = log_model_default("databricks-claude-sonnet-4-6")

    return Chat(
        provider=DatabricksProvider(
            model=model,
            workspace_client=workspace_client,
        ),
        system_prompt=system_prompt,
    )


class DatabricksProvider(OpenAICompletionsProvider):
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

        super().__init__(
            name=name,
            model=model,
            # The OpenAI() constructor will fail if no API key is present.
            # However, a dummy value is fine -- WorkspaceClient() handles the auth.
            api_key="not-used",
        )

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
        # Databricks injects a legacy httpx client even though OpenAI types it as
        # native httpx2. OpenAI 3 retains this as a runtime compatibility path.
        legacy_auth = cast(httpx.Auth | None, client._client.auth)
        self._async_client = create_openai_client(
            AsyncOpenAI,
            {
                "base_url": client.base_url,
                "api_key": "no-token",
                "http_client": httpx.AsyncClient(auth=legacy_auth),
            },
        )

    def list_models(self):
        raise NotImplementedError(
            ".list_models() is not yet implemented for Databricks. "
            "To view model availability online, see "
            "https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#-foundation-model-types"
        )

    # Databricks doesn't support stream_options
    def _chat_perform_args(
        self, stream, turns, tools, data_model=None, kwargs=None
    ) -> "SubmitInputArgs":
        kwargs2 = super()._chat_perform_args(stream, turns, tools, data_model, kwargs)

        if "stream_options" in kwargs2:
            del kwargs2["stream_options"]

        return kwargs2

    # GPT-OSS endpoints can return delta.content as a list of typed parts
    # instead of a plain string; normalize it in place, then reuse the base
    # class's existing string/reasoning handling.
    def stream_content(self, chunk, completion, turns=()) -> list[Content]:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is not None and isinstance(delta.content, list):
            text, reasoning = _normalize_content_parts(delta.content)
            delta.content = text
            if reasoning:
                setattr(delta, "reasoning", reasoning)
        return super().stream_content(chunk, completion, turns)

    # The streaming loop merges each chunk before it yields its content, so the
    # typed array reaches `model_dump()` unless it is normalized here as well.
    def stream_merge_chunks(self, completion, chunk):
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is not None and isinstance(delta.content, list):
            text, reasoning = _normalize_content_parts(delta.content)
            delta.content = text
            if reasoning:
                setattr(delta, "reasoning", reasoning)
        return super().stream_merge_chunks(completion, chunk)

    # Same normalization for the non-streaming/completed-response path.
    @staticmethod
    def _response_as_turn(
        completion: "ChatCompletion", has_data_model: bool
    ) -> AssistantTurn["ChatCompletion"]:
        message = completion.choices[0].message
        if isinstance(message.content, list):
            text, reasoning = _normalize_content_parts(message.content)
            message.content = text
            if reasoning:
                setattr(message, "reasoning", reasoning)
        return OpenAICompletionsProvider._response_as_turn(completion, has_data_model)

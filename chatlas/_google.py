from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

from pydantic import BaseModel

from ._chat import Chat
from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._logging import log_model_default
from ._merge import merge_dicts
from ._provider import Provider
from ._tokens import tokens_log
from ._tools import Tool
from ._turn import Turn, normalize_turns, user_turn

if TYPE_CHECKING:
    from google.genai.types import Content as GoogleContent
    from google.genai.types import (
        GenerateContentResponse,
        GenerateContentResponseDict,
        Part,
        PartDict,
    )

    from .types.google import ChatClientArgs, SubmitInputArgs
else:
    GenerateContentResponse = object


def ChatGoogle(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", GenerateContentResponse]:
    """
    Chat with a Google Gemini model.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    To use Google's models (i.e., Gemini), you'll need to sign up for an account
    and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python).
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[google]"`.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatGoogle

    chat = ChatGoogle(api_key=os.getenv("GOOGLE_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch.
        Do not provide non-`None` values for both `turns` and `system_prompt`.
        Each message in the list should be a dictionary with at least `role`
        (usually `system`, `user`, or `assistant`, but `tool` is also possible).
        Normally there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `GOOGLE_API_KEY` environment variable.
    kwargs
        Additional arguments to pass to the `genai.Client` constructor.

    Returns
    -------
    Chat
        A Chat object.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatGoogle(api_key="...")`)
    is the simplest way to get started, and is fine for interactive use, but is
    problematic for code that may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    GOOGLE_API_KEY=...
    ```

    ```python
    from chatlas import ChatGoogle
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatGoogle()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export GOOGLE_API_KEY=...
    ```
    """

    if model is None:
        model = log_model_default("gemini-2.0-flash")

    return Chat(
        provider=GoogleProvider(
            model=model,
            api_key=api_key,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt=system_prompt,
        ),
    )


class GoogleProvider(
    Provider[
        GenerateContentResponse, GenerateContentResponse, "GenerateContentResponseDict"
    ]
):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        kwargs: Optional["ChatClientArgs"],
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `google-genai` package. "
                "Install it with `pip install google-genai`."
            )

        self._model = model

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        self._client = genai.Client(**kwargs_full)

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    def chat_perform(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(turns, tools, data_model, kwargs)
        if stream:
            return self._client.models.generate_content_stream(**kwargs)
        else:
            return self._client.models.generate_content(**kwargs)

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(turns, tools, data_model, kwargs)
        if stream:
            return await self._client.aio.models.generate_content_stream(**kwargs)
        else:
            return await self._client.aio.models.generate_content(**kwargs)

    def _chat_perform_args(
        self,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        from google.genai.types import FunctionDeclaration, GenerateContentConfig
        from google.genai.types import Tool as GoogleTool

        kwargs_full: "SubmitInputArgs" = {
            "model": self._model,
            "contents": cast("GoogleContent", self._google_contents(turns)),
            **(kwargs or {}),
        }

        config = kwargs_full.get("config")
        if config is None:
            config = GenerateContentConfig()
        if isinstance(config, dict):
            config = GenerateContentConfig.model_construct(**config)

        if config.system_instruction is None:
            if len(turns) > 0 and turns[0].role == "system":
                config.system_instruction = turns[0].text

        if data_model:
            config.response_schema = data_model
            config.response_mime_type = "application/json"

        if tools:
            config.tools = [
                GoogleTool(
                    function_declarations=[
                        FunctionDeclaration.from_callable(
                            client=self._client, callable=tool.func
                        )
                        for tool in tools.values()
                    ]
                )
            ]

        kwargs_full["config"] = config

        return kwargs_full

    def stream_text(self, chunk) -> Optional[str]:
        try:
            # Errors if there is no text (e.g., tool request)
            return chunk.text
        except Exception:
            return None

    def stream_merge_chunks(self, completion, chunk):
        chunkd = chunk.model_dump()
        if completion is None:
            return cast("GenerateContentResponseDict", chunkd)
        return cast(
            "GenerateContentResponseDict",
            merge_dicts(completion, chunkd),  # type: ignore
        )

    def stream_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(
            completion,
            has_data_model,
        )

    def value_turn(self, completion, has_data_model) -> Turn:
        completion = cast("GenerateContentResponseDict", completion.model_dump())
        return self._as_turn(completion, has_data_model)

    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ):
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )

        res = self._client.models.count_tokens(**kwargs)
        return res.total_tokens or 0

    async def token_count_async(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ):
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )

        res = await self._client.aio.models.count_tokens(**kwargs)
        return res.total_tokens or 0

    def _token_count_args(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> dict[str, Any]:
        turn = user_turn(*args)

        kwargs = self._chat_perform_args(
            turns=[turn],
            tools=tools,
            data_model=data_model,
        )

        args_to_keep = ["model", "contents", "tools"]

        return {arg: kwargs[arg] for arg in args_to_keep if arg in kwargs}

    def _google_contents(self, turns: list[Turn]) -> list["GoogleContent"]:
        from google.genai.types import Content as GoogleContent

        contents: list["GoogleContent"] = []
        for turn in turns:
            if turn.role == "system":
                continue  # System messages are handled separately
            elif turn.role == "user":
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append(GoogleContent(role=turn.role, parts=parts))
            elif turn.role == "assistant":
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append(GoogleContent(role="model", parts=parts))
            else:
                raise ValueError(f"Unknown role {turn.role}")
        return contents

    def _as_part_type(self, content: Content) -> "Part":
        from google.genai.types import FunctionCall, FunctionResponse, Part

        if isinstance(content, ContentText):
            return Part.from_text(text=content.text)
        elif isinstance(content, ContentJson):
            return Part.from_text(text="<structured data/>")
        elif isinstance(content, ContentImageInline) and content.data:
            return Part.from_bytes(
                data=base64.b64decode(content.data),
                mime_type=content.image_content_type,
            )
        elif isinstance(content, ContentImageRemote):
            raise NotImplementedError(
                "Remote images aren't supported by Google (Gemini). "
                "Consider downloading the image and using content_image_file() instead."
            )
        elif isinstance(content, ContentToolRequest):
            return Part(
                function_call=FunctionCall(
                    id=content.id if content.name != content.id else None,
                    name=content.name,
                    # Goes in a dict, so should come out as a dict
                    args=cast(dict[str, Any], content.arguments),
                )
            )
        elif isinstance(content, ContentToolResult):
            if content.error:
                resp = {"error": content.error}
            else:
                resp = {"result": str(content.value)}
            return Part(
                # TODO: seems function response parts might need role='tool'???
                # https://github.com/googleapis/python-genai/blame/c8cfef85c/README.md#L344
                function_response=FunctionResponse(
                    id=content.id if content.name != content.id else None,
                    name=content.name,
                    response=resp,
                )
            )
        raise ValueError(f"Unknown content type: {type(content)}")

    def _as_turn(
        self,
        message: "GenerateContentResponseDict",
        has_data_model: bool,
    ) -> Turn:
        from google.genai.types import FinishReason

        candidates = message.get("candidates")
        if not candidates:
            return Turn("assistant", "")

        parts: list["PartDict"] = []
        finish_reason = None
        for candidate in candidates:
            content = candidate.get("content")
            if content:
                parts.extend(content.get("parts") or {})
            finish = candidate.get("finish_reason")
            if finish:
                finish_reason = finish

        contents: list[Content] = []
        for part in parts:
            text = part.get("text")
            if text:
                if has_data_model:
                    contents.append(ContentJson(json.loads(text)))
                else:
                    contents.append(ContentText(text))
            function_call = part.get("function_call")
            if function_call:
                # Seems name is required but id is optional?
                name = function_call.get("name")
                if name:
                    contents.append(
                        ContentToolRequest(
                            id=function_call.get("id") or name,
                            name=name,
                            arguments=function_call.get("args"),
                        )
                    )
            function_response = part.get("function_response")
            if function_response:
                # Seems name is required but id is optional?
                name = function_response.get("name")
                if name:
                    contents.append(
                        ContentToolResult(
                            id=function_response.get("id") or name,
                            value=function_response.get("response"),
                            name=name,
                        )
                    )

        usage = message.get("usage_metadata")
        tokens = (0, 0)
        if usage:
            tokens = (
                usage.get("prompt_token_count") or 0,
                usage.get("candidates_token_count") or 0,
            )

        tokens_log(self, tokens)

        if isinstance(finish_reason, FinishReason):
            finish_reason = finish_reason.name

        return Turn(
            "assistant",
            contents,
            tokens=tokens,
            finish_reason=finish_reason,
            completion=message,
        )


def ChatVertex(
    *,
    model: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", GenerateContentResponse]:
    """
    Chat with a Google Vertex AI model.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Python requirements

    `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[vertex]"`.
    :::

    ::: {.callout-note}
    ## Credentials

    To use Google's models (i.e., Vertex AI), you'll need to sign up for an account
    with [Vertex AI](https://cloud.google.com/vertex-ai), then specify the appropriate
    model, project, and location.
    :::

    Parameters
    ----------
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    project
        The Google Cloud project ID (e.g., "your-project-id"). If not provided, the
        GOOGLE_CLOUD_PROJECT environment variable will be used.
    location
        The Google Cloud location (e.g., "us-central1"). If not provided, the
        GOOGLE_CLOUD_LOCATION environment variable will be used.
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch.
        Do not provide non-`None` values for both `turns` and `system_prompt`.
        Each message in the list should be a dictionary with at least `role`
        (usually `system`, `user`, or `assistant`, but `tool` is also possible).
        Normally there is also a `content` field, which is a string.

    Returns
    -------
    Chat
        A Chat object.

    Examples
    --------

    ```python
    import os
    from chatlas import ChatVertex

    chat = ChatVertex(
        project="your-project-id",
        location="us-central1",
    )
    chat.chat("What is the capital of France?")
    ```
    """

    if kwargs is None:
        kwargs = {}

    kwargs["vertexai"] = True
    kwargs["project"] = project
    kwargs["location"] = location

    if model is None:
        model = log_model_default("gemini-2.0-flash")

    return Chat(
        provider=GoogleProvider(
            model=model,
            api_key=api_key,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt=system_prompt,
        ),
    )

from typing import TYPE_CHECKING, Literal, Optional, overload

from ._chat import Chat
from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._provider import Provider, ToolDef
from ._tokens import tokens_log
from ._turn import Turn, normalize_turns
from ._utils import inform_model_default

if TYPE_CHECKING:
    from google.generativeai.types.content_types import (
        ContentDict,
        FunctionDeclaration,
        PartType,
    )
    from google.generativeai.types.generation_types import GenerateContentResponse

    from .types._google_client import ProviderClientArgs
    from .types._google_create import SendMessageArgs
else:
    GenerateContentResponse = object


def ChatGoogle(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    kwargs: Optional["ProviderClientArgs"] = None,
) -> Chat["SendMessageArgs"]:
    """
    Chat with a Google Gemini model

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
        Additional arguments to pass to the `genai.GenerativeModel` constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    if model is None:
        model = inform_model_default("gemini-1.5-flash")

    turns = normalize_turns(
        turns or [],
        system_prompt=system_prompt,
    )

    return Chat(
        provider=GoogleProvider(
            turns=turns,
            model=model,
            api_key=api_key,
            kwargs=kwargs,
        ),
        turns=turns,
    )


class GoogleProvider(
    Provider[GenerateContentResponse, GenerateContentResponse, GenerateContentResponse]
):
    def __init__(
        self,
        *,
        turns: list[Turn],
        model: str,
        api_key: str | None,
        kwargs: Optional["ProviderClientArgs"],
    ):
        try:
            from google.generativeai import GenerativeModel
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `google-generativeai` package. "
                "Install it with `pip install google-generativeai`."
            )

        if api_key is not None:
            import google.generativeai as genai

            genai.configure(api_key=api_key)

        system_prompt = None
        if len(turns) > 0 and turns[0].role == "system":
            system_prompt = turns[0].text

        kwargs_full: "ProviderClientArgs" = {
            "model_name": model,
            "system_instruction": system_prompt,
            **(kwargs or {}),
        }

        self._client = GenerativeModel(**kwargs_full)

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ): ...

    def chat_perform(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, kwargs)
        return self._client.generate_content(**kwargs)

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, kwargs)
        return await self._client.generate_content_async(**kwargs)

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["SendMessageArgs"] = None,
    ) -> "SendMessageArgs":
        return {
            "contents": self._google_contents(turns),
            "stream": stream,
            "tools": self._gemini_tools(list(tools.values())) if tools else None,
            **(kwargs or {}),
        }

    def stream_text(self, chunk) -> Optional[str]:
        if chunk.parts:
            return chunk.text
        return None

    def stream_merge_chunks(self, completion, chunk):
        return chunk

    def stream_turn(self, completion) -> Turn:
        return self._as_turn(completion)

    def value_turn(self, completion) -> Turn:
        return self._as_turn(completion)

    def _google_contents(self, turns: list[Turn]) -> list["ContentDict"]:
        contents: list["ContentDict"] = []
        for turn in turns:
            if turn.role == "system":
                continue  # System messages are handled separately
            elif turn.role == "user":
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append({"role": turn.role, "parts": parts})
            elif turn.role == "assistant":
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append({"role": "model", "parts": parts})
            else:
                raise ValueError(f"Unknown role {turn.role}")
        return contents

    def _as_part_type(self, content: Content) -> "PartType":
        from google.generativeai.types.content_types import protos

        if isinstance(content, ContentText):
            return protos.Part(text=content.text)
        elif isinstance(content, ContentImageInline):
            return protos.Part(
                inline_data={
                    "mime_type": content.content_type,
                    "data": content.data,
                }
            )
        elif isinstance(content, ContentImageRemote):
            raise NotImplementedError(
                "Remote images aren't supported by Google (Gemini). "
                "Consider downloading the image and using content_image_file() instead."
            )
        elif isinstance(content, ContentToolRequest):
            return protos.Part(
                function_call={
                    "name": content.id,
                    "args": content.arguments,
                }
            )
        elif isinstance(content, ContentToolResult):
            return protos.Part(
                function_response={
                    "name": content.id,
                    "response": {
                        "value": str(content.value)
                        if content.value is not None
                        else content.error
                    },
                }
            )
        raise ValueError(f"Unknown content type: {type(content)}")

    def _as_turn(self, message: "GenerateContentResponse") -> Turn:
        contents = []
        for part in message.parts:
            if part.text:
                contents.append(ContentText(part.text))
            if part.function_call:
                func = part.function_call
                contents.append(
                    ContentToolRequest(
                        func.name,
                        name=func.name,
                        arguments=dict(func.args),
                    )
                )
            if part.function_response:
                func = part.function_response
                contents.append(
                    ContentToolResult(
                        func.name,
                        value=func.response,
                    )
                )

        usage = message.usage_metadata
        tokens = (
            usage.prompt_token_count,
            usage.candidates_token_count,
        )

        tokens_log("Google", tokens)

        return Turn("assistant", contents, tokens=tokens)

    def _gemini_tools(self, tools: list[ToolDef]) -> list["FunctionDeclaration"]:
        from google.generativeai.types.content_types import FunctionDeclaration

        res: list["FunctionDeclaration"] = []
        for tool in tools:
            fn = tool.schema["function"]
            params = None
            if fn["parameters"]["properties"]:
                params = {
                    "type": "object",
                    "properties": fn["parameters"]["properties"],
                    "required": fn["parameters"]["required"],
                }

            res.append(
                FunctionDeclaration(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=params,
                )
            )

        return res

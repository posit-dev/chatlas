from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Literal, overload

from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
    ContentUnion,
)
from ._content_pdf import parse_data_url
from ._turn import AssistantTurn, SystemTurn, Turn, UserTurn

if TYPE_CHECKING:
    import inspect_ai.model as i_model
    import inspect_ai.solver as i_solver
    import inspect_ai.tool as i_tool
    from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem
    from inspect_ai.tool import Content as InspectContent
    from inspect_ai.tool import ToolCall

Role = Literal["system", "user", "assistant"]


@overload
def turn_as_inspect_messages(
    turn: Turn, role: Literal["system"], model: str | None = None
) -> list[ChatMessageSystem]: ...


@overload
def turn_as_inspect_messages(
    turn: Turn, role: Literal["user"], model: str | None = None
) -> list[ChatMessage]: ...


@overload
def turn_as_inspect_messages(
    turn: Turn, role: Literal["assistant"], model: str | None = None
) -> list[ChatMessageAssistant]: ...


def turn_as_inspect_messages(
    turn: Turn, role: Role, model: str | None = None
) -> list[ChatMessageSystem] | list[ChatMessage] | list[ChatMessageAssistant]:
    """
    Translate a chatlas Turn into InspectAI ChatMessages.

    Parameters
    ----------
    turn
        The chatlas Turn to convert
    model
        The model name to include in assistant messages
    """
    (imodel, _, itool) = try_import_inspect()

    if isinstance(turn, SystemTurn):
        return [imodel.ChatMessageSystem(content=turn.text)]

    if isinstance(turn, UserTurn):
        tool_results: list[ContentToolResult] = []
        other_contents: list[InspectContent] = []
        for x in turn.contents:
            if isinstance(x, ContentToolResult):
                tool_results.append(x)
            else:
                other_contents.append(chatlas_content_as_inspect(x))

        res: list[ChatMessage] = []
        for x in tool_results:
            res.append(
                imodel.ChatMessageTool(
                    tool_call_id=x.id,
                    content=str(x.get_model_value()),
                    function=x.name,
                )
            )
        if other_contents:
            res.append(imodel.ChatMessageUser(content=other_contents))
        return res

    if isinstance(turn, AssistantTurn):
        tool_calls: list[ToolCall] = []
        other_contents: list[InspectContent] = []
        for x in turn.contents:
            if isinstance(x, ContentToolRequest):
                tool_calls.append(
                    itool.ToolCall(
                        id=x.id,
                        function=x.name,
                        arguments=(
                            x.arguments
                            if isinstance(x.arguments, dict)
                            else {"value": x.arguments}
                        ),
                    )
                )
            else:
                other_contents.append(chatlas_content_as_inspect(x))

        return [
            imodel.ChatMessageAssistant(
                source="generate",
                content=other_contents,
                tool_calls=tool_calls,
                model=model,
            )
        ]

    raise ValueError(f"Unknown turn role: {turn.role}")


def inspect_messages_as_turns(messages: list[ChatMessage]) -> list[Turn]:
    """Translate InspectAI ChatMessages into chatlas Turns."""
    (imodel, _, _) = try_import_inspect()

    turns: list[Turn] = []
    for msg in messages:
        if isinstance(msg, imodel.ChatMessageSystem):
            contents = [inspect_content_as_chatlas(x) for x in msg.content]
            turn = Turn(role="system", contents=contents)
        elif isinstance(msg, imodel.ChatMessageUser):
            contents = [inspect_content_as_chatlas(x) for x in msg.content]
            turn = Turn(role="user", contents=contents)
        elif isinstance(msg, imodel.ChatMessageAssistant):
            contents: list[Content] = []
            tool_calls = msg.tool_calls or []
            for x in tool_calls:
                contents.append(
                    ContentToolRequest(id=x.id, name=x.function, arguments=x.arguments)
                )
            for content in msg.content:
                contents.append(inspect_content_as_chatlas(content))
            turn = Turn(role="assistant", contents=contents)
        elif isinstance(msg, imodel.ChatMessageTool):
            contents = [
                ContentToolResult(
                    value=msg.content,
                    error=Exception(msg.error.message) if msg.error else None,
                    request=ContentToolRequest(
                        id=msg.tool_call_id or "",
                        name=msg.function or "",
                        arguments={},
                    ),
                )
            ]
            turn = Turn(role="user", contents=contents)
        else:
            raise ValueError(f"Unknown InspectAI ChatMessage type: {type(msg)}")
        turns.append(turn)

    return turns


def chatlas_content_as_inspect(content: ContentUnion) -> InspectContent:
    """Translate chatlas Content into InspectAI Content."""
    (_, _, itool) = try_import_inspect()

    if isinstance(content, ContentText):
        return itool.ContentText(text=content.text)
    elif isinstance(content, ContentImageRemote):
        return itool.ContentImage(image=content.url, detail=content.detail)
    elif isinstance(content, ContentImageInline):
        # Reconstruct the data URL from the base64 data and content type
        data_url = f"data:{content.image_content_type};base64,{content.data or ''}"
        return itool.ContentImage(image=data_url, detail="auto")
    elif isinstance(content, ContentPDF):
        doc = content.url
        if doc is None:
            doc = f"data:application/pdf;base64,{base64.b64encode(content.data).decode('ascii')}"
        return itool.ContentDocument(
            document=doc,
            mime_type="application/pdf",
            filename=content.filename,
        )
    elif isinstance(content, ContentJson):
        return itool.ContentData(data=content.value)
    elif isinstance(content, (ContentToolRequest, ContentToolResult)):
        # Tool request/results need to be handled at the Turn level
        # (i.e., by turn_as_messages)
        raise ValueError(
            f"Content of type {type(content)} cannot be directly translated to InspectAI content"
        )
    else:
        raise ValueError(
            f"Don't know how to translate chatlas content type of {type(content)} to InspectAI content"
        )


def inspect_content_as_chatlas(content: str | InspectContent) -> Content:
    """Translate InspectAI Content into chatlas Content."""
    (_, _, itool) = try_import_inspect()

    if isinstance(content, str):
        return ContentText(text=content)
    if isinstance(content, itool.ContentText):
        return ContentText(text=content.text)
    if isinstance(content, itool.ContentImage):
        if content.image.startswith("http://") or content.image.startswith("https://"):
            return ContentImageRemote(url=content.image, detail=content.detail)
        else:
            # Parse data URL to extract content type and base64 data
            # e.g., data:image/png;base64,....
            content_type, base64_data = parse_data_url(content.image)
            return ContentImageInline(
                data=base64_data,
                image_content_type=content_type,  # type: ignore
            )
    if isinstance(content, itool.ContentDocument):
        doc = content.document
        if content.mime_type == "application/pdf":
            url = None
            if doc.startswith("http://") or doc.startswith("https://"):
                url = doc
                data = b""
            else:
                data = base64.b64decode(doc.split(",", 1)[1])
            return ContentPDF(data=data, url=url, filename=content.filename)
        else:
            return ContentText(text=doc)
    if isinstance(content, itool.ContentData):
        return ContentJson(value=content.data)
    raise ValueError(
        f"Inspect AI content of type {type(content)} is not currently supported by chatlas"
    )


def try_import_inspect() -> "tuple[i_model, i_solver, i_tool]":  # pyright: ignore[reportInvalidTypeForm]
    try:
        import inspect_ai.model as imodel
        import inspect_ai.solver as isolver
        import inspect_ai.tool as itool

        return imodel, isolver, itool  # pyright: ignore[reportReturnType]
    except ImportError as e:
        raise ImportError(
            "This functionality requires `inspect-ai`. "
            "Install it with `pip install inspect-ai`."
        ) from e

from __future__ import annotations

from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._turn import Turn

try:
    import inspect_ai.model as imodel
    import inspect_ai.tool as itool

    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False


def turn_as_messages(turn: Turn, model: str | None = None) -> list:
    """
    Translate a chatlas Turn into InspectAI ChatMessages.

    Parameters
    ----------
    turn
        The chatlas Turn to convert
    model
        The model name to include in assistant messages

    Returns
    -------
    list
        A list of InspectAI ChatMessage objects

    Raises
    ------
    ImportError
        If inspect_ai is not installed
    ValueError
        If the turn has an unknown role
    """
    if not INSPECT_AVAILABLE:
        raise ImportError(
            "InspectAI integration requires the optional dependency `inspect-ai`. "
            "Install it with `pip install inspect-ai`."
        )

    if turn.role == "system":
        return [imodel.ChatMessageSystem(content=turn.text)]

    if turn.role == "user":
        tool_results: list[ContentToolResult] = []
        other_contents: list[itool.Content] = []
        for x in turn.contents:
            if isinstance(x, ContentToolResult):
                tool_results.append(x)
            else:
                other_contents.append(content_to_inspect(x))

        res: list[imodel.ChatMessage] = []
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

    if turn.role == "assistant":
        tool_calls: list[itool.ToolCall] = []
        other_contents: list[itool.Content] = []
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
                other_contents.append(content_to_inspect(x))

        return [
            imodel.ChatMessageAssistant(
                content=other_contents,
                tool_calls=tool_calls,
                model=model,
            )
        ]

    raise ValueError(f"Unknown turn role: {turn.role}")


def content_to_inspect(content: Content) -> itool.Content:
    """
    Translate chatlas Content into InspectAI Content.

    Parameters
    ----------
    content
        The chatlas Content object to convert

    Returns
    -------
    itool.Content
        An InspectAI Content object

    Raises
    ------
    ImportError
        If inspect_ai is not installed
    ValueError
        If the content type cannot be translated
    """
    if not INSPECT_AVAILABLE:
        raise ImportError(
            "InspectAI integration requires the optional dependency `inspect-ai`. "
            "Install it with `pip install inspect-ai`."
        )

    if isinstance(content, ContentText):
        return itool.ContentText(text=content.text)
    elif isinstance(content, ContentImageRemote):
        return itool.ContentImage(image=content.url, detail=content.detail)
    elif isinstance(content, ContentImageInline):
        return itool.ContentImage(image=content.data or "", detail="auto")
    elif isinstance(content, ContentPDF):
        return itool.ContentDocument(
            document=content.data.decode("utf-8"),
            mime_type="application/pdf",
        )
    elif isinstance(content, ContentJson):
        return itool.ContentData(data=content.value)
    elif isinstance(content, (ContentToolRequest, ContentToolResult)):
        raise ValueError(
            f"Content of type {type(content)} cannot be directly translated to InspectAI content"
        )
    else:
        raise ValueError(
            f"Don't know how to translate chatlas content type of {type(content)} to InspectAI content"
        )


def content_to_chatlas(content: str | itool.Content) -> Content:
    """
    Translate InspectAI Content into chatlas Content.

    Parameters
    ----------
    content
        The InspectAI Content object or string to convert

    Returns
    -------
    Content
        A chatlas Content object

    Raises
    ------
    ImportError
        If inspect_ai is not installed
    ValueError
        If the content type is not supported
    """
    if not INSPECT_AVAILABLE:
        raise ImportError(
            "InspectAI integration requires the optional dependency `inspect-ai`. "
            "Install it with `pip install inspect-ai`."
        )

    if isinstance(content, str):
        return ContentText(text=content)
    if isinstance(content, itool.ContentText):
        return ContentText(text=content.text)
    if isinstance(content, itool.ContentImage):
        if content.image.startswith("http://") or content.image.startswith("https://"):
            return ContentImageRemote(url=content.image, detail=content.detail)
        else:
            # derive content_type from base64 data
            # e.g., data:image/png;base64,....
            content_type = content.image.split(":")[1].split(";")[0]
            return ContentImageInline(
                data=content.image,
                image_content_type=content_type,  # type: ignore
            )
    if isinstance(content, itool.ContentDocument):
        if content.mime_type == "application/pdf":
            return ContentPDF(data=content.document.encode("utf-8"))
        else:
            return ContentText(text=content.document)
    if isinstance(content, itool.ContentData):
        return ContentJson(value=content.data)
    raise ValueError(
        f"Inspect AI content of type {type(content)} is not currently supported by chatlas"
    )

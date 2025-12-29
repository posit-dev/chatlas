"""
Tool result content expansion logic.

Very few providers support anything other than text results from tools.
Fortunately, we can fake them by unrolling the tool result to a forward
pointer to other user content items.

For example:
    ContentToolResult(value=ContentImageInline(...))

becomes:
    ContentToolResult("See <tool-content call-id='xyz'> below")
    ContentText("<tool-content call-id='xyz'>")
    ContentImageInline(...)
    ContentText("</tool-content>")
"""

from __future__ import annotations

from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentPDF,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
    ContentUnion,
)
from ._typing_extensions import TypeGuard


def expand_tool_result(content: ContentToolResult) -> list[ContentUnion]:
    """Expand a tool result that contains images/PDFs into separate content items."""
    request = content.request
    if request is None:
        return [content]

    value = content.value
    if is_image_or_pdf_content(value):
        return expand_tool_value(request, value)

    if isinstance(value, (list, tuple)) and any(
        is_image_or_pdf_content(x) for x in value
    ):
        if all(isinstance(x, (Content, str)) for x in value):
            return expand_tool_values(request, list(value))

    return [content]


def expand_tool_value(
    request: ContentToolRequest, value: ContentImageInline | ContentImageRemote | ContentPDF
) -> list[ContentUnion]:
    open_tag = f'<tool-content call-id="{request.id}">'

    return [
        ContentToolResult(
            value=f"See {open_tag} below.",
            request=request,
        ),
        ContentText(text=open_tag),
        value,
        ContentText(text="</tool-content>"),
    ]


def expand_tool_values(
    request: ContentToolRequest, values: list[Content | str]
) -> list[ContentUnion]:
    """Expand a tool result containing a list of images or PDFs."""
    open_tag = f'<tool-contents call-id="{request.id}">'

    expanded = [
        ContentToolResult(
            value=f"See {open_tag} below.",
            request=request,
        ),
        ContentText(text=open_tag),
    ]

    # Add each value wrapped in its own tags
    for item in values:
        expanded.extend(
            [
                ContentText(text="<tool-content>"),
                item if isinstance(item, Content) else ContentText(text=item),
                ContentText(text="</tool-content>"),
            ]
        )

    expanded.append(ContentText(text="</tool-contents>"))

    return expanded


def is_image_or_pdf_content(
    content: Content,
) -> TypeGuard[ContentImageInline | ContentImageRemote | ContentPDF]:
    """Check if content is an image or PDF type."""
    return isinstance(
        content,
        (ContentImageInline, ContentImageRemote, ContentPDF),
    )

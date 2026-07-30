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

from typing import Sequence

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


def merge_tool_results(contents: Sequence[ContentUnion]) -> list[ContentUnion]:
    """Combine results answering the same tool request into a single result.

    A tool can emit several results for one call: MCP tools do so whenever the
    server returns multiple content parts (text chunks plus a plot image, say).
    Providers key their wire-level result on the request id, so more than one
    result per id is rejected -- Anthropic, for instance, requires that "each
    tool_use must have a single result".

    The merged result takes the position of the first result in the group.
    """
    groups: dict[str, list[ContentToolResult]] = {}
    for x in contents:
        if isinstance(x, ContentToolResult) and x.request is not None:
            groups.setdefault(x.request.id, []).append(x)

    merged: list[ContentUnion] = []
    seen: set[str] = set()
    for x in contents:
        if not isinstance(x, ContentToolResult) or x.request is None:
            merged.append(x)
            continue
        request_id = x.request.id
        if request_id in seen:
            continue
        seen.add(request_id)
        group = groups[request_id]
        merged.append(group[0] if len(group) == 1 else combine_tool_results(group))

    return merged


def combine_tool_results(results: list[ContentToolResult]) -> ContentToolResult:
    """Fold a group of same-request results into one.

    Each part contributes the value it would have sent on its own, so a part's
    `model_format` is honored. `extra` is per-part display metadata -- every
    part was already delivered to callbacks with its own -- and there is no
    non-arbitrary way to merge several, so the combined result carries none.
    """
    request = results[0].request

    error = next((r.error for r in results if r.error is not None), None)
    if error is not None:
        return ContentToolResult(value=None, error=error, request=request)

    # An image/PDF value bypasses model_format rendering here: expand_tool_result
    # unrolls it by its raw value regardless of model_format, exactly as it does
    # for a single (non-combined) result, so rendering it here would only
    # destroy the Content object before that unrolling ever sees it.
    values = [
        r.value if is_image_or_pdf_content(r.value) else r.get_model_value()
        for r in results
    ]

    # expand_tool_result only unrolls a list when it holds image/PDF content
    # (that's the only case providers can't take inline), so the guard here
    # must match exactly -- any other Content (e.g. ContentJson) would
    # otherwise take the list route and reach the provider as a raw list.
    if any(is_image_or_pdf_content(v) for v in values):
        parts: list[Content | str] = [
            v if isinstance(v, (Content, str)) else str(v) for v in values
        ]
        return ContentToolResult(value=parts, model_format="as_is", request=request)

    return ContentToolResult(value="\n".join(str(v) for v in values), request=request)


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
    request: ContentToolRequest,
    value: ContentImageInline | ContentImageRemote | ContentPDF,
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
    content: object,
) -> TypeGuard[ContentImageInline | ContentImageRemote | ContentPDF]:
    """Check if content is an image or PDF type."""
    return isinstance(
        content,
        (ContentImageInline, ContentImageRemote, ContentPDF),
    )

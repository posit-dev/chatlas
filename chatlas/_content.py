from __future__ import annotations

import json
from pprint import pformat
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

ImageContentTypes = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]
"""
Allowable content types for images.
"""

ContentTypeEnum = Literal[
    "text",
    "image_remote",
    "image_inline",
    "tool_request",
    "tool_result",
    "json",
    "pdf",
]
"""
A discriminated union of all content types.
"""


class Content(BaseModel):
    """
    Base class for all content types that can be appear in a [](`~chatlas.Turn`)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    content_type: ContentTypeEnum

    def __str__(self):
        raise NotImplementedError

    def _repr_markdown_(self):
        raise NotImplementedError

    def __repr__(self, indent: int = 0):
        raise NotImplementedError


class ContentText(Content):
    """
    Text content for a [](`~chatlas.Turn`)
    """

    text: str
    content_type: ContentTypeEnum = "text"

    def __str__(self):
        return self.text

    def _repr_markdown_(self):
        return self.text

    def __repr__(self, indent: int = 0):
        text = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return " " * indent + f"<ContentText text='{text}'>"


class ContentImage(Content):
    """
    Base class for image content.

    This class is not meant to be used directly. Instead, use
    [](`~chatlas.content_image_url`), [](`~chatlas.content_image_file`), or
    [](`~chatlas.content_image_plot`).
    """

    pass


class ContentImageRemote(ContentImage):
    """
    Image content from a URL.

    This is the return type for [](`~chatlas.content_image_url`).
    It's not meant to be used directly.

    Parameters
    ----------
    url
        The URL of the image.
    detail
        A detail setting for the image. Can be `"auto"`, `"low"`, or `"high"`.
    """

    url: str
    detail: Literal["auto", "low", "high"] = "auto"

    content_type: ContentTypeEnum = "image_remote"

    def __str__(self):
        return f"![]({self.url})"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        return (
            " " * indent
            + f"<ContentImageRemote url='{self.url}' detail='{self.detail}'>"
        )


class ContentImageInline(ContentImage):
    """
    Inline image content.

    This is the return type for [](`~chatlas.content_image_file`) and
    [](`~chatlas.content_image_plot`).
    It's not meant to be used directly.

    Parameters
    ----------
    image_content_type
        The content type of the image.
    data
        The base64-encoded image data.
    """

    image_content_type: ImageContentTypes
    data: Optional[str] = None

    content_type: ContentTypeEnum = "image_inline"

    def __str__(self):
        return f"![](data:{self.image_content_type};base64,{self.data})"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        n_bytes = len(self.data) if self.data else 0
        return (
            " " * indent
            + f"<ContentImageInline content_type='{self.image_content_type}' size={n_bytes}>"
        )


class ContentToolRequest(Content):
    """
    A request to call a tool/function

    This content type isn't meant to be used directly. Instead, it's
    automatically generated by [](`~chatlas.Chat`) when a tool/function is
    requested by the model assistant.

    Parameters
    ----------
    id
        A unique identifier for this request.
    name
        The name of the tool/function to call.
    arguments
        The arguments to pass to the tool/function.
    """

    id: str
    name: str
    arguments: object

    content_type: ContentTypeEnum = "tool_request"

    def __str__(self):
        args_str = self._arguments_str()
        func_call = f"{self.name}({args_str})"
        comment = f"# tool request ({self.id})"
        return f"```python\n{comment}\n{func_call}\n```\n"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        args_str = self._arguments_str()
        return (
            " " * indent
            + f"<ContentToolRequest name='{self.name}' arguments='{args_str}' id='{self.id}'>"
        )

    def _arguments_str(self) -> str:
        if isinstance(self.arguments, dict):
            return ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return str(self.arguments)


class ContentToolResult(Content):
    """
    The result of calling a tool/function

    This content type isn't meant to be used directly. Instead, it's
    automatically generated by [](`~chatlas.Chat`) when a tool/function is
    called (in response to a [](`~chatlas.ContentToolRequest`)).

    Parameters
    ----------
    id
        The unique identifier of the tool request.
    value
        The value returned by the tool/function.
    name
        The name of the tool/function that was called.
    error
        An error message if the tool/function call failed.
    """

    id: str
    value: Any = None
    name: Optional[str] = None
    error: Optional[str] = None

    content_type: ContentTypeEnum = "tool_result"

    def _get_value(self, pretty: bool = False) -> str:
        if self.error:
            return f"Tool calling failed with error: '{self.error}'"
        if not pretty:
            return str(self.value)
        try:
            json_val = json.loads(self.value)  # type: ignore
            return pformat(json_val, indent=2, sort_dicts=False)
        except:  # noqa
            return str(self.value)

    # Primarily used for `echo="all"`...
    def __str__(self):
        comment = f"# tool result ({self.id})"
        value = self._get_value(pretty=True)
        return f"""```python\n{comment}\n{value}\n```"""

    # ... and for displaying in the notebook
    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        res = " " * indent
        res += f"<ContentToolResult value='{self.value}' id='{self.id}'"
        if self.error:
            res += f" error='{self.error}'"
        return res + ">"

    # The actual value to send to the model
    def get_final_value(self) -> str:
        return self._get_value()


class ContentJson(Content):
    """
    JSON content

    This content type primarily exists to signal structured data extraction
    (i.e., data extracted via [](`~chatlas.Chat`)'s `.extract_data()` method)

    Parameters
    ----------
    value
        The JSON data extracted
    """

    value: dict[str, Any]

    content_type: ContentTypeEnum = "json"

    def __str__(self):
        return json.dumps(self.value, indent=2)

    def _repr_markdown_(self):
        return f"""```json\n{self.__str__()}\n```"""

    def __repr__(self, indent: int = 0):
        return " " * indent + f"<ContentJson value={self.value}>"


class ContentPDF(Content):
    """
    PDF content

    This content type primarily exists to signal PDF data extraction
    (i.e., data extracted via [](`~chatlas.Chat`)'s `.extract_data()` method)

    Parameters
    ----------
    value
        The PDF data extracted
    """

    data: bytes
    content_type: ContentTypeEnum = "pdf"

    def __str__(self):
        return "<PDF document>"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        return " " * indent + f"<ContentPDF size={len(self.data)}>"


ContentUnion = Union[
    ContentText,
    ContentImageRemote,
    ContentImageInline,
    ContentToolRequest,
    ContentToolResult,
    ContentJson,
    ContentPDF,
]


def create_content(data: dict[str, Any]) -> ContentUnion:
    """
    Factory function to create the appropriate Content subclass based on the data.

    This is useful when deserializing content from JSON.
    """
    if not isinstance(data, dict):
        raise ValueError("Content data must be a dictionary")

    ct = data.get("content_type")

    if ct == "text":
        return ContentText.model_validate(data)
    elif ct == "image_remote":
        return ContentImageRemote.model_validate(data)
    elif ct == "image_inline":
        return ContentImageInline.model_validate(data)
    elif ct == "tool_request":
        return ContentToolRequest.model_validate(data)
    elif ct == "tool_result":
        return ContentToolResult.model_validate(data)
    elif ct == "json":
        return ContentJson.model_validate(data)
    elif ct == "pdf":
        return ContentPDF.model_validate(data)
    else:
        raise ValueError(f"Unknown content type: {ct}")

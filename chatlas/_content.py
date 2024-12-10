from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

ImageContentTypes = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]
"""
Allowable content types for images.
"""


class Content:
    """
    Base class for all content types that can be appear in a [](`~chatlas.Turn`)
    """

    def __str__(self):
        raise NotImplementedError

    def _repr_markdown_(self):
        raise NotImplementedError

    def __repr__(self, indent: int = 0):
        raise NotImplementedError


@dataclass
class ContentText(Content):
    """
    Text content for a [](`~chatlas.Turn`)
    """

    text: str

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


@dataclass
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

    def __str__(self):
        return f"![]({self.url})"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        return (
            " " * indent
            + f"<ContentImageRemote url='{self.url}' detail='{self.detail}'>"
        )


@dataclass
class ContentImageInline(ContentImage):
    """
    Inline image content.

    This is the return type for [](`~chatlas.content_image_file`) and
    [](`~chatlas.content_image_plot`).
    It's not meant to be used directly.

    Parameters
    ----------
    content_type
        The content type of the image.
    data
        The base64-encoded image data.
    """

    content_type: ImageContentTypes
    data: Optional[str] = None

    def __str__(self):
        return f"![](data:{self.content_type};base64,{self.data})"

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        n_bytes = len(self.data) if self.data else 0
        return (
            " " * indent
            + f"<ContentImageInline content_type='{self.content_type}' size={n_bytes}>"
        )


@dataclass
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

    def __str__(self):
        args_str = self._arguments_str()
        func_call = f"{self.name}({args_str})"
        comment = f"# tool request ({self.id})"
        return f"\n```python\n{comment}\n{func_call}\n```\n"

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


@dataclass
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
    error
        An error message if the tool/function call failed.
    """

    id: str
    value: Any = None
    error: Optional[str] = None

    def __str__(self):
        comment = f"# tool result ({self.id})"
        val = self.get_final_value()
        return f"""\n```python\n{comment}\n"{val}"\n```\n"""

    def _repr_markdown_(self):
        return self.__str__()

    def __repr__(self, indent: int = 0):
        res = " " * indent
        res += f"<ContentToolResult value='{self.value}' id='{self.id}'"
        if self.error:
            res += f" error='{self.error}'"
        return res + ">"

    def get_final_value(self) -> str:
        if self.error:
            return f"Tool calling failed with error: '{self.error}'"
        return str(self.value)


@dataclass
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

    def __str__(self):
        return json.dumps(self.value, indent=2)

    def _repr_markdown_(self):
        return f"""\n```json\n{self.__str__()}\n```\n"""

    def __repr__(self, indent: int = 0):
        return " " * indent + f"<ContentJson value={self.value}>"

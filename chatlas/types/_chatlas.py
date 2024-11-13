from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    Literal,
    Optional,
    TypeVar,
)

from .._typing_extensions import TypedDict

__all__ = (
    "Content",
    "ContentImage",
    "ContentImageInline",
    "ContentImageRemote",
    "ContentJson",
    "ContentText",
    "ContentToolRequest",
    "ContentToolResult",
    "ChatResponse",
    "ChatResponseAsync",
    "ImageContentTypes",
    "SubmitInputArgsT",
    "TokenUsage",
    "MISSING_TYPE",
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------------
# Generic types
# ---------------------------------------------------------------------------------


class AnyTypeDict(TypedDict, total=False):
    pass


SubmitInputArgsT = TypeVar("SubmitInputArgsT", bound=AnyTypeDict)
"""
A TypedDict representing the arguments that can be passed to the `.chat()`
method of a [](`~chatlas.Chat`) instance.
"""


# ---------------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------------


class MISSING_TYPE:
    """
    A singleton representing a missing value.
    """

    pass


MISSING = MISSING_TYPE()


class TokenUsage(TypedDict):
    """
    Token usage for a given provider (name).
    """

    name: str
    input: int
    output: int


# ---------------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------------


class ChatResponse:
    """
    Chat response object.

    This class wraps a generator that yields strings, and provides a `display()`
    method to show the content in a rich console, and a `get_string()` method to
    get the content as a string.
    """

    def __init__(self, generator: Generator[str, None]):
        self.generator = generator
        self.content = ""

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        chunk = next(self.generator)
        self.content += chunk  # Keep track of accumulated content
        return chunk

    def display(self):
        "Display the content in a rich console."
        from rich.live import Live
        from rich.markdown import Markdown

        with JupyterFriendlyConsole() as console:
            with Live(console=console, auto_refresh=False) as live:
                needs_display = True
                for _ in self:
                    live.update(Markdown(self.content), refresh=True)
                    needs_display = False
                if needs_display:
                    live.update(Markdown(self.content), refresh=True)

    def get_string(self) -> str:
        "Get the chat response content as a string."
        for _ in self:
            pass
        return self.content

    def __str__(self) -> str:
        return self.get_string()

    def __repr__(self) -> str:
        return (
            "ChatResponse object. Call `.display()` to show it in a rich"
            "console or `.get_string()` to get the content."
        )


class ChatResponseAsync:
    """
    A string-like class that uses a custom display hook to display itself in the console.
    Inherits from UserString to provide complete string interface.
    """

    def __init__(self, generator: AsyncGenerator[str, None]):
        self.generator = generator
        self.content = ""

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        chunk = await self.generator.__anext__()
        self.content += chunk  # Keep track of accumulated content
        return chunk

    async def display(self) -> None:
        "Display the content in a rich console."
        from rich.live import Live
        from rich.markdown import Markdown

        with JupyterFriendlyConsole() as console:
            with Live(console=console, auto_refresh=False) as live:
                needs_display = True
                async for _ in self:
                    live.update(Markdown(self.content), refresh=True)
                    needs_display = False
                if needs_display:
                    live.update(Markdown(self.content), refresh=True)

    async def get_string(self) -> str:
        "Get the chat response content as a string."
        async for _ in self:
            pass
        return self.content

    def __repr__(self) -> str:
        return (
            "ChatResponseAsync object. Call `.display()` to show it in a rich"
            "console or `.get_string()` to get the content."
        )


original_displayhook = sys.displayhook


def custom_displayhook(value: Any):
    """Custom display hook that handles our special class"""
    if isinstance(value, ChatResponse):
        value.display()
    else:
        original_displayhook(value)


sys.displayhook = custom_displayhook


@contextmanager
def JupyterFriendlyConsole():
    import rich.jupyter
    from rich.console import Console

    console = Console()

    # Prevent rich from inserting line breaks in a Jupyter context
    # (and, instead, rely on the browser to wrap text)
    console.soft_wrap = console.is_jupyter

    html_format = rich.jupyter.JUPYTER_HTML_FORMAT

    # Remove the `white-space:pre;` CSS style since the LLM's response is
    # (usually) already pre-formatted and essentially assumes a browser context
    rich.jupyter.JUPYTER_HTML_FORMAT = html_format.replace(
        "white-space:pre;", "word-break:break-word;"
    )
    yield console

    rich.jupyter.JUPYTER_HTML_FORMAT = html_format


# ---------------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------------


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

    pass


@dataclass
class ContentText(Content):
    text: str


class ContentImage(Content):
    pass


@dataclass
class ContentImageRemote(ContentImage):
    url: str
    detail: str = ""

    def __str__(self):
        return f"[remote image]: {self.url}"


@dataclass
class ContentImageInline(ContentImage):
    content_type: ImageContentTypes
    data: Optional[str] = None

    def __str__(self):
        n_bytes = len(self.data) if self.data else 0
        return f"[inline image]: {self.content_type} ({n_bytes} bytes)"


@dataclass
class ContentToolRequest(Content):
    id: str
    name: str
    arguments: dict[str, Any]

    def __str__(self):
        args_str = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"[tool request ({self.id})]: {self.name}({args_str})"


@dataclass
class ContentToolResult(Content):
    id: str
    value: Any = None
    error: Optional[str] = None

    def __str__(self):
        if self.error:
            return f"[tool result ({self.id})]: Error: {self.error}"
        return f"[tool result ({self.id})]: {self.value}"

    def get_final_value(self) -> Any:
        if self.error:
            return f"Tool calling failed with error: '{self.error}'"
        return str(self.value)


@dataclass
class ContentJson(Content):
    value: dict[str, Any]

    def __str__(self):
        return json.dumps(self.value, indent=2)

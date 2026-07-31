from __future__ import annotations

import base64
import inspect
import warnings
from pprint import pformat
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast, get_args

import orjson
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_serializer,
    field_validator,
    model_validator,
)

from ._typing_extensions import TypedDict, TypeIs
from ._utils import format_bytes, html_escape, truncate_lines

if TYPE_CHECKING:
    from htmltools import Tagified

    from ._tools import Tool, ToolBuiltIn


class ToolAnnotations(TypedDict, total=False):
    """
    Additional properties describing a Tool to clients.

    NOTE: all properties in ToolAnnotations are **hints**.
    They are not guaranteed to provide a faithful description of
    tool behavior (including descriptive properties like `title`).

    Clients should never make tool use decisions based on ToolAnnotations
    received from untrusted servers.
    """

    title: str
    """A human-readable title for the tool."""

    readOnlyHint: bool
    """
    If true, the tool does not modify its environment.
    Default: false
    """

    destructiveHint: bool
    """
    If true, the tool may perform destructive updates to its environment.
    If false, the tool performs only additive updates.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: true
    """

    idempotentHint: bool
    """
    If true, calling the tool repeatedly with the same arguments
    will have no additional effect on the its environment.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: false
    """

    openWorldHint: bool
    """
    If true, this tool may interact with an "open world" of external
    entities. If false, the tool's domain of interaction is closed.
    For example, the world of a web search tool is open, whereas that
    of a memory tool is not.
    Default: true
    """

    extra: dict[str, Any]
    """
    Additional metadata about the tool.
    """


ImageContentTypes = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
]
"""
Allowable content types for images.

Note that not every provider accepts every type here: `image/heic` and
`image/heif` are only supported by `ChatGoogle()` today. Providers that can't
accept a given type raise a clear error rather than silently sending it.
"""

HeicHeifImageTypes = Literal["image/heic", "image/heif"]
NonHeicImageContentTypes = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]
"""
The subset of [](`~chatlas.types.ImageContentTypes`) every provider accepts.

Returned by `check_image_content_type_supported()` so providers whose SDKs type
`media_type` this narrowly can use the result without casting.
"""

IMAGE_CONTENT_TYPES: tuple[ImageContentTypes, ...] = get_args(ImageContentTypes)
HEIC_HEIF_IMAGE_TYPES: tuple[HeicHeifImageTypes, ...] = get_args(HeicHeifImageTypes)

DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
BINARY_DOCUMENT_MIME_TYPES = frozenset(
    {
        DOCX_MIME_TYPE,
        XLSX_MIME_TYPE,
        "application/rtf",
        "application/msword",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.ms-excel",
    }
)
"""
`ContentDocument` MIME types that only `ChatOpenAI()`/`ChatOpenAICompletions()`
accept -- Anthropic and Google can't extract text from these binary formats and
require converting to plain text or PDF first.
"""


def check_image_content_type_supported(
    provider_name: str, content_type: ImageContentTypes
) -> NonHeicImageContentTypes:
    """Return `content_type`, or raise if `provider_name` can't accept it.

    Only `ChatGoogle()` supports HEIC/HEIF today; every other provider rejects
    them outright rather than sending bytes the API will reject anyway. The
    return value is narrowed to what's left, so callers passing it to an SDK
    that only types the universal four don't need a cast.
    """
    if is_heic_heif(content_type):
        raise ValueError(
            f"{provider_name} doesn't support {content_type} images. Convert "
            "to image/png, image/jpeg, image/webp, or image/gif first, or "
            "use ChatGoogle(), which supports HEIC/HEIF natively."
        )
    return content_type


def is_heic_heif(content_type: ImageContentTypes) -> TypeIs[HeicHeifImageTypes]:
    return content_type in HEIC_HEIF_IMAGE_TYPES


def is_image_content_type(content_type: str) -> TypeIs[ImageContentTypes]:
    return content_type in IMAGE_CONTENT_TYPES


class ToolInfo(BaseModel):
    """
    Serializable tool information

    This contains only the serializable parts of a Tool that are needed
    for ContentToolRequest to be JSON-serializable. This allows tool
    metadata to be preserved without including the non-serializable
    function reference.

    Parameters
    ----------
    name
        The name of the tool.
    description
        A description of what the tool does.
    parameters
        A dictionary describing the input parameters and their types.
    annotations
        Additional properties that describe the tool and its behavior.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    annotations: Optional[ToolAnnotations] = None

    @classmethod
    def from_tool(cls, tool: "Tool | ToolBuiltIn") -> "ToolInfo":
        """Create a ToolInfo from a Tool or ToolBuiltIn instance."""
        from ._tools import ToolBuiltIn

        if isinstance(tool, ToolBuiltIn):
            return cls(name=tool.name, description=tool.name, parameters={})
        else:
            # For regular tools, extract from schema
            func_schema = tool.schema["function"]
            return cls(
                name=tool.name,
                description=func_schema.get("description", ""),
                parameters=func_schema.get("parameters", {}),
                annotations=tool.annotations,
            )


ContentTypeEnum = Literal[
    "text",
    "image_remote",
    "image_inline",
    "tool_request",
    "tool_result",
    "tool_result_image",
    "tool_result_resource",
    "json",
    "pdf",
    "document",
    "uploaded",
    "thinking",
    "thinking_delta",
    "web_search_request",
    "web_search_results",
    "web_fetch_request",
    "web_fetch_results",
    "citation",
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

    def __repr__(self):
        return self.__str__()

    def _repr_markdown_(self):
        return self.__str__()


SourceTypeEnum = Literal["web"]


class Source(BaseModel):
    """Identity of a piece of evidence a citation or search result points to.

    Subclasses set a distinct ``type`` and add their identity fields. Today the
    only concrete source is :class:`WebSource`; file/document/RAG variants are
    added when that support lands.
    """

    type: SourceTypeEnum

    def __str__(self) -> str:
        return f"[{self.type} source]"


class WebSource(Source):
    """A web page surfaced by a search (not necessarily cited in the answer)."""

    type: SourceTypeEnum = "web"
    url: str
    title: Optional[str] = None

    def __str__(self) -> str:
        return self.url or self.title or "[web source]"


class ContentText(Content):
    """
    Text content for a [](`~chatlas.Turn`)
    """

    text: str
    content_type: ContentTypeEnum = "text"

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.text == "" or self.text.isspace():
            self.text = "[empty string]"

    def __add__(self, other: object) -> "ContentText":
        if not isinstance(other, ContentText):
            return NotImplemented  # type: ignore[return-value]
        return ContentText.model_construct(text=self.text + other.text)

    def __str__(self):
        return self.text


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
    data: str

    content_type: ContentTypeEnum = "image_inline"

    def __str__(self):
        return f"![](data:{self.image_content_type};base64,{self.data})"


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
    tool
        Serializable information about the tool. This is set internally by
        chatlas's tool calling loop and contains only the metadata needed
        for serialization (name, description, parameters, annotations).
    """

    id: str
    name: str
    arguments: object
    tool: Optional[ToolInfo] = None
    extra: dict[str, object] = Field(default_factory=dict)

    content_type: ContentTypeEnum = "tool_request"

    @field_serializer("extra")
    @classmethod
    def serialize_extra(cls, v: dict[str, object]) -> dict[str, object]:
        return serialize_dict_with_bytes(v)

    @field_validator("extra", mode="before")
    @classmethod
    def validate_extra(cls, v: object) -> object:
        if isinstance(v, dict):
            return validate_dict_with_bytes(v)
        return v

    def __str__(self):
        args_str = self._arguments_str()
        func_call = f"{self.name}({args_str})"
        comment = f"# 🔧 tool request ({self.id})"
        return f"```python\n{comment}\n{func_call}\n```\n"

    def _arguments_str(self) -> str:
        if isinstance(self.arguments, dict):
            return ", ".join(
                f"{k}={self._format_arg(v)}" for k, v in self.arguments.items()
            )
        return str(self.arguments)

    @staticmethod
    def _format_arg(value: object) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    def _repr_html_(self) -> str:
        return str(self.tagify())

    def tagify(self) -> Tagified:
        "Returns an HTML string suitable for passing to htmltools/shiny's `Chat()` component."
        try:
            from htmltools import HTML, TagList, head_content, tags
        except ImportError:
            raise ImportError(
                ".tagify() is only intended to be called by htmltools/shiny, ",
                "but htmltools is not installed. ",
            )

        html = f"<p></p><span class='chatlas-tool-request'>🔧 Running tool: <code>{html_escape(self.name, attr=False)}</code></span>"

        return TagList(
            HTML(html),
            head_content(tags.style(TOOL_CSS)),
        ).tagify()


class ContentToolResult(Content):
    """
    The result of calling a tool/function

    A content type representing the result of a tool function call. When a model
    requests a tool function, [](`~chatlas.Chat`) will create, (optionally)
    echo, (optionally) yield, and store this content type in the chat history.

    A tool function may also construct an instance of this class and return it.
    This is useful for a tool that wishes to customize how the result is handled
    (e.g., the format of the value sent to the model).

    Parameters
    ----------
    value
        The return value of the tool/function.
    model_format
        The format used for sending the value to the model. The default,
        `"auto"`, first attempts to format the value as a JSON string. If that
        fails, it gets converted to a string via `str()`. To force
        `orjson.dumps()` or `str()`, set to `"json"` or `"str"`. Finally,
        `"as_is"` is useful for doing your own formatting and/or passing a
        non-string value (e.g., a list or dict) straight to the model.
        Non-string values are useful for tools that return images or other
        'known' non-text content types.
    error
        An exception that occurred while invoking the tool. If this is set, the
        error message sent to the model and the value is ignored.
    extra
       Additional data associated with the tool result that isn't sent to the
       model.
    request
        Not intended to be used directly. It will be set when the
        :class:`~chatlas.Chat` invokes the tool.

    Note
    ----
    When `model_format` is `"json"` (or `"auto"`), and the value has a
    `.to_json()`/`.to_dict()` method, those methods are called to obtain the
    JSON representation of the value. This is convenient for classes, like
    `pandas.DataFrame`, that have a `.to_json()` method, but don't necessarily
    dump to JSON directly. If this happens to not be the desired behavior, set
    `model_format="as_is"` return the desired value as-is.
    """

    # public
    value: Any
    model_format: Literal["auto", "json", "str", "as_is"] = "auto"
    error: Optional[Exception] = None
    extra: Any = None

    # "private"
    request: Optional[ContentToolRequest] = None
    content_type: ContentTypeEnum = "tool_result"

    @field_serializer("error")
    @classmethod
    def serialize_error(cls, v: Optional[Exception]) -> Optional[str]:
        """Serialize Exception to string for JSON compatibility."""
        if v is None:
            return None
        return str(v)

    @field_validator("error", mode="before")
    @classmethod
    def validate_error(cls, v: Any) -> Optional[Exception]:
        """Accept string or Exception for error field."""
        if v is None:
            return None
        if isinstance(v, Exception):
            return v
        if isinstance(v, str):
            return Exception(v)
        return Exception(str(v))

    @property
    def id(self):
        if not self.request:
            raise ValueError("id is only available after the tool has been called")
        return self.request.id

    @property
    def name(self):
        if not self.request:
            raise ValueError("name is only available after the tool has been called")
        return self.request.name

    @property
    def arguments(self):
        if not self.request:
            raise ValueError(
                "arguments is only available after the tool has been called"
            )
        return self.request.arguments

    def __str__(self):
        return self.to_display_markdown()

    def to_display_markdown(self, max_lines: Optional[int] = None) -> str:
        """
        Render as a fenced code block, optionally capping the value's height.

        Parameters
        ----------
        max_lines
            Truncate the value to this many lines, replacing the remainder with a
            count of what was dropped. `None` (the default) emits the full value.
        """
        prefix = "✅ tool result" if not self.error else "❌ tool error"
        comment = f"# {prefix} ({self.id})"
        value = self._get_display_value()
        if max_lines is not None:
            value = truncate_lines(value, max_lines)
        return f"""```python\n{comment}\n{value}\n```"""

    # Format the value for display purposes
    def _get_display_value(self):
        if self.error:
            return f"Tool call failed with error: '{self.error}'"

        val = self.value

        # If value is already a dict or list, format it directly
        if isinstance(val, (dict, list)):
            return pformat(val, indent=2, sort_dicts=False)

        # For string values, try to parse as JSON
        if isinstance(val, str):
            try:
                json_val = orjson.loads(val)
                return pformat(json_val, indent=2, sort_dicts=False)
            except orjson.JSONDecodeError:
                # Not valid JSON, return as string
                return val

        return str(val)

    def get_model_value(self) -> object:
        "Get the actual value sent to the model."

        if self.error:
            return f"Tool call failed with error: '{self.error}'"

        val, mode = (self.value, self.model_format)

        if isinstance(val, str):
            return val

        if mode == "auto":
            try:
                return self._to_json(val)
            except Exception:
                return str(val)
        elif mode == "json":
            return self._to_json(val)
        elif mode == "str":
            return str(val)
        elif mode == "as_is":
            return val
        else:
            raise ValueError(f"Unknown format mode: {mode}")

    @staticmethod
    def _to_json(value: Any) -> object:
        if hasattr(value, "to_pandas") and callable(value.to_pandas):
            # Many (most?) df libs (polars, pyarrow, ...) have a .to_pandas()
            # method, and pandas has a .to_json() method
            value = value.to_pandas()

        if hasattr(value, "to_json") and callable(value.to_json):
            # pandas defaults to "columns", which is not ideal for LLMs
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
            sig = inspect.signature(value.to_json)
            if "orient" in list(sig.parameters.keys()):
                return value.to_json(orient="records")
            else:
                return value.to_json()

        # Support for df libs (beyond those with a .to_pandas() method)
        if hasattr(value, "__narwhals_dataframe__"):
            try:
                import narwhals

                val = cast(narwhals.DataFrame, narwhals.from_native(value))
                return val.to_pandas().to_json(orient="records")
            except ImportError:
                warnings.warn(
                    f"Tool result object of type {type(value)} appears to be a "
                    "narwhals-compatible DataFrame. If you run into issues with "
                    "the LLM not understanding this value, try installing narwhals: "
                    "`pip install narwhals`.",
                    ImportWarning,
                    stacklevel=2,
                )

        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()

        return orjson.dumps(value).decode("utf-8")

    def _repr_html_(self):
        return self.to_html()

    def tagify(self) -> Tagified:
        "A method for rendering this object via htmltools/shiny."
        try:
            from htmltools import HTML, TagList, head_content, tags
        except ImportError:
            raise ImportError(
                ".tagify() is only intended to be called by htmltools/shiny, ",
                "but htmltools is not installed. ",
            )

        return TagList(
            HTML(self.to_html()),
            head_content(tags.style(TOOL_CSS)),
        ).tagify()

    def to_html(self) -> str:
        """
        Render as an HTML string.

        Shared by `.tagify()` (shinychat) and the notebook echo display, so the two
        can't drift. Requires `TOOL_CSS` to be present on the page. The result is
        collapsed; `TOOL_CSS` bounds its height once expanded.
        """

        # Helper function to format code blocks (optionally with labels for arguments).
        # Labels are argument names, which come from the model's tool call, so they
        # need escaping just like the values do.
        def pre_code(code: str, label: str | None = None) -> str:
            if label:
                lbl = f"<span class='input-parameter-label'>{html_escape(label, attr=False)}</span>"
            else:
                lbl = ""
            return f"<pre>{lbl}<code>{html_escape(code, attr=False)}</code></pre>"

        # Helper function to wrap content in a <details> block.
        def details_block(summary: str, content: str, open_: bool = True) -> str:
            open_attr = " open" if open_ else ""
            return (
                f"<details{open_attr}><summary>{summary}</summary>{content}</details>"
            )

        # First, format the input parameters.
        args = self.arguments or {}
        if isinstance(args, dict):
            args = "".join(pre_code(str(v), label=k) for k, v in args.items())
        else:
            args = pre_code(str(args))

        params = (
            f"<strong>Input parameters:</strong>{args}" if args else ""
        )
        result = f"<strong>Result:</strong>{pre_code(self._get_display_value())}"

        # Put both the result and parameters into a container
        result_div = f'<div class="chatlas-tool-result-content">{result}{params}</div>'

        # Header for the top-level result details block. The tool name is
        # model-controlled, so it gets escaped too.
        name = html_escape(self.name, attr=False)
        if not self.error:
            header = f"Result from tool call: <code>{name}</code>"
        else:
            header = f"❌ Failed to call tool <code>{name}</code>"

        res = details_block(header, result_div, open_=False)

        return f'<div class="chatlas-tool-result">{res}</div>'

    def _arguments_str(self) -> str:
        if isinstance(self.arguments, dict):
            return ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return str(self.arguments)


class ContentJson(Content):
    """
    JSON content

    This content type primarily exists to signal structured data extraction
    (i.e., data extracted via [](`~chatlas.Chat`)'s `.chat_structured()` method)

    Parameters
    ----------
    value
        The JSON data extracted
    """

    value: dict[str, Any]

    content_type: ContentTypeEnum = "json"

    def __str__(self):
        val = orjson.dumps(self.value, option=orjson.OPT_INDENT_2).decode("utf-8")
        return f"""```json\n{val}\n```"""


def markdown_code_span(label: str) -> str:
    label = label.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
    longest_run = 0
    current_run = 0
    for character in label:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * (longest_run + 1)
    return f"{fence}{label}{fence}"


class ContentPDF(Content):
    """
    PDF content

    This content type primarily exists to signal PDF data extraction
    (i.e., data extracted via [](`~chatlas.Chat`)'s `.chat_structured()` method)

    Parameters
    ----------
    data
        The PDF's bytes. Optional when `url` is set.
    filename
        The name of the PDF file
    url
        An optional URL where the PDF can be accessed.
    """

    data: Optional[bytes] = None
    filename: str
    url: Optional[str] = None

    content_type: ContentTypeEnum = "pdf"

    @model_validator(mode="after")
    def _check_data_or_url(self) -> "ContentPDF":
        if self.data is None and self.url is None:
            raise ValueError("ContentPDF requires either `data` or `url` to be set.")
        return self

    @field_serializer("data")
    @classmethod
    def serialize_data(cls, v: Optional[bytes]) -> Optional[str]:
        if v is None:
            return None
        return base64.b64encode(v).decode("ascii")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: Optional[bytes | str]) -> Optional[bytes]:
        if isinstance(v, str):
            return base64.b64decode(v, validate=True)
        return v

    def __str__(self):
        detail = format_bytes(len(self.data)) if self.data is not None else self.url
        return markdown_code_span(f"[PDF {self.filename} · {detail}]")


class ContentDocument(Content):
    """
    Generic document content (plain text, Markdown, CSV, code, and -- on
    providers that support it -- docx/xlsx).

    This is the type returned by [](`~chatlas.content_document_file`). Unlike
    [](`~chatlas.ContentPDF`), documents carry a real `mime_type` since they
    span many formats; PDFs should always go through
    [](`~chatlas.content_pdf_file`)/[](`~chatlas.content_pdf_url`) instead,
    which unlock PDF-specific handling (page-image understanding on
    Anthropic, and URL passthrough).

    Parameters
    ----------
    data
        The document's bytes. Optional when `url` is set.
    filename
        The name of the document file.
    mime_type
        The document's MIME type (e.g. `"text/plain"`, `"text/csv"`,
        `"application/vnd.openxmlformats-officedocument.wordprocessingml.document"`).
        Not every provider accepts every MIME type -- providers that can't
        accept a given type raise a clear error.
    url
        An optional URL where the document can be accessed.
    """

    data: Optional[bytes] = None
    filename: str
    mime_type: str
    url: Optional[str] = None

    content_type: ContentTypeEnum = "document"

    @model_validator(mode="after")
    def _check_data_or_url(self) -> "ContentDocument":
        if self.data is None and self.url is None:
            raise ValueError(
                "ContentDocument requires either `data` or `url` to be set."
            )
        if self.mime_type == "application/pdf":
            raise ValueError(
                "ContentDocument doesn't support PDF files. Use "
                "content_pdf_file() or content_pdf_url() instead, which "
                "unlock PDF-specific handling (page-image understanding on "
                "Anthropic, and URL passthrough)."
            )
        return self

    @field_serializer("data")
    @classmethod
    def serialize_data(cls, v: Optional[bytes]) -> Optional[str]:
        if v is None:
            return None
        return base64.b64encode(v).decode("ascii")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: Optional[bytes | str]) -> Optional[bytes]:
        if isinstance(v, str):
            return base64.b64decode(v, validate=True)
        return v

    def __str__(self):
        return markdown_code_span(f"[document {self.filename} · {self.mime_type}]")


class ContentUploaded(Content):
    """
    A reference to a file already uploaded to a provider.

    Returned by `chat.files.upload(...)` and usable directly in `.chat()` so the
    file bytes are not re-sent each turn. Can also be constructed directly to
    reference a file uploaded out-of-band (e.g. a Google Vertex `gs://` URI).

    Parameters
    ----------
    id
        The provider's file identifier (OpenAI/Anthropic `file_id`, or a
        Google/Vertex URI such as `https://.../files/abc` or `gs://bucket/obj`).
    mime_type
        The file's MIME type. Determines image-vs-document serialization and is
        required by Google's file references.
    provider
        The provider the file was uploaded to (`"openai"`, `"anthropic"`, or
        `"google"`). Used to detect cross-provider misuse.
    extra
        A plain dict of provider-native metadata (filename, size, ...) when
        available. Unlike `FileMetadata.extra`, this can't hold the provider's
        own file object, since this type is serialized as part of a chat's turns.
    """

    id: str
    mime_type: str
    provider: str
    extra: dict[str, Any] = Field(default_factory=dict)

    content_type: ContentTypeEnum = "uploaded"

    def __str__(self):
        return markdown_code_span(f"[uploaded {self.id} · {self.mime_type}]")


class ContentThinking(Content):
    """
    Thinking/reasoning content

    Captures the model's internal reasoning process.

    Parameters
    ----------
    thinking
        The thinking/reasoning text from the model.
    extra
        Additional metadata associated with the thinking content (e.g.,
        encrypted content, status information).
    """

    thinking: str
    extra: Optional[dict[str, Any]] = None

    content_type: ContentTypeEnum = "thinking"

    @field_serializer("extra")
    @classmethod
    def serialize_extra(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if v is None:
            return None
        return serialize_dict_with_bytes(v)

    @field_validator("extra", mode="before")
    @classmethod
    def validate_extra(cls, v: object) -> object:
        if isinstance(v, dict):
            return validate_dict_with_bytes(v)
        return v

    def __add__(self, other: object) -> "ContentThinking":
        if not isinstance(other, ContentThinking):
            return NotImplemented  # type: ignore[return-value]
        return ContentThinking.model_construct(
            thinking=self.thinking + other.thinking,
            extra=other.extra if other.extra is not None else self.extra,
        )

    def __str__(self):
        return f"<thinking>\n{self.thinking}\n</thinking>\n"

    def _repr_html_(self):
        return str(self.tagify())

    def tagify(self) -> Tagified:
        try:
            from htmltools import HTML
        except ImportError:
            raise ImportError(
                ".tagify() is only intended to be called by htmltools/shiny, ",
                "but htmltools is not installed. ",
            )

        html = f"<details><summary>Thinking</summary>{self.thinking}</details>"

        return HTML(html)


class ContentThinkingDelta(Content):
    """
    A streaming fragment of thinking/reasoning content.

    Emitted during streaming to represent a chunk of the model's thinking.
    The ``phase`` attribute communicates block boundaries to downstream consumers.

    Parameters
    ----------
    thinking
        The thinking/reasoning text fragment.
    phase
        The phase of the thinking delta: ``"start"``, ``"body"``, or ``"end"``.
    """

    thinking: str
    phase: Literal["start", "body", "end"] = "body"

    content_type: ContentTypeEnum = "thinking_delta"

    def __add__(self, other: object) -> "ContentThinkingDelta":
        if not isinstance(other, ContentThinkingDelta):
            return NotImplemented  # type: ignore[return-value]
        return ContentThinkingDelta(
            thinking=self.thinking + other.thinking,
            phase=self.phase,
        )

    def __str__(self):
        return self.thinking


class ContentToolRequestSearch(Content):
    """
    A web search request from the model.

    This content type represents the model's request to search the web.
    It's automatically generated when a built-in web search tool is used.

    Parameters
    ----------
    query
        The search query.
    extra
        The raw provider-specific response data.
    """

    query: str
    extra: Optional[dict[str, Any]] = None

    content_type: ContentTypeEnum = "web_search_request"

    def __str__(self):
        return f"**web search request**: {self.query!r}"


class ContentToolResponseSearch(Content):
    """
    Web search results from the model.

    This content type represents the results of a web search.
    It's automatically generated when a built-in web search tool returns results.

    Parameters
    ----------
    sources
        The pages surfaced by the search.
    extra
        The raw provider-specific response data.
    """

    sources: list[WebSource]
    extra: Optional[dict[str, Any]] = None

    content_type: ContentTypeEnum = "web_search_results"

    def __str__(self):
        lines = "\n".join(f"* {s}" for s in self.sources)
        return f"**web search results**:\n{lines}"


class ContentToolRequestFetch(Content):
    """
    A web fetch request from the model.

    This content type represents the model's request to fetch a URL.
    It's automatically generated when a built-in web fetch tool is used.

    Parameters
    ----------
    url
        The URL to fetch.
    extra
        The raw provider-specific response data.
    """

    url: str
    extra: Optional[dict[str, Any]] = None

    content_type: ContentTypeEnum = "web_fetch_request"

    def __str__(self):
        return f"**web fetch request**: {self.url}"


class ContentToolResponseFetch(Content):
    """
    Web fetch results from the model.

    This content type represents the results of fetching a URL.
    It's automatically generated when a built-in web fetch tool returns results.

    Parameters
    ----------
    url
        The URL that was fetched.
    status
        A normalized, cross-provider outcome: ``"success"`` if content was
        retrieved, ``"error"`` if it was not, or ``None`` when the provider
        doesn't report an outcome. Providers expose finer-grained, non-aligned
        reasons (e.g. Anthropic's ``url_not_allowed``, Google's ``PAYWALL``);
        those are not normalized here but remain available in ``extra``.
    extra
        The raw provider-specific response data.
    """

    url: str
    status: Optional[Literal["success", "error"]] = None
    extra: Optional[dict[str, Any]] = None

    content_type: ContentTypeEnum = "web_fetch_results"

    def __str__(self):
        return f"**web fetch result**: {self.url}"


class ContentCitation(Content):
    """
    A source that grounds part of the assistant's answer.

    ``grounded_span`` is the span of the assistant's answer this citation
    grounds — the words a footnote marker attaches to. It is answer-side (from
    the reply). ``cited_quote`` is the source-side evidence quote, when the
    provider supplies one (e.g. Anthropic web search). ``source`` identifies the
    evidence; it is ``None`` when the citation grounds answer text with no
    resolvable source.
    """

    source: SerializeAsAny[Optional[Source]] = None
    grounded_span: Optional[str] = None
    cited_quote: Optional[str] = None
    extra: Optional[dict[str, Any]] = None
    content_type: ContentTypeEnum = "citation"

    @field_validator("source", mode="before")
    @classmethod
    def _rebuild_source(cls, v: Any) -> Any:
        return create_source(v) if isinstance(v, dict) else v

    # The label is bold rather than bracketed on purpose: `[citation]: <url>` is a
    # CommonMark link reference definition, so both the console and notebook
    # renderers would consume the line and display nothing.
    def __str__(self) -> str:
        label = (
            str(self.source) if self.source is not None else (self.grounded_span or "")
        )
        return f"**citation**: {label}"


ContentUnion = Union[
    ContentText,
    ContentImageRemote,
    ContentImageInline,
    ContentToolRequest,
    ContentToolResult,
    ContentJson,
    ContentPDF,
    ContentDocument,
    ContentUploaded,
    ContentThinking,
    ContentToolRequestSearch,
    ContentToolResponseSearch,
    ContentToolRequestFetch,
    ContentToolResponseFetch,
    ContentCitation,
]


ProviderAnnotation = Union[
    ContentCitation,
    ContentToolRequestSearch,
    ContentToolResponseSearch,
    ContentToolRequestFetch,
    ContentToolResponseFetch,
]
"""
Content a provider reports about its own server-side work, rather than content
the user authored. When produced by a provider, this content may carry a raw provider payload in `extra`.

Two consequences follow, and every provider has to handle both:

1. On the way out, these are the extra content types worth surfacing during
   streaming (beyond text and thinking).
2. On the way back in, only the provider that produced one can resend it, since
   `extra` is in that provider's own shape. Providers replay their own and drop
   the rest, so turns stay portable across providers.
"""

PROVIDER_ANNOTATION_TYPES = (
    ContentCitation,
    ContentToolRequestSearch,
    ContentToolResponseSearch,
    ContentToolRequestFetch,
    ContentToolResponseFetch,
)
"""Runtime `isinstance` form of `ProviderAnnotation` (keep the two in sync)."""


BYTES_SENTINEL = "__base64_bytes__"


def serialize_dict_with_bytes(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, bytes):
            result[key] = {BYTES_SENTINEL: base64.b64encode(value).decode("ascii")}
        else:
            result[key] = value
    return result


def validate_dict_with_bytes(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in d.items():
        if (
            isinstance(value, dict)
            and set(value.keys()) == {BYTES_SENTINEL}
            and isinstance(value[BYTES_SENTINEL], str)
        ):
            result[key] = base64.b64decode(value[BYTES_SENTINEL], validate=True)
        else:
            result[key] = value
    return result


def create_source(data: dict[str, Any]) -> Source:
    """Rebuild the concrete `Source` subclass from a serialized dict."""
    t = data.get("type")
    if t == "web":
        return WebSource.model_validate(data)
    raise ValueError(f"Unknown source type: {t}")


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
    elif ct == "document":
        return ContentDocument.model_validate(data)
    elif ct == "uploaded":
        return ContentUploaded.model_validate(data)
    elif ct == "thinking":
        return ContentThinking.model_validate(data)
    elif ct == "web_search_request":
        return ContentToolRequestSearch.model_validate(data)
    elif ct == "web_search_results":
        return ContentToolResponseSearch.model_validate(data)
    elif ct == "web_fetch_request":
        return ContentToolRequestFetch.model_validate(data)
    elif ct == "web_fetch_results":
        return ContentToolResponseFetch.model_validate(data)
    elif ct == "citation":
        return ContentCitation.model_validate(data)
    else:
        raise ValueError(f"Unknown content type: {ct}")


TOOL_CSS = """
/* Get dot to appear inline, even when in a paragraph following the request */
.chatlas-tool-request + p:has(.markdown-stream-dot) {
  display: inline;
}

/* Hide request when anything other than a dot follows it */
.chatlas-tool-request:not(:has(+ p .markdown-stream-dot)) {
  display: none;
}

.chatlas-tool-request, .chatlas-tool-result {
  font-weight: 300;
  font-size: 0.9rem;
}

.chatlas-tool-result {
  display: inline-block;
  width: 100%;
  margin-bottom: 1rem;
}

.chatlas-tool-result summary {
  list-style: none;
  cursor: pointer;
}

.chatlas-tool-result summary::after {
  content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-caret-right-fill' viewBox='0 0 16 16'%3E%3Cpath d='m12.14 8.753-5.482 4.796c-.646.566-1.658.106-1.658-.753V3.204a1 1 0 0 1 1.659-.753l5.48 4.796a1 1 0 0 1 0 1.506z'/%3E%3C/svg%3E");
  font-size: 1.15rem;
  margin-left: 0.25rem;
  vertical-align: middle;
}

.chatlas-tool-result details[open] summary::after {
  content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-caret-down-fill' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14 2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
}

.chatlas-tool-result-content {
  position: relative;
  border: 1px solid var(--bs-border-color, #0066cc);
  width: 100%;
  padding: 1rem;
  border-radius: var(--bs-border-radius, 0.2rem);
  margin-top: 1rem;
  margin-bottom: 1rem;
}

.chatlas-tool-result-content pre, .chatlas-tool-result-content code {
  background-color: var(--bs-body-bg, white) !important;
}

/* Bound a large result so it costs a fixed amount of vertical space.
   Consumers override the height by setting the custom property on an ancestor
   (chatlas' notebook display does this from set_echo_options()). */
.chatlas-tool-result-content pre {
  max-height: var(--chatlas-tool-result-max-height, 400px);
  overflow-y: auto;
}

.chatlas-tool-result-content .input-parameter-label {
  position: absolute;
  top: 0;
  width: 100%;
  text-align: center;
  font-weight: 300;
  font-size: 0.8rem;
  color: var(--bs-gray-600);
  background-color: var(--bs-body-bg);
  padding: 0.5rem;
  font-family: var(--bs-font-monospace, monospace);
}

pre:has(> .input-parameter-label) {
  padding-top: 1.5rem;
}

shiny-markdown-stream p:first-of-type:empty {
  display: none;
}
"""

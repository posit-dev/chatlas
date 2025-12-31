from __future__ import annotations

import inspect
import warnings
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Optional,
    cast,
    get_args,
    get_origin,
)

import openai
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from . import _utils
from ._content import (
    ContentImageInline,
    ContentPDF,
    ContentToolResult,
    ToolAnnotations,
)

__all__ = (
    "Tool",
    "ToolBuiltIn",
    "ToolRejectError",
)

if TYPE_CHECKING:
    from mcp import ClientSession as MCPClientSession
    from mcp import Tool as MCPTool
    from openai.types.chat import ChatCompletionToolParam


class Tool:
    """
    Define a tool

    Define a Python function for use by a chatbot. The function will always be
    invoked in the current Python process.

    Parameters
    ----------
    func
        The function to be invoked when the tool is called.
    name
        The name of the tool.
    description
        A description of what the tool does.
    parameters
        A dictionary describing the input parameters and their types.
    annotations
        Additional properties that describe the tool and its behavior.
    """

    func: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def __init__(
        self,
        *,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        name: str,
        description: str,
        parameters: dict[str, Any],
        annotations: "Optional[ToolAnnotations]" = None,
    ):
        self.name = name
        self.func = func
        self.annotations = annotations
        self._is_async = _utils.is_async_callable(func)
        self.schema: "ChatCompletionToolParam" = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    @classmethod
    def from_func(
        cls: type["Tool"],
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        name: Optional[str] = None,
        model: Optional[type[BaseModel]] = None,
        annotations: "Optional[ToolAnnotations]" = None,
    ) -> "Tool":
        """
        Create a Tool from a Python function

        Parameters
        ----------
        func
            The function to wrap as a tool.
        name
            The name of the tool. If not provided, the name will be inferred from the
            function's name.
        model
            A Pydantic model that describes the input parameters for the function.
            If not provided, the model will be inferred from the function's type hints.
            The primary reason why you might want to provide a model in
            Note that the name and docstring of the model takes precedence over the
            name and docstring of the function.
        annotations
            Additional properties that describe the tool and its behavior.

        Returns
        -------
        Tool
            A new Tool instance wrapping the provided function.

        Raises
        ------
        ValueError
            If there is a mismatch between model fields and function parameters.
        """

        if model is None:
            model = func_to_basemodel(func)

        _validate_model_vs_function(model, func)

        params = basemodel_to_param_schema(model)

        return cls(
            func=func,
            name=name or model.__name__ or func.__name__,
            description=model.__doc__ or func.__doc__ or "",
            parameters=params,
            annotations=annotations,
        )

    @classmethod
    def from_mcp(
        cls: type["Tool"],
        session: "MCPClientSession",
        mcp_tool: "MCPTool",
    ) -> "Tool":
        """
        Create a Tool from an MCP tool

        Parameters
        ----------
        session
            The MCP client session to use for calling the tool.
        mcp_tool
            The MCP tool to wrap.

        Returns
        -------
        Tool
            A new Tool instance wrapping the MCP tool.
        """

        async def _call(**args: Any) -> AsyncGenerator[ContentToolResult, None]:
            result = await session.call_tool(mcp_tool.name, args)

            # Raise an error if the tool call resulted in an error. It doesn't seem to be
            # very well defined how to get at the error message, but it appears that it gets
            # stored in the `text` attribute of the content. Also, empirically, the error
            # message seems to include `Error executing tool {tool_name}: ...`, so
            if result.isError:
                err_msg = getattr(
                    result.content[0],
                    "text",
                    f"Error executing tool {mcp_tool.name}.",
                )
                raise RuntimeError(err_msg)

            for content in result.content:
                if content.type == "text":
                    yield ContentToolResult(value=content.text)
                elif content.type == "image":
                    if content.mimeType not in (
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                    ):
                        raise ValueError(
                            f"Unsupported image MIME type: {content.mimeType}"
                        )

                    img = ContentImageInline(
                        data=content.data,
                        image_content_type=content.mimeType,
                    )
                    yield ContentToolResult(value=img)
                elif content.type == "resource":
                    from mcp.types import TextResourceContents

                    resource = content.resource
                    if isinstance(resource, TextResourceContents):
                        blob = resource.text.encode("utf-8")
                    else:
                        blob = resource.blob.encode("utf-8")

                    mime_type = content.resource.mimeType
                    if mime_type != "application/pdf":
                        raise ValueError(f"Unsupported resource MIME type: {mime_type}")

                    pdf = ContentPDF(data=blob, filename=f"{mcp_tool.name}-result.pdf")
                    yield ContentToolResult(value=pdf)
                else:
                    raise RuntimeError(f"Unexpected content type: {content.type}")

        params = mcp_tool_input_schema_to_param_schema(mcp_tool.inputSchema)

        # Convert MCP ToolAnnotations to our TypedDict format
        annotations = None
        if mcp_tool.annotations:
            annotations = cast(ToolAnnotations, mcp_tool.annotations.model_dump())

        return cls(
            func=_utils.wrap_async(_call),
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=params,
            annotations=annotations,
        )


class ToolBuiltIn:
    """
    Define a built-in provider-specific tool

    This class represents tools that are built into specific providers (like image
    generation). Unlike regular Tool objects, ToolBuiltIn instances pass raw
    provider-specific JSON directly through to the API.

    Parameters
    ----------
    name
        The name of the tool.
    definition
        The raw provider-specific tool definition as a dictionary.
    """

    def __init__(self, *, name: str, definition: dict[str, Any]):
        self.name = name
        self.definition = definition


class ToolRejectError(Exception):
    """
    Error to represent a tool call being rejected.

    This error is meant to be raised when an end user has chosen to deny a tool
    call. It can be raised in a tool function or in a `.on_tool_request()`
    callback registered via a :class:`~chatlas.Chat`. When used in the callback,
    the tool call is rejected before the tool function is invoked.

    Parameters
    ----------
    reason
        A string describing the reason for rejecting the tool call. This will be
        included in the error message passed to the LLM. In addition to the
        reason, the error message will also include "Tool call rejected." to
        indicate that the tool call was not processed.

    Raises
    -------
    ToolRejectError
        An error with a message informing the LLM that the tool call was
        rejected (and the reason why).

    Examples
    --------
    >>> import os
    >>> import chatlas as ctl
    >>>
    >>> chat = ctl.ChatOpenAI()
    >>>
    >>> def list_files():
    ...     "List files in the user's current directory"
    ...     while True:
    ...         allow = input(
    ...             "Would you like to allow access to your current directory? (yes/no): "
    ...         )
    ...         if allow.lower() == "yes":
    ...             return os.listdir(".")
    ...         elif allow.lower() == "no":
    ...             raise ctl.ToolRejectError(
    ...                 "The user has chosen to disallow the tool call."
    ...             )
    ...         else:
    ...             print("Please answer with 'yes' or 'no'.")
    >>>
    >>> chat.register_tool(list_files)
    >>> chat.chat("What files are available in my current directory?")
    """

    def __init__(self, reason: str = "The user has chosen to disallow the tool call."):
        message = f"Tool call rejected. {reason}"
        super().__init__(message)
        self.message = message


def func_to_schema(
    func: Callable[..., Any] | Callable[..., Awaitable[Any]],
    model: Optional[type[BaseModel]] = None,
) -> "ChatCompletionToolParam":
    if model is None:
        model = func_to_basemodel(func)

    # Throw if there is a mismatch between the model and the function parameters
    params = inspect.signature(func).parameters
    fields = model.model_fields
    diff = set(params) ^ set(fields)
    if diff:
        raise ValueError(
            f"`model` fields must match tool function parameters exactly. "
            f"Fields found in one but not the other: {diff}"
        )

    params = basemodel_to_param_schema(model)

    return {
        "type": "function",
        "function": {
            "name": model.__name__ or func.__name__,
            "description": model.__doc__ or func.__doc__ or "",
            "parameters": params,
        },
    }


def func_to_basemodel(func: Callable) -> type[BaseModel]:
    params = inspect.signature(func).parameters
    fields = {}

    for name, param in params.items():
        annotation = param.annotation
        annotated_field: Optional[FieldInfo] = None

        if annotation == inspect.Parameter.empty:
            warnings.warn(
                f"Parameter `{name}` of function `{name}` has no type hint. "
                "Using `Any` as a fallback."
            )
            annotation = Any
        # Check if annotation is Annotated[...] and extract Field metadata
        elif get_origin(annotation) is Annotated:
            args = get_args(annotation)
            # First arg is the actual type, rest are metadata
            annotation = args[0]
            for metadata in args[1:]:
                if isinstance(metadata, FieldInfo):
                    annotated_field = metadata
                    break

        # create_model() will error if the field name starts with `_` (since Pydantic
        # uses this to indicate private fields). We can work around this by using an alias.
        alias = None
        if name.startswith("_"):
            field_name, alias = (name.lstrip("_"), name)
        else:
            field_name, alias = (name, None)

        # Create the pydantic Field from a "normal" parameter
        if param.default != inspect.Parameter.empty:
            field = Field(default=param.default, alias=alias)
        else:
            field = Field(alias=alias)

        # If we have an Annotated FieldInfo, merge it with alias/default overrides
        if annotated_field is not None:
            field = FieldInfo.merge_field_infos(annotated_field, field)

        # Add the field to our fields dict
        fields[field_name] = (annotation, field)

    return create_model(func.__name__, **fields)


def basemodel_to_param_schema(model: type[BaseModel]) -> dict[str, object]:
    # Lean on openai's ability to translate BaseModel.model_json_schema()
    # to a valid tool schema (this wouldn't be impossible to do ourselves,
    # but it's fair amount of logic to substitute `$refs`, etc.)
    tool = openai.pydantic_function_tool(model)

    fn = tool["function"]
    if "parameters" not in fn:
        raise ValueError("Expected `parameters` in function definition.")

    params = rm_param_titles(fn["parameters"])

    return params


def _validate_model_vs_function(model: type[BaseModel], func: Callable) -> None:
    """Validate that model fields match function parameters."""

    sig_params = inspect.signature(func).parameters
    fields = model.model_fields

    param_names: set[str] = set()

    for field_name, field_info in fields.items():
        param_name = field_info.alias if field_info.alias else field_name

        if param_name not in sig_params:
            raise ValueError(
                f"`model` field `{field_name}` (param name `{param_name}`) "
                f"has no corresponding function parameter."
            )

        param_names.add(param_name)
        param = sig_params[param_name]
        func_has_default = param.default != inspect.Parameter.empty

        if func_has_default and param.default != field_info.default:
            model_default = (
                "no default"
                if field_info.default is PydanticUndefined
                else repr(field_info.default)
            )
            raise ValueError(
                f"Function parameter `{param_name}` has default `{param.default!r}`, "
                f"but model field `{field_name}` has {model_default}. "
                f"These must match in order to create a Tool."
            )

    # Check for function params without corresponding model fields
    extra_params = set(sig_params) - param_names
    if extra_params:
        raise ValueError(
            f"Function parameters {extra_params} have no corresponding model fields."
        )


def mcp_tool_input_schema_to_param_schema(
    input_schema: dict[str, Any],
) -> dict[str, object]:
    params = sanitize_schema(input_schema)

    if "additionalProperties" not in params:
        params["additionalProperties"] = False

    return params


def sanitize_schema(
    params: dict[str, object],
) -> dict[str, object]:
    """
    Sanitize JSON Schema for provider compatibility.

    - `title`: Pydantic includes titles at model/field level, but they're not needed
    - `format`: JSON Schema format hints (e.g., "uri", "date-time") that some
      providers like OpenAI reject
    - `required`: OpenAI requires all properties to be in required array, with
      optional params using anyOf with null type
    """
    if "title" in params:
        del params["title"]

    if "format" in params:
        del params["format"]

    if "properties" in params and isinstance(params["properties"], dict):
        required_keys = set(params.get("required", []))

        for key, prop in params["properties"].items():
            if isinstance(prop, dict):
                sanitize_schema(prop)

                # For optional properties (not in required), make them nullable
                # OpenAI requires all props in required array, using anyOf with
                # null type to indicate optionality
                if key not in required_keys:
                    prop_type = prop.get("type")
                    if prop_type and prop_type != "null":
                        # Convert to anyOf with null
                        prop["anyOf"] = [{"type": prop_type}, {"type": "null"}]
                        del prop["type"]
                    elif "anyOf" not in prop and "oneOf" not in prop:
                        # No type specified, just allow null
                        prop["anyOf"] = [prop.copy(), {"type": "null"}]

        # OpenAI requires all properties to be in required array
        params["required"] = list(params["properties"].keys())

    return params


# Keep for backwards compatibility
rm_param_titles = sanitize_schema

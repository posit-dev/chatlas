from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Optional

import openai
from pydantic import BaseModel, Field, create_model

from . import _utils
from ._content import (
    ContentToolResult,
    ContentToolResultImage,
    ContentToolResultResource,
)

__all__ = (
    "Tool",
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
    """

    func: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def __init__(
        self,
        *,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        self.name = name
        self.func = func
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
        model: Optional[type[BaseModel]] = None,
    ) -> "Tool":
        """
        Create a Tool from a Python function

        Parameters
        ----------
        func
            The function to wrap as a tool.
        model
            A Pydantic model that describes the input parameters for the function.
            If not provided, the model will be inferred from the function's type hints.
            The primary reason why you might want to provide a model in
            Note that the name and docstring of the model takes precedence over the
            name and docstring of the function.

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

        return cls(
            func=func,
            name=model.__name__ or func.__name__,
            description=model.__doc__ or func.__doc__ or "",
            parameters=params,
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

                    yield ContentToolResultImage(
                        value=content.data,
                        mime_type=content.mimeType,
                    )
                elif content.type == "resource":
                    from mcp.types import TextResourceContents

                    resource = content.resource
                    if isinstance(resource, TextResourceContents):
                        blob = resource.text.encode("utf-8")
                    else:
                        blob = resource.blob.encode("utf-8")

                    yield ContentToolResultResource(
                        value=blob, mime_type=content.resource.mimeType
                    )
                else:
                    raise RuntimeError(f"Unexpected content type: {content.type}")

        params = mcp_tool_input_schema_to_param_schema(mcp_tool.inputSchema)

        return cls(
            func=_utils.wrap_async(_call),
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=params,
        )


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

        if annotation == inspect.Parameter.empty:
            warnings.warn(
                f"Parameter `{name}` of function `{name}` has no type hint. "
                "Using `Any` as a fallback."
            )
            annotation = Any

        if param.default != inspect.Parameter.empty:
            field = Field(default=param.default)
        else:
            field = Field()

        # Add the field to our fields dict
        fields[name] = (annotation, field)

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


def mcp_tool_input_schema_to_param_schema(
    input_schema: dict[str, Any],
) -> dict[str, object]:
    params = rm_param_titles(input_schema)

    if "additionalProperties" not in params:
        params["additionalProperties"] = False

    return params


def rm_param_titles(
    params: dict[str, object],
) -> dict[str, object]:
    # For some reason, pydantic wants to include a title at the model and field
    # level. I don't think we actually need or want this.
    if "title" in params:
        del params["title"]

    if "properties" in params and isinstance(params["properties"], dict):
        for prop in params["properties"].values():
            if "title" in prop:
                del prop["title"]

    return params

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import openai
from pydantic import BaseModel, Field, create_model

from . import _utils

__all__ = (
    "Tool",
    "ToolRejectError",
)

if TYPE_CHECKING:
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
    model
        A Pydantic model that describes the input parameters for the function.
        If not provided, the model will be inferred from the function's type hints.
        The primary reason why you might want to provide a model in
        Note that the name and docstring of the model takes precedence over the
        name and docstring of the function.
    """

    func: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def __init__(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        model: Optional[type[BaseModel]] = None,
    ):
        self.func = func
        self._is_async = _utils.is_async_callable(func)
        self.schema = func_to_schema(func, model)
        self.name = self.schema["function"]["name"]


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

    params = fn["parameters"]

    # For some reason, openai (or pydantic?) wants to include a title
    # at the model and field level. I don't think we actually need or want this.
    if "title" in params:
        del params["title"]

    if "properties" in params and isinstance(params["properties"], dict):
        for prop in params["properties"].values():
            if "title" in prop:
                del prop["title"]

    return params

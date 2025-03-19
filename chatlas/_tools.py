from __future__ import annotations

import inspect
import json
import warnings
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional, Protocol

from pydantic import BaseModel, Field, create_model

from . import _utils

__all__ = (
    "Tool",
    "ToolResult",
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

    from ._content import ContentToolRequest


class Stringable(Protocol):
    def __str__(self) -> str: ...


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
        on_request: Optional[Callable[[ContentToolRequest], Stringable]] = None,
    ):
        self.func = func
        self._is_async = _utils.is_async_callable(func)
        self.schema = func_to_schema(func, model)
        self.name = self.schema["function"]["name"]
        self.on_request = on_request


class ToolResult:
    """
    A result from a tool invocation

    Return an instance of this class from a tool function in order to:

    1. Yield content for the user (i.e., the downstream consumer of a `.stream()` or `.chat()`)
       to display.
    2. Control how the tool result gets formatted for the model (i.e., the assistant).

    Parameters
    ----------
    assistant
        The tool result to send to the llm (i.e., assistant). If the result is
        not a string, `format_as` determines how to the value is formatted
        before sending it to the model.
    user
        A value to yield to the user (i.e., the consumer of a `.stream()`) when
        the tool is called. If `None`, no value is yielded. This is primarily
        useful for producing custom UI in the response output to indicate to the
        user that a tool call has completed (for example, return shiny UI here
        when `.stream()`-ing inside a shiny app).
    format_as
        How to format the `assistant` value for the model. The default,
        `"auto"`, first attempts to format the value as a JSON string. If that
        fails, it gets converted to a string via `str()`. To force
        `json.dumps()` or `str()`, set to `"json"` or `"str"`. Finally,
        `"as_is"` is useful for doing your own formatting and/or passing a
        non-string value (e.g., a list or dict) straight to the model.
        Non-string values are useful for tools that return images or other
        'known' non-text content types.
    """

    def __init__(
        self,
        assistant: Stringable,
        *,
        user: Optional[Stringable] = None,
        format_as: Literal["auto", "json", "str", "as_is"] = "auto",
    ):
        # TODO: if called when an active user session, perhaps we could
        # provide a smart default here
        self.user = user
        self.assistant = self._format_value(assistant, format_as)
        # TODO: we could consider adding an "emit value" -- that is, the thing to
        # display when `echo="all"` is used. I imagine that might be useful for
        # advanced users, but let's not worry about it until someone asks for it.
        # self.emit = emit

    def _format_value(self, value: Stringable, mode: str) -> Stringable:
        if isinstance(value, str):
            return value

        if mode == "auto":
            try:
                return json.dumps(value)
            except Exception:
                return str(value)
        elif mode == "json":
            return json.dumps(value)
        elif mode == "str":
            return str(value)
        elif mode == "as_is":
            return value
        else:
            raise ValueError(f"Unknown format mode: {mode}")


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
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The openai package is required for this functionality. "
            "Please install it with `pip install openai`."
        )

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

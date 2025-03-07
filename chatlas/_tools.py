from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol

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

    Return this value from a tool if you want to separate what gets sent
    to the model vs what value gets yielded to the user.

    Parameters
    ----------
    assistant
        The value to sent to the model. Must be stringify-able (i.e. have a `__str__` method).
    output
        A value to yield when a tool result occurs. If `None`, no value is yielded.
        This is primarily useful for allowing a tool result to create custom UI
        (for example, return shiny UI here when `.stream()`-ing in  a shiny app).
    """

    def __init__(
        self,
        assistant: Stringable,
        output: Optional[Stringable] = None,
    ):
        self.assistant = assistant
        self.output = output
        # TODO: we could consider adding an "emit value" -- that is, the thing to
        # display when `echo="all"` is used. I imagine that might be useful for
        # advanced users, but let's not worry about it until someone asks for it.
        # self.emit = emit


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

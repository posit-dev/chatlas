from __future__ import annotations

import inspect
import warnings
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal

from . import _utils
from ._typing_extensions import Required, TypedDict

__all__ = ("Tool",)


class Tool:
    """
    Define a tool

    Define a Python function for use by a chatbot. The function will always be
    invoked in the current Python process.

    Examples
    --------

    ```python
    from chatlas import Tool


    def add(a: int, b: int) -> int:
        return a + b


    chat = ChatOpenAI()

    # It's recommended to provide the tool description (and parameter descriptions)
    # via a docstring, but you can also provide them directly via Tool():
    chat.register_tool(
        Tool(
            add,
            description="Add two numbers.",
            parameter_descriptions={
                "a": "The first number.",
                "b": "The second number.",
            },
        ),
    )
    ```

    Parameters
    ----------
    func
        The function to define as a tool.
    name
        The name of the tool.
    description
        The description of the tool.
    parameter_descriptions
        Descriptions for the parameters of the function.
    """

    func: Callable[..., Any] | Callable[..., Awaitable[Any]]

    def __init__(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_descriptions: Optional[dict[str, str]] = None,
    ):
        self.func = func
        self._is_async = _utils.is_async_callable(func)
        self.schema = func_to_schema(
            func,
            name=name,
            description=description,
            parameter_descriptions=parameter_descriptions,
        )
        self.name = self.schema["function"]["name"]


class ToolSchemaProperty(TypedDict, total=False):
    type: Required[str]
    description: Required[str]


class ToolSchemaParams(TypedDict):
    type: Literal["object"]
    properties: dict[str, ToolSchemaProperty]
    required: list[str]


class ToolSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: ToolSchemaParams


class ToolSchema(TypedDict):
    type: Literal["function"]
    function: ToolSchemaFunction


def func_to_schema(
    func: Callable[..., Any] | Callable[..., Awaitable[Any]],
    name: str | None = None,
    description: str | None = None,
    parameter_descriptions: dict[str, str] | None = None,
) -> ToolSchema:
    name = name or func.__name__
    description = description or func.__doc__ or ""
    model = func_to_basemodel(func, name)
    params = basemodel_to_param_schema(
        model,
        name=name,
        description=description,
        parameter_descriptions=parameter_descriptions,
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


def func_to_basemodel(func: Callable, name: str) -> type[BaseModel]:
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

    return create_model(name, **fields)


def basemodel_to_param_schema(
    model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
    parameter_descriptions: dict[str, str] | None = None,
) -> ToolSchemaParams:
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
    tool = openai.pydantic_function_tool(
        model,
        name=name,
        description=description,
    )

    # Translate openai's tool schema format to our own
    fn = tool["function"]
    params: dict[str, Any] = {}
    if "parameters" in fn:
        params = fn["parameters"]

    properties: dict[str, ToolSchemaProperty] = {}
    if "properties" in params:
        properties = params["properties"]

    for k, v in properties.items():
        # Pydantic likes to include "title" in its schema, which we I don't think we
        # need (and Google will complain about)
        if "title" in v:
            del v["title"]
        # If description is falsy, provide a fallback from parameter_descriptions
        if not v.get("description", ""):
            if parameter_descriptions and k in parameter_descriptions:
                v["description"] = parameter_descriptions[k]

    required: list[str] = []
    if "required" in params:
        required = params["required"]

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }

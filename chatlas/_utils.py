import functools
import inspect
from types import NoneType
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    ParamSpec,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from typing_extensions import Literal, Required, TypedDict

__all__ = (
    "ToolFunction",
    "ToolSchema",
    "ToolSchemaFunction",
    "func_to_schema",
)

ToolFunctionSync = Callable[..., Any]
ToolFunctionAsync = Callable[..., Awaitable[Any]]
ToolFunction = Union[ToolFunctionSync, ToolFunctionAsync]


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
    func: ToolFunction,
    name: str | None = None,
    description: str | None = None,
    parameter_descriptions: dict[str, str] | None = None,
) -> ToolSchema:
    signature = inspect.signature(func)
    required: list[str] = []

    for nm, param in signature.parameters.items():
        if param.default is param.empty and param.kind not in [
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]:
            required.append(nm)

    annotations = get_type_hints(func, include_extras=True)

    param_desc = parameter_descriptions or {}

    params: ToolSchemaParams = {
        "type": "object",
        "properties": {
            k: type_to_json_schema(v, param_desc.get(k, None))
            for k, v in annotations.items()
            if k != "return"
        },
        "required": required,
    }

    desc = description or func.__doc__

    res: ToolSchema = {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": desc or "",
            "parameters": params,
        },
    }

    return res


def type_to_json_schema(
    t: type,
    desc: str | None = None,
) -> ToolSchemaProperty:
    origin = get_origin(t)
    args = get_args(t)
    if origin is Annotated:
        assert len(args) == 2
        assert desc is None or desc == ""
        assert isinstance(args[1], str)
        return type_to_json_schema(args[0], args[1])

    if origin is list:
        assert len(args) == 1
        return type_dict("array", desc, items=type_to_json_schema(args[0]))

    if origin is dict:
        assert len(args) == 2
        assert args[0] is str
        return type_dict(
            "object", desc, additionalProperties=type_to_json_schema(args[1])
        )

    if is_typeddict(t):
        annotations = get_type_hints(t, include_extras=True)
        return type_dict(
            "object",
            desc,
            properties={k: type_to_json_schema(v) for k, v in annotations.items()},
        )

    if t is dict:
        return type_dict("object", desc)
    if t is list:
        return type_dict("array", desc)
    if t is str:
        return type_dict("string", desc)
    if t is int:
        return type_dict("integer", desc)
    if t is float:
        return type_dict("number", desc)
    if t is bool:
        return type_dict("boolean", desc)
    if t is NoneType:
        return type_dict("null", desc)
    raise ValueError(f"Unsupported type: {t}")


def type_dict(
    type_: str,
    description: str | None,
    **kwargs: Any,
) -> ToolSchemaProperty:
    res: ToolSchemaProperty = {
        "type": type_,
        "description": description or "",
        **kwargs,  # type: ignore
    }
    return res


# Copied from shiny/_utils.py

R = TypeVar("R")  # Return type
P = ParamSpec("P")


def wrap_async(
    fn: Callable[P, R] | Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    Given a synchronous function that returns R, return an async function that wraps the
    original function. If the input function is already async, then return it unchanged.
    """

    if is_async_callable(fn):
        return fn

    fn = cast(Callable[P, R], fn)

    @functools.wraps(fn)
    async def fn_async(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)

    return fn_async


def is_async_callable(
    obj: Callable[P, R] | Callable[P, Awaitable[R]],
) -> TypeGuard[Callable[P, Awaitable[R]]]:
    """
    Determine if an object is an async function.

    This is a more general version of `inspect.iscoroutinefunction()`, which only works
    on functions. This function works on any object that has a `__call__` method, such
    as a class instance.

    Returns
    -------
    :
        Returns True if `obj` is an `async def` function, or if it's an object with a
        `__call__` method which is an `async def` function.
    """
    if inspect.iscoroutinefunction(obj):
        return True
    if hasattr(obj, "__call__"):  # noqa: B004
        if inspect.iscoroutinefunction(obj.__call__):  # type: ignore
            return True

    return False

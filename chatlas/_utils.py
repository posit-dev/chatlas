from __future__ import annotations

import asyncio
import functools
import inspect
import os
import re
from typing import Any, AsyncIterable, Awaitable, Callable, Iterable, TypeVar, cast

import httpx

from ._typing_extensions import ParamSpec, TypedDict, TypeGuard


class AnyTypeDict(TypedDict, total=False):
    pass


HTTPClientKwargs = TypeVar("HTTPClientKwargs", bound=AnyTypeDict)


def split_http_client_kwargs(
    kwargs: HTTPClientKwargs,
) -> tuple[HTTPClientKwargs, HTTPClientKwargs]:
    """Split kwargs, removing incompatible http_client for sync/async."""
    sync_kwargs = kwargs.copy()
    async_kwargs = kwargs.copy()

    http_client = kwargs.get("http_client")
    if isinstance(http_client, httpx.AsyncClient):
        sync_kwargs.pop("http_client")
    elif isinstance(http_client, httpx.Client):
        async_kwargs.pop("http_client")

    return sync_kwargs, async_kwargs


# --------------------------------------------------------------------
# wrap_async() and is_async_callable() was copied from shiny/_utils.py
# --------------------------------------------------------------------

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


def wrap_async_iterable(x: Iterable[Any] | AsyncIterable[Any]) -> AsyncIterable[Any]:
    """
    Given any iterable, return an async iterable. The async iterable will yield the
    values of the original iterable, but will also yield control to the event loop
    after each value. This is useful when you want to interleave processing with other
    tasks, or when you want to simulate an async iterable from a regular iterable.
    """

    if isinstance(x, AsyncIterable):
        return x

    if not isinstance(x, Iterable):
        raise TypeError("wrap_async_iterable requires an Iterable object.")

    return MakeIterableAsync(x)


class MakeIterableAsync:
    def __init__(self, iterable: Iterable[Any]):
        self.iterable = iterable

    def __aiter__(self):
        self.iterator = iter(self.iterable)
        return self

    async def __anext__(self):
        try:
            value = next(self.iterator)
            await asyncio.sleep(0)  # Yield control to the event loop
            return value
        except StopIteration:
            raise StopAsyncIteration


T = TypeVar("T")


def drop_none(x: dict[str, T | None]) -> dict[str, T]:
    return {k: v for k, v in x.items() if v is not None}


# https://docs.pytest.org/en/latest/example/simple.html#pytest-current-test-environment-variable
def is_testing():
    return os.environ.get("PYTEST_CURRENT_TEST", None) is not None


class MISSING_TYPE:
    """
    A singleton representing a missing value.
    """

    pass


MISSING = MISSING_TYPE()


# --------------------------------------------------------------------
# html_escape was copied from htmltools/_utils.py
# --------------------------------------------------------------------


HTML_ESCAPE_TABLE = {
    "&": "&amp;",
    ">": "&gt;",
    "<": "&lt;",
}

HTML_ATTRS_ESCAPE_TABLE = {
    **HTML_ESCAPE_TABLE,
    '"': "&quot;",
    "'": "&apos;",
    "\r": "&#13;",
    "\n": "&#10;",
}


def html_escape(text: str, attr: bool = True) -> str:
    table = HTML_ATTRS_ESCAPE_TABLE if attr else HTML_ESCAPE_TABLE
    if not re.search("|".join(table), text):
        return text
    for key, value in table.items():
        text = text.replace(key, value)
    return text

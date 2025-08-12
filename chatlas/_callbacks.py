from collections import OrderedDict
from typing import Any, Callable

from ._utils import is_async_callable


class CallbackManager:
    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[..., Any]] = OrderedDict()
        self._id: int = 1

    def add(self, callback: Callable[..., Any]) -> Callable[[], None]:
        callback_id = self._next_id()
        self._callbacks[callback_id] = callback

        def _rm_callback() -> None:
            self._callbacks.pop(callback_id, None)

        return _rm_callback

    def invoke(self, *args: Any, **kwargs: Any) -> None:
        if not self._callbacks:
            return

        # Invoke in reverse insertion order
        for callback_id in reversed(list(self._callbacks.keys())):
            callback = self._callbacks[callback_id]
            if is_async_callable(callback):
                raise RuntimeError(
                    "Can't use async callbacks with `.chat()`/`.stream()`."
                    "Async callbacks can only be used with `.chat_async()`/`.stream_async()`."
                )
            callback(*args, **kwargs)

    async def invoke_async(self, *args: Any, **kwargs: Any) -> None:
        if not self._callbacks:
            return

        # Invoke in reverse insertion order
        for callback_id in reversed(list(self._callbacks.keys())):
            callback = self._callbacks[callback_id]
            if is_async_callable(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)

    def count(self) -> int:
        return len(self._callbacks)

    def get_callbacks(self) -> list[Callable[..., Any]]:
        return list(self._callbacks.values())

    def _next_id(self) -> str:
        current_id = self._id
        self._id += 1
        return str(current_id)

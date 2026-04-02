from __future__ import annotations


class StreamController:
    """
    Cooperative cancellation handle for streaming responses.

    Create a controller and pass it to
    :meth:`~chatlas.Chat.stream` or :meth:`~chatlas.Chat.stream_async`
    via the ``controller`` argument, then call :meth:`cancel` from
    anywhere (e.g., a Shiny observer) to stop the stream after the
    current chunk.

    The same controller can be reused across multiple streams. Call
    :meth:`reset` to clear the cancelled state before starting a new
    stream.

    Examples
    --------
    ```python
    from chatlas import ChatOpenAI, StreamController

    chat = ChatOpenAI()
    ctrl = StreamController()

    i = 0
    for chunk in chat.stream("Write a story", controller=ctrl):
        i += 1
        print(chunk, end="")
        if i > 10:
            ctrl.cancel()

    # Partial response is preserved in history
    print(chat.get_turns())
    ```
    """

    def __init__(self):
        self._cancelled: bool = False
        self._reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel the stream. The reason is stored on the partial turn."""
        # Set reason before cancelled so that readers who see cancelled=True
        # will always find a reason. This is safe under CPython's GIL; on
        # free-threaded builds the worst case is a benign extra iteration
        # before the streaming loop notices the cancellation.
        self._reason = reason
        self._cancelled = True

    def reset(self) -> None:
        """Clear the cancelled state and reason."""
        self._cancelled = False
        self._reason = None

    def _ensure_ready(self) -> None:
        """Auto-reset if already cancelled (prevents stale controller bugs)."""
        if self._cancelled:
            import warnings

            warnings.warn(
                "StreamController was already cancelled — resetting automatically. "
                "Call controller.reset() explicitly to avoid this warning.",
                stacklevel=3,
            )
            self.reset()

    @property
    def cancelled(self) -> bool:
        """Whether the controller has been cancelled."""
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """The cancellation reason, or None if not cancelled."""
        return self._reason

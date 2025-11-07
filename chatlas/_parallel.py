"""Parallel chat execution for processing multiple prompts concurrently."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from pydantic import BaseModel

from ._chat import Chat
from ._content import ContentToolRequest, ContentToolResult, ToolInfo
from ._progress import ProgressTracker
from ._turn import user_turn

if TYPE_CHECKING:
    from ._batch_job import ContentT

__all__ = (
    "parallel_chat",
    "parallel_chat_text",
    "parallel_chat_structured",
)

ChatT = TypeVar("ChatT", bound=Chat)
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

@dataclass
class StructuredChatResult(Generic[BaseModelT, ChatT]):
    """Holds the result of a structured parallel chat request."""

    data: BaseModelT
    """The extracted structured data."""

    chat: ChatT
    """The Chat object."""


@overload
async def parallel_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["stop"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[ChatT]: ...


@overload
async def parallel_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["continue"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[ChatT | Exception]: ...


@overload
async def parallel_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[ChatT | Exception | None]: ...


async def parallel_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return", "continue", "stop"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[ChatT | Exception | None]:
    """
    Submit multiple chat prompts in parallel.

    If you have multiple prompts, you can submit them in parallel. This is
    typically considerably faster than submitting them in sequence, especially
    with providers like OpenAI and Google.

    If using [](`~chatlas.ChatOpenAI`) or [](`~chatlas.ChatAnthropic`) and if
    you're willing to wait longer, you might want to use
    [](`~chatlas.batch_chat()`) instead, as it comes with a 50% discount in
    return for taking up to 24 hours.

    Parameters
    ----------
    chat
        A base chat object.
    prompts
        A list of prompts. Each prompt can be a string or a list of
        string/Content objects.
    max_active
        The maximum number of simultaneous requests to send. For Anthropic,
        note that the number of active connections is limited primarily by
        the output tokens per minute limit (OTPM) which is estimated from
        the `max_tokens` parameter (defaults to 4096). If your usage tier
        limits you to 16,000 OTPM, you should either set `max_active = 4`
        (16,000 / 4096) or reduce `max_tokens` via `set_model_params()`.
    rpm
        Maximum number of requests per minute. Default is 500.
    on_error
        What to do when a request fails. One of:
          * `"return"` (the default): stop processing new requests,
             wait for in-flight requests to finish, then return.
          * `"continue"`: keep going, performing every request.
          * `"stop"`: stop processing and throw an error.
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list with one element for each prompt. Each element is either a Chat
    object (if successful), None (if the request wasn't submitted), or an
    error object (if it failed).

    Examples
    --------
    Basic usage with multiple prompts:

    ```python
    import asyncio
    import chatlas as ctl

    chat = ctl.ChatOpenAI()
    countries = ["Canada", "New Zealand", "Jamaica", "United States"]
    prompts = [f"What's the capital of {country}?" for country in countries]

    # NOTE: if running from a script, you'd need to wrap this in an async function
    # and call asyncio.run(main())
    chats = await ctl.parallel_chat(chat, prompts)
    ```

    Using with interpolation:

    ```python
    import chatlas as ctl

    chat = ctl.ChatOpenAI()
    template = "What's the capital of {{ country }}?"

    countries = ["Canada", "New Zealand", "Jamaica"]
    prompts = [ctl.interpolate(template, variables={"country": c}) for c in countries]

    chats = await ctl.parallel_chat(chat, prompts, max_active=5)
    ```

    See Also
    --------
    * :func:`~chatlas.parallel_chat_text` : Get just the text responses
    * :func:`~chatlas.parallel_chat_structured` : Extract structured data
    * :func:`~chatlas.batch_chat` : Batch API for discounted processing
    """
    return await _parallel_chat_impl(
        chat,
        prompts,
        max_active=max_active,
        rpm=rpm,
        on_error=on_error,
        kwargs=kwargs,
    )


@overload
async def parallel_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["stop"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[str]: ...


@overload
async def parallel_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["continue"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[str | Exception]: ...


@overload
async def parallel_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[str | Exception | None]: ...


async def parallel_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return", "continue", "stop"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[str | Exception | None]:
    """
    Submit multiple chat prompts in parallel and return text responses.

    This is a convenience function that wraps [](`~chatlas.parallel_chat()`) and
    extracts just the text content from each response.

    Parameters
    ----------
    chat
        A base chat object.
    prompts
        A list of prompts. Each prompt can be a string or a list of
        string/Content objects.
    max_active
        The maximum number of simultaneous requests to send.
    rpm
        Maximum number of requests per minute. Default is 500.
    on_error
        What to do when a request fails. One of:
          * `"return"` (the default): stop processing new requests,
             wait for in-flight requests to finish, then return.
          * `"continue"`: keep going, performing every request.
          * `"stop"`: stop processing and throw an error.
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list with one element for each prompt. Each element is either a string (if
    successful), None (if the request wasn't submitted), or an error object (if
    it failed).

    Examples
    --------
    ```python
    import chatlas as ctl

    chat = ctl.ChatOpenAI()

    countries = ["Canada", "New Zealand", "Jamaica", "United States"]
    prompts = [f"What's the capital of {country}?" for country in countries]

    # NOTE: if running from a script, you'd need to wrap this in an async function
    # and call asyncio.run(main())
    responses = await ctl.parallel_chat_text(chat, prompts)
    for country, response in zip(countries, responses):
        print(f"{country}: {response}")
    ```

    See Also
    --------
    * :func:`~chatlas.parallel_chat` : Get full Chat objects
    * :func:`~chatlas.parallel_chat_structured` : Extract structured data
    """
    chats = await parallel_chat(
        chat,
        prompts,
        max_active=max_active,
        rpm=rpm,
        on_error=on_error,
        kwargs=kwargs,
    )
    res: list[str | Exception | None] = []
    for x in chats:
        if not isinstance(x, Chat):
            res.append(x)
        else:
            last_turn = x.get_last_turn(role="assistant")
            assert last_turn is not None
            res.append(last_turn.text)
    return res


@overload
async def parallel_chat_structured(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    data_model: type[BaseModelT],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["stop"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[StructuredChatResult[BaseModelT, ChatT]]: ...


@overload
async def parallel_chat_structured(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    data_model: type[BaseModelT],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["continue"],
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[StructuredChatResult[BaseModelT, ChatT] | Exception]: ...


@overload
async def parallel_chat_structured(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    data_model: type[BaseModelT],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[StructuredChatResult[BaseModelT, ChatT] | Exception | None]: ...


async def parallel_chat_structured(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    data_model: type[BaseModelT],
    *,
    max_active: int = 10,
    rpm: int = 500,
    on_error: Literal["return", "continue", "stop"] = "return",
    kwargs: Optional[dict[str, Any]] = None,
) -> Sequence[StructuredChatResult[BaseModelT, ChatT] | Exception | None]:
    """
    Submit multiple chat prompts in parallel and extract structured data.

    This function processes multiple prompts concurrently and extracts
    structured data from each response according to the specified Pydantic model
    type.

    Parameters
    ----------
    chat
        A base chat object.
    prompts
        A list of prompts. Each prompt can be a string or a list of
        string/Content objects.
    data_model
        A Pydantic model class defining the structure to extract.
    max_active
        The maximum number of simultaneous requests to send.
    rpm
        Maximum number of requests per minute. Default is 500.
    on_error
        What to do when a request fails. One of:
          * `"return"` (the default): stop processing new requests,
             wait for in-flight requests to finish, then return.
          * `"continue"`: keep going, performing every request.
          * `"stop"`: stop processing and throw an error.
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list with one element for each prompt. Each element is either a
    `~chatlas.types.StructuredChatResult` (if successful), `None` (if the
    request wasn't submitted), or an error object (if it failed). Note that the
    `StructuredChatResult` contains both the extracted data (for convenience)
    and the full Chat object (for completeness).

    Examples
    --------
    Extract structured data from multiple prompts:

    ```python
    import chatlas as ctl
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int


    chat = ctl.ChatOpenAI()

    prompts = [
        "I go by Alex. 42 years on this planet and counting.",
        "Pleased to meet you! I'm Jamal, age 27.",
        "They call me Li Wei. Nineteen years young.",
        "Fatima here. Just celebrated my 35th birthday last week.",
    ]

    # NOTE: if running from a script, you'd need to wrap this in an async
    # function and call asyncio.run(main())
    people = await ctl.parallel_chat_structured(chat, prompts, Person)
    for person in people:
        print(f"{person.data.name} is {person.data.age} years old")
    ```

    See Also
    --------
    * :func:`~chatlas.parallel_chat` : Get full Chat objects
    * :func:`~chatlas.parallel_chat_text` : Get just the text responses
    * :func:`~chatlas.Chat.structured_data` : Extract data from a single chat
    """
    if not prompts:
        return []

    chats = await _parallel_chat_impl(
        chat,
        prompts,
        data_model=data_model,
        max_active=max_active,
        rpm=rpm,
        on_error=on_error,
        kwargs=kwargs,
    )

    results: list[StructuredChatResult[BaseModelT, ChatT] | Exception | None] = []
    for x in chats:
        if not isinstance(x, Chat):
            results.append(x)
        else:
            turn = x.get_last_turn(role="assistant")
            assert turn is not None
            dat = Chat._extract_turn_json(turn)
            d = data_model.model_validate(dat)
            results.append(StructuredChatResult(data=d, chat=x))

    return results


async def _parallel_chat_impl(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int,
    rpm: int,
    data_model: type[BaseModel] | None = None,
    on_error: Literal["return", "continue", "stop"] = "return",
    kwargs: dict[str, Any] | None = None,
) -> list[ChatT | Exception | None]:
    """
    Internal implementation of parallel chat execution with tool support.

    This function handles the multi-phase execution:
    1. Submit all prompts in parallel
    2. Process tools *sequentially* in submission order
    3. Submit tool results in parallel
    4. Repeat until all conversations are complete
    """
    if not prompts:
        return []

    rate_limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(max_active)

    # Copy the chat and empty its turns (so we can create a minimal fork for
    # each prompt)
    turns = chat.get_turns()
    chat_orig = copy.deepcopy(chat)
    chat_orig.set_turns([])

    # Initialize conversation states
    conversations: list[ConversationState] = []
    for i in range(len(prompts)):
        chat_i = copy.deepcopy(chat_orig)
        chat_i.set_turns(turns)
        conversations.append(ConversationState(chat=chat_i, index=i))

    # Helper to submit prompts in a rate-limited manner
    async def _submit_prompt(
        conv: ConversationState,
        prompt: ContentT | list[ContentT],
        error_controller: ErrorController,
        on_complete: Callable[[], None] = lambda: None,
    ):
        async with semaphore:
            await rate_limiter.acquire()

            # Check if we should skip due to earlier error
            if error_controller.should_stop_new_requests:
                on_complete()
                return

            # Mark that we're attempting to submit this conversation
            conv.was_submitted = True

            if not isinstance(prompt, list):
                prompt = [prompt]

            user_prompt = user_turn(*prompt)

            try:
                response = conv.chat._submit_turns_async(
                    user_prompt,
                    data_model=data_model,
                    echo="none",
                    stream=False,
                    kwargs=kwargs,
                )
                async for _ in response:
                    pass

            except Exception as e:
                conv.error = e
                error_controller.record_error(e)
            finally:
                on_complete()

            # Only check for tool requests if no error occurred
            if conv.error is None:
                last_turn = conv.chat.get_last_turn(role="assistant")
                assert last_turn is not None
                tool_requests = [
                    c for c in last_turn.contents if isinstance(c, ContentToolRequest)
                ]
                if tool_requests:
                    conv.pending_tool_requests = tool_requests
            conv.pending_tool_results = None

    # === PHASE 1: Submit initial prompts in parallel ===
    error_controller = ErrorController(on_error_mode=on_error)

    with ProgressTracker(
        f"Submitting {len(prompts)} prompts",
        total=len(prompts),
    ) as progress:
        tasks = [
            _submit_prompt(conv, prompt, error_controller, on_complete=progress.advance)
            for conv, prompt in zip(conversations, prompts)
        ]
        await asyncio.gather(*tasks)

    # For "stop" mode, raise immediately
    if on_error == "stop" and error_controller.first_error:
        raise error_controller.first_error

    # Helper to execute tools and attach results to the conversation
    async def _invoke_tools(
        conv: ConversationState,
        on_complete: Callable[[], None],
    ):
        requests = conv.pending_tool_requests
        assert requests is not None

        results: list[ContentToolResult] = []
        for x in requests:
            tool = conv.chat._tools.get(x.name)
            if tool is not None:
                x.tool = ToolInfo.from_tool(tool)

            tool_results = conv.chat._invoke_tool_async(x)
            async for res in tool_results:
                results.append(res)

        if results:
            conv.pending_tool_results = results
        conv.pending_tool_requests = None
        on_complete()

    # === PHASE 2+: Process tools and submit results until all conversations complete ===
    round_num = 1

    while True:
        # Errored conversations won't have pending_tool_requests set, so no need
        # to filter by error state here
        conversations_requesting_tools = [
            c for c in conversations if c.pending_tool_requests
        ]

        if len(conversations_requesting_tools) == 0:
            break

        # Process tool calls sequentially (in submission order)
        with ProgressTracker(
            f"Processing tools (round {round_num})",
            total=len(conversations_requesting_tools),
        ) as progress:
            for conv in conversations_requesting_tools:
                await _invoke_tools(conv, on_complete=progress.advance)

        # Submit pending tool results
        conversations_to_submit = [c for c in conversations if c.pending_tool_results]

        if len(conversations_to_submit) == 0:
            break

        with ProgressTracker(
            f"Submitting tool results (round {round_num})",
            total=len(conversations_to_submit),
        ) as progress:
            tasks = []
            for conv in conversations_to_submit:
                results = cast("list[ContentT]", conv.pending_tool_results)
                tasks.append(
                    _submit_prompt(
                        conv, results, error_controller, on_complete=progress.advance
                    )
                )
            await asyncio.gather(*tasks)

        # Check for "stop" mode after each round
        if on_error == "stop" and error_controller.first_error:
            raise error_controller.first_error

        round_num += 1

    # === PHASE 3: Return completed chats ===
    # For "stop" mode, raise if any error occurred
    if on_error == "stop" and error_controller.first_error:
        raise error_controller.first_error

    res: list[ChatT | Exception | None] = []
    for conv in conversations:
        if conv.error:
            res.append(conv.error)
        elif not conv.was_submitted:
            # Conversation was never submitted due to error handling
            res.append(None)
        else:
            res.append(conv.chat)

    return res


@dataclass
class ErrorController:
    """Tracks error state and controls early termination."""

    on_error_mode: Literal["return", "continue", "stop"]
    """The error handling mode."""

    first_error: Exception | None = None
    """The first error that occurred."""

    should_stop_new_requests: bool = False
    """Whether to stop processing new requests."""

    def record_error(self, error: Exception) -> None:
        """Record an error and determine if we should stop."""
        if self.first_error is None:
            self.first_error = error
            if self.on_error_mode in ("return", "stop"):
                self.should_stop_new_requests = True


@dataclass
class ConversationState(Generic[ChatT]):
    """Track the state of a single conversation in parallel_chat."""

    chat: ChatT
    """The chat object with accumulated conversation turns."""

    index: int
    """Position in the original prompts list."""

    pending_tool_requests: list[ContentToolRequest] | None = None
    """Tool requests that need to be processed."""

    pending_tool_results: list[ContentToolResult] | None = None
    """Tool results that need to be submitted back to the LLM."""

    error: Exception | None = None
    """If an error occurred during processing."""

    was_submitted: bool = False
    """Whether this conversation's prompt was ever submitted."""


class RateLimiter:
    """Simple rate limiter for controlling requests per minute."""

    def __init__(self, rpm: int = 500):
        """
        Initialize rate limiter.

        Parameters
        ----------
        rpm
            Maximum requests per minute
        """
        self.rpm = rpm
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_request_time = 0.0

    async def acquire(self) -> None:
        """Wait until it's safe to make another request."""
        if self.min_interval <= 0:
            return

        time_since_last = time.time() - self.last_request_time

        if self.min_interval > time_since_last:
            await asyncio.sleep(self.min_interval - time_since_last)

        self.last_request_time = time.time()

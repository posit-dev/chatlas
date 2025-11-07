"""Parallel chat execution for processing multiple prompts concurrently."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

from pydantic import BaseModel

from ._chat import Chat
from ._content import ContentToolRequest, ContentToolResult, ToolInfo
from ._progress import ProgressTracker
from ._turn import Turn, user_turn

if TYPE_CHECKING:
    from ._batch_job import ContentT

__all__ = (
    "parallel_chat",
    "parallel_chat_text",
    "parallel_chat_structured",
)

ChatT = TypeVar("ChatT", bound=Chat)
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


async def parallel_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    kwargs: Optional[dict[str, Any]] = None,
) -> list[ChatT]:
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
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list of Chat objects, one for each prompt.

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
    chats = await _parallel_chat_impl(
        chat, prompts, max_active=max_active, rpm=rpm, kwargs=kwargs
    )
    return cast(list[ChatT], chats)


async def parallel_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    kwargs: Optional[dict[str, Any]] = None,
) -> list[str]:
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
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list of text responses, one for each prompt.

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
        chat, prompts, max_active=max_active, rpm=rpm, kwargs=kwargs
    )
    texts: list[str] = []
    for x in chats:
        last_turn = x.get_last_turn()
        assert last_turn is not None
        texts.append(last_turn.text)
    return texts


async def parallel_chat_structured(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    data_model: type[BaseModelT],
    *,
    max_active: int = 10,
    rpm: int = 500,
    kwargs: Optional[dict[str, Any]] = None,
) -> list[BaseModelT]:
    """
    Submit multiple chat prompts in parallel and extract structured data.

    This function processes multiple prompts concurrently and extracts
    structured data from each response according to the specified Pydantic
    model type.

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
    kwargs
        Additional keyword arguments to pass to the chat method.

    Returns
    -------
    A list of structured data objects, one for each prompt, with each
    object being an instance of the specified Pydantic model.

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

    # NOTE: if running from a script, you'd need to wrap this in an async function
    # and call asyncio.run(main())
    people = await ctl.parallel_chat_structured(chat, prompts, Person)
    for person in people:
        print(f"{person.name} is {person.age} years old")
    ```

    See Also
    --------
    * :func:`~chatlas.parallel_chat` : Get full Chat objects
    * :func:`~chatlas.parallel_chat_text` : Get just the text responses
    * :func:`~chatlas.Chat.structured_data` : Extract data from a single chat
    """
    if not prompts:
        return []

    # Use the shared implementation with data_model parameter
    chats = await _parallel_chat_impl(
        chat,
        prompts,
        max_active=max_active,
        rpm=rpm,
        kwargs=kwargs,
        data_model=data_model,
    )

    # Extract structured data from each completed chat
    results: list[BaseModelT] = []
    for chat_obj in chats:
        last_turn = chat_obj.get_last_turn()
        assert last_turn is not None
        dat = Chat._extract_turn_json(last_turn)
        results.append(data_model.model_validate(dat))

    return results


async def _parallel_chat_impl(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    *,
    max_active: int = 10,
    rpm: int = 500,
    kwargs: Optional[dict[str, Any]] = None,
    data_model: type[BaseModel] | None = None,
) -> list[Chat]:
    """
    Internal implementation of parallel chat execution with tool support.

    This function handles the multi-phase execution:
    1. Submit all prompts in parallel
    2. Process tools sequentially in submission order
    3. Submit tool results in parallel
    4. Repeat until all conversations are complete

    Parameters
    ----------
    chat
        Base chat object to use as template
    prompts
        List of prompts to process
    max_active
        Maximum concurrent API calls
    rpm
        Requests per minute limit
    kwargs
        Additional arguments for chat submission
    data_model
        Optional Pydantic model for structured data extraction

    Returns
    -------
    List of completed Chat objects
    """
    if not prompts:
        return []

    rate_limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(max_active)

    # Make a "global" copy of the chat with no turns to use as a template
    # (to avoid a deep copy of turns for each prompt)
    turns = chat.get_turns()
    chat_global = copy.deepcopy(chat)
    chat_global.set_turns([])

    # Initialize conversation states
    conversations = [
        ConversationState(
            index=i,
            chat=copy.deepcopy(chat_global),
            pending_tool_results=None,
            is_complete=False,
            error=None,
        )
        for i in range(len(prompts))
    ]

    # Restore initial turns for each conversation
    for conv in conversations:
        conv.chat.set_turns(turns)

    # === PHASE 1: Submit initial prompts in parallel ===
    async def _submit_prompt(
        conv: ConversationState, prompt: ContentT | list[ContentT]
    ):
        """Submit a user prompt to the LLM."""
        async with semaphore:
            await rate_limiter.acquire()

            try:
                if not isinstance(prompt, list):
                    prompt = [prompt]

                user_prompt = user_turn(*prompt)

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

    with ProgressTracker(
        f"Submitting {len(prompts)} prompts",
        total=len(prompts),
    ) as progress:
        tasks = [
            _submit_prompt(conv, prompt) for conv, prompt in zip(conversations, prompts)
        ]
        await asyncio.gather(*tasks)
        progress.advance(len(prompts))

    # === PHASE 2+: Process tools and submit results until all conversations complete ===
    round_num = 1

    while True:
        # Check which conversations need tool processing
        conversations_needing_tools = [
            c for c in conversations if not c.is_complete and not c.error
        ]

        if not conversations_needing_tools:
            break  # All done!

        # Process tool calls sequentially (in submission order)
        with ProgressTracker(
            f"Processing tools (round {round_num})",
            total=len(conversations_needing_tools),
        ) as progress:
            for conv in conversations_needing_tools:
                last_turn = conv.chat.get_last_turn(role="assistant")
                if last_turn is None:
                    conv.is_complete = True
                    progress.advance()
                    continue

                # Extract and execute tool calls
                all_results: list[ContentToolResult] = []
                for content in last_turn.contents:
                    if isinstance(content, ContentToolRequest):
                        tool = chat_global._tools.get(content.name)
                        if tool is not None:
                            content.tool = ToolInfo.from_tool(tool)

                        try:
                            results = conv.chat._invoke_tool_async(content)
                            async for res in results:
                                all_results.append(res)
                        except Exception as e:
                            conv.error = e
                            break

                # If we got tool results, prepare to submit them
                if all_results and not conv.error:
                    conv.pending_tool_results = Turn(role="user", contents=all_results)
                else:
                    conv.is_complete = True  # No more tools needed

                progress.advance()

        # Submit all pending tool results in parallel
        conversations_to_submit = [
            c for c in conversations if c.pending_tool_results and not c.error
        ]

        if not conversations_to_submit:
            break  # No more tool results to submit

        async def _submit_tool_results(conv: ConversationState):
            """Submit tool results back to the LLM."""
            async with semaphore:
                await rate_limiter.acquire()

                try:
                    # pending_tool_results should never be None here (filtered above)
                    if conv.pending_tool_results is None:
                        return

                    response = conv.chat._submit_turns_async(
                        conv.pending_tool_results,
                        data_model=data_model,
                        echo="none",
                        stream=False,
                        kwargs=kwargs,
                    )
                    async for _ in response:
                        pass

                    conv.pending_tool_results = None

                except Exception as e:
                    conv.error = e

        with ProgressTracker(
            f"Submitting tool results (round {round_num})",
            total=len(conversations_to_submit),
        ) as progress:
            tasks = [_submit_tool_results(conv) for conv in conversations_to_submit]
            await asyncio.gather(*tasks)
            progress.advance(len(conversations_to_submit))

        round_num += 1

    # === PHASE 3: Return completed chats ===
    # Optionally: log or handle errors
    for conv in conversations:
        if conv.error:
            # Could log warning, or raise exception, or return None
            # For now, just include the chat as-is (with partial results)
            pass

    return [conv.chat for conv in conversations]


@dataclass
class ConversationState:
    """Track the state of a single conversation in parallel_chat."""

    index: int
    """Position in the original prompts list."""

    chat: Chat
    """The chat object with accumulated conversation turns."""

    pending_tool_results: Turn | None = None
    """Tool results that need to be submitted back to the LLM."""

    is_complete: bool = False
    """Whether this conversation is done (no more tools to execute)."""

    error: Exception | None = None
    """If an error occurred during processing."""


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

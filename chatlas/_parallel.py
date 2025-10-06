"""Parallel chat execution for processing multiple prompts concurrently."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ._chat import Chat

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
    from chatlas import ChatOpenAI, parallel_chat

    chat = ChatOpenAI()

    countries = ["Canada", "New Zealand", "Jamaica", "United States"]
    prompts = [f"What's the capital of {country}?" for country in countries]

    chats = parallel_chat(chat, prompts)
    for c in chats:
        print(c.last_turn().text)
    ```

    Using with interpolation:

    ```python
    from chatlas import ChatOpenAI, parallel_chat, interpolate

    chat = ChatOpenAI()
    template = "What's the capital of {{ country }}?"

    countries = ["Canada", "New Zealand", "Jamaica"]
    prompts = [interpolate(template, variables={"country": c}) for c in countries]

    chats = parallel_chat(chat, prompts, max_active=5)
    ```

    See Also
    --------
    * :func:`~chatlas.parallel_chat_text` : Get just the text responses
    * :func:`~chatlas.parallel_chat_structured` : Extract structured data
    * :func:`~chatlas.batch_chat` : Batch API for discounted processing
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(
            f"Processing {len(prompts)} prompts...",
            total=len(prompts),
        )

        async def _do_chat(prompt: ContentT | list[ContentT]):
            async with semaphore:
                await rate_limiter.acquire()

                chat_local = copy.deepcopy(chat_global)
                chat_local.set_turns(turns)

                if not isinstance(prompt, list):
                    prompt = [prompt]

                await chat_local.chat_async(
                    *prompt,
                    echo="none",
                    stream=False,
                    kwargs=kwargs,
                )
                progress.advance(task_id)
                return chat_local

        tasks = [_do_chat(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


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
    from chatlas import ChatOpenAI, parallel_chat_text

    chat = ChatOpenAI()

    countries = ["Canada", "New Zealand", "Jamaica", "United States"]
    prompts = [f"What's the capital of {country}?" for country in countries]

    responses = parallel_chat_text(chat, prompts)
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
    from chatlas import ChatOpenAI, parallel_chat_structured
    from pydantic import BaseModel


    class Person(BaseModel):
        name: str
        age: int


    chat = ChatOpenAI()

    prompts = [
        "I go by Alex. 42 years on this planet and counting.",
        "Pleased to meet you! I'm Jamal, age 27.",
        "They call me Li Wei. Nineteen years young.",
        "Fatima here. Just celebrated my 35th birthday last week.",
    ]

    people = parallel_chat_structured(chat, prompts, Person)
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

    rate_limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(max_active)

    # Make a "global" copy of the chat with no turns to use as a template
    # (to avoid a deep copy of turns for each prompt)
    turns = chat.get_turns()
    chat_global = copy.deepcopy(chat)
    chat_global.set_turns([])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(
            f"Processing {len(prompts)} prompts...", total=len(prompts)
        )

        async def _do_chat(prompt: ContentT | list[ContentT]):
            async with semaphore:
                await rate_limiter.acquire()

                chat_local = copy.deepcopy(chat_global)
                chat_local.set_turns(turns)

                if not isinstance(prompt, list):
                    prompt = [prompt]

                result = await chat_local.chat_structured_async(
                    *prompt,
                    data_model=data_model,
                    echo="none",
                    kwargs=kwargs,
                )
                progress.advance(task_id)
                return result

        tasks = [_do_chat(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


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

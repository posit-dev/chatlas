"""
Batch chat processing for submitting multiple requests simultaneously.

This module provides functionality for submitting multiple chat requests
in batches to providers that support it (currently OpenAI and Anthropic).
Batch processing can take up to 24 hours but offers significant cost savings
(up to 50% less than regular requests).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TypeVar, Union

from pydantic import BaseModel

from ._batch_job import BatchJob, ContentT
from ._chat import Chat

ChatT = TypeVar("ChatT", bound=Chat)
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def batch_chat(
    chat: ChatT,
    prompts: list[ContentT] | list[list[ContentT]],
    path: Union[str, Path],
    wait: bool = True,
) -> list[ChatT | None]:
    """
    Submit multiple chat requests in a batch.

    This function allows you to submit multiple chat requests simultaneously
    using provider batch APIs (currently OpenAI and Anthropic). Batch processing
    can take up to 24 hours but offers significant cost savings.

    Parameters
    ----------
    chat
        Chat instance to use for the batch
    prompts
        List of prompts to process. Each can be a string or list of strings.
    path
        Path to file (with .json extension) to store batch state
    wait
        If True, wait for batch to complete. If False, return None if incomplete.

    Returns
    -------
    List of Chat objects (one per prompt) if complete, None if wait=False and incomplete.
    Individual Chat objects may be None if their request failed.

    Example
    -------

    ```python
    from chatlas import ChatOpenAI

    chat = ChatOpenAI()
    prompts = [
        "What's the capital of France?",
        "What's the capital of Germany?",
        "What's the capital of Italy?",
    ]

    chats = batch_chat(chat, prompts, "capitals.json")
    for i, result_chat in enumerate(chats):
        if result_chat:
            print(f"Prompt {i + 1}: {result_chat.get_last_turn().text}")
    ```
    """
    job = BatchJob(chat, prompts, path, wait=wait)
    job.step_until_done()

    chats = []
    assistant_turns = job.result_turns()
    for user, assistant in zip(job.user_turns, assistant_turns):
        if assistant is not None:
            new_chat = copy.deepcopy(chat)
            new_chat.add_turn(user)
            new_chat.add_turn(assistant)
            chats.append(new_chat)
        else:
            chats.append(None)

    return chats


def batch_chat_text(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    path: Union[str, Path],
    wait: bool = True,
) -> list[str | None]:
    """
    Submit multiple chat requests in a batch and return text responses.

    This is a convenience function that returns just the text of the responses
    rather than full Chat objects.

    Parameters
    ----------
    chat
        Chat instance to use for the batch
    prompts
        List of prompts to process
    path
        Path to file (with .json extension) to store batch state
    wait
        If True, wait for batch to complete

    Return
    ------
    List of text responses (or None for failed requests)
    """
    chats = batch_chat(chat, prompts, path, wait=wait)

    texts = []
    for x in chats:
        if x is None:
            texts.append(None)
            continue
        last_turn = x.get_last_turn()
        if last_turn is None:
            texts.append(None)
            continue
        texts.append(last_turn.text)

    return texts


def batch_chat_structured(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    path: Union[str, Path],
    data_model: type[BaseModelT],
    wait: bool = True,
) -> list[BaseModelT | None]:
    """
    Submit multiple structured data requests in a batch.

    Parameters
    ----------
    chat
        Chat instance to use for the batch
    prompts
        List of prompts to process
    path
        Path to file (with .json extension) to store batch state
    data_model
        Pydantic model class for structured responses
    wait
        If True, wait for batch to complete

    Return
    ------
    List of structured data objects (or None for failed requests)
    """
    job = BatchJob(chat, prompts, path, data_model=data_model, wait=wait)
    result = job.step_until_done()

    if result is None:
        return []

    res: list[BaseModelT | None] = []
    assistant_turns = job.result_turns()
    for turn in assistant_turns:
        if turn is None:
            res.append(None)
        else:
            json = chat._extract_turn_json(turn)
            model = data_model.model_validate(json)
            res.append(model)

    return res


def batch_chat_completed(
    chat: Chat,
    prompts: list[ContentT] | list[list[ContentT]],
    path: Union[str, Path],
) -> bool:
    """
    Check if a batch job is completed without waiting.

    Parameters
    ----------
    chat
        Chat instance used for the batch
    prompts
        List of prompts used for the batch
    path
        Path to batch state file

    Returns
    -------
    True if batch is complete, False otherwise
    """
    job = BatchJob(chat, prompts, path, wait=False)
    stage = job.stage

    if stage == "submitting":
        return False
    elif stage == "waiting":
        status = job._poll()
        return not status.working
    elif stage == "retrieving" or stage == "done":
        return True
    else:
        raise ValueError(f"Unknown batch stage: {stage}")
